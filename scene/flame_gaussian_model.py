#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from .gaussian_model import GaussianModel
from flame_model.flame import FlameHead
from utils.graphics_utils import compute_face_orientation
from utils.uv_utils import PositionMapGenerator, load_uv_region_masks
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.mesh_sampling import (
    reweight_uvcoords_by_barycoords,
    hybrid_sampling_by_binding,
)


class FlameGaussianModel(GaussianModel):
    def __init__(
        self,
        sh_degree: int,
        disable_flame_static_offset=False,
        not_finetune_flame_params=False,
        n_shape=300,
        n_expr=100,
    ):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr
        self.mouth_mask_face_idsc = None
        self.flame_model = FlameHead(n_shape, n_expr, add_teeth=True).cuda()

        self.flame_param = None
        self.flame_param_orig = None

        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

    def uvcoords_sample(self):
        face_index, bary_coords = hybrid_sampling_by_binding(
            binding=self.binding,
            tex_coord=self.flame_model.verts_uvs.to("cuda"),
            uv_faces=self.flame_model.textures_idx.to("cuda"),
            uv_size=256,
        )
        prior_uvcoords_sample = reweight_uvcoords_by_barycoords(
            uvcoords=self.flame_model.verts_uvs.to("cuda"),
            uvfaces=self.flame_model.textures_idx.to("cuda"),
            face_index=face_index,
            bary_coords=bary_coords,
        )
        prior_uvcoords_sample = prior_uvcoords_sample[..., :2]

        return prior_uvcoords_sample

    def get_position_map(self):
        self.map_generator = PositionMapGenerator(
            verts = self.verts,
            faces=self.flame_model.faces.unsqueeze(0),
            uvfaces=self.flame_model.textures_idx.unsqueeze(0),
            uv_coords=self.flame_model.verts_uvs.unsqueeze(0),
            image_size=256,
            uv_size=256,
            device="cuda",
        )
        return self.map_generator.generate_position_map()
    
    def get_uv_mask(self):
        self.uv_mask = load_uv_region_masks("flame_model/assets/flame/uv_region_masks.pkl")
        return self.uv_mask

    def get_vertex_displace_map(self):
        return self.map_generator.displacement_map(self.verts)

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = (meshes if len(tgt_meshes) == 0 else tgt_meshes)

            self.num_timesteps = max(pose_meshes) + 1
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]["static_offset"])
                if static_offset.shape[1] != num_verts:
                    static_offset = torch.nn.functional.pad(
                        static_offset, (0, 0, 0, num_verts - static_offset.shape[1])
                    )
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                "shape": torch.from_numpy(meshes[0]["shape"]),
                "expr": torch.zeros([T, meshes[0]["expr"].shape[1]]),
                "rotation": torch.zeros([T, 3]),
                "neck_pose": torch.zeros([T, 3]),
                "jaw_pose": torch.zeros([T, 3]),
                "eyes_pose": torch.zeros([T, 6]),
                "translation": torch.zeros([T, 3]),
                "static_offset": static_offset,
                "dynamic_offset": torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param["expr"][i] = torch.from_numpy(mesh["expr"])
                self.flame_param["rotation"][i] = torch.from_numpy(mesh["rotation"])
                self.flame_param["neck_pose"][i] = torch.from_numpy(mesh["neck_pose"])
                self.flame_param["jaw_pose"][i] = torch.from_numpy(mesh["jaw_pose"])
                self.flame_param["eyes_pose"][i] = torch.from_numpy(mesh["eyes_pose"])
                self.flame_param["translation"][i] = torch.from_numpy(mesh["translation"])

            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()

            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            # import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if "shape" in flame_param:
            shape = flame_param["shape"]
        else:
            shape = self.flame_param["shape"]

        if "static_offset" in flame_param:
            static_offset = flame_param["static_offset"]
        else:
            static_offset = self.flame_param["static_offset"]

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param["expr"].cuda(),
            flame_param["rotation"].cuda(),
            flame_param["neck"].cuda(),
            flame_param["jaw"].cuda(),
            flame_param["eyes"].cuda(),
            flame_param["translation"].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = (
            self.flame_param_orig
            if original and self.flame_param_orig != None
            else self.flame_param
        )

        verts, verts_cano = self.flame_model(
            flame_param["shape"][None, ...],
            flame_param["expr"][[timestep]],
            flame_param["rotation"][[timestep]],
            flame_param["neck_pose"][[timestep]],
            flame_param["jaw_pose"][[timestep]],
            flame_param["eyes_pose"][[timestep]],
            flame_param["translation"][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param["static_offset"],
            dynamic_offset=flame_param["dynamic_offset"][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)

    def cluster_flame_conditions(self, dataset_len, use_keyframes=False, interval=25):
        """
        Cluster FLAME conditions (expr, jaw_pose, neck_pose, eyes_pose) via KMeans.
        Args:
            dataset_len (int): total number of frames.
            num_clusters (int): number of KMeans clusters.
            use_keyframes (bool): if True, cluster only keyframes (center of each interval); else all frames.
            interval (int): interval for keyframe sampling (default=25).
        Returns:
            clustered_timestamps (List[List[int]]): list of clusters, each containing frame indices.
        """
        # 1. Prepare frame indices
        if use_keyframes:
            idxs = np.arange(interval // 2, dataset_len, interval)
            if idxs[-1] >= dataset_len:
                idxs[-1] = dataset_len - 1
        else:
            idxs = np.arange(dataset_len)

        # 2. Stack FLAME parameters
        expr        = self.flame_param['expr'][:dataset_len]        # (T, 100)
        jaw         = self.flame_param['jaw_pose'][:dataset_len]    # (T, 3)
        neck        = self.flame_param['neck_pose'][:dataset_len]   # (T, 3)
        eyes        = self.flame_param['eyes_pose'][:dataset_len]   # (T, 6)
        rotation    = self.flame_param['rotation'][:dataset_len]    # (T, 3)
        translation = self.flame_param['translation'][:dataset_len] # (T, 3)

        # 2.1 Convert to CPU numpy for sklearn PCA
        expr_np = expr.cpu().detach().numpy()
        pose_np = torch.cat([jaw, neck, eyes, rotation], dim=1).cpu().detach().numpy()   # (T, 15)
        trans_np = translation.cpu().detach().numpy()                                     # (T, 3)

        # 2.2 Apply PCA to each group
        pca_expr = PCA(n_components=12, random_state=42).fit_transform(expr_np)    # ↓ 100 → 12
        pca_pose = PCA(n_components=12, random_state=42).fit_transform(pose_np)    # ↓ 15  → 12

        def norm_np(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)

        expr_norm = norm_np(pca_expr) * (0.3 ** 0.5)  
        pose_norm = norm_np(pca_pose) * (0.6 ** 0.5)
        trans_norm = norm_np(trans_np) * (0.1 ** 0.5)     
        feats_np = np.concatenate([expr_norm, pose_norm, trans_norm], axis=1)

        # 4. Adaptive KMeans clustering
        best_k = None
        best_score = -1
        best_labels = None
        search_range = range(5, 13)

        print("Searching optimal K based on silhouette score:")
        for K in search_range:
            kmeans_tmp = KMeans(n_clusters=K, random_state=42)
            labels_tmp = kmeans_tmp.fit_predict(feats_np)
            try:
                score = silhouette_score(feats_np, labels_tmp)
            except ValueError:
                continue  # Silhouette score not defined for certain degenerate clusterings
            print(f"K={K}, silhouette score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = K
                best_labels = labels_tmp

        print(f"\nSelected K = {best_k} with silhouette score = {best_score:.4f}")

        # 5. Group original frames by cluster: propagate cluster label of each sel-index to its interval window
        clustered_timestamps = [[] for _ in range(best_k)]
        if use_keyframes:
            half = interval // 2
            for sel_idx, label in enumerate(best_labels):
                center = idxs[sel_idx]
                start = max(0, center - half)
                end = min(dataset_len, center + half)
                clustered_timestamps[label].extend(range(start, end))
        else:
            # direct mapping: each frame gets its own label
            for frame_idx, label in zip(idxs, best_labels):
                clustered_timestamps[label].append(int(frame_idx))

        # 6. Deduplicate and sort
        clustered_timestamps = [sorted(set(lst)) for lst in clustered_timestamps]
        
        return clustered_timestamps, best_k

    def update_mesh_properties(self, verts, verts_cano):

        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), faces.squeeze(0), return_scale=True
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat)
        )  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces
        self.triangles = triangles

        # for mesh regularization
        self.verts_cano = verts_cano


    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.not_finetune_flame_params:
            return

        # pose
        self.flame_param["rotation"].requires_grad = True
        self.flame_param["neck_pose"].requires_grad = True
        self.flame_param["jaw_pose"].requires_grad = True
        self.flame_param["eyes_pose"].requires_grad = True
        params = [
            self.flame_param["rotation"],
            self.flame_param["neck_pose"],
            self.flame_param["jaw_pose"],
            self.flame_param["eyes_pose"],
        ]
        param_pose = {
            "params": params,
            "lr": training_args.flame_pose_lr,
            "name": "pose",
        }
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param["translation"].requires_grad = True
        param_trans = {
            "params": [self.flame_param["translation"]],
            "lr": training_args.flame_trans_lr,
            "name": "trans",
        }
        self.optimizer.add_param_group(param_trans)

        # expression
        self.flame_param["expr"].requires_grad = True
        param_expr = {
            "params": [self.flame_param["expr"]],
            "lr": training_args.flame_expr_lr,
            "name": "expr",
        }
        self.optimizer.add_param_group(param_expr)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs["has_target"]:
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {
                k: torch.from_numpy(v).cuda() for k, v in flame_param.items()
            }

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param["expr"].shape[
                0
            ]  # required by viewers

        if "motion_path" in kwargs and kwargs["motion_path"] is not None:
            motion_path = Path(kwargs["motion_path"])
            flame_param = np.load(str(motion_path))
            flame_param = {
                k: torch.from_numpy(v).cuda()
                for k, v in flame_param.items()
                if v.dtype == np.float32
            }

            self.flame_param = {
                # keep the static parameters
                "shape": self.flame_param["shape"],
                "static_offset": self.flame_param["static_offset"],
                # update the dynamic parameters
                "translation": flame_param["translation"],
                "rotation": flame_param["rotation"],
                "neck_pose": flame_param["neck_pose"],
                "jaw_pose": flame_param["jaw_pose"],
                "eyes_pose": flame_param["eyes_pose"],
                "expr": flame_param["expr"],
                "dynamic_offset": flame_param["dynamic_offset"],
            }
            self.num_timesteps = self.flame_param["expr"].shape[
                0
            ]  # required by viewers

        if "disable_fid" in kwargs and len(kwargs["disable_fid"]) > 0:
            mask = (self.binding[:, None] != kwargs["disable_fid"][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
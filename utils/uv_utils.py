import numpy as np
from skimage import io
import torch
import torch.nn as nn
global Meshes, load_obj, rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from typing import Optional
import pickle
import os
from skimage import io


class PositionMapGenerator:
    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        uvfaces: torch.Tensor,
        uv_coords: torch.Tensor,
        image_size: int = 256,
        uv_size: int = 256,
        device: str = "cuda",
    ):
        self.device = device
        self.verts = verts.to(device)
        self.faces = faces.to(device)
        self.uvfaces = uvfaces.to(device)
        self.uv_coords = uv_coords.to(device)

        self.render_uv = SRenderY(
            image_size=image_size,
            uv_size=uv_size,
            faces=self.faces,
            uvfaces=self.uvfaces,
            uvcoords=self.uv_coords,
            rasterizer_type="pytorch3d",
        ).to(device)

    def generate_position_map(
        self,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:

        vertices = self.normalize_vertices(self.verts)

        self.position_map = self.render_uv.world2uv(vertices).detach()

        if save_path is not None:
            position_map_show = (
                (self.position_map.squeeze(0).permute(1, 2, 0).cpu().numpy()/2+0.5) * 255
            )
            io.imsave(
                f"{save_path}/uv_position_map.jpg", position_map_show.astype(np.uint8)
            )

        return self.position_map
    
    def displacement_map(
        self,
        verts,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        
        verts = self.normalize_vertices(verts)
        position_map = self.render_uv.world2uv(verts).detach()

        displacement_map = position_map - self.position_map

        if save_path is not None:
            displacement_map_vis = (
                (displacement_map.squeeze(0).permute(1, 2, 0).cpu().numpy()/2+0.5) * 255 
            )
            io.imsave(
                f"{save_path}/uv_displace_map.jpg", displacement_map_vis.astype(np.uint8)
            )

        return displacement_map
    
    
    def normalize_vertices(self, vertices):
        """
        Normalize vertices to [0, 1] range per axis.

        Args:
            vertices: Tensor of shape (N, 3) or (B, N, 3)

        Returns:
            normalized vertices of the same shape
        """
        if vertices.dim() == 2:
            # Single mesh: (N, 3)
            x_min = vertices[:, 0].min()
            x_max = vertices[:, 0].max()
            y_min = vertices[:, 1].min()
            y_max = vertices[:, 1].max()
            z_min = vertices[:, 2].min()
            z_max = vertices[:, 2].max()

            vertices[:, 0] = (vertices[:, 0] - x_min) / (x_max - x_min) * 2 -1
            vertices[:, 1] = (vertices[:, 1] - y_min) / (y_max - y_min) * 2 -1
            vertices[:, 2] = (vertices[:, 2] - z_min) / (z_max - z_min) * 2 -1

        elif vertices.dim() == 3:
            # Batch of meshes: (B, N, 3)
            x_min = vertices[:, :, 0].min(dim=1, keepdim=True).values
            x_max = vertices[:, :, 0].max(dim=1, keepdim=True).values
            y_min = vertices[:, :, 1].min(dim=1, keepdim=True).values
            y_max = vertices[:, :, 1].max(dim=1, keepdim=True).values
            z_min = vertices[:, :, 2].min(dim=1, keepdim=True).values
            z_max = vertices[:, :, 2].max(dim=1, keepdim=True).values

            vertices[:, :, 0] = (vertices[:, :, 0] - x_min) / (x_max - x_min) * 2 -1
            vertices[:, :, 1] = (vertices[:, :, 1] - y_min) / (y_max - y_min) * 2 -1
            vertices[:, :, 2] = (vertices[:, :, 2] - z_min) / (z_max - z_min) * 2 -1

        else:
            raise ValueError("Input must be of shape (N, 3) or (B, N, 3)")

        return vertices
    

class UvRegionMask:
    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        uvfaces: torch.Tensor,
        uv_coords: torch.Tensor,
        image_size: int = 256,
        uv_size: int = 256,
        device: str = "cuda",
    ):
        self.device = device
        self.verts = verts.to(device)
        self.faces = faces.to(device)
        self.uvfaces = uvfaces.to(device)
        self.uv_coords = uv_coords.to(device)
        self.pkl_path = "flame_model/assets/flame/FLAME_masks.pkl"

        self.render_uv = SRenderY(
            image_size=image_size,
            uv_size=uv_size,
            faces=self.faces,
            uvfaces=self.uvfaces,
            uvcoords=self.uv_coords,
            rasterizer_type="pytorch3d",
        ).to(device)

    def load_flame_masks(self):
        """Load the FLAME_masks.pkl and return the dict of numpy arrays."""
        with open(self.pkl_path, "rb") as f:
            masks = pickle.load(f, encoding='latin1')
        return masks


    def generate_uv_region_mask(
        self,
        visualize: bool = True,
        save_dir: str = "flame_model/assets/flame"
    ) -> torch.Tensor:

        os.makedirs(save_dir, exist_ok=True)
        V = self.verts.shape[1]
        masks_dict = self.load_flame_masks()
        regions = ["eye_region", "nose", "lips", "forehead"]
        uv_masks = {}

        for region in regions:
            idx_np = masks_dict[region]
            idx = torch.from_numpy(idx_np).long().to(self.device)

            vert_colors = torch.zeros((1, V, 3), device="cuda")
            vert_colors[0, idx.clamp(max=V - 1), :] = 1.0
            uv_masks_raw = self.render_uv.world2uv(vert_colors).detach()

            uv_masks[region] = (uv_masks_raw[:, 0:1, :, :] > 0).float()  # shape: [1, 1, H, W]

        if visualize:
            os.makedirs("./uv_masks_vis", exist_ok=True)
            uv_masks_show = {}
            for region in regions:
                uv_masks_show[region] = (
                    (uv_masks[region].squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()) * 255
                )
                io.imsave(
                    f"./uv_masks_vis/{region}.png", uv_masks_show[region].astype(np.uint8)
                )

        pkl_path = os.path.join(save_dir, "uv_region_masks.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({k: v.cpu().numpy() for k, v in uv_masks.items()}, f)

        return uv_masks


def load_uv_region_masks(
    pkl_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Returns:
        uv_masks: dict[str, torch.Tensor], shape: [1, H, W]
    """
    with open(pkl_path, "rb") as f:
        mask_dict_np = pickle.load(f)

    uv_masks = {
        region: torch.from_numpy(mask).to(device=device, dtype=dtype)
        for region, mask in mask_dict_np.items()
    }

    return uv_masks


def dict2obj(d):

    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face2vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w - 1 - margin_x):
        for y in range(margin_y, h - 1 - margin_y):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


class Pytorch3dRasterizer(nn.Module):
    """Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            "image_size": image_size,
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": None,
            "max_faces_per_bin": None,
            "perspective_correct": False,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h > w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1] * h / w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0] * w / h

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1]
        )
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # print(image_size)
        # import ipdb; ipdb.set_trace()
        return pixel_vals


# modified from https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
class SRenderY(nn.Module):
    def __init__(
        self,
        image_size,
        faces,
        uvfaces,
        uvcoords,
        uv_size=256,
        rasterizer_type="pytorch3d",
    ):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        if rasterizer_type == "pytorch3d":
            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
        else:
            NotImplementedError

        # faces
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer(
            "dense_faces", torch.from_numpy(dense_triangles).long()[None, :, :]
        )
        self.register_buffer("faces", faces)
        self.register_buffer("raw_uvcoords", uvcoords)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1
        )  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face2vertices(uvcoords, uvfaces)
        self.register_buffer("uvcoords", uvcoords)
        self.register_buffer("uvfaces", uvfaces)
        self.register_buffer("face_uvcoords", face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = (
            torch.tensor([180, 180, 180])[None, None, :]
            .repeat(1, faces.max() + 1, 1)
            .to("cuda")
            .float()
            / 255.0
        )
        face_colors = face2vertices(colors, faces)
        self.register_buffer("face_colors", face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ]
        ).float()
        self.register_buffer("constant_factor", constant_factor)

    def world2uv(self, vertices):
        """
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = face2vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1),
            face_vertices,
        )[:, :3]
        return uv_vertices

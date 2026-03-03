import torch
import torch.nn as nn
from pytorch3d.structures       import Meshes
from pytorch3d.renderer.mesh    import rasterize_meshes

# modified from https://github.com/facebookresearch/pytorch3d
class Pytorch3dRasterizer(nn.Module):
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
            'cull_backfaces': True
        }
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings['image_size']
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings['blur_radius'],
            faces_per_pixel=raster_settings['faces_per_pixel'],
            bin_size=raster_settings['bin_size'],
            max_faces_per_bin=raster_settings['max_faces_per_bin'],
            perspective_correct=raster_settings['perspective_correct'],
            cull_backfaces=raster_settings['cull_backfaces']
        )

        return pix_to_face, bary_coords


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    Indexing the coordinates of the three vertices on each face.

    Args:
        vertices:   [bs, V, 3]
        faces:      [bs, F, 3]

    Return: 
        face_to_vertices: [bs, F, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))

    return vertices[faces.long()]


def hybrid_sampling_by_binding(
        binding: torch.Tensor,    # [N], each gs point's binding face index
        tex_coord: torch.Tensor,  # [V,2], UV coords per vertex
        uv_faces: torch.Tensor,   # [F,3], face-to-vertex indices
        uv_size: int = 256,
        device: str = 'cuda'
    ):
    """
    Hybrid sampling strategy:
    - Rasterize UV mesh to collect valid pixels per face.
    - For faces with valid pixels: uniformly sample barycentric coords from pixel pool.
    - For faces without pixels: fallback to analytic barycentric sampling.

    Args:
        binding: tensor of shape [N], face index for each gs point.
        tex_coord: tensor [V,2], UV coordinates.
        uv_faces: tensor [F,3], indices into tex_coord for each face.
        uv_size: resolution for UV rasterization.
        device: computation device.

    Returns:
        face_index: tensor [N], same as binding.
        bary_coords: tensor [N,3], sampled barycentric coordinates per gs point.
    """
    # move inputs to device
    binding = binding.to(device)
    tex_coord = tex_coord.to(device)
    uv_faces = uv_faces.to(device)

    # 1) Rasterization
    rasterizer = Pytorch3dRasterizer(uv_size)
    tex = tex_coord[None, ...]  # [1, V, 2]
    faces = uv_faces[None, ...]  # [1, F, 3]

    # append dummy W=1, map to NDC
    tex_ndc = torch.cat([tex, torch.ones_like(tex[..., :1])], dim=-1)  # [1, V, 3]
    tex_ndc = tex_ndc * 2 - 1
    tex_ndc[...,1] = -tex_ndc[...,1]

    pix_to_face, bary = rasterizer(tex_ndc.expand(1,-1,-1), faces.expand(1,-1,-1))
    pix_to_face = pix_to_face[0]  # [H, W]
    bary = bary[0]                # [H, W, 3]

    # mask and gather valid pixels
    valid_mask = pix_to_face != -1
    valid_face_index = pix_to_face[valid_mask]       # [P]
    valid_bary = bary[valid_mask]                     # [P, 3]
    pixel_coords = torch.nonzero(valid_mask, as_tuple=False)  # [P, 2]

    # determine which faces have any valid pixels
    F = uv_faces.shape[0]
    face_has_pixel = torch.zeros(F, dtype=torch.bool, device=device)
    faces_present = torch.unique(valid_face_index)
    face_has_pixel[faces_present] = True

    # prepare output container
    N = binding.shape[0]
    bary_out = torch.empty((N, 3), device=device)

    # iterate over each unique face in binding
    unique_faces = torch.unique(binding)
    for f in unique_faces:
        f_idx = f.item()
        # indices in binding for current face
        bind_inds = (binding == f_idx).nonzero(as_tuple=False).squeeze(1)
        count_needed = bind_inds.shape[0]

        if face_has_pixel[f_idx]:
            # get indices in valid arrays for this face
            mask_f = valid_face_index == f_idx
            idxs = torch.nonzero(mask_f, as_tuple=False).squeeze(1)  # [M]
            M = idxs.shape[0]

            # sort by raster pixel location for determinism
            coords = pixel_coords[idxs]
            scores = coords[:,0] * uv_size + coords[:,1]
            _, order = torch.sort(scores)
            sorted_idxs = idxs[order]

            if M >= count_needed:
                # uniform selection via linspace
                if count_needed == 1:
                    sel = sorted_idxs[M // 2: M // 2 + 1]
                else:
                    positions = torch.linspace(0, M-1, steps=count_needed, device=device).long()
                    sel = sorted_idxs[positions]
            else:
                # use all and sample extra with replacement
                extra = torch.randint(0, M, (count_needed - M,), device=device)
                sel = torch.cat([sorted_idxs, sorted_idxs[extra]], dim=0)

            sampled = valid_bary[sel]
        else:
            # analytic barycentric fallback
            u = torch.rand(count_needed, device=device)
            v = torch.rand(count_needed, device=device)
            sqrt_u = torch.sqrt(u)
            b0 = 1 - sqrt_u
            b1 = sqrt_u * (1 - v)
            b2 = sqrt_u * v
            sampled = torch.stack([b0, b1, b2], dim=1)

        # assign to output in original binding order
        bary_out[bind_inds] = sampled

    # face_index output is same as binding
    return binding, bary_out


def reweight_uvcoords_by_barycoords(
        uvcoords:    torch.Tensor,
        uvfaces:     torch.Tensor,
        face_index:  torch.Tensor,
        bary_coords: torch.Tensor,
    ):
    """
    Reweights the UV coordinates based on the barycentric coordinates for each face.

    Args:
        uvcoords:       [bs, V', 2].
        uvfaces:        [F, 3].
        face_index:     [N].
        bary_coords:    [N, 3].

    Returns:
        Reweighted UV coordinates, shape [bs, N, 2].
    """

    face_index = face_index.long()

    # homogeneous coordinates
    num_v           = uvcoords.shape[0]
    uvcoords        = torch.cat([uvcoords, torch.ones((num_v, 1)).to(uvcoords.device)], dim=1)
    # index attributes by face
    uvcoords        = uvcoords[None, ...]
    face_verts      = face_vertices(uvcoords,  uvfaces.expand(1, -1, -1))   # [1, F, 3, 3]
    # gather idnex for every splat
    N               = face_index.shape[0]
    face_index_3    = face_index.view(1, N, 1, 1).expand(1, N, 3, 3)
    position_vals   = face_verts.gather(1, face_index_3)
    # reweight
    position_vals   = (bary_coords[..., None] * position_vals).sum(dim = -2)

    return position_vals
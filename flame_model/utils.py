import numpy as np

def vert_uvs(V, uvs, faces_uvs, faces):
    """Compute uv coords for each vertex

    Args:
        V (int): num of vertices
        uvs (_type_): uv coords, N_uvx2
        faces_uvs (_type_): uv_ids in faces, N_fx3
        faces (_type_): vert_ids in faces, N_fx3
    """
    vert_uvs = np.ones((V, 2)) * -1
    for i, face in enumerate(faces):
        v1, v2, v3 = face
        uv1, uv2, uv3 = faces_uvs[i]

        vert_uvs[v1] = uvs[uv1]
        vert_uvs[v2] = uvs[uv2]
        vert_uvs[v3] = uvs[uv3]

    if (vert_uvs.sum(-1) < 0).sum():
        print("[Waring] Function 'vert_uvs': there are some verts that have no uv coords.")

    return vert_uvs.astype(np.float32)

def split_verts_for_unique_uv(V, uvs, faces_uvs, faces):
    """Split mesh verts to make verts and uvs 1<->1 match.

    Args:
        uvs (_type_): uv coords, N_uvx2
        faces_uvs (_type_): uv_ids in faces, N_fx3
        faces (_type_): vert_ids in faces, N_fx3
    """
    new_faces = faces.clone()
    extra_verts_ids = []

    vert_uvs = {}
    for i, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        uv1, uv2, uv3 = int(faces_uvs[i][0]), int(faces_uvs[i][1]), int(faces_uvs[i][2])

        if v1 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v1].items():
                if (uvs[uv1] == uv).all():
                    new_faces[i][0] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][0] = extra_id
                extra_verts_ids.append(v1)
                vert_uvs[v1][extra_id] = uvs[uv1]
        else:
            vert_uvs[v1] = {v1: uvs[uv1]}

        if v2 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v2].items():
                if (uvs[uv2] == uv).all():
                    new_faces[i][1] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][1] = extra_id
                extra_verts_ids.append(v2)
                vert_uvs[v2][extra_id] = uvs[uv2]
        else:
            vert_uvs[v2] = {v2: uvs[uv2]}

        if v3 in vert_uvs:
            match = False
            for v, uv in vert_uvs[v3].items():
                if (uvs[uv3] == uv).all():
                    new_faces[i][2] = v
                    match = True
                    break

            if not match:
                extra_id = V + len(extra_verts_ids)
                new_faces[i][2] = extra_id
                extra_verts_ids.append(v3)
                vert_uvs[v3][extra_id] = uvs[uv3]
        else:
            vert_uvs[v3] = {v3: uvs[uv3]}

    return extra_verts_ids, new_faces
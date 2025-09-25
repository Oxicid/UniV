# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'np' in locals():
    from .. import reload
    reload.reload(globals())

import numpy as np
from mathutils import Vector, Matrix
from ... import utypes


def extract_seams_umesh_ex(umesh: utypes.UMesh, coords_append):
    uv = umesh.uv
    if umesh.is_full_face_selected:
        for e in umesh.bm.edges:
            if e.seam:
                for crn in getattr(e, 'link_loops', ()):
                    coords_append(crn[uv].uv)
                    coords_append(crn.link_loop_next[uv].uv)
    else:
        if umesh.sync:
            for e in umesh.bm.edges:
                if e.seam:
                    for crn in getattr(e, 'link_loops', ()):
                        if not crn.face.hide:
                            coords_append(crn[uv].uv)
                            coords_append(crn.link_loop_next[uv].uv)
        else:
            if umesh.is_full_face_deselected:
                return
            for e in umesh.bm.edges:
                if e.seam:
                    for crn in getattr(e, 'link_loops', ()):
                        if crn.face.select:
                            coords_append(crn[uv].uv)
                            coords_append(crn.link_loop_next[uv].uv)

def extract_seams_umesh(umesh: utypes.UMesh):
    coords = []
    coords_append = coords.append
    extract_seams_umesh_ex(umesh, coords_append)
    return coords


def extract_seams_umeshes(umeshes: utypes.UMeshes) -> list[Vector]:
    coords = []
    coords_append = coords.append

    for umesh in umeshes:
        extract_seams_umesh_ex(umesh, coords_append)

    return coords


def extract_edges_with_seams(umesh: utypes.UMesh):
    edges = []
    edges_append = edges.append

    if umesh.is_full_face_selected:
        for e in umesh.bm.edges:
            if e.seam and hasattr(e, 'link_loops'):
                edges_append(e)
    else:
        if umesh.sync:
            for e in umesh.bm.edges:
                if e.seam and hasattr(e, 'link_loops'):
                    edges_append(e)
        else:
            if umesh.is_full_face_deselected:
                return []
            for e in umesh.bm.edges:
                if e.seam and hasattr(e, 'link_loops'):
                    edges_append(e)
    return edges


_local_verts = np.array([
    [-0.5, -0.5, -0.5, 1.0],
    [ 0.5, -0.5, -0.5, 1.0],
    [ 0.5,  0.5, -0.5, 1.0],
    [-0.5,  0.5, -0.5, 1.0],
    [-0.5, -0.5,  0.5, 1.0],
    [ 0.5, -0.5,  0.5, 1.0],
    [ 0.5,  0.5,  0.5, 1.0],
    [-0.5,  0.5,  0.5, 1.0],
], dtype=np.float32)

_edges_indexes = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.uint8)


def extraxt_cube_lines_for_orient_bound(orient_matrix: Matrix, dims):
    scale_matrix = Matrix.Diagonal(dims).to_4x4()
    mat = np.array(orient_matrix @ scale_matrix, dtype=np.float32).T

    verts = (_local_verts @ mat)[:, :3]
    edges = verts[_edges_indexes].reshape(-1, 3)
    return edges

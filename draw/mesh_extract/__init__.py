# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import bpy

if 'np' in locals():
    from .. import reload
    reload.reload(globals())

import numpy as np
from mathutils import Vector, Matrix
from ... import utypes
from ... import utils


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

def extract_non_sync_select_data(umesh: utypes.UMesh):
    assert not umesh.sync
    from mathutils.geometry import tessellate_polygon

    verts_or_edges = []
    flat_tris = []
    normals = []

    vert_select_get = utils.vert_select_get_func(umesh)
    edge_select_get = utils.edge_select_get_func(umesh)
    # uv = umesh.uv
    match bpy.context.tool_settings.uv_select_mode:
        case 'VERTEX':
            for f in utils.calc_visible_uv_faces_iter(umesh):
                selected_mask = [vert_select_get(crn) for crn in f.loops]
                if not any(selected_mask):
                    continue

                if all(selected_mask):
                    coords = [v.co for v in f.verts]
                    # verts_or_edges_extend(coords)
                    tessellated = tessellate_polygon((coords,))
                    for a, b, c in tessellated:
                        flat_tris.append(coords[a])
                        flat_tris.append(coords[b])
                        flat_tris.append(coords[c])
                    normals.extend([f.normal] * (len(tessellated) * 3))
                else:
                    for select_state, v in zip(selected_mask, f.verts):
                        if select_state:
                            verts_or_edges.append(v.co)
        case 'EDGE':
            for f in utils.calc_visible_uv_faces_iter(umesh):
                selected_mask = [edge_select_get(crn) for crn in f.loops]
                if not any(selected_mask):
                    continue

                if all(selected_mask):
                    coords = [v.co for v in f.verts]

                    # prev_co = coords[-1]
                    # for curr_co in coords:
                    #     verts_or_edges_append(prev_co)
                    #     verts_or_edges_append(curr_co)
                    #     prev_co = curr_co

                    tessellated = tessellate_polygon((coords,))
                    for a, b, c in tessellated:
                        flat_tris.append(coords[a])
                        flat_tris.append(coords[b])
                        flat_tris.append(coords[c])
                    normals.extend([f.normal] * (len(tessellated) * 3))
                else:
                    for select_state, crn in zip(selected_mask, f.loops):
                        if select_state:
                            verts_or_edges.append(crn.vert.co)
                            verts_or_edges.append(crn.link_loop_next.vert.co)
        case _:
            for f in utils.calc_selected_uv_faces_iter(umesh):
                coords = [v.co for v in f.verts]
                tessellated = tessellate_polygon((coords,))
                for a, b, c in tessellated:
                    flat_tris.append(coords[a])
                    flat_tris.append(coords[b])
                    flat_tris.append(coords[c])
                normals.extend([f.normal] * (len(tessellated) * 3))

    return verts_or_edges, (flat_tris, normals)




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

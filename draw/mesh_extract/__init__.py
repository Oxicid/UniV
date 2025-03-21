# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
from ... import types

def extract_seams(umeshes: types.UMeshes):
    coords = []
    coords_append = coords.append

    for umesh in umeshes:
        uv = umesh.uv
        if umesh.is_full_face_selected:
            for e in umesh.bm.edges:
                if e.seam and (corners := getattr(e, 'link_loops', None)):
                    for crn in corners:
                        coords_append(crn[uv].uv)
                        coords_append(crn.link_loop_next[uv].uv)
        else:
            if umesh.sync:
                for e in umesh.bm.edges:
                    if e.seam and (corners := getattr(e, 'link_loops', None)):
                        for crn in corners:
                            if not crn.face.hide:
                                coords_append(crn[uv].uv)
                                coords_append(crn.link_loop_next[uv].uv)
            else:
                if umesh.is_full_face_deselected:
                    continue
                for e in umesh.bm.edges:
                    if e.seam and (corners := getattr(e, 'link_loops', None)):
                        for crn in corners:
                            if crn.face.select:
                                coords_append(crn[uv].uv)
                                coords_append(crn.link_loop_next[uv].uv)
    return coords

def extract_edges_with_seams(umesh: types.UMesh):
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
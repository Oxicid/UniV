# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import typing

from bmesh.types import BMesh, BMFace, BMEdge, BMVert, BMLoop, BMLayerItem
from mathutils import Vector
from collections import deque

from .. import types


def shared_crn(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn:
        return shared

def shared_is_linked(crn: BMLoop, _shared_crn: BMLoop, uv: BMLayerItem):
    return crn.link_loop_next[uv].uv == _shared_crn[uv].uv and \
           crn[uv].uv == _shared_crn.link_loop_next[uv].uv

def shared_linked_crn_by_idx(crn: BMLoop, uv) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        if crn.link_loop_next[uv].uv == shared[uv].uv and crn[uv].uv == shared.link_loop_next[uv].uv:
            return shared

def set_faces_tag(faces, tag=True):
    for f in faces:
        f.tag = tag

def face_centroid_uv(f: BMFace, uv: BMLayerItem):
    value = Vector((0, 0))
    loops = f.loops
    for l in loops:
        value += l[uv].uv
    return value / len(loops)

def calc_face_area_uv(f, uv) -> float:
    area = 0.0
    for crn in f.loops:
        area += abs(crn[uv].uv.cross(crn.link_loop_next[uv].uv))
    return area

def calc_max_length_uv_crn(corners, uv) -> BMLoop:
    length = -1.0
    crn_ = None
    for crn in corners:
        if length < (length_ := (crn[uv].uv - crn.link_loop_next[uv].uv).length_squared):
            crn_ = crn
            length = length_
    return crn_

# Need implement disc_next disc_prev
# def calc_non_manifolds_uv(bm, uv):
#     for f in bm.faces:
#         for l in f.loops:  # Running through all the neighboring faces
#             link_face = l.link_loop_radial_next.face
#             for ll in link_face.loops:
#
#                 if ll[uv].uv != l[uv].uv:
#                     continue
#                 # Skip manifold
#                 if (l.link_loop_next[uv].uv == ll.link_loop_prev[uv].uv) or \
#                         (ll.link_loop_next[uv].uv == l.link_loop_prev[uv].uv):
#                     continue
#                 else:
#                     l[uv].select = True
#                     ll[uv].select = True


def calc_non_manifolds(bm: BMesh) -> tuple[set[BMVert], set[BMEdge]]:
    non_manifold_verts = set()
    for v in bm.verts:
        if v.hide:
            continue
        if not v.is_manifold:
            non_manifold_verts.add(v)

    non_manifold_edges = set()
    for f in bm.faces:
        if f.hide:
            continue
        for l in f.loops:
            link_face = l.link_loop_radial_next.face
            if link_face.hide:
                continue
            for ll in link_face.loops:
                if ll.vert != l.vert:
                    continue
                # Skip manifold
                if (l.link_loop_next.vert == ll.link_loop_prev.vert) or \
                        (ll.link_loop_next.vert == l.link_loop_prev.vert):
                    continue
                else:
                    if not l.edge.is_boundary:
                        non_manifold_edges.add(l.edge)
                    if not ll.edge.is_boundary:
                        non_manifold_edges.add(ll.edge)
    return non_manifold_verts, non_manifold_edges


def prev_disc(l: BMLoop) -> BMLoop:
    return l.link_loop_prev.link_loop_radial_prev


def linked_crn_uv(first: BMLoop, uv: BMLayerItem):
    first_vert = first.vert
    first_co = first[uv].uv
    linked = []
    bm_iter = first

    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert:  # Skip boundary or flipped
            bm_iter = first
            linked_cw = []
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                if first_vert != bm_iter.vert:  # Skip boundary or flipped
                    break

                if bm_iter == first:
                    break
                if first_co == bm_iter[uv].uv:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == first:
            break
        if first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_unordered(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    linked.remove(first)
    return linked

def linked_crn_uv_unordered_included(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = deque(l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co)
    return linked

def linked_crn_uv_by_tag_b(first: BMLoop, uv: BMLayerItem):
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first[uv].uv == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


def linked_crn_vert_uv_for_transform(first, uv):
    # Need tagging. tag == False - not append
    # assert utils.sync()
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first[uv].uv == bm_iter[uv].uv:
            bm_iter.tag = False
            linked.append(bm_iter)

    next_crn = first.link_loop_next
    if next_crn.tag:
        next_crn.tag = False
        linked.append(next_crn)

        bm_iter = next_crn
        while True:
            if (bm_iter := prev_disc(bm_iter)) == next_crn:
                break
            if not bm_iter.tag:
                continue
            if next_crn[uv].uv == bm_iter[uv].uv:
                bm_iter.tag = False
                linked.append(bm_iter)
    return linked

def linked_crn_uv_by_tag(first, uv):
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not (bm_iter.tag or bm_iter.link_loop_prev.tag):
            continue
        if first[uv].uv == bm_iter[uv].uv:  # TODO: Optimize and test
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_tag_c(crn: BMLoop, uv: BMLayerItem):
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.tag and l_crn[uv].uv == first_co]

def linked_crn_uv_by_face_index(first: BMLoop, uv: BMLayerItem):
    face_index = first.face.index
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.face.index == face_index and first[uv].uv == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_idx(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index with arg corner"""
    first_vert = crn.vert
    idx = crn.face.index
    first_co = crn[uv].uv
    linked = []
    bm_iter = crn

    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert:  # Skip boundary or flipped
            bm_iter = crn
            linked_cw = []
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                if first_vert != bm_iter.vert:  # Skip boundary or flipped
                    break

                if bm_iter == crn:
                    break
                if bm_iter.face.index == idx and first_co == bm_iter[uv].uv:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == crn:
            break
        if bm_iter.face.index == idx and first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_idx_unordered(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index without arg corner
    simular - linked_crn_uv_by_island_index_unordered
    """
    first_co = crn[uv].uv
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co]

def linked_crn_uv_by_idx_unordered_included(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index without arg corner
    simular - linked_crn_uv_by_island_index_unordered_included
    """
    first_co = crn[uv].uv
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx and l_crn[uv].uv == first_co]

def linked_crn_uv_by_island_index_unordered_included(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index with arg corner"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx and l_crn[uv].uv == first_co]

def linked_crn_uv_by_island_index_unordered(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index without arg corner"""
    first_co = crn[uv].uv
    linked = [l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx and l_crn[uv].uv == first_co]
    linked.remove(crn)
    return linked

def linked_crn_to_vert_by_face_index(crn):
    """Linked to vertex by face index without arg corner"""
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    linked.popleft()
    return linked

def linked_crn_to_vert_by_face_index_including(crn):
    """Linked to vertex by face index with arg corner"""
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    return linked

def select_linked_crn_uv_vert(first: BMLoop, uv: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:  # TODO: Optimize and test
            crn_uv_bm_iter.select = True

def select_crn_uv_edge(crn: BMLoop, uv):
    link_crn_next = crn.link_loop_next
    select_linked_crn_uv_vert(crn, uv)
    select_linked_crn_uv_vert(link_crn_next, uv)

    crn_uv_a = crn[uv]
    crn_uv_b = link_crn_next[uv]
    crn_uv_a.select = True
    crn_uv_a.select_edge = True
    crn_uv_b.select = True

def select_crn_uv_edge_with_shared_by_idx(crn: BMLoop, uv, force=False):
    idx = crn.face.index

    if (shared := crn.link_loop_radial_prev) != crn and shared.face.index == idx and shared_is_linked(crn, shared, uv):
        shared[uv].select_edge = True

    if force:
        crn_uv_a = crn[uv]
        crn_uv_a.select_edge = True
        for crn_a in linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
            crn_a[uv].select = True

        crn_uv_next = crn.link_loop_next
        for crn_b in linked_crn_uv_by_island_index_unordered_included(crn_uv_next, uv, idx):
            crn_b[uv].select = True
    else:
        crn_uv_a = crn[uv]
        crn_uv_a.select_edge = True
        if not crn_uv_a.select:
            for crn_a in linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                crn_a[uv].select = True

        crn_uv_next = crn.link_loop_next
        if not crn_uv_next[uv].select:
            for crn_b in linked_crn_uv_by_island_index_unordered_included(crn_uv_next, uv, idx):
                crn_b[uv].select = True


def deselect_linked_crn_uv_vert(first: BMLoop, uv: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:  # TODO: Optimize and test
            crn_uv_bm_iter.select = False

def deselect_crn_uv(first: BMLoop, uv: BMLayerItem):
    first[uv].select_edge = False

    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:  # TODO: Optimize and test
            if crn_uv_bm_iter.select:
                break
        else:
            first[uv].select = False

    second = first.link_loop_next
    bm_iter = second
    while True:
        if (bm_iter := prev_disc(bm_iter)) == second:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if second[uv].uv == crn_uv_bm_iter.uv:
            if crn_uv_bm_iter.select:
                break
        else:
            second[uv].select = False

def deselect_crn_uv_force(first: BMLoop, uv: BMLayerItem):
    first[uv].select_edge = False

    bm_iter = first
    first[uv].select = False
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False

    second = first.link_loop_next
    second[uv].select = False
    bm_iter = second
    while True:
        if (bm_iter := prev_disc(bm_iter)) == second:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if second[uv].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False

def copy_pos_to_target(crn, uv, idx):
    next_crn_co = crn.link_loop_next[uv].uv
    shared = shared_crn(crn)

    for _crn in linked_crn_uv_by_island_index_unordered_included(shared, uv, idx):
        _crn[uv].uv = next_crn_co

    crn_co = crn[uv].uv
    shared_next = shared_crn(crn).link_loop_next

    for _crn in linked_crn_uv_by_island_index_unordered_included(shared_next, uv, idx):
        _crn[uv].uv = crn_co

def copy_pos_to_target_with_select(crn, uv, idx):
    """Weld and Selects a common edge"""
    next_crn_co = crn.link_loop_next[uv].uv
    shared = shared_crn(crn)
    shared[uv].select_edge = True

    for _crn in linked_crn_uv_by_island_index_unordered_included(shared, uv, idx):
        _crn_uv = _crn[uv]
        _crn_uv.uv = next_crn_co
        _crn_uv.select = True

    crn_co = crn[uv].uv
    shared_next = shared_crn(crn).link_loop_next

    for _crn in linked_crn_uv_by_island_index_unordered_included(shared_next, uv, idx):
        _crn_uv = _crn[uv]
        _crn_uv.uv = crn_co
        _crn_uv.select = True

def weld_crn_edge_by_idx(crn: BMLoop, crn_pair, idx, uv: BMLayerItem):
    """For Weld OT"""
    coords_sum_a = Vector((0.0, 0.0))

    corners = []
    corners_append = corners.append

    first_co = crn[uv].uv
    for crn_a in crn.vert.link_loops:
        if crn_a.face.index == idx:
            crn_a_uv = crn_a[uv]
            crn_a_co = crn_a_uv.uv
            if crn_a_co == first_co:
                coords_sum_a += crn_a_co
                corners_append(crn_a_uv)

    second_co = crn_pair[uv].uv
    for crn_b in crn_pair.vert.link_loops:
        if crn_b.face.index == idx:
            crn_b_uv = crn_b[uv]
            crn_b_co = crn_b_uv.uv
            if crn_b_co == second_co:
                coords_sum_a += crn_b_co
                corners_append(crn_b_uv)

    avg_co_a = coords_sum_a / len(corners)

    for crn_ in corners:
        crn_.uv = avg_co_a

def is_flipped_uv(f, uv) -> bool:
    area = 0.0
    uvs = [l[uv].uv for l in f.loops]
    for i in range(len(uvs)):
        area += uvs[i - 1].cross(uvs[i])
    return area < 0

def point_inside_face(pt, f, uv):
    from mathutils.geometry import intersect_point_tri_2d
    corners = f.loops
    p1 = corners[0][uv].uv
    for i in range(1, len(corners)-1):
        if intersect_point_tri_2d(pt, p1, corners[i][uv].uv, corners[i+1][uv].uv):
            return True
    return False

def is_boundary_non_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(not l.face.select)

    # We get a clockwise corner, but linked to the end of the current corner
    if (next_linked_disc := crn.link_loop_radial_prev) == crn:
        return True
    if not next_linked_disc.face.select:
        return True
    return not (crn[uv].uv == next_linked_disc.link_loop_next[uv].uv and
                crn.link_loop_next[uv].uv == next_linked_disc[uv].uv)

def is_boundary_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(not l.face.hide)
    if (_shared_crn := crn.link_loop_radial_prev) == crn:
        return True
    if _shared_crn.face.hide:
        return True
    return not (crn[uv].uv == _shared_crn.link_loop_next[uv].uv and
                crn.link_loop_next[uv].uv == _shared_crn[uv].uv)

def calc_selected_uv_faces(umesh: 'types.UMesh') -> list[BMFace] | typing.Sequence[BMFace]:
    if umesh.is_full_face_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_face_selected:
            return umesh.bm.faces
        return [f for f in umesh.bm.faces if f.select]

    uv = umesh.uv
    if umesh.is_full_face_selected:
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return [f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops)]
        else:
            return [f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops)]
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return [f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops) and f.select]
    else:
        return [f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops) and f.select]

def calc_selected_uv_faces_iter(umesh: 'types.UMesh') -> 'typing.Generator[BMFace] | typing.Sequence':
    if umesh.is_full_face_deselected:
        return ()

    if umesh.sync:
        if umesh.is_full_face_selected:
            return umesh.bm.faces
        return (f for f in umesh.bm.faces if f.select)

    uv = umesh.uv
    if umesh.is_full_face_selected:
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return (f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops))
        else:
            return (f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops))
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return (f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops) and f.select)
    else:
        return (f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops) and f.select)

def calc_selected_verts(umesh: 'types.UMesh') -> list[BMVert] | typing.Any:  # noqa
    if umesh.is_full_vert_deselected:
        return []
    if umesh.is_full_vert_selected:
        return umesh.bm.verts
    return [v for v in umesh.bm.verts if v.select]

def calc_selected_edges(umesh: 'types.UMesh') -> list[BMEdge] | typing.Any:  # noqa
    if umesh.is_full_edge_deselected:
        return []
    if umesh.is_full_edge_selected:
        return umesh.bm.edges
    return [e for e in umesh.bm.edges if e.select]

def calc_visible_uv_faces(umesh) -> list[BMFace] | typing.Sequence[BMFace]:
    if umesh.is_full_face_selected:
        return umesh.bm.faces
    if umesh.sync:
        return [f for f in umesh.bm.faces if not f.hide]
    if umesh.is_full_face_deselected:
        return []
    return [f for f in umesh.bm.faces if f.select]

def calc_unselected_uv_faces(umesh: 'types.UMesh') -> list[BMFace]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return []
        return [f for f in umesh.bm.faces if not (f.select or f.hide)]
    if umesh.is_full_face_deselected:
        return []
    uv = umesh.uv
    return [f for f in umesh.bm.faces if f.select and any(not crn[uv].select_edge for crn in f.loops)]

def calc_uv_faces(umesh: 'types.UMesh', *, selected) -> list[BMFace]:
    if selected:
        return calc_selected_uv_faces(umesh)
    return calc_visible_uv_faces(umesh)

def calc_selected_uv_vert_corners(umesh: 'types.UMesh') -> list[BMLoop]:
    if umesh.is_full_vert_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_vert_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces for crn in f.loops if crn.vert.select]

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return [crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select]
    return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select]

def calc_selected_uv_corners_iter(umesh: 'types.UMesh') -> 'typing.Generator[BMLoop] | tuple':
    if umesh.sync:
        if umesh.is_full_vert_deselected:
            return ()

        if umesh.is_full_vert_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops)
        return (crn for f in umesh.bm.faces for crn in f.loops if crn.vert.select)

    if umesh.is_full_face_deselected:
        return ()

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return (crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select)
    return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select)

def calc_selected_uv_edge_corners(umesh: 'types.UMesh') -> list[BMLoop]:
    if umesh.is_full_face_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces for crn in f.loops if crn.edge.select]

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return [crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select_edge]
    return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select_edge]

def calc_visible_uv_corners(umesh: 'types.UMesh') -> list[BMLoop]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops]

    if umesh.is_full_face_deselected:
        return []
    if umesh.is_full_face_selected:
        return [crn for f in umesh.bm.faces for crn in f.loops]
    return [crn for f in umesh.bm.faces if f.select for crn in f.loops]

def calc_uv_corners(umesh: 'types.UMesh', *, selected) -> list[BMLoop]:
    if selected:
        return calc_selected_uv_vert_corners(umesh)
    return calc_visible_uv_corners(umesh)

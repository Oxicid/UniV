# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import typing
# from . import UMesh

from bmesh.types import BMesh, BMFace, BMEdge, BMVert, BMLoop, BMLayerItem
from mathutils import Vector
from collections import deque

from ..types import PyBMesh

def set_faces_tag(faces, tag=True):
    for f in faces:
        f.tag = tag

def face_centroid_uv(f: BMFace, uv_layer: BMLayerItem):
    value = Vector((0, 0))
    loops = f.loops
    for l in loops:
        value += l[uv_layer].uv
    return value / len(loops)


# Need implement disc_next disc_prev
# def calc_non_manifolds_uv(bm, uv_layer):
#     for f in bm.faces:
#         for l in f.loops:  # Running through all the neighboring faces
#             link_face = l.link_loop_radial_next.face
#             for ll in link_face.loops:
#
#                 if ll[uv_layer].uv != l[uv_layer].uv:
#                     continue
#                 # Skip manifold
#                 if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
#                         (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
#                     continue
#                 else:
#                     l[uv_layer].select = True
#                     ll[uv_layer].select = True


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

def linked_crn_uv(first: BMLoop, uv_layer: BMLayerItem):
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if first[uv_layer].uv == bm_iter[uv_layer].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_tag_b(first: BMLoop, uv_layer: BMLayerItem):
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first[uv_layer].uv == bm_iter[uv_layer].uv:
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
        if first[uv].uv == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked

def calc_crn_in_vert_by_tag(first: BMLoop):
    if not first.tag:
        return []
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.tag:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_face_index(first: BMLoop, uv_layer: BMLayerItem):
    face_index = first.face.index
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.face.index == face_index and first[uv_layer].uv == bm_iter[uv_layer].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_uv_by_face_index_b(first: BMLoop, uv: BMLayerItem, idx: int):
    first_co = first[uv].uv
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.face.index == idx and first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked

def linked_crn_by_face_index(crn):
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    linked.popleft()
    return linked

def linked_crn_by_face_index_including(crn):
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    return linked

def select_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = True

def select_crn_uv_edge(crn: BMLoop, uv_layer):
    link_crn_next = crn.link_loop_next
    select_linked_crn_uv_vert(crn, uv_layer)
    select_linked_crn_uv_vert(link_crn_next, uv_layer)

    crn_uv_a = crn[uv_layer]
    crn_uv_b = link_crn_next[uv_layer]
    crn_uv_a.select = True
    crn_uv_a.select_edge = True
    crn_uv_b.select = True

def deselect_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
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
        if first[uv].uv == crn_uv_bm_iter.uv:
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


# def deselect_crn_uv_extend(first: BMLoop, uv: BMLayerItem):
#     if not first[uv].select_edge:
#         return
#     first[uv].select_edge = False
#
#     shared_crn = first.link_loop_radial_prev
#     if shared_crn != first:
#         if first[uv].uv == shared_crn.link_loop_next[uv].uv and first.link_loop_next[uv].uv == shared_crn[uv].uv:
#             shared_crn[uv].select_edge = False
#             deselect_linked_crn_uv_vert(shared_crn, uv)
#             deselect_linked_crn_uv_vert(first, uv)
#         elif first[uv].uv == shared_crn.link_loop_next[uv].uv:
#             deselect_linked_crn_uv_vert(first, uv)
#         elif first.link_loop_next[uv].uv == shared_crn[uv].uv:
#             deselect_linked_crn_uv_vert(shared_crn, uv)
#
#
#     bm_iter = first
#     while True:
#         if (bm_iter := _prev_disc(bm_iter)) == first:
#             break
#         if not bm_iter.face.select:
#             continue
#         crn_uv_bm_iter = bm_iter[uv]
#         if first[uv].uv == crn_uv_bm_iter.uv:
#             if crn_uv_bm_iter.select:
#                 break
#         else:
#             first[uv].select = False
#
#     second = first.link_loop_next
#     bm_iter = first
#     while True:
#         if (bm_iter := _prev_disc(bm_iter)) == second:
#             break
#         if not bm_iter.face.select:
#             continue
#         crn_uv_bm_iter = bm_iter[uv]
#         if second[uv].uv == crn_uv_bm_iter.uv:
#             if crn_uv_bm_iter.select:
#                 break
#         else:
#             second[uv].select = False

def shared_crn(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn:
        return shared


def copy_pos_to_target(crn, uv, idx):
    next_crn_co = crn.link_loop_next[uv].uv
    shared = shared_crn(crn)

    source_corners = linked_crn_uv_by_face_index_b(shared, uv, idx)
    for _crn in source_corners:
        _crn[uv].uv = next_crn_co

    crn_co = crn[uv].uv
    shared_next = shared_crn(crn).link_loop_next

    source_corners = linked_crn_uv_by_face_index_b(shared_next, uv, idx)
    for _crn in source_corners:
        _crn[uv].uv = crn_co

def copy_pos_to_target_with_select(crn, uv, idx):
    """Weld and Selects a common edge"""
    next_crn_co = crn.link_loop_next[uv].uv
    shared = shared_crn(crn)
    shared[uv].select_edge = True

    source_corners = linked_crn_uv_by_face_index_b(shared, uv, idx)
    for _crn in source_corners:
        _crn_uv = _crn[uv]
        _crn_uv.uv = next_crn_co
        _crn_uv.select = True

    crn_co = crn[uv].uv
    shared_next = shared_crn(crn).link_loop_next

    source_corners = linked_crn_uv_by_face_index_b(shared_next, uv, idx)
    for _crn in source_corners:
        _crn_uv = _crn[uv]
        _crn_uv.uv = crn_co
        _crn_uv.select = True

def weld_crn_edge(crn: BMLoop, uv: BMLayerItem):
    crn_next = crn.link_loop_next
    shared = shared_crn(crn)
    shared_next = shared.link_loop_next

    index_a = crn.face.index
    index_b = shared.face.index

    corners_a = linked_crn_uv_by_face_index_b(crn, uv, index_a)
    corners_b = linked_crn_uv_by_face_index_b(crn_next, uv, index_a)

    corners_a += linked_crn_uv_by_face_index_b(shared_next, uv, index_b)
    corners_b += linked_crn_uv_by_face_index_b(shared, uv, index_b)

    coords_sum_a = Vector((0.0, 0.0))
    coords_sum_b = Vector((0.0, 0.0))

    for crn_a in corners_a:
        coords_sum_a += crn_a[uv].uv

    for crn_b in corners_b:
        coords_sum_b += crn_b[uv].uv

    avg_co_a = coords_sum_a / len(corners_a)
    avg_co_b = coords_sum_b / len(corners_b)

    for crn_a in corners_a:
        crn_a[uv].uv = avg_co_a

    for crn_b in corners_b:
        crn_b[uv].uv = avg_co_b


def is_boundary(crn: BMLoop, uv_layer: BMLayerItem):
    # assert(not l.face.select)

    # We get a clockwise corner, but linked to the end of the current corner
    if (next_linked_disc := crn.link_loop_radial_prev) == crn:
        return True
    if not next_linked_disc.face.select:
        return True
    return not (crn[uv_layer].uv == next_linked_disc.link_loop_next[uv_layer].uv and
                crn.link_loop_next[uv_layer].uv == next_linked_disc[uv_layer].uv)

def is_boundary_sync(crn: BMLoop, uv_layer: BMLayerItem):
    # assert(not l.face.hide)
    if (_shared_crn := crn.link_loop_radial_prev) == crn:
        return True
    if _shared_crn.face.hide:
        return True
    return not (crn[uv_layer].uv == _shared_crn.link_loop_next[uv_layer].uv and
                crn.link_loop_next[uv_layer].uv == _shared_crn[uv_layer].uv)

def shared_is_linked(crn: BMLoop, _shared_crn: BMLoop, uv_layer: BMLayerItem):
    return crn.link_loop_next[uv_layer].uv == _shared_crn[uv_layer].uv and \
           crn[uv_layer].uv == _shared_crn.link_loop_next[uv_layer].uv

def calc_selected_uv_faces(bm, uv_layer, sync) -> list[BMFace]:
    if PyBMesh.is_full_face_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return [f for f in bm.faces if f.select]

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops)]
        else:
            return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops)]
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select]
    else:
        return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select]

def calc_selected_uv_faces_b(umesh: 'UMesh') -> list[BMFace]:  # noqa
    if umesh.is_full_face_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_face_selected:
            return umesh.bm.faces
        return [f for f in umesh.bm.faces if f.select]

    uv = umesh.uv_layer
    if umesh.is_full_face_selected:
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return [f for f in umesh.bm.faces if all(l[uv].select for l in f.loops)]
        else:
            return [f for f in umesh.bm.faces if all(l[uv].select_edge for l in f.loops)]

    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return [f for f in umesh.bm.faces if all(l[uv].select for l in f.loops) and f.select]
    else:
        return [f for f in umesh.bm.faces if all(l[uv].select_edge for l in f.loops) and f.select]

def calc_selected_uv_faces_iter(bm, uv_layer, sync) -> 'typing.Generator[BMFace] | tuple':
    if PyBMesh.is_full_face_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return (f for f in bm.faces if f.select)

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops))
        else:
            return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops))
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select)
    else:
        return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select)

def calc_selected_verts(umesh: 'UMesh') -> list[BMVert] | typing.Any:  # noqa
    if umesh.is_full_vert_deselected:
        return []
    if umesh.is_full_vert_selected:
        return umesh.bm.verts
    return [v for v in umesh.bm.verts if v.select]

def calc_selected_edges(umesh: 'UMesh') -> list[BMEdge] | typing.Any:  # noqa
    if umesh.is_full_edge_deselected:
        return []
    if umesh.is_full_edge_selected:
        return umesh.bm.edges
    return [e for e in umesh.bm.edges if e.select]

def calc_visible_uv_faces(bm, uv_layer, sync) -> list[BMFace]:  # noqa
    if PyBMesh.is_full_face_selected(bm):
        return bm.faces
    if sync:
        return [f for f in bm.faces if not f.hide]
    if PyBMesh.is_full_face_deselected:
        return []
    return [f for f in bm.faces if f.select]

def calc_unselected_uv_faces(bm, uv, sync) -> list[BMFace]:
    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return []
        return [f for f in bm.faces if not (f.select or f.hide)]
    if PyBMesh.is_full_face_deselected(bm):
        return []
    return [f for f in bm.faces if f.select and any(not crn[uv].select_edge for crn in f.loops)]

def calc_uv_faces(bm, uv_layer, sync, *, selected) -> list[BMFace]:
    if selected:
        return calc_selected_uv_faces(bm, uv_layer, sync)
    return calc_visible_uv_faces(bm, uv_layer, sync)

def calc_selected_uv_corners(bm, uv_layer, sync) -> list[BMLoop]:
    if PyBMesh.is_full_vert_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return [l for f in bm.faces for l in f.loops]
        return [l for f in bm.faces for l in f.loops if l.vert.select]

    if PyBMesh.is_full_face_selected(bm):
        return [l for f in bm.faces for l in f.loops if l[uv_layer].select]
    return [l for f in bm.faces if f.select for l in f.loops if l[uv_layer].select]

def calc_selected_uv_corners_iter(bm, uv_layer, sync) -> 'typing.Generator[BMLoop] | tuple':
    if PyBMesh.is_full_vert_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return (l for f in bm.faces for l in f.loops)
        return (l for f in bm.faces for l in f.loops if l.vert.select)

    if PyBMesh.is_full_face_selected(bm):
        return (luv for f in bm.faces for luv in f.loops if luv[uv_layer].select)
    return (luv for f in bm.faces if f.select for luv in f.loops if luv[uv_layer].select)

def calc_visible_uv_corners(bm, sync) -> list[BMLoop]:
    if sync:
        return [luv for f in bm.faces if not f.hide for luv in f.loops]
    if PyBMesh.fields(bm).totfacesel == 0:
        return []
    return [luv for f in bm.faces if (f.select and not f.hide) for luv in f.loops]

def calc_uv_corners(bm, uv_layer, sync, *, selected) -> list[BMLoop]:
    if selected:
        return calc_selected_uv_corners(bm, uv_layer, sync)
    return calc_visible_uv_corners(bm, sync)

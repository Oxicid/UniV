# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import typing

from bmesh.types import BMesh, BMFace, BMEdge, BMVert, BMLoop, BMLayerItem
from math import isclose
from mathutils import Vector
from mathutils.geometry import area_tri, intersect_point_tri_2d
from collections import deque
from itertools import chain

from .. import types

def shared_crn(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn:
        return shared

def is_flipped_3d(crn):
    pair = crn.link_loop_radial_prev
    if pair == crn:
        return False
    return pair.vert == crn.vert

def shared_is_linked(crn: BMLoop, _shared_crn: BMLoop, uv: BMLayerItem):
    return crn.link_loop_next[uv].uv == _shared_crn[uv].uv and \
           crn[uv].uv == _shared_crn.link_loop_next[uv].uv

def is_pair(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    return crn.link_loop_next[uv].uv == _rad_prev[uv].uv and \
           crn[uv].uv == _rad_prev.link_loop_next[uv].uv

def is_pair_with_flip(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    if crn.vert == _rad_prev.vert:  # is flipped
        return crn[uv].uv == _rad_prev[uv].uv and \
            crn.link_loop_next[uv].uv == _rad_prev.link_loop_next[uv].uv
    return crn.link_loop_next[uv].uv == _rad_prev[uv].uv and \
           crn[uv].uv == _rad_prev.link_loop_next[uv].uv

def has_pair_with_ms(crn: BMLoop, uv: BMLayerItem):
    if crn.edge.seam or crn == (pair := crn.link_loop_radial_prev):
        return False
    if crn.vert == pair.vert:  # avoid flipped 3d
        return False
    return crn.link_loop_next[uv].uv == pair[uv].uv and \
           crn[uv].uv == pair.link_loop_next[uv].uv

def is_pair_by_idx(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    if crn == _rad_prev or crn.face.index != _rad_prev.face.index:
        return False
    return crn.link_loop_next[uv].uv == _rad_prev[uv].uv and \
           crn[uv].uv == _rad_prev.link_loop_next[uv].uv  # noqa


def shared_linked_crn_by_idx(crn: BMLoop, uv) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        if crn.link_loop_next[uv].uv == shared[uv].uv and crn[uv].uv == shared.link_loop_next[uv].uv:
            return shared

def shared_linked_crn_to_edge_by_idx(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
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

def calc_face_area_3d(f, scale) -> float:
    """newell cross"""
    n = Vector()
    corners = f.loops
    v_prev = corners[-1].vert.co * scale
    for crn in corners:
        v_curr = crn.vert.co * scale
        # inplace optimization ~20%: n += (v_prev.yzx - v_curr.yzx) * (v_prev.zxy + v_curr.zxy)
        v_prev_yzx = v_prev.yzx
        v_prev_zxy = v_prev.zxy

        v_prev_yzx -= v_curr.yzx
        v_prev_zxy += v_curr.zxy

        v_prev_yzx *= v_prev_zxy
        n += v_prev_yzx

        v_prev = v_curr
    return n.length

def calc_face_area_uv(f, uv) -> float:
    corners = f.loops
    if (n := len(corners)) == 4:
        l1 = corners[0][uv].uv
        l2 = corners[1][uv].uv
        l3 = corners[2][uv].uv
        l4 = corners[3][uv].uv

        return area_tri(l1, l2, l3) + area_tri(l3, l4, l1)
    elif n == 3:
        crn_a, crn_b, crn_c = corners
        return area_tri(crn_a[uv].uv, crn_b[uv].uv, crn_c[uv].uv)
    else:
        area = 0.0
        first_crn_co = corners[-1][uv].uv
        for crn in corners:
            next_crn_co = crn[uv].uv
            area += first_crn_co.cross(next_crn_co)
            first_crn_co = next_crn_co
        return abs(area) * 0.5

def calc_signed_face_area_uv(f, uv) -> float:
    area = 0.0
    corners = f.loops
    first_crn_co = corners[-1][uv].uv
    for crn in corners:
        next_crn_co = crn[uv].uv
        area += first_crn_co.cross(next_crn_co)
        first_crn_co = next_crn_co
    return area * 0.5

def calc_total_area_uv(faces, uv):
    return sum(calc_face_area_uv(f, uv) for f in faces)

def calc_total_area_3d(faces, scale):
    if scale:
        avg_scale = (sum(abs(s_) for s_ in scale) / 3)
        if all(isclose(abs(s_), avg_scale, abs_tol=0.01) for s_ in scale):
            return sum(f.calc_area() for f in faces) * avg_scale ** 2
        # newell_cross
        area = 0.0
        for f in faces:
            n = Vector()
            corners = f.loops
            v_prev = corners[-1].vert.co * scale
            for crn in corners:
                v_curr = crn.vert.co * scale
                # (inplace optimization ~20%) - n += (v_prev.yzx - v_curr.yzx) * (v_prev.zxy + v_curr.zxy)
                v_prev_yzx = v_prev.yzx
                v_prev_zxy = v_prev.zxy

                v_prev_yzx -= v_curr.yzx
                v_prev_zxy += v_curr.zxy

                v_prev_yzx *= v_prev_zxy
                n += v_prev_yzx

                v_prev = v_curr

            area += n.length
        return area * 0.5

    else:
        return sum(f.calc_area() for f in faces)


def calc_max_length_uv_crn(corners, uv) -> BMLoop:
    length = -1.0
    crn_ = None
    prev_co = corners[-1][uv].uv
    for crn in corners:
        curr_co = crn[uv].uv
        if length < (length_ := (prev_co - curr_co).length_squared):
            crn_ = crn
            length = length_
        prev_co = curr_co
    return crn_.link_loop_prev

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

def is_visible_func(sync):
    if sync:
        return lambda f: not f.hide
    else:
        return BMFace.select.__get__


def is_invisible_func(sync):
    if sync:
        return BMFace.hide.__get__
    else:
        return lambda f: not f.select

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

def linked_crn_to_vert_pair_iter(crn: BMLoop, uv, sync):
    """CW corners not reverse"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert
    iterated = False
    bm_iter = crn
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True
        # Finish CCW
        if pair_ccw in (prev_crn, crn) or (first_vert != pair_ccw.vert) or is_invisible(pair_ccw.face) or not is_pair(prev_crn, pair_ccw, uv):
            bm_iter = crn
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if (first_vert != next_crn.vert) or is_invisible(next_crn.face) or not is_pair(bm_iter, pair_cw, uv):
                    break
                yield next_crn
                bm_iter = next_crn
            break
        yield pair_ccw
        bm_iter = pair_ccw


def linked_crn_to_vert_pair(crn: BMLoop, uv, sync: bool):
    """Linked to arg corner by island index with arg corner"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if pair_ccw in (prev_crn, crn) or (first_vert != pair_ccw.vert) or is_invisible(pair_ccw.face) or not is_pair(prev_crn, pair_ccw, uv):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if (first_vert != next_crn.vert) or is_invisible(next_crn.face) or not is_pair(bm_iter, pair_cw, uv):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked

def linked_crn_to_vert_pair_with_seam(crn: BMLoop, uv, sync: bool):
    """Linked to arg corner by island index with arg corner"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if (pair_ccw in (prev_crn, crn) or
                (first_vert != pair_ccw.vert) or
                pair_ccw.edge.seam or
                is_invisible(pair_ccw.face) or
                not is_pair(prev_crn, pair_ccw, uv)
        ):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if ((first_vert != next_crn.vert)
                        or pair_cw.edge.seam
                        or is_invisible(next_crn.face)
                        or not is_pair(bm_iter, pair_cw, uv)
                ):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked

def linked_crn_to_vert_pair_by_idx_with_seam(crn: BMLoop, uv):
    """Linked to arg corner by island index with arg corner"""
    idx = crn.face.index
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if (pair_ccw in (prev_crn, crn) or
                (first_vert != pair_ccw.vert) or
                pair_ccw.edge.seam or
                pair_ccw.face.index != idx or
                not is_pair(prev_crn, pair_ccw, uv)
        ):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if ((first_vert != next_crn.vert)
                        or pair_cw.edge.seam
                        or next_crn.face.index != idx
                        or not is_pair(bm_iter, pair_cw, uv)
                ):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked


def linked_crn_uv_unordered(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    linked.remove(first)
    return linked

def linked_crn_uv_unordered_included(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    return linked

def linked_crn_uv_by_tag_b(first: BMLoop, uv: BMLayerItem):
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


# TODO: Replace with linked unordered (change logic in extend_from_linked)
def linked_crn_vert_uv_for_transform(first, uv):
    # Need tagging. tag == False - not append
    # assert utils.sync()
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first_co == bm_iter[uv].uv:
            bm_iter.tag = False
            linked.append(bm_iter)

    next_crn = first.link_loop_next
    next_crn_co = next_crn[uv].uv
    if next_crn.tag:
        next_crn.tag = False
        linked.append(next_crn)

        bm_iter = next_crn
        while True:
            if (bm_iter := prev_disc(bm_iter)) == next_crn:
                break
            if not bm_iter.tag:
                continue
            if next_crn_co == bm_iter[uv].uv:
                bm_iter.tag = False
                linked.append(bm_iter)
    return linked

def linked_crn_uv_by_crn_tag_unordered_included(crn, uv) -> list[BMLoop]:
    """Linked to arg corner by **crn** tag with arg corner and unordered"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.tag and l_crn[uv].uv == first_co]

def linked_crn_uv_by_face_tag_unordered_included(crn, uv) -> list[BMLoop]:
    """Linked to arg corner by **face** tag with arg corner and unordered"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.tag and l_crn[uv].uv == first_co]

def linked_crn_uv_by_face_index(first: BMLoop, uv: BMLayerItem):
    """Included Unordered"""
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

def linked_crn_to_vert_by_idx_3d(crn: BMLoop):
    """Linked to arg corner by island index with arg corner"""
    first_vert = crn.vert
    idx = crn.face.index
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
                if bm_iter.face.index == idx:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == crn:
            break
        if bm_iter.face.index == idx:
            linked.append(bm_iter)
    return linked

def linked_crn_to_vert_3d_iter(crn: BMLoop):
    """Linked to arg corner by visible faces"""
    first_vert = crn.vert
    bm_iter = crn

    while True:
        bm_iter_prev = bm_iter.link_loop_prev
        bm_iter = bm_iter_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert or bm_iter.face.hide:  # Skip boundary or flipped
            bm_iter = crn
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                # Skip boundary or flipped or seam clamp or hide clamp
                if first_vert != bm_iter.vert or bm_iter.face.hide:
                    break

                if bm_iter == crn:
                    break
                yield bm_iter
            break  # break first loop
        if bm_iter == crn:
            break
        yield bm_iter

def linked_crn_to_vert_with_seam_3d_iter(crn: BMLoop):
    """Linked to arg corner by visible faces"""
    first_vert = crn.vert
    bm_iter = crn

    while True:
        bm_iter_prev = bm_iter.link_loop_prev
        bm_iter = bm_iter_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert or bm_iter_prev.edge.seam or bm_iter.face.hide:  # Skip boundary or flipped
            bm_iter = crn
            while True:
                if bm_iter.edge.seam:  # clamp by seam
                    break
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                # Skip boundary or flipped or clamp by hide
                if first_vert != bm_iter.vert or bm_iter.face.hide:
                    break

                if bm_iter == crn:
                    break
                yield bm_iter
            break  # break first loop
        if bm_iter == crn:
            break
        yield bm_iter

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
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co]

def linked_crn_to_vert_by_face_index(crn):
    """Linked to vertex by face index without arg corner"""
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    linked.popleft()
    return linked

def linked_crn_to_vert_by_island_index_unordered(crn):
    """Linked to vertex by island index without arg corner"""
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx]

def select_linked_crn_uv_vert(first: BMLoop, uv: BMLayerItem):
    first_uv_co = first[uv].uv
    for crn in first.vert.link_loops:
        crn_uv = crn[uv]
        if first_uv_co == crn_uv.uv:
            crn_uv.select = True

def select_crn_uv_edge(crn: BMLoop, uv):
    crn[uv].select_edge = True
    select_linked_crn_uv_vert(crn, uv)
    select_linked_crn_uv_vert(crn.link_loop_next, uv)

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
    corners = f.loops
    prev = corners[-1][uv].uv
    for crn in corners:
        curr = crn[uv].uv
        area += prev.cross(curr)
        prev = curr
    return area < 0

def point_inside_face(pt, f, uv):
    corners = f.loops
    if (n := len(corners)) == 4:
        p1 = corners[0][uv].uv
        p2 = corners[1][uv].uv
        p3 = corners[2][uv].uv
        p4 = corners[3][uv].uv
        return intersect_point_tri_2d(pt, p1, p2, p3) or intersect_point_tri_2d(pt, p3, p4, p1)
    elif n == 3:
        crn_a, crn_b, crn_c = corners
        return intersect_point_tri_2d(pt, crn_a[uv].uv, crn_b[uv].uv, crn_c[uv].uv)
    else:
        p1 = corners[0][uv].uv
        p2 = corners[1][uv].uv
        for i in range(2, len(corners)):
            p3 = corners[i][uv].uv
            if intersect_point_tri_2d(pt, p1, p2, p3):
                return True
            p2 = p3
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
        if umesh.elem_mode == 'VERT':
            return [f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops)]
        else:
            return [f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops)]
    if umesh.elem_mode == 'VERT':
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
        if umesh.elem_mode == 'VERT':
            return (f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops))
        else:
            return (f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops))
    if umesh.elem_mode == 'VERT':
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

def calc_visible_uv_faces_iter(umesh: 'types.UMesh') -> typing.Iterable[BMFace]:
    if umesh.is_full_face_selected:
        return umesh.bm.faces
    if umesh.sync:
        return (f for f in umesh.bm.faces if not f.hide)

    if umesh.is_full_face_deselected:
        return []
    return (f for f in umesh.bm.faces if f.select)

def calc_visible_uv_faces(umesh) -> typing.Iterable[BMFace]:
    if umesh.is_full_face_selected:
        return umesh.bm.faces
    if umesh.sync:
        return [f for f in umesh.bm.faces if not f.hide]

    if umesh.is_full_face_deselected:
        return []
    return [f for f in umesh.bm.faces if f.select]

def calc_unselected_uv_faces_iter(umesh: 'types.UMesh') -> typing.Iterable[BMFace]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return []
        return (f for f in umesh.bm.faces if not (f.select or f.hide))
    else:
        if umesh.is_full_face_deselected:
            return []
        uv = umesh.uv
        return (f for f in umesh.bm.faces if f.select and not all(crn[uv].select_edge for crn in f.loops))  # TODO: Add by select_vert

def calc_unselected_uv_faces(umesh: 'types.UMesh') -> list[BMFace]:
    return list(calc_unselected_uv_faces_iter(umesh))

def calc_uv_faces(umesh: 'types.UMesh', *, selected) -> typing.Iterable[BMFace]:
    if selected:
        return calc_selected_uv_faces(umesh)
    return calc_visible_uv_faces(umesh)

def calc_selected_uv_vert_corners(umesh: 'types.UMesh') -> list[BMLoop]:
    if umesh.is_full_vert_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_vert_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select]

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return [crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select]
    return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select]

def calc_selected_uv_vert_corners_iter(umesh: 'types.UMesh') -> 'typing.Generator[BMLoop] | tuple':
    if umesh.sync:
        if umesh.is_full_vert_deselected:
            return ()

        if umesh.is_full_vert_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops)
        return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select)

    if umesh.is_full_face_deselected:
        return ()

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return (crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select)
    return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select)

def calc_selected_uv_edge_corners_iter(umesh: 'types.UMesh') -> typing.Iterable[BMLoop]:
    if umesh.sync:
        if umesh.is_full_edge_deselected:
            return ()

        if umesh.is_full_edge_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops)
        return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select)

    if umesh.is_full_face_deselected:
        return ()

    uv = umesh.uv
    if umesh.is_full_face_selected:
        return (crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select_edge)
    return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select_edge)

def calc_selected_uv_edge_corners(umesh: 'types.UMesh') -> list[BMLoop]:
    if umesh.is_full_face_deselected:
        return []

    if umesh.sync:
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select]

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

def calc_visible_uv_corners_iter(umesh: 'types.UMesh') -> typing.Iterable[BMLoop]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops)
        return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops)

    if umesh.is_full_face_deselected:
        return []
    if umesh.is_full_face_selected:
        return (crn for f in umesh.bm.faces for crn in f.loops)
    return (crn for f in umesh.bm.faces if f.select for crn in f.loops)

def calc_uv_corners(umesh: 'types.UMesh', *, selected) -> list[BMLoop]:
    if selected:
        return calc_selected_uv_vert_corners(umesh)
    return calc_visible_uv_corners(umesh)

class ShortPath:

    @staticmethod
    def vert_tag_add_adjacent_uv(heap, l_a: BMLoop, loops_prev: list[BMLoop | None], cost: list[float], uv, prioritize_corners, bound_priority_factor):
        from itertools import chain
        import heapq
        l_a_index = l_a.index
        uv_a = l_a[uv].uv

        # Loop over faces of face, but do so by first looping over loops.
        for l in chain(linked_crn_uv(l_a, uv), [l_a]):  # TODO: Add by index included and mark seam and bi-direct linked?
            #  'l_a' is already tagged, tag all adjacent.

            l.tag = False
            l_b = l.link_loop_next

            while True:
                if l_b.tag:
                    uv_b = l_b[uv].uv
                    # We know 'l_b' is not visited, check it out!
                    l_b_index = l_b.index
                    cost_cut = (uv_a - uv_b).length
                    if l_b in prioritize_corners:
                        cost_cut *= bound_priority_factor
                    cost_new = cost[l_a_index] + cost_cut

                    if cost[l_b_index] > cost_new:
                        cost[l_b_index] = cost_new
                        loops_prev[l_b_index] = l_a
                        heapq.heappush(heap, (cost_new, id(l_b), l_b))

                # This means we only step onto `l->prev` & `l->next`.
                if l_b == l.link_loop_next:
                    l_b = l.link_loop_prev.link_loop_prev
                if (l_b := l_b.link_loop_next) == l:
                    break

    @staticmethod
    def calc_path_uv_vert(isl: 'types.AdvIsland',
                          l_src: BMLoop,
                          l_dst: BMLoop,
                          exclude_corners_group: 'list[types.LoopGroup] | tuple',
                          prioritize_corners: set[BMLoop] | tuple = (),
                          bound_priority_factor=0.9) -> list[BMLoop]:
        import heapq
        from collections import deque
        path = deque()
        # BM_ELEM_TAG flag is used to store visited edges
        uv = isl.umesh.uv
        heap = []

        # NOTE: would pass BM_EDGE except we are looping over all faces anyway.
        # BM_mesh_elem_index_ensure(bm, BM_LOOP); NOTE: not needed for face tag.
        i = 0
        for f in isl:
            for crn in f.loops:
                crn.tag = True
                crn.index = i
                i += 1

        if exclude_corners_group:
            for exclude_lg in exclude_corners_group:
                for chain_corners in exclude_lg.chain_linked_corners:
                    for crn in chain_corners:
                        crn.tag = False

            # Restore tags that may have been excluded above.
            dst_corners = exclude_corners_group[0].chain_linked_corners[-1]
            src_corners = exclude_corners_group[2].chain_linked_corners[0]
            for crn in chain(dst_corners, src_corners):
                crn.tag = True

        # Allocate.
        loops_prev = [None] * i
        cost = [1e100] * i

        # Regular dijkstra the shortest path, but over UV loops instead of vertices.
        heapq.heappush(heap, (0.0, l_src.index, l_src))
        cost[l_src.index] = 0.0

        while heap:
            l = heapq.heappop(heap)[2]
            if (l.vert == l_dst.vert) and l[uv].uv == l_dst[uv].uv:
                while True:
                    path.appendleft(l)
                    if not (l := loops_prev[l.index]):
                        break
                break

            if l.tag:
                #  Adjacent loops are tagged while stepping to avoid 2x loops.
                l.tag = False
                ShortPath.vert_tag_add_adjacent_uv(heap, l, loops_prev, cost, uv, prioritize_corners, bound_priority_factor)

        return list(path)

    @staticmethod
    def path_to_loop_group_for_rect(path, umesh):
        assert path

        uv = umesh.uv
        chain_linked_corners = []
        for crn in path:
            linked = linked_crn_uv(crn, uv)
            linked.insert(0, crn)
            chain_linked_corners.append(linked)

        it = iter(path)
        prev_vert_co = next(it).vert.co

        weights = []
        for crn in it:
            cur_vert_co = crn.vert.co
            length = (prev_vert_co - cur_vert_co).length
            weights.append(length)
            prev_vert_co = cur_vert_co

        lg = types.LoopGroup(umesh)
        lg.corners = path
        lg.chain_linked_corners = chain_linked_corners
        lg.dirt = True
        lg.weights = weights
        return lg

    @staticmethod
    def calc_length_3d_and_uv_from_path(path, umesh):  # TODO: Add aspect
        uv = umesh.uv
        it = iter(path)
        first_crn = next(it)

        prev_uv_co = first_crn[uv].uv
        prev_vert_co = first_crn.vert.co

        total_length_3d = 0.0
        total_length_uv = 0.0

        for crn in it:
            cur_uv_co = crn[uv].uv
            cur_vert_co = crn.vert.co
            total_length_3d += (prev_vert_co - cur_vert_co).length
            total_length_uv += (prev_uv_co - cur_uv_co).length

            prev_uv_co = cur_uv_co
            prev_vert_co = cur_vert_co

        return total_length_3d, total_length_uv

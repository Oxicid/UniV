# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa

from bmesh.types import BMFace, BMLoop, BMLayerItem
from math import isclose
from mathutils import Vector
from mathutils.geometry import area_tri, intersect_point_tri_2d


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


def copy_pos_to_target_with_select(crn: BMLoop, uv, idx):
    """Weld and Selects a common edge"""
    from .bm_walk import linked_crn_uv_by_island_index_unordered_included
    next_crn_co = crn.link_loop_next[uv].uv
    shared = crn.link_loop_radial_prev
    shared[uv].select_edge = True

    for _crn in linked_crn_uv_by_island_index_unordered_included(shared, uv, idx):
        _crn_uv = _crn[uv]
        _crn_uv.uv = next_crn_co
        _crn_uv.select = True

    crn_co = crn[uv].uv
    shared_next = shared.link_loop_next

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
    """ NOTE: The algorithm is resistant to very small faces."""
    area = 0.0
    corners = f.loops
    prev = corners[-1][uv].uv
    for crn in corners:
        curr = crn[uv].uv
        area += prev.cross(curr)
        prev = curr
    return area < -2.99e-08  # Non-flipped faces also have a slightly negative value.


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

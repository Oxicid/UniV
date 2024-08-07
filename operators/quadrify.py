# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The code was taken and modified from the UvSquares addon: https://github.com/Radivarig/UvSquares/blob/master/uv_squares.py

import bpy

from math import hypot
from mathutils import Vector
from collections import defaultdict
from bmesh.types import BMLoopUV
from mathutils.geometry import area_tri

from .. import utils
from ..types import Islands, FaceIsland

class UNIV_OT_Quadrify(bpy.types.Operator):
    bl_idname = "uv.univ_quadrify"
    bl_label = "Quadrify"
    bl_description = "Align selected UV to rectangular distribution"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'WARNING'}, 'Active area must be UV')
            return {'CANCELLED'}

        sync = context.scene.tool_settings.use_uv_select_sync
        umeshes = utils.UMeshes(report=self.report)
        for umesh in umeshes:
            if islands := Islands.calc_selected_quad(umesh.bm, umesh.uv_layer, sync):
                for island in islands:
                    quad(umesh, island)
            umesh.update_tag = bool(islands)
        return umeshes.update()

def quad(umesh, island):
    co_and_linked_uv_corners = calc_co_and_linked_uv_corners_dict(island)
    uv = island.uv_layer

    def max_quad_uv_face_area(f):
        f_loops = f.loops
        l1 = f_loops[0][uv].uv
        l2 = f_loops[1][uv].uv
        l3 = f_loops[2][uv].uv
        l4 = f_loops[3][uv].uv

        return area_tri(l1, l2, l3) + area_tri(l3, l4, l1)

    target_face = max(island, key=max_quad_uv_face_area)
    shape_face(island.uv_layer, target_face, co_and_linked_uv_corners)
    follow_active_uv(umesh, target_face, island)

def calc_co_and_linked_uv_corners_dict(island: FaceIsland) -> defaultdict[Vector | list[BMLoopUV]]:
    uv_layer = island.uv_layer
    co_and_linked_uv_corners = defaultdict(list)
    for f in island:
        for corner in f.loops:
            uv_corner = corner[uv_layer]
            co: Vector = uv_corner.uv.copy().freeze()
            co_and_linked_uv_corners[co].append(uv_corner)  # noqa

    return co_and_linked_uv_corners

def shape_face(uv_layer, target_face, co_and_linked_uv_corners):
    corners = []
    for l in target_face.loops:
        corners.append(l[uv_layer])

    first_highest = corners[0]
    for c in corners:
        if c.uv.y > first_highest.uv.y:
            first_highest = c
    corners.remove(first_highest)

    second_highest = corners[0]
    for c in corners:
        if c.uv.y > second_highest.uv.y:
            second_highest = c

    if first_highest.uv.x < second_highest.uv.x:
        left_up = first_highest
        right_up = second_highest
    else:
        left_up = second_highest
        right_up = first_highest
    corners.remove(second_highest)

    first_lowest = corners[0]
    second_lowest = corners[1]

    if first_lowest.uv.x < second_lowest.uv.x:
        left_down = first_lowest
        right_down = second_lowest
    else:
        left_down = second_lowest
        right_down = first_lowest

    verts = [left_up, left_down, right_down, right_up]

    ratio_x, ratio_y = image_ratio()
    _min = float('inf')
    min_v = verts[0]
    for v in verts:
        if v is None:
            continue
        area = bpy.context.area
        if area.ui_type == 'UV':
            loc = area.spaces[0].cursor_location
            hyp = hypot(loc.x / ratio_x - v.uv.x, loc.y / ratio_y - v.uv.y)
            if hyp < _min:
                _min = hyp
                min_v = v

    make_uv_face_equal_rectangle(co_and_linked_uv_corners, left_up, right_up, right_down, left_down, min_v)


def make_uv_face_equal_rectangle(co_and_linked_uv_corners, left_up, right_up, right_down, left_down, start_v):
    if start_v is None:
        start_v = left_up.uv
    elif are_verts_quasi_equal(start_v, right_up):
        start_v = right_up.uv
    elif are_verts_quasi_equal(start_v, right_down):
        start_v = right_down.uv
    elif are_verts_quasi_equal(start_v, left_down):
        start_v = left_down.uv
    else:
        start_v = left_up.uv

    left_up = left_up.uv.copy().freeze()
    right_up = right_up.uv.copy().freeze()
    right_down = right_down.uv.copy().freeze()
    left_down = left_down.uv.copy().freeze()

    if start_v == left_up:
        final_scale_x = hypot_vert(left_up, right_up)
        final_scale_y = hypot_vert(left_up, left_down)
        curr_row_x = left_up.x
        curr_row_y = left_up.y

    elif start_v == right_up:
        final_scale_x = hypot_vert(right_up, left_up)
        final_scale_y = hypot_vert(right_up, right_down)
        curr_row_x = right_up.x - final_scale_x
        curr_row_y = right_up.y

    elif start_v == right_down:
        final_scale_x = hypot_vert(right_down, left_down)
        final_scale_y = hypot_vert(right_down, right_up)
        curr_row_x = right_down.x - final_scale_x
        curr_row_y = right_down.y + final_scale_y

    else:
        final_scale_x = hypot_vert(left_down, right_down)
        final_scale_y = hypot_vert(left_down, left_up)
        curr_row_x = left_down.x
        curr_row_y = left_down.y + final_scale_y

    for v in co_and_linked_uv_corners[left_up]:
        v.uv[:] = curr_row_x, curr_row_y

    for v in co_and_linked_uv_corners[right_up]:
        v.uv[:] = curr_row_x + final_scale_x, curr_row_y

    for v in co_and_linked_uv_corners[right_down]:
        v.uv[:] = curr_row_x + final_scale_x, curr_row_y - final_scale_y

    for v in co_and_linked_uv_corners[left_down]:
        v.uv[:] = curr_row_x, curr_row_y - final_scale_y


def follow_active_uv(umesh, f_act, faces):
    uv_layer = umesh.uv_layer

    # our own local walker
    def walk_face_init(faces, f_act):  # noqa
        # first tag all faces True (so we don't uv map them)
        utils.set_faces_tag(umesh.bm.faces)
        # then tag faces arg False
        utils.set_faces_tag(faces, False)
        # tag the active face True since we begin there
        f_act.tag = True

    def walk_face(f):  # noqa
        # all faces in this list must be tagged
        f.tag = True
        faces_a = [f]
        faces_b = []

        while faces_a:
            for f in faces_a:  # noqa
                for l in f.loops:  # noqa
                    l_edge = l.edge
                    if l_edge.is_manifold and not l_edge.seam:
                        l_other = l.link_loop_radial_next
                        f_other = l_other.face
                        if not f_other.tag:
                            yield l
                            f_other.tag = True
                            faces_b.append(f_other)
            # swap
            faces_a, faces_b = faces_b, faces_a
            faces_b.clear()

    def walk_edgeloop(l):  # noqa
        """
        Could make this a generic function
        """
        e_first = l.edge
        while True:
            e = l.edge  # noqa
            yield e

            # don't step past non-manifold edges
            if e.is_manifold:
                # walk around the quad and then onto the next face
                l = l.link_loop_radial_next  # noqa
                if len(l.face.verts) == 4:
                    l = l.link_loop_next.link_loop_next  # noqa
                    if l.edge is e_first:
                        break
                else:
                    break
            else:
                break

    def extrapolate_uv(fac,
                       l_a_outer, l_a_inner,
                       l_b_outer, l_b_inner):
        l_b_inner[:] = l_a_inner
        l_b_outer[:] = l_a_inner + ((l_a_inner - l_a_outer) * fac)

    def apply_uv(l_prev):
        l_a = [None, None, None, None]
        l_b = [None, None, None, None]

        l_a[0] = l_prev
        l_a[1] = l_a[0].link_loop_next
        l_a[2] = l_a[1].link_loop_next
        l_a[3] = l_a[2].link_loop_next

        #  l_b
        #  +-----------+
        #  |(3)        |(2)
        #  |           |
        #  |l_next(0)  |(1)
        #  +-----------+
        #        ^
        #  l_a   |
        #  +-----------+
        #  |l_prev(0)  |(1)
        #  |    (f)    |
        #  |(3)        |(2)
        #  +-----------+
        #  copy from this face to the one above.

        # get the other loops
        l_next = l_prev.link_loop_radial_next
        if l_next.vert != l_prev.vert:
            l_b[1] = l_next
            l_b[0] = l_b[1].link_loop_next
            l_b[3] = l_b[0].link_loop_next
            l_b[2] = l_b[3].link_loop_next
        else:
            l_b[0] = l_next
            l_b[1] = l_b[0].link_loop_next
            l_b[2] = l_b[1].link_loop_next
            l_b[3] = l_b[2].link_loop_next

        l_a_uv = [l[uv_layer].uv for l in l_a]  # noqa
        l_b_uv = [l[uv_layer].uv for l in l_b]  # noqa

        try:
            fac = edge_lengths[l_b[2].edge.index][0] / edge_lengths[l_a[1].edge.index][0]
        except ZeroDivisionError:
            fac = 1.0

        extrapolate_uv(fac,
                       l_a_uv[3], l_a_uv[0],
                       l_b_uv[3], l_b_uv[0])

        extrapolate_uv(fac,
                       l_a_uv[2], l_a_uv[1],
                       l_b_uv[2], l_b_uv[1])

    # Calculate average length per loop if needed
    umesh.bm.edges.index_update()
    edge_lengths: list[None | list[float]] = [None] * len(umesh.bm.edges)

    for f in faces:
        # we know it's a quad
        l_quad = f.loops
        l_pair_a = (l_quad[0], l_quad[2])
        l_pair_b = (l_quad[1], l_quad[3])

        for l_pair in (l_pair_a, l_pair_b):
            if edge_lengths[l_pair[0].edge.index] is None:

                edge_length_store = [-1.0]
                edge_length_accum = 0.0
                edge_length_total = 0

                for l in l_pair:
                    if edge_lengths[l.edge.index] is None:
                        for e in walk_edgeloop(l):
                            if edge_lengths[e.index] is None:
                                edge_lengths[e.index] = edge_length_store
                                edge_length_accum += e.calc_length()
                                edge_length_total += 1

                edge_length_store[0] = edge_length_accum / edge_length_total

    walk_face_init(faces, f_act)
    for l_prev_ in walk_face(f_act):
        apply_uv(l_prev_)

def image_ratio():
    ratio = 256, 256
    area = bpy.context.area
    if area and area.type == 'IMAGE_EDITOR':
        img = area.spaces[0].image
        if img and img.size[0] != 0:
            ratio = img.size
    return ratio

def are_verts_quasi_equal(a, b, allowed_error=0.00001):
    return all(abs(v) < allowed_error for v in a.uv - b.uv)

def hypot_vert(v1, v2):
    return hypot(*(v1 - v2))

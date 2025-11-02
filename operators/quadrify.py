# SPDX-FileCopyrightText: 2025 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The code was taken and modified from the UvSquares addon: https://github.com/Radivarig/UvSquares/blob/master/uv_squares.py

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import typing

from math import pi, atan2
from itertools import chain
from mathutils import Vector, Matrix
from bmesh.types import BMLoop, BMFace
from collections.abc import Callable

from .. import utils
from .. import utypes
from ..preferences import univ_settings
from ..utypes import AdvIslands, AdvIsland, UMeshes
from ..utils import linked_crn_uv_by_face_tag_unordered_included

QUAD_SIZE = 4


class UNIV_OT_Quadrify(bpy.types.Operator):
    bl_idname = "uv.univ_quadrify"
    bl_label = "Quadrify"
    bl_description = "Align selected UV to rectangular distribution"
    bl_options = {'REGISTER', 'UNDO'}

    shear: bpy.props.BoolProperty(name='Shear', default=False, description='Reduce shear within islands')
    xy_scale: bpy.props.BoolProperty(name='Scale Independently', default=True,
                                     description='Scale U and V independently')
    use_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        from ..preferences import prefs
        self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        self.mouse_pos = None
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.mouse_pos = utils.get_mouse_pos(context, event)
            return self.execute(context)

        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.shear or self.xy_scale:
            layout.prop(self, 'use_aspect')
        layout.prop(self, 'shear')
        layout.prop(self, 'xy_scale')

        layout.prop(univ_settings(), 'use_texel')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_selected = True
        self.islands_calc_type: Callable = Callable
        self.umeshes: UMeshes | None = None
        self.mouse_pos: Vector | None = None
        self.max_distance: float | None = None

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'WARNING'}, 'Active area must be UV')
            return {'CANCELLED'}

        self.umeshes = UMeshes(report=self.report)

        selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        if selected_umeshes:
            self.umeshes = selected_umeshes
            return self.quadrify_selected()
        elif unselected_umeshes and self.mouse_pos:
            self.umeshes = unselected_umeshes
            return self.quadrify_pick()
        else:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

    def quadrify_selected(self):
        texel = univ_settings().texel_density
        texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2

        counter = 0
        selected_non_quads_counter = 0
        for umesh in self.umeshes:
            umesh.update_tag = False
            if dirt_islands := AdvIslands.calc_extended_with_mark_seam(umesh):
                uv = umesh.uv
                umesh.value = umesh.check_uniform_scale(report=self.report)
                umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
                edge_lengths = []
                for d_island in dirt_islands:
                    links_static_with_quads, static_faces, non_quad_selected, quad_islands = self.split_by_static_faces_and_quad_islands(
                        d_island)
                    selected_non_quads_counter += len(non_quad_selected)
                    for isl in quad_islands:
                        utils.set_faces_tag(isl, True)
                        self.set_corner_tag_by_border_and_by_tag(isl)

                        if not edge_lengths:
                            edge_lengths = self.init_edge_sequence_from_umesh(umesh)

                        self.quad(isl, edge_lengths)
                        counter += 1
                        umesh.update_tag = True

                    if self.shear or self.xy_scale:
                        self.quad_normalize(quad_islands, umesh)

                    if univ_settings().use_texel:
                        for isl in quad_islands:

                            if isl.area_3d == -1.0:
                                isl.calc_area_3d(umesh.value)
                            if isl.area_uv == -1.0:
                                isl.calc_area_uv()

                            isl.calc_bbox()
                            isl.set_texel(texel, texture_size)

                    for static_crn, quad_corners in links_static_with_quads:
                        static_co = static_crn[uv].uv
                        min_dist_quad_crn = min(quad_corners, key=lambda q_crn: (q_crn[uv].uv - static_co).length)
                        static_co[:] = min_dist_quad_crn[uv].uv

        if selected_non_quads_counter:
            self.report({'WARNING'}, f"Ignored {selected_non_quads_counter} non-quad faces")
        elif not counter:
            return self.umeshes.update()

        self.umeshes.silent_update()
        return {'FINISHED'}

    @staticmethod
    def init_edge_sequence_from_umesh(umesh: utypes.UMesh) -> list[None | float]:
        idx = 0
        for f in umesh.bm.faces:
            for crn in f.loops:
                crn.index = idx
                idx += 1

        return [None] * umesh.total_corners

    @staticmethod
    def init_edge_sequence_from_island(island: utypes.FaceIsland) -> list[None | float]:
        idx = 0
        for f in island:
            for crn in f.loops:
                crn.index = idx
                idx += 1

        return [None] * idx

    def quad_normalize(self, quad_islands, umesh):
        # adjust and normalize
        quad_islands = AdvIslands(quad_islands, umesh)
        quad_islands.calc_tris_simple()
        quad_islands.calc_flat_uv_coords(save_triplet=True)
        quad_islands.calc_flat_unique_uv_coords()
        quad_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
        quad_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale
        from .texel import UNIV_OT_Normalize_VIEW3D
        for isl in quad_islands:
            old_center = isl.bbox.center
            isl.value = old_center
            new_center = UNIV_OT_Normalize_VIEW3D.individual_scale(self, isl)  # noqa
            isl.value = new_center
            if len(quad_islands) == 1:
                isl.set_position(old_center, new_center)
        if len(quad_islands) > 1:
            tot_area_uv, tot_area_3d = UNIV_OT_Normalize_VIEW3D.avg_by_frequencies(self, quad_islands)  # noqa
            UNIV_OT_Normalize_VIEW3D.normalize(self, quad_islands, tot_area_uv, tot_area_3d)  # noqa

    def quadrify_pick(self):
        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)

        for umesh in self.umeshes:
            if dirt_islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                for d_island in dirt_islands:
                    hit.find_nearest_island_by_crn(d_island)
        if not hit:
            self.report({'WARNING'}, "Islands not found")
            return {'CANCELLED'}

        links_static_with_quads, static_faces, quad_islands = self.split_by_static_faces_and_quad_islands_pick(
            hit.island)
        if not quad_islands:
            self.report({'WARNING'}, f"All {len(static_faces)} faces is non-quad")
            return {'CANCELLED'}

        for isl in quad_islands:
            utils.set_faces_tag(isl, True)
            self.set_corner_tag_by_border_and_by_tag(isl)
            edge_lengths = self.init_edge_sequence_from_island(isl)
            self.quad(isl, edge_lengths)

        umesh = hit.island.umesh
        uv = umesh.uv
        umesh.value = umesh.check_uniform_scale(report=self.report)
        umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0

        if univ_settings().use_texel:
            texel = univ_settings().texel_density
            texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2

            for isl in quad_islands:
                if isl.area_3d == -1.0:
                    isl.calc_area_3d(umesh.value)
                if isl.area_uv == -1.0:
                    isl.calc_area_uv()

                isl.calc_bbox()
                isl.set_texel(texel, texture_size)

        if self.shear or self.xy_scale:
            self.quad_normalize(quad_islands, umesh)

        for static_crn, quad_corners in links_static_with_quads:
            static_co = static_crn[uv].uv
            min_dist_quad_crn = min(quad_corners, key=lambda q_crn: (q_crn[uv].uv - static_co).length)
            static_co[:] = min_dist_quad_crn[uv].uv

        hit.island.umesh.update()
        if static_faces:
            self.report({'WARNING'}, f"Ignored {len(static_faces)} non-quad faces")
        return {'FINISHED'}

    def split_by_static_faces_and_quad_islands(self, island):
        umesh = island.umesh
        uv = umesh.uv
        quad_faces = []
        selected_non_quads = []
        static_faces = []
        face_select_get = utils.face_select_get_func(umesh)

        for f in island:
            if face_select_get(f):
                if len(f.loops) == QUAD_SIZE:
                    quad_faces.append(f)
                else:
                    selected_non_quads.append(f)
            else:
                static_faces.append(f)

        if not (static_faces or selected_non_quads):  # Full quad case
            return [], static_faces, selected_non_quads, [island]
        elif len(static_faces) + len(selected_non_quads) == len(island):  # Non quad case
            return [], static_faces, selected_non_quads, []

        utils.set_faces_tag(quad_faces)
        links_static_with_quads = self.store_links_static_with_quads(chain(static_faces, selected_non_quads), uv)
        fake_umesh = umesh.fake_umesh(quad_faces)
        # Calc sub-islands
        islands = [AdvIslands.island_type(i, umesh) for i in AdvIslands.calc_iter_ex(fake_umesh)]
        return links_static_with_quads, static_faces, selected_non_quads, islands

    def split_by_static_faces_and_quad_islands_pick(self, island) -> tuple[list[tuple[BMLoop, list[BMLoop]]], list[BMFace], list[AdvIsland]]:
        umesh = island.umesh
        uv = umesh.uv
        quad_faces = []
        static_faces = []

        for f in island:
            if len(f.loops) == QUAD_SIZE:
                quad_faces.append(f)
            else:
                static_faces.append(f)

        if not static_faces:
            return [], static_faces, [island]
        elif len(static_faces) == len(island):
            return [], static_faces, []

        utils.set_faces_tag(quad_faces)
        links_static_with_quads = self.store_links_static_with_quads(static_faces, uv)
        fake_umesh = umesh.fake_umesh(quad_faces)
        islands = [AdvIslands.island_type(i, umesh) for i in AdvIslands.calc_iter_ex(fake_umesh)]
        return links_static_with_quads, static_faces, islands

    @staticmethod
    def store_links_static_with_quads(faces: typing.Iterable[BMFace], uv):
        links_static_with_quads = []
        for f in faces:
            for crn in f.loops:
                if linked_corners := linked_crn_uv_by_face_tag_unordered_included(crn, uv):
                    links_static_with_quads.append((crn, linked_corners))
        return links_static_with_quads

    def quad(self, island: AdvIsland, edge_lengths):
        uv = island.umesh.uv
        max_quad_uv_face_area = self.get_face_score_fn(uv)

        target_face = max(island, key=max_quad_uv_face_area)
        temp_coords = [crn[uv].uv.copy() for crn in target_face.loops]
        self.face_to_rect(temp_coords)

        for crn, uv_co in zip(target_face.loops, temp_coords):
            for l_crn in linked_crn_uv_by_face_tag_unordered_included(crn, uv):
                l_crn[uv].uv = uv_co

        follow_active_uv(target_face, island, edge_lengths)

    def face_to_rect(self, coords: 'typing.MutableSequence[Vector] | list[Vector]'):
        best_crn_idx: int = self.get_best_crn_idx(coords)

        # Orient
        curr_uv_co = coords[best_crn_idx]
        prev_uv_co = coords[best_crn_idx - 1]
        next_uv_co = coords[0] if best_crn_idx == (len(coords) - 1) else coords[best_crn_idx + 1]

        delta_a = prev_uv_co - curr_uv_co
        delta_b = next_uv_co - curr_uv_co

        min_angle_a = -utils.find_min_rotate_angle(atan2(*delta_a.normalized()))
        min_angle_b = -utils.find_min_rotate_angle(atan2(*delta_b.normalized()))

        min_angle_is_prev = True
        min_angle = min_angle_a

        if abs(min_angle_b) < abs(min_angle_a):
            min_angle_is_prev = False
            min_angle = min_angle_b

        pivot = curr_uv_co.copy()
        rot_matrix = Matrix.Rotation(-min_angle, 2)
        diff = pivot - (rot_matrix @ pivot)
        for crn_co in coords:
            crn_co.rotate(rot_matrix)
            crn_co += diff

        # Rect next or prev corner
        delta_prev = (prev_uv_co - curr_uv_co).normalized()
        delta_next = (next_uv_co - curr_uv_co).normalized()

        # Fix zero lengths
        if delta_prev == Vector((0.0, 0.0)):
            prev_uv_co.y += 0.01
            delta_prev = Vector((0.0, 1.0))

        if delta_next == Vector((0.0, 0.0)):
            next_uv_co.x += 0.01
            delta_next = Vector((1.0, 0.0))

        # Get minimal rotation angle by orthogonal vectors
        if min_angle_is_prev:
            orto_pos = delta_prev.orthogonal()
            orto_neg = orto_pos.copy()
            orto_neg.negate()

            angle_a = delta_next.angle_signed(orto_pos, 0.0)
            angle_b = delta_next.angle_signed(orto_neg, 0.0)
        else:
            orto_pos = delta_next.orthogonal()
            orto_neg = orto_pos.copy()
            orto_neg.negate()

            angle_a = delta_prev.angle_signed(orto_pos, 0.0)
            angle_b = delta_prev.angle_signed(orto_neg, 0.0)

        min_angle = angle_a
        if abs(angle_b) < abs(angle_a):
            min_angle = angle_b

        rot_matrix = Matrix.Rotation(-min_angle, 2)
        diff = pivot - (rot_matrix @ pivot)

        if min_angle_is_prev:
            next_uv_co.rotate(rot_matrix)
            next_uv_co += diff
        else:
            prev_uv_co.rotate(rot_matrix)
            prev_uv_co += diff

        # Complete the rectangle.
        p3 = prev_uv_co + (next_uv_co - curr_uv_co)
        coords[best_crn_idx - 2] = p3

    @staticmethod
    def get_best_crn_idx(coords: list[Vector]):
        assert len(coords) == QUAD_SIZE
        angle_90 = pi / 2

        max_score = -1.0
        arg_max = 0
        for i in range(QUAD_SIZE):
            curr_uv_co = coords[i]
            prev_uv_co = coords[i - 1]
            next_uv_co = coords[0] if i == (QUAD_SIZE - 1) else coords[i + 1]

            delta_a = prev_uv_co - curr_uv_co
            delta_b = next_uv_co - curr_uv_co

            angle = delta_a.angle(delta_b, pi)
            rightness = 1.0 - min(abs(angle - angle_90) / angle_90, 1.0)

            dist = delta_a.length * delta_b.length

            score = rightness * 0.6 + dist * 0.01

            if score > max_score:
                max_score = score
                arg_max = i
        return arg_max

    @staticmethod
    def get_face_score_fn(uv_):
        def catcher(uv):
            def get_face_score_(f: BMFace):
                def calc_angle_2d():
                    c = l[uv].uv
                    prev = l.link_loop_prev[uv].uv
                    next_ = l.link_loop_next[uv].uv
                    return (prev - c).angle_signed(next_ - c, pi)

                # priority 90 degrees
                rightness = 0.0
                for l in f.loops:
                    a2d = abs(calc_angle_2d())
                    rightness += 1.0 - min(abs(a2d - pi/2) / (pi/2), 1.0)
                rightness /= QUAD_SIZE

                # diff between 2D and 3D angle, less == better
                angle_error = 0.0
                for l in f.loops:
                    a2d = calc_angle_2d()
                    a3d = l.calc_angle()
                    angle_error += abs(a2d - a3d)
                angle_error /= QUAD_SIZE
                angle_score = 1.0 - min(angle_error / pi, 1.0)  # normalize [0..1]

                # slight priority by uv area
                area2d = utils.calc_face_area_uv(f, uv)
                import math
                area_boost = math.log1p(area2d) * 0.1

                score = rightness * 0.6 + angle_score * 0.3 + area_boost
                return score
            return get_face_score_
        return catcher(uv_)

    @staticmethod
    def set_corner_tag_by_border_and_by_tag(island: AdvIsland):
        is_boundary = utils.is_boundary_func(island.umesh, invisible_check=False)
        for crn in island.corners_iter():
            if not crn.link_loop_radial_prev.face.tag:
                crn.tag = False
            else:
                crn.tag = not is_boundary(crn)


def follow_active_uv(f_act, island: AdvIsland, edge_lengths):
    uv = island.umesh.uv  # noqa

    def walk_face(f: BMFace):  # noqa
        # all faces in this list must be tagged
        f.tag = False
        faces_a = [f]
        faces_b = []

        while faces_a:
            for f in faces_a:  # noqa
                for l in f.loops:  # noqa
                    if l.tag:
                        l_other = l.link_loop_radial_prev
                        f_other = l_other.face
                        if f_other.tag:
                            yield l
                            f_other.tag = False
                            faces_b.append(f_other)
            # swap
            faces_a, faces_b = faces_b, faces_a
            faces_b.clear()

    def extrapolate_uv(fac,
                       l_a_outer, l_a_inner,
                       l_b_outer, l_b_inner):
        l_b_inner[:] = l_a_inner
        l_b_outer[:] = l_a_inner + ((l_a_inner - l_a_outer) * fac)

    def apply_uv(l_prev: BMLoop):
        l_a: list[BMLoop | None] = [None, None, None, None]  # TODO: Array convert to vars
        l_b: list[BMLoop | None] = [None, None, None, None]

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
        l_next = l_prev.link_loop_radial_prev
        assert l_next != l_prev
        l_b[1] = l_next
        l_b[0] = l_b[1].link_loop_next
        l_b[3] = l_b[0].link_loop_next
        l_b[2] = l_b[3].link_loop_next

        l_a_uv: list[Vector] = [l[uv].uv for l in l_a]  # noqa
        l_b_uv: list[Vector] = [l[uv].uv for l in l_b]  # noqa

        try:
            fac = edge_lengths[l_b[2].index] / edge_lengths[l_a[1].index]
        except ZeroDivisionError:
            fac = 1.0

        extrapolate_uv(fac,
                       l_a_uv[3], l_a_uv[0],
                       l_b_uv[3], l_b_uv[0])

        extrapolate_uv(fac,
                       l_a_uv[2], l_a_uv[1],
                       l_b_uv[2], l_b_uv[1])

    calc_avg_ring_length(edge_lengths, island)

    f_act.tag = False
    for l_prev_ in walk_face(f_act):
        apply_uv(l_prev_)


def calc_avg_ring_length(edge_lengths, island):
    for f in island:
        for ring_crn in f.loops:
            if edge_lengths[ring_crn.index] is None:
                corners = get_ring_corners_from_crn(ring_crn)

                avg_length = sum(crn.edge.calc_length() for crn in corners) / len(corners)
                for crn in corners:
                    edge_lengths[crn.index] = avg_length


# TODO: This algorithm does not always pass through all the edges, so we have to pass all 4 edges through this algorithm
def get_ring_corners_from_crn(first_crn: BMLoop):
    corners = [first_crn]

    # first direction
    iter_crn = first_crn
    while True:
        iter_crn = iter_crn.link_loop_next.link_loop_next
        corners.append(iter_crn)
        if not iter_crn.tag:
            break

        iter_crn = iter_crn.link_loop_radial_prev
        if iter_crn == first_crn:  # is circular
            return corners
        corners.append(iter_crn)

    # other dir
    if first_crn.tag:
        iter_crn = first_crn.link_loop_radial_prev
        if not iter_crn.tag:
            return corners

        while True:
            iter_crn = iter_crn.link_loop_next.link_loop_next
            corners.append(iter_crn)

            if not iter_crn.tag:
                break
            iter_crn = iter_crn.link_loop_radial_prev
            corners.append(iter_crn)
    return corners

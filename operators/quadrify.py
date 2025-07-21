# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The code was taken and modified from the UvSquares addon: https://github.com/Radivarig/UvSquares/blob/master/uv_squares.py

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from itertools import chain
from mathutils import Vector
from bmesh.types import BMLoopUV, BMLoop, BMFace
from collections.abc import Callable
from mathutils.geometry import area_tri

from .. import utils
from .. import types
from ..types import AdvIslands, AdvIsland
from ..utils import linked_crn_uv_by_face_tag_unordered_included

class UNIV_OT_Quadrify(bpy.types.Operator):
    bl_idname = "uv.univ_quadrify"
    bl_label = "Quadrify"
    bl_description = "Align selected UV to rectangular distribution"
    bl_options = {'REGISTER', 'UNDO'}

    shear: bpy.props.BoolProperty(name='Shear', default=False, description='Reduce shear within islands')
    xy_scale: bpy.props.BoolProperty(name='Scale Independently', default=True, description='Scale U and V independently')
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_selected = True
        self.islands_calc_type: Callable = Callable
        self.umeshes: types.UMeshes | None = None
        self.mouse_pos: Vector | None = None
        self.max_distance: float | None = None

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'WARNING'}, 'Active area must be UV')
            return {'CANCELLED'}

        self.umeshes = types.UMeshes(report=self.report)

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
                    links_static_with_quads, static_faces, non_quad_selected, quad_islands = self.split_by_static_faces_and_quad_islands(d_island)
                    selected_non_quads_counter += len(non_quad_selected)
                    for isl in quad_islands:
                        utils.set_faces_tag(isl, True)
                        set_corner_tag_by_border_and_by_tag(isl)  # TODO: Preserve flipped 3D

                        if not edge_lengths:
                            edge_lengths = self.init_edge_sequence_from_umesh(umesh)

                        quad(isl, edge_lengths)
                        counter += 1
                        umesh.update_tag = True

                    if self.shear or self.xy_scale:
                        self.quad_normalize(quad_islands, umesh)

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
    def init_edge_sequence_from_umesh(umesh: types.UMesh) -> list[None | float]:
        idx = 0
        for f in umesh.bm.faces:
            for crn in f.loops:
                crn.index = idx
                idx += 1

        return [None] * umesh.total_corners

    @staticmethod
    def init_edge_sequence_from_island(island: types.FaceIsland) -> list[None | float]:
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
        hit = types.IslandHit(self.mouse_pos, self.max_distance)

        for umesh in self.umeshes:
            if dirt_islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                for d_island in dirt_islands:
                    hit.find_nearest_island_by_crn(d_island)
        if not hit:
            self.report({'WARNING'}, "Islands not found")
            return {'CANCELLED'}

        links_static_with_quads, static_faces, quad_islands = self.split_by_static_faces_and_quad_islands_pick(hit.island)
        if not quad_islands:
            self.report({'WARNING'}, f"All {len(static_faces)} faces is non-quad")
            return {'CANCELLED'}

        for isl in quad_islands:
            utils.set_faces_tag(isl, True)
            set_corner_tag_by_border_and_by_tag(isl)
            edge_lengths = self.init_edge_sequence_from_island(isl)
            quad(isl, edge_lengths)

        umesh = hit.island.umesh
        uv = umesh.uv
        umesh.value = umesh.check_uniform_scale(report=self.report)
        umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
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
                if len(f.loops) == 4:
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

    def split_by_static_faces_and_quad_islands_pick(self, island):
        umesh = island.umesh
        uv = umesh.uv
        quad_faces = []
        static_faces = []

        for f in island:
            if len(f.loops) == 4:
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
    def store_links_static_with_quads(faces, uv):
        links_static_with_quads = []
        for f in faces:
            for crn in f.loops:
                if linked_corners := linked_crn_uv_by_face_tag_unordered_included(crn, uv):
                    links_static_with_quads.append((crn, linked_corners))
        return links_static_with_quads

def set_corner_tag_by_border_and_by_tag(island: AdvIsland):
    uv = island.umesh.uv
    for crn in island.corners_iter():
        prev = crn.link_loop_radial_prev
        if crn.edge.seam or crn == prev or not prev.face.tag:
            crn.tag = False
            continue
        crn.tag = utils.is_pair(crn, prev, uv)

def quad(island: AdvIsland, edge_lengths):
    uv = island.umesh.uv

    def max_quad_uv_face_area(f):
        f_loops = f.loops
        l1 = f_loops[0][uv].uv
        l2 = f_loops[1][uv].uv
        l3 = f_loops[2][uv].uv
        l4 = f_loops[3][uv].uv

        return area_tri(l1, l2, l3) + area_tri(l3, l4, l1)

    # TODO: Find most quare and large target face
    target_face = max(island, key=max_quad_uv_face_area)
    co_and_linked_uv_corners = calc_co_and_linked_uv_corners_dict(target_face, island.umesh.uv)
    shape_face(uv, target_face, co_and_linked_uv_corners)
    follow_active_uv(target_face, island, edge_lengths)

def calc_co_and_linked_uv_corners_dict(f, uv) -> dict[Vector, list[BMLoopUV]]:
    co_and_linked_uv_corners = {}
    for crn in f.loops:
        co: Vector = crn[uv].uv.copy().freeze()
        corners = linked_crn_uv_by_face_tag_unordered_included(crn, uv)
        co_and_linked_uv_corners[co] = [crn[uv] for crn in corners]

    return co_and_linked_uv_corners

def shape_face(uv, target_face, co_and_linked_uv_corners):
    corners = []
    for l in target_face.loops:
        corners.append(l[uv])

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

    make_uv_face_equal_rectangle(co_and_linked_uv_corners, left_up, right_up, right_down, left_down)


def make_uv_face_equal_rectangle(co_and_linked_uv_corners, left_up, right_up, right_down, left_down):
    left_up = left_up.uv.copy().freeze()
    right_up = right_up.uv.copy().freeze()
    right_down = right_down.uv.copy().freeze()
    left_down = left_down.uv.copy().freeze()

    final_scale_x = (left_up - right_up).length
    final_scale_y = (left_up - left_down).length
    curr_row_x = left_up.x
    curr_row_y = left_up.y

    for v in co_and_linked_uv_corners[left_up]:
        v.uv[:] = curr_row_x, curr_row_y

    for v in co_and_linked_uv_corners[right_up]:
        v.uv[:] = curr_row_x + final_scale_x, curr_row_y

    for v in co_and_linked_uv_corners[right_down]:
        v.uv[:] = curr_row_x + final_scale_x, curr_row_y - final_scale_y

    for v in co_and_linked_uv_corners[left_down]:
        v.uv[:] = curr_row_x, curr_row_y - final_scale_y


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
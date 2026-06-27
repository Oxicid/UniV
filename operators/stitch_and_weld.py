# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

_needs_reload = "bpy" in locals()

import bpy
import bl_math
import collections
from itertools import chain
from mathutils import Vector

from bmesh.types import BMLoop
from bpy.props import *

from .. import draw
from .. import utils
from .. import utypes
from ..utypes import (
    BBox,
    UMeshes,
    Islands,
    AdvIslands,
    AdvIsland,
    LoopGroup,
    LoopGroups
)
from ..preferences import prefs, univ_settings

if _needs_reload:
    from .. import reload
    reload.reload(globals())


class Stitch:
    def __init__(self):
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_position: Vector | None = None
        self.padding = 0.0
        self.zero_area_count = 0
        self.flipped_3d_count = 0
        if not hasattr(self, 'update_seams'):
            self.update_seams = True
        if not hasattr(self, 'between'):
            self.between = False

    def stitch(self):
        self.zero_area_count = 0
        self.flipped_3d_count = 0
        for umesh in self.umeshes:

            # This contains `Targets` and potentially `Transformed` islands, which will be sorted later.
            if self.between:
                islands = AdvIslands.calc_extended_with_mark_seam(umesh)
            else:
                islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            if len(islands) <= 1:
                continue

            # Prepare for stitch
            islands.indexing()
            umesh.set_corners_tag(False)
            if self.between:
                target_islands = islands.islands.copy()
            else:
                if umesh.elem_mode in ('VERT', 'EDGE'):
                    target_islands = [
                        isl for isl in islands if
                        utypes.IslandsBase.island_filter_is_any_edge_selected(isl.faces, umesh)]
                else:
                    target_islands = [
                        isl for isl in islands if
                        utypes.IslandsBase.island_filter_is_any_face_selected(isl.faces, umesh)]

            self.sort_by_dist_to_mouse_or_sel_edge_length(target_islands, umesh)

            if not target_islands:
                continue

            for t_isl in target_islands:
                t_isl.select_state = True

            exclude_indexes = {-1}
            for ref_isl in target_islands:
                if not ref_isl.tag:
                    continue
                ref_isl.tag = False
                balanced_target_islands = []

                temp_exclude_indexes = exclude_indexes.copy()
                temp_exclude_indexes.add(ref_isl[0].index)
                # TODO: Need adapt to flipped
                self.set_selected_boundary_tag_with_exclude_face_idx(ref_isl, temp_exclude_indexes)

                loop_groups = LoopGroups.calc_by_boundary_crn_tags_v2(ref_isl)
                filtered = self.split_lg_for_stitch(loop_groups)
                filtered.sort(key=lambda lg_: lg_.length_uv, reverse=True) # Prioritize multi-stitch by length

                if filtered:
                    # An island may not have any selected edges, but it can still be a reoriented island.
                    # Therefore, we add it to the exclude list, making sure an LG exists.
                    exclude_indexes.add(ref_isl[0].index)

                for ref_lg in filtered:
                    trans_lg = ref_lg.calc_shared_group_for_stitch()
                    trans_isl_index = trans_lg[0].face.index
                    exclude_indexes.add(trans_isl_index)

                    trans_isl = islands[trans_isl_index]
                    if trans_isl.select_state:
                        trans_isl.area_3d = ref_lg.length_3d
                        balanced_target_islands.append(trans_isl)

                    if self.padding:
                        self.reorient_to_target_with_padding(ref_isl, trans_isl, ref_lg, trans_lg, correct_flip=getattr(self, 'correct_flip', True))
                    else:
                        self.reorient_to_target(ref_isl, trans_isl, ref_lg, trans_lg, correct_flip=getattr(self, 'correct_flip', True))
                    umesh.update_tag = True

                while True:
                    stack = []
                    for balance_isl in balanced_target_islands:
                        if balance_isl.tag:
                            lg = self.balancing_filter_for_lgs(balance_isl, exclude_indexes)
                            if lg:
                                trans_lg = lg.calc_shared_group_for_stitch()
                                trans_isl_index = trans_lg[0].face.index
                                exclude_indexes.add(trans_isl_index)

                                trans_isl = islands[trans_isl_index]
                                if trans_isl.select_state:
                                    trans_isl.area_3d = lg.length_3d
                                    stack.append(trans_isl)

                                if self.padding:
                                    # NOTE: ref_isl is not the island from which ref_lg(lg) is derived
                                    # TODO: Check to pass balance_isl
                                    self.reorient_to_target_with_padding(ref_isl, trans_isl, lg, trans_lg, correct_flip=getattr(self, 'correct_flip', True))
                                else:
                                    self.reorient_to_target(ref_isl, trans_isl, lg, trans_lg, correct_flip=getattr(self, 'correct_flip', True))

                    balanced_target_islands = [b_isl for b_isl in balanced_target_islands if b_isl.tag]
                    balanced_target_islands.extend(stack)

                    if not balanced_target_islands:
                        break

        if self.zero_area_count:
            self.report({'WARNING'}, f'Found {self.zero_area_count} zero length edge loop. Use inspect tools to find the problem')  # noqa
        if self.flipped_3d_count:
            self.report({'WARNING'}, f'Found {self.flipped_3d_count} loops with 3D flipped faces. '  # noqa
                                     f'For correct result need recalculate normals')

    def reorient_to_target(self, ref_isl: AdvIsland, trans: AdvIsland, ref_lg: LoopGroup, trans_lg: LoopGroup, correct_flip=True):
        uv = ref_isl.umesh.uv

        is_flipped_3d = False
        if correct_flip:
            is_flipped_3d = trans_lg.is_flipped_3d
            self.flipped_3d_count += is_flipped_3d
            if is_flipped_3d:
                if (ref_lg.calc_signed_face_area() < 0) == (trans_lg.calc_signed_face_area() < 0):
                    trans.scale_simple(Vector((1, -1)))
            elif (ref_lg.calc_signed_face_area() < 0) != (trans_lg.calc_signed_face_area() < 0):
                trans.scale_simple(Vector((1, -1)))

        if ref_lg.is_cyclic:
            bbox, bbox_margin_corners = BBox.calc_bbox_with_extrema_corners(ref_lg, uv)
            xmin_crn, xmax_crn, ymin_crn, ymax_crn = bbox_margin_corners

            if bbox.width > bbox.height:
                pt_a1 = xmin_crn[uv].uv
                pt_a2 = xmax_crn[uv].uv

                if is_flipped_3d:
                    pt_b1 = utils.shared_crn(xmin_crn)[uv].uv
                    pt_b2 = utils.shared_crn(xmax_crn)[uv].uv
                else:
                    pt_b1 = utils.shared_crn(xmin_crn).link_loop_next[uv].uv
                    pt_b2 = utils.shared_crn(xmax_crn).link_loop_next[uv].uv
            else:
                pt_a1 = ymin_crn[uv].uv
                pt_a2 = ymax_crn[uv].uv

                if is_flipped_3d:
                    pt_b1 = utils.shared_crn(ymin_crn)[uv].uv
                    pt_b2 = utils.shared_crn(ymax_crn)[uv].uv
                else:
                    pt_b1 = utils.shared_crn(ymin_crn).link_loop_next[uv].uv
                    pt_b2 = utils.shared_crn(ymax_crn).link_loop_next[uv].uv

            # Rotate
            normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
            normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

            rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
            trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

            # Move
            center_ref = ref_lg.calc_bbox().center
            center_trans = trans_lg.calc_bbox().center
            trans.set_position(center_ref, center_trans)

            # Scale
            length_a = (pt_a1 - pt_a2).length
            length_b = (pt_b1 - pt_b2).length

            if length_a < 1e-06 or length_b < 1e-06:
                self.zero_area_count += 1
            else:
                scale = length_a / length_b
                trans.scale(Vector((scale, scale)), center_ref)

            trans_lg.copy_coords_from_ref(ref_lg, clean_seams=self.update_seams)

        else:
            pt_a1, pt_a2 = ref_lg.calc_begin_end_pt()
            pt_b1, pt_b2 = trans_lg.calc_begin_end_pt()

            # Rotate
            normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
            normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

            rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
            trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

            # Scale
            normal_a = pt_a1 - pt_a2
            normal_b = pt_b1 - pt_b2
            length_a = normal_a.length
            length_b = normal_b.length

            if length_a < 1e-06 or length_b < 1e-06:
                self.zero_area_count += 1
            else:
                scale = length_a / length_b
                trans.scale_simple(Vector((scale, scale)))

            # Move
            trans.set_position(pt_a1, pt_b1)
            trans_lg.copy_coords_from_ref(ref_lg, clean_seams=self.update_seams)

        # Select shared edges.
        if bpy.context.scene.tool_settings.uv_sticky_select_mode != 'DISABLED':
            set_select_edge = utils.edge_select_linked_set_func(trans_lg.umesh)
            for crn in ref_lg:
                set_select_edge(crn.link_loop_radial_prev, True)


    def reorient_to_target_with_padding(self, ref_isl: AdvIsland, trans: AdvIsland, ref_lg: LoopGroup, trans_lg: LoopGroup, correct_flip=True):
        uv = ref_isl.umesh.uv

        ref_is_flipped = False
        trans_is_flipped = False
        is_flipped_3d = False

        if correct_flip:
            is_flipped_3d = trans_lg.is_flipped_3d
            if trans_lg.is_flipped_3d:
                ref_is_flipped = ref_lg.calc_signed_face_area() < 0
                trans_is_flipped = trans_lg.calc_signed_face_area() < 0
                if ref_is_flipped == trans_is_flipped:
                    # trans_is_flipped ^= 1
                    trans.scale_simple(Vector((1, -1)))
            else:
                ref_is_flipped = ref_lg.calc_signed_face_area() < 0
                trans_is_flipped = trans_lg.calc_signed_face_area() < 0
                if ref_is_flipped != trans_is_flipped:
                    # trans_is_flipped ^= 1
                    trans.scale_simple(Vector((1, -1)))

        if ref_lg.is_cyclic:
            bbox, bbox_margin_corners = BBox.calc_bbox_with_extrema_corners(ref_lg, uv)
            xmin_crn, xmax_crn, ymin_crn, ymax_crn = bbox_margin_corners

            if bbox.width > bbox.height:
                pt_a1 = xmin_crn[uv].uv
                pt_a2 = xmax_crn[uv].uv

                if is_flipped_3d:
                    pt_b1 = utils.shared_crn(xmin_crn)[uv].uv
                    pt_b2 = utils.shared_crn(xmax_crn)[uv].uv
                else:
                    pt_b1 = utils.shared_crn(xmin_crn).link_loop_next[uv].uv
                    pt_b2 = utils.shared_crn(xmax_crn).link_loop_next[uv].uv
            else:
                pt_a1 = ymin_crn[uv].uv
                pt_a2 = ymax_crn[uv].uv

                if is_flipped_3d:
                    pt_b1 = utils.shared_crn(ymin_crn)[uv].uv
                    pt_b2 = utils.shared_crn(ymax_crn)[uv].uv
                else:
                    pt_b1 = utils.shared_crn(ymin_crn).link_loop_next[uv].uv
                    pt_b2 = utils.shared_crn(ymax_crn).link_loop_next[uv].uv

            # Rotate
            normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
            normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

            rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
            trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

            # Move
            center_ref = ref_lg.calc_bbox().center
            center_trans = trans_lg.calc_bbox().center
            trans.set_position(center_ref, center_trans)

            # Scale
            normal_a = pt_a1 - pt_a2
            normal_b = pt_b1 - pt_b2
            length_a = normal_a.length
            length_b = normal_b.length

            if length_a < 1e-06 or length_b < 1e-06:
                self.zero_area_count += 1
            else:
                scale = length_a / length_b
                bbox.scale(Vector((ref_isl.umesh.aspect, 1.0)))

                # Add padding scale
                min_length = bbox.min_length
                if min_length < 1e-06:
                    min_length = length_a
                pad_scale = bl_math.clamp((min_length-self.padding * 2) / min_length, 0.5, 1.5)

                # Check if trans_lg is the basic boundary, if so, then scaling should be negative (inner)
                trans.set_boundary_tag(match_idx=True)
                loop_groups = LoopGroups.calc_by_boundary_crn_tags(trans)
                if loop_groups:
                    trans_lg_is_basic_boundary = False
                    if len(loop_groups) != 1:
                        longest_border_lg = max(loop_groups, key=lambda lg: lg.length_uv)
                        vert = longest_border_lg[0].vert
                        if (len(longest_border_lg) == len(trans_lg) and
                                any(trans_crn.vert == vert for trans_crn in trans_lg)):
                            trans_lg_is_basic_boundary = True

                    else:
                        trans_lg_is_basic_boundary = True
                    if not trans_lg_is_basic_boundary:
                        pad_scale = 1 / pad_scale  # scale negative (inner)

                scale *= pad_scale
                trans.scale(Vector((scale, scale)), center_ref)

        else:
            pt_a1, pt_a2 = ref_lg.calc_begin_end_pt()
            pt_b1, pt_b2 = trans_lg.calc_begin_end_pt()

            # Rotate
            normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
            normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

            rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
            trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

            # Scale
            normal_a = pt_a1 - pt_a2
            normal_b = pt_b1 - pt_b2
            length_a = normal_a.length
            length_b = normal_b.length

            if length_a < 1e-06 or length_b < 1e-06:
                self.zero_area_count += 1
            else:
                scale = length_a / length_b
                trans.scale_simple(Vector((scale, scale)))

            # Move
            aspect_vec = Vector((1 / ref_isl.umesh.aspect, 1))
            orto = normal_a.orthogonal().normalized() * self.padding
            if orto == Vector((0, 0)):
                orto = (trans.bbox.center - ref_isl.bbox.center) * Vector((ref_isl.umesh.aspect, 1.0))
                orto = orto.normalized() * self.padding
            if orto == Vector((0, 0)):
                orto = Vector((self.padding, 0))
            orto *= aspect_vec

            correct_flipped_islands = (ref_is_flipped and not trans_is_flipped or
                                       ref_is_flipped and trans_is_flipped)
            if correct_flipped_islands:
                delta = (pt_a1 - pt_b1) - orto
            else:
                delta = (pt_a1 - pt_b1) + orto
            trans.move(delta)

    def balancing_filter_for_lgs(self, balance_isl, exclude_indexes):
        """Enhances multi-stitching steps for a more even distribution"""
        if not balance_isl.sequence:
            self.set_selected_boundary_tag_with_exclude_face_idx(balance_isl, exclude_indexes)
            loop_groups = LoopGroups.calc_by_boundary_crn_tags_v2(balance_isl)
            if loop_groups:
                filtered = self.split_lg_for_stitch(loop_groups)
                if len(filtered) == 1:
                    balance_isl.tag = False
                    return filtered[0]
                balance_isl.sequence = filtered
            else:
                balance_isl.tag = False
                return None

        # Selects an island for stitching based on a similar length to the one that was reoriented
        min_diff = float('inf')
        min_lg = None
        for lg in balance_isl.sequence:
            idx = lg[0].link_loop_radial_prev.face.index
            if idx not in exclude_indexes:
                diff = abs(lg.length_3d - balance_isl.area_3d)
                if diff < min_diff:
                    min_lg = lg
                    min_diff = diff
        if not min_lg:
            balance_isl.tag = False
        return min_lg

    @staticmethod
    def split_lg_for_stitch(lgs: LoopGroups) -> list[LoopGroup]:
        # TODO: Segment has a similar feature that groups data by indices. Use it, when refactor the LoopGroup to Segments.
        filtered_lg = []
        uv = lgs.umesh.uv
        def get_shared_face_idx(crn_):
            return crn_.link_loop_radial_prev.face.index

        for lg in lgs:
            if utils.all_equal(lg, key=get_shared_face_idx):
                filtered_lg.append(lg)
            else:
                split_lgs: list[list[BMLoop]] = utils.split_by_similarity(lg, get_shared_face_idx)

                # # Join same index LG, case when border loop circular but with different indexes
                start_crn = split_lgs[0][0]
                end_next_crn = split_lgs[-1][-1].link_loop_next
                # Circular case
                is_cyclic = start_crn.vert == end_next_crn.vert and start_crn[uv].uv == end_next_crn[uv].uv
                if is_cyclic:
                    # Join, if cyclic similar index.
                    crn_from_last_group = split_lgs[-1][-1]
                    if get_shared_face_idx(start_crn) == get_shared_face_idx(crn_from_last_group):
                        lg_start = split_lgs.pop()
                        lg_end = split_lgs[0]
                        del split_lgs[0]

                        lg_start.extend(lg_end)
                        lg_combined = LoopGroup(lgs.umesh)
                        lg_combined.corners = lg_start
                        filtered_lg.append(lg_combined)

                # Convert to LoopGroup.
                for sub_lg in split_lgs:
                    lg_combined = LoopGroup(lgs.umesh)
                    lg_combined.corners = sub_lg
                    filtered_lg.append(lg_combined)


        # There can be multiple shared segments between islands.
        # Here we remove redundant segments using indices taken from the shared face.
        groups = collections.defaultdict(list)
        for lg in filtered_lg:
            groups[get_shared_face_idx(lg[0])].append(lg)
        return [max(g, key=lambda lg_: lg_.length_uv) for g in groups.values()]  # Remove duplicates by length uv

    def set_selected_boundary_tag_with_exclude_face_idx(self, isl, exclude_idx: set):
        """Sets tags for the segments that will be used for stitching. Islands that have already been stitched are ignored."""
        is_bound = utils.is_boundary_func(isl.umesh, with_flipped_check=False)  # TODO: Check with flipped
        if self.between:
            for f in isl:
                for crn in f.loops:
                    pair = crn.link_loop_radial_prev
                    if pair.face.index in exclude_idx:
                        crn.tag = False
                        continue
                    crn.tag = is_bound(crn)
        else:
            get_edge_select = utils.edge_select_get_func(isl.umesh)
            for f in isl:
                for crn in f.loops:
                    if not get_edge_select(crn) or crn.link_loop_radial_prev.face.index in exclude_idx:
                        crn.tag = False
                        continue
                    crn.tag = is_bound(crn)

    def pick_reorient(self, ref_isl: AdvIsland, trans: AdvIsland, ref_lg: LoopGroup, trans_lg: LoopGroup, correct_flip=True):

        if correct_flip:
            is_flipped = trans_lg.is_flipped_3d
            if is_flipped:
                if (ref_lg.calc_signed_face_area() < 0) == (trans_lg.calc_signed_face_area() < 0):
                    trans.scale_simple(Vector((1, -1)))
            elif (ref_lg.calc_signed_face_area() < 0) != (trans_lg.calc_signed_face_area() < 0):
                trans.scale_simple(Vector((1, -1)))

        pt_a1, pt_a2 = ref_lg.calc_begin_end_pt()
        pt_b1, pt_b2 = trans_lg.calc_begin_end_pt()

        # Rotate
        normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
        normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

        rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
        trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

        # Scale
        normal_a = pt_a1 - pt_a2
        normal_b = pt_b1 - pt_b2
        length_a = normal_a.length
        length_b = normal_b.length

        if length_a < 1e-06 or length_b < 1e-06:
            self.report({'WARNING'}, 'Found zero length edge loop. Use inspect tools to find the problem')  # noqa
        else:
            scale = length_a / length_b
            trans.scale_simple(Vector((scale, scale)))

        # Move
        if self.padding:
            aspect_vec = Vector((1 / ref_isl.umesh.aspect, 1))
            orto = normal_a.orthogonal().normalized() * self.padding
            if orto == Vector((0, 0)):
                orto = (trans.bbox.center - ref_isl.bbox.center) * Vector((ref_isl.umesh.aspect, 1.0))
                orto = orto.normalized() * self.padding
            if orto == Vector((0, 0)):
                orto = Vector((self.padding, 0))
            orto *= aspect_vec
            delta = (pt_a1 - pt_b1) + orto
            trans.move(delta)
        else:
            trans.move(pt_a1 - pt_b1)
            trans_lg.copy_coords_from_ref(ref_lg, self.update_seams)


    def sort_by_dist_to_mouse_or_sel_edge_length(self, target_islands, umesh):

        if not utils.USE_GENERIC_UV_SYNC:
            if umesh.sync and self.mouse_position:
                if umesh.elem_mode in ('VERT', 'EDGE'):
                    if not umesh.has_selected_uv_faces():
                        target_islands.sort(key=lambda isl: utypes.IslandHit.closest_pt_to_selected_edge(isl, self.mouse_position))
                        return

        def calc_edge_length(isl: utypes.AdvIsland):
            uv = isl.umesh.uv
            aspect_vec = Vector((1 / isl.umesh.aspect, 1))

            total_length = 0.0
            is_boundary = utils.is_boundary_func(isl.umesh, with_seam=True, invisible_check=True)
            get_edge_select = utils.edge_select_get_func(isl.umesh)
            get_face_select = utils.face_select_get_func(isl.umesh)

            for crn in isl.corners_iter():
                if get_edge_select(crn) and is_boundary(crn):
                    diff = (crn[uv].uv - crn.link_loop_next[uv].uv) * aspect_vec
                    if get_face_select(crn.face):
                        # Prioritize edges with selected face, for correct targeting when circular stitches.
                        total_length += diff.length * 1.5
                    else:
                        total_length += diff.length

            return total_length

        target_islands.sort(key=lambda isl: calc_edge_length(isl), reverse=True)

    @staticmethod
    def filter_and_draw_lines(umeshes_a, umeshes_b):
        welded = []
        with_seam = []
        welded_append = welded.append
        with_seam_append = with_seam.append

        is_visible = utils.is_visible_func(umeshes_a.sync)

        # TODO: Optimize by update tag umesh (if umesh without tag_update -> not welded edges)
        for umesh in chain(umeshes_a, umeshes_b):
            uv = umesh.uv
            for e in umesh.sequence:
                if e.seam:
                    for crn in e.link_loops:
                        if is_visible(crn.face):
                            with_seam_append(crn[uv].uv)
                            with_seam_append(crn.link_loop_next[uv].uv)
                else:
                    toggle = False
                    for crn in e.link_loops:
                        if is_visible(crn.face):
                            if toggle:
                                welded_append(crn[uv].uv)
                                welded_append(crn.link_loop_next[uv].uv)
                            else:
                                welded_append(crn.link_loop_next[uv].uv)
                                welded_append(crn[uv].uv)
                                toggle = True

        draw.LinesDrawSimple.draw_register(with_seam, draw.DrawCallSeams2D.get_color())

        welded_color = (0.1, 0.1, 1.0, 1.0)
        draw.DotLinesDrawSimple.draw_register(welded, welded_color)

    @staticmethod
    def clear_seams_from_selected_edges(umeshes) -> int:
        counter_seam = 0
        for umesh in umeshes:
            counter_seam_local = 0
            for e in umesh.bm.edges:
                if e.select and e.seam:
                    e.seam = False
                    counter_seam_local += 1
            counter_seam += counter_seam_local
            umesh.update_tag = bool(counter_seam_local)
        return counter_seam

# TODO: Implement delayed report class
LAST_WELD_BY_DISTANCE_TIME = 0.0
LAST_WELD_BY_DISTANCE_COUNTERS = (0, 0)


# noinspection PyTypeHints
class UNIV_OT_Weld(bpy.types.Operator, Stitch):
    bl_idname = "uv.univ_weld"
    bl_label = "Weld"
    bl_description = "Weld selected UV vertices\n\n" \
                     "If there are paired and unpaired selections with no connections \nat the same time in the off sync mode, \n" \
                     "the paired connection is given priority, but when you press again, \nthe unpaired selections are also connected.\n" \
                     "This prevents unwanted connections.\n" \
                     "Works like Stitch if everything is welded in the island.\n\n" \
                     "Context keymaps on button:\n" \
                     "Default - Weld\n" \
                     "Alt - Weld by Distance\n\n" \
                     "Has [W] keymap"
    bl_options = {'REGISTER', 'UNDO'}

    use_by_distance: BoolProperty(name='By Distance', default=False)
    distance: FloatProperty(name='Distance', default=0.0005, min=0, soft_max=0.05, step=0.0001)  # noqa
    weld_by_distance_type: EnumProperty(name='Weld by', default='BY_ISLANDS', items=(
        ('ALL', 'All', ''),
        ('BY_ISLANDS', 'By Islands', '')
    ))

    flip: BoolProperty(name='Flip', default=False, options={'HIDDEN'})
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        if self.use_by_distance:
            layout.row(align=True).prop(self, 'weld_by_distance_type', expand=True)
        row = layout.row(align=True)
        row.prop(self, "use_by_distance", text="")
        row.active = self.use_by_distance
        row.prop(self, 'distance', slider=True)
        layout.prop(self, 'use_aspect')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_position = Vector(context.region.view2d.region_to_view(
                    event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.use_by_distance = event.alt

        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Stitch.__init__(self)
        self.update_seams = True

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio() if self.use_aspect else 1.0

        if self.use_by_distance:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_by_context()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

            if not self.umeshes:
                return self.umeshes.update()

            if self.weld_by_distance_type == 'BY_ISLANDS':
                counter_welded, counter_seams = self.weld_by_distance_island(extended=bool(selected_umeshes))
            else:
                counter_welded, counter_seams = self.weld_by_distance_all(selected=bool(selected_umeshes))

            global LAST_WELD_BY_DISTANCE_TIME
            global LAST_WELD_BY_DISTANCE_COUNTERS
            from time import perf_counter

            delta_time = (perf_counter() - LAST_WELD_BY_DISTANCE_TIME)
            if delta_time > 0.5:
                if LAST_WELD_BY_DISTANCE_COUNTERS != (counter_welded, counter_seams) or delta_time > 1.5:
                    info = ''
                    if counter_welded:
                        info = f'Welded {counter_welded} vertices. '
                    if counter_seams:
                        info += f'Cleaned {counter_seams} seams.'
                    if info:
                        self.report({'INFO'}, 'Weld by Distance: ' + info)

                    LAST_WELD_BY_DISTANCE_TIME = perf_counter()
                    LAST_WELD_BY_DISTANCE_COUNTERS = (counter_welded, counter_seams)
        else:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

            if not self.umeshes:
                return self.umeshes.update()

            for umesh in chain(selected_umeshes, visible_umeshes):
                umesh.sequence = draw.mesh_extract.extract_edges_with_seams(umesh)

            if not selected_umeshes and self.mouse_position:
                hit = utypes.CrnEdgeHit(self.mouse_position, self.max_distance)
                for umesh in self.umeshes:
                    hit.find_nearest_crn_by_visible_faces(umesh)
                self.pick_weld(hit)
                self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
                bpy.context.area.tag_redraw()
                return {'FINISHED'}

            self.weld()
            self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
            bpy.context.area.tag_redraw()

        self.umeshes.update(info='Not found verts for weld')
        return {'FINISHED'}

    def weld(self):
        from ..utils import weld_crn_edge_by_idx

        # Weld by index pair select edges
        islands_of_mesh = []
        for umesh in self.umeshes:
            uv = umesh.uv
            update_tag = False
            islands = AdvIslands.calc_extended_any_edge_non_manifold(umesh)
            if islands:
                umesh.set_corners_tag(False)
                islands.indexing()

                for idx, isl in enumerate(islands):
                    isl.set_selected_crn_edge_tag()

                    for crn in isl.iter_corners_by_tag():
                        shared = crn.link_loop_radial_prev
                        if shared == crn:
                            crn.tag = False
                            continue

                        # Skip non-inner boundary edges for stitch
                        if shared.face.index != idx:
                            crn.tag = False
                            shared.tag = False
                            # TODO: Count boundary edges for stitch operator, for avoid overhead
                            continue

                        # Single select preserve system.
                        if not shared.tag:
                            # Contain inner edges for avoid overhead for single select preserve system.
                            isl.sequence.append(crn)
                            continue

                        # CPU Bound
                        crn_next = crn.link_loop_next
                        shared_next = shared.link_loop_next

                        is_splitted_a = crn[uv].uv.to_tuple() != shared_next[uv].uv.to_tuple()
                        is_splitted_b = crn_next[uv].uv.to_tuple() != shared[uv].uv.to_tuple()

                        if is_splitted_a and is_splitted_b:
                            weld_crn_edge_by_idx(crn, shared_next, idx, uv)
                            weld_crn_edge_by_idx(crn_next, shared, idx, uv)
                            update_tag = True
                        elif is_splitted_a:
                            weld_crn_edge_by_idx(crn, shared_next, idx, uv)
                            update_tag = True
                        elif is_splitted_b:
                            weld_crn_edge_by_idx(crn_next, shared, idx, uv)
                            update_tag = True

                        if crn.edge.seam:
                            crn.edge.seam = False
                            update_tag = True

                        crn.tag = False
                        shared.tag = False

            umesh.update_tag = update_tag

            if islands:
                islands_of_mesh.append(islands)

        if self.umeshes.update_tag:
            return

        # Weld unpair selected edges
        if not self.umeshes.sync or any(u.sync_valid for u in self.umeshes):
            from ..utils import linked_crn_uv_by_idx_unordered_included
            from ..utils import is_pair

            for islands in islands_of_mesh:
                umesh = islands.umesh
                # There can be cases where one mesh has valid UVs and another does not, so we filter out such cases accordingly.
                if umesh.sync:
                    if not umesh.sync_valid:
                        continue

                update_tag = False
                uv = umesh.uv
                edge_linked_select_set = utils.edge_select_linked_set_func(umesh)

                for isl in islands:
                    single_selected_inner_edges = isl.sequence
                    if single_selected_inner_edges:
                        update_tag = True
                        for crn in single_selected_inner_edges:
                            crn.edge.seam = False

                            # Weld verts
                            next_crn_co = crn.link_loop_next[uv].uv
                            shared = crn.link_loop_radial_prev
                            for shared_l_crn in linked_crn_uv_by_idx_unordered_included(shared, uv):
                                shared_l_crn[uv].uv = next_crn_co

                            crn_co = crn[uv].uv
                            shared_next = shared.link_loop_next
                            for shared_l_crn in linked_crn_uv_by_idx_unordered_included(shared_next, uv):
                                shared_l_crn[uv].uv = crn_co

                            # TODO: Check selection on flipped faces
                            edge_linked_select_set(shared, True)

                umesh.update_tag = update_tag

            if self.umeshes.update_tag:
                return

        self.stitch()

    def weld_by_distance_island(self, extended):
        counter_seams = 0
        counter_welded = 0
        for umesh in self.umeshes:
            get_vert_select = utils.vert_select_get_func(umesh)
            is_boundary = utils.is_boundary_func(umesh, with_seam=False, with_flipped_check=False)

            uv = umesh.uv
            local_counter_seams = 0
            local_counter_welded = 0
            all_linked_corners = set()

            islands = Islands.calc_any_extended_or_visible_non_manifold(umesh, extended=extended)
            islands.indexing()

            for idx, isl in enumerate(islands):
                unique_verts_3d = {v for f in isl for v in f.verts}
                if extended:
                    for v in unique_verts_3d:
                        if not v.select:  # Valid for sync and non-sync
                            continue

                        linked = []
                        for crn in getattr(v, 'link_loops', ()):
                            if get_vert_select(crn) and crn.face.index == idx:
                                linked.append(crn)
                        all_linked_corners.update(linked)
                        corners_groups = self.weld_corners_in_vert(linked, uv)
                        local_counter_welded += sum(len(g) for g in corners_groups)
                else:
                    for v in unique_verts_3d:
                        linked = []
                        for crn in getattr(v, 'link_loops', ()):
                            if crn.face.index == idx:
                                linked.append(crn)
                        all_linked_corners.update(linked)
                        corners_groups = self.weld_corners_in_vert(linked, uv)
                        local_counter_welded += sum(len(g) for g in corners_groups)

                for crn in all_linked_corners:
                    if crn.edge.seam:
                        if not is_boundary(crn) and crn.link_loop_radial_prev.face.index == idx:
                            crn.edge.seam = False
                            local_counter_seams += 1

                    if crn.link_loop_prev.edge.seam:
                        prev = crn.link_loop_prev
                        if not is_boundary(prev) and prev.link_loop_radial_prev.face.index == idx:
                            prev.edge.seam = False
                            local_counter_seams += 1

            counter_seams += local_counter_seams
            counter_welded += local_counter_welded
            umesh.update_tag = bool(local_counter_seams + local_counter_welded)

        return counter_welded, counter_seams

    def weld_by_distance_all(self, selected):
        counter_seams = 0
        counter_welded = 0
        for umesh in self.umeshes:
            uv = umesh.uv
            local_counter_seams = 0
            local_counter_welded = 0

            all_linked_corners = set()
            is_invisible = utils.is_invisible_func(umesh.sync)
            if selected:
                get_vert_select = utils.vert_select_get_func(umesh)
                for v in umesh.bm.verts:
                    if not v.select:
                        continue

                    linked = []
                    for crn in getattr(v, 'link_loops', ()):
                        if get_vert_select(crn) and not is_invisible(crn.face):
                            linked.append(crn)
                    all_linked_corners.update(linked)
                    corners_groups = self.weld_corners_in_vert(linked, uv)
                    local_counter_welded += sum(len(g) for g in corners_groups)
            else:
                for v in umesh.bm.verts:
                    linked = []
                    for crn in getattr(v, 'link_loops', ()):
                        if not is_invisible(crn.face):
                            linked.append(crn)
                    all_linked_corners.update(linked)
                    corners_groups = self.weld_corners_in_vert(linked, uv)
                    local_counter_welded += sum(len(g) for g in corners_groups)

            is_boundary = utils.is_boundary_func(umesh, with_seam=False, with_flipped_check=False)
            for crn in all_linked_corners:
                if crn.edge.seam:
                    if not is_boundary(crn):
                        crn.edge.seam = False
                        local_counter_seams += 1

                if crn.link_loop_prev.edge.seam:
                    if not is_boundary(crn.link_loop_prev):
                        crn.link_loop_prev.edge.seam = False
                        local_counter_seams += 1

            counter_seams += local_counter_seams
            counter_welded += local_counter_welded
            umesh.update_tag = bool(local_counter_seams + local_counter_welded)

        return counter_welded, counter_seams

    def weld_corners_in_vert(self, crn_in_vert, uv):
        if utils.all_equal(_crn[uv].uv.to_tuple() for _crn in crn_in_vert):
            return []

        corners_groups = self.calc_distance_corners_groups(crn_in_vert, uv)
        for group in corners_groups:
            # TODO: Add weighted center by face 'relax'
            value = Vector((0, 0))
            for c in group:
                value += c[uv].uv
            avg = value / len(group)
            for c in group:
                c[uv].uv = avg
        return corners_groups

    def calc_distance_corners_groups(self, crn_in_vert: list[BMLoop], uv) -> list[list[BMLoop]]:
        corners_groups = []
        visited = set()

        for corner_first in crn_in_vert:
            if corner_first in visited:
                continue

            group = [corner_first]
            visited.add(corner_first)

            i = 0
            while i < len(group):
                a = group[i]
                for b in crn_in_vert:
                    if b in visited:
                        continue
                    if (a[uv].uv - b[uv].uv).length <= self.distance:
                        group.append(b)
                        visited.add(b)
                i += 1

            if not utils.all_equal(c[uv].uv.to_tuple() for c in group):
                corners_groups.append(group)

        return corners_groups

    def pick_weld(self, hit: utypes.CrnEdgeHit):
        if not hit:
            self.report({'WARNING'}, 'Edge not found within a given radius')
            return

        sync = hit.umesh.sync
        ref_crn = hit.crn
        shared = ref_crn.link_loop_radial_prev
        is_visible = utils.is_visible_func(sync)
        if shared == ref_crn or not is_visible(shared.face):
            self.report({'INFO'}, 'Edge is boundary')
            return

        uv = hit.umesh.uv
        e = ref_crn.edge
        if utils.is_pair_with_flip(ref_crn, shared, uv):
            if e.seam:
                e.seam = False
                hit.umesh.update()
                return
            return

        # Fast calculate, if edge has non-manifold links
        if utils.is_flipped_3d(ref_crn):
            if ref_crn[uv].uv.to_tuple() == shared[uv].uv.to_tuple():  # check a
                shared_next = shared.link_loop_next
                for crn in [shared_next] + utils.linked_crn_to_vert_pair_with_seam(shared_next, uv, sync):
                    crn[uv].uv = ref_crn.link_loop_next[uv].uv  # join b
                e.seam = False
                hit.umesh.update()
                return
            elif ref_crn.link_loop_next[uv].uv.to_tuple() == shared.link_loop_next[uv].uv.to_tuple():  # check b
                for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                    crn[uv].uv = ref_crn[uv].uv  # join a
                e.seam = False
                hit.umesh.update()
                return
        else:
            if ref_crn[uv].uv.to_tuple() == shared.link_loop_next[uv].uv.to_tuple():  # check a
                for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                    crn[uv].uv = ref_crn.link_loop_next[uv].uv  # join b
                e.seam = False
                hit.umesh.update()
                return
            elif ref_crn.link_loop_next[uv].uv.to_tuple() == shared[uv].uv.to_tuple():  # check b
                shared_next = shared.link_loop_next
                for crn in [shared_next] + utils.linked_crn_to_vert_pair_with_seam(shared_next, uv, sync):
                    crn[uv].uv = ref_crn[uv].uv  # join a
                e.seam = False
                hit.umesh.update()
                return

        if utils.is_flipped_3d(ref_crn):
            self.report({'WARNING'}, 'Edge has 3D flipped face, need recalculate normals')

        if utils.is_flipped_3d(ref_crn):
            ref_isl, isl_set = hit.calc_island_non_manifold_with_flip()
        else:
            ref_isl, isl_set = hit.calc_island_non_manifold()

        if shared.face in isl_set:
            if utils.is_flipped_3d(ref_crn):
                for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                    crn[uv].uv = ref_crn[uv].uv

                shared_next = shared.link_loop_next
                for crn in [shared_next] + utils.linked_crn_to_vert_pair_with_seam(shared_next, uv, sync):
                    crn[uv].uv = ref_crn.link_loop_next[uv].uv
                e.seam = False
                hit.umesh.update()
            else:
                shared_next_crn = shared.link_loop_next
                for crn in [shared_next_crn] + utils.linked_crn_to_vert_pair_with_seam(shared_next_crn, uv, sync):
                    crn[uv].uv = ref_crn[uv].uv

                for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                    crn[uv].uv = ref_crn.link_loop_next[uv].uv
                e.seam = False
                hit.umesh.update()
        else:
            ref_lg = LoopGroup(hit.umesh)
            ref_lg.corners = [hit.crn]
            trans_lg = ref_lg.calc_shared_group_for_stitch()

            hit.crn = shared
            trans_isl, _ = hit.calc_island_with_seam()
            self.pick_reorient(ref_isl, trans_isl, ref_lg, trans_lg)
            hit.umesh.update()
        return


class UNIV_OT_Weld_VIEW3D(UNIV_OT_Weld, utypes.RayCast):
    bl_idname = "mesh.univ_weld"

    def invoke(self, context, event):
        if event.value == 'PRESS':
            self.init_data_for_ray_cast(event)
            return self.execute(context)
        self.use_by_distance = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Stitch.__init__(self)
        utypes.RayCast.__init__(self)
        self.update_seams = True

    def execute(self, context):
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0

        if self.use_by_distance:
            self.weld_by_distance_from_3d()
        else:
            res = self.weld_by_edge_from_3d()
            if res:
                return res

        self.umeshes.update(info='Not found elements for weld')
        return {'FINISHED'}

    def weld_by_distance_from_3d(self):
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_by_context()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        umeshes_without_uv = self.umeshes.filtered_by_uv_exist()
        self.umeshes.verify_uv()

        if not self.umeshes and not umeshes_without_uv:
            self.report({'WARNING'}, 'Not found edges for manipulate')
            return

        if self.umeshes:
            if self.weld_by_distance_type == 'BY_ISLANDS':
                counter_welded, counter_seams = self.weld_by_distance_island(extended=bool(selected_umeshes))
            else:
                counter_welded, counter_seams = self.weld_by_distance_all(selected=bool(selected_umeshes))
            counter_seams += self.clear_seams_from_selected_edges(umeshes_without_uv)

            global LAST_WELD_BY_DISTANCE_TIME
            global LAST_WELD_BY_DISTANCE_COUNTERS
            from time import perf_counter

            delta_time = (perf_counter() - LAST_WELD_BY_DISTANCE_TIME)
            if delta_time > 0.5:
                if LAST_WELD_BY_DISTANCE_COUNTERS != (counter_welded, counter_seams) or delta_time > 1.5:
                    info = ''
                    if counter_welded:
                        info = f'Welded {counter_welded} vertices. '
                    if counter_seams:
                        info += f'Cleaned {counter_seams} seams.'
                    if info:
                        self.report({'INFO'}, 'Weld by Distance: ' + info)

                    LAST_WELD_BY_DISTANCE_TIME = perf_counter()
                    LAST_WELD_BY_DISTANCE_COUNTERS = (counter_welded, counter_seams)

        self.umeshes.umeshes.extend(umeshes_without_uv.umeshes.copy())

    def weld_by_edge_from_3d(self):
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update(info='Not found edges for manipulate')
        if not selected_umeshes and self.mouse_pos_from_3d:
            hit = self.ray_cast(prefs().max_pick_distance)
            if hit:
                if len(hit.umesh.bm.loops.layers.uv):
                    hit.umesh.verify_uv()
                    self.pick_weld(hit)
                else:
                    if not hit.crn.edge.seam:
                        return {'CANCELLED'}
                    hit.crn.edge.seam = False
                    hit.umesh.update()
            return {'FINISHED'}

        umeshes_without_uv = self.umeshes.filtered_by_uv_exist()
        self.umeshes.verify_uv()
        self.weld()
        self.clear_seams_from_selected_edges(umeshes_without_uv)
        self.umeshes.umeshes.extend(umeshes_without_uv.umeshes.copy())
        return None


# noinspection PyTypeHints
class UNIV_OT_Stitch(bpy.types.Operator, Stitch, utils.PaddingHelper):
    bl_idname = "uv.univ_stitch"
    bl_label = 'Stitch'
    bl_description = "Stitch selected UV vertices by proximity\n\n" \
                     "Default - Stitch\n" \
                     "Alt - Stitch Between\n\n" \
                     "Has [Shift + W] keymap. \n" \
                     "In sync mode when calling stitch via keymap, the stitch priority is done by mouse cursor.\n" \
                     "In other cases of pairwise selection, prioritization occurs by island size"
    bl_options = {'REGISTER', 'UNDO'}

    between: BoolProperty(name='Between', default=False, description='Attention, it is unstable')
    update_seams: BoolProperty(name='Update Seams', default=True)
    use_aspect: BoolProperty(name='Correct Aspect', default=True)
    correct_flip: BoolProperty(name='Correct Flip', default=True)
    padding_multiplayer: bpy.props.FloatProperty(name='Padding Multiplayer', default=0, min=-32, soft_min=0,
                                                 soft_max=4, max=32)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "between")
        layout.prop(self, "update_seams")
        layout.prop(self, "use_aspect")
        layout.prop(self, "correct_flip")
        if not self.between:
            self.draw_padding()

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_position = Vector(context.region.view2d.region_to_view(
                    event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.between = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Stitch.__init__(self)

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.update_tag = False
        if self.between:
            visible_umeshes = self.umeshes.filtered_by_selected_uv_faces()
            selected_umeshes = self.umeshes
        else:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        self.calc_padding()

        for umesh in chain(selected_umeshes, visible_umeshes):
            umesh.aspect = utils.get_aspect_ratio() if self.use_aspect else 1.0
            umesh.sequence = draw.mesh_extract.extract_edges_with_seams(umesh)

        if self.between:
            self.stitch()
        else:
            if not self.umeshes:
                return self.umeshes.update()
            if not selected_umeshes and self.mouse_position:
                self.report_padding()

                hit = utypes.CrnEdgeHit(self.mouse_position, self.max_distance)
                for umesh in self.umeshes:
                    hit.find_nearest_crn_by_visible_faces(umesh)

                if hit:
                    self.pick_stitch(hit)
                else:
                    self.report({'WARNING'}, 'Edge not found within a given radius')

                self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
                bpy.context.area.tag_redraw()
                return {'FINISHED'}

            self.stitch()

        self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
        bpy.context.area.tag_redraw()

        self.umeshes.update(info='Not found islands for stitch')
        return {'FINISHED'}

    def pick_stitch(self, hit: utypes.CrnEdgeHit):
        sync = hit.umesh.sync
        hit.umesh.update_tag = True
        ref_crn = hit.crn
        shared = ref_crn.link_loop_radial_prev
        is_visible = utils.is_visible_func(sync)
        if shared == ref_crn or not is_visible(shared.face):
            self.report({'WARNING'}, 'Edge is boundary')
            return

        if ref_crn.link_loop_next.vert != shared.vert:
            self.report({'WARNING'}, 'Edge has 3D flipped face, for correct result need recalculate normals')

        uv = hit.umesh.uv
        e = ref_crn.edge
        if not self.padding and utils.is_pair_with_flip(ref_crn, shared, uv):
            if e.seam:
                e.seam = False
                hit.umesh.update()
            else:
                self.report({'INFO'}, 'The edge was already stitched, no action was taken')
            return

        ref_isl, isl_set = hit.calc_island_with_seam()

        if shared.face in isl_set:
            self.report({'WARNING'}, 'It is not possible to use Stitch on itself, use Weld operator')
        else:
            ref_lg = LoopGroup(hit.umesh)
            ref_lg.corners = [hit.crn]
            trans_lg = ref_lg.calc_shared_group_for_stitch()

            hit.crn = shared
            trans_isl, _ = hit.calc_island_with_seam()
            self.pick_reorient(ref_isl, trans_isl, ref_lg, trans_lg, correct_flip=self.correct_flip)
            hit.umesh.update()


class UNIV_OT_Stitch_VIEW3D(UNIV_OT_Stitch, utypes.RayCast):
    bl_idname = "mesh.univ_stitch"

    def invoke(self, context, event):
        if event.value == 'PRESS':
            self.init_data_for_ray_cast(event)
            return self.execute(context)
        self.between = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Stitch.__init__(self)

    def execute(self, context):
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()
        self.umeshes.update_tag = False

        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0

        settings = univ_settings()
        self.padding = int(settings.padding * self.padding_multiplayer) / \
            min(int(settings.size_x), int(settings.size_y))

        if self.between:
            self.stitch_between()
        else:
            res = self.stitch_by_edge()
            if res:
                return res
        self.umeshes.update(info='Not found islands for stitch')
        return {'FINISHED'}

    def stitch_between(self):
        self.umeshes.filtered_by_selected_uv_faces()
        without_uv = self.umeshes.filtered_by_uv_exist()
        self.umeshes.verify_uv()
        self.stitch()

        self.clear_seams_from_selected_edges(without_uv)
        self.umeshes.umeshes.extend(without_uv.umeshes.copy())

    def stitch_by_edge(self):
        settings = univ_settings()
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return None
        if not selected_umeshes and self.mouse_pos_from_3d:
            if self.padding:
                img_size = utils.get_active_image_size()
                if img_size:  # TODO: Get active image size from material id
                    if min(int(settings.size_x), int(settings.size_y)) != min(img_size):
                        self.report({'WARNING'}, 'Global and Active texture sizes have different values, '
                                                 'which will result in incorrect padding.')

            hit = self.ray_cast(prefs().max_pick_distance)
            if hit:
                if len(hit.umesh.bm.loops.layers.uv):
                    hit.umesh.verify_uv()
                    self.pick_stitch(hit)
                else:
                    if not hit.crn.edge.seam:
                        return {'CANCELLED'}
                    hit.crn.edge.seam = False
                    hit.umesh.update()
                    return {'FINISHED'}
            return {'FINISHED'}

        without_uv = self.umeshes.filtered_by_uv_exist()
        self.umeshes.verify_uv()
        self.stitch()
        self.clear_seams_from_selected_edges(without_uv)
        self.umeshes.umeshes.extend(without_uv.umeshes.copy())
        return None

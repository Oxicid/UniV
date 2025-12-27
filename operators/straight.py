# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import numpy as np

from .. import utils
from .. import utypes
from mathutils import Vector, Matrix

class StraightIsland:
    def __init__(self, isl: utypes.AdvIsland, segment: utypes.Segment):
        self.isl = isl
        self.segment = segment
        self.pivot: Vector | None = None
        self.is_flipped_to_avoid_unflip_after_unwrap = False


class UNIV_OT_Straight(bpy.types.Operator):
    bl_idname = "uv.univ_straight"
    bl_label = "Straight"
    bl_description = "Straighten selected edge-chain and relax the rest of the UV Island"
    bl_options = {'REGISTER', 'UNDO'}

    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        from ..preferences import univ_settings
        self.layout.prop(univ_settings(), 'use_texel')
        self.layout.prop(self, 'use_correct_aspect')

    def execute(self, context):
        assert (context.area.ui_type == 'UV')

        umeshes = utypes.UMeshes(report=self.report)
        if umeshes.elem_mode not in ('VERT', 'EDGE'):
            self.report({'WARNING'}, f'Use Vertex or Edge mode')
            return {'CANCELLED'}

        if self.use_correct_aspect:
            umeshes.calc_aspect_ratio(from_mesh=False)

        umeshes.fix_context()
        umeshes.filter_by_selected_uv_edges()

        temporary_hidden_islands = []
        deselect_islands = []

        straight_islands = []

        for umesh in umeshes:
            need_hide = self.need_hide(umesh)

            islands = utypes.AdvIslands.calc_visible_with_mark_seam(umesh)
            islands.indexing()

            is_boundary = utils.is_boundary_func(umesh, with_seam=False)
            get_edge_select = utils.edge_select_get_func(umesh)
            get_face_select = utils.face_select_get_func(umesh)

            for idx, isl in enumerate(islands):
                to_segmenting_corners = []
                for crn in isl.corners_iter():
                    pair_face = crn.link_loop_radial_prev.face

                    if not get_edge_select(crn):
                        crn.tag = False
                    elif get_face_select(crn.face):
                        crn.tag = False
                    elif not is_boundary(crn) and get_face_select(pair_face) and pair_face.index == idx:
                        crn.tag = False
                    else:
                        crn.tag = True
                        to_segmenting_corners.append(crn)

                if to_segmenting_corners:
                    isl.apply_aspect_ratio()

                    segments = utypes.Segments.from_tagged_corners(to_segmenting_corners, umesh)
                    segment = max(segments.segments, key=lambda seg__: seg__.length_uv)

                    straight_isl = StraightIsland(isl, segment)
                    if isl.should_flip_after_unwrap():
                        isl.scale_simple(Vector((-1.0, 1.0)))
                        straight_isl.is_flipped_to_avoid_unflip_after_unwrap = True

                    straight_islands.append(straight_isl)

                elif need_hide:
                    temporary_hidden_islands.append(isl)

        # Used for valid sync and non-sync for invalid corners
        for isl in deselect_islands:
            isl.select = False

        # Used for non-valid sync
        for isl in temporary_hidden_islands:
            for f in isl:
                f.hide_set(True)

        if not straight_islands:
            self.report({'WARNING'}, f"Loops not found")
            return {'CANCELLED'}

        for straight_isl in straight_islands:
            isl = straight_isl.isl
            uv = isl.umesh.uv
            segment: utypes.Segment = straight_isl.segment

            if segment.is_circular:
                pivot = self.distribute_by_circle_and_get_pivot(segment)
                straight_isl.pivot = pivot
            else:
                segment.calc_chain_linked_corners()

                # Distribute
                card_vec = utils.vec_to_cardinal(segment.end_co - segment.start_co)
                start = segment.start_co
                end = start + (card_vec * segment.length)
                segment.distribute(start, end, True)

                straight_isl.pivot = start.copy()

            # Set pins
            for linked in segment.chain_linked_corners:
                for crn in linked:
                    crn[uv].pin_uv = True

            # Mark Seam
            is_boundary = utils.is_boundary_func(isl.umesh)
            for crn in isl.corners_iter():
                if is_boundary(crn):
                    crn.edge.seam = True

            isl.select = True

        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=False, use_subsurf_data=False, margin=0)

        # Deselect islands and restore edge selection and clear pins.
        for straight_isl in straight_islands:
            # NOTE: The aspect ratio must be restored before applying the texel in order to use the correct UV area.
            straight_isl.isl.reset_aspect_ratio()
            straight_isl.isl.select = False

        for isl in temporary_hidden_islands:
            for f in isl:
                f.hide_set(False)

        for straight_isl in straight_islands:
            isl = straight_isl.isl
            umesh = isl.umesh
            set_edge_select = utils.edge_select_linked_set_func(umesh)

            segment: utypes.Segment = straight_isl.segment
            for adv_crn in segment:
                if adv_crn.invert:
                    set_edge_select(adv_crn.crn.link_loop_prev, True)
                else:
                    set_edge_select(adv_crn.crn, True)

            # Here we compensate the aspect for the pivot, since it was calculated with a normalized aspect ratio.
            pivot_after_aspect = straight_isl.pivot * Vector((1 / umesh.aspect, 1))
            isl._bbox = utypes.BBox.calc_bbox([pivot_after_aspect])

            utils.set_global_texel(isl, calc_bbox=False)

            # Restore flips
            if straight_isl.is_flipped_to_avoid_unflip_after_unwrap:
                isl.scale_simple(Vector((-1.0, 1.0)))

            uv = umesh.uv
            for linked in segment.chain_linked_corners:
                for crn in linked:
                    crn[uv].pin_uv = False

        umeshes.update()
        return {'FINISHED'}

    @staticmethod
    def need_hide(umesh):
        if umesh.sync:
            if utils.USE_GENERIC_UV_SYNC:
                if not umesh.sync_valid:
                    return True
            else:
                return True
        return False

    @staticmethod
    def distribute_by_circle_and_get_pivot(segment: utypes.Segment):
        segment.calc_chain_linked_corners()

        # Get pivot by weighted centroids, for minimize shifts
        edge_centroids = np.array([adv_crn.center_uv for adv_crn in segment])
        edge_weights = np.array(segment.calc_lengths_uv())

        pivot = np.average(edge_centroids, weights=edge_weights, axis=0)
        pivot = Vector(pivot)

        # Get weighted avg radius
        radius_seq = np.linalg.norm(edge_centroids - pivot, axis=1)
        avg_radius = np.average(radius_seq, weights=edge_weights)
        avg_radius = max(avg_radius, 0.0001)

        tar_adv_crn = segment[0]
        tar_pt = tar_adv_crn.curr_pt.copy()

        vec = (pivot - tar_pt).normalized()
        if vec == Vector((0, 0)):
            vec = Vector((-1, 0))

        start_pt = vec * avg_radius

        start_angle = math.pi
        last_angle = -math.pi
        if tar_adv_crn.invert or tar_adv_crn.is_pair:
            start_angle = -start_angle
            last_angle = -last_angle

        angles = utils.weighted_linear_space(start_angle, last_angle, edge_weights)

        uv = segment.umesh.uv
        for corners, angle in zip(segment.chain_linked_corners, angles):
            rot_matrix = Matrix.Rotation(angle, 2)
            co = (start_pt @ rot_matrix) + pivot

            for l_crn in corners:
                l_crn[uv].uv = co
        return pivot
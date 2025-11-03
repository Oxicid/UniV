# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from .. import utils
from .. import utypes


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

        zero_length_counter = 0
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
                    segments = utypes.Segments.from_tagged_corners(to_segmenting_corners, umesh)
                    segments.segments.sort(key=lambda seg__: seg__.length, reverse=True)
                    segment = segments.segments[0]
                    if segment.is_circular:  # TODO: Implement circular straight
                        segments = segment.break_by_cardinal_dir()
                        segments.segments.sort(key=lambda seg__: seg__.length, reverse=True)
                        segment = segments.segments[0]
                        if segment.is_circular:
                            if need_hide:
                                temporary_hidden_islands.append(isl)
                            else:
                                deselect_islands.append(isl)
                            zero_length_counter += 1
                            continue
                    isl.sequence = segment
                    straight_islands.append(isl)
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
            if zero_length_counter:
                self.report({'WARNING'}, f"Found {zero_length_counter} loops")
            else:
                self.report({'WARNING'}, f"Loops not found")
            return {'CANCELLED'}

        for isl in straight_islands:
            segment: utypes.Segment = isl.sequence
            segment.calc_chain_linked_corners()

            # Distribute
            card_vec = utils.vec_to_cardinal(segment.end_co - segment.start_co)
            start = segment.start_co
            end = start + (card_vec * segment.length)
            segment.distribute(start, end, True)

            # Set pins
            uv = isl.umesh.uv
            for linked in segment.chain_linked_corners:
                for crn in linked:
                    crn[uv].pin_uv = True

            # Mark Seam
            is_boundary = utils.is_boundary_func(isl.umesh)
            for crn in isl.corners_iter():
                if is_boundary(crn):
                    crn.edge.seam = True

            isl.select = True
            isl.apply_aspect_ratio()

        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=False, use_subsurf_data=False, margin=0)

        # Deselect islands and restore edge selection and clear pins
        for isl in straight_islands:
            isl.reset_aspect_ratio()
            isl.select = False

        for isl in temporary_hidden_islands:
            for f in isl:
                f.hide_set(False)

        for isl in straight_islands:
            set_edge_select = utils.edge_select_linked_set_func(isl.umesh)

            segment: utypes.Segment = isl.sequence
            for adv_crn in segment:
                if adv_crn.invert:
                    set_edge_select(adv_crn.crn.link_loop_prev, True)
                else:
                    set_edge_select(adv_crn.crn, True)

            isl._bbox = utypes.BBox.calc_bbox([segment.start_co])
            utils.set_global_texel(isl, calc_bbox=False)

            uv = isl.umesh.uv
            for linked in segment.chain_linked_corners:
                for crn in linked:
                    crn[uv].pin_uv = False

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

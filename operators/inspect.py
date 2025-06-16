# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import enum
import bl_math

from mathutils.geometry import area_tri
from bpy.props import *
from bpy.types import Operator

from .. import utils
from .. import types

class Inspect(enum.IntFlag):
    Overlap = enum.auto()
    OverlapWithModifier = enum.auto()
    InexactOverlap = enum.auto()
    SelfOverlap = enum.auto()
    TroubleOverlapFace = enum.auto()
    OverlapByPadding = enum.auto()
    DoubleVertices3D = enum.auto()
    __pass2 = enum.auto()

    Zero = enum.auto()
    __pass3 = enum.auto()

    Flipped = enum.auto()
    Flipped3D = enum.auto()
    InsideNormalsOrient = enum.auto()
    Rotated = enum.auto()
    __pass4 = enum.auto()

    NonManifold = enum.auto()
    NonSplitted = enum.auto()
    TileIsect = enum.auto()
    __pass5 = enum.auto()
    __pass6 = enum.auto()

    OverScaled = enum.auto()
    OverStretched = enum.auto()
    AngleStretch = enum.auto()
    __pass7 = enum.auto()

    Concave = enum.auto()
    DeduplicateUVLayers = enum.auto()
    RepairAfterJoin = enum.auto()
    __pass8 = enum.auto()
    IncorrectBMeshTags = enum.auto()
    Other = enum.auto()


class UNIV_OT_Check_Zero(Operator):
    bl_idname = "uv.univ_check_zero"
    bl_label = "Zero"
    bl_description = "Select degenerate UVs (zero area UV triangles)"
    bl_options = {'REGISTER', 'UNDO'}

    precision: FloatProperty(name='Precision', default=1e-6, min=0, soft_max=0.001, step=0.0001, precision=7)  # noqa

    def draw(self, context):
        self.layout.prop(self, 'precision', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = self.zero(self.precision, umeshes)

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    @staticmethod
    def zero(precision, umeshes, batch_inspect=False):
        sync = umeshes.sync
        if sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        precision *= 0.001
        total_counter = 0
        select_set = utils.face_select_linked_func(sync)
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            local_counter = 0
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if face.hide if sync else not face.select:
                    continue

                area = area_tri(tris_a[uv].uv, tris_b[uv].uv, tris_c[uv].uv)
                if area <= precision:
                    select_set(face, uv)
                    local_counter += 1
            umesh.update_tag |= bool(local_counter)
            total_counter += local_counter

        if not batch_inspect:
            umeshes.update()
        return total_counter


class UNIV_OT_Check_Flipped(Operator):
    bl_idname = "uv.univ_check_flipped"
    bl_label = "Flipped"
    bl_description = "Select flipped UV faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = self.flipped(umeshes)

        if not total_counter:
            self.report({'INFO'}, 'Flipped faces not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} flipped faces')
        return {'FINISHED'}

    @staticmethod
    def flipped(umeshes, batch_inspect=False):
        sync = umeshes.sync
        if sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        total_counter = 0
        select_set = utils.face_select_linked_func(sync)
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            local_counter = 0
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if face.hide if sync else not face.select:
                    continue

                a = tris_a[uv].uv
                b = tris_b[uv].uv
                c = tris_c[uv].uv

                signed_area = a.cross(b) + b.cross(c) + c.cross(a)
                if signed_area < 0:
                    select_set(face, uv)
                    local_counter += 1
            umesh.update_tag |= bool(local_counter)
            total_counter += local_counter
        if batch_inspect is False:
            umeshes.update()
        return total_counter


class UNIV_OT_Check_Non_Splitted(Operator):
    bl_idname = 'uv.univ_check_non_splitted'
    bl_label = 'Non-Splitted'
    bl_description = "Selects the edges where seams should be marked and unwrapped without connection"
    bl_options = {'REGISTER', 'UNDO'}

    use_auto_smooth: bpy.props.BoolProperty(name='Use Auto Smooth', default=True)
    user_angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'use_auto_smooth')
        layout.prop(self, 'user_angle', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        bpy.ops.uv.select_all(action='DESELECT')

        # clamp angle
        if self.use_auto_smooth:
            max_angle_from_obj_smooth = max(umesh.smooth_angle for umesh in umeshes)
            self.user_angle = bl_math.clamp(self.user_angle, 0.0, max_angle_from_obj_smooth)

        result = self.select_inner(umeshes, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def select_inner(umeshes, use_auto_smooth, user_angle, batch_inspect=False):
        non_seam_counter = 0
        angle_counter = 0
        sharps_counter = 0
        seam_counter = 0
        mtl_counter = 0

        sync = umeshes.sync
        is_boundary = utils.is_boundary_sync if sync else utils.is_boundary_non_sync

        for umesh in umeshes:
            to_select = set()
            if use_auto_smooth:
                angle = min(umesh.smooth_angle, user_angle)
            else:
                angle = user_angle

            uv = umesh.uv
            for crn in utils.calc_visible_uv_corners(umesh):
                if (pair_crn := crn.link_loop_radial_prev) in to_select:
                    continue

                edge = crn.edge
                if is_boundary(crn, uv):
                    if crn.edge.seam:
                        continue
                    if edge.is_boundary:
                        continue
                elif not edge.smooth:
                    sharps_counter += 1
                elif edge.calc_face_angle() >= angle:
                    angle_counter += 1
                elif edge.seam:
                    seam_counter += 1
                elif pair_crn.face.material_index != crn.face.material_index:
                    mtl_counter += 1
                else:
                    continue
                to_select.add(crn)

            umesh.sequence = to_select
            umesh.update_tag |= bool(to_select)

        if any(umesh.sequence for umesh in umeshes):
            if batch_inspect:
                # Avoid switch elem mode if all edges selected for batch inspect
                select_get = utils.edge_select_linked_get_func(sync)
                all_selected = True
                for umesh in umeshes:
                    uv = umesh.uv
                    if not all(select_get(crn_edge, uv) for crn_edge in umesh.sequence):
                        all_selected = False
                        break
                if all_selected:
                    return non_seam_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

            select_set = utils.edge_select_linked_set_func(sync)
            if umeshes.elem_mode not in ('EDGE', 'VERTEX'):
                umeshes.elem_mode = 'EDGE'

            for umesh in umeshes:
                uv = umesh.uv
                for edge in umesh.sequence:
                    select_set(edge, True, uv)

        return non_seam_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

    @staticmethod
    def data_formatting(counters):
        non_seam_counter, angle_counter, sharps_counter, seam_counter, mtl_counter = counters
        r_text = ''
        if non_seam_counter:
            r_text += f'Non-Seam - {non_seam_counter}. '
        if angle_counter:
            r_text += f'Sharp Angles - {angle_counter}. '
        if sharps_counter:
            r_text += f'Mark Sharps - {sharps_counter}. '
        if seam_counter:
            r_text += f'Mark Seams - {seam_counter}. '
        if mtl_counter:
            r_text += f'Materials - {mtl_counter}. '

        if r_text:
            r_text = f'Found: {sum(counters)} non splitted edges. {r_text}'
        return r_text


class UNIV_OT_Check_Overlap(Operator):
    bl_idname = 'uv.univ_check_overlap'
    bl_label = 'Overlap'
    bl_description = "Select all UV faces which overlap each other.\n" \
                     "Unlike the default operator, this one informs about the number of faces with conflicts"
    bl_options = {'REGISTER', 'UNDO'}

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ANY', items=(('ANY', 'Any', ''), ('INEXACT', 'Inexact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.0008, min=0.0, soft_min=0.00005, soft_max=0.00999)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        if self.check_mode == 'INEXACT':
            layout.prop(self, 'threshold')
        layout.row(align=True).prop(self, 'check_mode', expand=True)

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False

        count = self.overlap_check(umeshes, self.check_mode, self.threshold)
        self.report(*self.get_info_from_count(count, self.check_mode))
        umeshes.silent_update()
        return {'FINISHED'}

    @staticmethod
    def get_info_from_count(count, check_mode):
        if count:
            if check_mode == 'INEXACT':
                return {'WARNING'}, f"Found {count} islands with inexact overlap"
            else:
                return {'WARNING'}, f"Found about {count} edges with overlap"
        else:
            if check_mode == 'INEXACT':
                return {'INFO'}, f"Inexact islands with overlap not found"
            else:
                return {'INFO'}, f"Edges with overlap not found"

    @staticmethod
    def overlap_check(umeshes, check_mode, threshold=0.001) -> int:
        if umeshes.sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        bpy.ops.uv.select_overlap()

        count = 0
        if check_mode == 'INEXACT':
            all_islands = []
            for umesh in umeshes:
                adv_islands = types.AdvIslands.calc_extended_with_mark_seam(umesh)
                # The following subdivision is needed to ignore the exact self overlaps that are created from the flipped face
                for isl in reversed(adv_islands):
                    if isl.has_flip_with_noflip():
                        adv_islands.islands.remove(isl)
                        noflip, flipped = isl.calc_islands_by_flip_with_mark_seam()
                        adv_islands.islands.extend(noflip)
                        adv_islands.islands.extend(flipped)
                all_islands.extend(adv_islands)

            overlapped = types.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
            for isl in overlapped:
                if isinstance(isl, types.AdvIsland):
                    count += 1
                else:
                    isl.select = False
                    isl.umesh.update_tag = True
        else:
            for umesh in umeshes:
                if umesh.sync:
                    count += umesh.total_edge_sel
                else:
                    count += len(utils.calc_selected_uv_edge_corners(umesh))

        return count

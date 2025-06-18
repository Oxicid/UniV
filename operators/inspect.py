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
    OverlapInexact = enum.auto()
    OverlapSelf = enum.auto()
    OverlapTroubleFace = enum.auto()
    OverlapByMaterial = enum.auto()
    OverlapWithModifier = enum.auto()
    __pass1 = enum.auto()
    __pass2 = enum.auto()

    AllOverlapFlags = (Overlap | OverlapWithModifier | OverlapInexact |
                       OverlapSelf | OverlapTroubleFace | OverlapByMaterial)
    Zero = enum.auto()
    __pass3 = enum.auto()

    OverlapByPadding = enum.auto()
    DoubleVertices3D = enum.auto()
    TileIsect = enum.auto()


    Flipped = enum.auto()
    Flipped3D = enum.auto()
    InsideNormalsOrient = enum.auto()
    Rotated = enum.auto()
    __pass4 = enum.auto()

    NonManifold = enum.auto()
    NonSplitted = enum.auto()

    OverScaled = enum.auto()
    OverStretched = enum.auto()
    AngleStretch = enum.auto()
    __pass5 = enum.auto()

    Concave = enum.auto()
    DeduplicateUVLayers = enum.auto()
    RepairAfterJoin = enum.auto()
    __pass6 = enum.auto()
    IncorrectBMeshTags = enum.auto()
    Other = enum.auto()

    @classmethod
    def default_value_for_settings(cls):
        return cls.Overlap | cls.Zero | cls.Flipped | cls.NonSplitted

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

        total_counter = self.zero(umeshes, self.precision)
        umeshes.update()

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    @staticmethod
    def zero(umeshes, precision=1e-6):
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
        umeshes.update()

        if not total_counter:
            self.report({'INFO'}, 'Flipped faces not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} flipped faces')
        return {'FINISHED'}

    @staticmethod
    def flipped(umeshes):
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

        result = self.non_splitted(umeshes, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def non_splitted(umeshes, use_auto_smooth, user_angle, batch_inspect=False):
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
                    for umesh in umeshes:
                        umesh.sequence = []
                    return non_seam_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

            select_set = utils.edge_select_linked_set_func(sync)
            if umeshes.elem_mode not in ('EDGE', 'VERTEX'):
                umeshes.elem_mode = 'EDGE'

            for umesh in umeshes:
                uv = umesh.uv
                for edge in umesh.sequence:
                    select_set(edge, True, uv)
                umesh.sequence = []
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

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ALL', items=(('ALL', 'All', ''), ('INEXACT', 'Inexact', '')))
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

class UNIV_OT_BatchInspectFlags(Operator):
    bl_idname = 'uv.univ_batch_inspect_flags'
    bl_label = 'Flags'
    bl_description = "Inspect Flags"

    flag: IntProperty(name='Flag', default=0, min=0)

    def execute(self, context):
        from ..preferences import univ_settings
        tag = Inspect(self.flag)

        # Turn off the old (all) overlap flag if it doesn't match the new one.
        if tag in Inspect.AllOverlapFlags:
            if tag not in Inspect(univ_settings().batch_inspect_flags):
                univ_settings().batch_inspect_flags &= ~Inspect.AllOverlapFlags

        univ_settings().batch_inspect_flags ^= tag
        return {'FINISHED'}

INSPECT_INFO = {}

class UNIV_OT_BatchInspect(Operator):
    bl_idname = 'uv.univ_batch_inspect'
    bl_label = 'Inspect'
    bl_description = "Batch Inspect by Enabled tags"
    bl_options = {'REGISTER', 'UNDO'}

    inspect_all: BoolProperty(name='Inspect All', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        if 'Hidden' in INSPECT_INFO:
            box = col.box()
            box.label(text='Hidden faces found — the result may be incorrect', icon='INFO')
            col.separator()

        for inspect_flag in ('Overlap', 'Zero', 'Flipped', 'Non-Splitted'):
            if info := INSPECT_INFO.get(inspect_flag):
                box = col.box()
                box.label(text=f'{inspect_flag}: ' + info)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.inspect_all = event.alt
        return self.execute(context)

    def execute(self, context):
        from ..preferences import univ_settings
        flags = Inspect(univ_settings().batch_inspect_flags)
        global INSPECT_INFO
        INSPECT_INFO.clear()
        if not flags:
            self.report({'WARNING'}, 'Not found enabled flags')
            return {'CANCELLED'}

        umeshes = types.UMeshes()
        umeshes.update_tag = False

        umeshes.fix_context()

        if not umeshes:
            has_uv_maps = any(obj.data.uv_layers for obj in bpy.context.selected_objects if obj.type == 'MESH')
            self.report({'WARNING'}, 'Not found meshes with polygons' if has_uv_maps else 'Not found meshes with UV maps')
            return {'CANCELLED'}

        if (flags & Inspect.AllOverlapFlags) or self.inspect_all:
            if Inspect.Overlap in flags or (self.inspect_all and not (flags & Inspect.AllOverlapFlags)):
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'ALL'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'ALL')[1]

            elif Inspect.OverlapInexact in flags:
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'INEXACT'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'INEXACT')[1]


        if Inspect.Zero in flags or self.inspect_all:
            if count := UNIV_OT_Check_Zero.zero(umeshes):
                INSPECT_INFO['Zero'] = f'Detected {count} degenerate triangles'

        if Inspect.Flipped in flags or self.inspect_all:
            if count := UNIV_OT_Check_Flipped.flipped(umeshes):
                INSPECT_INFO['Flipped'] = f'Detected {count} flipped faces'

        if Inspect.NonSplitted in flags or self.inspect_all:  # Last check, because it switches elem mode to EDGE.
            result = UNIV_OT_Check_Non_Splitted.non_splitted(umeshes, use_auto_smooth=True, user_angle=180, batch_inspect=True)
            if info := UNIV_OT_Check_Non_Splitted.data_formatting(result):
                INSPECT_INFO['Non-Splitted'] = info


        has_hidden_faces = False
        for umesh in umeshes:
            if has_hidden_faces:
                break
            if umesh.is_full_face_selected:
                continue

            if umeshes.sync:
                for f in umesh.bm.faces:
                    if f.hide:
                        has_hidden_faces = True
                        break
            else:
                for f in umesh.bm.faces:
                    if not f.select:
                        has_hidden_faces = True
                        break
        if has_hidden_faces:
            if not INSPECT_INFO:
                self.report({'WARNING'}, 'No problems detected, but hidden faces could lead to incorrect results.')
                return {'FINISHED'}
            INSPECT_INFO['Hidden'] = True

        if not INSPECT_INFO:
            self.report({'INFO'}, 'No errors detected.')
        else:
            umeshes.silent_update()
            self.report({'WARNING'}, f'Detected {len(INSPECT_INFO)} errors.')
            return context.window_manager.invoke_popup(self, width=400)
        return {'FINISHED'}

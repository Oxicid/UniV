# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import bl_math

from mathutils.geometry import area_tri
from bpy.props import *
from bpy.types import Operator

from .. import utils
from .. import types


class UNIV_OT_Check_Zero(Operator):
    bl_idname = "uv.univ_check_zero"
    bl_label = "Zero"
    bl_description = "Select degenerate UVs (zero area UV triangles)"
    bl_options = {'REGISTER', 'UNDO'}

    precision: FloatProperty(name='Precision', default=0.0001, min=0, soft_max=0.001, step=0.0001, precision=7)  # noqa

    def draw(self, context):
        self.layout.prop(self, 'precision', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        umeshes = types.UMeshes()
        total_counter = self.zero(self.precision, umeshes, sync)

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    @staticmethod
    def zero(precision, umeshes, sync):
        if sync:
            utils.set_select_mode_mesh('FACE')
        bpy.ops.uv.select_all(action='DESELECT')

        precision = precision * 0.001
        total_counter = 0
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue

            local_counter = 0
            uv = umesh.uv
            loop_triangles = umesh.bm.calc_loop_triangles()
            for tris in loop_triangles:
                if sync:
                    if tris[0].face.hide:
                        continue
                else:
                    if not tris[0].face.select:
                        continue
                area = area_tri(tris[0][uv].uv, tris[1][uv].uv, tris[2][uv].uv)
                if area < precision:
                    if sync:
                        tris[0].face.select = True
                    else:
                        for crn in tris[0].face.loops:
                            crn[uv].select = True
                            crn[uv].select_edge = True
                    local_counter += 1
            umesh.update_tag = bool(local_counter)
            total_counter += local_counter
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
        total_counter = self.flipped(umeshes)

        if not total_counter:
            self.report({'INFO'}, 'Flipped faces not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} flipped faces')
        return {'FINISHED'}

    @staticmethod
    def flipped(umeshes):
        sync = umeshes.sync
        if sync:
            utils.set_select_mode_mesh('FACE')
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = 0
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue

            local_counter = 0
            uv = umesh.uv
            loop_triangles = umesh.bm.calc_loop_triangles()
            for tris in loop_triangles:
                if sync:
                    if tris[0].face.hide:
                        continue
                else:
                    if not tris[0].face.select:
                        continue
                a = tris[0][uv].uv
                b = tris[1][uv].uv
                c = tris[2][uv].uv

                area = a.cross(b) + b.cross(c) + c.cross(a)
                if area < 0:
                    if sync:
                        tris[0].face.select = True
                    else:
                        for crn in tris[0].face.loops:
                            crn_uv = crn[uv]
                            crn_uv.select = True
                            crn_uv.select_edge = True
                    local_counter += 1
            umesh.update_tag = bool(local_counter)
            total_counter += local_counter
        umeshes.update()
        return total_counter

class UNIV_OT_Check_Non_Splitted(Operator):
    bl_idname = 'uv.univ_check_non_splitted'
    bl_label = 'Non-Splitted'
    bl_options = {'REGISTER', 'UNDO'}

    check_non_seam: bpy.props.BoolProperty(name='Check Non-Seam', default=True)
    use_auto_smooth: bpy.props.BoolProperty(name='Use Auto Smooth', default=True)
    user_angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'check_non_seam')
        layout.prop(self, 'use_auto_smooth')
        layout.prop(self, 'user_angle', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        bpy.ops.uv.select_all(action='DESELECT')

        # clamp angle
        if self.use_auto_smooth:
            max_angle_from_obj_smooth = max(umesh.smooth_angle for umesh in umeshes)
            self.user_angle = bl_math.clamp(self.user_angle, 0.0, max_angle_from_obj_smooth)

        if umeshes.sync:
            if utils.get_select_mode_mesh() != 'EDGE':
                utils.set_select_mode_mesh('EDGE')
        else:
            if utils.get_select_mode_uv() not in ('EDGE', 'VERTEX'):
                utils.set_select_mode_uv('EDGE')
        result = self.select_inner(umeshes, self.check_non_seam, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def select_inner(umeshes, check_non_seam, use_auto_smooth, user_angle):
        non_seam_counter = 0
        angle_counter = 0
        sharps_counter = 0
        seam_counter = 0
        mtl_counter = 0

        sync = umeshes.sync
        is_boundary = utils.is_boundary_sync if sync else utils.is_boundary_non_sync

        for umesh in umeshes:
            local_non_seam_counter = 0
            local_angle_counter = 0
            local_sharps_counter = 0
            local_seam_counter = 0
            local_mtl_counter = 0

            if use_auto_smooth:
                angle = min(umesh.smooth_angle, user_angle)
            else:
                angle = user_angle

            uv = umesh.uv
            for crn in utils.calc_visible_uv_corners(umesh):
                edge = crn.edge
                shared_crn = crn.link_loop_radial_prev
                if sync:
                    if edge.select:
                        continue
                else:
                    if shared_crn[uv].select_edge:
                        continue

                if is_boundary(crn, uv):
                    if not check_non_seam or crn.edge.seam:
                        continue
                    local_non_seam_counter += 1
                    if shared_crn != crn:
                        shared_crn[uv].select = True
                        shared_crn[uv].select_edge = True
                        shared_crn.link_loop_next[uv].select = True
                        shared_crn.link_loop_next[uv].select = True
                elif not edge.smooth:
                    local_sharps_counter += 1
                elif edge.calc_face_angle() >= angle:
                    local_angle_counter += 1
                elif edge.seam:
                    local_seam_counter += 1
                elif shared_crn.face.material_index != crn.face.material_index:
                    local_mtl_counter += 1
                else:
                    continue

                if sync:
                    edge.select = True
                else:
                    utils.select_crn_uv_edge(crn, uv)

            if update_tag := (local_non_seam_counter or
                              local_angle_counter or
                              local_sharps_counter or
                              local_seam_counter or
                              local_mtl_counter):
                umesh.bm.select_flush_mode()
                non_seam_counter += local_non_seam_counter
                angle_counter += local_angle_counter
                sharps_counter += local_sharps_counter
                seam_counter += local_seam_counter
                mtl_counter += local_mtl_counter

            umesh.update_tag = update_tag
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
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.uv.select_overlap()
        count = 0
        for umesh in types.UMeshes():
            if umesh.sync:
                count += umesh.total_edge_sel
            else:
                if umesh.is_full_face_deselected:
                    continue
                uv = umesh.uv
                if umesh.is_full_face_selected:
                    for f in umesh.bm.faces:
                        for crn in f.loops:
                            count += crn[uv].select_edge
                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            for crn in f.loops:
                                count += crn[uv].select_edge
        if count:
            self.report({'WARNING'}, f"Found about {count} edges with overlap")
        else:
            self.report({'INFO'}, f"Edges with overlap not found")
        return {'FINISHED'}

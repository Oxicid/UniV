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


from ..utils import UMeshes

class UNIV_OT_Check_Zero(Operator):
    bl_idname = "uv.univ_check_zero"
    bl_label = "Select Degenerate"
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
        umeshes = UMeshes()
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
            uv = umesh.uv_layer
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
    bl_label = "Select Flipped"
    bl_description = "Select flipped UV faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
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
            uv = umesh.uv_layer
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
    bl_label = 'Select Non-Splitted'
    bl_options = {'REGISTER', 'UNDO'}

    use_auto_smooth: bpy.props.BoolProperty(name='Use Auto Smooth', default=True)
    user_angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        self.layout.prop(self, 'use_auto_smooth')
        self.layout.prop(self, 'user_angle', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = utils.UMeshes()
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
        result = self.select_inner(umeshes, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def select_inner(umeshes, use_auto_smooth, user_angle):
        angle_counter = 0
        sharps_counter = 0
        seam_counter = 0
        mtl_counter = 0

        sync = umeshes.sync
        is_boundary = utils.is_boundary_sync if sync else utils.is_boundary

        for umesh in umeshes:
            local_angle_counter = 0
            local_sharps_counter = 0
            local_seam_counter = 0
            local_mtl_counter = 0

            if use_auto_smooth:
                angle = min(umesh.smooth_angle, user_angle)
            else:
                angle = user_angle

            uv = umesh.uv_layer
            for crn in utils.calc_visible_uv_corners(umesh.bm, umesh.sync):
                edge = crn.edge
                shared_crn = crn.link_loop_radial_prev
                if sync:
                    if edge.select:
                        continue
                else:
                    if shared_crn[uv].select_edge:
                        continue

                if is_boundary(crn, uv):
                    continue

                if not edge.smooth:
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

            if update_tag := (local_angle_counter or local_sharps_counter or local_seam_counter or local_mtl_counter):
                umesh.bm.select_flush_mode()
                angle_counter += local_angle_counter
                sharps_counter += local_sharps_counter
                seam_counter += local_seam_counter
                mtl_counter += local_mtl_counter

            umesh.update_tag = update_tag
        return angle_counter, sharps_counter, seam_counter, mtl_counter

    @staticmethod
    def data_formatting(counters):
        angle_counter, sharps_counter, seam_counter, mtl_counter = counters
        r_text = ''
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
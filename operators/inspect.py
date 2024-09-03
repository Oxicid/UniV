# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

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
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH'

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
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH'

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

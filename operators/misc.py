# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from bpy.types import Operator
from bpy.props import *
from .. import utils
from ..types import UMeshes


class UNIV_OT_Pin(Operator):
    bl_idname = 'uv.univ_pin'
    bl_label = 'Pin'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Set/clear selected UV vertices as anchored between multiple unwrap operations\n\n" \
                     f"Default - Set Pin \n" \
                     f"Ctrl or Alt- Clear Pin\n\n" \
                     f"This button is used to free the 'P' button for the Pack operator."

    clear: BoolProperty(name='Clear', default=False)

    def __init__(self):
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.clear = (event.ctrl or event.alt)
        return self.execute(context)

    def execute(self, context):
        self.umeshes = UMeshes()
        set_pin_state = not self.clear

        if context.mode == 'EDIT_MESH':
            if self.umeshes.sync:
                has_selected = any(u.total_vert_sel for u in self.umeshes)
            else:
                has_selected = any(any(utils.calc_selected_uv_vert_corners_iter(u)) for u in self.umeshes)

            if has_selected:
                bpy.ops.uv.pin(clear=self.clear)
                return {'FINISHED'}
            else:
                for umesh in self.umeshes:
                    uv = umesh.uv
                    for crn in (visible_corners := utils.calc_visible_uv_corners(umesh)):
                        crn[uv].pin_uv = set_pin_state

                    umesh.update_tag = bool(visible_corners)
        else:
            for umesh in self.umeshes:
                uv = umesh.uv
                for f in umesh.bm.faces:
                    for crn in f.loops:
                        crn[uv].pin_uv = set_pin_state

        return self.umeshes.update()

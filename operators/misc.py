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
from ..preferences import settings


class UNIV_OT_Pin(Operator):
    bl_idname = 'uv.univ_pin'
    bl_label = 'Pin'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Set/clear selected UV vertices as anchored between multiple unwrap operations\n" \
                     f"With sync mode disabled, Edge mode switches to Vertex since the pins are not visible in edge mode\n\n" \
                     f"Default - Set Pin \n" \
                     f"Ctrl or Alt- Clear Pin\n\n" \
                     f"This button is used to free the 'P' button for the Pack operator"

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
                utils.set_select_mode_uv('VERTEX')
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


class UNIV_OT_TD_PresetsProcessing(Operator):
    bl_idname = "scene.univ_td_presets_processing"
    bl_label = "Presets Processing"

    operation_type: EnumProperty(default='ADD',
                                 options={'SKIP_SAVE'},
                                 items=(('ADD', 'Add', ''),
                                        ('REMOVE', 'Remove', ''),
                                        ('REMOVE_ALL', 'Remove All', ''))
                                 )

    def execute(self, _context):
        match self.operation_type:
            case 'ADD':
                self.add()
            case 'REMOVE':
                self.remove()
            case 'REMOVE_ALL':
                settings().texels_presets.clear()
                settings().active_td_index = -1
        for a in utils.get_areas_by_type('VIEW_3D'):
            a.tag_redraw()
        for a in utils.get_areas_by_type('IMAGE_EDITOR'):
            a.tag_redraw()

        return {'FINISHED'}

    def add(self):
        if len(td_presets := settings().texels_presets) >= 8:
            self.report({'WARNING'}, 'The preset limit of 8 units has been reached')
            return

        active_td_index = self.sanitize_index()

        my_user = settings().texels_presets.add()
        my_user.name = str(round(settings().texel_density))
        my_user.texel = settings().texel_density

        if len(td_presets) > 1:
            td_presets.move(len(td_presets), active_td_index + 1)
            settings().active_td_index = active_td_index + 1
        else:
            settings().active_td_index = len(td_presets) - 1

    def remove(self):
        if not len(td_presets := settings().texels_presets):
            self.report({'WARNING'}, 'The preset is empty')
            return
        active_td_index = self.sanitize_index()
        if len(td_presets) == active_td_index - 1:
            settings().active_td_index = -1
        td_presets.remove(active_td_index)
        self.sanitize_index()

    @staticmethod
    def sanitize_index():
        active_td_index = settings().active_td_index
        td_presets = settings().texels_presets

        if active_td_index < 0:
            active_td_index = len(td_presets) + active_td_index
        if active_td_index < 0 or active_td_index >= len(td_presets):
            active_td_index = len(td_presets) - 1
        settings().active_td_index = active_td_index
        return active_td_index

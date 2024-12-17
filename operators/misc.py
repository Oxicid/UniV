# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from bpy.types import Operator
from bpy.props import *
from collections import Counter
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


class UNIV_OT_Join(Operator):
    bl_idname = "object.univ_join"
    bl_label = "Join"
    bl_description = "Join with preserve uv channels"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object  # TODO: Without selected

    def execute(self, context):
        active_obj_type = context.active_object.type
        if active_obj_type == 'EMPTY':
            self.report({'WARNING'}, f"Empty object cannot be joined")
            return {'CANCELLED'}
        counter_obj_for_join = 0
        for obj in context.selected_objects:
            if obj.type == active_obj_type:
                counter_obj_for_join += 1
                if counter_obj_for_join > 1:
                    break
        else:
            self.report({'WARNING'}, f"There must be more than one {active_obj_type.capitalize()} type object")
            return {'CANCELLED'}

        if active_obj_type == 'MESH':
            conflict_attr_name = '_CONFLICT_WITH_UV'

            meshes = {obj.data for obj in context.selected_objects if obj != context.active_object and obj.type == 'MESH'}
            removed_extra_channels_counter = 0
            for m in meshes:
                removed_extra_channels_counter += self.sanitize_uv(m)
            removed_extra_channels_counter += self.sanitize_uv(context.active_object.data)

            uv_names_counter = [Counter() for _ in range(8)]
            active_uv_size = len(context.active_object.data.uv_layers)

            for mesh in meshes:
                for idx, uv in enumerate(mesh.uv_layers):
                    uv_names_counter[idx][uv.name] += 1

            uv_names = [uv.name for uv in context.active_object.data.uv_layers]
            max_index = active_uv_size

            for idx, counter in enumerate(uv_names_counter[active_uv_size:]):
                max_key = max(counter, key=counter.get, default=None)
                if max_key is None:
                    break
                uv_names.append(max_key)
                max_index = max(idx+1, max_index)

            for idx, uv_name in enumerate(uv_names):
                if conflict_attr_name in uv_name:
                    uv_names[idx] = uv_name.replace(conflict_attr_name, str(idx))

            for idx, uv in enumerate(context.active_object.data.uv_layers):
                if uv.name in uv_names_counter[idx]:
                    del uv_names_counter[idx][uv.name]

            conflicts_counter = sum(len(c) for c in uv_names_counter)
            meshes.add(context.active_object.data)

            for mesh in meshes:
                for attr in mesh.attributes:
                    if any((uv.name == attr.name) for uv in mesh.uv_layers):
                        continue

                    if attr.name in uv_names:
                        conflicts_counter += 1
                        attr.name += conflict_attr_name

            added_uvs_counter = 0
            for mesh in meshes:
                added_uvs_counter += self.add_missed_uvs(max_index, mesh, uv_names)
                self.rename_uvs(uv_names, mesh)

            info = ''
            if conflicts_counter:
                info += f"Resolver {conflicts_counter} names conflicts. "
            if removed_extra_channels_counter:
                info += f"Removed {removed_extra_channels_counter} extra channels in total."
            if added_uvs_counter:
                info += f"Added {added_uvs_counter} channels in total."

            if info:
                self.report({'WARNING'}, info)

        return bpy.ops.object.join()

    @staticmethod
    def add_missed_uvs(max_index, mesh, names):
        mesh_uv_size = len(mesh.uv_layers)
        if max_index == 0 or mesh_uv_size == max_index:
            return 0
        assert max_index <= 8
        assert mesh_uv_size < max_index

        import numpy as np
        do_init = len(mesh.uv_layers) == 0
        if not do_init:
            uv_coords = np.empty(len(mesh.loops) * 2, dtype='float32')
            uv = mesh.uv_layers[-1]
            uv.data.foreach_get("uv", uv_coords)

        counter = 0
        while (mesh_uv_size := len(mesh.uv_layers)) != max_index:
            uv = mesh.uv_layers.new(name=names[mesh_uv_size], do_init=do_init)
            counter += 1
            if not do_init:
                uv.data.foreach_set("uv", uv_coords)  # noqa
        return counter

    def rename_uvs(self, names, mesh):
        assert len(mesh.uv_layers) == len(names)
        for _ in range(10):
            renamed = True
            for uv, new_name in zip(mesh.uv_layers, names):
                if uv.name != new_name:
                    uv.name = new_name
                    renamed = False
            if renamed:
                return True
        return False
        self.report({'WARNING'}, f'Mesh {mesh.name} do not rename uv layers')

    def sanitize_uv(self, mesh):
        counter = 0
        if len(mesh.uv_layers) > 8:
            for uv in reversed(mesh.uv_layers[8:]):
                mesh.uv_layers.remove(uv)
                counter += 1
            self.report({'WARNING'}, f'Mesh {mesh.name} delete {counter} extra channels')
        return counter

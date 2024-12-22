# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import bmesh
import numpy as np

from bpy.types import Operator
from bpy.props import *
from collections import Counter
from .. import utils
from ..types import UMeshes
from .. import preferences
from ..preferences import univ_settings

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
                univ_settings().texels_presets.clear()
                univ_settings().active_td_index = -1
        for a in utils.get_areas_by_type('VIEW_3D'):
            a.tag_redraw()
        for a in utils.get_areas_by_type('IMAGE_EDITOR'):
            a.tag_redraw()

        return {'FINISHED'}

    def add(self):
        if len(td_presets := univ_settings().texels_presets) >= 8:
            self.report({'WARNING'}, 'The preset limit of 8 units has been reached')
            return

        active_td_index = self.sanitize_index()

        my_user = univ_settings().texels_presets.add()
        my_user.name = str(round(univ_settings().texel_density))
        my_user.texel = univ_settings().texel_density

        if len(td_presets) > 1:
            td_presets.move(len(td_presets), active_td_index + 1)
            univ_settings().active_td_index = active_td_index + 1
        else:
            univ_settings().active_td_index = len(td_presets) - 1

    def remove(self):
        if not len(td_presets := univ_settings().texels_presets):
            self.report({'WARNING'}, 'The preset is empty')
            return
        active_td_index = self.sanitize_index()
        if len(td_presets) == active_td_index - 1:
            univ_settings().active_td_index = -1
        td_presets.remove(active_td_index)
        self.sanitize_index()

    @staticmethod
    def sanitize_index():
        active_td_index = univ_settings().active_td_index
        td_presets = univ_settings().texels_presets

        if active_td_index < 0:
            active_td_index = len(td_presets) + active_td_index
        if active_td_index < 0 or active_td_index >= len(td_presets):
            active_td_index = len(td_presets) - 1
        univ_settings().active_td_index = active_td_index
        return active_td_index


class UNIV_OT_Join(Operator):
    bl_idname = "object.univ_join"
    bl_label = "Join"
    bl_description = "Join with preserve uv channels"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # TODO: Sanitize UV-names in materials (scip if not assign material)
        # TODO: Fix uv names in other obj by sanitized material
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


class UNIV_OT_UV_Layers_Manager(Operator):
    bl_idname = 'uv.univ_layers_manager'
    bl_label = 'UV Maps'
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(default='ADD',
                                 options={'HIDDEN'},
                                 items=(('ADD', 'Add', ''),
                                        ('REMOVE', 'Remove', ''),
                                        ('RESET_NAMES', 'Reset Names', ''),

                                        ('SORT', 'Sort', ''),
                                        ('SYNC_SEAMS', 'Sync Seams', ''),
                                        ('SET_ACTIVE_RENDER', 'Set Active Render', ''),
                                        )
                         )

    @classmethod
    def poll(cls, context):
        return context.active_object

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        return self.execute(context)

    def execute(self, context):
        return {'FINISHED'}

    @staticmethod
    def update_uv_layers_props():
        preferences.UV_LAYERS_ENABLE = False
        try:
            UNIV_OT_UV_Layers_Manager.update_uv_layers_props_ex()
        except:  # noqa
            if preferences.debug():
                import traceback
                traceback.print_exc()
        finally:
            preferences.UV_LAYERS_ENABLE = True

    @staticmethod
    def update_uv_layers_props_ex():
        settings = univ_settings()
        context = bpy.context
        active_obj = context.active_object
        if not active_obj or active_obj.type != 'MESH':
            if settings.uv_layers_size:
                settings.uv_layers_size = 0
                utils.update_univ_panels()
            return

        if context.mode == 'EDIT_MESH':
            selected_objects = context.objects_in_mode_unique_data
        else:
            selected_objects = (obj_ for obj_ in context.selected_objects if obj_.type == 'MESH')

        uv_presets = settings.uv_layers_presets
        UNIV_OT_UV_Layers_Manager.sanitize_size(uv_presets)

        act_obj_uv_layers = active_obj.data.uv_layers
        if act_obj_uv_layers:
            act_obj_uv_layers_size = len(act_obj_uv_layers)
            if act_obj_uv_layers_size > 8:
                for preset, uv in zip(uv_presets, act_obj_uv_layers):
                    preset.name = uv.name
                    preset.flag = 2
                settings.uv_layers_size = 8
                utils.update_univ_panels()
                return
            act_obj_uv_idx = 0
            act_obj_uv_render_idx = 0
            for idx, uv in enumerate(act_obj_uv_layers):
                if uv.active:
                    act_obj_uv_idx = idx
                if uv.active_render:
                    act_obj_uv_render_idx = idx

            act_obj_uv_names = tuple(uv.name for uv in act_obj_uv_layers)
            act_obj_uv_names_tags = [True for _ in act_obj_uv_names]
            for obj in selected_objects:
                uv_layers = obj.data.uv_layers
                uv_layers_size = len(uv_layers)
                # Frequent case
                if uv_layers_size == act_obj_uv_layers_size:
                    for idx, uv in enumerate(uv_layers):

                        if act_obj_uv_names_tags[idx]:
                            if act_obj_uv_names[idx] != uv.name:
                                act_obj_uv_names_tags[idx] = False

                        if uv.active:
                            if act_obj_uv_idx != idx:
                                act_obj_uv_names_tags[idx] = False

                        if uv.active_render:
                            if act_obj_uv_render_idx != idx:
                                act_obj_uv_names_tags[idx] = False

                elif uv_layers_size > act_obj_uv_layers_size:
                    settings.uv_layers_size = uv_layers_size
                    for idx, uv in enumerate(uv_layers):
                        if idx == 8:
                            break
                        preset = uv_presets[idx]
                        if idx < act_obj_uv_layers_size:
                            preset.name = act_obj_uv_names[idx]
                            preset.flag = not act_obj_uv_names_tags[idx]
                        else:
                            preset.name = uv.name
                            preset.flag = 2

                    if settings.uv_layers_active_idx >= uv_layers_size-1:
                        settings.uv_layers_active_idx = uv_layers_size-1  # clamp
                        settings.uv_layers_active_render_idx = act_obj_uv_render_idx
                    utils.update_univ_panels()
                    return

                else:  # uv_layers_size < act_obj_uv_layers_size:
                    settings.uv_layers_size = act_obj_uv_layers_size
                    if not uv_layers_size:
                        for idx in range(act_obj_uv_layers_size):
                            preset = uv_presets[idx]
                            preset.name = act_obj_uv_names[idx]
                            preset.flag = 2

                        settings.uv_layers_active_idx = act_obj_uv_idx
                        settings.uv_layers_active_render_idx = -1
                        utils.update_univ_panels()
                        return

                    for idx in range(act_obj_uv_layers_size):
                        preset = uv_presets[idx]
                        if idx < uv_layers_size:
                            preset.name = act_obj_uv_names[idx]
                            preset.flag = not act_obj_uv_names_tags[idx]
                        else:
                            preset.name = act_obj_uv_names[idx]
                            preset.flag = 2

                    settings.uv_layers_active_render_idx = -1
                    if settings.uv_layers_active_idx > act_obj_uv_layers_size-1:
                        settings.uv_layers_active_idx = act_obj_uv_layers_size-1
                    utils.update_univ_panels()
                    return

            for idx in range(act_obj_uv_layers_size):
                preset = uv_presets[idx]
                preset.name = act_obj_uv_names[idx]
                preset.flag = not act_obj_uv_names_tags[idx]

            settings.uv_layers_size = act_obj_uv_layers_size
            settings.uv_layers_active_idx = act_obj_uv_idx
            settings.uv_layers_active_render_idx = act_obj_uv_render_idx
        else:
            obj_with_max_uv = max(selected_objects, key=lambda ob: len(ob.data.uv_layers))
            uv_layers = obj_with_max_uv.data.uv_layers
            uv_layers_size = len(uv_layers)

            settings.uv_layers_size = uv_layers_size

            for preset, uv in zip(uv_presets, uv_layers):
                preset.name = uv.name
                preset.flag = 2

            settings.uv_layers_active_idx = 0
            settings.uv_layers_active_render_idx = -1
        utils.update_univ_panels()

    @staticmethod
    def sanitize_size(presets):
        if (size := len(presets)) == 8:
            return

        if size < 8:
            for _ in range(8-size):
                presets.add()
        else:
            for i in range(len(presets), 8, -1):
                presets.remove(i-1)

    @staticmethod
    def univ_uv_layers_update(_, deps):
        from .. import ui
        for update_obj in deps.updates:
            if update_obj.is_updated_transform:
                return
            else:
                ui.REDRAW_UV_LAYERS = True
                return

    @staticmethod
    def append_handler_with_delay():
        if univ_settings().uv_layers_show:
            bpy.app.handlers.depsgraph_update_post.append(UNIV_OT_UV_Layers_Manager.univ_uv_layers_update)
        return

class UNIV_OT_MoveUpDownBase(Operator):
    bl_options = {'REGISTER', 'UNDO'}
    with_names: BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'
        # and (settings := univ_settings()).uv_layers_size != settings.uv_layers_active_idx+1  TODO: Bug report

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.with_names = not event.alt
        return self.execute(context)

    def move_uv_bm(self, umesh, idx, up, with_names=False):
        idx_inc_dec = 1
        layers = umesh.obj.data.uv_layers
        if up:
            if idx == 0 or len(layers) == 0 or len(layers) < idx + 1:
                return False
        else:
            idx_inc_dec = -1
            if len(layers) <= idx + 1:
                return False

        other_idx = idx - idx_inc_dec
        if umesh.is_edit_bm:
            self._swap_uv_bm(umesh, idx, other_idx, with_names)
        else:
            self._swap_uv_mesh(umesh, idx, other_idx, with_names)

    @staticmethod
    def _swap_uv_bm(umesh, idx, other_idx, with_names, change_active_idx=True):
        uv = umesh.uv
        layers = umesh.obj.data.uv_layers

        if layers.active_index != idx:
            layers.active_index = idx

        coords = [crn[uv].uv.copy() for f in umesh.bm.faces for crn in f.loops]
        layers.active_index = other_idx
        it = (crn[uv].uv for f in umesh.bm.faces for crn in f.loops)

        for uv_a_copy, uv_b in zip(coords, it):
            uv_b_copy = uv_b.copy()
            uv_b[:] = uv_a_copy
            uv_a_copy[:] = uv_b_copy

        layers.active_index = idx

        it = (crn for f in umesh.bm.faces for crn in f.loops)
        for uv_b_copy, crn_a in zip(coords, it):
            crn_a[uv].uv = uv_b_copy

        if change_active_idx:  # small optimization
            layers.active_index = other_idx

        if with_names:
            name_a = layers[idx].name
            name_b = layers[other_idx].name
            layers[idx].name = 'temp'
            layers[other_idx].name = name_a
            layers[idx].name = name_b

    @staticmethod
    def _swap_uv_mesh(umesh, idx, other_idx, with_names, change_active_idx=True):
        size = len(umesh.obj.data.loops) * 2
        uvs_a = np.empty(size, dtype='float32')
        uvs_b = np.empty(size, dtype='float32')

        layers = umesh.obj.data.uv_layers
        uv_layer_a = layers[idx]
        uv_layer_b = layers[other_idx]
        uv_layer_a.data.foreach_get("uv", uvs_a)
        uv_layer_b.data.foreach_get("uv", uvs_b)

        uv_layer_a.data.foreach_set("uv", uvs_b)
        uv_layer_b.data.foreach_set("uv", uvs_a)

        if change_active_idx:  # small optimization
            if layers.active_index != other_idx:
                layers.active_index = other_idx

        if with_names:
            name_a = layers[idx].name
            name_b = layers[other_idx].name
            layers[idx].name = 'temp'
            layers[other_idx].name = name_a
            layers[idx].name = name_b

class UNIV_OT_MoveUp(UNIV_OT_MoveUpDownBase):
    bl_idname = 'mesh.univ_move_up'
    bl_label = 'Up'

    def execute(self, context):
        settings = univ_settings()
        if settings.uv_layers_active_idx == 0:
            self.report({'WARNING'}, 'Cannot move up')
            return {'CANCELLED'}

        umeshes = UMeshes.calc_any_unique()
        for umesh in umeshes:
            if self.move_uv_bm(umesh, settings.uv_layers_active_idx, up=True, with_names=self.with_names):
                if umesh.is_edit_bm:
                    umesh.update()
        umeshes.free()
        return {'FINISHED'}

class UNIV_OT_MoveDown(UNIV_OT_MoveUpDownBase):
    bl_idname = 'mesh.univ_move_down'
    bl_label = 'Down'

    def execute(self, context):
        settings = univ_settings()
        if settings.uv_layers_size == settings.uv_layers_active_idx+1:
            self.report({'WARNING'}, 'Cannot move down')
            return {'CANCELLED'}

        umeshes = UMeshes.calc_any_unique()
        for umesh in umeshes:
            if self.move_uv_bm(umesh, settings.uv_layers_active_idx, up=False, with_names=self.with_names):
                if umesh.is_edit_bm:
                    umesh.update()
        umeshes.free()
        return {'FINISHED'}

class UNIV_OT_Add(Operator):
    bl_idname = 'mesh.univ_add'
    bl_label = 'Add'

    add_missed: BoolProperty(name='Add with Missed', default=False)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.add_missed = event.alt
        return self.execute(context)

    def execute(self, context):
        settings = univ_settings()
        objects = utils.calc_any_unique_obj()
        target_min_size = min(len(obj.data.uv_layers) for obj in objects)+1
        if self.add_missed:
            target_max_size = max(len(obj.data.uv_layers) for obj in objects)+1
            if target_min_size != target_max_size:
                target_min_size = target_max_size-1

        if target_min_size > 8:
            self.report({'WARNING'}, 'The limit of 8 channels has been reached.')
            return {'CANCELLED'}

        for obj in objects:
            if bpy.context.mode == 'EDIT_MESH':
                if self.add_missed_uvs_bm(obj, target_min_size, settings.uv_layers_active_idx):
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            else:
                if self.add_missed_uvs(obj, target_min_size, settings.uv_layers_active_idx):
                    obj.data.update()
        return {'FINISHED'}

    @staticmethod
    def add_missed_uvs(obj, target_size, active_index):
        assert target_size <= 8
        mesh = obj.data
        mesh_uv_size = len(mesh.uv_layers)
        if target_size == 0 or mesh_uv_size == target_size or mesh_uv_size >= 8:
            return 0

        counter = 0
        if mesh.uv_layers:
            uv_coords = np.empty(len(mesh.loops) * 2, dtype='float32')
            if active_index > len(mesh.uv_layers)-1:
                uv = mesh.uv_layers[-1]
            else:
                uv = mesh.uv_layers[active_index]
            uv.data.foreach_get("uv", uv_coords)
            while (len(mesh.uv_layers)) < target_size:
                uv = mesh.uv_layers.new(do_init=False)
                uv.data.foreach_set("uv", uv_coords)  # noqa
                counter += 1
        else:
            while (len(mesh.uv_layers)) < target_size:
                mesh.uv_layers.new(do_init=True)
                counter += 1
        return counter

    @staticmethod
    def add_missed_uvs_bm(obj, target_size, active_index):
        assert target_size <= 8
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        uv_size = len(bm.loops.layers.uv)
        if target_size == 0 or uv_size == target_size or uv_size >= 8:
            return 0

        if uv_size == 0:
            while (len(bm.loops.layers.uv)) < target_size:
                bm.loops.layers.uv.new('UVMap')
            return target_size

        counter = 0
        uv = bm.loops.layers.uv.verify()
        if uv_size >= active_index+1:
            if mesh.uv_layers.active_index != active_index:
                mesh.uv_layers.active_index = active_index

        coords = [crn[uv].uv.copy() for f in bm.faces for crn in f.loops]

        while (len(bm.loops.layers.uv)) < target_size:
            bm.loops.layers.uv.new('UVMap')
            obj.data.uv_layers.active_index = len(mesh.uv_layers)-1
            corners = (crn for f in bm.faces for crn in f.loops)
            for crn, uv_co in zip(corners, coords):
                crn[uv].uv = uv_co
            counter += 1
        bm = bm  # noqa
        return counter

class UNIV_OT_Remove(Operator):
    bl_idname = 'mesh.univ_remove'
    bl_label = 'Remove'

    remove_all: BoolProperty(name='Remove All', default=False)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.remove_all = event.alt
        return self.execute(context)

    def execute(self, context):
        settings = univ_settings()
        objects = utils.calc_any_unique_obj()
        max_size = max(len(obj.data.uv_layers) for obj in objects)

        if max_size == 0:
            self.report({'WARNING'}, 'All uv maps removed')
            return {'CANCELLED'}

        target_idx = settings.uv_layers_active_idx
        for obj in objects:
            mesh = obj.data
            uv_layers = mesh.uv_layers
            if not uv_layers:
                continue

            if self.remove_all:
                if bpy.context.mode == 'EDIT_MESH':
                    bm = bmesh.from_edit_mesh(mesh)
                    for uv_layer in reversed(bm.loops.layers.uv):
                        bm.loops.layers.uv.remove(uv_layer)
                    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
                else:
                    for uv_layer in reversed(uv_layers):
                        uv_layers.remove(uv_layer)
                    obj.data.update()
            else:
                if len(uv_layers) >= target_idx+1:
                    if bpy.context.mode == 'EDIT_MESH':
                        bm = bmesh.from_edit_mesh(mesh)
                        bm.loops.layers.uv.remove(bm.loops.layers.uv[target_idx])
                        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                    else:
                        uv_layers.remove(uv_layers[target_idx])
                        obj.data.update()

        return {'FINISHED'}

class UNIV_OT_SetActiveRender(Operator):
    bl_idname = 'mesh.univ_active_render_set'
    bl_label = 'Remove'

    idx: IntProperty(name='Set Active', default=0, min=0, max=8, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        objects = utils.calc_any_unique_obj()

        for obj in objects:
            mesh = obj.data
            uv_layers = mesh.uv_layers
            if len(uv_layers) >= self.idx+1:
                if not uv_layers[self.idx].active_render:
                    uv_layers[self.idx].active_render = True
        return {'FINISHED'}

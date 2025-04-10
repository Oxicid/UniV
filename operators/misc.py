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
from .. import types
from ..types import UMeshes
from .. import preferences
from ..preferences import prefs, univ_settings
from mathutils import Vector

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.umeshes.fix_context()
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
            objects = utils.calc_any_unique_obj()
            removed_extra_channels_counter = sum(self.sanitize_uv(obj.data) for obj in objects)

            uv_names, conflicts_counter = self.sanitize_attr_names_and_get_names(objects)

            max_uv_size = max(len(obj.data.uv_layers) for obj in objects)
            added_uvs_counter = 0
            for obj in objects:
                added_uvs_counter += self.add_missed_uvs(obj, max_uv_size)
                self.rename_uvs(uv_names, obj.data)

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
    def sanitize_attr_names_and_get_names(objects):
        conflict_attr_name = '_CONFLICT_WITH_UV'
        active_uv_layers = bpy.context.active_object.data.uv_layers
        uv_names_counter = [Counter() for _ in range(8)]
        for obj in objects:
            for idx, uv in enumerate(obj.data.uv_layers):
                uv_names_counter[idx][uv.name] += 1

        uv_names = [uv.name for uv in active_uv_layers]
        for idx, counter in enumerate(uv_names_counter[len(active_uv_layers):]):
            frequent_uv_name = max(counter, key=counter.get, default=None)
            if frequent_uv_name is None:
                break
            uv_names.append(frequent_uv_name)

        # Remove potential '_CONFLICT_WITH_UV' name in uv maps
        for idx, uv_name in enumerate(uv_names):
            if conflict_attr_name in uv_name:
                uv_names[idx] = uv_name.replace(conflict_attr_name, str(idx))

        # Remove names for count conflict names
        for idx, uv in enumerate(active_uv_layers):
            if uv.name in uv_names_counter[idx]:
                del uv_names_counter[idx][uv.name]
        conflicts_counter = sum(len(c) for c in uv_names_counter)

        # Rename attributes for resolve potential name conflict
        for obj in objects:
            for attr in obj.data.attributes:
                if any((uv.name == attr.name) for uv in obj.data.uv_layers):
                    continue

                if attr.name in uv_names:
                    conflicts_counter += 1
                    attr.name += conflict_attr_name
        return uv_names, conflicts_counter

    @staticmethod
    def add_missed_uvs(obj, target_size, active_index=8):
        assert target_size <= 8
        mesh = obj.data
        mesh_uv_size = len(mesh.uv_layers)
        if target_size == 0 or mesh_uv_size == target_size or mesh_uv_size >= 8:
            return 0

        if mesh.uv_layers:
            if active_index > len(mesh.uv_layers)-1:
                uv = mesh.uv_layers[-1]
            else:
                uv = mesh.uv_layers[active_index]

            uv_coords = np.empty(len(mesh.loops) * 2, dtype='float32')
            uv.data.foreach_get("uv", uv_coords)
            while (len(mesh.uv_layers)) < target_size:
                uv = mesh.uv_layers.new(do_init=False)
                uv.data.foreach_set("uv", uv_coords)  # noqa
        else:
            while (len(mesh.uv_layers)) < target_size:
                mesh.uv_layers.new(do_init=True)
        return target_size - mesh_uv_size

    def rename_uvs(self, names, mesh):
        assert len(mesh.uv_layers) == len(names)
        changed = False
        for _ in range(10):
            renamed = True
            for uv, new_name in zip(mesh.uv_layers, names):
                if uv.name != new_name:
                    uv.name = new_name
                    renamed = False
                    changed = True
            if renamed:
                return changed
        if preferences.debug():
            self.report({'WARNING'}, f'Mesh {mesh.name} do not rename uv layers')
        return changed

    def sanitize_uv(self, mesh):
        counter = 0
        if len(mesh.uv_layers) > 8:
            for uv in reversed(mesh.uv_layers[8:]):
                mesh.uv_layers.remove(uv)
                counter += 1
            self.report({'WARNING'}, f'Mesh {mesh.name} delete {counter} extra channels')
        return counter


class UNIV_OT_Hide(Operator):
    bl_idname = "uv.univ_hide"
    bl_label = 'Hide'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Hide selected UV"

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if not (context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV'):
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}

        if event.value == 'PRESS':
            self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
            self.mouse_pos = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        is_incorrect_hide_mod_for_non_sync = not utils.sync() and utils.get_select_mode_mesh_reversed() != 'FACE'
        if is_incorrect_hide_mod_for_non_sync:
            utils.set_select_mode_mesh('FACE')

        self.umeshes = UMeshes(report=self.report)
        self.umeshes.fix_context()

        if self.umeshes:
            self.umeshes.elem_mode = utils.get_select_mode_mesh_reversed()
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_pos:
            return self.pick_hide()

        if self.umeshes.sync:
            if self.umeshes.elem_mode == 'FACE':
                return bpy.ops.uv.hide(unselected=False)

            self.umeshes.filter_by_selected_mesh_faces()
            if not self.umeshes:
                # TODO: Implement hide by view box
                return bpy.ops.uv.hide(unselected=False)

            if self.umeshes.elem_mode == 'VERTEX':
                self.vert_hide_sync_preprocessing()
            elif self.umeshes.elem_mode == 'EDGE':
                self.edge_hide_sync_preprocessing()

            res = bpy.ops.uv.hide(unselected=False)
            for umesh in self.umeshes:
                if umesh.sequence:
                    for f in umesh.sequence:
                        f.hide_set(False)
                    umesh.update()
                if preferences.debug():
                    if umesh.total_face_sel or umesh.total_edge_sel or umesh.total_vert_sel:
                        self.report({'WARNING'}, 'Undefined Behavior: Has selected elements even after applying Hide operation')
            return res
        else:
            # bpy.ops.uv.hide sometimes works incorrectly in 'FACE' mode too,
            # maybe it's something to do with not updating bm.select_mode
            for umesh in self.umeshes:
                if is_incorrect_hide_mod_for_non_sync:
                    umesh.bm.select_mode = {'FACE'}
                uv = umesh.uv
                update_tag = False
                for f in utils.calc_visible_uv_faces(umesh):
                    if any(crn[uv].select for crn in f.loops):
                        f.select = False
                        update_tag = True
                umesh.update_tag = update_tag
            return self.umeshes.update()

    def pick_hide(self):
        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            for isl in types.AdvIslands.calc_visible_with_mark_seam(umesh):
                hit.find_nearest_island_by_crn(isl)

        if not hit:
            self.report({'INFO'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        hit.island.hide_first()
        hit.island.umesh.update()
        return {'FINISHED'}

    def vert_hide_sync_preprocessing(self):
        def is_hide_face():
            for crn in f.loops:
                crn_vert = crn.vert
                if crn_vert.select:
                    if all(not f__.select for f__ in crn_vert.link_faces):
                        return True
                    for crn__ in utils.linked_crn_to_vert_pair_iter(crn, uv, True):
                        if crn__.face.select:
                            return True
            return False

        for umesh in self.umeshes:
            uv = umesh.uv
            visible_faces = utils.calc_visible_uv_faces(umesh)
            for f in visible_faces:
                # Skip selected and unselected face
                if f.select or not any(crn_.vert.select for crn_ in f.loops):
                    continue
                if not is_hide_face():
                    umesh.sequence.append(f)

    def edge_hide_sync_preprocessing(self):
        def is_hide_face():
            for crn in f.loops:
                if crn.edge.select:
                    pair = crn.link_loop_radial_prev
                    pair_face = pair.face
                    if pair_face.hide:
                        return True

                    if utils.is_pair(crn, crn.link_loop_radial_prev, uv):
                        return True
                    else:
                        if pair_face.select:
                            continue
                        else:
                            return True
            return False

        for umesh in self.umeshes:
            uv = umesh.uv
            visible_faces = utils.calc_visible_uv_faces(umesh)
            for f in visible_faces:
                # Skip selected and unselected face
                if f.select or not any(crn_.edge.select for crn_ in f.loops):
                    continue

                if not is_hide_face():
                    umesh.sequence.append(f)


LAST_MOUSE_POS = -100_000, -100_000
REPEAT_MOUSE_POS_COUNT = 0

class UNIV_OT_SetCursor2D(Operator):
    bl_idname = "uv.univ_set_cursor_2d"
    bl_label = 'Set Cursor 2D'

    # TODO: Implement 3D Ctrl + Shift + Right Mouse Button for consistent with 2D

    def invoke(self, context, event):
        if not (context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV'):
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}

        global LAST_MOUSE_POS
        global REPEAT_MOUSE_POS_COUNT
        int_mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        if LAST_MOUSE_POS == int_mouse_pos:
            REPEAT_MOUSE_POS_COUNT += 1
        else:
            REPEAT_MOUSE_POS_COUNT = 0
            LAST_MOUSE_POS = int_mouse_pos

        if REPEAT_MOUSE_POS_COUNT >= 3:
            REPEAT_MOUSE_POS_COUNT = 0

        max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        mouse_pos = Vector(context.region.view2d.region_to_view(*int_mouse_pos))

        pt = None
        min_dist = max_distance
        mouse_pos = mouse_pos

        if REPEAT_MOUSE_POS_COUNT != 1:
            if context.mode == 'EDIT_MESH' and REPEAT_MOUSE_POS_COUNT == 0:
                grid_pt = Vector(utils.round_threshold(v, 1/2) for v in mouse_pos)
                if (grid_dist := (grid_pt - mouse_pos).length * 2.0) < min_dist:
                    pt = grid_pt
                    min_dist = grid_dist
            else:
                zoom = types.View2D.get_zoom(context.region.view2d)
                divider = 1 / 8 if zoom <= 1600 else 1 / 64
                divider = divider if zoom <= 12800 else 1 / 64 / 8

                grid_pt = Vector(utils.round_threshold(v, divider) for v in mouse_pos)
                if (grid_dist := (grid_pt - mouse_pos).length) < min_dist:
                    pt = grid_pt
                    min_dist = grid_dist

        if context.mode == 'EDIT_MESH' and REPEAT_MOUSE_POS_COUNT == 0:
            zero_pt = Vector((0.0, 0.0))
            for umesh in (umeshes := UMeshes()):
                uv = umesh.uv
                if prefs().snap_points_default == 'ALL':
                    for f in utils.calc_visible_uv_faces(umesh):
                        face_center_sum = zero_pt.copy()
                        corners = f.loops
                        prev_co = corners[-1][uv].uv
                        for crn in corners:
                            cur_co = crn[uv].uv
                            if (dist := (cur_co - mouse_pos).length) < min_dist:
                                pt = cur_co
                                min_dist = dist
                            edge_center = (prev_co + cur_co) * 0.5
                            if (dist := (edge_center - mouse_pos).length) < min_dist:
                                pt = edge_center
                                min_dist = dist

                            prev_co = cur_co
                            face_center_sum += cur_co
                        face_center = face_center_sum / len(corners)
                        if (dist := (face_center - mouse_pos).length) < min_dist:
                            pt = face_center
                            min_dist = dist

                elif umeshes.elem_mode == 'VERTEX':
                    for f in utils.calc_visible_uv_faces(umesh):
                        for crn in f.loops:
                            uv_co = crn[uv].uv
                            if (length := (mouse_pos - uv_co).length) < min_dist:
                                pt = uv_co
                                min_dist = length
                elif umeshes.elem_mode == 'EDGE':
                    for f in utils.calc_visible_uv_faces(umesh):
                        corners = f.loops
                        prev_co = corners[-1][uv].uv
                        for crn in corners:
                            cur_co = crn[uv].uv
                            if (dist := (cur_co - mouse_pos).length) < min_dist:
                                pt = cur_co
                                min_dist = dist
                            edge_center = (prev_co + cur_co) * 0.5
                            if (dist := (edge_center - mouse_pos).length) < min_dist:
                                pt = edge_center
                                min_dist = dist
                            prev_co = cur_co
                else:
                    for f in utils.calc_visible_uv_faces(umesh):
                        face_center_sum = zero_pt.copy()
                        corners = f.loops
                        for crn in corners:
                            cur_co = crn[uv].uv
                            if (dist := (cur_co - mouse_pos).length) < min_dist:
                                pt = cur_co
                                min_dist = dist
                            face_center_sum += cur_co
                        face_center = face_center_sum / len(corners)
                        if (dist := (face_center - mouse_pos).length) < min_dist:
                            pt = face_center
                            min_dist = dist

        if not pt:
            if REPEAT_MOUSE_POS_COUNT == 1:
                self.report({'INFO'}, 'Force Set Cursor 2D to Mouse position')
            pt = mouse_pos
        elif REPEAT_MOUSE_POS_COUNT == 2:
            self.report({'INFO'}, 'Force Set Cursor 2D to Grid')

        context.space_data.cursor_location = pt
        return {'FINISHED'}


class UNIV_OT_Focus(Operator):
    bl_idname = "uv.univ_focus"
    bl_label = 'Focus'

    def invoke(self, context, event):
        if not (context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV'):
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}
        assert context.mode == 'EDIT_MESH'

        bounds = types.BBox()
        umeshes = UMeshes()
        color = (1,1,0,1)
        for umesh in umeshes:
            uv = umesh.uv
            bounds.update(crn[uv].uv for crn in utils.calc_selected_uv_vert_corners_iter(umesh))
        if bounds == types.BBox():
            color = (0,1,1,1)
            for umesh in umeshes:
                uv = umesh.uv
                bounds.update(crn[uv].uv for crn in utils.calc_visible_uv_corners_iter(umesh))
        if bounds == types.BBox():
            color = (1,0,0,1)
            bounds.xmin = 0.0
            bounds.ymin = 0.0
            bounds.xmax = 1.0
            bounds.ymax = 1.0

        draw_data = bounds.draw_data_lines()


        n_panel_width = next(r.width for r in context.area.regions if r.type == 'UI')
        bounds.scale(1.2)  # Add padding

        space_data = context.area.spaces.active
        sima = types.SpaceImage.get_fields(space_data)

        image_size = [256, 256]
        aspect = [1, 1]
        if space_data.image:
            image_width_, image_height_ = space_data.image.size
            if image_height_:
                aspect = list(space_data.image.display_aspect)
                image_size[:] = image_width_, image_height_
        image_size_without_aspect = image_size.copy()

        image_size[0] *= aspect[0]
        image_size[1] *= aspect[1]

        # adjust offset and zoom
        c_region = types.ARegion.get_fields(context.region)

        size_y = c_region.winrct.height / ((bounds.height + 0.00001) * image_size[1])
        size_x = (c_region.winrct.width - n_panel_width) / ((bounds.width+ 0.00001) * image_size[0])

        zoom = min(size_x, size_y)
        if zoom > 100.0:
            zoom = 100.0

        sima.xof = round((bounds.center_x - 0.5) * image_size[0] + (n_panel_width / zoom) / 2)
        sima.yof = round((bounds.center_y - 0.5) * image_size[1])

        # sima_zoom_set
        old_zoom = sima.zoom
        sima.zoom = zoom
        if zoom < 0.1 or zoom > 4.0:
            w, h = image_size_without_aspect
            w *= zoom
            h *= zoom
            if (w < 4) and (h < 4) and zoom < old_zoom:
                sima.zoom = old_zoom
            elif c_region.winrct.width <= zoom:
                sima.zoom = old_zoom
            elif c_region.winrct.height <= zoom:
                sima.zoom = old_zoom

        from .select import add_draw_rect
        add_draw_rect(draw_data, color)

        context.region.tag_redraw()
        return {'FINISHED'}


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
        if not active_obj or active_obj.type != 'MESH' or not context.selected_objects:
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
    @bpy.app.handlers.persistent
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
        try:
            if univ_settings().uv_layers_show:
                bpy.app.handlers.depsgraph_update_post.append(UNIV_OT_UV_Layers_Manager.univ_uv_layers_update)
        except Exception as e:
            print('Failed to add a handler for UV Layer system.', e)

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

    def move_uv_bm(self, obj, idx, up, with_names=False):
        idx_inc_dec = 1
        layers = obj.data.uv_layers
        if up:
            if idx == 0 or len(layers) == 0 or len(layers) < idx + 1:
                return False
        else:
            idx_inc_dec = -1
            if len(layers) <= idx + 1:
                return False

        other_idx = idx - idx_inc_dec
        if bpy.context.mode == 'EDIT_MESH':
            self._swap_uv_bm(obj, idx, other_idx, with_names)
        else:
            self._swap_uv_mesh(obj, idx, other_idx, with_names)

    @staticmethod
    def _swap_uv_bm(obj, idx, other_idx, with_names, change_active_idx=True):
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        uv = bm.loops.layers.uv.verify()
        layers = mesh.uv_layers

        if layers.active_index != idx:
            layers.active_index = idx

        coords = [crn[uv].uv.copy() for f in bm.faces for crn in f.loops]
        layers.active_index = other_idx
        it = (crn[uv].uv for f in bm.faces for crn in f.loops)

        for uv_a_copy, uv_b in zip(coords, it):
            uv_b_copy = uv_b.copy()
            uv_b[:] = uv_a_copy
            uv_a_copy[:] = uv_b_copy

        layers.active_index = idx

        it = (crn for f in bm.faces for crn in f.loops)
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
    def _swap_uv_mesh(obj, idx, other_idx, with_names, change_active_idx=True):
        size = len(obj.data.loops) * 2
        uvs_a = np.empty(size, dtype='float32')
        uvs_b = np.empty(size, dtype='float32')

        layers = obj.data.uv_layers
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

        for obj in utils.calc_any_unique_obj():
            if self.move_uv_bm(obj, settings.uv_layers_active_idx, up=True, with_names=self.with_names):
                if bpy.context.mode == 'EDIT_MESH':
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return {'FINISHED'}

class UNIV_OT_MoveDown(UNIV_OT_MoveUpDownBase):
    bl_idname = 'mesh.univ_move_down'
    bl_label = 'Down'

    def execute(self, context):
        settings = univ_settings()
        if settings.uv_layers_size == settings.uv_layers_active_idx+1:
            self.report({'WARNING'}, 'Cannot move down')
            return {'CANCELLED'}

        for obj in utils.calc_any_unique_obj():
            if self.move_uv_bm(obj, settings.uv_layers_active_idx, up=False, with_names=self.with_names):
                if bpy.context.mode == 'EDIT_MESH':
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return {'FINISHED'}

class UNIV_OT_Add(Operator):
    bl_idname = 'mesh.univ_add'
    bl_label = 'Add'
    bl_options = {'REGISTER', 'UNDO'}

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
        if not (objects := utils.calc_any_unique_obj()):
            self.report({'WARNING'}, 'Objects not found')
            return {'CANCELLED'}

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
                if UNIV_OT_Join.add_missed_uvs(obj, target_min_size, settings.uv_layers_active_idx):
                    obj.data.update()
        return {'FINISHED'}

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
            uv = bm.loops.layers.uv.new('UVMap')
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
    bl_options = {'REGISTER', 'UNDO'}

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
        if not (objects := utils.calc_any_unique_obj()):
            self.report({'WARNING'}, 'Objects not found')
            return {'CANCELLED'}

        max_size = max(len(obj.data.uv_layers) for obj in objects)

        if max_size == 0:
            self.report({'WARNING'}, 'All uv maps removed')
            return {'CANCELLED'}

        target_idx = univ_settings().uv_layers_active_idx
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

class UNIV_OT_FixUVs(UNIV_OT_Join):
    bl_idname = "mesh.univ_fix_uvs"
    bl_label = "Fix UVs"
    bl_description = "Fix channels"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    # def invoke(self, context, event):
    #     if event.value == 'PRESS':
    #         return self.execute(context)
    #     self.sorted_by_name = event.alt  # TODO: Add sorted by name
    #     return self.execute(context)

    def execute(self, context):
        if not (objects := utils.calc_any_unique_obj()):
            self.report({'WARNING'}, 'Objects not found')
            return {'CANCELLED'}

        removed_extra_channels_counter = sum(self.sanitize_uv(obj.data) for obj in objects)

        uv_names, conflicts_counter = self.sanitize_attr_names_and_get_names(objects)

        max_uv_size = max(len(obj.data.uv_layers) for obj in objects)
        if not max_uv_size:
            if preferences.debug():
                self.report({'WARNING'}, 'UVs not found')
            return {'CANCELLED'}

        added_uvs_counter = 0
        for obj in objects:
            if bpy.context.mode == 'EDIT_MESH':
                if UNIV_OT_Add.add_missed_uvs_bm(obj, max_uv_size, 8):
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            else:
                if res := self.add_missed_uvs(obj, max_uv_size):
                    added_uvs_counter += res

            self.rename_uvs(uv_names, obj.data)

        active_uv_idx = 0
        active_render_uv_idx = 0
        for idx, uv in enumerate(bpy.context.active_object.data.uv_layers):
            if uv.active:
                active_uv_idx = idx
            if uv.active_render:
                active_render_uv_idx = idx

        for obj in objects:
            mesh = obj.data
            uv_layers = mesh.uv_layers
            uv_layers[active_uv_idx].active = True
            uv_layers[active_render_uv_idx].active_render = True
            mesh.update()

        if bpy.context.mode == 'EDIT_MESH':
            if bpy.context.area.type == 'VIEW_3D':
                bpy.ops.mesh.univ_seam_border(selected=False, mtl=False, by_sharps=False)  # noqa
            else:
                bpy.ops.uv.univ_seam_border(selected=False, mtl=False, by_sharps=False)  # noqa

        info = ''
        if conflicts_counter:
            info += f"Resolver {conflicts_counter} names conflicts. "
        if removed_extra_channels_counter:
            info += f"Removed {removed_extra_channels_counter} extra channels in total."
        if added_uvs_counter:
            info += f"Added {added_uvs_counter} channels in total."

        if info:
            self.report({'WARNING'}, info)

        return {'FINISHED'}

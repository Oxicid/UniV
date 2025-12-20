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
from .. import utypes
from ..utypes import UMeshes, BBox, BBox3D
from .. import preferences
from ..preferences import prefs, univ_settings
from mathutils import Vector


class UNIV_OT_Pin(Operator):
    bl_idname = 'uv.univ_pin'
    bl_label = 'Pin'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Set/clear selected UV vertices as anchored between multiple unwrap operations\n" \
        f"With sync mode disabled, Edge mode switches to Vertex since the pins are not visible in edge mode\n\n" \
        f"This button is used to free the 'P' button for the Pack operator"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        from .transform import UNIV_OT_Align_pie
        self.umeshes = UMeshes()
        self.umeshes.update_tag = False

        if context.mode == 'EDIT_MESH':
            selected, visible = self.umeshes.filtered_by_selected_and_visible_uv_verts()
            self.umeshes = selected if selected else visible

            if selected:
                for umesh in self.umeshes:
                    if umesh.elem_mode == 'VERT':
                        umesh.sequence = utils.calc_selected_uv_vert(umesh)
                    elif umesh.elem_mode == 'EDGE':
                        corners = utils.calc_selected_uv_edge_iter(umesh)
                        umesh.sequence = UNIV_OT_Align_pie.get_unique_linked_corners_from_crn_edge(umesh, corners)
                    else:
                        corners = (crn for f in utils.calc_selected_uv_faces_iter(umesh) for crn in f.loops)
                        umesh.sequence = UNIV_OT_Align_pie.get_unique_linked_corners_from_crn_vert(umesh, corners)

                    umesh.update_tag = bool(umesh.sequence)
            else:
                for umesh in self.umeshes:
                    umesh.sequence = utils.calc_visible_uv_corners(umesh)
                    umesh.update_tag = bool(umesh.sequence)
        else:
            for umesh in self.umeshes:
                umesh.sequence = [crn for f in umesh.bm.faces for crn in f.loops]
                umesh.update_tag = bool(umesh.sequence)

        all_pinned = True
        for umesh in self.umeshes:
            uv = umesh.uv
            corners = umesh.sequence
            if not all(crn[uv].pin_uv for crn in corners):
                all_pinned = False
                break


        for umesh in self.umeshes:
            uv = umesh.uv
            corners = umesh.sequence

            if all_pinned:
                if any(crn[uv].pin_uv for crn in corners):
                    umesh.update_tag = True
                    for crn in corners:
                        crn[uv].pin_uv = False
            else:
                if not all(crn[uv].pin_uv for crn in corners):
                    umesh.update_tag = True
                    for crn in corners:
                        crn[uv].pin_uv = True

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
        my_user.size_x = univ_settings().size_x
        my_user.size_y = univ_settings().size_y
        from . import checker
        size_name = checker.UNIV_OT_Checker.resolution_values_to_name(int(my_user.size_x), int(my_user.size_y))
        my_user.name += ' ' + size_name

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
            if active_index > len(mesh.uv_layers) - 1:
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
    bl_description = f"Hide selected or unselected UV"

    unselected: BoolProperty(name='Unselected', default=False)

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
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.fix_context()
        if not self.umeshes.sync:
            # Fix incorrect for hide in non-sync mode
            if utils.get_select_mode_mesh() != 'FACE':
                self.umeshes.sync = True
                self.umeshes._elem_mode = ''  # noqa
                self.umeshes.elem_mode = 'FACE'
                utils.update_area_by_type('VIEW_3D')
            self.umeshes.sync = False

        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_pos:
            return self.pick_hide()

        if utils.USE_GENERIC_UV_SYNC:
            return bpy.ops.uv.hide(unselected=self.unselected)

        # Legacy

        # Unselected
        if self.unselected:
            self.umeshes.umeshes.extend(visible_umeshes.umeshes.copy())
            for umesh in self.umeshes:
                unselected_faces = utils.calc_unselected_uv_faces(umesh)
                if umesh.sync:
                    for f in unselected_faces:
                        f.hide = True
                else:
                    for f in unselected_faces:
                        f.select = False
                umesh.update_tag = bool(unselected_faces)
                if unselected_faces:
                    umesh.bm.select_flush(True)
            self.umeshes.update(info='Not found unselected faces')
            return {'FINISHED'}
        # Selected
        if self.umeshes.sync:
            if self.umeshes.elem_mode == 'FACE':
                return bpy.ops.uv.hide(unselected=False)

            self.umeshes.filter_by_selected_mesh_faces()
            if not self.umeshes:
                # TODO: Implement hide by view box
                return bpy.ops.uv.hide(unselected=False)

            if self.umeshes.elem_mode == 'VERT':
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
                        self.report({'WARNING'},
                                    'Undefined Behavior: Has selected elements even after applying Hide operation')
            return res
        else:
            # bpy.ops.uv.hide sometimes works incorrectly in 'FACE' mode too,
            # maybe it's something to do with not updating bm.select_mode
            for umesh in self.umeshes:
                uv = umesh.uv
                update_tag = False
                for f in utils.calc_visible_uv_faces(umesh):
                    if any(crn[uv].select for crn in f.loops):
                        f.select = False
                        update_tag = True
                umesh.update_tag = update_tag
            return self.umeshes.update()

    def pick_hide(self):
        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)
        all_islands = []
        for umesh in self.umeshes:
            for isl in utypes.AdvIslands.calc_visible_with_mark_seam(umesh):
                if self.unselected:
                    all_islands.append(isl)
                hit.find_nearest_island_by_crn(isl)

        if not hit:
            self.report({'INFO'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        if self.unselected:
            if len(all_islands) == 1:
                self.report({'INFO'}, 'No found unpicked islands for hiding')
                return {'FINISHED'}

            hit.island.tag = False
            hit.island.umesh.update_tag = False
            for isl in all_islands:
                if isl.tag:
                    isl.hide_first()
                    isl.umesh.update_tag = True
            return self.umeshes.update()
        else:
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

        # Set active trim
        self.set_active_trim(mouse_pos)

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
                zoom = utypes.View2D.get_zoom(context.region.view2d)
                divider = 1 / 8 if zoom <= 1600 else 1 / 64
                divider = divider if zoom <= 12800 else 1 / 64 / 8

                grid_pt = Vector(utils.round_threshold(v, divider) for v in mouse_pos)
                if (grid_dist := (grid_pt - mouse_pos).length) < min_dist:
                    pt = grid_pt
                    min_dist = grid_dist

        # TODO: Implement Object Mode
        if context.mode == 'EDIT_MESH' and REPEAT_MOUSE_POS_COUNT == 0:
            umeshes = UMeshes()
            pt, min_dist = self.snap_to_trim(pt, mouse_pos, min_dist)

            zero_pt = Vector((0.0, 0.0))
            for umesh in umeshes:
                uv = umesh.uv
                if prefs().snap_points_default == 'ALL':
                    for f in utils.calc_visible_uv_faces_iter(umesh):
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

                elif umeshes.elem_mode == 'VERT':
                    for f in utils.calc_visible_uv_faces_iter(umesh):
                        for crn in f.loops:
                            uv_co = crn[uv].uv
                            if (length := (mouse_pos - uv_co).length) < min_dist:
                                pt = uv_co
                                min_dist = length
                elif umeshes.elem_mode == 'EDGE':
                    for f in utils.calc_visible_uv_faces_iter(umesh):
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
                    for f in utils.calc_visible_uv_faces_iter(umesh):
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
        if context.space_data.pivot_point != 'CURSOR':
            from . import toggle
            from .. import draw
            toggle.PREV_PIVOT = context.space_data.pivot_point
            context.space_data.pivot_point = 'CURSOR'

            draw.TextDraw.max_draw_time = 1.8
            draw.TextDraw.draw(f"Switch Pivot to 'Cursor'")
        return {'FINISHED'}

    @staticmethod
    def snap_to_trim(pt, mouse_pos, min_dist):
        if not prefs().use_trims:
            return pt, min_dist

        for trim in utils.get_trim_bboxes():
            center = trim.center
            if (dist := (center - mouse_pos).length) < min_dist:
                pt = center
                min_dist = dist

            for crn_pt in trim.draw_data_verts():
                if (dist := (crn_pt - mouse_pos).length) < min_dist:
                    pt = crn_pt
                    min_dist = dist

            for (line_a, line_b) in utils.reshape_to_pair(trim.draw_data_lines()):
                line_center = (line_a + line_b) * 0.5
                if (dist := (line_center - mouse_pos).length) < min_dist:
                    pt = line_center
                    min_dist = dist

        return pt, min_dist


    @staticmethod
    def set_active_trim(mouse_pos):
        idx = -1
        min_dist = float('inf')
        if prefs().use_trims:
            for i, trim in enumerate(preferences.prefs().trims_presets):
                if not trim.visible:
                    continue
                bb = BBox(trim.x, trim.x + trim.width, trim.y, trim.y + trim.height)

                if mouse_pos in bb:
                    for (l_a, l_b) in utils.reshape_to_pair(bb.draw_data_lines()):
                        _, dist = utils.intersect_point_line_segment(mouse_pos, l_a, l_b)
                        if dist < min_dist:
                            min_dist = dist
                            idx = i

            if idx != -1:
                prefs().active_trim_index = idx


class UNIV_OT_Focus(Operator):
    bl_idname = "uv.univ_focus"
    bl_label = 'Focus'

    def invoke(self, context, event):
        if not (context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV'):
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}
        assert context.mode == 'EDIT_MESH'

        bounds = BBox()
        umeshes = UMeshes()
        color = (1, 1, 0, 1)
        for umesh in umeshes:
            uv = umesh.uv
            if umesh.sync and umesh.elem_mode in ('FACE', 'ISLAND'):
                bounds.update(crn[uv].uv for f in utils.calc_selected_uv_faces_iter(umesh) for crn in f.loops)
            else:
                bounds.update(crn[uv].uv for crn in utils.calc_selected_uv_vert_iter(umesh))
        if bounds == BBox():
            color = (0, 1, 1, 1)
            for umesh in umeshes:
                uv = umesh.uv
                bounds.update(crn[uv].uv for crn in utils.calc_visible_uv_corners_iter(umesh))
        if bounds == BBox():
            color = (1, 0, 0, 1)
            bounds.xmin = 0.0
            bounds.ymin = 0.0
            bounds.xmax = 1.0
            bounds.ymax = 1.0

        lines = bounds.draw_data_lines()

        n_panel_width = next(r.width for r in context.area.regions if r.type == 'UI')
        tools_width = next(r.width for r in context.area.regions if r.type == 'TOOLS')
        bounds.scale(1.2)  # Add padding

        space_data = context.area.spaces.active
        sima = utypes.SpaceImage.get_fields(space_data)

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
        c_region = utypes.ARegion(context.region)

        zero_division_avoid = 0.00001
        size_y = c_region.winrct.height / ((bounds.height + zero_division_avoid) * image_size[1])
        size_x = (c_region.winrct.width - n_panel_width - tools_width) / ((bounds.width + zero_division_avoid) * image_size[0])

        zoom = max(min(size_x, size_y), 0.05)
        if zoom > 100.0:
            zoom = 100.0

        offset_x = (n_panel_width - tools_width) / zoom / 2
        sima.xof = round((bounds.center_x - 0.5) * image_size[0] + offset_x)
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

        from .. import draw
        draw.LinesDrawSimple.draw_register(lines, color)

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

        if context.mode == 'EDIT_MESH':
            if not active_obj or active_obj.type != 'MESH':
                if settings.uv_layers_size:
                    settings.uv_layers_size = 0
                    utils.update_univ_panels()
                return
            selected_objects = context.objects_in_mode_unique_data
        else:

            if not active_obj or active_obj.type != 'MESH':
                if settings.uv_layers_size:
                    settings.uv_layers_size = 0
                    utils.update_univ_panels()
                return
            selected_objects = (obj_ for obj_ in context.selected_objects if obj_.type == 'MESH')

        act_obj_uv_layers = active_obj.data.uv_layers

        uv_presets = settings.uv_layers_presets
        UNIV_OT_UV_Layers_Manager.sanitize_size(uv_presets)

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

                    if settings.uv_layers_active_idx >= uv_layers_size - 1:
                        settings.uv_layers_active_idx = uv_layers_size - 1  # clamp
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
                    if settings.uv_layers_active_idx > act_obj_uv_layers_size - 1:
                        settings.uv_layers_active_idx = act_obj_uv_layers_size - 1
                    utils.update_univ_panels()
                    return

            for idx in range(act_obj_uv_layers_size):
                preset = uv_presets[idx]
                preset.name = act_obj_uv_names[idx]
                preset.flag = not act_obj_uv_names_tags[idx]

            settings.uv_layers_size = act_obj_uv_layers_size
            settings.uv_layers_active_idx = act_obj_uv_idx
            settings.uv_layers_active_render_idx = act_obj_uv_render_idx
        elif selected_objects:
            obj_with_max_uv = max(selected_objects, key=lambda ob: len(ob.data.uv_layers))
            uv_layers = obj_with_max_uv.data.uv_layers
            uv_layers_size = len(uv_layers)

            settings.uv_layers_size = uv_layers_size

            for preset, uv in zip(uv_presets, uv_layers):
                preset.name = uv.name
                preset.flag = 2

            settings.uv_layers_active_idx = 0
            settings.uv_layers_active_render_idx = -1
        else:
            if settings.uv_layers_size:
                settings.uv_layers_size = 0
                utils.update_univ_panels()
            return
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
        for update_obj in deps.updates:
            if update_obj.is_updated_transform:
                return
            else:
                from .. import ui
                ui.REDRAW_UV_LAYERS = True
                return

    @staticmethod
    def append_handler_with_delay():
        try:
            if univ_settings().uv_layers_show:
                bpy.app.handlers.depsgraph_update_post.append(UNIV_OT_UV_Layers_Manager.univ_uv_layers_update)
        except Exception as e:
            print('UniV: Failed to add a handler for UV Layer system.', e)


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
        other_uv = bm.loops.layers.uv[other_idx]
        layers = mesh.uv_layers

        coords = [crn[uv].uv.copy() for f in bm.faces for crn in f.loops]
        it = (crn[other_uv].uv for f in bm.faces for crn in f.loops)

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
    bl_description = ("Move Up UV Layer \n"
                      "Alt+Click - Moves only the UV layer, keeping name in place.")

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
    bl_description = ("Move Down UV Layer \n"
                      "Alt+Click - Moves only the UV layer, keeping name in place.")

    def execute(self, context):
        settings = univ_settings()
        if settings.uv_layers_size == settings.uv_layers_active_idx + 1:
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
    bl_description = ("Add UV Layer \n"
                      "Alt+Click - Add missed UV layers.")

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

        target_min_size = min(len(obj.data.uv_layers) for obj in objects) + 1
        if self.add_missed:
            target_max_size = max(len(obj.data.uv_layers) for obj in objects) + 1
            if target_min_size != target_max_size:
                target_min_size = target_max_size - 1

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
        if uv_size >= active_index + 1:
            if mesh.uv_layers.active_index != active_index:
                mesh.uv_layers.active_index = active_index

        coords = [crn[uv].uv.copy() for f in bm.faces for crn in f.loops]

        while (len(bm.loops.layers.uv)) < target_size:
            uv = bm.loops.layers.uv.new('UVMap')
            obj.data.uv_layers.active_index = len(mesh.uv_layers) - 1
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
    bl_description = ("Remove UV Layer \n"
                      "Alt+Click - Remove all UV layers.")

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
                if len(uv_layers) >= target_idx + 1:
                    if bpy.context.mode == 'EDIT_MESH':
                        bm = bmesh.from_edit_mesh(mesh)
                        bm.loops.layers.uv.remove(bm.loops.layers.uv[target_idx])
                        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                    else:
                        uv_layers.remove(uv_layers[target_idx])
                        obj.data.update()

        return {'FINISHED'}


class UNIV_OT_CopyToLayer(Operator):
    bl_idname = 'uv.univ_copy_to_layer'
    bl_label = 'Copy'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        umeshes = UMeshes(report=self.report)

        copy_from = int(univ_settings().copy_to_layers_from)
        copy_to = int(univ_settings().copy_to_layers_to)

        if copy_from != 0:
            if copy_from == copy_to:
                self.report({'WARNING'}, 'The From and To indexes are identical')
                return {'CANCELLED'}

        source_and_target_same_count = 0
        missed_source_meshes_count = 0
        missed_target_meshes_count = 0
        if umeshes.is_edit_mode:
            if context.area.type != 'IMAGE_EDITOR':
                umeshes.set_sync()
                umeshes.sync_invalidate()

            selected, visible = umeshes.filtered_by_selected_and_visible_uv_edges()
            umeshes = selected if selected else visible
            if not umeshes:
                return umeshes.update()

            for umesh in reversed(umeshes):
                if len(umesh.obj.data.uv_layers) == 1:
                    umeshes.umeshes.remove(umesh)
            if not umeshes:
                return umeshes.update(info='Not found meshes with 2 and more uvs for copy coordinates')

            for umesh in reversed(umeshes):
                if copy_from == 0:
                    donor_uv = umesh.uv
                else:
                    if len(umesh.obj.data.uv_layers) < copy_from:
                        umeshes.umeshes.remove(umesh)
                        missed_source_meshes_count += 1
                        continue
                    donor_uv = umesh.bm.loops.layers.uv[copy_from-1]

                faces = utils.calc_selected_uv_faces(umesh) if selected else utils.calc_visible_uv_faces(umesh)
                if copy_to == 0:
                    for idx in range(len(umesh.obj.data.uv_layers)):
                        recipient_uv = umesh.bm.loops.layers.uv[idx]
                        if donor_uv.name == recipient_uv.name:
                            continue

                        for f in faces:
                            for crn in f.loops:
                                crn[recipient_uv].uv = crn[donor_uv].uv
                else:
                    if len(umesh.obj.data.uv_layers) < copy_to:
                        umeshes.umeshes.remove(umesh)
                        missed_target_meshes_count += 1
                        continue

                    recipient_uv = umesh.bm.loops.layers.uv[copy_to-1]
                    if recipient_uv.name == donor_uv.name:
                        umeshes.umeshes.remove(umesh)
                        source_and_target_same_count += 1
                        continue

                    for f in faces:
                        for crn in f.loops:
                            crn[recipient_uv].uv = crn[donor_uv].uv

        else:
            if not umeshes:
                return umeshes.update()

            for umesh in reversed(umeshes):
                if len(umesh.obj.data.uv_layers) == 1:
                    umeshes.umeshes.remove(umesh)
            if not umeshes:
                return umeshes.update(info='Not found meshes with 2 and more uvs for copy coordinates')

            for umesh in reversed(umeshes):
                if copy_from == 0:
                    donor_uv = umesh.obj.data.uv_layers.active
                else:
                    if len(umesh.obj.data.uv_layers) < copy_from:
                        umeshes.umeshes.remove(umesh)
                        missed_source_meshes_count += 1
                        continue
                    donor_uv = umesh.obj.data.uv_layers[copy_from - 1]

                size = len(umesh.obj.data.loops) * 2
                donor_uv_coords = np.empty(size, dtype='float32')
                donor_uv.data.foreach_get("uv", donor_uv_coords)

                if copy_to == 0:
                    for idx in range(len(umesh.obj.data.uv_layers)):
                        recipient_uv = umesh.obj.data.uv_layers[idx]
                        if donor_uv.name == recipient_uv.name:
                            continue
                        recipient_uv.data.foreach_set("uv", donor_uv_coords)
                else:
                    if len(umesh.obj.data.uv_layers) < copy_to:
                        umeshes.umeshes.remove(umesh)
                        missed_target_meshes_count += 1
                        continue

                    recipient_uv = umesh.obj.data.uv_layers[copy_to - 1]
                    if recipient_uv.name == donor_uv.name:
                        umeshes.umeshes.remove(umesh)
                        source_and_target_same_count += 1
                        continue
                    recipient_uv.data.foreach_set("uv", donor_uv_coords)

        info = ''
        if source_and_target_same_count:
            info += f'{source_and_target_same_count} meshes has same source and target UV channel. '
        if missed_source_meshes_count:
            info += f'{missed_source_meshes_count} meshes do not have the required source UV channel to provide coordinates for transfer. '
        if missed_target_meshes_count:
            info += f'{missed_target_meshes_count} meshes do not have the required target UV channel to receive source coordinates.'
        if info:
            self.report({'WARNING'}, info)

        for umesh in umeshes:
            umesh.obj.update_tag()
        return {'FINISHED'}


class UNIV_OT_SetActiveRender(Operator):
    bl_idname = 'mesh.univ_active_render_set'
    bl_label = 'Remove'
    bl_options = {'REGISTER', 'UNDO'}

    idx: IntProperty(name='Set Active', default=0, min=0, max=8, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        objects = utils.calc_any_unique_obj()

        for obj in objects:
            mesh = obj.data
            uv_layers = mesh.uv_layers
            if len(uv_layers) >= self.idx + 1:
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
    #
    #     self.lock_overlap = event.shift
    #     return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

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
            info += f"Resolver {conflicts_counter} uv names conflicts. "
        if removed_extra_channels_counter:
            info += f"Removed {removed_extra_channels_counter} extra channels in total."
        if added_uvs_counter:
            info += f"Added {added_uvs_counter} channels in total."

        if info:
            self.report({'WARNING'}, info)

        return {'FINISHED'}


class UNIV_OT_Flatten(Operator):
    bl_idname = 'mesh.univ_flatten'
    bl_label = 'Flatten'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Convert 3d coords to 2D from uv map\n\n" \
                     "Context keymaps on button:\n" \
                     "\t\tDefault - Mesh\n" \
                     "\t\tShift - Shape Keys\n" \
                     "\t\tAlt - Modifier" \

    axis: EnumProperty(name="Axis", default='z', items=(
        ('z', 'Bottom', ''),
        ('y', 'Side', ''),
        ('x', 'Front', '')))

    flatten_type: EnumProperty(name="Flatten Type", default='MESH', items=(
                                    ('MESH', 'Mesh', ''),
                                    ('SHAPE_KEY', 'Shape Key', ''),
                                    ('MODIFIER', 'Modifier', '')))
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)
    mix_factor: FloatProperty(name='Mix Factor', default=1, min=0, max=1)
    weld_distance: FloatProperty(name='Weld Distance', default=0.00001, min=0)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.alt:
            self.flatten_type = 'MODIFIER'
        elif event.shift:
            self.flatten_type = 'SHAPE_KEY'
        else:
            self.flatten_type = 'MESH'

        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'axis', expand=True)
        layout.prop(self, 'use_correct_aspect')
        if self.flatten_type == 'MODIFIER':
            layout.prop(self, 'mix_factor')
            layout.prop(self, 'weld_distance')
        layout.row(align=True).prop(self, 'flatten_type', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.fix_context()
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        if not self.umeshes:
            return self.umeshes.update()
        if self.use_correct_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=True)

        if self.umeshes.is_edit_mode:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes
            if not self.umeshes:
                return self.umeshes.update(info='Not found faces for manipulate')

            if self.apply_gn():
                return {'FINISHED'}

            if selected_umeshes:
                for umesh in self.umeshes:
                    uv = umesh.uv
                    split_edges = set()
                    for f in (selected_faces := utils.calc_selected_uv_faces(umesh)):
                        for crn in f.loops:
                            if not crn.link_loop_radial_prev.face.select or utils.is_boundary_sync(crn, uv):
                                split_edges.add(crn.edge)
                    if split_edges:
                        bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                        for e in split_edges:
                            e.select = True
                        umesh.bm.select_flush(True)
                    if self.flatten_type == 'SHAPE_KEY':
                        continue
                    self.apply_coords(selected_faces, umesh)
                if self.flatten_type == 'SHAPE_KEY':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    for umesh in self.umeshes:
                        self.apply_shape_keys((f for f in umesh.obj.data.polygons if f.select), umesh)
                        umesh.obj.data.update_tag()
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                    return {'FINISHED'}

            else:
                for umesh in self.umeshes:
                    uv = umesh.uv
                    split_edges = set()
                    for f in (visible_faces := utils.calc_visible_uv_faces(umesh)):
                        for crn in f.loops:
                            if utils.is_boundary_sync(crn, uv):
                                split_edges.add(crn.edge)
                    bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                    if self.flatten_type == 'SHAPE_KEY':
                        continue
                    self.apply_coords(visible_faces, umesh)

                if self.flatten_type == 'SHAPE_KEY':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    for umesh in self.umeshes:
                        self.apply_shape_keys((f for f in umesh.obj.data.polygons if not f.hide), umesh)
                        umesh.obj.data.update_tag()
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        else:
            if self.apply_gn():
                return {'FINISHED'}

            for umesh in self.umeshes:
                uv = umesh.uv
                split_edges = set()
                for f in umesh.bm.faces:
                    for crn in f.loops:
                        if not utils.is_pair(crn, crn.link_loop_radial_prev, uv):
                            split_edges.add(crn.edge)
                if split_edges:
                    bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                    umesh.update()

                if self.flatten_type == 'SHAPE_KEY':
                    umesh.free()
                    self.apply_shape_keys((f for f in umesh.obj.data.polygons if f.select), umesh)
                    umesh.obj.data.update_tag()
                else:
                    self.apply_coords(umesh.bm.faces, umesh)
            if self.flatten_type == 'SHAPE_KEY':
                return {'FINISHED'}

        self.umeshes.silent_update()
        self.umeshes.free()
        return {'FINISHED'}

    def apply_coords(self, faces, umesh):
        uv = umesh.uv
        bb3d = BBox3D.get_from_umesh(umesh)
        bb = bb3d.to_bbox_2d(self.axis)
        max_length = bb.max_length
        if umesh.aspect != 1:
            max_length = self.aspect_to_scale(umesh.aspect).to_2d() * max_length
        delta = Vector((-0.5, -0.5))

        for f in faces:
            for crn in f.loops:
                if self.axis == 'z':
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d()
                elif self.axis == 'y':
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d().zxy
                else:
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d().xzy

        umesh.bm.normal_update()

    def apply_shape_keys(self, faces, umesh: utypes.UMesh):
        if not umesh.obj.data.shape_keys:
            bb3d = BBox3D.get_from_umesh(umesh)
        else:
            base_sk_data = umesh.obj.data.shape_keys.key_blocks[0].data
            coords = (base_sk_data[i].co for i in range(len(umesh.obj.data.vertices)))
            bb3d = BBox3D.calc_bbox(coords)

        bb = bb3d.to_bbox_2d(self.axis)
        max_length = bb.max_length
        if umesh.aspect != 1:
            max_length = self.aspect_to_scale(umesh.aspect).to_2d() * max_length
        delta = Vector((-0.5, -0.5))

        uv_data = umesh.obj.data.uv_layers.active.data

        if not umesh.obj.data.shape_keys:
            umesh.obj.shape_key_add(name="model", from_mix=True)
            sk = umesh.obj.shape_key_add(name="uv", from_mix=True)
        else:
            if not (sk := umesh.obj.data.shape_keys.key_blocks.get('uv')):
                sk = umesh.obj.shape_key_add(name="uv", from_mix=True)

        idx = 0
        for idx, sk_ in enumerate(umesh.obj.data.shape_keys.key_blocks):
            if umesh.obj.data.shape_keys.key_blocks[idx] == sk:
                break

        umesh.obj.active_shape_key_index = idx
        umesh.obj.active_shape_key.value = 1

        corners = umesh.obj.data.loops
        sk_data = sk.data
        for poly in faces:
            for loop_index in poly.loop_indices:
                uv_co = uv_data[loop_index].uv
                vert_index = corners[loop_index].vertex_index

                if self.axis == 'z':
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d()
                elif self.axis == 'y':
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d().zxy
                else:
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d().xzy

    def apply_gn(self):
        if self.flatten_type == 'MODIFIER':
            if bpy.app.version < (4, 1, 0):
                self.report({'WARNING'}, 'Modifier types is not supported in Blender versions below 4.1')
                return True
            node_group = self.get_flatten_node_group()
            self.create_gn_flatter_modifier(node_group)
            utils.update_area_by_type('VIEW_3D')
            return True

    def create_gn_flatter_modifier(self, node_group):
        axis = {'z': 2, 'y': 3, 'x': 4}
        for umesh in self.umeshes:
            has_checker_modifier = False

            for m in umesh.obj.modifiers:
                if not isinstance(m, bpy.types.NodesModifier):
                    continue
                if m.name.startswith('UniV Flatten'):
                    has_checker_modifier = True
                    if m.node_group != node_group:
                        m.node_group = node_group

                    m['Socket_2'] = umesh.uv.name
                    m['Socket_3'] = axis[self.axis]
                    m['Socket_4'] = self.aspect_to_scale(umesh.aspect)
                    m['Socket_5'] = self.weld_distance
                    m['Socket_6'] = self.mix_factor
                    umesh.obj.update_tag()
                    break

            if not has_checker_modifier:
                m = umesh.obj.modifiers.new(name='UniV Flatten', type='NODES')
                m.node_group = node_group
                m['Socket_2'] = umesh.uv.name
                m['Socket_3'] = axis[self.axis]
                m['Socket_4'] = self.aspect_to_scale(umesh.aspect)
                m['Socket_5'] = self.weld_distance
                m['Socket_6'] = self.mix_factor

    def get_flatten_node_group(self):
        """Get exist flatten node group"""
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Flatten'):
                if self.flatten_node_group_is_changed(ng):
                    if ng.users == 0:
                        bpy.data.node_groups.remove(ng)
                else:
                    return ng
        return self._create_flatten_node_group()

    @staticmethod
    def flatten_node_group_is_changed(ng):
        items = ng.interface.items_tree
        if len(items) != 7:
            return True
        expect_types = (
            'Geometry',
            'Geometry',
            'String',
            'Menu',
            'Vector',
            'Float',
            'Float'
        )
        for str_typ, item in zip(expect_types, items):
            bpy_type = (getattr(bpy.types, 'NodeTreeInterfaceSocket' + str_typ))
            if not isinstance(item.rna_type, bpy_type):
                return True
        return False

    @staticmethod
    def _create_flatten_node_group():
        bb = bpy.data.node_groups.new(type='GeometryNodeTree', name="UniV Flatten")

        # Interface
        # Socket Geometry
        geometry_socket = bb.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        geometry_socket.attribute_domain = 'POINT'

        # Socket Geometry
        geometry_socket_1 = bb.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        geometry_socket_1.attribute_domain = 'POINT'

        # Socket UV Map
        uv_map_socket = bb.interface.new_socket(name="UV Map", in_out='INPUT', socket_type='NodeSocketString')
        uv_map_socket.default_value = "UVMap"
        uv_map_socket.subtype = 'NONE'
        uv_map_socket.attribute_domain = 'POINT'

        # Socket Axis
        axis_socket = bb.interface.new_socket(name="Axis", in_out='INPUT', socket_type='NodeSocketMenu')
        axis_socket.attribute_domain = 'POINT'
        axis_socket.force_non_field = True

        # Socket Aspect Ratio
        aspect_ratio_socket = bb.interface.new_socket(
            name="Aspect Ratio", in_out='INPUT', socket_type='NodeSocketVector')
        aspect_ratio_socket.default_value = (1.0, 1.0, 0.0)
        aspect_ratio_socket.min_value = 0.01
        aspect_ratio_socket.max_value = 10000
        aspect_ratio_socket.default_attribute_name = "Aspect Ratio"
        aspect_ratio_socket.force_non_field = True

        # Socket Distance
        distance_socket = bb.interface.new_socket(name="Distance", in_out='INPUT', socket_type='NodeSocketFloat')
        distance_socket.default_value = 0.00001
        distance_socket.min_value = 0.0
        distance_socket.subtype = 'DISTANCE'
        distance_socket.attribute_domain = 'POINT'
        distance_socket.force_non_field = True

        # Socket Factor
        factor_socket = bb.interface.new_socket(name="Factor", in_out='INPUT', socket_type='NodeSocketFloat')
        factor_socket.default_value = 1.0
        factor_socket.min_value = 0.0
        factor_socket.max_value = 1.0
        factor_socket.subtype = 'FACTOR'
        factor_socket.attribute_domain = 'POINT'
        factor_socket.force_non_field = True

        # initialize UniV Flatten nodes
        # node Group Input
        group_input = bb.nodes.new("NodeGroupInput")

        # node Group Output
        group_output = bb.nodes.new("NodeGroupOutput")
        group_output.is_active_output = True

        # node Split Edges
        split_edges = bb.nodes.new("GeometryNodeSplitEdges")
        split_edges.name = "Split Edges"

        # node Set Position
        set_position = bb.nodes.new("GeometryNodeSetPosition")
        set_position.name = "Set Position"

        # node Named Attribute
        named_attribute = bb.nodes.new("GeometryNodeInputNamedAttribute")
        named_attribute.label = "UV Attribute"
        named_attribute.data_type = 'FLOAT_VECTOR'

        # node Bounding Box
        bounding_box = bb.nodes.new("GeometryNodeBoundBox")

        # node Vector Math
        vector_math = bb.nodes.new("ShaderNodeVectorMath")
        vector_math.name = "Vector Math"
        vector_math.operation = 'SUBTRACT'

        # node Separate XYZ
        separate_xyz_widths = bb.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz_widths.label = "BBox Widths"

        # Remap
        remap_to_center = bb.nodes.new("ShaderNodeVectorMath")
        remap_to_center.label = "Remap to centered range"
        remap_to_center.operation = 'SUBTRACT'
        remap_to_center.inputs[1].default_value = (0.5, 0.5, 0.0)

        # Scale UV
        max_length_1 = bb.nodes.new("ShaderNodeMath")
        max_length_1.label = "Max Length"
        max_length_1.operation = 'MAXIMUM'

        max_length_2 = bb.nodes.new("ShaderNodeMath")
        max_length_2.label = "Max Length"
        max_length_2.operation = 'MAXIMUM'

        max_length_3 = bb.nodes.new("ShaderNodeMath")
        max_length_3.label = "Max Length"
        max_length_3.operation = 'MAXIMUM'

        scale_uv = bb.nodes.new("ShaderNodeVectorMath")
        scale_uv.label = "Scale UV"
        scale_uv.operation = 'MULTIPLY'

        # node Menu Switch
        menu_switch = bb.nodes.new("GeometryNodeMenuSwitch")
        menu_switch.label = "Axis Index Menu"
        menu_switch.data_type = 'INT'
        menu_switch.enum_items.clear()

        for idx, item in enumerate(['Bottom', 'Side', 'Front']):
            menu_switch.enum_items.new(item)
            menu_switch.inputs[idx + 1].default_value = idx

        # node Index Switch
        index_switch = bb.nodes.new("GeometryNodeIndexSwitch")
        index_switch.label = "Get scale from bbox"
        index_switch.name = "Index Switch"
        index_switch.data_type = 'FLOAT'
        index_switch.index_switch_items.clear()
        for _ in range(3):
            index_switch.index_switch_items.new()

        # node Index Switch
        index_switch_001 = bb.nodes.new("GeometryNodeIndexSwitch")
        index_switch_001.label = "Switch scale by axis"
        index_switch_001.data_type = 'VECTOR'
        index_switch_001.index_switch_items.clear()
        for _ in range(3):
            index_switch_001.index_switch_items.new()

        # node Side swizzle separate
        separate_xyz_001 = bb.nodes.new("ShaderNodeSeparateXYZ")

        # node Side swizzle combine
        combine_xyz = bb.nodes.new("ShaderNodeCombineXYZ")

        # node Front swizzle separate
        separate_xyz_002 = bb.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz_002.label = "Front swizzle separate"

        # node Front swizzle combine
        combine_xyz_001 = bb.nodes.new("ShaderNodeCombineXYZ")
        combine_xyz_001.label = "Front swizzle combine"

        # node Merge by Distance
        merge_by_distance = bb.nodes.new("GeometryNodeMergeByDistance")
        merge_by_distance.name = "Merge by Distance"

        is_inputs = False  # TODO: use version check instead
        try:
            merge_by_distance.mode = 'ALL'
        except:  # noqa
            is_inputs = True
            merge_by_distance.inputs[2].default_value = 'All'

        # node Aspect Ratio Scale
        vector_math_004 = bb.nodes.new("ShaderNodeVectorMath")
        vector_math_004.label = "Aspect Ratio Scale"
        vector_math_004.operation = 'MULTIPLY'

        # node Factor
        mix_factor = bb.nodes.new("ShaderNodeMix")
        mix_factor.label = "Factor"
        mix_factor.blend_type = 'MIX'
        mix_factor.data_type = 'VECTOR'
        mix_factor.factor_mode = 'UNIFORM'

        # node Position
        position = bb.nodes.new("GeometryNodeInputPosition")
        position.name = "Position"

        # node No UVMap case
        switch = bb.nodes.new("GeometryNodeSwitch")
        switch.label = "No UVMap case"
        switch.input_type = 'FLOAT'
        switch.inputs[1].default_value = 0.0

        # Set locations
        group_input.location = (-1465, -10)
        group_output.location = (1600, 30)
        split_edges.location = (685, 1)
        set_position.location = (1138, 28)
        named_attribute.location = (-782, -155)
        bounding_box.location = (-1170, -592)
        vector_math.location = (-978, -587)
        separate_xyz_widths.location = (-785, -590)
        max_length_1.location = (-585, -395)
        remap_to_center.location = (-585, -145)
        scale_uv.location = (42, -192)
        menu_switch.location = (-1170, -380)
        index_switch.location = (-360, -460)
        max_length_2.location = (-586, -560)
        index_switch_001.location = (685, -120)
        separate_xyz_001.location = (268, -250)
        combine_xyz.location = (460, -250)
        max_length_3.location = (-590, -730)
        separate_xyz_002.location = (268, -400)
        combine_xyz_001.location = (460, -400)
        merge_by_distance.location = (1315, 18)
        vector_math_004.location = (-165, -460)
        mix_factor.location = (920, -40)
        position.location = (685, -335)
        switch.location = (680, 160)

        # initialize bb links
        # group_input.Geometry -> split_edges.Mesh
        bb.links.new(group_input.outputs[0], split_edges.inputs[0])
        # split_edges.Mesh -> set_position.Geometry
        bb.links.new(split_edges.outputs[0], set_position.inputs[0])
        # group_input.Geometry -> bounding_box.Geometry
        bb.links.new(group_input.outputs[0], bounding_box.inputs[0])

        # named_attribute.Attribute -> vector_math_001.Vector
        bb.links.new(named_attribute.outputs[0], remap_to_center.inputs[0])
        # vector_math_001.Vector -> vector_math_002.Vector
        bb.links.new(remap_to_center.outputs[0], scale_uv.inputs[0])

        # vector_math.Vector -> separate_xyz.Vector
        bb.links.new(vector_math.outputs[0], separate_xyz_widths.inputs[0])
        # separate_xyz.X -> math.Value
        bb.links.new(separate_xyz_widths.outputs[0], max_length_1.inputs[0])
        # separate_xyz.Y -> math.Value
        bb.links.new(separate_xyz_widths.outputs[1], max_length_1.inputs[1])
        # math.Value -> index_switch.0
        bb.links.new(max_length_1.outputs[0], index_switch.inputs[1])
        # separate_xyz.Y -> math_001.Value
        bb.links.new(separate_xyz_widths.outputs[1], max_length_2.inputs[0])
        # separate_xyz.Z -> math_001.Value
        bb.links.new(separate_xyz_widths.outputs[2], max_length_2.inputs[1])
        # math_001.Value -> index_switch.1
        bb.links.new(max_length_2.outputs[0], index_switch.inputs[2])
        # separate_xyz.X -> math_002.Value
        bb.links.new(separate_xyz_widths.outputs[0], max_length_3.inputs[0])
        # separate_xyz.Z -> math_002.Value
        bb.links.new(separate_xyz_widths.outputs[2], max_length_3.inputs[1])
        # math_002.Value -> index_switch.2
        bb.links.new(max_length_3.outputs[0], index_switch.inputs[3])
        # vector_math_002.Vector -> index_switch_001.0
        bb.links.new(scale_uv.outputs[0], index_switch_001.inputs[1])
        # vector_math_002.Vector -> separate_xyz_001.Vector
        bb.links.new(scale_uv.outputs[0], separate_xyz_001.inputs[0])
        # vector_math_002.Vector -> separate_xyz_002.Vector
        bb.links.new(scale_uv.outputs[0], separate_xyz_002.inputs[0])

        # bounding_box.Max -> vector_math.Vector
        bb.links.new(bounding_box.outputs[2], vector_math.inputs[0])
        # bounding_box.Min -> vector_math.Vector
        bb.links.new(bounding_box.outputs[1], vector_math.inputs[1])
        # group_input.UV Map -> named_attribute.Name
        bb.links.new(group_input.outputs[1], named_attribute.inputs[0])
        # merge_by_distance.Geometry -> group_output.Geometry
        bb.links.new(merge_by_distance.outputs[0], group_output.inputs[0])
        # group_input.Axis -> menu_switch.Menu
        bb.links.new(group_input.outputs[2], menu_switch.inputs[0])
        # menu_switch.Output -> index_switch.Index
        bb.links.new(menu_switch.outputs[0], index_switch.inputs[0])
        # separate_xyz_001.Z -> combine_xyz.X
        bb.links.new(separate_xyz_001.outputs[2], combine_xyz.inputs[0])
        # separate_xyz_001.X -> combine_xyz.Y
        bb.links.new(separate_xyz_001.outputs[0], combine_xyz.inputs[1])
        # separate_xyz_001.Y -> combine_xyz.Z
        bb.links.new(separate_xyz_001.outputs[1], combine_xyz.inputs[2])
        # combine_xyz.Vector -> index_switch_001.1
        bb.links.new(combine_xyz.outputs[0], index_switch_001.inputs[2])
        # menu_switch.Output -> index_switch_001.Index
        bb.links.new(menu_switch.outputs[0], index_switch_001.inputs[0])
        # separate_xyz_002.X -> combine_xyz_001.X
        bb.links.new(separate_xyz_002.outputs[0], combine_xyz_001.inputs[0])
        # separate_xyz_002.Z -> combine_xyz_001.Y
        bb.links.new(separate_xyz_002.outputs[2], combine_xyz_001.inputs[1])
        # separate_xyz_002.Y -> combine_xyz_001.Z
        bb.links.new(separate_xyz_002.outputs[1], combine_xyz_001.inputs[2])
        # combine_xyz_001.Vector -> index_switch_001.2
        bb.links.new(combine_xyz_001.outputs[0], index_switch_001.inputs[3])
        # set_position.Geometry -> merge_by_distance.Geometry
        bb.links.new(set_position.outputs[0], merge_by_distance.inputs[0])
        # group_input.Distance -> merge_by_distance.Distance
        if is_inputs:
            bb.links.new(group_input.outputs[4], merge_by_distance.inputs[3])
        else:
            bb.links.new(group_input.outputs[4], merge_by_distance.inputs[2])
        # index_switch.Output -> vector_math_004.Vector
        bb.links.new(index_switch.outputs[0], vector_math_004.inputs[0])
        # vector_math_004.Vector -> vector_math_002.Vector
        bb.links.new(vector_math_004.outputs[0], scale_uv.inputs[1])
        # group_input.Aspect Ratio -> vector_math_004.Vector
        bb.links.new(group_input.outputs[3], vector_math_004.inputs[1])
        # index_switch_001.Output -> mix.B
        bb.links.new(index_switch_001.outputs[0], mix_factor.inputs[5])
        # position.Position -> mix.A
        bb.links.new(position.outputs[0], mix_factor.inputs[4])
        # mix.Result -> set_position.Position
        bb.links.new(mix_factor.outputs[1], set_position.inputs[2])
        # named_attribute.Exists -> switch.Switch
        bb.links.new(named_attribute.outputs[1], switch.inputs[0])
        # group_input.Factor -> switch.True
        bb.links.new(group_input.outputs[5], switch.inputs[2])
        # switch.Output -> mix.Factor
        bb.links.new(switch.outputs[0], mix_factor.inputs[0])
        return bb

    @staticmethod
    def aspect_to_scale(aspect_y):
        if aspect_y > 1:
            return Vector((aspect_y, 1, 0))
        else:
            return Vector((1, 1/aspect_y, 0))


class UNIV_OT_FlattenCleanup(Operator):
    bl_idname = 'mesh.univ_flatten_clean_up'
    bl_label = 'Flatten'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Remove Flatten modifiers and shape keys and unused nodes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)
        self.umeshes.fix_context()
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        removed_sk_counter = 0
        removed_modifiers = 0
        removed_geometry_nodes = 0
        if self.umeshes:
            has_shape_keys = self.has_flatten_shape_keys()
            if has_shape_keys:
                if self.umeshes.is_edit_mode:
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

            for umesh in self.umeshes:
                # Remove shape keys
                if umesh.obj.data.shape_keys and umesh.obj.data.shape_keys.key_blocks.get('uv'):
                    if len(umesh.obj.data.shape_keys.key_blocks) == 2:
                        umesh.obj.shape_key_clear()
                        removed_sk_counter += 1
                    else:
                        sk = umesh.obj.data.shape_keys.key_blocks.get('uv')
                        umesh.obj.shape_key_remove(sk)
                        removed_sk_counter += 1

                # Remove modifiers
                for m in reversed(umesh.obj.modifiers):
                    if isinstance(m, bpy.types.NodesModifier) and m.name.startswith('UniV Flatten'):
                        umesh.obj.modifiers.remove(m)
                        removed_modifiers += 1

                umesh.obj.data.update_tag()
                if self.umeshes.is_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

        # Remove geometry nodes
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Flatten'):
                if ng.users == 0:
                    bpy.data.node_groups.remove(ng)
                    removed_geometry_nodes += 1

        info = ''
        if removed_sk_counter:
            info += f"Removed {removed_sk_counter} shape keys. "
        if removed_modifiers:
            info += f"Removed {removed_modifiers} modifiers. "
        if not removed_modifiers and removed_geometry_nodes:
            info += f"Removed {removed_geometry_nodes} node groups."

        if info:
            self.report({'INFO'}, info)

        return {'FINISHED'}

    def has_flatten_shape_keys(self):
        for umesh in self.umeshes:
            if umesh.obj.data.shape_keys:
                if umesh.obj.data.shape_keys.key_blocks.get('uv'):
                    return True
        return False

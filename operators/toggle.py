# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import traceback

from .. import info
from .. import utils
from ..preferences import force_debug, prefs, stable
from ..types import ARegion

from collections import defaultdict
from bpy.props import *
from bpy.types import Operator

AREA: bpy.types.Area | None = None
AREA_TICKS: bool = False

class UNIV_OT_SplitUVToggle(Operator):
    bl_idname = 'wm.univ_split_uv_toggle'
    bl_label = 'Split UV Toggle'

    mode: EnumProperty(name='Mode',
                       default='SPLIT',
                       items=(
                           ('SPLIT', 'Split', ''),
                           ('SWITCH', 'Switch', ''),
                           ('NEW_WINDOWS', 'New Windows', ''),
                           ('SWAP', 'Swap', '')
                       ))

    def invoke(self, context, event):
        if event.value == 'PRESS':
            self.mode = 'SPLIT'
            return self.execute(context)

        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'SPLIT'
            case True, False, False:
                self.mode = 'NEW_WINDOWS'
            case False, True, False:
                self.mode = 'SWITCH'
            case False, False, True:
                self.mode = 'SWAP'
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement.")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        if context.area is None or context.area.type not in ('VIEW_3D', 'IMAGE_EDITOR'):
            self.report({'WARNING'}, f"Active area must be a 3D Viewport, UV Editor or Image Editor")
            return {'CANCELLED'}
        if context.window.screen.show_fullscreen is True and self.mode != 'SWITCH':
            if self.mode == 'SPLIT':
                self.report({'INFO'}, "You can't split in full screen mode. Switch Mode is used instead.")
            elif self.mode == 'NEW_WINDOWS':
                self.report({'INFO'}, "You cannot create new windows in Full Screen Mode. Switch Mode is used instead.")
            elif self.mode == 'SWAP':
                self.report({'INFO'}, "You can't swap in full screen mode. Switch Mode is used instead.")
            return self.switch_toggle(context)

        if self.mode == 'SPLIT':
            return self.split_toggle(context)
        elif self.mode == 'SWITCH':
            return self.switch_toggle(context)
        elif self.mode == 'NEW_WINDOWS':
            return self.create_new_windows(context)
        elif self.mode == 'SWAP':
            return self.swap_toggle(context)
        self.report({'WARNING'}, f"Mode: {self.mode} not implemented")
        return {'CANCELLED'}

    def split_toggle(self, context):
        active_area = context.area
        if active_area.type == 'VIEW_3D':
            # Close ui_type
            for area in context.screen.areas:
                if area.ui_type == 'UV':
                    if context.area.height == area.height:
                        with context.temp_override(area=area):
                            bpy.ops.screen.area_close()

                        return {'FINISHED'}

            # Change area_type (if area != ui_type) from 3D VIEW
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    if context.area.height == area.height:
                        area.ui_type = 'UV'

                        image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                        if hasattr(image_editor, 'show_gizmo_navigate'):
                            image_editor.show_gizmo_navigate = False

                        area.tag_redraw()
                        c_region = ARegion.get_fields(ARegion.get_n_panel_from_area(area))
                        c_region.info()

                        self.category_setter_register(area)
                        return {'FINISHED'}

            # If there are two 'VIEW_3D' windows, it switches to IMAGE_EDITOR.
            for area in context.screen.areas:
                if (area != active_area) and (area.type == 'VIEW_3D') and (active_area.height == area.height):
                    active_area.type = 'IMAGE_EDITOR'
                    active_area.ui_type = 'UV'

                    image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                    if hasattr(image_editor, 'show_gizmo_navigate'):
                        image_editor.show_gizmo_navigate = False

                    self.category_setter_register(active_area)
                    return {'FINISHED'}

            # Split VIEW_3D
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)

            for area in context.screen.areas:
                if area.ui_type == 'VIEW_3D' and area != active_area:
                    if area.height == area.height:
                        area.ui_type = 'UV'

                        image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                        if hasattr(image_editor, 'show_gizmo_navigate'):
                            image_editor.show_gizmo_navigate = False

                        self.category_setter_register(area)
                        return {'FINISHED'}

        # Change ui_type to ShaderNodeTree from area_type
        elif active_area.type == 'IMAGE_EDITOR' and active_area.ui_type != 'UV':
            c_region = ARegion.get_fields(ARegion.get_n_panel_from_area(active_area))
            active_area.ui_type = 'UV'
            if not c_region.alignment:
                bpy.ops.wm.context_toggle(data_path='space_data.show_region_ui')
            self.category_setter_register(active_area)
            return {'FINISHED'}

        elif active_area.type == 'IMAGE_EDITOR':
            # If there are two 'IMAGE_EDITOR' windows, it switches to VIEW_3D.
            old_areas = set(context.screen.areas[:])
            for area in old_areas:
                if (area != active_area) and (area.type == 'IMAGE_EDITOR') and (active_area.height == area.height):
                    active_area.type = 'VIEW_3D'
                    self.category_setter_register(area)
                    return {'FINISHED'}

            # Close VIEW_3D from ui_type
            for area in context.screen.areas:
                if area.ui_type == 'VIEW_3D':
                    if active_area.height == area.height:
                        with context.temp_override(area=active_area):
                            bpy.ops.screen.area_close()
                        return {'FINISHED'}

            # Split from UV
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
            active_area.ui_type = 'VIEW_3D'

            new_area, = set(context.screen.areas[:]) - old_areas
            c_region = ARegion.get_fields(ARegion.get_n_panel_from_area(new_area))
            if c_region.alignment:
                self.category_setter_register(new_area)
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, f"Active space must be a 3D VIEW or IMAGE_EDITOR")
            return {'CANCELLED'}

    def switch_toggle(self, context):
        active_area = context.area
        if active_area.type == 'VIEW_3D':
            active_area.type = 'IMAGE_EDITOR'
            active_area.ui_type = 'UV'
            self.category_setter_register(active_area)
            return {'FINISHED'}

        if active_area.type == 'IMAGE_EDITOR':
            if active_area.ui_type != 'UV':
                active_area.ui_type = 'UV'
                self.category_setter_register(active_area)
                return {'FINISHED'}
            active_area.type = 'VIEW_3D'  # TODO: Add set_univ_category for 3D after implement 3D VIEW Panel
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, f"Active space must be a 3D VIEW or IMAGE_EDITOR")
            return {'CANCELLED'}

    def create_new_windows(self, context):
        if context.area.type in ('VIEW_3D', 'IMAGE_EDITOR'):
            screens_before = set(bpy.data.screens[:])
            bpy.ops.wm.window_new()

            area = context.area
            area.type = 'IMAGE_EDITOR'
            area.ui_type = 'UV'

            new_screen = set(bpy.data.screens) - screens_before
            c_region = ARegion.get_fields(ARegion.get_n_panel_from_area(area))
            if c_region.alignment == 1:
                with bpy.context.temp_override(area=area, screen=new_screen):  # noqa
                    bpy.ops.wm.context_toggle(data_path='space_data.show_region_ui')

                    image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                    image_editor.uv_editor.show_stretch = True
                    image_editor.uv_editor.display_stretch_type = 'AREA'
                    if hasattr(image_editor, 'show_gizmo_navigate'):
                        image_editor.show_gizmo_navigate = False

            self.category_setter_register(area)

            visible_screens = {w.screen for w in context.window_manager.windows}
            for screen in bpy.data.screens:
                if ('temp' in screen.name) and (screen not in visible_screens):
                    screen.user_clear()

            return {'FINISHED'}
        self.report({'WARNING'}, f"Active space must be a 3D VIEW or IMAGE_EDITOR")
        return {'CANCELLED'}

    def swap_toggle(self, context):
        active_area = context.area
        if active_area.type == 'VIEW_3D':
            for area in context.screen.areas:
                if area.ui_type == 'UV':
                    if context.area.height == area.height:
                        area.type = 'VIEW_3D'
                        area.tag_redraw()
                        active_area.type = 'IMAGE_EDITOR'
                        active_area.ui_type = 'UV'  # TODO: Implement N-Panel swap

                        image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                        image_editor.uv_editor.show_stretch = True
                        image_editor.uv_editor.display_stretch_type = 'AREA'
                        if hasattr(image_editor, 'show_gizmo_navigate'):
                            image_editor.show_gizmo_navigate = False

                        active_area.tag_redraw()
                        return {'FINISHED'}
            self.report({'WARNING'}, f"UV area not found for swap")
            return {'CANCELLED'}

        elif active_area.type == 'IMAGE_EDITOR':
            for area in context.screen.areas:
                if area.ui_type == 'VIEW_3D':
                    if active_area.height == area.height:
                        active_area.type = 'VIEW_3D'
                        active_area.tag_redraw()
                        area.type = 'IMAGE_EDITOR'
                        area.ui_type = 'UV'

                        image_editor = [space for space in area.spaces if space.type == 'IMAGE_EDITOR'][0]
                        image_editor.uv_editor.show_stretch = True
                        image_editor.uv_editor.display_stretch_type = 'AREA'
                        if hasattr(image_editor, 'show_gizmo_navigate'):
                            image_editor.show_gizmo_navigate = False

                        area.tag_redraw()
                        return {'FINISHED'}
            self.report({'WARNING'}, f"3D VIEW area not found for swap")
            return {'CANCELLED'}

        else:
            self.report({'WARNING'}, f"Active space must be a 3D VIEW or IMAGE_EDITOR")
            return {'CANCELLED'}

    def category_setter_register(self, area):
        global AREA
        global AREA_TICKS
        area.tag_redraw()
        AREA = area
        AREA_TICKS = 0
        bpy.app.timers.register(self.register_)

    @staticmethod
    def register_():
        global AREA
        global AREA_TICKS

        if AREA is None:
            return None
        if AREA_TICKS == 0:
            AREA_TICKS = 1
            return 0.01

        if AREA_TICKS < 3 and not UNIV_OT_SplitUVToggle.set_univ_category(AREA):
            try:
                for scr in bpy.data.screens:
                    if AREA in scr.areas[:]:
                        with bpy.context.temp_override(area=AREA, screen=scr):  # noqa
                            bpy.ops.wm.context_toggle(data_path='space_data.show_region_ui')
                            bpy.context.space_data.uv_editor.show_stretch = True
                            bpy.context.space_data.uv_editor.display_stretch_type = 'AREA'
                            if hasattr(bpy.context.space_data, 'show_gizmo_navigate'):
                                bpy.context.space_data.show_gizmo_navigate = False
                        break
            except TypeError:
                if force_debug():
                    traceback.print_exc()
            AREA_TICKS += 1
            return 0.01
        AREA = None
        return None

    @staticmethod
    def set_univ_category(area):
        if stable():
            return True
        try:
            if ARegion.set_active_category('UniV', area):
                area.tag_redraw()
            return True
        except (AttributeError, Exception):
            if force_debug():
                traceback.print_exc()
            return False

class UNIV_OT_SyncUVToggle(Operator):
    bl_idname = 'uv.univ_sync_uv_toggle'
    bl_label = 'Sync UV Toggle'

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        tool_settings = context.tool_settings
        convert_to_sync = not tool_settings.use_uv_select_sync
        tool_settings.use_uv_select_sync = convert_to_sync

        self.sync_uv_selection_mode(convert_to_sync)

        umeshes = utils.UMeshes()
        for umesh in umeshes:
            if convert_to_sync:
                self.to_sync(umesh)
            else:
                self.disable_sync(umesh)
        return umeshes.update()

    @staticmethod
    def sync_uv_selection_mode(sync):
        if sync:
            if utils.get_select_mode_uv() == 'VERTEX':
                utils.set_select_mode_mesh('VERTEX')
            elif utils.get_select_mode_uv() == 'EDGE':
                utils.set_select_mode_mesh('EDGE')
            else:
                utils.set_select_mode_mesh('FACE')

        else:
            if utils.get_select_mode_mesh() == 'VERTEX':
                utils.set_select_mode_uv('VERTEX')
            elif utils.get_select_mode_mesh() == 'EDGE':
                utils.set_select_mode_uv('EDGE')
            else:
                if utils.get_select_mode_uv() == 'ISLAND':
                    return
                utils.set_select_mode_uv('FACE')

    @staticmethod
    def disable_sync(umesh):
        uv = umesh.uv_layer
        if utils.get_select_mode_mesh() == 'FACE':
            if umesh.is_full_face_selected:
                for face in umesh.bm.faces:
                    for loop in face.loops:
                        loop_uv = loop[uv]
                        loop_uv.select = True
                        loop_uv.select_edge = True
            elif umesh.is_full_face_deselected:
                for face in umesh.bm.faces:
                    if not face.hide:
                        face.select = True
                        for loop in face.loops:
                            loop_uv = loop[uv]
                            loop_uv.select = False
                            loop_uv.select_edge = False
            else:
                for face in umesh.bm.faces:
                    sel_state = face.select
                    for loop in face.loops:
                        loop_uv = loop[uv]
                        loop_uv.select = sel_state
                        loop_uv.select_edge = sel_state
                    face.select = True
        else:
            if umesh.is_full_face_selected:
                for face in umesh.bm.faces:
                    for crn in face.loops:
                        crn_uv = crn[uv]
                        crn_uv.select = True
                        crn_uv.select_edge = True
                return
            if umesh.is_full_face_deselected:
                for face in umesh.bm.faces:
                    for crn in face.loops:
                        crn_uv = crn[uv]
                        crn_uv.select = False
                        crn_uv.select_edge = False
                    face.select = True
                return

            for vert in umesh.bm.verts:
                if hasattr(vert, 'link_loops'):
                    sel_state = vert.select
                    for crn in vert.link_loops:
                        crn[uv].select = sel_state
            if not umesh.is_full_face_selected:
                for face in umesh.bm.faces:
                    face.select = True

            if umesh.is_full_face_deselected:
                return

            # Select corner edge
            if umesh.is_full_face_selected:
                corners = (crn__ for face in umesh.bm.faces for crn__ in face.loops)
            else:
                corners = (crn__ for face in umesh.bm.faces if face.select for crn__ in face.loops)

            for crn in corners:
                crn_uv = crn[uv]
                crn_uv.select_edge = crn_uv.select and crn.link_loop_next[uv].select

            # Deselect corner vertex, without linked selected corner edge
            if utils.get_select_mode_mesh() == 'EDGE':
                for vert in umesh.bm.verts:
                    if not (vert.select and hasattr(vert, 'link_loops')):
                        continue
                    if any(crn_[uv].select for crn_ in vert.link_loops if crn_.face.select):
                        crn_groups = defaultdict(list)
                        for crn in vert.link_loops:
                            if crn.face.select:
                                crn_groups[crn[uv].uv.copy().freeze()].append(crn)

                        for corners in crn_groups.values():
                            if not any(crn[uv].select_edge or crn.link_loop_prev[uv].select_edge for crn in corners):
                                for crn in corners:
                                    crn[uv].select = False

    @staticmethod
    def to_sync(umesh):
        uv_layer = umesh.uv_layer
        if utils.get_select_mode_uv() in ('FACE', 'ISLAND'):
            for face in umesh.bm.faces:
                face.select = all(loop[uv_layer].select_edge or loop[uv_layer].select for loop in face.loops)

        elif utils.get_select_mode_uv() == 'VERTEX':
            for vert in umesh.bm.verts:
                if hasattr(vert, 'link_loops'):
                    vert.select = any(loop[uv_layer].select for loop in vert.link_loops)
        else:
            for edge in umesh.bm.edges:
                if hasattr(edge, 'link_loops'):
                    edge.select = any(loop[uv_layer].select_edge for loop in edge.link_loops)
        umesh.bm.select_flush_mode()


def univ_header_sync_btn(self, context):
    if prefs().show_split_toggle_uv_button:
        if context.mode == 'EDIT_MESH':
            layout = self.layout
            layout.operator('uv.univ_sync_uv_toggle', text='', icon='UV_SYNC_SELECT')

def univ_header_split_btn(self, _context):
    if prefs().show_split_toggle_uv_button:
        layout = self.layout
        layout.operator('wm.univ_split_uv_toggle', text='', icon='SCREEN_BACK')

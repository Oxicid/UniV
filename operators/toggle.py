# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import time
import traceback

from .. import utils
from ..preferences import force_debug, prefs, stable
from .. import types
from ..types import ARegion

from collections import defaultdict
from bpy.props import *
from bpy.types import Operator

AREA: bpy.types.Area | None = None
AREA_TICKS: bool = False

class UNIV_OT_SplitUVToggle(Operator):
    bl_idname = 'wm.univ_split_uv_toggle'
    bl_label = 'Split UV Toggle'
    bl_options = {'UNDO'}

    mode: EnumProperty(name='Mode',
                       default='SPLIT',
                       items=(
                           ('SPLIT', 'Split', ''),
                           ('SWITCH', 'Switch', ''),
                           ('NEW_WINDOWS', 'New Windows', ''),
                           ('SWAP', 'Swap', '')
                       ))

    def invoke(self, context, event):
        self.mouse_x_pos = event.mouse_x

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
                self.report({'INFO'}, f"Event: {utils.event_to_string(event)} not implement.")
                return {'CANCELLED'}
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_x_pos = 0

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
        # TODO: If there's another editor to the left of the UV window,
        #  closing the UV window expands the left editor instead of the right one (VIEW 3D).
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
                        self.category_setter_register(area)
                        return {'FINISHED'}

            # If there are two 'VIEW_3D' windows, it switches to IMAGE_EDITOR.
            for area in context.screen.areas:
                if (area != active_area) and (area.type == 'VIEW_3D') and (active_area.height == area.height):
                    active_area.type = 'IMAGE_EDITOR'
                    active_area.ui_type = 'UV'

                    image_editor = [space for space in active_area.spaces if space.type == 'IMAGE_EDITOR'][0]
                    if hasattr(image_editor, 'show_gizmo_navigate'):
                        image_editor.show_gizmo_navigate = False

                    self.category_setter_register(active_area)
                    return {'FINISHED'}

            # Split VIEW_3D - create UV area
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)

            target_area = None
            mouse_in_right_side = self.mouse_x_pos > (active_area.x + active_area.width // 2)
            if prefs().split_toggle_uv_by_cursor and mouse_in_right_side:
                target_area = active_area
            else:
                for area in context.screen.areas:
                    if area.ui_type == 'VIEW_3D' and area != active_area:
                        if area.height == area.height:
                            target_area = area
                            break

            if target_area:
                target_area.ui_type = 'UV'

                image_editor = next(space for space in target_area.spaces if space.type == 'IMAGE_EDITOR')
                if hasattr(image_editor, 'show_gizmo_navigate'):
                    image_editor.show_gizmo_navigate = False

                self.category_setter_register(target_area)
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, 'Splitted area not found')
                return {'CANCELLED'}

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

            # Split from UV - create 3D area
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
            new_area, = set(context.screen.areas[:]) - old_areas

            mouse_in_right_side = self.mouse_x_pos > (active_area.x + active_area.width // 2)
            if not (prefs().split_toggle_uv_by_cursor and mouse_in_right_side):
                new_area, active_area = active_area, new_area

            active_area.ui_type = 'VIEW_3D'
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


class UNIV_OT_TogglePanelsByCursor(Operator):
    """Idea inspired by Machin3. Useful for freeing up the T and N keys."""
    bl_idname = 'wm.univ_toggle_panels_by_cursor'
    bl_label = 'Toggle Panels By Cursor'
    bl_options = {'UNDO'}

    def invoke(self, context, event):
        self.mouse_x_pos = event.mouse_x
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_x_pos = -50_000

    def execute(self, context):
        if context.area is None:
            return {'PASS_THROUGH'}

        if self.mouse_x_pos == -50_000:
            self.report({'WARNING'}, "UniV: The operator 'Toggle Panels By Cursor' should be called via INVOKE_DEFAULT")
            return {'CANCELLED'}

        active_area = context.area
        mouse_in_right_side = self.mouse_x_pos > (active_area.x + active_area.width // 2)

        if mouse_in_right_side:
            for space in context.area.spaces:
                if hasattr(space, 'show_region_ui'):
                    return bpy.ops.wm.context_toggle(data_path='space_data.show_region_ui')
        else:
            for space in context.area.spaces:
                if hasattr(space, 'show_region_toolbar'):
                    return bpy.ops.wm.context_toggle(data_path='space_data.show_region_toolbar')
        return {'CANCELLED'}

PREV_PIVOT = ''
PREV_PIVOT_TIME = 0.0
class UNIV_OT_TogglePivot(Operator):
    bl_idname = 'uv.univ_toggle_pivot'
    bl_label = 'Toggle Pivot'
    bl_options = {'UNDO'}
    bl_description = ('Toggles the pivot between Bounding Box and Individual Origins. \n'
                      'If toggled again within 1.8 seconds, it switches from Bounding Box or Individual to Cursor.')


    def execute(self, context):
        from time import perf_counter
        from .. import draw
        global PREV_PIVOT
        global PREV_PIVOT_TIME
        curr_time = perf_counter()
        curr_pivot = context.space_data.pivot_point  # noqa

        draw_time = 1.8
        draw.TextDraw.max_draw_time = draw_time
        readable_text = {'INDIVIDUAL_ORIGINS': 'Individual',
                         'CENTER': 'Boundary Box',
                         'MEDIAN': 'Median',
                         'CURSOR': 'Cursor' }

        if (curr_time - PREV_PIVOT_TIME) <= draw_time:
            if PREV_PIVOT == 'CENTER' and curr_pivot == 'INDIVIDUAL_ORIGINS':
                context.space_data.pivot_point = 'CURSOR'
            elif PREV_PIVOT == 'INDIVIDUAL_ORIGINS' and curr_pivot == 'CENTER':
                context.space_data.pivot_point = 'CURSOR'
            elif PREV_PIVOT == 'INDIVIDUAL_ORIGINS' and curr_pivot == 'MEDIAN':
                context.space_data.pivot_point = 'CURSOR'
            elif PREV_PIVOT == 'CENTER' and curr_pivot == 'MEDIAN':
                context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
            elif PREV_PIVOT == 'MEDIAN' and curr_pivot == 'CENTER':
                context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
            else:
                if curr_pivot == 'CENTER':
                    context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
                else:
                    context.space_data.pivot_point = 'CENTER'

            draw.TextDraw.draw(f'Switch to {readable_text[context.space_data.pivot_point]!r}')

        else:
            if PREV_PIVOT == '' or PREV_PIVOT == curr_pivot:
                if curr_pivot == 'CENTER':
                    context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
                else:
                    context.space_data.pivot_point = 'CENTER'
            else:
                draw.TextDraw.draw(f'Toggle: {readable_text[context.space_data.pivot_point]!r} ➔ {readable_text[PREV_PIVOT]!r}')
                context.space_data.pivot_point = PREV_PIVOT

        PREV_PIVOT = curr_pivot
        PREV_PIVOT_TIME = curr_time
        return {'FINISHED'}


class UNIV_OT_SyncUVToggle(Operator):
    bl_idname = 'uv.univ_sync_uv_toggle'
    bl_label = 'Sync UV Toggle'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        tool_settings = context.tool_settings
        convert_to_sync = not tool_settings.use_uv_select_sync
        tool_settings.use_uv_select_sync = convert_to_sync

        self.sync_uv_selection_mode(convert_to_sync)

        umeshes = types.UMeshes()
        for umesh in umeshes:
            if convert_to_sync:
                self.to_sync(umesh)
            else:
                self.disable_sync(umesh)
        return umeshes.update()

    @staticmethod
    def sync_uv_selection_mode(sync):
        if sync:
            if utils.get_select_mode_uv() == 'VERT':
                utils.set_select_mode_mesh('VERT')
            elif utils.get_select_mode_uv() == 'EDGE':
                utils.set_select_mode_mesh('EDGE')
            else:
                utils.set_select_mode_mesh('FACE')

        else:
            if utils.get_select_mode_mesh() == 'VERT':
                utils.set_select_mode_uv('VERT')
            elif utils.get_select_mode_mesh() == 'EDGE':
                utils.set_select_mode_uv('EDGE')
            else:
                if utils.get_select_mode_uv() == 'ISLAND':
                    return
                utils.set_select_mode_uv('FACE')

    @staticmethod
    def disable_sync(umesh):
        uv = umesh.uv
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
        uv = umesh.uv
        if utils.get_select_mode_uv() in ('FACE', 'ISLAND'):
            for face in umesh.bm.faces:
                face.select = all(loop[uv].select_edge or loop[uv].select for loop in face.loops)

        elif utils.get_select_mode_uv() == 'VERT':
            for vert in umesh.bm.verts:
                if hasattr(vert, 'link_loops'):
                    vert.select = any(loop[uv].select for loop in vert.link_loops)
        else:
            for edge in umesh.bm.edges:
                if hasattr(edge, 'link_loops'):
                    edge.select = any(loop[uv].select_edge for loop in edge.link_loops)
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


LAST_STRETCH_TYPE: str = ''
LAST_STRETCH_TIME: float = 0.0
STRETCH_SPACE_DATA: bpy.types.Space | None = None

class UNIV_OT_StretchUVToggle(Operator):
    bl_idname = 'uv.univ_stretch_uv_toggle'
    bl_label = 'Stretch UV Toggle'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "A single press toggles Stretch type on and off. Double press switches the Stretch type"

    swap: BoolProperty(name='Swap', default=False)

    def execute(self, context):
        active_area = context.area
        if active_area.type == 'IMAGE_EDITOR' and active_area.ui_type != 'UV':
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}

        global LAST_STRETCH_TYPE
        global STRETCH_SPACE_DATA

        if not self.swap:
            STRETCH_SPACE_DATA = None

        uv_editor = context.space_data.uv_editor
        if self.swap:
            global LAST_STRETCH_TIME
            LAST_STRETCH_TIME = time.perf_counter()
            uv_editor.show_stretch = True
            if not LAST_STRETCH_TYPE or LAST_STRETCH_TYPE == 'ANGLE':
                uv_editor.display_stretch_type = 'AREA'
                LAST_STRETCH_TYPE = 'AREA'
            else:
                uv_editor.display_stretch_type = 'ANGLE'
                LAST_STRETCH_TYPE = 'ANGLE'

            umeshes = types.UMeshes.calc(verify_uv=False)
            count_non_default_scale = sum(bool(umesh.check_uniform_scale()) for umesh in umeshes)
            umeshes.free()

            txt = LAST_STRETCH_TYPE.capitalize()
            if count_non_default_scale:
                txt = [f'Warning: The scale hasn`t been applied to {count_non_default_scale} objects', txt]
            from ..  import draw
            draw.TextDraw.draw(txt)
        else:
            STRETCH_SPACE_DATA = context.space_data
            bpy.app.timers.register(self.register_,  first_interval=0.22)

        return {'FINISHED'}

    @staticmethod
    def register_():
        global LAST_STRETCH_TYPE
        global STRETCH_SPACE_DATA
        global LAST_STRETCH_TIME

        if isinstance(STRETCH_SPACE_DATA, type(None)):
            return None

        delta = time.perf_counter() - LAST_STRETCH_TIME
        if delta < 0.22:
            STRETCH_SPACE_DATA = None
            return None

        uv_editor = STRETCH_SPACE_DATA.uv_editor
        if uv_editor.show_stretch:
            uv_editor.show_stretch = False
            LAST_STRETCH_TYPE = uv_editor.display_stretch_type
            return None

        uv_editor.show_stretch = True
        if not LAST_STRETCH_TYPE:
            LAST_STRETCH_TYPE = 'ANGLE'

        uv_editor.display_stretch_type = LAST_STRETCH_TYPE

        umeshes = types.UMeshes.calc(verify_uv=False)
        count_non_default_scale = sum(bool(umesh.check_uniform_scale()) for umesh in umeshes)
        umeshes.free()

        txt = LAST_STRETCH_TYPE.capitalize()
        if count_non_default_scale:
            txt = [f'Warning: The scale hasn`t been applied to {count_non_default_scale} objects', txt]

        from .. import draw
        draw.TextDraw.draw(txt)

        STRETCH_SPACE_DATA = None
        return None

class UNIV_OT_ShowModifiedUVEdgeToggle(Operator):
    bl_idname = 'uv.univ_show_modified_uv_edges_toggle'
    bl_label = 'Show Modified UV Edges Toggle'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_area = context.area
        if active_area.type == 'IMAGE_EDITOR' and active_area.ui_type != 'UV':
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}
        context.space_data.uv_editor.show_modified_edges ^= 1
        return {'FINISHED'}

class UNIV_OT_ModifiersToggle(Operator):
    bl_idname = 'view3d.univ_modifiers_toggle'
    bl_label = 'Toggle Modifiers'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_obj = context.active_object
        selected_objects = context.selected_objects.copy()

        if active_obj and active_obj not in selected_objects:
            selected_objects.append(active_obj)

        modifier_status = [mod for obj in selected_objects if obj.type != 'EMPTY'
                           for mod in obj.modifiers if mod.type != 'COLLISION']
        if not modifier_status:
            self.report({'INFO'}, 'Not found modifiers')
            return {'FINISHED'}
        show_status = not all(mod.show_viewport for mod in modifier_status)
        for mod in modifier_status:
            if mod.show_viewport != show_status:
                mod.show_viewport = show_status
        return {'FINISHED'}


class UNIV_OT_WorkspaceToggle(Operator):
    bl_idname = 'wm.univ_workspace_toggle'
    bl_label = 'Toggle Workspace'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode not in ('EDIT_MESH', 'OBJECT') or context.area.type != 'VIEW_3D':
            return {'PASS_THROUGH'}

        tools = bpy.context.workspace.tools
        active_tool_name = tools.from_space_view3d_mode(context.mode).idname

        if not ToggleHandlers.owners:
            ToggleHandlers.subscribe_to_tool()

        if active_tool_name == 'tool.univ':
            if bpy.context.mode == 'EDIT_MESH':
                if ToggleHandlers.last_tool_edit not in ('', 'tool.univ') and \
                        self.contain_tool_by_name(ToggleHandlers.last_tool_edit):
                    bpy.ops.wm.tool_set_by_id(name=ToggleHandlers.last_tool_edit)
                else:
                    bpy.ops.wm.tool_set_by_id(name='builtin.select_box')
            else:
                if ToggleHandlers.last_tool_object not in ('', 'tool.univ') and \
                        self.contain_tool_by_name(ToggleHandlers.last_tool_object):
                    bpy.ops.wm.tool_set_by_id(name=ToggleHandlers.last_tool_object)
                else:
                    bpy.ops.wm.tool_set_by_id(name='builtin.select_box')
        else:
            if bpy.context.mode == 'EDIT_MESH':
                ToggleHandlers.last_tool_edit = active_tool_name
            else:
                ToggleHandlers.last_tool_object = active_tool_name
            bpy.ops.wm.tool_set_by_id(name='tool.univ')

        return {'FINISHED'}

    @staticmethod
    def contain_tool_by_name(name, space_type: str='VIEW_3D'):
        from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
        tool_helper_cls = ToolSelectPanelHelper._tool_class_from_space_type(space_type)  # noqa

        for item in ToolSelectPanelHelper._tools_flatten(  # noqa
                tool_helper_cls.tools_from_context(bpy.context, mode=bpy.context.mode)):
            if getattr(item, 'idname', None) == name:
                return True
        return False

class ToggleHandlers:
    last_tool_object = ''
    last_tool_edit = ''
    owners = []

    @staticmethod
    def univ_toggle_rna_callback():
        tools = bpy.context.workspace.tools

        if (mode := bpy.context.mode) == 'EDIT_MESH':
            if tool := tools.from_space_view3d_mode(mode):
                idname = tool.idname
                if idname != 'tool.univ':
                    ToggleHandlers.last_tool_edit = idname
        elif mode == 'OBJECT':
            if tool := tools.from_space_view3d_mode(mode):
                idname = tool.idname
                if idname != 'tool.univ':
                    ToggleHandlers.last_tool_object = idname

    @classmethod
    def subscribe_to_tool(cls):
        cls.owners.clear()
        for ws in bpy.data.workspaces:
            owner = ('UniV', ws.name)
            cls.owners.append(owner)
            bpy.msgbus.subscribe_rna(
                key=ws.path_resolve("tools", False),
                owner=owner,
                args=(),
                notify=cls.univ_toggle_rna_callback)

    @classmethod
    def unsubscribe_from_tool(cls):
        for owner in cls.owners:
            bpy.msgbus.clear_by_owner(owner)
        cls.owners.clear()

    @staticmethod
    @bpy.app.handlers.persistent
    def univ_tool_load_handler(_):
        ToggleHandlers.subscribe_to_tool()

    @classmethod
    def register_handler(cls):
        cls.unregister_handler()
        bpy.app.handlers.load_post.append(cls.univ_tool_load_handler)

    @classmethod
    def unregister_handler(cls):
        cls.unsubscribe_from_tool()
        for handler in reversed(bpy.app.handlers.load_post):
            if handler.__name__ == 'univ_tool_load_handler':
                bpy.app.handlers.load_post.remove(handler)

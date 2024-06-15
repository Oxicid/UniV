import bpy
import traceback

from .. import info
from ..types import ARegion

from bpy.props import *
from bpy.types import Operator

AREA: bpy.types.Area | None = None
AREA_SKIP: bool = False

class UNIV_OT_SplitUVToggle(Operator):
    bl_idname = 'wm.split_uv_toggle'
    bl_label = 'Split UV Toggle'

    mode: EnumProperty(name='Mode', default='SPLIT', items=(('SPLIT', 'Split', ''), ('SWITCH', 'Switch', ''), ('NEW_WINDOWS', 'New Windows', '')))

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
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement.")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        if context.window.screen.show_fullscreen is True and self.mode != 'SWITCH':
            if self.mode == 'SPLIT':
                self.report({'INFO'}, "You can 't split in full screen mode. Switch Mode is used instead.")
            elif self.mode == 'NEW_WINDOWS':
                self.report({'INFO'}, "You cannot create new windows in Full Screen Mode. Switch Mode is used instead.")
            return self.switch_toggle(context)

        if self.mode == 'SPLIT':
            return self.split_toggle(context)
        elif self.mode == 'SWITCH':
            return self.switch_toggle(context)
        elif self.mode == 'NEW_WINDOWS':
            return self.create_new_windows(context)
        self.report({'WARNING'}, f"Mode: {self.mode} not implemented")
        return {'CANCELLED'}

    def split_toggle(self, context):
        active_area = context.area
        if active_area.type == 'VIEW_3D':
            # Close ui_type
            for area in context.screen.areas:
                if area.ui_type == 'UV':
                    if context.area.height == area.height:
                        bpy.ops.screen.area_close({'area': area})
                        return {'FINISHED'}

            # Change area_type (if area != ui_type) from 3D VIEW
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    if context.area.height == area.height:
                        area.ui_type = 'UV'
                        self.category_setter_register(area)
                        return {'FINISHED'}

            # If there are two 'VIEW_3D' windows, it switches to IMAGE_EDITOR.
            for area in context.screen.areas:
                if (area != active_area) and (area.type == 'VIEW_3D') and (active_area.height == area.height):
                    active_area.type = 'IMAGE_EDITOR'
                    active_area.ui_type = 'UV'
                    self.category_setter_register(active_area)
                    return {'FINISHED'}

            # Split VIEW_3D
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)

            for area in context.screen.areas:
                if area.ui_type == 'VIEW_3D' and area != active_area:
                    if area.height == area.height:
                        area.ui_type = 'UV'
                        self.category_setter_register(area)
                        return {'FINISHED'}

        # Change ui_type to ShaderNodeTree from area_type
        elif active_area.type == 'IMAGE_EDITOR' and active_area.ui_type != 'UV':
            active_area.ui_type = 'UV'
            self.category_setter_register(active_area)
            return {'FINISHED'}

        elif active_area.type == 'IMAGE_EDITOR':
            # If there are two 'IMAGE_EDITOR' windows, it switches to VIEW_3D.
            for area in context.screen.areas:
                if (area != active_area) and (area.type == 'IMAGE_EDITOR') and (active_area.height == area.height):
                    active_area.type = 'VIEW_3D'
                    # self.category_setter_register(active_area)
                    return {'FINISHED'}

            # Close VIEW_3D from ui_type
            for area in context.screen.areas:
                if area.ui_type == 'VIEW_3D':
                    if active_area.height == area.height:
                        with context.temp_override(area=area):
                            bpy.ops.screen.area_close({'area': active_area})
                        return {'FINISHED'}

            # Split from ui_type
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
            active_area.ui_type = 'VIEW_3D'
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

            with bpy.context.temp_override(screen=new_screen):  # noqa
                bpy.ops.wm.context_toggle(data_path='space_data.show_region_ui')
            self.category_setter_register(area)

            visible_screens = {w.screen for w in context.window_manager.windows}
            for screen in bpy.data.screens:
                if ('temp' in screen.name) and (screen not in visible_screens):
                    screen.user_clear()

            return {'FINISHED'}
        self.report({'WARNING'}, f"Active space must be a 3D VIEW or IMAGE_EDITOR")
        return {'CANCELLED'}

    def category_setter_register(self, area):
        global AREA
        global AREA_SKIP
        area.tag_redraw()
        AREA = area
        AREA_SKIP = False
        bpy.app.timers.register(self.register_)

    @staticmethod
    def register_():
        global AREA
        global AREA_SKIP

        if AREA is None:
            return None
        if AREA_SKIP is False:
            AREA.tag_redraw()
            AREA_SKIP = True
            return 0.01

        UNIV_OT_SplitUVToggle.set_univ_category(AREA)
        AREA = None
        return None

    @staticmethod
    def set_univ_category(area):
        try:
            if ARegion.set_active_category('UniV', area):
                area.tag_redraw()
        except (AttributeError, Exception):
            if False:  # TODD: Implement Debug
                traceback.print_exc()

def univ_header_btn(self, _context):
    layout = self.layout
    layout.operator('wm.split_uv_toggle', text='', icon='ARROW_LEFTRIGHT')

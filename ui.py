# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.types import Panel, Menu

from .icons import icons
from .preferences import univ_settings, prefs

try:
    from . import univ_pro
except ImportError:
    univ_pro = None

REDRAW_UV_LAYERS = True

class UNIV_PT_General(Panel):
    bl_label = ''
    bl_idname = 'UNIV_PT_General'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

    @staticmethod
    def draw_align_buttons(where):
        def ly_wide_icon_op(layer, direct, icon):
            row = layer.row(align=True)
            row.ui_units_x = 3
            row.scale_x = 2.05
            row.operator('uv.univ_align', text="", icon_value=icon).direction = direct

        def ly_mid_mid_op(layer, direct, icon):
            row = layer.row(align=True)
            row.operator('uv.univ_align', text="", icon_value=icon).direction = direct
            row.scale_x = 2

        col_main = where.column(align=True)
        row_top = col_main.row(align=True)

        ly_wide_icon_op(row_top, 'LEFT_UPPER', icons.arrow_top_left)
        ly_wide_icon_op(row_top.row(), 'UPPER', icons.arrow_top)
        ly_wide_icon_op(row_top, 'RIGHT_UPPER', icons.arrow_top_right)

        row_middle = col_main.row().row(align=True)
        ly_wide_icon_op(row_middle, 'LEFT', icons.arrow_left)

        row_mid_middle = row_middle.row().row(align=True)

        ly_mid_mid_op(row_mid_middle, 'HORIZONTAL', icons.horizontal_c)
        ly_mid_mid_op(row_mid_middle.row(), 'CENTER', icons.center)
        ly_mid_mid_op(row_mid_middle.row(), 'VERTICAL', icons.vertical_b)
        ly_wide_icon_op(row_middle, 'RIGHT', icons.arrow_right)

        row_bottom = col_main.row(align=True)
        ly_wide_icon_op(row_bottom, 'LEFT_BOTTOM', icons.arrow_bottom_left)
        ly_wide_icon_op(row_bottom.row(), 'BOTTOM', icons.arrow_bottom)
        ly_wide_icon_op(row_bottom, 'RIGHT_BOTTOM', icons.arrow_bottom_right)

    @staticmethod
    def draw_texel_density(layer, prefix):
        settings = univ_settings()
        split = layer.split(align=True)
        row = split.row(align=True)
        set_idname = prefix + '.univ_texel_density_set'
        row.operator(set_idname).custom_texel = -1.0
        row.operator(prefix + '.univ_texel_density_get')
        row.prop(settings, 'texel_density', text='')
        row.operator(prefix + '.univ_select_texel_density', text='', icon_value=icons.arrow)
        row.popover(panel='UNIV_PT_td_presets_manager', text='', icon_value=icons.settings_a)

        split = layer.split(align=False)
        row = split.row(align=True)
        for idx, preset in enumerate(settings.texels_presets):
            if idx and (idx+1) % 4 == 1:
                split = layer.split(align=False)
                row = split.row(align=True)
            row.operator(set_idname, text=preset.name).custom_texel = preset.texel

    @staticmethod
    def draw_uv_layers(layout):
        settings = univ_settings()
        if not settings.uv_layers_show:
            return
        global REDRAW_UV_LAYERS

        if REDRAW_UV_LAYERS:
            from .operators.misc import UNIV_OT_UV_Layers_Manager
            bpy.app.timers.register(UNIV_OT_UV_Layers_Manager.update_uv_layers_props, first_interval=0.1)
            REDRAW_UV_LAYERS = False

        layout.label(text='UV Maps')
        row = layout.row()
        col = row.column()
        col.template_list(
            listtype_name="UNIV_UL_UV_LayersManager",
            list_id="",
            dataptr=settings,
            propname="uv_layers_presets",
            active_dataptr=settings,
            active_propname="uv_layers_active_idx",
            rows=4,
            maxrows=4,
            columns=4,
            # type='GRID'
        )
        col = row.column(align=True)
        col.operator('mesh.univ_add', icon='ADD', text='')
        col.operator('mesh.univ_remove', icon='REMOVE', text='')
        col.separator(factor=0.25)
        col.operator('mesh.univ_move_up', icon='TRIA_UP', text='')
        col.operator('mesh.univ_move_down', icon='TRIA_DOWN', text='')
        col.separator(factor=0.25)
        col.operator('mesh.univ_fix_uvs', icon='EVENT_F', text='')

    def draw_header(self, context):
        layout = self.layout
        row = layout.split(factor=.5)
        row.popover(panel='UNIV_PT_GlobalSettings', text="", icon_value=icons.settings_b)
        row.label(text='UniV')

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        col = layout.column(align=True)

        col.label(text='Transform')
        row = col.row(align=True)
        row.operator('uv.univ_crop', icon_value=icons.crop).axis = 'XY'
        row.operator('uv.univ_crop', text='', icon_value=icons.x).axis = 'X'
        row.operator('uv.univ_crop', text='', icon_value=icons.y).axis = 'Y'

        row = col.row(align=True)
        row.operator('uv.univ_fill', icon_value=icons.fill).axis = 'XY'
        row.operator('uv.univ_fill', text='', icon_value=icons.x).axis = 'X'
        row.operator('uv.univ_fill', text='', icon_value=icons.y).axis = 'Y'

        row = col.row(align=True)
        row.operator('uv.univ_reset_scale', icon_value=icons.reset).axis = 'XY'
        row.operator('uv.univ_reset_scale', text='', icon_value=icons.x).axis = 'X'
        row.operator('uv.univ_reset_scale', text='', icon_value=icons.y).axis = 'Y'

        row = col.row(align=True)
        row.operator('uv.univ_orient', icon_value=icons.orient).edge_dir = 'BOTH'
        row.operator('uv.univ_orient', text='', icon_value=icons.arrow_right).edge_dir = 'HORIZONTAL'
        row.operator('uv.univ_orient', text='', icon_value=icons.arrow_top).edge_dir = 'VERTICAL'

        col_for_align = col.column()
        col_for_align.separator(factor=0.5)
        self.draw_align_buttons(col_for_align)

        col = layout.column()
        col_align = col.column(align=True)
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_rotate', icon_value=icons.rotate)
        row.operator('uv.univ_flip', icon_value=icons.flip)

        split = col_align.split(align=True)
        split.operator('uv.univ_sort', icon_value=icons.sort)
        row = split.row(align=True)
        row.operator('uv.univ_distribute', icon_value=icons.distribute)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_home', icon_value=icons.home)
        row.operator('uv.univ_shift', icon_value=icons.shift)

        split = col_align.split(align=True)
        split.operator('uv.univ_random', icon_value=icons.random)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_adjust_td', icon_value=icons.adjust)
        row.operator('uv.univ_normalize', icon_value=icons.normalize)

        self.draw_texel_density(col_align, 'uv')

        # Pack
        col_align = col.column(align=True)
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.scale_y = 1.3
        # row.scale_x = 2
        row.operator('uv.univ_pack', icon_value=icons.pack)
        row.popover(panel='UNIV_PT_PackSettings', text='', icon_value=icons.settings_a)

        # Misc
        col_align = col.column(align=True)

        col_align.label(text='Misc')
        if univ_pro:
            split = col_align.split(align=True)
            split.operator('uv.univ_rectify', icon_value=icons.rectify)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_quadrify', icon_value=icons.quadrify)
        row.operator('uv.univ_straight', icon_value=icons.straight)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_relax', icon_value=icons.relax)
        row.operator('uv.univ_unwrap', icon_value=icons.unwrap)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_weld', icon_value=icons.weld)
        row.operator('uv.univ_stitch', icon_value=icons.stitch)

        split = col_align.split(align=True)
        split.scale_y = 1.3
        split.operator('uv.univ_stack', icon_value=icons.stack)

        # Select
        col_align.label(text='Select')
        col_align = col.column(align=True)

        if univ_pro:
            row = col_align.row(align=True)
            row.operator('uv.univ_select_flat', icon_value=icons.flat)
            row.operator('uv.univ_select_loop', icon_value=icons.loop_select)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_grow', icon_value=icons.grow)
        row.operator('uv.univ_select_edge_grow', icon_value=icons.edge_grow)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_linked', icon_value=icons.linked)
        row.operator('uv.univ_select_by_cursor', icon_value=icons.cursor)

        row = col_align.row(align=True)
        row.operator('uv.univ_select_border', icon_value=icons.border)
        row.operator('uv.univ_select_stacked', icon_value=icons.select_stacked)

        col_align.separator(factor=0.35)

        row = col_align.row(align=True)
        row.operator('uv.univ_select_border_edge_by_angle', icon_value=icons.border_by_angle).edge_dir = 'BOTH'
        row.operator('uv.univ_select_border_edge_by_angle', text='', icon_value=icons.horizontal_a).edge_dir = 'HORIZONTAL'
        row.operator('uv.univ_select_border_edge_by_angle', text='', icon_value=icons.vertical_a).edge_dir = 'VERTICAL'

        row = col_align.row(align=True)
        row.operator('uv.univ_select_square_island', icon_value=icons.square).shape = 'SQUARE'
        row.operator('uv.univ_select_square_island', text='',  icon_value=icons.horizontal_a).shape = 'HORIZONTAL'
        row.operator('uv.univ_select_square_island', text='',  icon_value=icons.vertical_a).shape = 'VERTICAL'

        row = col_align.row(align=True)
        row.operator('uv.univ_select_by_area', text='Small', icon_value=icons.small).size_mode = 'SMALL'
        row.operator('uv.univ_select_by_area', text='Medium', icon_value=icons.medium).size_mode = 'MEDIUM'
        row.operator('uv.univ_select_by_area', text='Large', icon_value=icons.large).size_mode = 'LARGE'

        # Inspect
        col_align = col.column(align=True)
        col_align.label(text='Inspect')

        split = col_align.split(align=True)
        split.operator('uv.univ_check_zero', icon_value=icons.zero)
        split.operator('uv.univ_check_flipped', icon_value=icons.flipped)

        split = col_align.split(align=True)
        split.operator('uv.univ_check_non_splitted', icon_value=icons.non_splitted)
        split.operator('uv.univ_check_overlap', icon_value=icons.overlap)

        # Mark
        col_align = col.column(align=True)
        col_align.label(text='Mark')
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_pin', icon_value=icons.pin)

        split = col_align.split(align=True)
        split.operator('uv.univ_cut', icon_value=icons.cut)
        split.operator('uv.univ_seam_border', icon_value=icons.border_seam)

        layout.label(text='Texture')
        row = layout.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.univ_checker', icon_value=icons.checker)
        row.operator('wm.univ_checker_cleanup', text='', icon_value=icons.remove)

        self.draw_uv_layers(self.layout)


class UNIV_PT_General_VIEW_3D(Panel):
    bl_label = ''
    bl_idname = 'UNIV_PT_General_VIEW3D'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "UniV"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        col = layout.column(align=True)

        col_align = col.column(align=True)
        col_align.label(text='Mark')
        split = col_align.split(align=True)
        split.operator('mesh.univ_cut', icon_value=icons.cut)
        split.operator('mesh.univ_seam_border', icon_value=icons.border_seam)

        split = col_align.split(align=True)
        split.operator('mesh.univ_angle', icon_value=icons.border_by_angle)

        col_align.label(text='Project')
        if univ_pro:
            row = col_align.row(align=True)
            row.operator('mesh.univ_transfer', icon_value=icons.transfer)
        row = col_align.row(align=True)
        row.operator('mesh.univ_normal', icon_value=icons.normal)
        row.operator('mesh.univ_box_project', icon_value=icons.box)

        row = col_align.row(align=True)
        row.operator('mesh.univ_smart_project', icon_value=icons.smart)
        row.operator('mesh.univ_view_project', icon_value=icons.view)

        col_align.label(text='Misc')
        row = col_align.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.univ_stack', icon_value=icons.stack)

        col_align.label(text='Transform')
        col_align.operator('mesh.univ_gravity', icon_value=icons.gravity)

        row = col_align.row(align=True)
        row.operator('mesh.univ_adjust_td', icon_value=icons.adjust)
        row.operator('mesh.univ_normalize', icon_value=icons.normalize)

        UNIV_PT_General.draw_texel_density(col_align, 'mesh')

        col_align.label(text='Select')
        if univ_pro:
            row = col_align.row(align=True)
            row.operator('mesh.univ_select_flat', icon_value=icons.flat)
            row.operator('mesh.univ_select_loop', icon_value=icons.loop_select)
        row = col_align.row(align=True)
        row.operator('mesh.univ_select_grow', icon_value=icons.grow)
        row.operator('mesh.univ_select_edge_grow', icon_value=icons.edge_grow)

        col_align.label(text='Texture')
        row = col_align.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.univ_checker', icon_value=icons.checker)
        row.operator('wm.univ_checker_cleanup', text='', icon_value=icons.remove)

        UNIV_PT_General.draw_uv_layers(layout)


class UNIV_PT_GlobalSettings(Panel):
    bl_idname = 'UNIV_PT_GlobalSettings'
    bl_label = 'Global Settings'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_options = {"INSTANCED"}
    bl_category = "UniV"

    def draw(self, context):
        self.draw_global_settings(self.layout)

    @staticmethod
    def draw_global_settings(layout):
        settings = univ_settings()

        row = layout.row(align=True, heading='Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        layout.prop(settings, 'padding', slider=True)
        layout.separator()
        layout.prop(settings, 'uv_layers_show')
        layout.prop(prefs(), 'enable_uv_layers_sync_borders_seam')


class UNIV_PT_PackSettings(Panel):
    bl_idname = 'UNIV_PT_PackSettings'
    bl_label = 'Pack Settings'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_options = {"INSTANCED"}
    bl_category = "UniV"

    def draw(self, context):
        layout = self.layout
        settings = univ_settings()

        row = layout.row(align=True, heading='Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        layout.prop(settings, 'padding', slider=True)
        layout.separator()

        if not bpy.app.version >= (3, 6, 0):
            layout.prop(settings, 'rotate', toggle=1)
        else:
            row = layout.row(align=True)
            row.prop(settings, 'shape_method', expand=True)

            row = layout.row(align=True)
            row.prop(settings, 'scale', toggle=1)
            row.prop(settings, 'rotate', toggle=1)

            if settings.rotate:
                row = layout.row().column()
                row.scale_x = 1.5
                row.alignment = 'CENTER'
                row.prop(settings, 'rotate_method', text='Rotation Method')

            if settings.pin:
                row.prop(settings, 'pin_method', text='Lock Method       ')

            self.layout.prop(settings, 'pin')
            layout.prop(settings, 'merge_overlap')
        layout.prop(settings, 'udim_source')


class UNIV_UL_TD_PresetsManager(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):  # noqa
        row = layout.row(align=0)
        row.prop(item, 'name', text=str(index+1), emboss=True)
        row.prop(item, 'texel', text='TD', emboss=False)


class UNIV_UL_UV_LayersManager(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):  # noqa
        # TODO: Redraw if undo???
        settings = univ_settings()
        if index >= settings.uv_layers_size:
            return
        row = layout.row(align=0)
        if flag := item.flag:  # noqa
            if flag == 2:
                row.alert = True
            else:
                row.active = False
        row.prop(item, 'name', text='', emboss=False, icon='GROUP_UVS')  # noqa
        icon = 'RESTRICT_RENDER_OFF' if settings.uv_layers_active_render_idx == index else 'RESTRICT_RENDER_ON'
        row.operator('mesh.univ_active_render_set', text='', icon=icon).idx = index


class UNIV_PT_TD_PresetsManager(Panel):
    bl_label = 'Texel Density Presets Manager'
    bl_idname = 'UNIV_PT_td_presets_manager'
    bl_space_type = 'IMAGE_EDITOR'
    bl_options = {'INSTANCED'}
    bl_region_type = 'UI'
    bl_category = 'UniV'

    def draw(self, context):
        settings = univ_settings()

        layout = self.layout
        layout.label(text=f"Texel Density: {round(settings.texel_density, 4)}")
        row = layout.row(align=True, heading='Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        row = layout.row()
        col = row.column()
        col.scale_x = 0.5
        col.template_list(
            listtype_name="UNIV_UL_TD_PresetsManager",
            list_id="",
            dataptr=settings,  # noqa
            propname="texels_presets",
            active_dataptr=settings,  # noqa
            active_propname="active_td_index",
            maxrows=9
        )

        col = row.column(align=True)
        col.operator('scene.univ_td_presets_processing', icon='ADD', text="").operation_type = 'ADD'
        col.operator('scene.univ_td_presets_processing', icon='REMOVE', text="").operation_type = 'REMOVE'
        col.separator()
        col.operator('scene.univ_td_presets_processing', icon='TRASH', text="").operation_type = 'REMOVE_ALL'


class IMAGE_MT_PIE_univ_edit(Menu):
    bl_label = 'UniV Pie'

    # @classmethod
    # def poll(cls, context):
    #     return context.mode == 'EDIT'

    def draw(self, _context):
        # Angle
        pie = self.layout.menu_pie()
        pie.scale_x = 1.25
        pie.scale_y = 2.0
        split = pie.split()

        col = split.column(align=True)

        split = pie.split()
        col = split.column(align=True)
        row = col.row(align=True)
        row.operator("uv.univ_sync_uv_toggle", icon='UV_SYNC_SELECT')

        # MultiLoop
        split = pie.split()
        col = split.column(align=True)

        # Boundary loop
        split = pie.split()
        col = split.column(align=True)
        row = col.row()

        # Toggle
        split = pie.split()
        col = split.column(align=True)

        # View
        split = pie.split()
        col = split.column(align=True)

class VIEW3D_MT_PIE_univ_obj(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        # Angle
        pie = self.layout.menu_pie()
        # pie.scale_x = 1.25
        # pie.scale_y = 2.0
        split = pie.split()

        col = split.column(align=True)
        row = col.row(align=True)

        split = pie.split()
        col = split.column(align=False)
        # col.scale_x = 1.25
        col.scale_y = 2.0
        col.operator("view3d.univ_modifiers_toggle", text='Toggle Modifiers', icon='HIDE_OFF')

        # MultiLoop
        split = pie.split()
        col = split.column(align=False)
        col.alignment = 'CENTER'
        UNIV_PT_General.draw_uv_layers(col)

        # Boundary loop
        split = pie.split()
        col = split.column(align=True)
        # row = col.row()

        # Toggle
        split = pie.split()
        col = split.column(align=True)

        # View
        split = pie.split()
        col = split.column(align=True)

class VIEW3D_MT_PIE_univ_edit(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        pie = self.layout.menu_pie()
        # pie.scale_x = 1.25
        # pie.scale_y = 2.0
        split = pie.split()

        # Angle
        col = split.column(align=True)
        row = col.row(align=True)
        row.scale_x = 1.2
        row.scale_y = 2.0
        row.operator("mesh.univ_select_flat", icon_value=icons.flat)

        split = pie.split()
        col = split.column(align=True)
        col.scale_x = 1.2
        col.scale_y = 2.0
        col.operator("view3d.univ_modifiers_toggle", text='Toggle Modifiers', icon='HIDE_OFF')

        # MultiLoop
        split = pie.split()
        col = split.column(align=True)
        row = col.row(align=True)
        row.scale_x = 1.2
        row.scale_y = 1.75
        if univ_pro:
            row.operator("mesh.univ_select_loop", icon_value=icons.loop_select)
        else:
            row.operator("mesh.loop_multi_select", text='Loop', icon_value=icons.loop_select).ring=False
        row.operator("mesh.loop_multi_select", text='Ring').ring=True
        row.operator("mesh.region_to_loop", text='Select to Loop', icon="SELECT_SET")

        col = col.column()
        col.separator()
        UNIV_PT_General.draw_uv_layers(col)
        # Boundary loop
        split = pie.split()
        col = split.column(align=True)
        row = col.row()

        # Toggle
        split = pie.split()
        col = split.column(align=True)

        # View
        split = pie.split()
        col = split.column(align=True)
        col.scale_x = 1.2
        col.scale_y = 2.0
        col.operator("mesh.univ_checker", icon_value=icons.checker)

        split = pie.split()

        col = split.column(align=True)

        split = pie.split()
        col = split.column(align=True)


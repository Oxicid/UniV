# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import bpy
from bpy.types import Panel, Menu, WorkSpaceTool

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
            row.scale_x = 2.1
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
        row.operator(prefix + '.univ_texel_density_get', icon_value=icons.td_get)
        row.operator(set_idname, icon_value=icons.td_set).td_preset_idx = -1
        row.prop(settings, 'texel_density', text='')
        row.operator(prefix + '.univ_select_texel_density', text='', icon_value=icons.arrow)
        if prefix == 'uv':
            row.popover(panel='UNIV_PT_td_presets_manager', text='', icon_value=icons.settings_a)
        else:
            row.popover(panel='UNIV_PT_td_presets_manager_view_3d', text='', icon_value=icons.settings_a)

        split = layer.split()
        row = split.row(align=True)
        for idx, preset in enumerate(settings.texels_presets):
            if idx and (idx+1) % 4 == 1:
                split = layer.split()
                row = split.row(align=True)
            row.operator(set_idname, text=preset.name).td_preset_idx = idx

    @staticmethod
    def draw_uv_layers(layout, ui_list='UNIV_UL_UV_LayersManager'):
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
            listtype_name=ui_list,
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
        col.popover(panel='UNIV_PT_layers_manager', text='', icon_value=icons.settings_a)

    def draw_header(self, context):
        layout = self.layout
        row = layout.split(factor=.5)
        row.popover(panel='UNIV_PT_GlobalSettings', text="", icon_value=icons.settings_b)
        row.label(text='UniV')

    def draw(self, context):
        layout = self.layout
        if prefs().use_csa_mods:
            layout.operator_context = 'INVOKE_DEFAULT'
        else:
            layout.operator_context = 'EXEC_DEFAULT'
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
        row = split.row(align=True)
        row.operator('uv.univ_stack', icon_value=icons.stack)
        if univ_pro:
            row.operator('uv.univ_select_similar', text='', icon_value=icons.arrow)

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
        row.operator('uv.univ_select_square_island', icon_value=icons.square).shape = 'SQUARE'
        row.operator('uv.univ_select_square_island', text='H-Rect',  icon_value=icons.horizontal_a).shape = 'HORIZONTAL'
        row.operator('uv.univ_select_square_island', text='V-Rect',  icon_value=icons.vertical_a).shape = 'VERTICAL'

        row = col_align.row(align=True)
        row.operator('uv.univ_select_by_area', text='Small', icon_value=icons.small).size_mode = 'SMALL'
        row.operator('uv.univ_select_by_area', text='Medium', icon_value=icons.medium).size_mode = 'MEDIUM'
        row.operator('uv.univ_select_by_area', text='Large', icon_value=icons.large).size_mode = 'LARGE'

        # Mark
        col_align = col.column(align=True)
        col_align.label(text='Mark')
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_pin', icon_value=icons.pin)

        split = col_align.split(align=True)
        split.operator('uv.univ_cut', icon_value=icons.cut)
        split.operator('uv.univ_seam_border', icon_value=icons.border_seam)

        # Other
        split = col_align.column()
        split.label(text='Other')

        row = split.row(align=True)
        row.scale_y = 1.35
        row.operator('uv.univ_batch_inspect', icon_value=icons.zero)
        row.popover(panel='UNIV_PT_BatchInspectSettings', text='', icon_value=icons.settings_a)

        row = split.row(align=True)
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

    @classmethod
    def poll(cls, context):
        return prefs().show_view_3d_panel

    def draw_header(self, context):
        layout = self.layout
        row = layout.split(factor=.5)
        row.popover(panel='UNIV_PT_GlobalSettings', text="", icon_value=icons.settings_b)
        row.label(text='UniV')

    def draw(self, context):
        layout = self.layout
        if prefs().use_csa_mods:
            layout.operator_context = 'INVOKE_DEFAULT'
        else:
            layout.operator_context = 'EXEC_DEFAULT'
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
        row.operator('mesh.univ_relax', icon_value=icons.relax)
        row.operator('mesh.univ_unwrap', icon_value=icons.unwrap)

        row = col_align.row(align=True)
        row.operator('mesh.univ_weld', icon_value=icons.weld)
        row.operator('mesh.univ_stitch', icon_value=icons.stitch)
        row = col_align.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.univ_stack', icon_value=icons.stack)
        if univ_pro:
            row.operator('mesh.univ_select_similar', text='', icon_value=icons.arrow)

        col_align.label(text='Transform')
        col_align.operator('mesh.univ_gravity', icon_value=icons.gravity)
        col_align.operator('mesh.univ_reset_scale', icon_value=icons.reset).axis = 'XY'

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

        col_align.label(text='Other')
        row = col_align.row(align=True)
        row.operator('mesh.univ_flatten', icon_value=icons.flatten)
        row.operator('mesh.univ_flatten_clean_up', icon_value=icons.remove, text='')

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

        row = layout.row(align=True, heading='Global Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        layout.prop(settings, 'padding', slider=True)
        layout.separator()
        layout.prop(settings, 'uv_layers_show')

        indent_px = 16
        split = layout.split(factor=indent_px / bpy.context.region.width)
        _ = split.column()
        col = split.column()
        col.active = settings.uv_layers_show
        col.prop(prefs(), 'enable_uv_layers_sync_borders_seam')

        layout.prop(prefs(), 'use_csa_mods')
        layout.operator('wm.univ_show_addon_preferences')


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

        if uvpm_exist := hasattr(context.scene, 'uvpm3_props'):
            layout.prop(settings, 'use_uvpm')

        row = layout.row(align=True, heading='Global Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        layout.prop(settings, 'padding', slider=True)
        layout.separator()

        if settings.use_uvpm:
            if uvpm_exist:
                self.draw_uvpm()
            else:
                layout.label(text='UVPackmaster not found')
            return

        if not bpy.app.version >= (3, 6, 0):
            layout.prop(settings, 'rotate', toggle=True)
        else:
            row = layout.row(align=True)
            row.prop(settings, 'shape_method', expand=True)

            row = layout.row(align=True)
            row.prop(settings, 'scale', toggle=True)
            row.prop(settings, 'rotate', toggle=True)

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

    def draw_uvpm(self):
        layout = self.layout
        settings = univ_settings()
        uvpm_settings = bpy.context.scene.uvpm3_props

        if hasattr(uvpm_settings, 'default_main_props'):
            uvpm_main_props = uvpm_settings.default_main_props
        else:
            uvpm_main_props = uvpm_settings

        row = layout.row(align=True)
        row.prop(settings, 'scale', toggle=True)
        row.prop(uvpm_main_props, 'rotation_enable', text='Rotation', toggle=True)
        row.prop(uvpm_main_props, 'flipping_enable', text='Flip', toggle=True)
        if settings.scale:
            row = layout.row(align=True)
            row.prop(uvpm_main_props, 'normalize_scale', text='Normalize', toggle=True)
            row.prop(uvpm_main_props, 'heuristic_allow_mixed_scales', text='Mixed Scale', toggle=True)
        if uvpm_main_props.rotation_enable:
            row = layout.row(align=True)
            row.prop(uvpm_main_props, 'pre_rotation_disable', text='Pre-Rotation Disable', toggle=True)
            subrow = row.column()
            subrow.scale_x = 0.8
            subrow.prop(uvpm_main_props, 'rotation_step', text='Step')

        layout.prop(uvpm_main_props, 'lock_overlapping_enable', text='Lock Overlaps')
        if uvpm_main_props.lock_overlapping_enable:
            row = layout.row(align=True)
            row.prop(uvpm_main_props, 'lock_overlapping_mode', expand=True)

        layout.prop(uvpm_main_props.numbered_groups_descriptors.lock_group, 'enable', text='Lock Groups')


class UNIV_PT_BatchInspectSettings(Panel):
    bl_idname = 'UNIV_PT_BatchInspectSettings'
    bl_label = 'Batch Inspect Settings'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_options = {"INSTANCED"}
    bl_category = "UniV"

    def draw(self, context):
        from .operators.inspect import Inspect
        settings = univ_settings()
        flags = settings.batch_inspect_flags

        def draw_tag_button(flag: Inspect):
            is_enabled = bool(flags & flag)
            row.operator('uv.univ_batch_inspect_flags',
                         text='',
                         icon='CHECKBOX_HLT' if is_enabled else 'CHECKBOX_DEHLT',  # noqa
                         depress=is_enabled).flag = flag

        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'
        col = layout.column(align=True)

        row = col.row(align=True)
        row.operator('uv.univ_check_overlap', icon_value=icons.overlap).check_mode = 'ALL'
        draw_tag_button(Inspect.Overlap)

        row = col.row(align=True)
        row.operator('uv.univ_check_overlap', text='Inexact', icon_value=icons.overlap).check_mode = 'INEXACT'
        draw_tag_button(Inspect.OverlapInexact)

        # row = col.row(align=True)
        # row.operator('uv.univ_check_overlap', text='Self', icon_value=icons.overlap)  # .check_mode = 'SELF'
        # draw_tag_button(Inspect.OverlapSelf)
        #
        # row = col.row(align=True)
        # row.operator('uv.univ_check_overlap', text='Trouble', icon_value=icons.overlap)  # .check_mode = 'TROUBLE'
        # draw_tag_button(Inspect.OverlapTroubleFace)
        #
        # row = col.row(align=True)
        # row.operator('uv.univ_check_overlap', text='By Material', icon_value=icons.overlap)  # .check_mode = 'MATERIAL'
        # draw_tag_button(Inspect.OverlapByMaterial)
        #
        # row = col.row(align=True)
        # row.operator('uv.univ_check_overlap', text='With Modifier', icon_value=icons.overlap)  # .check_mode = 'MODIFIER'
        # draw_tag_button(Inspect.OverlapWithModifier)

        col.separator()

        row = col.row(align=True)
        row.operator('uv.univ_check_over', icon_value=icons.over)
        draw_tag_button(Inspect.Over)

        row = col.row(align=True)
        row.operator('uv.univ_check_zero', icon_value=icons.zero)
        draw_tag_button(Inspect.Zero)

        row = col.row(align=True)
        row.operator('uv.univ_check_non_splitted', icon_value=icons.non_splitted)
        draw_tag_button(Inspect.NonSplitted)

        row = col.row(align=True)
        row.operator('uv.univ_check_flipped', icon_value=icons.flipped)
        draw_tag_button(Inspect.Flipped)

        row = col.row(align=True)
        row.operator('uv.univ_check_other', icon_value=icons.random)
        draw_tag_button(Inspect.Other)


class UNIV_UL_TD_PresetsManager(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, 'name', text='', emboss=False)
        row.prop(item, 'texel', text='TD', emboss=False)

class UNIV_UL_UV_LayersManager(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0, flt_flag=0):
        # TODO: Redraw if undo???
        settings = univ_settings()
        if index >= settings.uv_layers_size:
            return
        if flag := item.flag:  # noqa
            if flag == 2:
                layout.alert = True
            else:
                layout.active = False
        layout.prop(item, 'name', text='', emboss=False, icon='GROUP_UVS')  # noqa
        icon = 'RESTRICT_RENDER_OFF' if settings.uv_layers_active_render_idx == index else 'RESTRICT_RENDER_ON'
        layout.operator('mesh.univ_active_render_set', text='', icon=icon).idx = index


class UNIV_UL_UV_LayersManagerV2(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0, flt_flag=0):
        settings = univ_settings()
        if index >= settings.uv_layers_size:
            return
        if flag := item.flag:  # noqa
            if flag == 2:
                layout.alert = True
            else:
                layout.active = False
        layout.prop(item, 'name', text='', emboss=False)  # noqa
        icon = 'RESTRICT_RENDER_OFF' if settings.uv_layers_active_render_idx == index else 'RESTRICT_RENDER_ON'
        layout.operator('mesh.univ_active_render_set', text='', icon=icon).idx = index


class UNIV_PT_TD_PresetsManager(Panel):
    bl_label = 'Texel Density Presets Manager'
    bl_idname = 'UNIV_PT_td_presets_manager'
    bl_space_type = 'IMAGE_EDITOR'
    bl_options = {'INSTANCED'}
    bl_region_type = 'UI'
    bl_category = 'UniV'

    def draw(self, context):
        self.draw_ex(self.layout)

    @staticmethod
    def draw_ex(layout, ot_prefix='uv'):
        if prefs().use_csa_mods:
            layout.operator_context = 'INVOKE_DEFAULT'
        else:
            layout.operator_context = 'EXEC_DEFAULT'

        settings = univ_settings()
        layout.label(text=f"Texel Density: {round(settings.texel_density, 4)}")
        row = layout.row(align=True, heading='Global Size')
        row.prop(settings, 'size_x', text='')
        row.prop(settings, 'lock_size', text='', icon='LOCKED' if settings.lock_size else 'UNLOCKED')
        row.prop(settings, 'size_y', text='')

        layout.separator()
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator(ot_prefix + ".univ_calc_uv_area", icon_value=icons.area)
        row.operator(ot_prefix + ".univ_calc_uv_coverage", icon_value=icons.coverage)
        if ot_prefix == 'uv':
            col.operator("uv.univ_texel_density_from_texture")
        row = col.row(align=True)
        row.operator('uv.univ_texel_density_from_physical_size')
        row = row.split().row(align=True)
        row.scale_x = 0.7
        row.prop(settings, 'texture_physical_size', expand=True, text='')

        col.operator("mesh.univ_calc_udims_from_3d_area")
        layout.separator()

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

        td_idx = univ_settings().active_td_index
        if td_idx < 0:
            return

        td_presets = univ_settings().texels_presets
        if len(td_presets) >= td_idx + 1:
            preset = td_presets[td_idx]
            layout.prop(preset, 'texel')
            row = layout.row(align=True)
            row.prop(preset, 'size_x')
            row.prop(preset, 'size_y')


class UNIV_PT_TD_PresetsManager_VIEW3D(Panel):
    bl_label = 'Texel Density Presets Manager'
    bl_idname = 'UNIV_PT_td_presets_manager_view_3d'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'INSTANCED'}
    bl_category = 'UniV'

    def draw(self, context):
        UNIV_PT_TD_PresetsManager.draw_ex(self.layout, 'mesh')


class UNIV_PT_TD_LayersManager(Panel):
    bl_label = 'Layers Manager'
    bl_idname = 'UNIV_PT_layers_manager'
    bl_space_type = 'IMAGE_EDITOR'
    bl_options = {'INSTANCED'}
    bl_region_type = 'UI'
    bl_category = 'UniV'

    def draw(self, context):
        layout = self.layout
        if prefs().use_csa_mods:
            layout.operator_context = 'INVOKE_DEFAULT'
        else:
            layout.operator_context = 'EXEC_DEFAULT'

        settings = univ_settings()
        layout.operator('mesh.univ_fix_uvs', icon='EVENT_F')

        row = layout.row(align=True)
        row.prop(settings, 'copy_to_layers_from', text='')
        row.label(text='', icon_value=icons.shift)
        row.prop(settings, 'copy_to_layers_to', text='')
        row.separator(factor=0.35)
        row.operator('uv.univ_copy_to_layer')


class IMAGE_MT_PIE_univ_edit(Menu):
    bl_label = 'UniV Pie'

    # @classmethod
    # def poll(cls, context):
    #     return context.mode == 'EDIT'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        if univ_pro:
            pie.operator("uv.univ_select_flat", icon_value=icons.flat)
        else:
            pie.split()

        pie.operator("uv.univ_sync_uv_toggle", icon='UV_SYNC_SELECT')

        # Bottom
        col = pie.column(align=True)
        col.separator(factor=18)
        col.scale_x = 0.8

        row = col.row(align=True)
        row.scale_y = 1.35
        row.operator('uv.univ_select_by_vertex_count', text='Tris').polygone_type = 'TRIS'
        row.operator('uv.univ_select_by_vertex_count', text='Quad').polygone_type = 'QUAD'
        row.operator('uv.univ_select_by_vertex_count', text='N-Gone').polygone_type = 'NGONE'

        UNIV_PT_General.draw_uv_layers(col, 'UNIV_UL_UV_LayersManagerV2')

        # Upper
        pie.operator("uv.univ_toggle_pivot", icon='PIVOT_ACTIVE')
        # Left Upper
        pie.split()
        # Right Upper
        pie.operator("mesh.univ_checker", text='Toggle Checker', icon_value=icons.checker)
        # Left Bottom
        if univ_pro:
            pie.operator("uv.univ_select_loop", icon_value=icons.loop_select)
        else:
            pie.split()


class IMAGE_MT_PIE_univ_align(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        pie.operator('uv.univ_align_pie', text='Left', icon_value=icons.arrow_left).direction = 'LEFT'
        pie.operator('uv.univ_align_pie', text='Right', icon_value=icons.arrow_right).direction = 'RIGHT'
        pie.operator('uv.univ_align_pie', text='Bottom', icon_value=icons.arrow_bottom).direction = 'BOTTOM'
        pie.operator('uv.univ_align_pie', text='Upper', icon_value=icons.arrow_top).direction = 'UPPER'

        col = pie.column()
        col.scale_x = 1.2
        col.scale_y = 1.2
        row = col.row(align=True)
        row.alignment = 'CENTER'
        row.prop(univ_settings(), 'align_island_mode', expand=True, icon_only=True)
        col.separator(factor=0.2)
        row = col.row(align=True)
        row.prop(univ_settings(), 'align_mode', expand=True, icon_only=True)

        pie.operator('uv.univ_align_pie', text='Center', icon_value=icons.center).direction = 'CENTER'
        pie.operator('uv.univ_align_pie', text='Horizontal', icon_value=icons.horizontal_c).direction = 'HORIZONTAL'
        pie.operator('uv.univ_align_pie', text='Vertical', icon_value=icons.vertical_b).direction = 'VERTICAL'


class IMAGE_MT_PIE_univ_misc(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator('uv.univ_relax', icon_value=icons.relax)
        # Right
        pie.operator('uv.univ_unwrap', icon_value=icons.unwrap)
        # Bottom
        pie.operator('uv.univ_stack', icon_value=icons.stack)
        # Upper
        if univ_pro:
            pie.operator('uv.univ_rectify', icon_value=icons.rectify)
        else:
            pie.split()

        # Left Upper
        pie.operator('uv.univ_quadrify', icon_value=icons.quadrify)
        # Right Upper
        pie.operator('uv.univ_straight', icon_value=icons.straight)
        # Left Bottom
        pie.operator('uv.univ_weld', icon_value=icons.weld)
        # Right Bottom
        pie.operator('uv.univ_stitch', icon_value=icons.stitch)


class VIEW3D_MT_PIE_univ_misc(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator('mesh.univ_relax', icon_value=icons.relax)
        # Right
        pie.operator('mesh.univ_unwrap', icon_value=icons.unwrap)
        # Bottom
        pie.operator('mesh.univ_stack', icon_value=icons.stack)

        # Upper
        pie.split()

        # Left Upper
        pie.split()
        # pie.operator('uv.univ_quadrify', icon_value=icons.quadrify)
        # Right Upper
        pie.split()
        # pie.operator('uv.univ_straight', icon_value=icons.straight)
        # Left Bottom
        pie.operator('mesh.univ_weld', icon_value=icons.weld)
        # Right Bottom
        pie.operator('mesh.univ_stitch', icon_value=icons.stitch)


class VIEW3D_MT_PIE_univ_obj(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.split()

        # Right
        pie.operator("view3d.univ_modifiers_toggle", text='Toggle Modifiers', icon='HIDE_OFF')

        # Bottom
        split = pie.split()
        col = split.column(align=True)
        col.separator(factor=18)
        UNIV_PT_General.draw_uv_layers(col, 'UNIV_UL_UV_LayersManagerV2')

        # Upper
        pie.split()

        # Left Upper
        pie.split()

        # Right Upper
        pie.operator("mesh.univ_checker", text='Toggle Checker', icon_value=icons.checker)

        # Left Bottom
        pie.split()
        # Right Bottom
        pie.operator("wm.univ_workspace_toggle", icon_value=icons.unwrap)


class VIEW3D_MT_PIE_univ_edit(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        if univ_pro:
            pie.operator("mesh.univ_select_flat", icon_value=icons.flat)
        else:
            pie.operator("mesh.faces_select_linked_flat")
        # Right
        pie.operator("view3d.univ_modifiers_toggle", text='Toggle Modifiers', icon='HIDE_OFF')

        # Bottom

        col = pie.column()
        col.separator(factor=18)
        col.scale_x = 0.8

        row = col.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.loop_multi_select', text='Ring').ring = True
        row.operator('mesh.loop_to_region', text='Inner')

        row = col.row(align=True)
        row.scale_y = 1.35
        row.operator('mesh.univ_select_by_vertex_count', text='Tris').polygone_type = 'TRIS'
        row.operator('mesh.univ_select_by_vertex_count', text='Quad').polygone_type = 'QUAD'
        row.operator('mesh.univ_select_by_vertex_count', text='N-Gone').polygone_type = 'NGONE'

        UNIV_PT_General.draw_uv_layers(col, 'UNIV_UL_UV_LayersManagerV2')

        # Upper
        pie.operator("mesh.region_to_loop", text='To Loop', icon="SELECT_SET")

        # Left Upper
        pie.operator("mesh.select_nth", icon_value=icons.checker).offset = 1
        # Right Upper
        pie.operator("mesh.univ_checker", text="Toggle Checker", icon_value=icons.checker)
        # Left Bottom
        if univ_pro:
            pie.operator("mesh.univ_select_loop", icon_value=icons.loop_select)
        else:
            pie.operator("mesh.loop_multi_select", text='Loop').ring = False
        # Right Bottom
        pie.operator("wm.univ_workspace_toggle", icon_value=icons.unwrap)


class IMAGE_MT_PIE_univ_transform(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator('uv.univ_rotate', icon_value=icons.rotate)

        # Right
        pie.operator('uv.univ_flip', icon_value=icons.flip)

        # Bottom
        col = pie.column(align=True)

        col.separator(factor=12)
        col.scale_x = 1.35
        col.scale_y = 1.35

        row = col.row(align=True)
        row.operator('uv.univ_crop', icon_value=icons.crop).axis = 'XY'
        row.operator('uv.univ_crop', text='', icon_value=icons.x).axis = 'X'
        row.operator('uv.univ_crop', text='', icon_value=icons.y).axis = 'Y'

        row = col.row(align=True)
        row.operator('uv.univ_fill', icon_value=icons.fill).axis = 'XY'
        row.operator('uv.univ_fill', text='', icon_value=icons.x).axis = 'X'
        row.operator('uv.univ_fill', text='', icon_value=icons.y).axis = 'Y'

        col.separator(factor=0.35)

        row = col.row(align=True)
        row.operator('uv.univ_home', icon_value=icons.home)
        row.operator('uv.univ_shift', icon_value=icons.shift)

        row = col.row(align=True)
        row.operator('uv.univ_random', icon_value=icons.random)

        # Upper
        pie.operator('uv.univ_orient', icon_value=icons.orient).edge_dir = 'BOTH'

        # Left Upper
        pie.operator('uv.univ_orient', text='H-Orient', icon_value=icons.horizontal_a).edge_dir = 'HORIZONTAL'

        # Right Upper
        pie.operator('uv.univ_orient', text='V-Orient', icon_value=icons.vertical_a).edge_dir = 'VERTICAL'

        # Left Bottom
        pie.operator('uv.univ_sort', icon_value=icons.sort)

        # Right Bottom
        pie.operator('uv.univ_distribute', icon_value=icons.distribute)


class IMAGE_MT_PIE_univ_texel(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator('uv.univ_adjust_td', icon_value=icons.adjust)

        # Right
        pie.operator('uv.univ_normalize', icon_value=icons.normalize)

        # Bottom
        split = pie.split()
        col = split.column()
        col.separator(factor=18)
        row = col.row(align=True)
        settings = univ_settings()
        row.prop(settings, 'texel_density')
        row.operator('uv.univ_select_texel_density', text='', icon_value=icons.arrow)
        row.popover(panel='UNIV_PT_td_presets_manager', text='', icon_value=icons.settings_a)

        split = col.split()
        row = split.row(align=True)
        for idx, preset in enumerate(settings.texels_presets):
            if idx and (idx+1) % 4 == 1:
                split = col.split()
                row = split.row(align=True)
            row.operator('uv.univ_texel_density_set', text=preset.name).td_preset_idx = idx

        # Upper
        pie.operator('uv.univ_reset_scale', icon_value=icons.reset)

        # Left Upper
        pie.operator('uv.univ_calc_uv_area', icon_value=icons.area)

        # Right Upper
        pie.operator('uv.univ_calc_uv_coverage', icon_value=icons.coverage)

        # Left Bottom
        pie.operator('uv.univ_texel_density_get', icon_value=icons.td_get)

        # Right Bottom
        pie.operator('uv.univ_texel_density_set', icon_value=icons.td_set).td_preset_idx = -1


class VIEW3D_MT_PIE_univ_texel(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator('mesh.univ_adjust_td', icon_value=icons.adjust)

        # Right
        pie.operator('mesh.univ_normalize', icon_value=icons.normalize)

        # Bottom
        split = pie.split()
        col = split.column()
        col.separator(factor=18)
        row = col.row(align=True)
        settings = univ_settings()
        row.prop(settings, 'texel_density')
        row.operator('mesh.univ_select_texel_density', text='', icon_value=icons.arrow)
        row.popover(panel='UNIV_PT_td_presets_manager_view_3d', text='', icon_value=icons.settings_a)

        split = col.split()
        row = split.row(align=True)
        for idx, preset in enumerate(settings.texels_presets):
            if idx and (idx+1) % 4 == 1:
                split = col.split()
                row = split.row(align=True)
            row.operator('mesh.univ_texel_density_set', text=preset.name).td_preset_idx = idx

        # Upper
        pie.operator('mesh.univ_reset_scale', icon_value=icons.reset)

        # Left Upper
        pie.operator('mesh.univ_calc_uv_area', icon_value=icons.area)

        # Right Upper
        pie.operator('mesh.univ_calc_uv_coverage', icon_value=icons.coverage)

        # Left Bottom
        pie.operator('mesh.univ_texel_density_get', icon_value=icons.td_get)

        # Right Bottom
        pie.operator('mesh.univ_texel_density_set', icon_value=icons.td_set).td_preset_idx = -1


class VIEW3D_MT_PIE_univ_favorites_edit(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator("mesh.univ_weld", icon_value=icons.weld)
        # Right
        pie.operator("mesh.univ_cut", icon_value=icons.cut)

        # Bottom
        split = pie.split()
        col = split.column(align=True)
        col.separator(factor=18)
        col.menu_contents("SCREEN_MT_user_menu")

        # Upper
        pie.operator("mesh.univ_gravity", icon_value=icons.gravity)
        # Left Upper
        pie.operator("mesh.univ_relax", icon_value=icons.relax)
        # Right Upper
        pie.operator("mesh.univ_unwrap", icon_value=icons.unwrap)
        # Left Bottom
        pie.operator("mesh.univ_stack", icon_value=icons.stack)
        # Right Bottom
        pie.operator("mesh.univ_angle", icon_value=icons.border_by_angle)


class IMAGE_MT_PIE_univ_favorites_edit(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator("uv.univ_weld", icon_value=icons.weld)
        # Right
        pie.operator("uv.univ_cut", icon_value=icons.cut)

        # Bottom
        split = pie.split()
        col = split.column(align=True)
        col.separator(factor=18)
        col.menu_contents("SCREEN_MT_user_menu")

        # Upper
        pie.operator("uv.univ_orient", icon_value=icons.orient)
        # Left Upper
        pie.operator("uv.univ_relax", icon_value=icons.relax)
        # Right Upper
        pie.operator("uv.univ_unwrap", icon_value=icons.unwrap)
        # Left Bottom
        pie.operator("uv.univ_stack", icon_value=icons.stack)
        # Right Bottom
        pie.operator("uv.univ_pin", icon_value=icons.pin)


class VIEW3D_MT_PIE_univ_projection(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator("mesh.univ_normal", icon_value=icons.normal)
        # Right
        pie.operator("mesh.univ_box_project", icon_value=icons.box)

        # Bottom
        pie.split()
        # Upper
        if univ_pro:
            pie.operator("mesh.univ_transfer", icon_value=icons.transfer)
        else:
            pie.split()

        # Left Upper
        pie.operator("mesh.univ_smart_project", icon_value=icons.smart)
        # Right Upper
        pie.operator("mesh.univ_view_project", icon_value=icons.view)
        # Left Bottom
        # Right Bottom


class IMAGE_MT_PIE_univ_inspect(Menu):
    bl_label = 'UniV Pie'

    def draw(self, _context):
        layout = self.layout
        layout.operator_context = 'EXEC_DEFAULT'

        pie = layout.menu_pie()

        # Left
        pie.operator("uv.univ_check_overlap", icon_value=icons.overlap)
        # Right
        pie.operator("uv.univ_check_non_splitted", icon_value=icons.non_splitted)
        # Bottom
        pie.split()
        # Upper
        pie.operator("uv.univ_batch_inspect", icon_value=icons.zero)
        # Left Upper
        pie.operator('uv.univ_check_over', icon_value=icons.over)
        # Right Upper
        pie.operator("uv.univ_check_other", icon_value=icons.random)
        # Left Bottom
        pie.operator("uv.univ_check_zero", icon_value=icons.zero)
        # Right Bottom
        pie.operator("uv.univ_check_flipped", icon_value=icons.flipped)


class UNIV_WT_object_VIEW3D(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'OBJECT'
    bl_idname = 'tool.univ'
    bl_description = ''
    bl_label = 'UniV'
    bl_icon = os.path.join(os.path.dirname(__file__), 'icons', 'univ')
    bl_keymap = ()

    # @staticmethod
    # def draw_settings(context, layout, tool):
    # if prefs().use_csa_mods:
    #     layout.operator_context = 'INVOKE_DEFAULT'
    # else:
    #     layout.operator_context = 'EXEC_DEFAULT'
    #     col = layout.column(align=True)
    #
    #     col_align = col.column(align=True)
    #     col_align.label(text='Mark')
    #     split = col_align.split(align=True)
    #     split.operator('mesh.univ_cut', icon_value=icons.cut)
    #     split.operator('mesh.univ_seam_border', icon_value=icons.border_seam)


class UNIV_WT_edit_VIEW3D(WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'tool.univ'
    bl_description = ''
    bl_label = 'UniV'
    bl_icon = os.path.join(os.path.dirname(__file__), 'icons', 'univ')
    bl_keymap = ()

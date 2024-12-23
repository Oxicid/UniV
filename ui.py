# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.types import Panel
from .preferences import univ_settings

REDRAW_UV_LAYERS = True

class UNIV_PT_General(Panel):
    bl_label = ''
    bl_idname = 'UNIV_PT_General'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

    @staticmethod
    def draw_align_buttons(where):
        def ly_wide_text_op(layout, direction, *, text):
            row = layout.row(align=True)
            row.operator('uv.univ_align', text=text).direction = direction

        where.alignment = 'EXPAND'
        colMain = where.column(align=True)
        rowTop = colMain.row(align=True)

        ly_wide_text_op(rowTop, 'LEFT_UPPER', text='↖')
        ly_wide_text_op(rowTop.row(), 'UPPER', text='↑')
        ly_wide_text_op(rowTop, 'RIGHT_UPPER', text='↗')
        ##
        rowMiddle = colMain.row().row(align=True)
        ly_wide_text_op(rowMiddle, 'LEFT', text='← ')
        rowMidMiddle = rowMiddle.row().row(align=True)

        ly_wide_text_op(rowMidMiddle, 'HORIZONTAL', text='—')
        ly_wide_text_op(rowMidMiddle, 'CENTER', text='+')
        ly_wide_text_op(rowMidMiddle, 'VERTICAL', text='|')
        ly_wide_text_op(rowMiddle, 'RIGHT', text=' →')
        ##
        rowBottom = colMain.row(align=True)
        ly_wide_text_op(rowBottom, 'LEFT_BOTTOM', text='↙')
        ly_wide_text_op(rowBottom.row(), 'BOTTOM', text='↓')
        ly_wide_text_op(rowBottom, 'RIGHT_BOTTOM', text='↘')

    @staticmethod
    def draw_texel_density(layer, prefix):
        settings = univ_settings()
        split = layer.split(align=True)
        row = split.row(align=True)
        set_idname = prefix + '.univ_texel_density_set'
        row.operator(set_idname).custom_texel = -1.0
        row.operator(prefix + '.univ_texel_density_get')
        row.prop(settings, 'texel_density', text='')
        row.operator(prefix + '.univ_select_texel_density', text='', icon='RESTRICT_SELECT_OFF')
        row.popover(panel='UNIV_PT_td_presets_manager', text='', icon='SETTINGS')

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
        # print(f'{REDRAW_UV_LAYERS=}')
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
        col.separator()
        col.operator('mesh.univ_move_up', icon='TRIA_UP', text='')
        col.operator('mesh.univ_move_down', icon='TRIA_DOWN', text='')

    def draw_header(self, context):
        layout = self.layout
        row = layout.split(factor=.35)
        row.popover(panel='UNIV_PT_GlobalSettings', text="", icon='PREFERENCES')
        row.label(text='UniV')

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        col = layout.column(align=True)

        col.label(text='Transform')
        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_crop').axis = 'XY'
        row = split.row(align=True)
        row.operator('uv.univ_crop', text='X').axis = 'X'
        row.operator('uv.univ_crop', text='Y').axis = 'Y'

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_fill').axis = 'XY'
        row = split.row(align=True)
        row.operator('uv.univ_fill', text='X').axis = 'X'
        row.operator('uv.univ_fill', text='Y').axis = 'Y'

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_orient').edge_dir = 'BOTH'
        row = split.row(align=True)
        row.operator('uv.univ_orient', text='→').edge_dir = 'HORIZONTAL'
        row.operator('uv.univ_orient', text='↑').edge_dir = 'VERTICAL'

        col_for_align = col.column()
        col_for_align.separator(factor=0.5)
        self.draw_align_buttons(col_for_align)

        col = layout.column()
        col_align = col.column(align=True)
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_rotate')
        row.operator('uv.univ_flip')

        split = col_align.split(align=True)
        split.operator('uv.univ_sort')
        row = split.row(align=True)
        row.operator('uv.univ_distribute')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_home')
        row.operator('uv.univ_shift')

        split = col_align.split(align=True)
        split.operator('uv.univ_random')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_adjust_td')
        row.operator('uv.univ_normalize')

        self.draw_texel_density(col_align, 'uv')

        # Pack
        col_align = col.column(align=True)
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.scale_y = 1.5
        # row.scale_x = 2
        row.operator('uv.univ_pack')
        row.popover(panel='UNIV_PT_PackSettings', text="", icon='SETTINGS')

        # Misc
        col_align = col.column(align=True)

        col_align.label(text='Misc')
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_quadrify')
        row.operator('uv.univ_straight')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_relax')
        row.operator('uv.univ_unwrap')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_weld')
        row.operator('uv.univ_stitch')

        split = col_align.split(align=True)
        split.operator('uv.univ_pin', icon='PINNED')

        split = col_align.split(align=True)
        split.scale_y = 1.5
        split.operator('uv.univ_stack')

        # Select
        col_align.label(text='Select')
        col_align = col.column(align=True)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_grow')
        row.operator('uv.univ_select_edge_grow')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_linked')
        row.operator('uv.univ_select_by_cursor')

        row = col_align.split(align=True)
        row.operator('uv.univ_select_border')

        split = col_align.split(factor=0.65, align=True)
        split.operator('uv.univ_select_border_edge_by_angle').edge_dir = 'BOTH'
        row = split.row(align=True)
        row.operator('uv.univ_select_border_edge_by_angle', text='H').edge_dir = 'HORIZONTAL'
        row.operator('uv.univ_select_border_edge_by_angle', text='V').edge_dir = 'VERTICAL'

        split = col_align.split(factor=0.65, align=True)
        split.operator('uv.univ_select_square_island').shape = 'SQUARE'
        row = split.row(align=True)
        row.operator('uv.univ_select_square_island', text='H').shape = 'HORIZONTAL'
        row.operator('uv.univ_select_square_island', text='V').shape = 'VERTICAL'

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_by_area', text='Small').size_mode = 'SMALL'
        row.operator('uv.univ_select_by_area', text='Medium').size_mode = 'MEDIUM'
        row.operator('uv.univ_select_by_area', text='Large').size_mode = 'LARGE'

        # Inspect
        col_align = col.column(align=True)
        col_align.label(text='Inspect')

        split = col_align.split(align=True)
        split.operator('uv.univ_check_zero')
        split.operator('uv.univ_check_flipped')

        split = col_align.split(align=True)
        split.operator('uv.univ_check_non_splitted')
        split.operator('uv.univ_check_overlap')

        # Seam
        col_align = col.column(align=True)
        col_align.label(text='Seam')
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_cut')
        split.operator('uv.univ_seam_border')

        layout.label(text='Texture')
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')

        self.draw_uv_layers(self.layout)


class UNIV_PT_General_VIEW_3D(UNIV_PT_General):
    bl_idname = 'UNIV_PT_General_VIEW3D'
    bl_space_type = 'VIEW_3D'

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        col = layout.column(align=True)

        col_align = col.column(align=True)
        col_align.label(text='Seam')
        split = col_align.split(align=True)
        split.operator('mesh.univ_cut')
        split.operator('mesh.univ_seam_border')

        split = col_align.split(align=True)
        split.operator('mesh.univ_angle')

        col_align.label(text='Project')
        row = col_align.row(align=True)
        row.operator('mesh.univ_normal')
        row.operator('mesh.univ_box_project')

        col_align.label(text='Stack')
        row = col_align.row(align=True)
        row.operator('mesh.univ_stack', text='Stack')

        col_align.label(text='Transform')
        col_align.operator('mesh.univ_gravity')

        row = col_align.row(align=True)
        row.operator('mesh.univ_adjust_td')
        row.operator('mesh.univ_normalize')

        self.draw_texel_density(col_align, 'mesh')

        col_align.label(text='Texture')
        row = col_align.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')

        self.draw_uv_layers(layout)


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

        UNIV_PT_GlobalSettings.draw_global_settings(layout)

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

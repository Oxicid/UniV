# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.types import Panel

class UNIV_PT_General(Panel):
    bl_label = "UniV"
    bl_idname = 'UNIV_PT_General'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"
    bl_options = {'HIDE_HEADER'}

    @staticmethod
    def draw_align_buttons(where, *, alignment='EXPAND', scale_x=1.0):
        def ly_wide_text_op(layout, direction, *, text):
            row = layout.row(align=True)
            row.operator('uv.univ_align', text=text).direction = direction

        def ly_mid_mid_text_op(layout, direction, *, text, opt='uv.univ_align'):
            row = layout.row(align=True)
            row.operator(opt, text=text).direction = direction

        where.alignment = alignment
        colMain = where.column(align=True)
        colMain.scale_x = scale_x
        rowTop = colMain.row(align=True)

        ly_wide_text_op(rowTop, 'LEFT_UPPER', text='↖')
        ly_wide_text_op(rowTop.row(), 'UPPER', text='↑')
        ly_wide_text_op(rowTop, 'RIGHT_UPPER', text='↗')
        ##
        rowMiddle = colMain.row().row(align=True)
        ly_mid_mid_text_op(rowMiddle, 'LEFT', text='← ')
        rowMidMiddle = rowMiddle.row().row(align=True)
        ly_mid_mid_text_op(rowMidMiddle, 'HORIZONTAL', text='—')
        ly_mid_mid_text_op(rowMidMiddle, 'CENTER', text='+')
        ly_mid_mid_text_op(rowMidMiddle, 'VERTICAL', text='|')
        ly_mid_mid_text_op(rowMiddle, 'RIGHT', text=' →')
        ##
        rowBottom = colMain.row(align=True)
        ly_wide_text_op(rowBottom, 'LEFT_BOTTOM', text='↙')
        ly_wide_text_op(rowBottom.row(), 'BOTTOM', text='↓')
        ly_wide_text_op(rowBottom, 'RIGHT_BOTTOM', text='↘')

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
        row.operator('uv.univ_adjust_td')
        row.operator('uv.univ_normalize')

        split = col_align.split(align=True)
        split.operator('uv.univ_home')

        split = col_align.split(align=True)
        split.operator('uv.univ_random')

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

        row = col_align.split(align=True)#.row(align=True)
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
        split.operator('mesh.univ_seam_border')

        self.layout.label(text='Texture')
        row = self.layout.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')
        # row.alignment = 'RIGHT'


class UNIV_PT_General_VIEW_3D(Panel):
    bl_label = "UniV"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "UniV"

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
        col_align.operator('mesh.univ_orient_view3d')

        row = col_align.row(align=True)
        row.operator('mesh.univ_adjust_td')
        row.operator('mesh.univ_normalize')

        col_align.label(text='Texture')
        row = col_align.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')
        row.alignment = 'RIGHT'

class UNIV_PT_PackSettings(Panel):
    bl_idname = 'UNIV_PT_PackSettings'
    bl_label = 'Pack Settings'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_options = {"INSTANCED"}
    bl_category = "UniV"

    def draw(self, context):
        settings = context.scene.univ_settings  # noqa

        layout = self.layout
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

        layout.separator()

        row = layout.row(align=True)
        row.alignment = 'LEFT'
        row.prop(settings, 'texture_size')
        row.prop(settings, 'padding', slider=True)

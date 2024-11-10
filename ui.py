# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

from bpy.types import Panel

class UNIV_PT_General(Panel):
    bl_label = "UniV"
    bl_idname = 'UNIV_PT_General'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

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
        split.operator('uv.univ_crop', text='Crop').axis = 'XY'
        row = split.row(align=True)
        row.operator('uv.univ_crop', text='X').axis = 'X'
        row.operator('uv.univ_crop', text='Y').axis = 'Y'

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_fill', text='Fill').axis = 'XY'
        row = split.row(align=True)
        row.operator('uv.univ_fill', text='X').axis = 'X'
        row.operator('uv.univ_fill', text='Y').axis = 'Y'

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_orient', text='Orient').edge_dir = 'BOTH'
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
        row.operator('uv.univ_rotate', text='Rotate')
        row.operator('uv.univ_flip', text='Flip')

        split = col_align.split(align=True)
        split.operator('uv.univ_sort', text='Sort')
        row = split.row(align=True)
        row.operator('uv.univ_distribute', text='Distribute')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_adjust_td', text='Adjust')
        row.operator('uv.univ_normalize', text='Normalize')

        split = col_align.split(align=True)
        split.operator('uv.univ_home', text='Home')

        split = col_align.split(align=True)
        split.operator('uv.univ_random', text='Random')

        split = col_align.split(align=True)
        split.scale_y = 1.5
        split.operator('uv.univ_pack', text='Pack')

        # Misc
        col_align = col.column(align=True)

        col_align.label(text='Misc')
        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_quadrify', text='Quadrify')
        row.operator('uv.univ_straight', text='Straight')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_relax', text='Relax')
        row.operator('uv.univ_unwrap', text='Unwrap')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_weld', text='Weld')
        row.operator('uv.univ_stitch', text='Stitch')

        split = col_align.split(align=True)
        split.operator('uv.univ_pin', text='Pin', icon='PINNED')

        split = col_align.split(align=True)
        split.scale_y = 1.5
        split.operator('uv.univ_stack', text='Stack')

        # Select
        col_align.label(text='Select')
        col_align = col.column(align=True)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_grow', text='Grow')
        row.operator('uv.univ_select_edge_grow', text='Edge Grow')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_linked', text='Linked')
        row.operator('uv.univ_select_by_cursor', text='Cursor')

        row = col.split().row(align=True)
        row.operator('uv.univ_select_border', text='Border')

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_select_border_edge_by_angle', text='Border by Angle').edge_dir = 'BOTH'
        row = split.row(align=True)
        row.operator('uv.univ_select_border_edge_by_angle', text='H').edge_dir = 'HORIZONTAL'
        row.operator('uv.univ_select_border_edge_by_angle', text='V').edge_dir = 'VERTICAL'

        split = col.split(factor=0.65, align=True)
        split.operator('uv.univ_select_square_island', text='Square').shape = 'SQUARE'
        row = split.row(align=True)
        row.operator('uv.univ_select_square_island', text='H').shape = 'HORIZONTAL'
        row.operator('uv.univ_select_square_island', text='V').shape = 'VERTICAL'

        # Inspect
        col_align = col.column(align=True)
        col_align.label(text='Inspect')

        split = col_align.split(align=True)
        split.operator('uv.univ_check_zero', text='Zero')
        split.operator('uv.univ_check_flipped', text='Flipped')

        split = col_align.split(align=True)
        split.row(align=True).operator('uv.univ_check_non_splitted', text='Non-Splitted')

        # Seam
        col_align = col.column(align=True)
        col_align.label(text='Seam')
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_cut', text='Cut')

        self.layout.label(text='Texture')
        row = self.layout.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker', text='Checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')
        row.alignment = 'RIGHT'


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
        split.operator('mesh.univ_cut', text='Cut')
        split.operator('mesh.univ_angle', text='Angle')

        layout.label(text='Project')
        row = self.layout.row(align=True)
        row.operator('mesh.univ_normal', text='Normal')
        row.operator('mesh.univ_box_project', text='Box')

        layout.label(text='Stack')
        row = self.layout.row(align=True)
        row.operator('mesh.univ_stack', text='Stack')

        layout.label(text='Transform')
        layout.operator('mesh.univ_orient_view3d', text='Orient')

        row = layout.row(align=True)
        row.operator('mesh.univ_adjust_td', text='Adjust')
        row.operator('mesh.univ_normalize', text='Normalize')

        layout.label(text='Texture')
        row = self.layout.row(align=True)
        row.scale_y = 1.5
        row.operator('mesh.univ_checker', text='Checker')
        row.operator('wm.univ_checker_cleanup', text='', icon='TRASH')
        row.alignment = 'RIGHT'

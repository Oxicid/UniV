# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

from bpy.types import Panel

class UNIV_PT_General(Panel):
    bl_label = "UniV"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

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

        col_align = col.column(align=True)
        col_align.separator(factor=0.35)
        row = col_align.row(align=True)
        col = row.column(align=True)
        col.operator('uv.univ_align', text='↖').direction = 'LEFT_UPPER'
        col.operator('uv.univ_align', text='← ').direction = 'LEFT'
        col.operator('uv.univ_align', text='↙').direction = 'LEFT_BOTTOM'

        col = row.column(align=True)
        col.operator('uv.univ_align', text='↑').direction = 'UPPER'
        col.operator('uv.univ_align', text='+').direction = 'CENTER'
        col.operator('uv.univ_align', text='↓').direction = 'BOTTOM'

        col = row.column(align=True)
        col.operator('uv.univ_align', text='↗').direction = 'RIGHT_UPPER'
        col.operator('uv.univ_align', text=' →').direction = 'RIGHT'
        col.operator('uv.univ_align', text='↘').direction = 'RIGHT_BOTTOM'

        row_tr = col_align.row(align=True)
        col = row_tr.column(align=True)

        row = col.row(align=True)
        row.operator('uv.univ_align', text='—').direction = 'HORIZONTAL'
        row.operator('uv.univ_align', text='|').direction = 'VERTICAL'

        col_align = col.column(align=True)
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_rotate', text='Rotate')
        row.operator('uv.univ_flip', text='Flip')

        split = col_align.split(align=True)
        split.operator('uv.univ_sort', text='Sort')
        row = split.row(align=True)
        row.operator('uv.univ_distribute', text='Distribute')

        split = col_align.split(align=True)
        split.operator('uv.univ_home', text='Home')

        split = col_align.split(align=True)
        split.operator('uv.univ_random', text='Random')

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
        split.operator('mesh.univ_stack', text='Stack')

        # Select
        col_align.label(text='Select')
        col_align = col.column(align=True)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_linked', text='Linked')
        row.operator('uv.univ_single', text='Single')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_by_cursor', text='Cursor')
        row.operator('uv.univ_select_view', text='View')

        row = col.split().row(align=True)
        row.operator('uv.univ_select_border', text='Border')
        row.operator('uv.univ_select_inner', text='Inner')

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
        split.operator('uv.univ_select_zero', text='Zero')
        split.operator('uv.univ_select_flipped', text='Flipped')

        # Seam
        col_align = col.column(align=True)
        col_align.label(text='Seam')
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_cut', text='Cut')

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

        self.layout.label(text='Project')
        row = self.layout.row(align=True)
        row.operator('mesh.univ_plane', text='Plane')
        row.operator('mesh.univ_box_project', text='Box')

        self.layout.label(text='Stack')
        row = self.layout.row(align=True)
        row.operator('mesh.univ_stack', text='Stack')

        self.layout.label(text='Transform')
        self.layout.operator('mesh.univ_orient_view3d', text='Orient')

"""
Created by Oxicid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from bpy.types import Panel

class UNIV_PT_General(Panel):
    bl_label = "UniV"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        box = layout.box()
        col = box.column(align=True)

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

        # Select
        col_align = col.column(align=True)
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_select_linked', text='Linked')
        row.operator('uv.univ_select_view', text='View')

        split = col_align.split(align=True)
        split.operator('uv.univ_single', text='Single')

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

        # Quadrify
        col_align = col.column(align=True)
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_quad', text='Quad')
        row.operator('uv.univ_straight', text='Straight')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_relax', text='Relax')

        # Inspect
        col_align = col.column(align=True)
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('uv.univ_select_zero', text='Zero')
        split.operator('uv.univ_select_flipped', text='Flipped')

        split = col_align.split(align=True)
        row = split.row(align=True)
        row.operator('uv.univ_weld', text='Weld')
        row.operator('uv.univ_stitch', text='Stitch')

        # Seam
        col_align = col.column(align=True)
        col_align.separator(factor=0.35)

        split = col_align.split(align=True)
        split.operator('mesh.univ_cut', text='Cut')
        split.operator('mesh.univ_angle', text='Angle')

class UNIV_PT_General_VIEW_3D(Panel):
    bl_label = "UniV"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "UniV"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        box = layout.box()
        col = box.column(align=True)

        col_align = col.column(align=True)
        split = col_align.split(align=True)
        split.operator('mesh.univ_cut', text='Cut')
        split.operator('mesh.univ_angle', text='Angle')

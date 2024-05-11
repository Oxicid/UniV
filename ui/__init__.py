from bpy.types import Panel

class UNIV_PT_General(Panel):
    bl_label = " "
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UniV"

    # def draw_header(self, context):

    def draw(self, context):
        layout = self.layout
        layout.operator_context = "INVOKE_DEFAULT"
        box = layout.box()
        col = box.column(align=True)

        # row = col.row(align=True)
        # row.operator('uv.univ_crop', text="Crop")
        # row.operator('uv.univ_fill', text="Fill")
        #
        # row = col.row(align=True)
        # row.operator('uv.univ_align_edge', text="Align Edge")
        #
        # row = col.row(align=True)
        # row.operator('uv.univ_align_world', text="Align World")

        col_align = col.column(align=True)

        row = col_align.row(align=True)
        col = row.column(align=True)
        col.operator('uv.univ_align', text="↖").direction = 'LEFT_UPPER'
        col.operator('uv.univ_align', text="← ").direction = 'LEFT'
        col.operator('uv.univ_align', text="↙").direction = 'LEFT_BOTTOM'

        col = row.column(align=True)
        col.operator('uv.univ_align', text="↑").direction = 'UPPER'
        col.operator('uv.univ_align', text="+").direction = 'CENTER'
        col.operator('uv.univ_align', text="↓").direction = 'BOTTOM'

        col = row.column(align=True)
        col.operator('uv.univ_align', text="↗").direction = 'RIGHT_UPPER'
        col.operator('uv.univ_align', text=" →").direction = 'RIGHT'
        col.operator('uv.univ_align', text="↘").direction = 'RIGHT_BOTTOM'

        row_tr = col_align.row(align=True)
        col = row_tr.column(align=True)
        # col.scale_x = 0.5
        row = col.row(align=True)
        row.operator('uv.univ_align', text="—").direction = 'HORIZONTAL'
        row.operator('uv.univ_align', text="|").direction = 'VERTICAL'

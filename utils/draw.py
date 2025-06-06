# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import gpu
import typing

from mathutils import Vector, Color
from bmesh.types import BMLoop
from bl_math import clamp


def rgb_to_hex(rgb):
    return f"#{int(clamp(rgb[0]) * 255.0):02x}" \
           f"{int(clamp(rgb[1]) * 255.0):02x}" \
           f"{int(clamp(rgb[2]) * 255.0):02x}"


def hex_to_rgb(hexcode):
    import binascii
    unhex = binascii.unhexlify(hexcode[1:])
    assert len(unhex) == 3, f"Expected hexcode size - 7, given size - {len(hexcode)}"
    return Color(unhex) / 255

def hsv_to_rgb(h, s, v):
    """Saturate is mutable"""
    # Get from https://stackoverflow.com/a/31628808/21538444
    import numpy as np
    shape = h.shape
    i = np.int_(h*6.)
    f = h*6.-i

    q = f
    t = 1.0 - f
    i = np.ravel(i)
    f = np.ravel(f)
    i %= 6

    t = np.ravel(t)
    q = np.ravel(q)

    clist = (1-s * np.vstack([np.zeros_like(f), np.ones_like(f), q, t])) * v

    # 0:v 1:p 2:q 3:t
    order = np.array([[0, 3, 1], [2, 0, 1], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:, None]]

    return rgb.reshape(shape+(3,))

def color_for_groups(groups):
    """Return flat colors by group"""
    import numpy as np
    np.random.seed((id(groups)+2) % np.iinfo(np.int32).max)

    groups_size = len(groups)
    h = np.random.uniform(low=0.0, high=1.0, size=groups_size)
    s = np.random.uniform(low=0.8, high=0.8, size=groups_size)
    v = np.random.uniform(low=0.8, high=0.8, size=groups_size)

    rgb = hsv_to_rgb(h, s, v)

    first_and_end_point = 2
    return np.repeat(rgb, [len(g)*first_and_end_point for g in groups], axis=0).tolist()  # TODO: Bug report np.ndarray for color, incorrect work


class UNIV_OT_Draw_Test(bpy.types.Operator):
    bl_idname = 'uv.univ_draw_test'
    bl_label = 'Draw Test'
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        self.mouse_position = ()
        self.view = None
        self.handler = None
        self.area = None

        self.shader = None
        self.shader_smooth_color = None

        self.batch = None
        self.batch_smooth_color = None
        self.batch_smooth_color_2 = None

        self.points = ()
        self.mid_points = ()
        self.texts = ()

    def invoke(self, context, event):
        self.area = context.area
        self.view = context.region.view2d
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.shader_smooth_color = gpu.shader.from_builtin('SMOOTH_COLOR')
        self.register_draw()
        from .. import types

        self.umeshes = types.UMeshes()
        self.umesh = self.umeshes[0]
        self.uv = self.umesh.uv

        self.test_invoke(event)

        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        try:
            return self.modal_ex(context, event)
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            self.umeshes.silent_update()
            self.exit()
            return {'FINISHED'}

    def modal_ex(self, _context, event):
        # print()
        # print(f'{event.type = }')
        # print(f'{event.value = }')

        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            self.test(event)
            self.area.tag_redraw()

        if event.type == 'LEFTMOUSE':
            self.test_invoke(event)
            self.area.tag_redraw()

        if event.type in ('ESC', 'RIGHTMOUSE'):
            return self.exit()

        return {'RUNNING_MODAL'}

    def test_invoke(self, _event):
        from .. import types
        umesh = self.umeshes[0]
        groups = types.LoopGroup.calc_dirt_loop_groups(umesh)
        self.calc_from_corners(groups, umesh.uv)

    def test(self, event):
        pt = self.get_mouse_pos(event)
        self.points = (pt,)

    def calc_from_corners(self, groups: typing.Sequence[typing.Sequence[BMLoop]] | typing.Sequence[BMLoop] | BMLoop, uv=None, exact=False):
        from gpu_extras.batch import batch_for_shader
        if not groups:
            self.mid_points = []
            self.texts = []
            return

        if isinstance(groups, BMLoop):
            groups = [[groups]]
        elif isinstance(groups[0], BMLoop):
            groups = [groups]

        if uv is None:
            uv = self.uv

        offset_lines = self.uv_crn_groups_to_lines_with_offset(groups, uv, exact=exact)
        color = color_for_groups(groups)
        self.calc_text_data_from_lines(offset_lines)

        self.shader_smooth_color = gpu.shader.from_builtin('SMOOTH_COLOR')
        self.batch_smooth_color = batch_for_shader(self.shader_smooth_color, 'LINES', {"pos": offset_lines, 'color': color})
        self.batch_smooth_color_2 = batch_for_shader(self.shader_smooth_color, 'POINTS', {"pos": offset_lines[::2], 'color': color[::2]})

    def calc_from_segments(self, groups: typing.Sequence[typing.Sequence['CrnEdgeGrow']] | typing.Sequence['CrnEdgeGrow'] | 'CrnEdgeGrow'):  # noqa
        from gpu_extras.batch import batch_for_shader
        if not groups:
            self.mid_points = []
            self.texts = []
            return

        if type(groups).__name__ == 'CrnEdgeGrow':
            groups = [[groups]]
        elif type(groups[0]).__name__ == 'CrnEdgeGrow':
            groups = [groups]

        offset_lines = self.uv_segments_to_lines_with_offset(groups)
        color = color_for_groups(groups)
        self.calc_text_data_from_lines(offset_lines)

        self.shader_smooth_color = gpu.shader.from_builtin('SMOOTH_COLOR')
        self.batch_smooth_color = batch_for_shader(self.shader_smooth_color, 'LINES', {"pos": offset_lines, 'color': color})
        self.batch_smooth_color_2 = batch_for_shader(self.shader_smooth_color, 'POINTS', {"pos": offset_lines[::2], 'color': color[::2]})

    def get_mouse_pos(self, event):
        return Vector(self.view.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def register_draw(self):
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(self.univ_test_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def univ_test_draw_callback(self):
        if bpy.context.area.ui_type != 'UV':
            return

        import blf
        from gpu_extras.batch import batch_for_shader

        gpu.state.blend_set('ALPHA')
        gpu.state.point_size_set(8)
        gpu.state.line_width_set(2)

        try:
            self.shader.bind()
        except ReferenceError:
            return

        self.shader.uniform_float("color", (1, 1, 0, 0.5))

        batch_nearest = batch_for_shader(self.shader, 'POINTS', {"pos": self.points})
        self.shader.uniform_float("color", (1, 0.2, 0, 1))
        batch_nearest.draw(self.shader)
        if self.batch_smooth_color:
            self.shader_smooth_color.bind()
            self.batch_smooth_color.draw(self.shader_smooth_color)
            self.batch_smooth_color_2.draw(self.shader_smooth_color)

            blf.size(font_id := 0, 350)
            blf.position(font_id, 0, 0, 0)
            scale = 0.000015  # * 0.5

            def draw_texts(mid_points, texts):
                blf_draw = blf.draw
                m_translate = gpu.matrix.translate
                for pt_, text_ in zip(mid_points, texts):

                    m_translate(pt_)
                    blf_draw(font_id, text_)

            blf.color(font_id, 0.8, 0.0, 0.0, 1.0)

            with gpu.matrix.push_pop():
                gpu.matrix.scale((scale, scale))
                draw_texts(self.mid_points, self.texts)

        self.area.tag_redraw()

        gpu.state.blend_set('NONE')

    @staticmethod
    def uv_crn_groups_to_lines_with_offset(groups: typing.Sequence[typing.Sequence[BMLoop]], uv, line_offset=0.008, exact=False):
        """exact - correct line offset for flipped faces"""
        import numpy as np
        from .ubm import is_flipped_uv
        size = sum(len(group) for group in groups)

        edges = np.empty(shape=(size*2, 2), dtype='float32')
        idx = 0
        if exact:
            for group in groups:
                for crn in group:
                    start_edge = crn[uv].uv
                    end_edge = crn.link_loop_next[uv].uv

                    # offset
                    nx, ny = (end_edge - start_edge)
                    if not is_flipped_uv(crn.face, uv):
                        ny = -ny
                    n = Vector((ny, nx))
                    n.normalize()
                    n *= line_offset

                    edges[idx] = (start_edge + n).to_tuple()
                    idx += 1
                    edges[idx] = (end_edge + n).to_tuple()
                    idx += 1
        else:
            for group in groups:
                for crn in group:
                    start_edge = crn[uv].uv
                    end_edge = crn.link_loop_next[uv].uv

                    # offset
                    nx, ny = (end_edge - start_edge)

                    n = Vector((-ny, nx))
                    n.normalize()
                    n *= line_offset

                    edges[idx] = (start_edge + n).to_tuple()
                    idx += 1
                    edges[idx] = (end_edge + n).to_tuple()
                    idx += 1
        return edges

    @staticmethod
    def uv_segments_to_lines_with_offset(segments: typing.Sequence[typing.Sequence], line_offset=0.008):
        """exact - correct line offset for flipped faces"""
        import numpy as np
        size = sum(len(seg) for seg in segments)

        edges = np.empty(shape=(size*2, 2), dtype='float32')
        idx = 0

        for seg in segments:
            for adv_crn in seg:
                start_edge = adv_crn.curr_pt
                end_edge = adv_crn.next_pt

                # offset
                nx, ny = (end_edge - start_edge)

                n = Vector((-ny, nx))
                n.normalize()
                n *= line_offset

                edges[idx] = (start_edge + n).to_tuple()
                idx += 1
                edges[idx] = (end_edge + n).to_tuple()
                idx += 1

        return edges

    def calc_text_data_from_lines(self, edges: typing.Sequence | typing.Any, scale=0.000015):
        # import blf
        import numpy as np
        size = len(edges) // 2

        edges: np.ndarray = edges.reshape(len(edges)//2, 2, 2)
        edges_midpoints = np.mean(edges, axis=1)

        # blf.size(font_id := 0, 350)
        # blf.position(font_id, 0, 0, 0)

        texts = np.arange(size, dtype='uint32').astype(str)

        # texts_dim = np.empty(shape=(size, 2), dtype='float32')
        # blf_dimensions = blf.dimensions
        # for idx, text in enumerate(texts):
        #     texts_dim[idx] = blf_dimensions(font_id, text)

        # texts_dim *= scale * 0.5

        edges_midpoints *= 1 / scale

        # edges_midpoints -= texts_dim  # shift text center

        shifted_midpoints = np.roll(edges_midpoints, shift=1, axis=0)

        shifted_midpoints[0] = (0.0, 0.0)
        edges_midpoints -= shifted_midpoints

        self.mid_points = edges_midpoints

        self.texts = texts

    def exit(self):
        if not (self.handler is None):
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler, 'WINDOW')

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.ui_type == 'UV':
                        area.tag_redraw()

        return {'FINISHED'}
    
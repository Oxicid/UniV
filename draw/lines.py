# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import bpy
import typing
import gpu
import numpy.typing as npt
import mathutils
from . import shaders

from time import perf_counter as time
from gpu_extras.batch import batch_for_shader


class LinesDrawSimple:
    start_time = time()
    max_draw_time = 1.5
    handler: None = None
    shader: gpu.types.GPUShader | None = None
    batch: gpu.types.GPUBatch | None = None
    color: tuple = (1, 1, 0, 1)
    # target_area: bpy.types.Area = None

    @classmethod
    def draw_register(cls, data: list[mathutils.Vector] | npt.NDArray, color: tuple = (1, 1, 0, 1)):
        if not data:
            return
        cls.start_time = time()
        cls.color = color

        cls.shader = shaders.POLYLINE_UNIFORM_COLOR_2D
        cls.batch = batch_for_shader(cls.shader, 'LINES', {"pos": data})

        sima = bpy.types.SpaceImageEditor
        if not (cls.handler is None):
            sima.draw_handler_remove(cls.handler, 'WINDOW')

        cls.handler = sima.draw_handler_add(cls.draw_callback_px, (), 'WINDOW', 'POST_VIEW')
        bpy.app.timers.register(cls.uv_area_draw_timer)

    @classmethod
    def uv_area_draw_timer(cls):
        if cls.handler is None:
            cls.max_draw_time = 1.5
            return None
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = 1.5
        return None

    @classmethod
    def draw_callback_px(cls):
        if bpy.context.area.ui_type != 'UV':
            return

        shaders.set_line_width(2)
        shaders.blend_set_alpha()

        cls.shader.bind()
        cls.shader.uniform_float("color", cls.color)
        shaders.set_line_width_vk(cls.shader)
        cls.batch.draw(cls.shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()


class LinesDrawSimple3D:
    start_time = time()
    max_draw_time = 1.5
    handler: None = None
    shader: gpu.types.GPUShader | None = None
    batch: gpu.types.GPUBatch | None = None
    color: tuple = (1, 1, 0, 1)
    # target_area: bpy.types.Area = None

    @classmethod
    def draw_register(cls, data: list[mathutils.Vector] | npt.NDArray, color: tuple = (1, 1, 0, 1)):
        if not len(data):
            return
        cls.start_time = time()
        cls.color = color

        cls.shader = shaders.POLYLINE_UNIFORM_COLOR_3D
        cls.batch = batch_for_shader(cls.shader, 'LINES', {"pos": data})

        v3d = bpy.types.SpaceView3D
        if not (cls.handler is None):
            v3d.draw_handler_remove(cls.handler, 'WINDOW')

        cls.handler = v3d.draw_handler_add(cls.draw_callback_px, (), 'WINDOW', 'POST_VIEW')
        bpy.app.timers.register(cls.univ_view3d_draw_timer)

    @classmethod
    def univ_view3d_draw_timer(cls):
        if cls.handler is None:
            cls.max_draw_time = 1.5
            return None
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceView3D.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = 1.5
        return None

    @classmethod
    def draw_callback_px(cls):
        if bpy.context.area.type != 'VIEW_3D':
            return

        shaders.set_line_width(2)
        shaders.blend_set_alpha()

        cls.shader.bind()
        cls.shader.uniform_float("color", cls.color)
        shaders.set_line_width_vk(cls.shader)
        cls.batch.draw(cls.shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()


class DotLinesDrawSimple:
    start_time = time()
    max_draw_time = 1.5
    handler: None = None
    shader: gpu.types.GPUShader | None = None
    batch: gpu.types.GPUBatch | None = None
    color: tuple = (0, 1, 1, 1)
    # target_area: bpy.types.Area = None

    @classmethod
    def draw_register(cls, data: list[mathutils.Vector] | npt.NDArray, color: tuple = (0, 1, 1, 1)):
        if not data:
            return
        cls.start_time = time()
        cls.color = color

        if not cls.shader:
            cls.create_shader_info()

        arc_lengths = []
        arc_lengths_append = arc_lengths.append
        it = iter(data)
        for a in it:
            b = next(it)
            arc_lengths_append(0)
            arc_lengths_append((a - b).length)

        cls.batch = batch_for_shader(cls.shader, 'LINES', {"pos": data, 'arc_length': arc_lengths})

        sima = bpy.types.SpaceImageEditor
        if not (cls.handler is None):
            sima.draw_handler_remove(cls.handler, 'WINDOW')

        cls.handler = sima.draw_handler_add(cls.draw_callback_px, (), 'WINDOW', 'POST_VIEW')
        bpy.app.timers.register(cls.uv_area_draw_timer)

    @classmethod
    def uv_area_draw_timer(cls):
        if cls.handler is None:
            cls.max_draw_time = 1.5
            return None
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()
        cls.handler = None
        cls.max_draw_time = 1.5
        return None

    @classmethod
    def draw_callback_px(cls):
        area = bpy.context.area
        if area.ui_type != 'UV':
            return

        shaders.set_line_width(3)
        shaders.blend_set_alpha()

        cls.shader.bind()

        from .. import utypes
        reg = next(r for r in area.regions if r.type == 'WINDOW')
        zoom = utypes.View2D.get_zoom(reg.view2d) / 10

        matrix = gpu.matrix.get_projection_matrix()
        cls.shader.uniform_float("vpm", matrix)
        cls.shader.uniform_float("color", cls.color)
        cls.shader.uniform_float("scale", zoom)
        # shaders.set_line_width_vk(cls.shader)  # TODO: Dot shader not support line_width
        cls.batch.draw(cls.shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()

    @classmethod
    def create_shader_info(cls):
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('FLOAT', "v_arc_length")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "vpm")
        shader_info.push_constant('FLOAT', "scale")
        shader_info.push_constant('VEC4', "color")
        shader_info.vertex_in(0, 'VEC2', "pos")
        shader_info.vertex_in(1, 'FLOAT', "arc_length")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "out_color")

        shader_info.vertex_source(
            "void main()"
            "{"
            "  v_arc_length = arc_length;"
            "  gl_Position = vpm * vec4(pos, 0.0f, 1.0f);"
            "}"
        )

        shader_info.fragment_source(
            "void main()"
            "{"
            "  if (mod(v_arc_length, 1.0 / scale) > 0.4 / scale) discard;"
            "  out_color = color;"
            "}"
        )

        cls.shader = gpu.shader.create_from_info(shader_info)

class SegmentsDrawSimple:
    start_time = time()
    default_max_draw_time = 20
    max_draw_time = 10
    handler: None = None
    shader: gpu.types.GPUShader | None = None
    shader2: gpu.types.GPUShader | None = None
    batch: gpu.types.GPUBatch | None = None
    batch2: gpu.types.GPUBatch | None = None
    mid_points = []
    texts = []
    # target_area: bpy.types.Area = None


    @classmethod
    def draw_register(cls, groups: "typing.Sequence[typing.Iterable]"):
        from gpu_extras.batch import batch_for_shader
        from ..utils import UNIV_OT_Draw_Test, color_for_groups
        if not groups:
            return
        cls.start_time = time()

        if getattr(bpy.context.preferences.system, "gpu_backend", None) == "VULKAN":
            cls.shader = gpu.shader.from_builtin('SMOOTH_COLOR')
            cls.shader2 = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
        else:
            cls.shader = gpu.shader.from_builtin('SMOOTH_COLOR')
            cls.shader2 = gpu.shader.from_builtin('SMOOTH_COLOR')

        offset_lines = UNIV_OT_Draw_Test.uv_segments_to_lines_with_offset(groups)
        color = color_for_groups(groups)
        cls.mid_points, cls.texts = UNIV_OT_Draw_Test.calc_text_data_from_lines(offset_lines)

        cls.batch = batch_for_shader(cls.shader, 'LINES', {"pos": offset_lines, 'color': color})
        cls.batch_2 = batch_for_shader(cls.shader2, 'POINTS', {"pos": offset_lines[::2], 'color': color[::2]})

        sima = bpy.types.SpaceImageEditor
        if not (cls.handler is None):
            sima.draw_handler_remove(cls.handler, 'WINDOW')

        cls.handler = sima.draw_handler_add(cls.draw_callback_px, (), 'WINDOW', 'POST_VIEW')
        bpy.app.timers.register(cls.uv_area_draw_timer)


    @classmethod
    def uv_area_draw_timer(cls):
        if cls.handler is None:
            cls.max_draw_time = cls.default_max_draw_time
            return None
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = cls.default_max_draw_time
        return None

    @classmethod
    def draw_callback_px(cls):
        if bpy.context.area.ui_type != 'UV':
            return

        import blf
        from ..draw import shaders

        shaders.blend_set_alpha()
        shaders.set_point_size(8)
        # if not is_vulkan_enabled:
        shaders.set_line_width(2)

        # try:
        #     cls.shader.bind()
        # except ReferenceError:
        #     return

        cls.batch.draw(cls.shader)
        cls.batch_2.draw(cls.shader2)

        font_id = 0
        blf.size(font_id, 350)
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
            draw_texts(cls.mid_points, cls.texts)


        shaders.blend_set_none()
        shaders.set_point_size(1)
        shaders.set_line_width(1)
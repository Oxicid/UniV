# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu
import numpy.typing as npt
import mathutils

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

        cls.shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR' if bpy.app.version < (3, 5, 0) else 'UNIFORM_COLOR')
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
            return
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = 1.5
        return

    @classmethod
    def draw_callback_px(cls):
        if bpy.context.area.ui_type != 'UV':
            return

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(2)
            bgl.glEnable(bgl.GL_ALPHA)
        else:
            gpu.state.line_width_set(2)
            gpu.state.blend_set('ALPHA')

        cls.shader.bind()
        cls.shader.uniform_float("color", cls.color)
        cls.batch.draw(cls.shader)

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(1)
            bgl.glDisable(bgl.GL_BLEND)  # noqa
        else:
            gpu.state.line_width_set(1)
            gpu.state.blend_set('NONE')

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

        cls.shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR' if bpy.app.version < (3, 5, 0) else 'UNIFORM_COLOR')
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
            return
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceView3D.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = 1.5
        return

    @classmethod
    def draw_callback_px(cls):
        if bpy.context.area.type != 'VIEW_3D':
            return

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(2)
            bgl.glEnable(bgl.GL_ALPHA)
        else:
            gpu.state.line_width_set(2)
            gpu.state.blend_set('ALPHA')

        cls.shader.bind()
        cls.shader.uniform_float("color", cls.color)
        cls.batch.draw(cls.shader)

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(1)
            bgl.glDisable(bgl.GL_BLEND)  # noqa
        else:
            gpu.state.line_width_set(1)
            gpu.state.blend_set('NONE')


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
            return
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV' :
                a.tag_redraw()
        cls.handler = None
        cls.max_draw_time = 1.5
        return

    @classmethod
    def draw_callback_px(cls):
        area = bpy.context.area
        if area.ui_type != 'UV':
            return

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(3)
            bgl.glEnable(bgl.GL_ALPHA)
        else:
            gpu.state.line_width_set(3)
            gpu.state.blend_set('ALPHA')

        cls.shader.bind()
        reg = next(r for r in area.regions if r.type == 'WINDOW')

        from .. import types
        zoom = types.View2D.get_zoom(reg.view2d)/10

        matrix = gpu.matrix.get_projection_matrix()
        cls.shader.uniform_float("vpm", matrix)
        cls.shader.uniform_float("color", cls.color)
        cls.shader.uniform_float("scale", zoom)
        cls.batch.draw(cls.shader)

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(1)
            bgl.glDisable(bgl.GL_BLEND)  # noqa
        else:
            gpu.state.line_width_set(1)
            gpu.state.blend_set('NONE')

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
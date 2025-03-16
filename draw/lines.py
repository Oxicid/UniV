# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu

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
    def draw_register(cls, data, color: tuple = (1, 1, 0, 1)):
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
            return
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2
        bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()
        cls.handler = None
        return

    @classmethod
    def draw_callback_px(cls):
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

        for a in bpy.context.screen.areas:
            if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                a.tag_redraw()

        if bpy.app.version < (3, 5, 0):
            import bgl
            bgl.glLineWidth(1)
            bgl.glDisable(bgl.GL_BLEND)  # noqa
        else:
            gpu.state.line_width_set(1)
            gpu.state.blend_set('NONE')

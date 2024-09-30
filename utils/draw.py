# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
from random import seed, random

import bpy
import gpu
import numpy as np
from bmesh.types import BMLoop
from mathutils import Vector


def color_for_groups(groups) -> list[list[float]]:
    """Return flat colors by group"""
    colors = []
    for i in range(len(groups)):
        seed(i)
        c0 = random()
        seed(hash(c0))
        c1 = random()
        seed(hash(c1))
        c2 = random()
        colors.append((c0, c1, c2))

    return np.repeat(colors, [len(g)*2 for g in groups], axis=0).tolist()  # noqa

def uv_crn_groups_to_lines_with_offset(groups: list[list[BMLoop]], uv, line_offset=0.008):
    edges = []
    for group in groups:
        for crn in group:
            edges.append(crn[uv].uv.copy())
            edges.append(crn.link_loop_next[uv].uv.copy())

    edge_iter = iter(edges)
    for _ in range(int(len(edges) / 2)):
        start_edge = next(edge_iter)
        end_edge = next(edge_iter)

        nx, ny = (end_edge - start_edge)
        n = Vector((-ny, nx))
        n.normalize()
        n *= line_offset

        start_edge += n
        end_edge += n

class UNIV_OT_Draw_Test(bpy.types.Operator):
    bl_idname = 'uv.univ_draw_test'
    bl_label = 'Draw Test'
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        self.points = ()
        self.mouse_position = ()
        self.view = None
        self.handler = None
        self.area = None
        self.shader = None
        self.umeshes = None
        self.batch = None

    def invoke(self, context, event):
        self.area = context.area
        self.view = context.region.view2d
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.register_draw()
        from .. import types

        self.umeshes = types.UMeshes()

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

        if event.type in ('ESC', 'RIGHTMOUSE'):
            return self.exit()

        return {'RUNNING_MODAL'}

    def test(self, event):
        pt = self.get_mouse_pos(event)
        self.points = (pt,)

    def get_mouse_pos(self, event):
        return Vector(self.view.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def register_draw(self):
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(self.univ_test_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def univ_test_draw_callback(self):
        if bpy.context.area.ui_type != 'UV':
            return

        gpu.state.point_size_set(4)
        gpu.state.blend_set('ALPHA')

        self.shader.bind()
        self.shader.uniform_float("color", (1, 1, 0, 0.5))
        from gpu_extras.batch import batch_for_shader

        batch_nearest = batch_for_shader(self.shader, 'POINTS', {"pos": self.points})
        self.shader.uniform_float("color", (1, 0.2, 0, 1))
        batch_nearest.draw(self.shader)

        self.area.tag_redraw()

        gpu.state.blend_set('NONE')

    def exit(self):
        if not (self.handler is None):
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler, 'WINDOW')

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.ui_type == 'UV':
                        area.tag_redraw()

        return {'FINISHED'}
    
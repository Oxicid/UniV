# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import bmesh
import contextlib
from . import shaders
from . import mesh_extract
from .text import TextDraw
from ..utypes import UMesh
from ..preferences import univ_settings
from .lines import LinesDrawSimple, LinesDrawSimple3D, DotLinesDrawSimple
from gpu_extras.batch import batch_for_shader


class DrawerSeamsProcessing:
    @staticmethod
    def draw_fn_2d(shader, batch, color):
        shaders.set_line_width(2)
        shaders.blend_set_alpha()

        shader.bind()
        shader.uniform_float("color", color)
        shaders.set_line_width_vk(shader)
        batch.draw(shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()

    @staticmethod
    def data_to_batch(data, shader):
        if not data:
            return None
        return batch_for_shader(shader, 'LINES', {"pos": data})

    @staticmethod
    def get_color():
        return *bpy.context.preferences.themes[0].view_3d.edge_seam, 0.8

    @staticmethod
    def get_shader():
        return shaders.POLYLINE_UNIFORM_COLOR

    @staticmethod
    def is_enable():
        try:
            from .. import univ_pro
            return univ_settings().overlay_2d_enable
        except ImportError:
            return False


class DrawCall:
    def __init__(self, draw_fn, shader, color, batch):
        self.draw_fn = draw_fn
        self.shader = shader
        self.color = color
        self.batch = batch
        # To avoid iterating over all mesh elements again,
        # UniV operators can control Update by adding extended draw elements on top of existing draw elements.
        self.coords_extend = []
        self.batch_extend = None

    def __call__(self):
        if self.batch:
            self.draw_fn(self.shader, self.batch, self.color)
        if self.batch_extend:
            self.draw_fn(self.shader, self.batch_extend, self.color)


class Drawer2D:
    draw_objects: dict[str | list[DrawCall]] = {}
    drawers = []
    shaders_with_color = []
    mesh_extractors_with_batch = []
    dirt = True
    handler = None
    sync = True
    frozen = False

    @classmethod
    def update_drawer_data(cls):
        drawers = []
        shaders_with_color = []
        mesh_extractors_with_batch = []
        if True:  # if seam
            drawers.append(DrawerSeamsProcessing.draw_fn_2d)
            shaders_with_color.append((DrawerSeamsProcessing.get_shader(), DrawerSeamsProcessing.get_color()))
            mesh_extractors_with_batch.append((mesh_extract.extract_seams_umesh, DrawerSeamsProcessing.data_to_batch))

        cls.dirt = True
        cls.draw_objects.clear()
        cls.drawers = drawers
        cls.shaders_with_color = shaders_with_color
        cls.mesh_extractors_with_batch = mesh_extractors_with_batch


    @classmethod
    def update(cls):
        if bpy.context.mode != 'EDIT_MESH':
            cls.dirt = True
            return

        if cls.sync != bpy.context.tool_settings.use_uv_select_sync:
            cls.sync = not cls.sync
            cls.dirt = True
            cls.draw_objects.clear()

        draw_objects = cls.draw_objects
        if not draw_objects:
            cls.dirt = True

        if not cls.dirt:
            return

        unique_objects_with_uv = [obj for obj in bpy.context.objects_in_mode_unique_data if obj.data.uv_layers]
        for obj in unique_objects_with_uv:
            obj_id = obj.name
            if obj_id not in draw_objects:
                umesh = UMesh(bmesh.from_edit_mesh(obj.data), obj)

                draw_calls_seq = []
                for drawer, (shader, color), (extract, batch) in zip(cls.drawers, cls.shaders_with_color, cls.mesh_extractors_with_batch):
                    data = extract(umesh)
                    draw_call = DrawCall(drawer, shader, color, batch(data, shader))
                    draw_calls_seq.append(draw_call)

                draw_objects[obj_id] = draw_calls_seq

        # Delete extra objects after renaming
        if len(unique_objects_with_uv) != len(draw_objects):
            names = {obj.name for obj in unique_objects_with_uv}
            for draw_obj_key in draw_objects.copy():
                if draw_obj_key not in names:
                    del draw_objects[draw_obj_key]

        cls.dirt = False


    @staticmethod
    def univ_drawer_2d_callback():
        if bpy.context.area.ui_type == 'UV':
            Drawer2D.update()

            for draw_calls_seq in Drawer2D.draw_objects.values():
                for draw_call in draw_calls_seq:
                    draw_call()

    @classmethod
    def register(cls):
        cls.unregister()
        if univ_settings().overlay_2d_enable:
            Drawer2D.update_drawer_data()
            cls.handler = bpy.types.SpaceImageEditor.draw_handler_add(
                cls.univ_drawer_2d_callback, (), 'WINDOW', 'POST_VIEW')

            bpy.app.handlers.depsgraph_update_post.append(Drawer2D.univ_drawer_2d_update)
            cls.sync = bpy.context.tool_settings.use_uv_select_sync

    @classmethod
    def unregister(cls):
        if cls.handler is not None:
            bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')

        for update_handler in reversed(bpy.app.handlers.depsgraph_update_post):
            if update_handler.__name__ == Drawer2D.univ_drawer_2d_update.__name__:
                bpy.app.handlers.depsgraph_update_post.remove(update_handler)

        cls.draw_objects.clear()
        cls.mesh_extractors_with_batch.clear()
        cls.shaders_with_color.clear()
        cls.drawers.clear()

        cls.dirt = True
        cls.frozen = False
        cls.handler = None

    @staticmethod
    @bpy.app.handlers.persistent
    def univ_drawer_2d_update(_, deps):
        draw_objects = Drawer2D.draw_objects

        if not draw_objects:
            return

        if bpy.context.mode != 'EDIT_MESH':
            Drawer2D.dirt = True
            draw_objects.clear()
            return

        from bpy.types import Object
        for update_obj in deps.updates:
            obj = update_obj.id
            if type(obj) == Object:
                Drawer2D.dirt = True
                try:
                    del draw_objects[obj.name]
                except: pass # noqa

                if not draw_objects:
                    return

    @staticmethod
    def append_handler_with_delay():
        try:
            from ..preferences import univ_settings
            if univ_settings().overlay_2d_enable:
                Drawer2D.register()
        except Exception as e:
            print('UniV: Failed to add a handler for Drawer2D system.', e)

    @staticmethod
    def update_data(self, _context):
        if self.overlay_2d_enable:
            Drawer2D.register()
        else:
            Drawer2D.unregister()

    @classmethod
    def is_valid(cls):
        assert len(cls.mesh_extractors_with_batch) == len(cls.shaders_with_color)
        assert len(cls.mesh_extractors_with_batch) == len(cls.drawers)

        if cls.dirt:
            return

        if bpy.context.mode != 'EDIT_MESH':
            return

        for obj in bpy.context.objects_in_mode_unique_data:
            if obj.data.uv_layers:
                assert id(obj) in cls.draw_objects
            else:
                assert id(obj) not in cls.draw_objects

    @classmethod
    @contextlib.contextmanager
    def freeze(cls):
        cls.frozen = True
        try:
            yield
        finally:
            cls.frozen = False
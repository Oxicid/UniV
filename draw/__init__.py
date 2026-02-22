# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu
import bmesh
import contextlib

from mathutils import Vector
from gpu_extras.batch import batch_for_shader

from . import shaders
from . import mesh_extract
from .text import TextDraw
from ..utypes import UMesh
from ..preferences import univ_settings, prefs
from .lines import LinesDrawSimple, LinesDrawSimple3D, DotLinesDrawSimple



class DrawerNonSyncSelectProcessing:
    @staticmethod
    def draw_fn_2d(shader, batch, verts_lines_color, world_matrix):
        verts_lines_shader, tris_shader = shader
        verts_lines_batch, tris_batch = batch

        shaders.blend_set_alpha()
        view_3d_theme = bpy.context.preferences.themes[0].view_3d
        shaders.set_point_size(view_3d_theme.vertex_size + 1.0)
        edge_width = getattr(view_3d_theme, 'edge_width', 1.0) + 1.0
        shaders.set_line_width(edge_width)

        gpu.state.depth_test_set('ALWAYS' if univ_settings().overlay_toggle_xray else 'LESS')  # enable deps-test
        gpu.state.depth_mask_set(True)  # write in depth-buffer

        if tris_batch:
            tris_shader.bind()
            rv3d = bpy.context.region_data
            mvp = rv3d.perspective_matrix @ world_matrix
            tris_shader.uniform_float("mvp", mvp)

            normal_matrix = world_matrix.to_3x3().inverted_safe().transposed()
            tris_shader.uniform_float("normal_matrix", normal_matrix)

            view_dir = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
            tris_shader.uniform_float("light_dir", view_dir.xy)

            tris_batch.draw(tris_shader)

        gpu.state.depth_test_set('ALWAYS')  # enable deps-test
        gpu.state.depth_mask_set(True)
        if verts_lines_batch:
            verts_lines_shader.bind()
            verts_lines_shader.uniform_float("color", verts_lines_color)
            try:
                # TODO: Split shaders by elem mode
                shaders.set_line_width_vk(verts_lines_shader, edge_width)
            except: pass  # noqa
            rv3d = bpy.context.region_data

            with gpu.matrix.push_pop():
                gpu.matrix.load_projection_matrix(rv3d.window_matrix)
                gpu.matrix.load_matrix(rv3d.view_matrix @ world_matrix)
                verts_lines_batch.draw(verts_lines_shader)

        shaders.set_point_size(1)
        shaders.set_line_width(1)
        shaders.blend_set_none()

    @staticmethod
    def data_to_batch(data, shader):
        verts_lines, (tris, normals) = data
        verts_lines_shader, tris_shader = shader

        if not verts_lines and not tris:
            return None

        verts_lines_batch = None
        tris_batch = None
        match bpy.context.tool_settings.uv_select_mode:
            case 'VERTEX':
                if verts_lines:
                    verts_lines_batch = batch_for_shader(verts_lines_shader, 'POINTS', {"pos": verts_lines})
            case 'EDGE':
                if verts_lines:
                    verts_lines_batch = batch_for_shader(verts_lines_shader, 'LINES', {"pos": verts_lines})

        if tris:
            tris_batch = batch_for_shader(tris_shader, 'TRIS', {"pos": tris, "normal": normals})

        if verts_lines_batch or tris_batch:
            return verts_lines_batch, tris_batch
        return None


    @staticmethod
    def get_color():
        if bpy.context.tool_settings.uv_select_mode == 'VERTEX':
            return univ_settings().overlay_3d_uv_vert_color
        else:
            return univ_settings().overlay_3d_uv_edge_color

    @classmethod
    def update_color(cls):
        for idx, drawer in enumerate(Drawer3D.drawers):
            if drawer is cls.draw_fn_2d:
                shader, c = Drawer3D.shaders_with_color[idx]
                color = cls.get_color()
                Drawer3D.shaders_with_color[idx] = (shader, color)
        for draw_calls in Drawer3D.draw_objects.values():
            for draw_call in draw_calls:
                if draw_call.draw_fn is cls.draw_fn_2d:
                    draw_call.color = cls.get_color()


    @staticmethod
    def get_shader():
        if bpy.context.tool_settings.uv_select_mode == 'VERTEX':
            return shaders.POINT_UNIFORM_COLOR_3D, shaders.FLAT_SHADING_UNIFORM_COLOR_3D_FOR_UV_FACE_SELECT
        else:
            return shaders.POLYLINE_UNIFORM_COLOR_3D, shaders.FLAT_SHADING_UNIFORM_COLOR_3D_FOR_UV_FACE_SELECT

    @staticmethod
    def is_enable():
        try:
            from .. import univ_pro
            return univ_settings().overlay_3d_enable
        except ImportError:
            return False



class DrawCallSeams2D:
    def __init__(self, batch: gpu.types.GPUBatch):
        self.shader = shaders.POLYLINE_UNIFORM_COLOR
        self.color = self.get_color()
        self.batch = batch
        # To avoid iterating over all mesh elements again,
        # UniV operators can control Update by adding extended draw elements on top of existing draw elements.
        # self.coords_extend = []
        # self.batch_extend = None

    def __call__(self):
        shaders.set_line_width(2)
        shaders.blend_set_alpha()

        self.shader.bind()
        self.shader.uniform_float("color", self.color)
        shaders.set_line_width_vk(self.shader)
        self.batch.draw(self.shader)
        # if self.batch_extend:
        #     self.batch_extend.draw(self.shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()

    @classmethod
    def init(cls, umesh: UMesh) -> 'DrawCallSeams2D | None':
        data = mesh_extract.extract_seams_umesh(umesh)
        if data:
            return cls(batch_for_shader(shaders.POLYLINE_UNIFORM_COLOR, 'LINES', {"pos": data}))
        return None

    if bpy.app.version >= (4, 5, 0):
        @staticmethod
        def get_color():
            return *bpy.context.preferences.themes[0].view_3d.seam, 0.5
    else:
        @staticmethod
        def get_color():
            return *bpy.context.preferences.themes[0].view_3d.edge_seam, 0.5

    @staticmethod
    def is_enable():
        try:
            from .. import univ_pro
            return univ_settings().overlay_2d_enable
        except ImportError:
            return False

class DrawCallConstraints2D:
    def __init__(self, batch_v: gpu.types.GPUBatch | None, batch_h: gpu.types.GPUBatch | None):
        self.shader = shaders.POLYLINE_UNIFORM_COLOR
        self.color_v = (0.1, 0.1, 0.8, 0.5)
        self.color_h = (0.85, 1.0, 0.0, 0.5)

        self.batch_v = batch_v
        self.batch_h = batch_h

    def __call__(self):
        shaders.blend_set_alpha()
        shaders.set_line_width(3)

        self.shader.bind()
        shaders.set_line_width_vk(self.shader, 3)

        if self.batch_v:
            self.shader.uniform_float("color", self.color_v)
            self.batch_v.draw(self.shader)
        if self.batch_h:
            self.shader.uniform_float("color", self.color_h)
            self.batch_h.draw(self.shader)

        shaders.set_line_width(1)
        shaders.blend_set_none()

    @classmethod
    def init(cls, umesh: UMesh) -> 'DrawCallConstraints2D | None':
        constraints_attr = umesh.bm.edges.layers.int.get('univ_constraints')
        if not constraints_attr:
            return None

        v_coords, h_coords = cls.extract_data(umesh, constraints_attr)
        if not (v_coords or h_coords):
            return None

        v_batch = None
        h_batch = None

        if v_coords:
            v_batch = batch_for_shader(shaders.POLYLINE_UNIFORM_COLOR, 'LINES', {"pos": v_coords})
        if h_coords:
            h_batch = batch_for_shader(shaders.POLYLINE_UNIFORM_COLOR, 'LINES', {"pos": h_coords})

        return cls(v_batch, h_batch)

    @staticmethod
    def is_enable():
        try:
            from .. import univ_pro
            return univ_settings().overlay_2d_enable
        except ImportError:
            return False

    @staticmethod
    def extract_data(umesh: UMesh, attr):
        """"""
        v_coords = []
        h_coords = []
        v_coords_append = v_coords.append
        h_coords_append = h_coords.append

        uv = umesh.uv
        if umesh.is_full_face_selected:
            for e in umesh.bm.edges:
                if edge_idx := e[attr]:
                    for crn in getattr(e, 'link_loops', ()):
                        bits = edge_idx & 3

                        if bits == 2:  # vertical
                            v_coords_append(crn[uv].uv)
                            v_coords_append(crn.link_loop_next[uv].uv)
                        elif bits == 3:  # horizontal
                            h_coords_append(crn[uv].uv)
                            h_coords_append(crn.link_loop_next[uv].uv)

                        edge_idx >>= 2
        else:
            if umesh.sync:
                for e in umesh.bm.edges:
                    if edge_idx := e[attr]:
                        for crn in getattr(e, 'link_loops', ()):
                            if not crn.face.hide:
                                bits = edge_idx & 3

                                if bits == 2:  # vertical
                                    v_coords_append(crn[uv].uv)
                                    v_coords_append(crn.link_loop_next[uv].uv)
                                elif bits == 3:  # horizontal
                                    h_coords_append(crn[uv].uv)
                                    h_coords_append(crn.link_loop_next[uv].uv)

                                edge_idx >>= 2
            else:
                if umesh.is_full_face_deselected:
                    return [], []
                for e in umesh.bm.edges:
                    if edge_idx := e[attr]:
                        for crn in getattr(e, 'link_loops', ()):
                            if crn.face.select:
                                bits = edge_idx & 3

                                if bits == 2:  # vertical
                                    v_coords_append(crn[uv].uv)
                                    v_coords_append(crn.link_loop_next[uv].uv)
                                elif bits == 3:  # horizontal
                                    h_coords_append(crn[uv].uv)
                                    h_coords_append(crn.link_loop_next[uv].uv)

                                edge_idx >>= 2
        return v_coords, h_coords

class DrawCall3D:
    def __init__(self, draw_fn, shader, color, batch, world_matrix):
        self.draw_fn = draw_fn
        self.shader = shader
        self.color = color
        self.batch = batch
        self.world_matrix = world_matrix
        # To avoid iterating over all mesh elements again,
        # UniV operators can control Update by adding extended draw elements on top of existing draw elements.
        self.coords_extend = []
        self.batch_extend = None

    def __call__(self):
        if self.batch:
            self.draw_fn(self.shader, self.batch, self.color, self.world_matrix)
        if self.batch_extend:
            self.draw_fn(self.shader, self.batch_extend, self.color, self.world_matrix)


class TrimDrawer:
    line_shader = None
    tris_shader = None
    batch_lines = None
    batch_tris = None
    handler = None
    pause = False  # for updates

    @classmethod
    def data_to_batch(cls):
        if not cls.is_enable():
            return

        boxes_lines = []
        boxes_lines_colors = []

        boxes_tris = []
        boxes_tris_color = []
        active = prefs().active_trim_index

        line_opacity = prefs().trim_line_opacity
        tris_opacity = prefs().trim_tris_opacity

        from .. import utils
        aspect = utils.get_aspect_ratio()
        aspect_x = min(1.0, 1.0 / aspect)
        aspect_y = min(1.0, aspect)

        for idx, trim in enumerate(prefs().trims_presets):
            if trim.visible:
                bb = trim.to_bbox()
                boxes_lines.extend(bb.draw_data_lines())
                boxes_tris.extend(bb.draw_data_tris())

                if idx == active:
                    center = bb.center
                    cursor_size = trim.get_cursor_size()

                    off_x = cursor_size * aspect_x
                    off_y = cursor_size * aspect_y

                    boxes_lines.append(center - Vector((off_x, 0)))
                    boxes_lines.append(center + Vector((off_x, 0)))
                    boxes_lines.append(center - Vector((0, off_y)))
                    boxes_lines.append(center + Vector((0, off_y)))

                    boxes_lines_colors.extend([[*trim.color, line_opacity+0.3]] * 12)
                    boxes_tris_color.extend([[*trim.color, tris_opacity+0.15]] * 6)
                else:
                    boxes_lines_colors.extend([[*trim.color, line_opacity]] * 8)
                    boxes_tris_color.extend([[*trim.color, tris_opacity]] * 6)

        if cls.line_shader is None:
            cls.tris_shader, cls.line_shader = cls.get_shader()
        cls.batch_lines = batch_for_shader(cls.line_shader, 'LINES', {"pos": boxes_lines, 'color': boxes_lines_colors})
        cls.batch_tris = batch_for_shader(cls.tris_shader, 'TRIS', {"pos": boxes_tris, 'color': boxes_tris_color})

    @staticmethod
    def get_shader():
        # TODO: Standardize
        return gpu.shader.from_builtin('SMOOTH_COLOR'), gpu.shader.from_builtin('POLYLINE_FLAT_COLOR')

    @staticmethod
    def is_enable():
        try:
            from .. import univ_pro
            return univ_settings().use_trims
        except ImportError:
            return False

    @staticmethod
    def univ_drawer_2d_callback():
        if bpy.context.area.ui_type == 'UV':
            # shaders.set_line_width(line_width)
            shaders.blend_set_alpha()
            TrimDrawer.line_shader.bind()

            # shaders.set_line_width_vk(TrimDrawer.shader, line_width)
            try:
                TrimDrawer.line_shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
                TrimDrawer.line_shader.uniform_float("lineWidth", prefs().trim_line_width)
            except: pass  # noqa
            TrimDrawer.batch_lines.draw(TrimDrawer.line_shader)
            TrimDrawer.batch_tris.draw(TrimDrawer.tris_shader)

            # shaders.set_line_width(1)
            shaders.blend_set_none()

    @classmethod
    def register(cls):
        cls.unregister()
        if cls.is_enable():
            cls.data_to_batch()
            cls.handler = bpy.types.SpaceImageEditor.draw_handler_add(
                cls.univ_drawer_2d_callback, (), 'WINDOW', 'POST_VIEW')

    @classmethod
    def unregister(cls):
        if cls.handler:
            try:
                bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')
            except Exception as e:
                print(e)

        cls.handler = None

    @staticmethod
    def append_handler_with_delay():
        try:
            TrimDrawer.register()
        except Exception as e:
            print('UniV: Failed to add a handler for Drawer2D system.', e)


class DrawerSubscribeRNA:
    sync_owner = None
    uv_mode_owner = None

    @staticmethod
    def univ_uv_sync_rna_callback():
        Drawer2D.update_data(univ_settings(), bpy.context)
        Drawer3D.update_data(univ_settings(), bpy.context)

    @classmethod
    def subscribe_to_uv_sync(cls):
        owner = object()
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.ToolSettings, "use_uv_select_sync"),
            owner=owner,
            args=(),
            notify=cls.univ_uv_sync_rna_callback,
        )
        cls.sync_owner = owner

    @classmethod
    def subscribe_to_uv_mode(cls):
        owner = object()
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.ToolSettings, "uv_select_mode"),
            owner=owner,
            args=(),
            notify=cls.univ_uv_sync_rna_callback,
        )
        cls.uv_mode_owner = owner

    @classmethod
    def subscribe(cls):
        cls.subscribe_to_uv_sync()
        cls.subscribe_to_uv_mode()

    @classmethod
    def unsubscribe(cls):
        if cls.sync_owner:
            bpy.msgbus.clear_by_owner(cls.sync_owner)
            cls.sync_owner = None

        if cls.uv_mode_owner:
            bpy.msgbus.clear_by_owner(cls.uv_mode_owner)
            cls.uv_mode_owner = None

    @staticmethod
    @bpy.app.handlers.persistent
    def univ_drawer_load_handler(_):
        DrawerSubscribeRNA.subscribe()

    @classmethod
    def register_handler(cls):
        cls.unregister_handler()
        bpy.app.handlers.load_post.append(cls.univ_drawer_load_handler)

    @classmethod
    def unregister_handler(cls):
        cls.unsubscribe()
        for handler in reversed(bpy.app.handlers.load_post):
            if handler.__name__ == cls.univ_drawer_load_handler.__name__:
                bpy.app.handlers.load_post.remove(handler)

def has_crash_modal_running():
    """Built-in Blender modal operators cause crashes, so we stop drawing elements while they are being executed."""
    if bpy.app.version >= (4, 2, 0):
        for window in bpy.context.window_manager.windows:
            for op in window.modal_operators:
                if not op.bl_idname.startswith(('UV_OT_univ_', 'WM_OT_sk_screencast_keys')):
                    return True
        return False
    else:
        # In older versions, modal operators cannot be selectively excluded,
        # so any modal operator interrupts drawing.
        from .. import utypes
        for window in bpy.context.window_manager.windows:
            win = utypes.wmWindow.get_fields(window)  # TODO: Check in old versions
            for handle in win.modalhandlers:
                if handle.type == 3:  # WM_HANDLER_TYPE_OP
                    return True
        return False

@bpy.app.handlers.persistent
def univ_drawer_update_tracker(_, deps):
    draw_objects_2d = Drawer2D.draw_objects
    draw_objects_3d = Drawer3D.draw_objects

    if not draw_objects_2d and not draw_objects_3d:
        return

    if bpy.context.mode != 'EDIT_MESH':
        Drawer2D.dirt = True
        Drawer3D.dirt = True
        draw_objects_2d.clear()
        draw_objects_3d.clear()
        return

    from bpy.types import Object
    for update_obj in deps.updates:
        obj = update_obj.id
        if type(obj) == Object:
            Drawer2D.dirt = True
            Drawer3D.dirt = True
            obj_name = obj.name
            try:
                del draw_objects_2d[obj_name]
            except: pass # noqa

            try:
                del draw_objects_3d[obj_name]
            except: pass # noqa

            if not draw_objects_2d and not draw_objects_3d:
                return

def has_update_tracker():
    for update_handler in bpy.app.handlers.depsgraph_update_post:
        if update_handler.__name__ == univ_drawer_update_tracker.__name__:
            return True
    return False

def safe_remove_update_tracker():
    if univ_settings().overlay_2d_enable or univ_settings().overlay_3d_enable:
        return
    for update_handler in reversed(bpy.app.handlers.depsgraph_update_post):
        if update_handler.__name__ == univ_drawer_update_tracker.__name__:
            bpy.app.handlers.depsgraph_update_post.remove(update_handler)


class Drawer2D:
    draw_objects: dict[str | list[DrawCallSeams2D]] = {}
    drawers: list[type[DrawCallSeams2D]] = []
    dirt = True
    handler = None
    sync = True
    frozen = False

    @classmethod
    def update_drawer_data(cls):
        drawers: list[type[DrawCallSeams2D | DrawCallConstraints2D]] = []
        if True:  # if constraints
            drawers.append(DrawCallConstraints2D)

        if True:  # if seam
            drawers.append(DrawCallSeams2D)

        cls.dirt = True
        cls.draw_objects.clear()
        cls.drawers = drawers


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

        if has_crash_modal_running():
            cls.dirt = True
            return

        unique_objects_with_uv = [obj for obj in bpy.context.objects_in_mode_unique_data if obj.data.uv_layers]
        for obj in unique_objects_with_uv:
            obj_id = obj.name
            if obj_id not in draw_objects:
                umesh = UMesh(bmesh.from_edit_mesh(obj.data), obj)

                draw_calls_seq = []
                for drawer in cls.drawers:
                    draw_call = drawer.init(umesh)
                    if draw_call:
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

            if not has_update_tracker():
                bpy.app.handlers.depsgraph_update_post.append(univ_drawer_update_tracker)
            cls.sync = bpy.context.tool_settings.use_uv_select_sync

    @classmethod
    def unregister(cls):
        if cls.handler:
            try:
                bpy.types.SpaceImageEditor.draw_handler_remove(cls.handler, 'WINDOW')
            except Exception as e:
                print(e)

        safe_remove_update_tracker()

        cls.draw_objects.clear()
        cls.drawers.clear()

        cls.dirt = True
        cls.frozen = False
        cls.handler = None

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

class Drawer3D:
    draw_objects: dict[str | list[DrawCall3D]] = {}
    drawers = []
    shaders_with_color = []
    mesh_extractors_with_batch = []
    dirt = True
    handler = None
    sync = True
    frozen = False
    uv_select_mode = ''

    prev_operators_size = 0
    first_operator_bl_idname = ''
    last_operator_bl_idname = ''

    @classmethod
    def update_drawer_data(cls):
        drawers = []
        shaders_with_color = []
        mesh_extractors_with_batch = []
        if True:  # if seam
            drawers.append(DrawerNonSyncSelectProcessing.draw_fn_2d)
            shaders_with_color.append((DrawerNonSyncSelectProcessing.get_shader(), DrawerNonSyncSelectProcessing.get_color()))
            mesh_extractors_with_batch.append((mesh_extract.extract_non_sync_select_data, DrawerNonSyncSelectProcessing.data_to_batch))

        cls.dirt = True
        cls.draw_objects.clear()
        cls.drawers = drawers
        cls.shaders_with_color = shaders_with_color
        cls.mesh_extractors_with_batch = mesh_extractors_with_batch
        cls.uv_select_mode = bpy.context.tool_settings.uv_select_mode

        # Clean update tracker data
        operators = bpy.context.window_manager.operators
        operators_size = len(operators)

        cls.prev_operators_size = operators_size
        if operators_size:
            cls.first_operator_bl_idname = operators[0].bl_idname
            if operators_size >= 2:
                cls.last_operator_bl_idname = operators[-1].bl_idname
            else:
                cls.last_operator_bl_idname = ''
        else:
            cls.first_operator_bl_idname = ''
            cls.last_operator_bl_idname = ''

    @classmethod
    def update(cls):
        if bpy.context.mode != 'EDIT_MESH':
            cls.dirt = True
            return

        if cls.detect_update():
            # print('Update detected')
            cls.dirt = True
            cls.draw_objects.clear()

        if bpy.context.tool_settings.uv_select_mode != cls.uv_select_mode:
            if bpy.context.tool_settings.uv_select_mode == 'FACE' and cls.uv_select_mode == 'ISLAND':
                cls.uv_select_mode = bpy.context.tool_settings.uv_select_mode
            else:
                cls.update_drawer_data()

        draw_objects = cls.draw_objects
        if not draw_objects:
            cls.dirt = True

        if has_crash_modal_running():
            cls.dirt = True
            return

        if not cls.dirt:
            return

        unique_objects_with_uv = [obj for obj in bpy.context.objects_in_mode_unique_data if obj.data.uv_layers]
        for obj in unique_objects_with_uv:
            obj_id = obj.name
            if obj_id not in draw_objects:
                umesh = UMesh(bmesh.from_edit_mesh(obj.data), obj)
                umesh.sync = False
                world_matrix = umesh.obj.matrix_world.copy()

                draw_calls_seq = []
                for drawer, (shader, color), (extract, to_batch) in zip(cls.drawers, cls.shaders_with_color, cls.mesh_extractors_with_batch):
                    data = extract(umesh)
                    draw_call = DrawCall3D(drawer, shader, color, to_batch(data, shader), world_matrix)
                    draw_calls_seq.append(draw_call)

                draw_objects[obj_id] = draw_calls_seq

        # Delete extra objects after renaming
        if len(unique_objects_with_uv) != len(draw_objects):
            names = {obj.name for obj in unique_objects_with_uv}
            for draw_obj_key in draw_objects.copy():
                if draw_obj_key not in names:
                    del draw_objects[draw_obj_key]

        cls.dirt = False
        from .. import utils
        utils.update_area_by_type('VIEW_3D')

    @classmethod
    def detect_update(cls):
        """Updates to uv selection in non-sync mode cannot be determined via update handlers, so operator history checks must be used."""
        operators = bpy.context.window_manager.operators
        operators_size = len(operators)

        if not cls.draw_objects or not operators_size:
            cls.prev_operators_size = operators_size
            if operators_size:
                cls.last_operator_bl_idname = operators[-1].bl_idname
                cls.first_operator_bl_idname = operators[0].bl_idname
            else:
                cls.last_operator_bl_idname = ''
                cls.first_operator_bl_idname = ''
            return False

        last_operator_bl_idname = operators[-1].bl_idname
        if cls.prev_operators_size != operators_size:
            cls.prev_operators_size = operators_size
            # Skip non-uv select operators
            if (not last_operator_bl_idname.startswith('UV_OT') or
                    last_operator_bl_idname.startswith('UV_OT_univ_') or
                    'select' not in last_operator_bl_idname):
                # NOTE: Control update UniV operators manually
                cls.last_operator_bl_idname = last_operator_bl_idname
                if operators_size == 1:
                    cls.first_operator_bl_idname = ''
                return False
            return True

        if operators_size == 1:
            if cls.last_operator_bl_idname != last_operator_bl_idname:
                cls.last_operator_bl_idname = last_operator_bl_idname
                return True

        else:
            first_operator_bl_idname = operators[0].bl_idname
            if (cls.last_operator_bl_idname != last_operator_bl_idname and
                cls.first_operator_bl_idname != first_operator_bl_idname):
                cls.last_operator_bl_idname = last_operator_bl_idname
                cls.first_operator_bl_idname = first_operator_bl_idname
                return True
        return False


    @staticmethod
    def univ_drawer_3d_callback():
        if bpy.context.tool_settings.use_uv_select_sync:
            Drawer3D.dirt = True
            Drawer3D.draw_objects.clear()
            return

        if not any(area.ui_type == 'UV'
                   for win in bpy.context.window_manager.windows
                   for area in win.screen.areas):
            Drawer3D.dirt = True
            Drawer3D.draw_objects.clear()
            return

        Drawer3D.update()

        for draw_calls_seq in Drawer3D.draw_objects.values():
            for draw_call in draw_calls_seq:
                draw_call()

    @classmethod
    def register(cls):
        cls.unregister()
        if univ_settings().overlay_3d_enable:
            cls.sync = bpy.context.tool_settings.use_uv_select_sync
            cls.uv_select_mode = bpy.context.tool_settings.uv_select_mode
            Drawer3D.update_drawer_data()
            cls.handler =  bpy.types.SpaceView3D.draw_handler_add(
                cls.univ_drawer_3d_callback, (), 'WINDOW', 'POST_VIEW')

            if not has_update_tracker():
                bpy.app.handlers.depsgraph_update_post.append(univ_drawer_update_tracker)

    @classmethod
    def unregister(cls):
        if cls.handler:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(cls.handler, 'WINDOW')
            except Exception as e:
                print(e)

        safe_remove_update_tracker()

        cls.draw_objects.clear()
        cls.mesh_extractors_with_batch.clear()
        cls.shaders_with_color.clear()
        cls.drawers.clear()

        cls.dirt = True
        cls.frozen = False
        cls.handler = None
        cls.uv_select_mode = ''

    @staticmethod
    def append_handler_with_delay():
        try:
            from ..preferences import univ_settings
            if univ_settings().overlay_3d_enable:
                Drawer3D.register()
        except Exception as e:
            print('UniV: Failed to add a handler for Drawer2D system.', e)

    @staticmethod
    def update_data(self, _context):
        if self.overlay_3d_enable:
            Drawer3D.register()
        else:
            Drawer3D.unregister()

    @classmethod
    @contextlib.contextmanager
    def freeze(cls):
        cls.frozen = True
        try:
            yield
        finally:
            cls.frozen = False
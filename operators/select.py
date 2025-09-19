# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu
import math

from math import sqrt, isclose
from bl_math import lerp
from mathutils import Vector
from bpy.props import *
from bpy.types import Operator
from bmesh.types import BMFace, BMLoop
from gpu_extras.batch import batch_for_shader
from time import perf_counter as time
from collections.abc import Callable

from .. import utils
from .. import utypes
from ..preferences import prefs, univ_settings
from ..utypes import Islands, AdvIslands, AdvIsland, BBox, UMeshes, MeshIslands, UnionIslands

from ..utils import (
    face_centroid_uv,
    is_boundary_non_sync,
)


uv_handle = None
start = time()
shader: gpu.types.GPUShader | None = None
batch: gpu.types.GPUBatch | None = None


def draw_callback_px(color):
    global shader
    global batch
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

    for a in bpy.context.screen.areas:
        if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
            a.tag_redraw()


def uv_area_draw_timer():
    global start
    global uv_handle
    if uv_handle is None:
        return
    counter = time() - start
    if counter < 0.8:
        return 0.2
    bpy.types.SpaceImageEditor.draw_handler_remove(uv_handle, 'WINDOW')

    for a in bpy.context.screen.areas:
        if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
            a.tag_redraw()
    uv_handle = None
    return


def add_draw_rect(data, color=(1, 1, 0, 1)):
    global start
    global shader
    global batch
    global uv_handle

    start = time()
    if getattr(bpy.context.preferences.system, "gpu_backend", None) == "VULKAN":
        shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    else:
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR' if bpy.app.version < (3, 5, 0) else 'UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": data})

    if not (uv_handle is None):
        bpy.types.SpaceImageEditor.draw_handler_remove(uv_handle, 'WINDOW')

    uv_handle = bpy.types.SpaceImageEditor.draw_handler_add(draw_callback_px, (color,), 'WINDOW', 'POST_VIEW')
    bpy.app.timers.register(uv_area_draw_timer)


class UNIV_OT_SelectLinked(Operator):
    bl_idname = 'uv.univ_select_linked'
    bl_label = 'Linked'
    bl_description = "Select all UV vertices linked to the active UV map"
    bl_options = {'REGISTER', 'UNDO'}

    deselect: bpy.props.BoolProperty(name='Deselect', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.deselect = event.ctrl
        return self.execute(context)

    def execute(self, context):
        if self.deselect is False:
            umeshes = UMeshes.calc(verify_uv=False)
            umeshes.fix_context()

            if context.area.ui_type == 'UV':
                return bpy.ops.uv.select_linked()
            if uv_areas := [area for area in context.screen.areas if area.ui_type == 'UV']:
                with context.temp_override(area=uv_areas[0]):  # noqa
                    return bpy.ops.uv.select_linked()
            return {'CANCELLED'}
        else:
            sync = bpy.context.scene.tool_settings.use_uv_select_sync
            return self.deselect_linked(sync=sync)

    def deselect_linked(self, sync):
        umeshes = UMeshes(report=self.report)
        umeshes.fix_context()

        if sync and umeshes.elem_mode == 'VERT':
            for umesh in umeshes:
                if umesh.is_full_vert_selected or umesh.is_full_vert_deselected:
                    umesh.update_tag = False
                    continue
                has_full_selected = False
                half_selected = []
                if islands := Islands.calc_visible(umesh):
                    for island in islands:
                        select_info = island.info_select()
                        if select_info == utypes.eInfoSelectFaceIsland.HALF_SELECTED:
                            half_selected.append(island)
                        elif select_info == utypes.eInfoSelectFaceIsland.FULL_SELECTED:
                            has_full_selected |= True
                            island.set_tag()

                is_update = bool(half_selected)

                if is_update and not has_full_selected:
                    for half_sel in half_selected:
                        for f in half_sel:
                            for v in f.verts:
                                v.select = False

                elif is_update and has_full_selected:
                    for half_sel in half_selected:
                        for f in half_sel:
                            verts = f.verts
                            if any(f.tag for v in verts for f in v.link_faces):
                                continue
                            for v in verts:
                                v.select = False

                if is_update:
                    umesh.bm.select_flush_mode()

                umesh.update_tag = is_update
            return umeshes.update(info='No islands for deselect')

        if sync and umeshes.elem_mode == 'EDGE':
            for umesh in umeshes:
                if umesh.is_full_edge_selected or umesh.is_full_edge_deselected:
                    umesh.update_tag = False
                    continue
                has_full_selected = False
                half_selected = []
                if islands := Islands.calc_visible(umesh):
                    for island in islands:
                        select_info = island.info_select()
                        if select_info == utypes.eInfoSelectFaceIsland.HALF_SELECTED:
                            half_selected.append(island)
                        elif select_info == utypes.eInfoSelectFaceIsland.FULL_SELECTED:
                            has_full_selected |= True
                            island.set_tag()

                is_update = bool(half_selected)

                if is_update and not has_full_selected:
                    for half_sel in half_selected:
                        for f in half_sel:
                            for e in f.edges:
                                e.select = False

                elif is_update and has_full_selected:
                    for half_sel in half_selected:
                        for f in half_sel:
                            edges = f.edges
                            for e in edges:
                                if any(ff.tag for ff in e.link_faces):
                                    continue
                                e.select = False

                if is_update:
                    umesh.bm.select_flush_mode()

                umesh.update_tag = is_update
            return umeshes.update(info='No islands for deselect')

        for umesh in umeshes:
            if sync and umeshes.elem_mode == 'FACE':
                if umesh.is_full_edge_selected or umesh.is_full_edge_deselected:
                    umesh.update_tag = False
                    continue
            is_update = False
            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    if update_state := (island.info_select() == utypes.eInfoSelectFaceIsland.HALF_SELECTED):
                        island.select = False
                    is_update |= update_state
            umesh.update_tag = is_update

            if is_update and sync:
                umesh.bm.select_flush_mode()

        return umeshes.update(info='No islands for deselect')


class UNIV_OT_Select_By_Cursor(Operator):
    bl_idname = "uv.univ_select_by_cursor"
    bl_label = "Cursor"
    bl_description = "Select by Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    face_mode: BoolProperty(name='Face Mode', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.face_mode = event.alt

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITIONAL'
        else:
            self.mode = 'SELECT'

        return self.execute(context)

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'INFO'}, f"UV area not found")
            return {'CANCELLED'}

        umeshes = UMeshes(report=self.report)
        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        umeshes.filter_by_visible_uv_faces()

        tile_co = utils.get_tile_from_cursor()
        view_rect = BBox.init_from_minmax(tile_co, tile_co+Vector((1, 1)))
        view_rect.pad(Vector((-2e-08, -2e-08)))

        view_island = AdvIsland()
        view_island._bbox = view_rect
        view_island.flat_coords = view_rect.draw_data_tris()

        if self.mode == 'ADDITIONAL':
            self.additional(umeshes, view_island)
        elif self.mode == 'DESELECT':
            self.deselect(umeshes, view_island)
        else:
            self.select(umeshes, view_island)

        add_draw_rect(view_rect.draw_data_lines())

        return umeshes.update()

    def select(self, umeshes, view_island):
        for umesh in umeshes:
            if self.face_mode:
                uv = umesh.uv
                bb = view_island.bbox
                face_select_set = utils.face_select_set_func(umesh)
                for f in utils.calc_visible_uv_faces_iter(umesh):
                    face_select_set(f, face_centroid_uv(f, uv) in bb)
            else:
                adv_islands = AdvIslands.calc_visible(umesh)
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords()

                for island in adv_islands:
                    island.select = island.is_overlap(view_island)
                umesh.update_tag = bool(adv_islands)

    def additional(self, umeshes: UMeshes, view_island):
        for umesh in umeshes:
            has_update = False
            if not umesh.has_full_selected_uv_faces():
                if self.face_mode:
                    uv = umesh.uv
                    bb = view_island.bbox
                    face_select_set = utils.face_select_set_func(umesh)
                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if face_centroid_uv(f, uv) in bb:
                            face_select_set(f, True)
                            has_update = True
                else:
                    adv_islands = AdvIslands.calc_visible(umesh)
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()

                    for island in adv_islands:
                        select_info = island.info_select()
                        if select_info == utypes.eInfoSelectFaceIsland.FULL_SELECTED:
                            continue
                        if island.is_overlap(view_island):
                            island.select = True
                            has_update = True

            umesh.update_tag = has_update

    def deselect(self, umeshes, view_island):
        has_update = False
        for umesh in umeshes:
            if umesh.has_selected_uv_faces():
                if self.face_mode:
                    uv = umesh.uv
                    bb = view_island.bbox
                    face_select_set = utils.face_select_set_func(umesh)

                    for f in utils.calc_visible_uv_faces_iter(umesh):
                        if face_centroid_uv(f, uv) in bb:
                            face_select_set(f, False)
                            has_update = True
                else:
                    adv_islands = AdvIslands.calc_visible(umesh)
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()

                    for island in adv_islands:
                        select_info = island.info_select()
                        if select_info == utypes.eInfoSelectFaceIsland.UNSELECTED:
                            continue
                        if island.is_overlap(view_island):
                            island.select = False
                            has_update = True
            umesh.update_tag = has_update


class UNIV_OT_Select_Square_Island(Operator):
    bl_idname = 'uv.univ_select_square_island'
    bl_label = 'Square'
    bl_description = 'Select Square Island'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    shape: EnumProperty(name='Shape', default='HORIZONTAL', items=(
        ('HORIZONTAL', 'Horizontal', ''),
        ('SQUARE', 'Square', ''),
        ('VERTICAL', 'Vertical', ''),
    ))

    threshold: FloatProperty(name='Square Threshold', default=0.05, min=0, max=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITIONAL'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        if self.umeshes.sync:
            self.umeshes.elem_mode = 'FACE'

        self.umeshes.filter_by_visible_uv_faces()

        if self.mode == 'DESELECT':
            self.deselect()
        elif self.mode == 'ADDITIONAL':
            self.addition()
        else:
            self.select()
        self.umeshes.silent_update()
        return {'FINISHED'}

    def select(self):
        for umesh in self.umeshes:
            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    island.select = self.is_target_island(island)
            umesh.update_tag = bool(islands)

    def deselect(self):
        # TODO: Fix calc_extended_any_elem_with_mark_seam for face/island elem mode
        if self.umeshes.sync or self.umeshes.elem_mode in ('FACE', 'ISLAND'):
            island_calc_type = Islands.calc_extended_with_mark_seam
        else:
            island_calc_type = Islands.calc_extended_any_elem_with_mark_seam

        for umesh in self.umeshes:
            update_tag = False
            if umesh.has_selected_uv_faces():
                for island in island_calc_type(umesh):
                    if self.is_target_island(island):
                        island.select = False
                        update_tag = True
            umesh.update_tag = update_tag

    def addition(self):
        for umesh in self.umeshes:
            update_tag = False
            if not umesh.has_full_selected_uv_faces():
                for island in Islands.calc_non_full_selected_with_mark_seam(umesh):
                    if self.is_target_island(island):
                        island.select = True
                        update_tag = True
            umesh.update_tag = update_tag

    def is_target_island(self, island):
        percent, close_to_square = self.percent_and_close_to_square(island)
        if self.shape == 'SQUARE':
            return close_to_square
        elif self.shape == 'HORIZONTAL':
            return percent > 0 and not close_to_square
        else:
            return percent < 0 and not close_to_square

    def percent_and_close_to_square(self, island):
        bbox = island.calc_bbox()
        width = bbox.width
        height = bbox.height

        if width == 0 and height == 0:
            width = height = 1
        elif width == 0:
            width = 1e-06
        elif height == 0:
            height = 1e-06

        percent = (width - height) / height
        return percent, math.isclose(percent, 0, abs_tol=self.threshold)


class UNIV_OT_Select_Border(Operator):
    bl_idname = 'uv.univ_select_border'
    bl_label = 'Border'
    bl_description = 'Select border edges'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    border_mode: EnumProperty(name='Border', default='BORDER', items=(
        ('BORDER', 'Border', ''),
        ('BORDER_BETWEEN', 'Border Between', ''),
        ('BORDER_EDGE_BY_ANGLE', 'Border Edge by Angle', ''),
        ('ALL_EDGE_BY_ANGLE', 'All Edge by Angle', ''),
    ))

    edge_dir: EnumProperty(name='Direction', default='HORIZONTAL', items=(
        ('BOTH', 'Both', ''),
        ('HORIZONTAL', 'Horizontal', ''),
        ('VERTICAL', 'Vertical', ''),
    ))

    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)
    angle: FloatProperty(name='Angle', default=math.radians(5), min=0, max=math.radians(45.001), subtype='ANGLE')

    def draw(self, context):
        if self.border_mode in ('BORDER_EDGE_BY_ANGLE', 'ALL_EDGE_BY_ANGLE'):
            row = self.layout.row(align=True)
            row.prop(self, 'edge_dir', expand=True)
            layout = self.layout
            layout.prop(self, 'angle', slider=True)
            layout.prop(self, 'use_correct_aspect')

        col = self.layout.column(align=True)
        col.prop(self, 'border_mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_vec = Vector((1, 0))
        self.y_vec = Vector((0, 1))
        self.angle_45 = math.pi / 4
        self.angle_135 = math.pi * 0.75
        self.edge_orient = self.x_vec
        self.negative_ange = 0
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.border_mode = 'BORDER_BETWEEN' if event.alt else 'BORDER'

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        if context.scene.tool_settings.use_uv_select_sync:  # TODO: Adapt for new sync system
            bpy.ops.uv.univ_sync_uv_toggle()  # noqa

        if self.border_mode in ('BORDER_EDGE_BY_ANGLE', 'ALL_EDGE_BY_ANGLE'):
            return self.select_edge_by_angle()

        self.umeshes = UMeshes(report=self.report)
        if self.umeshes.elem_mode not in ('EDGE', 'VERT'):
            self.umeshes.elem_mode = 'EDGE'

        self.umeshes.filter_by_visible_uv_faces()

        if self.border_mode == 'BORDER':
            self.select_border()
        else:
            self.select_border_between()
        self.umeshes.silent_update()
        return {'FINISHED'}

    def select_border(self):
        # TODO: Add match by mark seam
        for umesh in self.umeshes:
            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            edge_select_get = utils.edge_select_get_func(umesh)
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            if self.mode == 'SELECT':  # TODO: Add behavior, border by selected faces
                for crn in corners:
                    edge_select_set(crn, is_boundary_non_sync(crn, uv))

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary_non_sync(crn, uv):
                        self.deselect_edge_with_linked_edges_for_border(crn, uv, edge_select_set)
            else:  # 'ADDITION'
                for crn in corners:
                    if edge_select_get(crn):
                        continue
                    if is_boundary_non_sync(crn, uv):
                        edge_select_set(crn, True)

    def select_border_between(self):
        # bpy.ops.uv.select_all(action='DESELECT')
        for umesh in self.umeshes:
            uv = umesh.uv
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            if self.mode == 'SELECT':
                islands = Islands.calc_extended(umesh)

                for _f in umesh.bm.faces:  # TODO: Add deselect method in umesh
                    for _crn in _f.loops:
                        _crn_uv = _crn[uv]
                        _crn_uv.select = False
                        _crn_uv.select_edge = False

                islands.indexing(force=False)
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:  # No match hidden faces
                                continue
                            if shared_crn.face.index != f.index:
                                edge_select_set(crn, True)
                                edge_select_set(shared_crn, True)

            elif self.mode == 'DESELECT':
                islands = Islands.calc_extended(umesh)

                islands.indexing(force=False)
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:
                                continue
                            if shared_crn.face.index != f.index:
                                self.deselect_edge_with_linked_edges_for_border(crn, uv, edge_select_set)
                                self.deselect_edge_with_linked_edges_for_border(shared_crn, uv, edge_select_set)
            else:  # 'ADDITION'
                islands = Islands.calc_extended(umesh)

                islands.indexing(force=False)
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:
                                continue
                            if shared_crn.face.index != f.index:
                                edge_select_set(crn, True)
                                edge_select_set(shared_crn, True)
                umesh.update_tag = bool(islands)

    # Select Edge by Angle

    def select_edge_by_angle(self):
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.elem_mode = 'EDGE'

        if self.use_correct_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=False)

        self.edge_orient = self.x_vec if self.edge_dir == 'HORIZONTAL' else self.y_vec
        self.negative_ange = math.pi - self.angle

        if self.border_mode == 'BORDER_EDGE_BY_ANGLE':
            if self.edge_dir == 'BOTH':
                self.select_both_border()
            else:
                self.select_hv_border()
        else:
            if self.edge_dir == 'BOTH':
                self.select_both()
            else:
                self.select_hv()

        return self.umeshes.update()

    def select_hv(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            aspect_for_x = umesh.aspect
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    a = vec.angle(self.edge_orient, 0)

                    between_angle = a <= self.angle or a >= self.negative_ange
                    edge_select_set(crn, between_angle)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    a = vec.angle(self.edge_orient, 0)

                    if a <= self.angle or a >= self.negative_ange:
                        edge_select_set(crn, False)

            else:  # 'ADDITIONAL'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if edge_select_get(crn):
                        continue

                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    a = vec.angle(self.edge_orient, 0)

                    if a <= self.angle or a >= self.negative_ange:
                        edge_select_set(crn, True)

    def select_both(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            aspect_for_x = umesh.aspect
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    x_angle = vec.angle(self.x_vec, 0)
                    y_angle = vec.angle(self.y_vec, 0)

                    between_angle = x_angle <= self.angle or x_angle >= self.negative_ange or \
                        y_angle <= self.angle or y_angle >= self.negative_ange
                    edge_select_set(crn, between_angle)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    x_angle = vec.angle(self.x_vec, 0)
                    y_angle = vec.angle(self.y_vec, 0)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        edge_select_set(crn, False)

            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if edge_select_get(crn):
                        continue

                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    x_angle = vec.angle(self.x_vec, 0)
                    y_angle = vec.angle(self.y_vec, 0)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        edge_select_set(crn, True)

    def select_both_border(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            aspect_for_x = umesh.aspect
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                for crn in corners:
                    if is_boundary_non_sync(crn, uv):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        x_angle = vec.angle(self.x_vec, 0)
                        y_angle = vec.angle(self.y_vec, 0)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            edge_select_set(crn, True)
                            continue
                    edge_select_set(crn, False)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary_non_sync(crn, uv):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        x_angle = vec.angle(self.x_vec, 0)
                        y_angle = vec.angle(self.y_vec, 0)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            self.deselect_edge_with_linked_edges_for_border(crn, uv, edge_select_set)

            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if edge_select_get(crn):
                        continue

                    if is_boundary_non_sync(crn, uv):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        x_angle = vec.angle(self.x_vec, 0)
                        y_angle = vec.angle(self.y_vec, 0)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            edge_select_set(crn, True)

    def select_hv_border(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            aspect_for_x = umesh.aspect
            edge_select_set = utils.edge_select_linked_set_func(umesh)
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                for crn in corners:
                    if is_boundary_non_sync(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        vec.x *= aspect_for_x
                        a = vec.angle(self.edge_orient, 0)

                        if a <= self.angle or a >= self.negative_ange:
                            edge_select_set(crn, True)
                            continue
                    edge_select_set(crn, False)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary_non_sync(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        vec.x *= aspect_for_x
                        a = vec.angle(self.edge_orient, 0)

                        if a <= self.angle or a >= self.negative_ange:
                            self.deselect_edge_with_linked_edges_for_border(crn, uv, edge_select_set)

            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)  # TODO: add calc_unselected_uv_edges func
                for crn in corners:
                    if edge_select_get(crn):
                        continue
                    if is_boundary_non_sync(crn, uv):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        a = vec.angle(self.edge_orient, 0)

                        if a <= self.angle or a >= self.negative_ange:
                            edge_select_set(crn, True)

    @staticmethod
    def deselect_edge_with_linked_edges_for_border(crn, uv, select_linked_func):
        for crn_ in utils.linked_crn_uv_unordered_included(crn, uv):
            select_linked_func(crn_, False)
            select_linked_func(crn_.link_loop_prev, False)

        for crn_ in utils.linked_crn_uv_unordered_included(crn.link_loop_next, uv):
            select_linked_func(crn_, False)
            select_linked_func(crn_.link_loop_prev, False)


class UNIV_OT_Select_Pick(Operator):
    bl_idname = 'uv.univ_select_pick'
    bl_label = 'Pick Select'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        self.umeshes = UMeshes()
        self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        self.mouse_pos = utils.get_mouse_pos(context, event)
        return self.pick_select()

    def pick_select(self):
        if self.umeshes.sync and not self.select:
            self.umeshes.elem_mode = 'FACE'

        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            if self.select:
                if umesh.has_full_selected_uv_faces():
                    continue
            else:
                if not umesh.has_selected_uv_verts():
                    continue

            for isl in Islands.calc_visible_with_mark_seam(umesh):
                if self.select:  # Skip full selected island
                    if isl.is_full_face_selected():
                        continue
                else:  # Skip full deselected islands
                    if not isl.is_full_face_selected():
                        continue
                hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            return {'CANCELLED'}

        umesh = hit.island.umesh
        hit.island.select = self.select
        # TODO:Notify if has extra selection in sync non face mode
        umesh.update()

        return {'FINISHED'}


class UNIV_OT_SelectLinkedPick_VIEW3D(bpy.types.Macro):
    bl_idname = 'mesh.univ_select_linked_pick'
    bl_label = 'Select Linked Pick'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment


class UNIV_OT_DeselectLinkedPick_VIEW3D(bpy.types.Macro):
    bl_idname = 'mesh.univ_deselect_linked_pick'
    bl_label = 'Deselect Linked Pick'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment


class UNIV_OT_SelectLinked_VIEW3D(Operator):
    bl_idname = 'mesh.univ_select_linked'
    bl_label = 'Select Linked'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)
    delimit: EnumProperty(name='Delimit', default='NORMAL',
                          items=(('NORMAL', 'Normal', ''),
                                 ('MATERIAL', 'Material', ''),
                                 ('SEAM', 'Seam', ''),
                                 ('SHARP', 'Sharp', ''),
                                 ('UV', 'UVs', '')
                                 ))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'select')
        self.layout.prop(self, 'delimit', expand=True)

    def execute(self, context):
        if self.select:
            return bpy.ops.mesh.select_linked(delimit={self.delimit})
        else:
            self.deselect()
            return {'FINISHED'}

    def deselect(self):
        umeshes = UMeshes.calc_any_unique(verify_uv=False)
        umeshes.set_sync(True)
        umeshes.filter_by_partial_selected_uv_faces()
        umeshes.update_tag = False

        match self.delimit:
            case 'NORMAL':
                calc_type = MeshIslands.calc_iter_non_manifold_ex
            case 'MATERIAL':
                calc_type = MeshIslands.calc_by_material_non_manifold_iter_ex
            case 'SEAM':
                calc_type = MeshIslands.calc_with_markseam_non_manifold_iter_ex
            case 'SHARP':
                calc_type = MeshIslands.calc_by_sharps_non_manifold_iter_ex
            case 'UV':
                calc_type = Islands.calc_with_markseam_iter_ex
            case _:
                raise

        for umesh in umeshes:
            if self.delimit == 'UV' and not umesh.obj.data.uv_layers:
                calc_type_ = MeshIslands.calc_with_markseam_non_manifold_iter_ex
            else:
                calc_type_ = calc_type
            Islands.tag_filter_visible(umesh)
            for isl in calc_type_(umesh):
                if not utils.all_equal((f.select for f in isl)):
                    umeshes.elem_mode = 'FACE'
                    for f in isl:
                        f.select = False
                    umesh.update_tag = True
        umeshes.update()


# TODO: Grow after 0.3 (within 0.3-1.5 sec) sec and no effect repeat - without seam clamp
class UNIV_OT_Select_Grow_Base(Operator):
    bl_label = 'Grow'
    bl_options = {'REGISTER', 'UNDO'}

    grow: BoolProperty(name='Select', default=True)
    # TODO: Improve clamp
    clamp_on_seam: BoolProperty(name='Clamp on Seam', default=True,
                                description="Edge Grow clamp on edges with seam, but if the original edge has seam, this effect is ignored")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc_islands: Callable = Callable
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)


class UNIV_OT_Select_Grow(UNIV_OT_Select_Grow_Base):
    bl_idname = 'uv.univ_select_grow'
    bl_description = "Select more UV vertices connected to initial selection\n\n" \
                     "Default - Grow\n" \
                     "Ctrl or Alt - Shrink\n\n" \
                     "Has [Ctrl + Scroll Up/Down] keymap"

    def execute(self, context):
        self.umeshes = UMeshes()
        self.umeshes.filter_by_partial_selected_uv_elem_by_mode()

        # TODO: Implement without island calc (use linked with pair iter and mark seam)
        if self.clamp_on_seam:
            self.calc_islands = Islands.calc_visible_with_mark_seam
        else:
            self.calc_islands = Islands.calc_visible

        if self.grow:
            return self.grow_select()
        else:
            return self.shrink()

    def is_sticky_off_in_face_mode_non_sync(self):
        return (self.umeshes.elem_mode == 'FACE' and not self.umeshes.sync and
                bpy.context.scene.tool_settings.uv_sticky_select_mode == 'DISABLED')

    def grow_select(self):
        sync = self.umeshes.sync
        if self.clamp_on_seam:
            linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
        else:
            linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

        for umesh in self.umeshes:
            uv = umesh.uv
            islands = self.calc_islands(umesh)
            islands.indexing()
            to_select = []
            to_select_append = to_select.append
            if self.is_sticky_off_in_face_mode_non_sync():
                face_select_get = utils.face_select_get_func(umesh)

                for f in utils.calc_unselected_uv_faces_iter(umesh):
                    if any(face_select_get(l_crn.face)
                           for crn in f.loops
                           for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                        to_select_append(f)
            else:
                # TODO: Remove calc island
                for idx, isl in enumerate(islands):
                    if sync:
                        if self.umeshes.elem_mode == 'FACE':
                            # To optimize performance, the logic should be split based on whether there are many selected faces or just a few.
                            for f in isl:
                                if not f.select and any(l_crn.face.select for crn in f.loops if crn.vert.select
                                                        for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx)):
                                    to_select_append(f)
                        else:
                            if umesh.is_full_face_deselected:  # optimize case when only one vertex/edge selected
                                for f in isl:
                                    if any(v.select for v in f.verts):
                                        to_select_append(f)
                            else:
                                for f in isl:
                                    if not f.select and self.is_grow_face(f, uv, idx):
                                        to_select_append(f)
                    else:
                        if self.umeshes.elem_mode == 'EDGE':
                            for f in isl:
                                selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    to_select_append(f)
                        else:
                            for f in isl:
                                selected_corners = sum(crn[uv].select for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    to_select_append(f)
            if to_select:
                if sync:
                    for f in to_select:
                        f.select = True
                else:
                    if self.is_sticky_off_in_face_mode_non_sync():
                        face_select_set = utils.face_select_set_func(umesh)
                        for f in to_select:
                            face_select_set(f, True)
                    else:
                        for f in to_select:
                            idx = f.index
                            for crn in f.loops:
                                if not crn[uv].select:
                                    for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                                        l_crn[uv].select = True
                        for isl in islands:
                            for f in isl:
                                for crn in f.loops:
                                    if crn[uv].select and crn.link_loop_next[uv].select:
                                        crn[uv].select_edge = True
                umesh.update()

        return {'FINISHED'}

    def shrink(self):
        sync = self.umeshes.sync
        if self.clamp_on_seam:
            linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
        else:
            linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

        for umesh in self.umeshes:
            uv = umesh.uv
            islands = self.calc_islands(umesh)
            islands.indexing()
            to_deselect = []
            to_deselect_append = to_deselect.append

            if self.is_sticky_off_in_face_mode_non_sync():
                face_select_get = utils.face_select_get_func(umesh)
                for f in utils.calc_selected_uv_faces(umesh):
                    if not all(face_select_get(l_crn.face)
                               for crn in f.loops
                               for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                        to_deselect_append(f)

            else:
                for idx, isl in enumerate(islands):
                    if sync:
                        if self.umeshes.elem_mode == 'FACE':
                            for f in isl:
                                if f.select and any(not l_crn.face.select for crn in f.loops
                                                    for l_crn in utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx)):
                                    to_deselect_append(f)
                        else:
                            if umesh.is_full_face_deselected:
                                for f in isl:
                                    if not f.select:
                                        if any(v.select for v in f.verts):
                                            to_deselect_append(f)
                            else:
                                for f in isl:
                                    if not f.select and self.is_shrink_face(f, uv, idx):
                                        to_deselect_append(f)
                    else:
                        if self.umeshes.elem_mode == 'EDGE':
                            for f in isl:
                                selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    to_deselect_append(f)
                        else:
                            for f in isl:
                                selected_corners = sum(crn[uv].select for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    to_deselect_append(f)

            if to_deselect:
                if sync:
                    if self.umeshes.elem_mode == 'FACE':
                        for f in to_deselect:
                            f.select = False
                    else:
                        for f in to_deselect:
                            f.select = False
                            for v in f.verts:
                                v.select = False
                    umesh.bm.select_flush(False)
                else:
                    if self.is_sticky_off_in_face_mode_non_sync():
                        face_select_set = utils.face_select_set_func(umesh)
                        for f in to_deselect:
                            face_select_set(f, False)
                    else:
                        # TODO: Implement face linked deselect with pair check and mark seam
                        for f in to_deselect:
                            idx = f.index
                            for crn in f.loops:
                                if crn[uv].select:
                                    for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                                        l_crn[uv].select = False

                        for isl in islands:
                            for f in isl:
                                for crn in f.loops:
                                    if not (crn[uv].select and crn.link_loop_next[uv].select):
                                        crn[uv].select_edge = False

                umesh.update()

        return {'FINISHED'}

    @staticmethod
    def is_grow_face(face: BMFace, uv, idx):
        for crn in face.loops:
            crn_vert = crn.vert
            if not crn_vert.select:
                continue

            if len(utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx)) + 1 == len(crn_vert.link_loops):
                return True

            crn_edge = crn.edge

            if crn_edge.is_boundary and crn_edge.select:
                return True

            if crn_vert.select and crn.link_loop_next.vert.select and not (crn_edge.seam or utils.is_boundary_sync(crn, uv)):
                return True

        return False

    @staticmethod
    def handle_deselect_vertex(face: BMFace, idx):
        for v in face.verts:
            if v.select:
                for ff in v.link_faces:
                    if ff.index not in (idx, -1):
                        if ff.select:
                            break
                else:
                    v.select = False

    def is_shrink_face(self, face: BMFace, _uv, idx):
        has_selected_verts = False  # noqa
        for v in face.verts:
            if v.select:
                has_selected_verts = True
                if not v.is_boundary:
                    break
                for ff in v.link_faces:
                    if ff.index not in (idx, -1):
                        break
        else:
            return True

        if has_selected_verts:
            for crn in face.loops:
                if crn.vert.select:
                    for ff in crn.vert.link_faces:
                        if ff.index not in (idx, -1):
                            if ff.select:
                                self.handle_deselect_vertex(face, idx)
                                return False
            return True
        return False


class UNIV_OT_Select_Grow_VIEW3D(UNIV_OT_Select_Grow_Base):
    bl_idname = 'mesh.univ_select_grow'
    bl_description = "Select more vertices connected to initial selection\n\n" \
                     "Default - Grow\n" \
                     "Ctrl or Alt - Shrink\n\n" \
                     "Has [Ctrl + Scroll Up/Down] keymap"

    def execute(self, context):
        self.umeshes = UMeshes.calc_any_unique(verify_uv=False)

        self.umeshes.set_sync()
        self.calc_islands = MeshIslands.calc_visible_with_mark_seam if self.clamp_on_seam else MeshIslands.calc_visible
        if self.grow:
            return self.grow_select()
        else:
            return self.shrink()

    def grow_select(self):
        has_updates = False
        linked_crn_to_vert = utils.linked_crn_to_vert_with_seam_3d_iter if self.clamp_on_seam else utils.linked_crn_to_vert_3d_iter
        if self.umeshes.elem_mode == 'VERT':
            self.umeshes.filter_by_selected_mesh_verts()

            for umesh in self.umeshes:
                if umesh.is_full_vert_selected:
                    continue

                to_select = set()
                for v in umesh.bm.verts:
                    if not v.select:
                        continue
                    if v.is_wire:
                        to_select.update(ee
                                         for ee in v.link_edges
                                         if ee.is_wire and not ee.select and not ee.hide)
                    else:
                        selection_states_from_linked_faces = [f.select for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue

                        elif any(selection_states_from_linked_faces):
                            all_linked_faces_with_select = []
                            all_linked_faces_without_select = []

                            link_corners_to_vert = {crn for crn in v.link_loops if not crn.face.hide}
                            while link_corners_to_vert:
                                crn = link_corners_to_vert.pop()

                                linked_corners = set(crn_ for crn_ in linked_crn_to_vert(crn))
                                link_corners_to_vert -= linked_corners

                                faces = [crn_.face for crn_ in linked_corners]
                                faces.append(crn.face)

                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                            if all_linked_faces_with_select:
                                for faces in all_linked_faces_with_select:
                                    to_select.update(f for f in faces if not f.select)
                            else:
                                for faces in all_linked_faces_without_select:
                                    to_select.update(faces)

                        else:  # extend all visible unselected
                            for f in v.link_faces:
                                if not f.hide:
                                    to_select.add(f)
                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.bm.select_flush(True)
                    umesh.update()

        elif self.umeshes.elem_mode == 'EDGE':
            self.umeshes.filter_by_selected_mesh_edges()

            for umesh in self.umeshes:
                if umesh.is_full_edge_selected:
                    continue

                to_select = set()
                for e in umesh.bm.edges:
                    if not e.select:
                        continue
                    if e.is_wire:
                        to_select.update(ee
                                         for v in e.verts for ee in v.link_edges
                                         if ee.is_wire and not ee.select and not ee.hide)
                    else:
                        selection_states_from_linked_faces = [f.select for v in e.verts for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue
                        elif any(selection_states_from_linked_faces):
                            all_linked_faces_with_select = []
                            all_linked_faces_without_select = []

                            for crn in e.link_loops:
                                if crn.face.hide:
                                    continue
                                faces = list(crn_.face for crn_ in linked_crn_to_vert(crn))
                                faces.append(crn.face)

                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                                # Do not combine crn and crn.next in "faces", otherwise grow becomes redundant
                                faces = [crn_.face for crn_ in linked_crn_to_vert(crn.link_loop_next)]
                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                            if all_linked_faces_with_select:
                                for faces in all_linked_faces_with_select:
                                    to_select.update(f for f in faces if not f.select)
                            else:
                                for faces in all_linked_faces_without_select:
                                    to_select.update(faces)

                        else:  # extend all visible unselected
                            for crn in e.link_loops:
                                to_select.update(crn_.face for crn_ in linked_crn_to_vert(crn))
                                to_select.update(crn_.face for crn_ in linked_crn_to_vert(crn.link_loop_next))
                                if not crn.face.hide:
                                    to_select.add(crn.face)
                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.bm.select_flush(True)
                    umesh.update()
        else:
            self.umeshes.filter_by_selected_mesh_faces()

            for umesh in self.umeshes:
                if umesh.is_full_face_selected:
                    continue

                to_select = set()
                for f in utils.calc_selected_uv_faces(umesh):
                    for crn in f.loops:
                        if all(ff.select for ff in crn.vert.link_faces):  # x2.5 performance
                            continue
                        to_select.update(crn_.face
                                         for crn_ in linked_crn_to_vert(crn)
                                         if not crn_.face.select)

                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.update()

        if not has_updates:
            self.report({'INFO'}, 'Not found faces for grow select')
        return {'FINISHED'}

    def shrink(self):
        has_updates = False

        linked_crn_to_vert = utils.linked_crn_to_vert_with_seam_3d_iter if self.clamp_on_seam else utils.linked_crn_to_vert_3d_iter
        if self.umeshes.elem_mode == 'VERT':
            self.umeshes.filter_by_selected_mesh_verts()

            for umesh in self.umeshes:
                if umesh.is_full_vert_selected:
                    continue

                to_deselect = set()
                for v in umesh.bm.verts:
                    if not v.select:
                        continue
                    if v.is_wire:
                        if any(ee.is_wire and not ee.select and not ee.hide for ee in v.link_edges):
                            to_deselect.add(v)
                    else:
                        selection_states_from_linked_faces = [f.select for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue

                        elif any(selection_states_from_linked_faces):
                            link_corners_to_vert = {crn for crn in v.link_loops if not crn.face.hide}
                            while link_corners_to_vert:
                                crn = link_corners_to_vert.pop()

                                linked_corners = set(crn_ for crn_ in linked_crn_to_vert(crn))
                                link_corners_to_vert -= linked_corners

                                if crn.face.select and all(crn_.face.select for crn_ in linked_corners):
                                    break
                            else:  # not break
                                to_deselect.add(v)

                        else:  # shrink all visible unselected
                            to_deselect.add(v)
                for v in to_deselect:
                    v.select = False

                if to_deselect:
                    has_updates = True
                    umesh.bm.select_flush(False)
                    umesh.update()

        elif self.umeshes.elem_mode == 'EDGE':
            self.umeshes.filter_by_selected_mesh_edges()

            for umesh in self.umeshes:
                if umesh.is_full_edge_selected:
                    continue

                to_deselect = set()
                for e in umesh.bm.edges:
                    if not e.select:
                        continue
                    if e.is_wire:
                        if any(ee.is_wire and not ee.select and not ee.hide
                               for v in e.verts for ee in v.link_edges):
                            to_deselect.add(e)
                    else:
                        selection_states_from_linked_faces = [f.select for v in e.verts for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue
                        elif any(selection_states_from_linked_faces):
                            for crn in e.link_loops:
                                if crn.face.hide:
                                    continue
                                if crn.face.select:
                                    if all(crn_.face.select for crn_ in linked_crn_to_vert(crn)) or \
                                            all(crn_.face.select for crn_ in linked_crn_to_vert(crn.link_loop_next)):
                                        break
                            else:  # not break
                                to_deselect.add(e)
                        else:  # shrink all visible unselected
                            to_deselect.add(e)
                for e in to_deselect:
                    e.select = False

                if to_deselect:
                    has_updates = True
                    umesh.bm.select_flush(False)

                    for e in to_deselect:
                        if e.is_wire:
                            continue
                        for v in e.verts:
                            for ee in v.link_edges:
                                if not ee.select or ee.is_wire:
                                    continue
                                if all(not f.select for f in ee.link_faces):
                                    ee.select = False

                    umesh.update()
        else:
            self.umeshes.filter_by_selected_mesh_faces()

            for umesh in self.umeshes:
                if umesh.is_full_face_selected:
                    continue

                to_deselect = set()
                for f in utils.calc_selected_uv_faces(umesh):
                    for crn in f.loops:
                        if all(ff.select for ff in crn.vert.link_faces):  # x2.5 performance
                            continue
                        if any(not crn_.face.select for crn_ in linked_crn_to_vert(crn)):
                            to_deselect.add(f)
                            break

                for f in to_deselect:
                    f.select = False

                if to_deselect:
                    has_updates = True
                    umesh.update()

        if not has_updates:
            self.report({'INFO'}, 'Not found faces for shrink deselect')
        return {'FINISHED'}


class UNIV_OT_Select_Edge_Grow_Base(Operator):
    bl_label = 'Edge Grow'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Edge Grow/Shrink Select\n\n" \
                     "Default - Grow Select \n" \
                     "Ctrl or Alt - Shrink Select\n\n" \
                     "Has [Alt + Scroll Up/Down] keymap, but it conflicts with the Frame Offset operator"

    clamp_on_seam: BoolProperty(name='Clamp on Seam', default=True,
                                description="Edge Grow clamp on edges with seam, but if the original edge has seam, this effect is ignored")
    grow: BoolProperty(name='Select', default=True, description='Grow/Shrink toggle')
    max_angle: FloatProperty(name='Angle', default=math.radians(20), min=math.radians(1), soft_min=math.radians(5), max=math.radians(90), subtype='ANGLE',
                             description="Max select angle.")
    prioritize_sharps: BoolProperty(name='Prioritize Sharps', default=True,
                                    description='Gives 35% priority to an edge that has a Mark Sharp, works if there are more than 4 linked edges.')
    boundary_by_boundary: BoolProperty(name='Boundary by Boundary', default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc_islands: Callable = Callable
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.grow:
            layout.prop(self, 'prioritize_sharps')
        layout.prop(self, 'boundary_by_boundary')
        layout.prop(self, 'clamp_on_seam')
        layout.prop(self, 'max_angle')
        layout.prop(self, 'grow')


class UNIV_OT_Select_Edge_Grow_VIEW2D(UNIV_OT_Select_Edge_Grow_Base):
    bl_idname = 'uv.univ_select_edge_grow'

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)

        if self.umeshes.elem_mode not in ('VERT', 'EDGE'):
            self.report({'INFO'}, f'Edge Grow not work in "{self.umeshes.elem_mode}" mode, run grow instead')
            return bpy.ops.uv.univ_select_grow(grow=self.grow, clamp_on_seam=self.clamp_on_seam)  # noqa

        if self.clamp_on_seam:
            self.calc_islands = Islands.calc_extended_any_edge_with_markseam
        else:
            self.calc_islands = Islands.calc_extended_any_edge

        self.umeshes.filter_by_selected_uv_edges()
        self.umeshes.update_tag = False

        if self.grow:
            self.grow_select()
            self.umeshes.update(info='Not found edges for grow select')
            return {'FINISHED'}

        self.shrink_select()
        # TODO: Deselect single vert in VERTEX mode when repeat press operator (when prev deselect failed)
        self.umeshes.update(info='Not found edges for shrink select')
        return {'FINISHED'}

    def grow_select(self):
        for umesh in self.umeshes:
            islands = self.calc_islands(umesh)
            islands.indexing()

            grew = []
            uv = umesh.uv
            for isl in islands:
                for crn in isl.calc_selected_edge_corners_iter():
                    with_seam = not self.clamp_on_seam or crn.edge.seam
                    selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                    if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                        if not with_seam:
                            if grow_prev_crn.edge.seam:
                                continue  # TODO: Remove continue?
                        grew.append(grow_prev_crn)

                    if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                        if not with_seam:
                            if grow_next_crn.edge.seam:
                                continue
                        grew.append(grow_next_crn)

            if umesh.sync:
                for grew_crn in grew:
                    grew_crn.edge.select = True
            else:
                for grew_crn in grew:
                    utils.select_crn_uv_edge_with_shared_by_idx(grew_crn, uv, force=True)
            umesh.update_tag = bool(grew)

    def shrink_select(self):
        for umesh in self.umeshes:
            islands = self.calc_islands(umesh)
            islands.indexing()

            uv = umesh.uv
            shrink = []
            for isl in islands:
                for crn in isl.calc_selected_edge_corners_iter():
                    # TODO: To avoid confusion, change with_seam to with_seam_clamp
                    #  (don't forget to change the logic itself, otherwise with_seam is interpreted as a match with seams).
                    with_seam = not self.clamp_on_seam or crn.edge.seam
                    selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                    if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                        if not with_seam and grow_prev_crn.edge.seam:
                            grow_prev_crn = None

                    if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                        if not with_seam and grow_next_crn.edge.seam:
                            grow_next_crn = None

                    if grow_prev_crn or grow_next_crn:
                        shrink.append(crn)

            if shrink:
                if umesh.sync:
                    edge_deselect = utils.edge_deselect_safe_3d_func(umesh)
                    for crn in shrink:
                        edge_deselect(crn)
                    umesh.bm.select_history.validate()  # Active elem validate
                else:
                    edge_select_set = utils.edge_select_linked_set_func(umesh)
                    for crn in shrink:
                        edge_select_set(crn, False)
                umesh.update_tag = True

    def grow_prev(self, crn, selected_dir, uv, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        prev_crn = crn.link_loop_prev
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        cur_linked_corners = utils.linked_crn_uv_by_island_index_unordered(crn, uv, crn.face.index)

        if is_clamped(cur_linked_corners, shared, prev_crn, with_seam, uv):
            return None

        if not len(cur_linked_corners):
            if selected_dir.angle(crn[uv].uv - prev_crn[uv].uv, max_angle) <= max_angle:
                return prev_crn
        elif len(cur_linked_corners) == 3 and len(crn.vert.link_loops) == 4 \
                and shared \
                and len(cur_quad_linked_crn_uv := utils.linked_crn_uv_by_idx(crn, uv)) == 3 \
                and utils.shared_linked_crn_by_idx(cur_quad_linked_crn_uv[1], uv):  # noqa # pylint:disable=used-before-assignment
            return cur_quad_linked_crn_uv[1]
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in cur_linked_corners:
                angle_ = selected_dir.angle(crn_[uv].uv - crn_.link_loop_next[uv].uv, max_angle)

                if self.prioritize_sharps:
                    if not crn.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_by_idx(crn_, uv))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(crn_[uv].uv - prev_crn_[uv].uv, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_by_idx(prev_crn_, uv))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_

            return min_crn
        return False

    def grow_next(self, crn, selected_dir, uv, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        next_crn = crn.link_loop_next
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        next_linked_corners = utils.linked_crn_uv_by_island_index_unordered(
            crn.link_loop_next, uv, crn.link_loop_next.face.index)

        if is_clamped(next_linked_corners, shared, next_crn, with_seam, uv):
            return None

        if not len(next_linked_corners):
            if selected_dir.angle(next_crn.link_loop_next[uv].uv - next_crn[uv].uv, max_angle) <= max_angle:
                return next_crn

        elif len(next_linked_corners) == 3 and len(next_crn.vert.link_loops) == 4 \
                and shared \
                and len(next_quad_linked_crn_uv := utils.linked_crn_uv_by_idx(next_crn, uv)) == 3 \
                and utils.shared_linked_crn_by_idx(next_quad_linked_crn_uv[1].link_loop_prev, uv):  # noqa # pylint:disable=used-before-assignment
            return next_quad_linked_crn_uv[1].link_loop_prev
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in next_linked_corners:
                angle_ = selected_dir.angle(crn_.link_loop_next[uv].uv - crn_[uv].uv, max_angle)

                if self.prioritize_sharps:
                    if not crn.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_by_idx(crn_, uv))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(prev_crn_[uv].uv - crn_[uv].uv, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_by_idx(prev_crn_, uv))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_
            return min_crn
        return False

    def is_clamped_by_selected_and_seams(self, linked_corners, shared, next_or_prev_crn, with_seam, uv):
        # Skip if selected or with seam
        if self.umeshes.sync:
            if next_or_prev_crn.edge.select:
                return True

            if with_seam:
                for crn__ in linked_corners:
                    if crn__.edge.select:
                        return True
                    if (prev_crn__ := crn__.link_loop_prev) != shared:
                        if prev_crn__.edge.select:
                            return True
            else:
                if next_or_prev_crn.edge.seam:
                    return True
                for crn__ in linked_corners:
                    crn_edge = crn__.edge
                    if crn_edge.select or crn_edge.seam:
                        return True
                    if (prev_crn__ := crn__.link_loop_prev) != shared:
                        prev_crn_edge = prev_crn__.edge
                        if prev_crn_edge.seam or prev_crn_edge.select:
                            return True
        else:
            if next_or_prev_crn[uv].select_edge:
                return True
            if with_seam:
                for crn__ in linked_corners:
                    if crn__[uv].select_edge:
                        return True
                    if (prev_crn__ := crn__.link_loop_prev) != shared:
                        if prev_crn__[uv].select_edge:
                            return True
            else:
                if next_or_prev_crn.edge.seam:
                    return True
                for crn__ in linked_corners:
                    if crn__[uv].select_edge or crn__.edge.seam:
                        return True
                    if (prev_crn__ := crn__.link_loop_prev) != shared:
                        if prev_crn__.edge.seam or prev_crn__[uv].select_edge:
                            return True


class UNIV_OT_Select_Edge_Grow_VIEW3D(UNIV_OT_Select_Edge_Grow_Base):
    bl_idname = 'mesh.univ_select_edge_grow'

    max_angle: FloatProperty(name='Angle', default=math.radians(40), min=math.radians(1), soft_min=math.radians(5), max=math.radians(90), subtype='ANGLE',
                             description="Max select angle.")

    def execute(self, context):
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)

        if self.umeshes.elem_mode not in ('VERT', 'EDGE'):
            return bpy.ops.mesh.univ_select_grow(grow=self.grow, clamp_on_seam=self.clamp_on_seam)  # noqa

        self.umeshes.set_sync()
        if self.clamp_on_seam:
            self.calc_islands = MeshIslands.calc_extended_any_edge_with_markseam
        else:
            self.calc_islands = MeshIslands.calc_extended_any_edge

        self.umeshes.update_tag = False
        if self.grow:
            self.grow_select()
            self.umeshes.update(info='Not found edges for grow select')
            return {'FINISHED'}

        self.shrink_select()
        self.umeshes.update(info='Not found edges for shrink select')
        return {'FINISHED'}

    def grow_select(self):
        for umesh in reversed(self.umeshes):
            if islands := self.calc_islands(umesh):
                islands.indexing()
                grew = []
                for isl in islands:
                    for crn in isl.calc_selected_edge_corners_iter():
                        with_seam = not self.clamp_on_seam or crn.edge.seam
                        selected_dir = crn.link_loop_next.vert.co - crn.vert.co

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam:
                                if grow_prev_crn.edge.seam:
                                    continue
                            grew.append(grow_prev_crn)

                        if grow_next_crn := self.grow_next(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam:
                                if grow_next_crn.edge.seam:
                                    continue
                            grew.append(grow_next_crn)

                for grew_crn in grew:
                    grew_crn.edge.select = True
                umesh.update_tag = bool(grew)

    def shrink_select(self):
        for umesh in self.umeshes:
            if islands := self.calc_islands(umesh):
                islands.indexing()
                shrink = []
                for isl in islands:
                    for crn in isl.calc_selected_edge_corners_iter():
                        with_seam = not self.clamp_on_seam or crn.edge.seam
                        selected_dir = crn.link_loop_next.vert.co - crn.vert.co

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam and grow_prev_crn.edge.seam:
                                grow_prev_crn = None

                        if grow_next_crn := self.grow_next(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam and grow_next_crn.edge.seam:
                                grow_next_crn = None

                        if grow_prev_crn or grow_next_crn:
                            shrink.append(crn)

                if shrink:
                    edge_deselect = utils.edge_deselect_safe_3d_func(umesh)
                    for crn in shrink:
                        edge_deselect(crn)
                    umesh.bm.select_history.validate()
                    umesh.update_tag = True

    def grow_prev(self, crn, selected_dir, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        prev_crn = crn.link_loop_prev
        shared = utils.shared_linked_crn_to_edge_by_idx(crn)
        cur_linked_corners = utils.linked_crn_to_vert_by_island_index_unordered(crn)

        if is_clamped(cur_linked_corners, shared, prev_crn, with_seam):
            return None

        if not len(cur_linked_corners):
            if selected_dir.angle(crn.vert.co - prev_crn.vert.co, max_angle) <= max_angle:
                return prev_crn
        elif len(cur_linked_corners) == 3 and len(crn.vert.link_loops) == 4 \
                and shared \
                and len(cur_quad_linked_crn_uv := utils.linked_crn_to_vert_by_idx_3d(crn)) == 3 \
                and utils.shared_linked_crn_to_edge_by_idx(cur_quad_linked_crn_uv[1]):  # noqa # pylint:disable=used-before-assignment
            return cur_quad_linked_crn_uv[1]
        # TODO: Implement border and border with quad
        # elif not shared and len(cur_linked_corners) == 1 \
        # and (shared_prev_crn := utils.shared_linked_crn_to_edge_by_idx(prev_crn))
        else:
            # TODO: Implement angles by normal projection
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in cur_linked_corners:
                angle_ = selected_dir.angle(crn_.vert.co - crn_.link_loop_next.vert.co, max_angle)
                if self.prioritize_sharps:
                    if not crn_.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(crn_))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(crn_.vert.co - prev_crn_.vert.co, max_angle)
                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(prev_crn_))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_

            return min_crn
        return False

    def grow_next(self, crn, selected_dir, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        next_crn = crn.link_loop_next
        shared = utils.shared_linked_crn_to_edge_by_idx(crn)
        next_linked_corners = utils.linked_crn_to_vert_by_island_index_unordered(next_crn)

        if is_clamped(next_linked_corners, shared, next_crn, with_seam):
            return None

        if not len(next_linked_corners):
            if selected_dir.angle(next_crn.link_loop_next.vert.co - next_crn.vert.co, max_angle) <= max_angle:
                return next_crn

        elif len(next_linked_corners) == 3 and len(next_crn.vert.link_loops) == 4 \
                and shared \
                and len(next_quad_linked_crn_uv := utils.linked_crn_to_vert_by_idx_3d(next_crn)) == 3 \
                and utils.shared_linked_crn_to_edge_by_idx(next_quad_linked_crn_uv[1].link_loop_prev):  # noqa # pylint:disable=used-before-assignment
            return next_quad_linked_crn_uv[1].link_loop_prev
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in next_linked_corners:
                angle_ = selected_dir.angle(crn_.link_loop_next.vert.co - crn_.vert.co, max_angle)

                if self.prioritize_sharps:
                    if not crn_.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(crn_))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(prev_crn_.vert.co - crn_.vert.co, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(prev_crn_))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_
            return min_crn
        return False

    @staticmethod
    def is_clamped_by_selected_and_seams(linked_corners, shared, next_or_prev_crn, with_seam):
        # Skip if selected or with seam
        if next_or_prev_crn.edge.select:
            return True

        if with_seam:
            for crn in linked_corners:
                if crn.edge.select:
                    return True
                if (prev_crn__ := crn.link_loop_prev) != shared:
                    if prev_crn__.edge.select:
                        return True
        else:
            if next_or_prev_crn.edge.seam:
                return True
            for crn in linked_corners:
                crn_edge = crn.edge
                if crn_edge.select or crn_edge.seam:
                    return True
                if (prev_crn__ := crn.link_loop_prev) != shared:
                    prev_crn_edge = prev_crn__.edge
                    if prev_crn_edge.seam or prev_crn_edge.select:
                        return True


class UNIV_OT_SelectTexelDensity_VIEW3D(Operator):
    bl_idname = "mesh.univ_select_texel_density"
    bl_label = 'Select by TD'
    bl_description = "Select by Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    compare_type: EnumProperty(name='Compare Type', default='LESS', items=(
        ('LESS', 'Less', ''),
        ('EQUAL', 'Equal', ''),
        ('GREATER', 'Greater', ''),
    ))
    island_mode: EnumProperty(name='Mode', default='ISLAND', items=(('ISLAND', 'Island', ''), ('FACE', 'Face', '')))

    target_texel: FloatProperty(name='Texel', default=512, min=1, soft_min=32, soft_max=2048, max=10_000)
    threshold: FloatProperty(name='Threshold', default=0.01, min=0, soft_min=0.0001, soft_max=50, max=10_000)

    def draw(self, context):
        layout = self.layout
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'island_mode', expand=True)
        layout.prop(self, 'target_texel', slider=True)
        layout.prop(self, 'threshold', slider=True)
        layout.row(align=True).prop(self, 'compare_type', expand=True)

    def invoke(self, context, event):
        self.target_texel = univ_settings().texel_density

        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        self.island_mode = 'FACE' if event.alt else 'ISLAND'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        umeshes = UMeshes()

        if not self.bl_idname.startswith('UV'):
            umeshes.set_sync()

        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        umeshes.filter_by_visible_uv_faces()
        has_elem = False
        counter = 0
        counter_skipped = 0
        for umesh in umeshes:
            umesh.update_tag = False
            has_selected = umesh.has_selected_uv_verts()
            if self.island_mode == 'ISLAND':
                islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            else:
                islands = [AdvIsland([f], umesh) for f in utils.calc_visible_uv_faces(umesh)]
            if islands:
                has_elem = True
                scale = umesh.check_uniform_scale(self.report)
                for isl in islands:
                    isl.calc_area_3d(scale)
                    isl.calc_area_uv()

                    area_3d = sqrt(isl.area_3d)
                    area_uv = sqrt(isl.area_uv) * texture_size

                    texel = area_uv / area_3d if area_3d else 0

                    if not (compared_result := isclose(texel, self.target_texel, abs_tol=self.threshold)):
                        if self.compare_type == 'LESS':
                            compared_result = texel < self.target_texel
                        elif self.compare_type == 'GREATER':
                            compared_result = texel > self.target_texel

                    if self.mode == 'SELECT':
                        if compared_result:
                            if has_selected and isl.is_full_face_selected():
                                counter_skipped += 1
                                continue
                            counter += 1
                            isl.select = True
                            umesh.update_tag = True
                        elif has_selected:
                            if isl.is_full_face_deselected:
                                continue
                            isl.select = False
                            umesh.update_tag = True
                    elif self.mode == 'ADDITION':
                        if compared_result:
                            if has_selected and isl.is_full_face_selected():
                                counter_skipped += 1
                                continue
                            counter += 1
                            isl.select = True
                            umesh.update_tag = True
                    else:  # self.mode == 'DESELECT':
                        if compared_result:
                            if isl.is_full_face_deselected:
                                counter_skipped += 1
                                continue
                            counter += 1
                            isl.select = False
                            umesh.update_tag = True

        if not has_elem:
            self.report({'WARNING'}, f'{self.island_mode.capitalize() + "s"} not found')
        else:
            if not counter and not counter_skipped:
                self.report({'WARNING'}, f'No found {self.island_mode.capitalize() + "s"} in the specified texel')
        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_SelectTexelDensity(UNIV_OT_SelectTexelDensity_VIEW3D):
    bl_idname = "uv.univ_select_texel_density"


class UNIV_OT_Tests(utils.UNIV_OT_Draw_Test):
    def test_invoke(self, _event):

        # from .. import univ_pro
        umesh = self.umeshes[0]
        uv = umesh.uv

        islands = AdvIslands.calc_visible(umesh)
        islands.indexing()

        if lgs := utypes.LoopGroup.calc_dirt_loop_groups(umesh):
            # umesh.tag_visible_corners()
            # for lg in lgs:
            #     lg.extend_from_linked()

            self.calc_from_corners(lgs, uv)

        umesh.update()


class UNIV_OT_SelectByArea(Operator):
    bl_idname = "uv.univ_select_by_area"
    bl_label = 'Select by Area'
    bl_description = "Select by Area"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    size_mode: EnumProperty(name='Size Mode', default='SMALL', items=(
        ('SMALL', 'Small', ''),
        ('MEDIUM', 'Medium', ''),
        ('LARGE', 'Large', ''),
    ))
    size_type: EnumProperty(name='Size Mode', default='AREA', items=(
        ('AREA', 'Area', ''),
        ('X', 'Size X', ''),
        ('Y', 'Size Y', ''),
    ))

    threshold: FloatProperty(name='Threshold', default=0.005, min=0, soft_min=0.005, max=0.5, subtype='FACTOR')
    lower_slider: FloatProperty(name='Low', default=0.1, min=0, max=0.9, subtype='PERCENTAGE',
                                update=lambda self, _: setattr(self, 'higher_slider', self.lower_slider+0.05) if self.higher_slider-0.05 < self.lower_slider else None)
    higher_slider: FloatProperty(name='High', default=0.8, min=0.1, max=1, subtype='PERCENTAGE',
                                 update=lambda self, _: setattr(self, 'lower_slider', self.higher_slider-0.05) if self.higher_slider-0.05 < self.lower_slider else None)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'mode', expand=True)
        layout.row(align=True).prop(self, 'size_type', expand=True)
        layout.row(align=True).prop(self, 'size_mode', expand=True)

        row = layout.row(align=True)
        row.label(text=f'   Small: {self.lower_slider*100:.2f}%')
        row.label(text=f'  Medium{((self.higher_slider-self.lower_slider)*100):.2f}%')
        row.label(text=f'   Large {(1-self.higher_slider)*100:.2f}%')

        row = layout.row(align=True)
        row.prop(self, 'lower_slider', slider=True)
        row.prop(self, 'higher_slider', slider=True)
        layout.prop(self, 'threshold', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        if umeshes.sync:
            umeshes.elem_mode = 'FACE'
        umeshes.filter_by_visible_uv_faces()

        min_value = float('inf')
        max_value = float('-inf')
        islands_of_mesh = []
        counter = 0
        counter_skipped = 0
        for umesh in reversed(umeshes):
            umesh.update_tag = False
            if islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                islands_of_mesh.append(islands)
                islands.value = umesh.has_selected_uv_verts()

                if self.size_type == 'AREA':
                    for isl in islands:
                        area = isl.calc_area_uv()
                        isl.value = area
                        min_value = min(area, min_value)
                        max_value = max(area, max_value)
                else:
                    for isl in islands:
                        bbox = isl.calc_bbox()
                        if self.size_type == 'X':
                            size = bbox.width
                        else:
                            size = bbox.height
                        isl.value = size
                        min_value = min(size, min_value)
                        max_value = max(size, max_value)
            else:
                umeshes.umeshes.remove(umesh)

        if self.size_mode == 'SMALL':
            lower = min_value
            higher = lerp(min_value, max_value, self.lower_slider)
        elif self.size_mode == 'MEDIUM':
            lower = lerp(min_value, max_value, self.lower_slider)
            higher = lerp(min_value, max_value, self.higher_slider)
        else:  # self.size_mode == 'LARGE':
            lower = lerp(min_value, max_value, self.higher_slider)
            higher = max_value
        lower -= lower * self.threshold
        higher += higher * self.threshold

        for islands in islands_of_mesh:
            umesh = islands.umesh
            has_selected = islands.value
            for isl in islands:
                if self.mode == 'SELECT':
                    if lower <= isl.value <= higher:
                        if has_selected and isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        isl.select = True
                        umesh.update_tag = True
                    elif has_selected:
                        if isl.is_full_face_deselected:
                            continue
                        isl.select = False
                        umesh.update_tag = True
                elif self.mode == 'ADDITION':
                    if lower <= isl.value <= higher:
                        if has_selected and isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        isl.select = True
                        umesh.update_tag = True
                else:  # self.mode == 'DESELECT':
                    if lower <= isl.value <= higher:
                        if isl.is_full_face_deselected:
                            counter_skipped += 1
                            continue
                        counter += 1
                        isl.select = False
                        umesh.update_tag = True

        if not islands_of_mesh:
            self.report({'WARNING'}, f'Islands not found')
        else:
            if not counter and not counter_skipped:
                self.report({'WARNING'}, f'No found in the specified size')
        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_Stacked(Operator):
    bl_idname = "uv.univ_select_stacked"
    bl_label = 'Stacked'
    bl_description = "Select exact overlapped islands"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    threshold: bpy.props.FloatProperty(name='Distance', default=0.001, min=0.0, soft_min=0.00005, soft_max=0.00999)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'mode', expand=True)
        layout.prop(self, 'threshold', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        all_islands = []
        for umesh in reversed(umeshes):
            umesh.update_tag = False
            if islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                all_islands.extend(islands)

        union_islands = UnionIslands.calc_overlapped_island_groups(all_islands, self.threshold)

        counter = 0
        counter_skipped = 0
        for union_isl in union_islands:
            if self.mode == 'SELECT':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        isl.select = True
                        isl.umesh.update_tag = True
                else:
                    if union_isl.is_full_face_deselected:
                        continue
                    union_isl.select = False
                    union_isl.umesh.update_tag = True
            elif self.mode == 'ADDITION':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue

                        counter += 1
                        isl.select = True
                        isl.umesh.update_tag = True
            else:  # self.mode == 'DESELECT':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_deselected:
                            counter_skipped += 1
                            continue
                        counter += 1
                        isl.select = False
                        isl.umesh.update_tag = True

        if not union_islands:
            self.report({'WARNING'}, f'Islands not found')
        else:
            if not counter and not counter_skipped:
                self.report({'WARNING'}, f'No found stacked islands')
        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_SelectByVertexCount_VIEW3D(Operator):
    bl_idname = "mesh.univ_select_by_vertex_count"
    bl_label = 'Select by Vertex Count'
    bl_description = "Select by Vertex Count"
    bl_options = {'REGISTER', 'UNDO'}

    elem_mode: EnumProperty(name='Elem Mode', default='FACE', items=(
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    polygone_type: EnumProperty(name='Polygone Type', default='TRIS', items=(
        ('TRIS', 'Tris', ''),
        ('QUAD', 'Quad', ''),
        ('NGONE', 'N-Gone', ''),
    ))
    use_face_target_size: BoolProperty(name='Use target face size', default=False)
    face_target_size: IntProperty(name='Face Size', min=3, soft_max=32, default=4)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'elem_mode', expand=True)
        layout.row(align=True).prop(self, 'mode', expand=True)
        row = layout.row(align=True)
        row.active = not self.use_face_target_size
        row.prop(self, 'polygone_type', expand=True)

        row = layout.row(align=True)
        row.prop(self, "use_face_target_size", text="")
        row.active = self.use_face_target_size
        row.prop(self, 'face_target_size')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        if self.bl_idname.startswith('UV'):
            umeshes = UMeshes()
            island_type = AdvIslands.calc_visible_with_mark_seam
        else:
            umeshes = UMeshes.calc_any_unique(verify_uv=False)
            island_type = MeshIslands.calc_visible_with_mark_seam
            umeshes.set_sync()

        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        counter = 0
        counter_without_effect = 0
        is_target_face = self.get_is_target_face_func()
        face_select_get = self.get_face_select_func(umeshes)
        face_select_set = self.set_face_select_func(umeshes)
        has_any_select = self.has_any_elem_select_func(umeshes)

        for umesh in umeshes:
            uv = umesh.uv
            local_counter = 0
            if self.mode == 'SELECT':
                has_update = False
                if self.elem_mode == 'FACE':
                    for f in utils.calc_visible_uv_faces_iter(umesh):
                        if is_target_face(f):
                            if face_select_get(f, uv):
                                counter_without_effect += 1
                            else:
                                local_counter += 1
                                face_select_set(f, True, uv)
                        elif has_any_select(f, uv):
                            has_update = True
                            face_select_set(f, False, uv)
                else:
                    for isl in island_type(umesh):
                        if all(is_target_face(f) for f in isl):
                            if isl.has_all_face_select:
                                counter_without_effect += 1
                            else:
                                local_counter += 1
                                isl.select = True
                        elif isl.has_any_elem_select:
                            has_update = True
                            isl.select = False

                if has_update or local_counter:
                    umesh.update()
                counter += local_counter
                continue

            elif self.mode == 'DESELECT':
                if self.elem_mode == 'FACE':
                    for f in utils.calc_selected_uv_faces_iter(umesh):
                        if is_target_face(f):
                            local_counter += 1
                            face_select_set(f, False, uv)
                else:
                    for isl in island_type(umesh):
                        if isl.has_any_elem_select:
                            if all(is_target_face(f) for f in isl):
                                local_counter += 1
                                isl.select = False
            else:  # self.mode == 'ADDITION':
                if self.elem_mode == 'FACE':
                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if is_target_face(f):
                            local_counter += 1
                            face_select_set(f, True, uv)
                else:
                    for isl in island_type(umesh):
                        if not isl.has_all_face_select:
                            if all(is_target_face(f) for f in isl):
                                local_counter += 1
                                isl.select = True

            if local_counter:
                umesh.update()
                counter += local_counter

        elem_name = self.elem_mode.capitalize() + 's'
        if self.mode == 'SELECT':
            if counter and counter_without_effect:
                print(f"UniV: Select by Vertex Count: "
                      f"Found {counter+counter_without_effect} {elem_name} for select, {counter_without_effect} of them were already selected")
            elif counter and not counter_without_effect:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for select")
            elif not counter and counter_without_effect:
                self.report({'INFO'}, f"Found {counter_without_effect} {elem_name} for select, that were all initially selected")
            else:
                self.report({'INFO'}, f"Not found {elem_name} for select")

        elif self.mode == 'DESELECT':
            if counter:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for deselect")
            else:
                self.report({'INFO'}, f"No {elem_name} found to deselect — they may have been unselected initially")
        else:  # self.mode == 'ADDITION':
            if counter:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for additional select")
            else:
                self.report({'INFO'}, f"No {elem_name} found to additional select — they may have been selected initially")

        for umesh in umeshes:
            umesh.check_faces_exist(self.report)
        return {'FINISHED'}

    def get_is_target_face_func(self):
        if self.polygone_type == 'TRIS':
            is_target_polygon = lambda f_: len(f_.loops) == 3
        elif self.polygone_type == 'QUAD':
            is_target_polygon = lambda f_: len(f_.loops) == 4
        else:
            is_target_polygon = lambda f_: len(f_.loops) >= 5
        if self.use_face_target_size:
            is_target_polygon = lambda f_: len(f_.loops) == self.face_target_size
        return is_target_polygon

    @staticmethod
    def get_face_select_func(umeshes):
        if umeshes.sync:
            return lambda f, uv: BMFace.select.__get__(f)
        else:
            if umeshes.elem_mode == 'EDGE':
                return lambda f, uv: all(crn[uv].select_edge for crn in f.loops)
            else:
                return lambda f, uv: all(crn[uv].select for crn in f.loops)

    @staticmethod
    def set_face_select_func(umeshes):
        if umeshes.sync:
            return lambda f, state, uv: BMFace.select.__set__(f, state)
        else:
            def func(f, state, uv):
                for crn in f.loops:
                    crn_uv = crn[uv]
                    crn_uv.select = state
                    crn_uv.select_edge = state
            return func

    @staticmethod
    def has_any_elem_select_func(umeshes):
        if umeshes.sync:
            if umeshes.elem_mode == 'FACE':
                return lambda f, uv: f.select
            else:
                return lambda f, uv: any(v.select for v in f.verts)
        else:
            def func(f, uv):
                for crn in f.loops:
                    if crn[uv].select:
                        return True
                return False
            return func


class UNIV_OT_SelectByVertexCount_VIEW2D(UNIV_OT_SelectByVertexCount_VIEW3D):
    bl_idname = "uv.univ_select_by_vertex_count"

    elem_mode: EnumProperty(name='Elem Mode', default='ISLAND', items=(
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))


class UNIV_OT_SelectMode(Operator):
    bl_idname = "uv.univ_select_mode"
    bl_label = 'Select Mode'
    bl_description = "Set Select Mode with sticky disabled in Face Mode"
    bl_options = {'REGISTER', 'UNDO'}

    type: EnumProperty(name='Type', default='VERTEX', items=(
        ('VERTEX', 'Vertex', ''),
        ('EDGE', 'Edge', ''),
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))

    @classmethod
    def poll(cls, context):
        if context.scene.tool_settings.use_uv_select_sync:
            return bpy.ops.mesh.select_mode.poll()  # noqa
        else:
            return bpy.ops.uv.select_mode.poll()  # noqa

    def execute(self, context):
        if utils.sync():
            if self.type == 'ISLAND':
                # TODO: Implement toggle island checkbox for new sync select system
                return {'CANCELLED'}

            update = False
            if bpy.app.version >= (4, 5, 0):
                tool_settings = context.scene.tool_settings
                sticky_mode = tool_settings.uv_sticky_select_mode
                if self.type == 'FACE':
                    if sticky_mode != 'DISABLED':
                        update = True
                        tool_settings.uv_sticky_select_mode = 'DISABLED'
                else:
                    if sticky_mode != 'SHARED_LOCATION':
                        update = True
                        tool_settings.uv_sticky_select_mode = 'SHARED_LOCATION'
            elem_type = self.type
            if elem_type == 'VERTEX':
                elem_type = 'VERT'

            res = bpy.ops.mesh.select_mode(type=elem_type, use_extend=False, use_expand=False)
            if update or res == {'FINISHED'}:
                return {'FINISHED'}
            else:
                return res
        else:
            update = False
            tool_settings = context.scene.tool_settings
            sticky_mode = tool_settings.uv_sticky_select_mode
            if self.type == 'FACE':
                if sticky_mode != 'DISABLED':
                    update = True
                    tool_settings.uv_sticky_select_mode = 'DISABLED'
            else:
                if sticky_mode != 'SHARED_LOCATION':
                    update = True
                    tool_settings.uv_sticky_select_mode = 'SHARED_LOCATION'

            current_mode = utils.get_select_mode_uv()
            if self.type == 'ISLAND' and current_mode == 'FACE':
                # TODO: This behavior needs to be integrated into the Blender source.
                umeshes = UMeshes()
                umeshes.filter_by_selected_uv_faces()
                umeshes.update_tag = False

                for umesh in umeshes:
                    Islands.tag_filter_visible(umesh)
                    select_get = utils.face_select_get_func(umesh)
                    for faces in Islands.calc_with_markseam_iter_ex(umesh):
                        if not utils.all_equal(faces, select_get):
                            utypes.FaceIsland(faces, umesh).select = False
                            umesh.update_tag = True
                update |= umeshes.update_tag
                umeshes.silent_update()

            res = bpy.ops.uv.select_mode(type=self.type)
            if update or res == {'FINISHED'}:
                return {'FINISHED'}
            else:
                return res

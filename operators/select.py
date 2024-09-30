# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu
import math
import numpy as np

from mathutils import Vector
from bpy.props import *
from bpy.types import Operator
from bmesh.types import BMLoop, BMLayerItem
from gpu_extras.batch import batch_for_shader
from time import perf_counter as time

from .. import utils
from .. import types
from ..preferences import prefs
from ..types import Islands, AdvIslands, AdvIsland,  BBox, UMeshes

from ..utils import (
    face_centroid_uv,
    select_linked_crn_uv_vert,
    deselect_linked_crn_uv_vert,
    is_boundary,
)


uv_handle = None
start = time()
shader: gpu.types.GPUShader | None = None
batch: gpu.types.GPUBatch | None = None

def draw_callback_px():
    global shader
    global batch
    shader.bind()
    shader.uniform_float("color", (1, 1, 0, 1))
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

def add_draw_rect(data):
    global start
    global shader
    global batch
    global uv_handle

    start = time()

    shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR' if bpy.app.version < (3, 5, 0) else 'UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": data})

    if not (uv_handle is None):
        bpy.types.SpaceImageEditor.draw_handler_remove(uv_handle, 'WINDOW')

    uv_handle = bpy.types.SpaceImageEditor.draw_handler_add(draw_callback_px, (), 'WINDOW', 'POST_VIEW')
    bpy.app.timers.register(uv_area_draw_timer)

class UNIV_OT_SelectLinked(Operator):
    bl_idname = 'uv.univ_select_linked'
    bl_label = 'Select Linked'
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
            if context.area.ui_type == 'UV':
                return bpy.ops.uv.select_linked()
            if uv_areas := [area for area in context.screen.areas if area.uv_type == 'UV']:
                with context.temp_override(area=uv_areas[0]):  # noqa
                    return bpy.ops.uv.select_linked()
            return {'CANCELLED'}
        else:
            sync = bpy.context.scene.tool_settings.use_uv_select_sync
            return self.deselect_linked(sync=sync)

    def deselect_linked(self, sync):
        umeshes = UMeshes(report=self.report)
        mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()

        if sync and mode == 'VERTEX':
            for umesh in umeshes:
                if types.PyBMesh.is_full_vert_selected(umesh.bm) or types.PyBMesh.is_full_vert_deselected(umesh.bm):
                    umesh.update_tag = False
                    continue
                has_full_selected = False
                half_selected = []
                if islands := Islands.calc_visible(umesh):
                    for island in islands:
                        select_info = island.info_select(sync)
                        if select_info == types.eInfoSelectFaceIsland.HALF_SELECTED:
                            half_selected.append(island)
                        elif select_info == types.eInfoSelectFaceIsland.FULL_SELECTED:
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

        if sync and mode == 'EDGE':
            for umesh in umeshes:
                if types.PyBMesh.is_full_edge_selected(umesh.bm) or types.PyBMesh.is_full_edge_deselected(umesh.bm):
                    umesh.update_tag = False
                    continue
                has_full_selected = False
                half_selected = []
                if islands := Islands.calc_visible(umesh):
                    for island in islands:
                        select_info = island.info_select(sync)
                        if select_info == types.eInfoSelectFaceIsland.HALF_SELECTED:
                            half_selected.append(island)
                        elif select_info == types.eInfoSelectFaceIsland.FULL_SELECTED:
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
            if sync and mode == 'FACE':
                if types.PyBMesh.is_full_edge_selected(umesh.bm) or types.PyBMesh.is_full_edge_deselected(umesh.bm):
                    umesh.update_tag = False
                    continue
            is_update = False
            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    if update_state := (island.info_select(sync) == types.eInfoSelectFaceIsland.HALF_SELECTED):
                        island.deselect_set(mode=mode, sync=sync)
                    is_update |= update_state
            umesh.update_tag = is_update

            if is_update and sync:
                umesh.bm.select_flush_mode()

        return umeshes.update(info='No islands for deselect')


class UNIV_OT_Select_By_Cursor(Operator):
    bl_idname = "uv.univ_select_by_cursor"
    bl_label = "Select by Cursor"
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

        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        umeshes = UMeshes(report=self.report)
        elem_mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()

        tile_co = utils.get_tile_from_cursor()
        view_rect = BBox.init_from_minmax(tile_co, tile_co+Vector((1, 1)))
        view_rect.pad(Vector((-2e-08, -2e-08)))

        view_island = AdvIsland()
        view_island._bbox = view_rect
        view_island.flat_coords = view_rect.draw_data_tris()

        if sync and self.face_mode:
            utils.set_select_mode_mesh('FACE')

        args = (umeshes, elem_mode, self.face_mode, view_island, sync)

        if self.mode == 'ADDITIONAL':
            self._additional(*args)
        elif self.mode == 'DESELECT':
            self._deselect(*args)
        else:
            self._select(*args)

        add_draw_rect(view_rect.draw_data_lines())

        return umeshes.update()

    @staticmethod
    def _additional(umeshes: UMeshes, elem_mode, face_mode, view_island, sync):
        for umesh in umeshes:
            if sync:
                if elem_mode == 'VERTEX' and types.PyBMesh.is_full_vert_selected(umesh.bm) \
                        or elem_mode == 'EDGE' and types.PyBMesh.is_full_edge_selected(umesh.bm) \
                        or elem_mode == 'FACE' and types.PyBMesh.is_full_face_selected(umesh.bm):
                    umesh.update_tag = False
                    continue

            if face_mode:
                bb = view_island.bbox
                uv = umesh.uv
                if sync:
                    for f in umesh.bm.faces:
                        if not f.select:
                            if face_centroid_uv(f, uv) in bb:
                                f.select = True
                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv) in bb:
                                for _l in f.loops:
                                    luv = _l[uv]
                                    luv.select = True
                                    luv.select_edge = True
                umesh.bm.select_flush_mode()

            else:
                has_update = False
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=False):
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()
                    for island in adv_islands:
                        select_info = island.info_select(sync, elem_mode)
                        if select_info == types.eInfoSelectFaceIsland.FULL_SELECTED:
                            continue
                        if island.is_overlap(view_island):
                            island.select = True
                            has_update = True

                if sync and has_update and elem_mode in ('VERTEX', 'EDGE'):
                    umesh.bm.select_flush_mode()

                umesh.update_tag = has_update

    @staticmethod
    def _select(umeshes, elem_mode, face_mode, view_island, sync):
        for umesh in umeshes:

            if face_mode:
                bb = view_island.bbox
                uv = umesh.uv
                if sync:
                    for f in umesh.bm.faces:
                        if face_centroid_uv(f, uv) in bb:
                            f.select = True
                        else:
                            f.select = False
                            for v in f.verts:
                                v.select = False

                            for e in f.edges:
                                e.select = False

                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv) in bb:
                                for crn in f.loops:
                                    crn_uv = crn[uv]
                                    crn_uv.select = True
                                    crn_uv.select_edge = True
                            else:
                                for crn in f.loops:
                                    crn_uv = crn[uv]
                                    crn_uv.select = False
                                    crn_uv.select_edge = False
                umesh.bm.select_flush_mode()

            else:
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=False):
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()

                    if sync and elem_mode in ('VERTEX', 'EDGE'):
                        for island in adv_islands:
                            island.tag = island.is_overlap(view_island)

                        for island in adv_islands:
                            if not island.tag:
                                island.select = False
                        for island in adv_islands:
                            if island.tag:
                                island.select = True

                        umesh.bm.select_flush_mode()

                    else:
                        for island in adv_islands:
                            island._select_ex(island.is_overlap(view_island), sync, elem_mode)  # noqa # pylint: disable=W0212

                umesh.update_tag = bool(adv_islands)

    @staticmethod
    def _deselect(umeshes, elem_mode, face_mode, view_island, sync):
        for umesh in umeshes:
            sync_elem_mode = utils.get_select_mode_mesh()
            if sync_elem_mode == 'VERTEX' and types.PyBMesh.is_full_vert_deselected(umesh.bm) \
                    or sync_elem_mode == 'EDGE' and types.PyBMesh.is_full_edge_deselected(umesh.bm) \
                    or sync_elem_mode == 'FACE' and types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            if face_mode:
                bb = view_island.bbox
                uv = umesh.uv
                if sync:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv) in bb:
                                f.select = False

                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv) in bb:
                                for _l in f.loops:
                                    luv = _l[uv]
                                    luv.select = False
                                    luv.select_edge = False

                umesh.bm.select_flush_mode()

            else:
                has_update = False
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=False):
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()
                    for island in adv_islands:
                        select_info = island.info_select(sync, elem_mode)
                        if select_info == types.eInfoSelectFaceIsland.UNSELECTED:
                            continue
                        if island.is_overlap(view_island):
                            island.select = False
                            has_update = True

                if sync and has_update and elem_mode in ('VERTEX', 'EDGE'):
                    umesh.bm.select_flush_mode()
                umesh.update_tag = has_update


class UNIV_OT_Select_Square_Island(Operator):
    bl_idname = 'uv.univ_select_square_island'
    bl_label = 'Select Square Island'
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

    def __init__(self):
        self.sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync
        self.elem_mode = utils.get_select_mode_mesh() if self.sync else utils.get_select_mode_uv()
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
        if self.sync and utils.get_select_mode_mesh() != 'FACE':
            if self.mode != 'ADDITIONAL':
                utils.set_select_mode_mesh('FACE')
                self.elem_mode = 'FACE'

        self.umeshes = UMeshes(report=self.report)

        if self.mode == 'DESELECT':
            self.deselect()
        elif self.mode == 'ADDITIONAL':
            self.addition()
        else:
            self.select()

        return self.umeshes.update()

    def select(self):
        for umesh in self.umeshes:
            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    percent, close = self.percent_and_is_square_close(island)

                    if self.shape == 'SQUARE':
                        island.select = close
                    elif self.shape == 'HORIZONTAL':
                        island.select = percent > 0 and not close
                    else:
                        island.select = percent < 0 and not close
            umesh.update_tag = bool(islands)

    def deselect(self):
        for umesh in self.umeshes:
            update_tag = False
            if self.sync and types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    percent, close = self.percent_and_is_square_close(island)

                    if self.shape == 'SQUARE':
                        deselect = close
                    elif self.shape == 'HORIZONTAL':
                        deselect = percent > 0 and not close
                    else:
                        deselect = percent < 0 and not close

                    if deselect:
                        island.select = False
                        update_tag = True
            umesh.update_tag = update_tag

    def addition(self):
        for umesh in self.umeshes:
            update_tag = False
            if self.sync and types.PyBMesh.is_full_face_selected(umesh.bm):
                umesh.update_tag = False
                continue

            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    percent, close = self.percent_and_is_square_close(island)

                    if self.shape == 'SQUARE':
                        select = close
                    elif self.shape == 'HORIZONTAL':
                        select = percent > 0 and not close
                    else:
                        select = percent < 0 and not close

                    if select:
                        island.select = True
                        update_tag = True

            umesh.update_tag = update_tag

    def percent_and_is_square_close(self, island):
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


class UNIV_OT_Select_Border_Edge_by_Angle(Operator):
    bl_idname = 'uv.univ_select_border_edge_by_angle'
    bl_label = 'Select Border Edge by Angle'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    edge_dir: EnumProperty(name='Direction', default='HORIZONTAL', items=(
        ('BOTH', 'Both', ''),
        ('HORIZONTAL', 'Horizontal', ''),
        ('VERTICAL', 'Vertical', ''),
    ))
    border: EnumProperty(name='Border', default='BORDER', items=(
        ('BORDER', 'Border', ''),
        ('ALL', 'All', ''),
    ))

    angle: FloatProperty(name='Angle', default=math.radians(5), min=0, max=math.radians(45.001), subtype='ANGLE')

    def draw(self, context):
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'edge_dir', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'border', expand=True)
        layout = self.layout
        layout.prop(self, 'angle', slider=True)

    def __init__(self):
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

        self.border = 'ALL' if event.alt else 'BORDER'

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        if context.scene.tool_settings.use_uv_select_sync:
            bpy.ops.uv.univ_sync_uv_toggle()  # noqa
        if utils.get_select_mode_uv() != 'EDGE':
            utils.set_select_mode_uv('EDGE')

        self.edge_orient = self.x_vec if self.edge_dir == 'HORIZONTAL' else self.y_vec
        self.umeshes = UMeshes(report=self.report)

        self.negative_ange = math.pi - self.angle

        if self.border == 'BORDER':
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
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)

            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv]
                    crn_uv_b = crn.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        to_select_corns.append(crn)
                    else:
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn_uv_b.select = False
                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            elif self.mode == 'DESELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv]
                    crn_uv_b = crn.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        self.deselect_crn_uv_edge(crn, uv)
                    else:
                        if crn_uv_a.select_edge:
                            to_select_corns.append(crn)
                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            else:  # 'ADDITIONAL'
                for loop in corners:
                    crn_uv_a = loop[uv]
                    if crn_uv_a.select_edge:
                        continue
                    crn_uv_b = loop.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        select_linked_crn_uv_vert(loop, uv)
                        select_linked_crn_uv_vert(loop.link_loop_next, uv)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn_uv_b.select = True

    def select_both(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv]
                    crn_uv_b = crn.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    x_angle = vec.angle(self.x_vec)
                    y_angle = vec.angle(self.y_vec)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        to_select_corns.append(crn)
                    else:
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn_uv_b.select = False
                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            elif self.mode == 'DESELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv]
                    crn_uv_b = crn.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    x_angle = vec.angle(self.x_vec)
                    y_angle = vec.angle(self.y_vec)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        self.deselect_crn_uv_edge(crn, uv)
                    else:
                        if crn_uv_a.select_edge:
                            to_select_corns.append(crn)
                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv]
                    if crn_uv_a.select_edge:
                        continue
                    crn_uv_b = crn.link_loop_next[uv]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    x_angle = vec.angle(self.x_vec)
                    y_angle = vec.angle(self.y_vec)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        select_linked_crn_uv_vert(crn, uv)
                        select_linked_crn_uv_vert(crn.link_loop_next, uv)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn_uv_b.select = True

    def select_both_border(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        x_angle = vec.angle(self.x_vec)
                        y_angle = vec.angle(self.y_vec)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            to_select_corns.append(crn)
                        else:
                            crn_uv_a.select = False
                            crn_uv_a.select_edge = False
                            crn_uv_b.select = False
                    else:
                        crn_uv_a = crn[uv]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv].select = False

                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        x_angle = vec.angle(self.x_vec)
                        y_angle = vec.angle(self.y_vec)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            self.deselect_crn_uv_edge_for_border(crn, uv)
                # TODO: Refactor
                for crn in _corners:
                    crn_uv = crn[uv]
                    if crn_uv.select_edge:
                        self.select_crn_uv_edge_sequence([crn], uv)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv):
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        x_angle = vec.angle(self.x_vec)
                        y_angle = vec.angle(self.y_vec)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            select_linked_crn_uv_vert(crn, uv)
                            select_linked_crn_uv_vert(crn.link_loop_next, uv)
                            crn_uv_a.select = True
                            crn_uv_a.select_edge = True
                            crn_uv_b.select = True

    def select_hv_border(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            to_select_corns.append(crn)
                        else:
                            crn_uv_a.select = False
                            crn_uv_a.select_edge = False
                            crn_uv_b.select = False
                    else:
                        crn_uv_a = crn[uv]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv].select = False

                self.select_crn_uv_edge_sequence(to_select_corns, uv)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            self.deselect_crn_uv_edge_for_border(crn, uv)
                            # Removing the notches
                            if is_boundary(crn.link_loop_next, uv):
                                crn_uv_c = crn.link_loop_next[uv]
                                crn_uv_c.select = False
                                crn_uv_c.select_edge = False
                            if is_boundary(crn.link_loop_prev, uv):
                                crn_uv_d = crn.link_loop_prev[uv]
                                crn_uv_d.select = False
                                crn_uv_d.select_edge = False

                # TODO: Refactor
                for crn in _corners:
                    crn_uv = crn[uv]
                    if crn_uv.select_edge:
                        self.select_crn_uv_edge_sequence([crn], uv)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv):
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            select_linked_crn_uv_vert(crn, uv)
                            select_linked_crn_uv_vert(crn.link_loop_next, uv)
                            crn_uv_a.select = True
                            crn_uv_a.select_edge = True
                            crn_uv_b.select = True

    @staticmethod
    def select_crn_uv_edge_sequence(to_select_corns: list[BMLoop], uv):
        for crn in to_select_corns:
            link_crn_next = crn.link_loop_next
            select_linked_crn_uv_vert(crn, uv)
            select_linked_crn_uv_vert(link_crn_next, uv)

            crn_uv_a = crn[uv]
            crn_uv_b = link_crn_next[uv]
            crn_uv_a.select = True
            crn_uv_a.select_edge = True
            crn_uv_b.select = True

    @staticmethod
    def select_crn_uv_edge(crn: BMLoop, uv):
        link_crn_next = crn.link_loop_next
        select_linked_crn_uv_vert(crn, uv)
        select_linked_crn_uv_vert(link_crn_next, uv)

        crn_uv_a = crn[uv]
        crn_uv_b = link_crn_next[uv]
        crn_uv_a.select = True
        crn_uv_a.select_edge = True
        crn_uv_b.select = True

    @staticmethod
    def deselect_crn_uv_edge(crn: BMLoop, uv: BMLayerItem):
        link_crn_next = crn.link_loop_next
        deselect_linked_crn_uv_vert(crn, uv)
        deselect_linked_crn_uv_vert(link_crn_next, uv)

        crn_uv_a = crn[uv]
        crn_uv_b = link_crn_next[uv]
        crn_uv_a.select = False
        crn_uv_a.select_edge = False
        crn_uv_b.select = False

    @staticmethod
    def deselect_crn_uv_edge_for_border(crn: BMLoop, uv):
        def _deselect_linked_crn_uv_vert(first: BMLoop):
            bm_iter = first
            while True:
                if (bm_iter := bm_iter.link_loop_prev.link_loop_radial_prev) == first:
                    break
                crn_uv_bm_iter = bm_iter[uv]
                if first[uv].uv == crn_uv_bm_iter.uv:
                    crn_uv_bm_iter.select = False
                    crn_uv_bm_iter.select_edge = False
                    bm_iter.link_loop_prev[uv].select = False
                    bm_iter.link_loop_prev[uv].select_edge = False

        _deselect_linked_crn_uv_vert(crn)
        link_crn_next = crn.link_loop_next
        deselect_linked_crn_uv_vert(crn, uv)
        deselect_linked_crn_uv_vert(link_crn_next, uv)

        crn_uv_a = crn[uv]
        crn.link_loop_prev[uv].select_edge = False
        crn_uv_b = link_crn_next[uv]
        crn_uv_a.select = False
        crn_uv_a.select_edge = False
        crn_uv_b.select = False

class UNIV_OT_Select_Border(Operator):
    bl_idname = 'uv.univ_select_border'
    bl_label = 'Select Border'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    border_mode: EnumProperty(name='Border Mode', default='ALL', items=(
        ('ALL', 'All', ''),
        ('BETWEEN', 'Between', ''),
    ))

    def draw(self, context):
        row = self.layout.row(align=True)
        row.prop(self, 'border_mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)

    def __init__(self):
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.border_mode = 'BETWEEN' if event.alt else 'ALL'

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        if context.scene.tool_settings.use_uv_select_sync:
            bpy.ops.uv.univ_sync_uv_toggle()  # noqa
            self.sync = False
        if utils.get_select_mode_uv() not in ('EDGE', 'VERTEX'):
            utils.set_select_mode_uv('EDGE')

        self.umeshes = UMeshes(report=self.report)
        if self.border_mode == 'ALL':
            self.select_border()
        else:
            self.select_border_between()
        return self.umeshes.update()

    def select_border(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv):
                        to_select_corns.append(crn)
                    else:
                        crn_uv_a = crn[uv]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv].select = False

                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge_sequence(to_select_corns, uv)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv):
                        UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(crn, uv)
                for crn in _corners:
                    crn_uv = crn[uv]
                    if crn_uv.select_edge:
                        UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge_sequence([crn], uv)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv):
                        select_linked_crn_uv_vert(crn, uv)
                        select_linked_crn_uv_vert(crn.link_loop_next, uv)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn.link_loop_next[uv].select = True

    def select_border_between(self):
        # bpy.ops.uv.select_all(action='DESELECT')
        for umesh in self.umeshes:
            uv = umesh.uv
            if self.mode == 'SELECT':
                islands = Islands.calc_extended(umesh)

                for _f in umesh.bm.faces:
                    for _crn in _f.loops:
                        _crn_uv = _crn[uv]
                        _crn_uv.select = False
                        _crn_uv.select_edge = False

                islands.indexing()
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:
                                continue
                            if shared_crn.face.index != f.index:
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv)

            elif self.mode == 'DESELECT':
                islands = Islands.calc_extended(umesh)

                islands.indexing()
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:
                                continue
                            if shared_crn.face.index != f.index:
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(shared_crn, uv)
                                # Removing the notches
                                if is_boundary(crn.link_loop_next, uv):
                                    crn_uv_c = crn.link_loop_next[uv]
                                    crn_uv_c.select = False
                                    crn_uv_c.select_edge = False
            else:  # 'ADDITION'
                islands = Islands.calc_extended(umesh)

                islands.indexing()
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn: BMLoop
                            if crn == (shared_crn := crn.link_loop_radial_prev):
                                continue
                            if not shared_crn.face.tag:
                                continue
                            if shared_crn.face.index != f.index:
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv)
                umesh.update_tag = bool(islands)

class UNIV_OT_Select_Pick(Operator):
    bl_idname = 'uv.univ_select_pick'
    bl_label = 'Pick Select'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)

    @staticmethod
    def get_mouse_pos(event, view):
        return Vector(view.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def __init__(self):
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None
        self.umeshes: UMeshes | None = None

    def invoke(self, context, event):
        self.umeshes = UMeshes()
        self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        self.mouse_pos = self.get_mouse_pos(event, context.region.view2d)
        return self.pick_select()

    def pick_select(self):
        min_dist = float('inf')
        min_isl = None
        sync = self.umeshes.sync
        is_sync_face_mode = utils.get_select_mode_mesh() == 'FACE'

        for umesh in self.umeshes:
            if self.select:
                if sync and umesh.is_full_face_selected:
                    continue
            else:
                if sync:
                    if is_sync_face_mode:
                        if umesh.is_full_face_deselected:
                            continue
                    else:
                        if umesh.is_full_vert_deselected:
                            continue

            uv = umesh.uv
            for isl in Islands.calc_visible_with_mark_seam(umesh):
                if self.select:  # Skip full selected island
                    if sync:
                        if all(f.select for f in isl):
                            continue
                    else:
                        if all(crn[uv].select for f in isl for crn in f.loops):
                            continue
                else:  # Skip full deselected islands
                    if sync:
                        if is_sync_face_mode:
                            if all(not f.select for f in isl):
                                continue
                        else:
                            if all(not v.select for f in isl for v in f.verts):
                                continue
                    else:
                        if all(not crn[uv].select for f in isl for crn in f.loops):
                            continue

                if not (res := self.find_nearest_island(self.mouse_pos, min_dist, isl, uv)) is None:
                    min_dist = res
                    min_isl = isl

        if not min_isl or (self.max_distance < min_dist):
            return {'CANCELLED'}

        umesh = min_isl.umesh

        if sync:
            if self.select:
                for f in min_isl:
                    f.select = True
            else:
                if utils.get_select_mode_mesh() == 'FACE':
                    for f in min_isl:
                        f.select = False
                else:
                    if any(f.select for f in min_isl):
                        for f in min_isl:
                            f.select = False

                        for f in utils.calc_selected_uv_faces_iter(umesh):
                            if f.select:
                                for v in f.verts:
                                    v.select = True
                                for e in f.verts:
                                    e.select = True
                    else:
                        for f in min_isl:
                            for v in f.verts:
                                v.select = False
                            for e in f.verts:
                                e.select = False
                        umesh.bm.select_flush_mode()
        else:
            select_state = self.select
            uv = umesh.uv
            for f in min_isl:
                for crn in f.loops:
                    crn_uv = crn[uv]
                    crn_uv.select = select_state
                    crn_uv.select_edge = select_state

        umesh.update()

        return {'FINISHED'}

    @staticmethod
    def find_nearest_island(pt, min_dist, isl, uv):
        from ..utils import closest_pt_to_line
        first_dist = min_dist
        for f in isl:
            face_center = Vector((0.0, 0.0))
            for crn in f.loops:
                close_pt = closest_pt_to_line(pt, crn[uv].uv, crn.link_loop_next[uv].uv)

                face_center += close_pt
                if (dist := (close_pt-pt).length) < min_dist:
                    min_dist = dist
                # TODO: Check Point inside face, if crn.edge == other_crn.edge

            if (dist := (face_center / len(f.loops) - pt).length) < min_dist:
                min_dist = dist

        if first_dist != min_dist:
            return min_dist

class UNIV_OT_Pick(utils.UNIV_OT_Draw_Test):
    @utils.profile
    def test(self, event):
        from numpy import mean as np_mean
        from numpy import array as np_array
        from numpy import roll as np_roll
        from numpy.linalg import norm as np_distance
        from ..utils import closest_pts_to_lines

        pt = self.get_mouse_pos(event)
        pt_np = np.array([pt], dtype='float32')
        u = self.umeshes[0]
        uv = u.uv

        min_pt = ()
        min_dist = float('inf')

        for f in u.bm.faces:
            l_a = np_array([crn[uv].uv.to_tuple() for crn in f.loops], dtype='float32')
            l_b = np_roll(l_a, shift=1, axis=0)

            closest_points = closest_pts_to_lines(pt_np, l_a, l_b)
            distance = np_distance(pt_np - closest_points, axis=1)

            min_index = np.argmin(distance)
            if distance[min_index] < min_dist:
                min_dist = distance[min_index]
                min_pt = closest_points[min_index]

            face_center = np_mean(l_a, axis=0)
            distance_face_center = np_distance(pt_np - face_center, axis=1)

            if distance_face_center < min_dist:
                min_dist = distance_face_center
                min_pt = face_center

        self.points = (min_pt,)

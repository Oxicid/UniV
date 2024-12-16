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
from bmesh.types import BMFace, BMLoop, BMLayerItem
from gpu_extras.batch import batch_for_shader
from time import perf_counter as time

from .. import utils
from .. import types
from ..preferences import prefs, settings
from ..types import Islands, AdvIslands, AdvIsland,  BBox, UMeshes

from ..utils import (
    face_centroid_uv,
    select_linked_crn_uv_vert,
    deselect_linked_crn_uv_vert,
    is_boundary_non_sync,
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

        if sync and umeshes.elem_mode == 'VERTEX':
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

        if sync and umeshes.elem_mode == 'EDGE':
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
            if sync and umeshes.elem_mode == 'FACE':
                if types.PyBMesh.is_full_edge_selected(umesh.bm) or types.PyBMesh.is_full_edge_deselected(umesh.bm):
                    umesh.update_tag = False
                    continue
            is_update = False
            if islands := Islands.calc_visible(umesh):
                for island in islands:
                    if update_state := (island.info_select(sync) == types.eInfoSelectFaceIsland.HALF_SELECTED):
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
                            island.select = island.is_overlap(view_island)

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
    bl_label = 'Border by Angle'
    bl_description = 'Select border edges by angle'
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
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
                        crn_uv_a = crn[uv]
                        crn_uv_b = crn.link_loop_next[uv]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            self.deselect_crn_uv_edge_for_border(crn, uv)
                            # Removing the notches
                            if is_boundary_non_sync(crn.link_loop_next, uv):
                                crn_uv_c = crn.link_loop_next[uv]
                                crn_uv_c.select = False
                                crn_uv_c.select_edge = False
                            if is_boundary_non_sync(crn.link_loop_prev, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
    bl_label = 'Border'
    bl_description = 'Select border edges'
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
            if self.mode == 'SELECT':  # TODO: Add behavior, border by select (not All)
                to_select_corns = []
                for crn in corners:
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
                    if is_boundary_non_sync(crn, uv):
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
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv)

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
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(shared_crn, uv)
                                # Removing the notches
                                if is_boundary_non_sync(crn.link_loop_next, uv):
                                    crn_uv_c = crn.link_loop_next[uv]
                                    crn_uv_c.select = False
                                    crn_uv_c.select_edge = False
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
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv)
                umesh.update_tag = bool(islands)


class UNIV_OT_Select_Pick(Operator):
    bl_idname = 'uv.univ_select_pick'
    bl_label = 'Pick Select'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)

    def __init__(self):
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
        sync = self.umeshes.sync
        is_sync_face_mode = self.umeshes.elem_mode == 'FACE'  # TODO: Test with: and sync

        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            if self.select:
                # TODO: Use has_non_full_sel (implement)
                if (sync and umesh.is_full_face_selected) or (not sync and umesh.is_full_face_deselected):
                    continue
            else:
                if sync:
                    if (is_sync_face_mode and umesh.is_full_face_deselected) or \
                            (not is_sync_face_mode and umesh.is_full_vert_deselected):
                        continue

            for isl in Islands.calc_visible_with_mark_seam(umesh):
                if self.select:  # Skip full selected island
                    if isl.is_full_face_selected:
                        continue
                else:  # Skip full deselected islands
                    if (is_sync_face_mode and isl.is_full_face_deselected) or \
                            (not is_sync_face_mode and isl.is_full_vert_deselected):
                        continue
                hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            return {'CANCELLED'}

        umesh = hit.island.umesh

        if sync:
            if self.select:
                hit.island.select = True
            else:
                if self.umeshes.elem_mode == 'FACE':
                    hit.island.select = False
                else:
                    if any(f.select for f in hit.island):
                        hit.island.select_all_elem = False
                        for f in utils.calc_selected_uv_faces_iter(umesh):
                            for v in f.verts:
                                v.select = True
                            for e in f.edges:
                                e.select = True
                    else:
                        for f in hit.island:
                            for v in f.verts:
                                v.select = False
                            for e in f.edges:
                                e.select = False
                        umesh.bm.select_flush_mode()
        else:
            hit.island.select = self.select

        umesh.update()

        return {'FINISHED'}


class UNIV_OT_Select_Grow(Operator):
    bl_idname = 'uv.univ_select_grow'
    bl_label = 'Grow'
    bl_description = "Select more UV vertices connected to initial selection\n\n" \
                     "Default - Grow\n" \
                     "Ctrl or Alt - Shrink\n\n" \
                     "Has [Ctrl + Scroll Up/Down] keymap"
    bl_options = {'REGISTER', 'UNDO'}

    grow: BoolProperty(name='Select', default=True)

    def __init__(self):
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)

    def execute(self, context):
        self.umeshes = UMeshes()
        if self.grow:
            return self.grow_select()
        else:
            return self.shrink()

    def grow_select(self):
        sync = self.umeshes.sync
        is_sync_face_mode = utils.get_select_mode_mesh() == 'FACE' and sync

        for umesh in self.umeshes:
            if (sync and umesh.is_full_face_selected) or (not sync and umesh.is_full_face_deselected):
                continue
            if is_sync_face_mode and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            if islands := Islands.calc_visible_with_mark_seam(umesh):
                islands.indexing()
                for idx, isl in enumerate(islands):
                    if sync:
                        if self.umeshes.elem_mode == 'FACE':
                            for f in isl:
                                if not f.select:
                                    f.tag = any(l_crn.face.select for crn in f.loops for l_crn in utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx))
                        else:
                            if umesh.is_full_face_deselected:
                                for f in isl:
                                    if not f.select:
                                        f.tag = any(v.select for v in f.verts)
                            else:
                                for f in isl:
                                    if not f.select:
                                        f.tag = self.is_grow_face(f, uv, idx)
                    else:
                        if self.umeshes.elem_mode == 'EDGE':
                            for f in isl:
                                selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    f.tag = True
                        else:
                            for f in isl:
                                selected_corners = sum(crn[uv].select for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    f.tag = True
                if sync:
                    for isl in islands:
                        for f in isl:
                            if f.tag:
                                f.select = True
                else:
                    for idx, isl in enumerate(islands):
                        for f in isl:
                            if not f.tag:
                                continue
                            for crn in f.loops:
                                if crn[uv].select:
                                    continue
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
        is_sync_face_mode = utils.get_select_mode_mesh() == 'FACE' and sync

        for umesh in self.umeshes:
            if sync:
                if (is_sync_face_mode and umesh.is_full_face_deselected) or \
                        (not is_sync_face_mode and umesh.is_full_vert_deselected):
                    continue
            else:
                if umesh.is_full_face_deselected:
                    continue

            uv = umesh.uv
            if islands := Islands.calc_visible_with_mark_seam(umesh):
                islands.indexing()
                for idx, isl in enumerate(islands):
                    if sync:
                        if self.umeshes.elem_mode == 'FACE':
                            for f in isl:
                                if f.select:
                                    f.tag = any(not l_crn.face.select for crn in f.loops
                                                for l_crn in utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx))
                        else:
                            if umesh.is_full_face_deselected:
                                for f in isl:
                                    if not f.select:
                                        f.tag = any(v.select for v in f.verts)
                            else:
                                for f in isl:
                                    if not f.select:
                                        f.tag = self.is_shrink_face(f, uv, idx)
                    else:
                        if self.umeshes.elem_mode == 'EDGE':
                            for f in isl:
                                selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    f.tag = True
                        else:
                            for f in isl:
                                selected_corners = sum(crn[uv].select for crn in f.loops)
                                if selected_corners and selected_corners != len(f.loops):
                                    f.tag = True

                if sync:
                    if self.umeshes.elem_mode == 'FACE':
                        for isl in islands:
                            for f in isl:
                                if f.tag:
                                    f.select = False
                        # umesh.bm.select_flush_mode()
                    else:
                        for isl in islands:
                            for f in isl:
                                if f.tag:
                                    f.select = False
                                    for v in f.verts:
                                        v.select = False
                    umesh.bm.select_flush(False)
                else:
                    for idx, isl in enumerate(islands):
                        for f in isl:
                            if not f.tag:
                                continue
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


class UNIV_OT_Select_Edge_Grow_VIEW2D(Operator):
    bl_idname = 'uv.univ_select_edge_grow'
    bl_label = 'Edge Grow'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Edge Grow/Shrink Select\n\n" \
                     f"Default - Grow Select \n" \
                     f"Ctrl or Alt - Shrink Select\n\n" \
                     f"Has [Alt + Scroll Up/Down] keymap, but it conflicts with the Frame Offset operator"

    clamp_on_seam: BoolProperty(name='Clamp on Seam', default=True,
                                description="Edge Grow clamp on edges with seam, but if the original edge has seam, this effect is ignored")
    grow: BoolProperty(name='Select', default=True, description='Grow/Shrink toggle')
    max_angle: FloatProperty(name='Angle', default=math.radians(20), min=math.radians(1), soft_min=math.radians(5), max=math.radians(90), subtype='ANGLE',
                             description="Max select angle. If edge topology contain 4 quad faces without border edge, this effect is ignored.")

    def __init__(self):
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)

        if self.umeshes.sync:
            self.report({'INFO'}, f'Edge Grow not work in Sync mode, run grow instead')
            return bpy.ops.uv.univ_select_grow(grow=self.grow)  # noqa

        if self.umeshes.elem_mode not in ('VERTEX', 'EDGE'):
            self.report({'INFO'}, f'Edge Grow not work in "{self.umeshes.elem_mode}" mode, run grow instead')
            return bpy.ops.uv.univ_select_grow(grow=self.grow)  # noqa

        if self.grow:
            self.grow_select()
            self.umeshes.update(info='Not found edges for grow select')
            return {'FINISHED'}

        self.shrink_select()
        self.umeshes.update(info='Not found edges for shrink select')
        return {'FINISHED'}

    def grow_select(self):
        calc_islands = Islands.calc_extended_any_edge_with_markseam if self.clamp_on_seam else Islands.calc_extended_any_edge
        for umesh in reversed(self.umeshes):
            uv = umesh.uv
            update = False
            if islands := calc_islands(umesh):  # noqa
                islands.indexing()
                for isl in islands:
                    grew = []
                    # corners = (crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge)
                    corners = [crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge]
                    for crn in corners:

                        with_seam = not self.clamp_on_seam or crn.edge.seam
                        selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam):
                            if not with_seam:
                                if grow_prev_crn.edge.seam:
                                    continue
                            grew.append(grow_prev_crn)

                        if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam):
                            if not with_seam:
                                if grow_next_crn.edge.seam:
                                    continue
                            grew.append(grow_next_crn)

                    for grew_crn in grew:
                        utils.select_crn_uv_edge_with_shared_by_idx(grew_crn, uv, force=True)

                    update |= bool(grew)

            if not update:
                self.umeshes.umeshes.remove(umesh)

    def shrink_select(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            update = False
            if islands := Islands.calc_extended_any_edge_with_markseam(umesh):
                islands.indexing()
                for isl in islands:
                    shrink = []
                    # corners = (crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge)
                    corners = [crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge]
                    for crn in corners:
                        with_seam = crn.edge.seam
                        selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam):
                            if not with_seam and grow_prev_crn.edge.seam:
                                grow_prev_crn = None

                        if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam):
                            if not with_seam and grow_next_crn.edge.seam:
                                grow_next_crn = None

                        if grow_prev_crn or grow_next_crn:
                            shrink.append((crn, grow_prev_crn, grow_next_crn))

                    for cur_crn, prev_crn, next_crn in shrink:
                        cur_crn[uv].select_edge = False

                        if shared__ := utils.shared_linked_crn_by_idx(cur_crn, uv):
                            shared__[uv].select_edge = False

                        if prev_crn and next_crn:
                            for deselect_crn in utils.linked_crn_uv_by_idx_unordered_included(cur_crn, uv):
                                deselect_crn[uv].select = False
                            for deselect_crn in utils.linked_crn_uv_by_idx_unordered_included(cur_crn.link_loop_next, uv):
                                deselect_crn[uv].select = False
                        elif prev_crn:
                            for deselect_crn in utils.linked_crn_uv_by_idx_unordered_included(cur_crn, uv):
                                deselect_crn[uv].select = False

                            # Deselect if not have selected edges
                            linked_next = utils.linked_crn_uv_by_idx_unordered_included(cur_crn.link_loop_next, uv)
                            if not any(crn__[uv].select_edge or crn__.link_loop_prev[uv].select_edge for crn__ in linked_next):
                                for deselect_crn in linked_next:
                                    deselect_crn[uv].select = False
                        else:
                            for deselect_crn in utils.linked_crn_uv_by_idx_unordered_included(cur_crn.link_loop_next, uv):
                                deselect_crn[uv].select = False

                            linked_prev = utils.linked_crn_uv_by_idx_unordered_included(cur_crn, uv)
                            if not any(crn__[uv].select_edge or crn__.link_loop_prev[uv].select_edge for crn__ in linked_prev):
                                for deselect_crn in linked_prev:
                                    deselect_crn[uv].select = False

                    update |= bool(shrink)

            umesh.update_tag = update

    @staticmethod
    def grow_prev(crn, selected_dir, uv, max_angle, with_seam) -> 'BMLoop | None | False':
        prev_crn = crn.link_loop_prev
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        cur_linked_corners = utils.linked_crn_uv_by_island_index_unordered(crn, uv, crn.face.index)

        # Skip if selected or with seam
        if prev_crn[uv].select_edge:
            return None

        if with_seam:
            for crn__ in cur_linked_corners:
                if crn__[uv].select_edge:
                    return None
                if (prev_crn__ := crn__.link_loop_prev) != shared:
                    if prev_crn__[uv].select_edge:
                        return None
        else:
            if prev_crn.edge.seam:
                return None
            for crn__ in cur_linked_corners:
                if crn__[uv].select_edge or crn__.edge.seam:
                    return None
                if (prev_crn__ := crn__.link_loop_prev) != shared:
                    if prev_crn__.edge.seam or prev_crn__[uv].select_edge:  # or crn__.edge.seam or crn__[uv].select_edge:
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
                if angle_ < angle:
                    angle = angle_
                    min_crn = crn_
                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(crn_[uv].uv - prev_crn_[uv].uv, max_angle)
                    if angle_ < angle:
                        angle = angle_
                        min_crn = crn_.link_loop_prev

            return min_crn
        return False

    @staticmethod
    def grow_next(crn, selected_dir, uv, max_angle, with_seam) -> 'BMLoop | None | False':
        next_crn = crn.link_loop_next
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        next_linked_corners = utils.linked_crn_uv_by_island_index_unordered(crn.link_loop_next, uv, crn.link_loop_next.face.index)

        # Skip if selected or with seam
        if with_seam:
            for crn__ in next_linked_corners:
                if crn__[uv].select_edge:
                    return None
                if (prev_crn__ := crn__.link_loop_prev) != shared:
                    if prev_crn__[uv].select_edge:
                        return None
        else:
            if next_crn.edge.seam:
                return None
            for crn__ in next_linked_corners:
                if crn__[uv].select_edge or crn__.edge.seam:
                    return None
                if (prev_crn__ := crn__.link_loop_prev) != shared:
                    if prev_crn__.edge.seam or prev_crn__[uv].select_edge:
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
                if angle_ < angle:
                    angle = angle_
                    min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(prev_crn_[uv].uv - crn_[uv].uv, max_angle)
                    if angle_ < angle:
                        angle = angle_
                        min_crn = crn_.link_loop_prev
            return min_crn
        return False


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
    island_mode: EnumProperty(name='Mode', default='ISLAND', items=(('ISLAND', 'Island', ''), ('FACE', 'Face', '')))

    target_texel: FloatProperty(name='Texel', default=512, min=1, soft_min=32, soft_max=2048, max=10_000)
    threshold: FloatProperty(name='Texel', default=0.01, min=0, soft_max=50, max=10_000)

    def draw(self, context):
        layout = self.layout
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'island_mode', expand=True)
        layout.prop(self, 'target_texel', slider=True)
        layout.prop(self, 'threshold', slider=True)

    def invoke(self, context, event):
        self.target_texel = settings().texel_density

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
        texture_size = (int(settings().size_x) + int(settings().size_y)) / 2
        umeshes = types.UMeshes()
        if umeshes.sync and utils.get_select_mode_mesh() != 'FACE':
            utils.set_select_mode_mesh('FACE')

        if not self.bl_idname.startswith('UV'):
            umeshes.set_sync()

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

                    if self.mode == 'SELECT':
                        if isclose(texel, self.target_texel, abs_tol=self.threshold):
                            if has_selected and isl.is_full_face_selected:
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
                        if isclose(texel, self.target_texel, abs_tol=self.threshold):
                            if has_selected and isl.is_full_face_selected:
                                counter_skipped += 1
                                continue
                            counter += 1
                            isl.select = True
                            umesh.update_tag = True
                    else:  # self.mode == 'DESELECT':
                        if isclose(texel, self.target_texel, abs_tol=self.threshold):
                            if isl.is_full_face_deselected:
                                counter_skipped += 1
                                continue
                            counter += 1
                            isl.select = False
                            umesh.update_tag = True

        if not has_elem:
            self.report({'WARNING'}, f'{self.island_mode.capitalize() + "s"} not found')
        else:
            sel_or_deselect = "deselected" if self.mode == "DESELECT" else "selected"
            if counter:
                self.report({'INFO'}, f'{sel_or_deselect.capitalize()} {counter} {self.island_mode + "s"}')
            else:
                if counter_skipped:
                    self.report({'INFO'}, f'The {self.island_mode.capitalize() + "s"} were already {sel_or_deselect}')
                else:
                    self.report({'WARNING'}, f'No found {self.island_mode.capitalize() + "s"} in the specified texel')
        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_SelectTexelDensity(UNIV_OT_SelectTexelDensity_VIEW3D):
    bl_idname = "uv.univ_select_texel_density"


class UNIV_OT_Tests(utils.UNIV_OT_Draw_Test):
    pass

    # def test_invoke(self, _event):
    #     self.max_angle = math.radians(20)  # noqa
    #
    #     for umesh in self.umeshes:
    #         uv = umesh.uv
    #         update = False
    #         if islands := Islands.calc_extended_any_edge_with_markseam(umesh):
    #             islands.indexing()
    #             for isl in islands:
    #                 grew = []
    #                 # corners = (crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge)
    #                 corners = [crn_ for f in isl for crn_ in f.loops if crn_[uv].select_edge]
    #                 for crn in corners:
    #
    #                     with_seam = crn.edge.seam
    #                     selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv
    #
    #                     if grow_prev_crn := UNIV_OT_Select_Edge_Grow_VIEW2D.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam):
    #                         if not with_seam:
    #                             if grow_prev_crn.edge.seam:
    #                                 continue
    #
    #                         cur_linked_corners_ = utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, crn.face.index)
    #                         cur_linked_corners_.extend([crn.link_loop_prev for crn in cur_linked_corners_])
    #
    #                         self.calc_from_corners(cur_linked_corners_, uv, exact=True)
    #                         grew.append(grow_prev_crn)
    #
    #                     # if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam):
    #                     #     if not with_seam:
    #                     #         if grow_next_crn.edge.seam:
    #                     #             continue
    #                     #     grew.append(grow_next_crn)
    #
    #                 for grew_crn in grew:
    #                     utils.select_crn_uv_edge_with_shared_by_idx(grew_crn, uv, force=True)
    #
    #                 update |= bool(grew)
    #
    #         umesh.update_tag = update


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

    threshold: FloatProperty(name='Threshold', default=0.01, min=0, soft_min=0.01, soft_max=0.1, max=0.5, subtype='FACTOR')
    lower_slider: FloatProperty(name='Low', default=0.2, min=0, max=0.9, subtype='PERCENTAGE',
        update=lambda self, _: setattr(self, 'higher_slider', self.lower_slider+0.05) if self.higher_slider-0.05 < self.lower_slider else None)
    higher_slider: FloatProperty(name='High', default=0.8, min=0.1, max=1, subtype='PERCENTAGE',
        update=lambda self, _: setattr(self, 'lower_slider', self.higher_slider-0.05) if self.higher_slider-0.05 < self.lower_slider else None)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'mode', expand=True)
        layout.row(align=True).prop(self, 'size_type', expand=True)
        layout.row(align=True).prop(self, 'size_mode', expand=True)
        layout.label(text=f'Small: {self.lower_slider*100:.2f}% \t\t\t'
                          f'Medium{((self.higher_slider-self.lower_slider)*100):.2f}% \t\t\t'
                          f'Large {(1-self.higher_slider)*100:.2f}%')
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
        umeshes = types.UMeshes()
        if umeshes.sync and utils.get_select_mode_mesh() != 'FACE':
            utils.set_select_mode_mesh('FACE')

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
        lower -= self.threshold
        higher += self.threshold

        for islands in islands_of_mesh:
            umesh = islands.umesh
            has_selected = islands.value
            for isl in islands:
                if self.mode == 'SELECT':
                    if lower <= isl.value <= higher:
                        if has_selected and isl.is_full_face_selected:
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
                        if has_selected and isl.is_full_face_selected:
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
            sel_or_deselect = "deselected" if self.mode == "DESELECT" else "selected"
            if counter:
                self.report({'INFO'}, f'{sel_or_deselect.capitalize()} {counter} islands')
            else:
                if counter_skipped:
                    self.report({'INFO'}, f'The islands were already{sel_or_deselect}')
                else:
                    self.report({'WARNING'}, f'No found in the specified size')
        umeshes.silent_update()
        return {'FINISHED'}

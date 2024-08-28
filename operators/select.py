# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import gpu
import math

from mathutils import Vector
from mathutils.geometry import area_tri
from bpy.props import *
from bpy.types import Operator
from bmesh.types import BMLoop, BMLayerItem
from gpu_extras.batch import batch_for_shader
from time import perf_counter as time

from .. import utils
from .. import types
from ..types import Islands, AdvIslands, AdvIsland,  MeshIslands, BBox


from ..utils import (
    UMeshes,
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
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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
        umeshes = utils.UMeshes(report=self.report)
        mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()

        if sync and mode == 'VERTEX':
            for umesh in umeshes:
                if types.PyBMesh.is_full_vert_selected(umesh.bm) or types.PyBMesh.is_full_vert_deselected(umesh.bm):
                    umesh.update_tag = False
                    continue
                has_full_selected = False
                half_selected = []
                if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, sync):
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
                if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, sync):
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
            if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, sync):
                for island in islands:
                    if update_state := (island.info_select(sync) == types.eInfoSelectFaceIsland.HALF_SELECTED):
                        island.deselect_set(mode=mode, sync=sync)
                    is_update |= update_state
            umesh.update_tag = is_update

            if is_update and sync:
                umesh.bm.select_flush_mode()

        return umeshes.update(info='No islands for deselect')

class UNIV_OT_SelectView(Operator):
    bl_idname = 'uv.univ_select_view'
    bl_label = 'Select View'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    face_mode: BoolProperty(name='Face Mode', default=False)

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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
        umeshes = utils.UMeshes(report=self.report)
        elem_mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()

        reg = context.area.regions[-1]
        assert reg.type == 'WINDOW'

        view_rect = BBox.init_from_minmax(
            Vector(reg.view2d.region_to_view(reg.x, reg.y)),
            Vector(reg.view2d.region_to_view(reg.x + reg.width, reg.y + reg.height))
        )

        padding = -(view_rect.min_length * 0.1)
        view_rect.pad(Vector((padding, padding)))

        view_island = AdvIsland([], None, None)  # noqa
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
                uv_layer = umesh.uv_layer
                if sync:
                    for f in umesh.bm.faces:
                        if not f.select:
                            if face_centroid_uv(f, uv_layer) in bb:
                                f.select = True
                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv_layer) in bb:
                                for _l in f.loops:
                                    luv = _l[uv_layer]
                                    luv.select = True
                                    luv.select_edge = True
                umesh.bm.select_flush_mode()

            else:
                has_update = False
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=False):
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
                uv_layer = umesh.uv_layer
                if sync:
                    for f in umesh.bm.faces:
                        if face_centroid_uv(f, uv_layer) in bb:
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
                            if face_centroid_uv(f, uv_layer) in bb:
                                for _l in f.loops:
                                    luv = _l[uv_layer]
                                    luv.select = True
                                    luv.select_edge = True
                            else:
                                for _l in f.loops:
                                    luv = _l[uv_layer]
                                    luv.select = False
                                    luv.select_edge = False
                umesh.bm.select_flush_mode()

            else:
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=False):
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
                uv_layer = umesh.uv_layer
                if sync:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv_layer) in bb:
                                f.select = False

                else:
                    for f in umesh.bm.faces:
                        if f.select:
                            if face_centroid_uv(f, uv_layer) in bb:
                                for _l in f.loops:
                                    luv = _l[uv_layer]
                                    luv.select = False
                                    luv.select_edge = False

                umesh.bm.select_flush_mode()

            else:
                has_update = False
                if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=False):
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

class UNIV_OT_Select_By_Cursor(UNIV_OT_SelectView):
    bl_idname = "uv.univ_select_by_cursor"
    bl_label = "Select by Cursor"
    bl_description = "Select by Cursor"

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'INFO'}, f"UV area not found")
            return {'CANCELLED'}

        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        umeshes = utils.UMeshes(report=self.report)
        elem_mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()

        tile_co = utils.get_tile_from_cursor()
        view_rect = BBox.init_from_minmax(tile_co, tile_co+Vector((1, 1)))
        view_rect.pad(Vector((-2e-08, -2e-08)))

        view_island = AdvIsland([], None, None)  # noqa
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

class UNIV_OT_Single(Operator):
    bl_idname = 'uv.univ_single'
    bl_label = 'Single'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    invert: BoolProperty(name='Invert', default=False)

    def __init__(self):
        self.sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.invert = event.alt

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITIONAL'
        else:
            self.mode = 'SELECT'

        return self.execute(context)

    def execute(self, context):

        if not self.sync:
            if self.mode == 'ADDITIONAL':
                bpy.ops.uv.univ_sync_uv_toggle()  # noqa
            bpy.context.scene.tool_settings.use_uv_select_sync = True
            self.sync = True

        if utils.get_select_mode_mesh() != 'FACE':
            utils.set_select_mode_mesh('FACE')

        self.umeshes = utils.UMeshes(report=self.report)
        self.select()

        return self.umeshes.update()

    def select(self):
        total_selected = 0
        total_deselected = 0

        for umesh in self.umeshes:
            if self.mode == 'ADDITIONAL' and types.PyBMesh.is_full_face_selected(umesh.bm):
                umesh.update_tag = False
                continue
            elif self.mode == 'DESELECT' and types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            selected = 0
            deselected = 0
            if islands := AdvIslands.calc_visible(umesh.bm, umesh.uv_layer, self.sync):
                islands.indexing()

                for mesh_island in MeshIslands.calc_visible(umesh):
                    indexes = {f.index for f in mesh_island}
                    state = len(indexes) == 1
                    if self.invert:
                        state ^= 1

                    if self.mode == 'SELECT':
                        if state:
                            mesh_island.select = True
                            selected += 1
                        else:
                            mesh_island.deselect(mode='FACE')
                            deselected += 1
                    elif self.mode == 'DESELECT':
                        if state:
                            mesh_island.deselect(mode='FACE')
                            deselected += 1
                    else:
                        if state:
                            mesh_island.select = True
                            deselected += 1

            umesh.update_tag = bool(selected + deselected)

        if total_selected or total_deselected:
            selected_deselected_info = ''
            if total_selected:
                selected_deselected_info = f'Total selected = {total_selected}. '
            if total_deselected:
                selected_deselected_info += f'Total deselected = {total_deselected}'

            self.report({'INFO'}, selected_deselected_info)

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
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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

        self.umeshes = utils.UMeshes(report=self.report)

        if self.mode == 'DESELECT':
            self.deselect()
        elif self.mode == 'ADDITIONAL':
            self.addition()
        else:
            self.select()

        return self.umeshes.update()

    def select(self):
        for umesh in self.umeshes:
            if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, self.sync):
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

            if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, self.sync):
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

            if islands := Islands.calc_visible(umesh.bm, umesh.uv_layer, self.sync):
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
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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
        self.umeshes = utils.UMeshes(report=self.report)

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

            uv_layer = umesh.uv_layer
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)

            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    crn_uv_b = crn.link_loop_next[uv_layer]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        to_select_corns.append(crn)
                    else:
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn_uv_b.select = False
                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            elif self.mode == 'DESELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    crn_uv_b = crn.link_loop_next[uv_layer]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        self.deselect_crn_uv_edge(crn, uv_layer)
                    else:
                        if crn_uv_a.select_edge:
                            to_select_corns.append(crn)
                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            else:  # 'ADDITIONAL'
                for loop in corners:
                    crn_uv_a = loop[uv_layer]
                    if crn_uv_a.select_edge:
                        continue
                    crn_uv_b = loop.link_loop_next[uv_layer]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    a = vec.angle(self.edge_orient)

                    if a <= self.angle or a >= self.negative_ange:
                        select_linked_crn_uv_vert(loop, uv_layer)
                        select_linked_crn_uv_vert(loop.link_loop_next, uv_layer)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn_uv_b.select = True

    def select_both(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv_layer = umesh.uv_layer
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    crn_uv_b = crn.link_loop_next[uv_layer]

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
                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            elif self.mode == 'DESELECT':
                to_select_corns = []
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    crn_uv_b = crn.link_loop_next[uv_layer]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    x_angle = vec.angle(self.x_vec)
                    y_angle = vec.angle(self.y_vec)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        self.deselect_crn_uv_edge(crn, uv_layer)
                    else:
                        if crn_uv_a.select_edge:
                            to_select_corns.append(crn)
                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    if crn_uv_a.select_edge:
                        continue
                    crn_uv_b = crn.link_loop_next[uv_layer]

                    vec = crn_uv_a.uv - crn_uv_b.uv
                    x_angle = vec.angle(self.x_vec)
                    y_angle = vec.angle(self.y_vec)

                    if x_angle <= self.angle or x_angle >= self.negative_ange or \
                            y_angle <= self.angle or y_angle >= self.negative_ange:
                        select_linked_crn_uv_vert(crn, uv_layer)
                        select_linked_crn_uv_vert(crn.link_loop_next, uv_layer)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn_uv_b.select = True

    def select_both_border(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv_layer = umesh.uv_layer
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv_layer):
                        crn_uv_a = crn[uv_layer]
                        crn_uv_b = crn.link_loop_next[uv_layer]

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
                        crn_uv_a = crn[uv_layer]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv_layer].select = False

                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv_layer):
                        crn_uv_a = crn[uv_layer]
                        crn_uv_b = crn.link_loop_next[uv_layer]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        x_angle = vec.angle(self.x_vec)
                        y_angle = vec.angle(self.y_vec)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            self.deselect_crn_uv_edge_for_border(crn, uv_layer)
                # TODO: Refactor
                for crn in _corners:
                    crn_uv = crn[uv_layer]
                    if crn_uv.select_edge:
                        self.select_crn_uv_edge_sequence([crn], uv_layer)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv_layer):
                        crn_uv_b = crn.link_loop_next[uv_layer]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        x_angle = vec.angle(self.x_vec)
                        y_angle = vec.angle(self.y_vec)

                        if x_angle <= self.angle or x_angle >= self.negative_ange or \
                                y_angle <= self.angle or y_angle >= self.negative_ange:
                            select_linked_crn_uv_vert(crn, uv_layer)
                            select_linked_crn_uv_vert(crn.link_loop_next, uv_layer)
                            crn_uv_a.select = True
                            crn_uv_a.select_edge = True
                            crn_uv_b.select = True

    def select_hv_border(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue

            uv_layer = umesh.uv_layer
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv_layer):
                        crn_uv_a = crn[uv_layer]
                        crn_uv_b = crn.link_loop_next[uv_layer]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            to_select_corns.append(crn)
                        else:
                            crn_uv_a.select = False
                            crn_uv_a.select_edge = False
                            crn_uv_b.select = False
                    else:
                        crn_uv_a = crn[uv_layer]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv_layer].select = False

                self.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv_layer):
                        crn_uv_a = crn[uv_layer]
                        crn_uv_b = crn.link_loop_next[uv_layer]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            self.deselect_crn_uv_edge_for_border(crn, uv_layer)
                            # Removing the notches
                            if is_boundary(crn.link_loop_next, uv_layer):
                                crn_uv_c = crn.link_loop_next[uv_layer]
                                crn_uv_c.select = False
                                crn_uv_c.select_edge = False
                            if is_boundary(crn.link_loop_prev, uv_layer):
                                crn_uv_d = crn.link_loop_prev[uv_layer]
                                crn_uv_d.select = False
                                crn_uv_d.select_edge = False

                # TODO: Refactor
                for crn in _corners:
                    crn_uv = crn[uv_layer]
                    if crn_uv.select_edge:
                        self.select_crn_uv_edge_sequence([crn], uv_layer)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv_layer):
                        crn_uv_b = crn.link_loop_next[uv_layer]

                        vec = crn_uv_a.uv - crn_uv_b.uv
                        a = vec.angle(self.edge_orient)

                        if a <= self.angle or a >= self.negative_ange:
                            select_linked_crn_uv_vert(crn, uv_layer)
                            select_linked_crn_uv_vert(crn.link_loop_next, uv_layer)
                            crn_uv_a.select = True
                            crn_uv_a.select_edge = True
                            crn_uv_b.select = True

    @staticmethod
    def select_crn_uv_edge_sequence(to_select_corns: list[BMLoop], uv_layer):
        for crn in to_select_corns:
            link_crn_next = crn.link_loop_next
            select_linked_crn_uv_vert(crn, uv_layer)
            select_linked_crn_uv_vert(link_crn_next, uv_layer)

            crn_uv_a = crn[uv_layer]
            crn_uv_b = link_crn_next[uv_layer]
            crn_uv_a.select = True
            crn_uv_a.select_edge = True
            crn_uv_b.select = True

    @staticmethod
    def select_crn_uv_edge(crn: BMLoop, uv_layer):
        link_crn_next = crn.link_loop_next
        select_linked_crn_uv_vert(crn, uv_layer)
        select_linked_crn_uv_vert(link_crn_next, uv_layer)

        crn_uv_a = crn[uv_layer]
        crn_uv_b = link_crn_next[uv_layer]
        crn_uv_a.select = True
        crn_uv_a.select_edge = True
        crn_uv_b.select = True

    @staticmethod
    def deselect_crn_uv_edge(crn: BMLoop, uv_layer: BMLayerItem):
        link_crn_next = crn.link_loop_next
        deselect_linked_crn_uv_vert(crn, uv_layer)
        deselect_linked_crn_uv_vert(link_crn_next, uv_layer)

        crn_uv_a = crn[uv_layer]
        crn_uv_b = link_crn_next[uv_layer]
        crn_uv_a.select = False
        crn_uv_a.select_edge = False
        crn_uv_b.select = False

    @staticmethod
    def deselect_crn_uv_edge_for_border(crn: BMLoop, uv_layer):
        def _deselect_linked_crn_uv_vert(first: BMLoop):
            bm_iter = first
            while True:
                if (bm_iter := bm_iter.link_loop_prev.link_loop_radial_prev) == first:
                    break
                crn_uv_bm_iter = bm_iter[uv_layer]
                if first[uv_layer].uv == crn_uv_bm_iter.uv:
                    crn_uv_bm_iter.select = False
                    crn_uv_bm_iter.select_edge = False
                    bm_iter.link_loop_prev[uv_layer].select = False
                    bm_iter.link_loop_prev[uv_layer].select_edge = False

        _deselect_linked_crn_uv_vert(crn)
        link_crn_next = crn.link_loop_next
        deselect_linked_crn_uv_vert(crn, uv_layer)
        deselect_linked_crn_uv_vert(link_crn_next, uv_layer)

        crn_uv_a = crn[uv_layer]
        crn.link_loop_prev[uv_layer].select_edge = False
        crn_uv_b = link_crn_next[uv_layer]
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
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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

        self.umeshes = utils.UMeshes(report=self.report)
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

            uv_layer = umesh.uv_layer
            corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)
            if self.mode == 'SELECT':
                to_select_corns = []
                for crn in corners:
                    if is_boundary(crn, uv_layer):
                        to_select_corns.append(crn)
                    else:
                        crn_uv_a = crn[uv_layer]
                        crn_uv_a.select = False
                        crn_uv_a.select_edge = False
                        crn.link_loop_next[uv_layer].select = False

                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge_sequence(to_select_corns, uv_layer)

            elif self.mode == 'DESELECT':
                _corners = list(corners)
                for crn in _corners:
                    if is_boundary(crn, uv_layer):
                        UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(crn, uv_layer)
                for crn in _corners:
                    crn_uv = crn[uv_layer]
                    if crn_uv.select_edge:
                        UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge_sequence([crn], uv_layer)

            else:  # 'ADDITION'
                for crn in corners:
                    crn_uv_a = crn[uv_layer]
                    if crn_uv_a.select_edge:
                        continue
                    if is_boundary(crn, uv_layer):
                        select_linked_crn_uv_vert(crn, uv_layer)
                        select_linked_crn_uv_vert(crn.link_loop_next, uv_layer)
                        crn_uv_a.select = True
                        crn_uv_a.select_edge = True
                        crn.link_loop_next[uv_layer].select = True

    def select_border_between(self):
        # bpy.ops.uv.select_all(action='DESELECT')
        for umesh in self.umeshes:
            uv_layer = umesh.uv_layer
            if self.mode == 'SELECT':
                islands = Islands.calc_extended(umesh.bm, umesh.uv_layer, self.sync)

                for _f in umesh.bm.faces:
                    for _crn in _f.loops:
                        _crn_uv = _crn[uv_layer]
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
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv_layer)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv_layer)

            elif self.mode == 'DESELECT':
                islands = Islands.calc_extended(umesh.bm, umesh.uv_layer, self.sync)

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
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(crn, uv_layer)
                                UNIV_OT_Select_Border_Edge_by_Angle.deselect_crn_uv_edge_for_border(shared_crn, uv_layer)
                                # Removing the notches
                                if is_boundary(crn.link_loop_next, uv_layer):
                                    crn_uv_c = crn.link_loop_next[uv_layer]
                                    crn_uv_c.select = False
                                    crn_uv_c.select_edge = False
            else:  # 'ADDITION'
                islands = Islands.calc_extended(umesh.bm, umesh.uv_layer, self.sync)

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
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(crn, uv_layer)
                                UNIV_OT_Select_Border_Edge_by_Angle.select_crn_uv_edge(shared_crn, uv_layer)
                umesh.update_tag = bool(islands)


class UNIV_OT_Select_Inner(Operator):
    bl_idname = 'uv.univ_select_inner'
    bl_label = 'Select Inner'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))
    seam: BoolProperty(name='Seam', default=True)
    sharp: BoolProperty(name='Sharp', default=True)
    mtl: BoolProperty(name='Mtl', default=True)

    def draw(self, context):
        self.layout.prop(self, 'angle', slider=True)

        row = self.layout.row(align=True)
        row.prop(self, 'sharp', toggle=True)
        row.prop(self, 'seam', toggle=True)
        row.prop(self, 'mtl', toggle=True)

        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)

    def __init__(self):
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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

    def execute(self, context):
        self.umeshes = utils.UMeshes(report=self.report)

        if self.sync:
            if utils.get_select_mode_mesh() != 'EDGE':
                utils.set_select_mode_mesh('EDGE')
            if self.mode == 'SELECT':
                bpy.ops.mesh.select_all(action='DESELECT')
            self.select_inner_sync()

        else:
            if utils.get_select_mode_uv() not in ('EDGE', 'VERTEX'):
                utils.set_select_mode_uv('EDGE')
            if self.mode == 'SELECT':
                bpy.ops.uv.select_all(action='DESELECT')
            self.select_inner()

        self.umeshes.update()
        return {'FINISHED'}

    def select_inner(self):
        for umesh in self.umeshes:
            if types.PyBMesh.is_full_face_deselected(umesh.bm):
                umesh.update_tag = False
                continue
            update_tag = False
            uv = umesh.uv_layer
            if self.mode == 'ADDITION':
                corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops if not crn[uv].select_edge)
            else:
                corners = (crn for face in umesh.bm.faces if face.select for crn in face.loops)

            if self.mode == 'DESELECT' and utils.get_select_mode_uv() == 'EDGE':
                corners = list(corners)

            for crn in corners:
                shared_crn = crn.link_loop_radial_prev
                if shared_crn == crn:
                    continue
                if not shared_crn.face.select:
                    continue
                if not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv):
                    continue
                if (self.sharp and not crn.edge.smooth) or \
                        crn.edge.calc_face_angle() >= self.angle or \
                        (self.seam and crn.edge.seam) or \
                        (self.mtl and shared_crn.face.material_index != crn.face.material_index):
                    if self.mode == 'DESELECT':
                        utils.deselect_crn_uv_force(crn, uv)
                    else:
                        utils.select_crn_uv_edge(crn, uv)
                    update_tag = True

            if update_tag and self.mode == 'DESELECT' and utils.get_select_mode_uv() == 'EDGE':
                for crn in corners:
                    crv_uv = crn[uv]
                    if crv_uv.select_edge:
                        if not (crv_uv.select and crn.link_loop_next[uv].select):
                            crv_uv.select_edge = False

            umesh.update_tag = update_tag

    def select_inner_sync(self):
        for umesh in self.umeshes:
            update_tag = False
            uv = umesh.uv_layer
            if self.mode == 'ADDITION':
                corners = (crn for face in umesh.bm.faces if not face.hide for crn in face.loops if not crn.edge.select)
            else:
                corners = (crn for face in umesh.bm.faces if not face.hide for crn in face.loops)

            for crn in corners:
                shared_crn = crn.link_loop_radial_prev
                if shared_crn == crn:
                    continue
                if shared_crn.face.hide:
                    continue
                if not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv):
                    continue
                if (self.sharp and not crn.edge.smooth) or \
                        crn.edge.calc_face_angle() >= self.angle or \
                        (self.seam and crn.edge.seam) or \
                        (self.mtl and shared_crn.face.material_index != crn.face.material_index):
                    if self.mode == 'DESELECT':
                        crn.edge.select = False
                        crn.link_loop_next.edge.select = False
                        crn.link_loop_prev.edge.select = False
                    else:
                        crn.edge.select = True
                    update_tag = True
            umesh.update_tag = update_tag
            if update_tag:
                umesh.bm.select_flush_mode()

class UNIV_OT_Select_Zero(bpy.types.Operator):
    bl_idname = "uv.univ_select_zero"
    bl_label = "Select Degenerate"
    bl_description = "Select degenerate UVs (zero area UV triangles)"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    precision: bpy.props.FloatProperty(name='Precision', default=0.0001, min=0, soft_max=0.001, step=0.0001, precision=7)  # noqa

    def draw(self, context):
        self.layout.prop(self, 'precision', slider=True)
        self.layout.row(align=True).prop(self, 'mode', expand=True)

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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

    def __init__(self):
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        total_counter = self.zero()

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    def zero(self):
        self.umeshes = UMeshes()
        if self.sync:
            if self.mode == 'SELECT':
                bpy.ops.mesh.select_all(action='DESELECT')
            utils.set_select_mode_mesh('FACE')
        else:
            if self.mode == 'SELECT':
                bpy.ops.uv.select_all(action='DESELECT')
        select_state = False if self.mode == 'DESELECT' else True
        precision = self.precision * 0.001
        total_counter = 0
        for umesh in self.umeshes:
            if not self.sync and umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue

            local_counter = 0
            uv = umesh.uv_layer
            loop_triangles = umesh.bm.calc_loop_triangles()
            for tris in loop_triangles:
                if self.sync:
                    if tris[0].face.hide:
                        continue
                else:
                    if not tris[0].face.select:
                        continue
                area = area_tri(tris[0][uv].uv, tris[1][uv].uv, tris[2][uv].uv)
                if area < precision:
                    if self.sync:
                        tris[0].face.select = select_state
                    else:
                        for crn in tris[0].face.loops:
                            crn[uv].select = select_state
                            crn[uv].select_edge = select_state
                    local_counter += 1
            umesh.update_tag = bool(local_counter)
            total_counter += local_counter
        self.umeshes.update()
        return total_counter

class UNIV_OT_Select_Flipped(bpy.types.Operator):
    bl_idname = "uv.univ_select_flipped"
    bl_label = "Select Flipped"
    bl_description = "Select flipped UV faces"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    def draw(self, context):
        self.layout.row(align=True).prop(self, 'mode', expand=True)

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

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

    def __init__(self):
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        total_counter = self.zero()

        if not total_counter:
            self.report({'INFO'}, 'Flipped faces not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} flipped faces')
        return {'FINISHED'}

    def zero(self):
        self.umeshes = UMeshes()
        if self.sync:
            if self.mode == 'SELECT':
                bpy.ops.mesh.select_all(action='DESELECT')
            utils.set_select_mode_mesh('FACE')
        else:
            if self.mode == 'SELECT':
                bpy.ops.uv.select_all(action='DESELECT')
        select_state = False if self.mode == 'DESELECT' else True

        total_counter = 0
        for umesh in self.umeshes:
            if not self.sync and umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue

            local_counter = 0
            uv = umesh.uv_layer
            loop_triangles = umesh.bm.calc_loop_triangles()
            for tris in loop_triangles:
                if self.sync:
                    if tris[0].face.hide:
                        continue
                else:
                    if not tris[0].face.select:
                        continue
                a = tris[0][uv].uv
                b = tris[1][uv].uv
                c = tris[2][uv].uv

                area = a.cross(b) + b.cross(c) + c.cross(a)
                if area < 0:
                    if self.sync:
                        tris[0].face.select = select_state
                    else:
                        for crn in tris[0].face.loops:
                            crn[uv].select = select_state
                            crn[uv].select_edge = select_state
                    local_counter += 1
            umesh.update_tag = bool(local_counter)
            total_counter += local_counter
        self.umeshes.update()
        return total_counter

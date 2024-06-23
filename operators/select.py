import bpy
import gpu
# import math
# import numpy as np

from bpy.types import Operator
from bpy.props import *

from .. import utils
from .. import info
from .. import types
from ..utils import UMeshes, face_centroid_uv
from ..types import Islands, AdvIslands, AdvIsland  # , FaceIsland, UnionIslands

from mathutils import Vector
from time import perf_counter as time
from gpu_extras.batch import batch_for_shader


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
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    batch = batch_for_shader(shader, 'LINES', {"pos": data})

    if not (uv_handle is None):
        bpy.types.SpaceImageEditor.draw_handler_remove(uv_handle, 'WINDOW')

    uv_handle = bpy.types.SpaceImageEditor.draw_handler_add(draw_callback_px, (), 'WINDOW', 'POST_VIEW')
    bpy.app.timers.register(uv_area_draw_timer)

class UNIV_OT_SelectLinked(Operator):
    bl_idname = 'uv.univ_select_linked'
    bl_label = 'Select Linked'
    bl_options = {'REGISTER', 'UNDO'}

    deselect: bpy.props.BoolProperty(name='Mode', default=False)

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
        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.deselect = False
            case True, False, False:
                self.deselect = True
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement. \n\n")
                return {'CANCELLED'}
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
                        island.deselect(mode=mode, sync=sync)
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
        ('DESELECT_INVERTED', 'Deselect Inverted', ''),
    ))
    face_mode = BoolProperty(name='Face Mode', default=False)

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

        view_rect = types.View2D.get_rect(context.area.regions[-1].view2d).copy()
        view_rect.xmax -= bpy.context.preferences.system.ui_scale  # category panel compensation

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
                                # for v in f.verts:
                                #     v.select = True
                                #
                                # for e in f.edges:
                                #     e.select = True
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
                            island.select(elem_mode, sync)
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
                    # TODO: Implement border selection safe
                    for f in umesh.bm.faces:
                        if face_centroid_uv(f, uv_layer) in bb:
                            f.select = True
                            # # TODO: Optimize that moment
                            # for v in f.verts:
                            #     v.select = True
                            #
                            # for e in f.edges:
                            #     e.select = True
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
                                island.deselect(elem_mode, sync)
                        for island in adv_islands:
                            if island.tag:
                                island.select(elem_mode, sync)

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
                                # for v in f.verts:
                                #     v.select = False
                                #
                                # for e in f.edges:
                                #     e.select = False

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
                            island.deselect(elem_mode, sync)
                            has_update = True

                if sync and has_update and elem_mode in ('VERTEX', 'EDGE'):
                    umesh.bm.select_flush_mode()
                umesh.update_tag = has_update

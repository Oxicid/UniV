# import bpy
# import math
import typing

import bmesh
import mathutils

import numpy as np

# from mathutils import Vector
from collections import defaultdict

from .bench import timer, profile
from . import umath
from .umath import *
from .other import *
from .ubm import *
from ..types import PyBMesh

class UMesh:
    def __init__(self, bm, obj, is_edit_bm=True):
        self.bm: bmesh.types.BMesh = bm
        self.obj: bpy.types.Object = obj
        self.uv_layer: bmesh.types.BMLayerItem = bm.loops.layers.uv.verify()
        self.is_edit_bm: bool = is_edit_bm
        self.update_tag: bool = True

    def update(self, force=False):
        if not self.update_tag:
            return False
        if self.is_edit_bm:
            bmesh.update_edit_mesh(self.obj.data, loop_triangles=force, destructive=force)
        else:
            self.bm.to_mesh(self.obj.data)
        return True

    def free(self):
        self.bm.free()

    def ensure(self, face=True, edge=False, vert=False, force=False):
        if self.is_edit_bm or not force:
            return
        if face:
            self.bm.faces.ensure_lookup_table()
        if edge:
            self.bm.edges.ensure_lookup_table()
        if vert:
            self.bm.verts.ensure_lookup_table()

    def __del__(self):
        if not self.is_edit_bm:
            self.bm.free()


class UMeshes:
    def __init__(self, umeshes=None, *, report=None):
        if umeshes is None:
            self._sel_ob_with_uv()
        else:
            self.umeshes: list[UMesh] = umeshes
        self.report_obj = report
        self._cancel = False

    def report(self, info_type={'INFO'}, info="No uv for manipulate"):  # noqa
        if self.report_obj is None:
            print(info_type, info)
            return
        self.report_obj(info_type, info)

    def cancel_with_report(self, info_type: set[str]={'INFO'}, info: str ="No uv for manipulate"): # noqa
        self._cancel = True
        self.report(info_type, info)

    def update(self, force=False, info_type={'INFO'}, info="No uv for manipulate"):  # noqa
        if self._cancel is True:
            return {'CANCELLED'}
        if sum(umesh.update(force=force) for umesh in self.umeshes):
            return {'FINISHED'}
        if info:
            self.report(info_type, info)
        return {'CANCELLED'}

    def final(self):
        if self._cancel is True:
            return True
        return any(umesh.update_tag for umesh in self.umeshes)

    def ensure(self, face=True, edge=False, vert=False, force=False):
        for umesh in self.umeshes:
            umesh.ensure(face, edge, vert, force)

    def loop(self):
        active = bpy.context.active_object
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        area = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'][0]
        with bpy.context.temp_override(area=area):
            bpy.ops.object.select_all(action='DESELECT')

        for umesh in self.umeshes:
            bpy.context.view_layer.objects.active = umesh.obj
            umesh.obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)

            bm = bmesh.from_edit_mesh(umesh.obj.data)
            yield UMesh(bm, umesh.obj)

            umesh.obj.select_set(False)
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        for umesh in self.umeshes:
            umesh.obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.context.view_layer.objects.active = active

    @classmethod
    def sel_ob_with_uv(cls):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    data_and_objects[obj.data].append(obj)

            for data, obj in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, obj, False))

        return cls(bmeshes)

    def _sel_ob_with_uv(self):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    data_and_objects[obj.data].append(obj)

            for data, obj in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, obj, False))
        self.umeshes = bmeshes

    def __iter__(self) -> typing.Iterator[UMesh]:
        return iter(self.umeshes)

    def __getitem__(self, item):
        return self.umeshes[item]

    def __len__(self):
        return len(self.umeshes)

    def __bool__(self):
        return bool(self.umeshes)


def calc_selected_uv_faces(bm, uv_layer, sync) -> list[bmesh.types.BMFace]:
    if PyBMesh.is_full_face_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return [f for f in bm.faces if f.select]

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops)]
        else:
            return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops)]
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select]
    else:
        return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select]

def calc_selected_uv_faces_iter(bm, uv_layer, sync) -> 'typing.Generator[bmesh.types.BMFace] | tuple':
    if PyBMesh.is_full_face_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return (f for f in bm.faces if f.select)

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops))
        else:
            return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops))
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select)
    else:
        return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select)

def calc_visible_uv_faces(bm, uv_layer, sync) -> list[bmesh.types.BMFace]:  # noqa
    if PyBMesh.is_full_face_selected(bm):
        return bm.faces
    if sync:
        return [f for f in bm.faces if not f.hide]
    return [f for f in bm.faces if f.select]

def calc_uv_faces(bm, uv_layer, sync, *, selected) -> list[bmesh.types.BMFace]:
    if selected:
        return calc_selected_uv_faces(bm, uv_layer, sync)
    return calc_visible_uv_faces(bm, uv_layer, sync)

def calc_selected_uv_corners(bm, uv_layer, sync) -> list[bmesh.types.BMLoop]:
    if PyBMesh.is_full_vert_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return [l for f in bm.faces for l in f.loops]
        return [l for f in bm.faces for l in f.loops if l.vert.select]

    if PyBMesh.is_full_face_selected(bm):
        return [l for f in bm.faces for l in f.loops if l[uv_layer].select]
    return [l for f in bm.faces if f.select for l in f.loops if l[uv_layer].select]

def calc_selected_uv_corners_iter(bm, uv_layer, sync) -> 'typing.Generator[bmesh.types.BMLoop] | tuple':
    if PyBMesh.is_full_vert_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return (l for f in bm.faces for l in f.loops)
        return (l for f in bm.faces for l in f.loops if l.vert.select)

    if PyBMesh.is_full_face_selected(bm):
        return (luv for f in bm.faces for luv in f.loops if luv[uv_layer].select)
    return (luv for f in bm.faces if f.select for luv in f.loops if luv[uv_layer].select)

def calc_visible_uv_corners(bm, sync) -> list[bmesh.types.BMLoop]:
    if sync:
        return [luv for f in bm.faces if not f.hide for luv in f.loops]
    if PyBMesh.fields(bm).totfacesel == 0:
        return []
    return [luv for f in bm.faces if (f.select and not f.hide) for luv in f.loops]

def calc_uv_corners(bm, uv_layer, sync, *, selected) -> list[bmesh.types.BMLoop]:
    if selected:
        return calc_selected_uv_corners(bm, uv_layer, sync)
    return calc_visible_uv_corners(bm, sync)

def calc_avg_normal():
    umeshes = UMeshes.sel_ob_with_uv()
    size = sum(len(umesh.bm.faces) for umesh in umeshes)

    normals = np.empty(3 * size).reshape((-1, 3))
    areas = np.empty(size)

    i = 0
    for umesh in umeshes:
        for f in umesh.bm.faces:
            normals[i] = f.normal.to_tuple()
            areas[i] = f.calc_area()
            i += 1

    weighted_normals = normals * areas[:, np.newaxis]
    summed_normals = np.sum(weighted_normals, axis=0)

    return summed_normals / np.linalg.norm(summed_normals)

def find_min_rotate_angle(angle):
    return -(round(angle / (math.pi / 2)) * (math.pi / 2) - angle)


def calc_min_align_angle(points):
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)


def calc_min_align_angle_pt(points):
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)

def get_cursor_location() -> Vector:
    if bpy.context.area.ui_type == 'UV':
        return bpy.context.space_data.cursor_location.copy()
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.ui_type == 'UV':
                return area.spaces.active.cursor_location.copy()

def get_tile_from_cursor() -> Vector:
    if cursor := get_cursor_location():
        return Vector((math.floor(val) for val in cursor))

def set_cursor_location(loc):
    if bpy.context.area.ui_type == 'UV':
        bpy.context.space_data.cursor_location = loc
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.ui_type == 'UV':
                area.spaces.active.cursor_location = loc
                return

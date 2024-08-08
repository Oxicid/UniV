# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing  # noqa
import math  # noqa

import bmesh
import mathutils  # noqa

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
        self.sync: bool = sync()

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

    def check_uniform_scale(self, report=None):
        _, _, scale = self.obj.matrix_world.decompose()
        if not vec_isclose_to_uniform(scale, 0.01):
            if report:
                report({'WARNING'}, f'Object {self.obj.name} has non-uniform scale: {scale}')
            return False
        return True

    @property
    def is_full_face_selected(self):
        return PyBMesh.is_full_face_selected(self.bm)

    @property
    def is_full_face_deselected(self):
        return PyBMesh.fields(self.bm).totfacesel == 0

    @property
    def is_full_edge_selected(self):
        return PyBMesh.is_full_edge_selected(self.bm)

    @property
    def is_full_edge_deselected(self):
        return PyBMesh.is_full_edge_deselected(self.bm)

    @property
    def is_full_vert_selected(self):
        return PyBMesh.is_full_vert_selected(self.bm)

    @property
    def is_full_vert_deselected(self):
        return PyBMesh.is_full_vert_deselected(self.bm)

    @property
    def total_vert_sel(self):
        return PyBMesh.fields(self.bm).totvertsel

    @property
    def total_edge_sel(self):
        return PyBMesh.fields(self.bm).totedgesel

    @property
    def total_face_sel(self):
        return PyBMesh.fields(self.bm).totfacesel

    @property
    def total_corners(self):
        return PyBMesh.fields(self.bm).totloop

    @property
    def has_any_selected_crn_non_sync(self):
        if PyBMesh.is_full_face_deselected(self.bm):
            return False

        uv = self.uv_layer
        if PyBMesh.is_full_face_selected(self.bm):
            for f in self.bm.faces:
                for _crn in f.loops:
                    crn_uv = _crn[uv]
                    if crn_uv.select or crn_uv.select_edge:
                        return True
            return False

        for f in self.bm.faces:
            if f.select:
                for _crn in f.loops:
                    crn_uv = _crn[uv]
                    if crn_uv.select or crn_uv.select_edge:
                        return True
        return False

    @property
    def has_any_selected_crn_edge_non_sync(self):
        if PyBMesh.is_full_face_deselected(self.bm):
            return False

        uv = self.uv_layer
        if PyBMesh.is_full_face_selected(self.bm):
            for f in self.bm.faces:
                for crn in f.loops:
                    if crn[uv].select_edge:
                        return True
            return False

        for f in self.bm.faces:
            if f.select:
                for crn in f.loops:
                    if crn[uv].select_edge:
                        return True
        return False

    @property
    def has_any_selected_crn_vert_non_sync(self):
        if PyBMesh.is_full_face_deselected(self.bm):
            return False

        uv = self.uv_layer
        if PyBMesh.is_full_face_selected(self.bm):
            for f in self.bm.faces:
                for crn in f.loops:
                    if crn[uv].select:
                        return True
            return False

        for f in self.bm.faces:
            if f.select:
                for crn in f.loops:
                    if crn[uv].select:
                        return True
        return False

    @property
    def smooth_angle(self):
        if hasattr(self.obj.data, 'use_auto_smooth'):
            if self.obj.data.use_auto_smooth:
                return self.obj.data.auto_smooth_angle  # noqa
        else:
            for mod in self.obj.modifiers:
                if 'Smooth by Angle' not in mod.name:
                    continue
                if not (mod.show_in_editmode and mod.show_viewport):
                    continue
                if 'Input_1' in mod:
                    if isinstance(value := mod['Input_1'], float):
                        return value
        return math.radians(180.0)

    def tag_hidden_corners(self):
        corners = (_crn for f in self.bm.faces for _crn in f.loops)
        if self.sync:
            if self.is_full_face_selected:
                for crn in corners:
                    crn.tag = False
            else:
                for f in self.bm.faces:
                    h_tag = f.hide
                    for crn in f.loops:
                        crn.tag = h_tag
        else:
            if self.is_full_face_deselected:
                for crn in corners:
                    crn.tag = True
            else:
                for f in self.bm.faces:
                    s_tag = f.select
                    for crn in f.loops:
                        crn.tag = s_tag

    def tag_visible_corners(self):
        corners = (_crn for f in self.bm.faces for _crn in f.loops)
        if self.sync:
            if self.is_full_face_selected:
                for crn in corners:
                    crn.tag = True
            else:
                for f in self.bm.faces:
                    h_tag = not f.hide
                    for crn in f.loops:
                        crn.tag = h_tag
        else:
            if self.is_full_face_deselected:
                for crn in corners:
                    crn.tag = False
            else:
                for f in self.bm.faces:
                    s_tag = f.select
                    for crn in f.loops:
                        crn.tag = s_tag

    def tag_selected_corners(self, both=False):
        corners = (_crn for f in self.bm.faces for _crn in f.loops)
        if self.sync:
            if self.is_full_face_selected:
                for crn in corners:
                    crn.tag = True
            else:
                for f in self.bm.faces:
                    if f.hide:
                        for crn in f.loops:
                            crn.tag = False
                    else:
                        for crn in f.loops:
                            crn.tag = crn.edge.select
        else:
            if self.is_full_face_deselected:
                for crn in corners:
                    crn.tag = False
            else:
                uv = self.uv_layer
                if both:
                    for f in self.bm.faces:
                        if f.select:
                            for crn in f.loops:
                                crn_uv = crn[uv]
                                crn.tag = crn_uv.select_edge or crn_uv.select
                        else:
                            for crn in f.loops:
                                crn.tag = False
                else:
                    for f in self.bm.faces:
                        if f.select:
                            for crn in f.loops:
                                crn.tag = crn[uv].select_edge
                        else:
                            for crn in f.loops:
                                crn.tag = False

    def tag_selected_faces(self, both=False):
        if self.sync:
            if self.is_full_face_selected:
                self.set_face_tag()
            else:
                for f in self.bm.faces:
                    f.tag = not f.hide
        else:
            if self.is_full_face_deselected:
                self.set_face_tag(False)
            else:
                uv = self.uv_layer
                if both:
                    for f in self.bm.faces:
                        if f.select:
                            f.tag = all(crn[uv].select_edge or crn[uv].select for crn in f.loops)
                        else:
                            f.tag = False
                else:
                    for f in self.bm.faces:
                        if f.select:
                            f.tag = all(crn[uv].select_edge for crn in f.loops)
                        else:
                            f.tag = False

    def tag_selected_edge_linked_crn_sync(self):
        if PyBMesh.is_full_edge_selected(self.bm):
            self.set_tag()
            return
        if PyBMesh.is_full_edge_deselected(self.bm):
            self.set_tag(False)
            return

        self.set_tag(False)

        for e in self.bm.edges:
            if e.select:
                for v in e.verts:
                    for crn in v.link_loops:
                        crn.tag = not crn.face.hide

    def set_tag(self, state=True):
        for f in self.bm.faces:
            if f.select:
                for crn in f.loops:
                    crn.tag = state

    def set_face_tag(self, state=True):
        for f in self.bm.faces:
            f.tag = state

    def calc_selected_faces(self) -> list[BMFace] or bmesh.types.BMFaceSeq:
        if self.is_full_face_deselected:
            return []

        if self.is_full_face_selected:
            return self.bm.faces
        return [f for f in self.bm.faces if f.select]

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
        return {'CANCELLED'}

    def update(self, force=False, info_type={'INFO'}, info="No uv for manipulate"):  # noqa
        if self._cancel is True:
            return {'CANCELLED'}
        if sum(umesh.update(force=force) for umesh in self.umeshes):
            return {'FINISHED'}
        if info:
            self.report(info_type, info)
        return {'CANCELLED'}

    def silent_update(self):
        for umesh in self:
            umesh.update()

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
        with bpy.context.temp_override(area=area):  # noqa
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

    def set_sync(self, state=True):
        for umesh in self:
            umesh.sync = state

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

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, objs[0], False))
        self.umeshes = bmeshes

    @classmethod
    def calc(cls):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH':
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, objs[0], False))
        return cls(bmeshes)

    def filter_selected_faces(self):
        for umesh in reversed(self.umeshes):
            if umesh.is_full_face_deselected:
                self.umeshes.remove(umesh)

    def __iter__(self) -> typing.Iterator[UMesh]:
        return iter(self.umeshes)

    def __getitem__(self, item):
        return self.umeshes[item]

    def __len__(self):
        return len(self.umeshes)

    def __bool__(self):
        return bool(self.umeshes)


class NoInit:
    def __getattribute__(self, item):
        raise AttributeError(f'Object not initialized')

    def __bool__(self):
        raise AttributeError(f'Object not initialized')

    def __len__(self):
        raise AttributeError(f'Object not initialized')

def sync():
    return bpy.context.scene.tool_settings.use_uv_select_sync

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

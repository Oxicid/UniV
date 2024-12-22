# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing  # noqa

import bpy
import bmesh

from collections import defaultdict
from math import pi

from .. import utils
from ..types import PyBMesh

class FakeBMesh:
    def __init__(self, isl):
        self.faces = isl

class UMesh:
    def __init__(self, bm, obj, is_edit_bm=True):
        self.bm: bmesh.types.BMesh | FakeBMesh = bm
        self.obj: bpy.types.Object = obj
        self.uv: bmesh.types.BMLayerItem = bm.loops.layers.uv.verify()
        self.is_edit_bm: bool = is_edit_bm
        self.update_tag: bool = True
        self.sync: bool = utils.sync()
        # self.islands_calc_type
        # self.islands_calc_subtype
        self.value: float | int | utils.NoInit = utils.NoInit()  # value for different purposes
        self.aspect: float = 1.0

    def update(self, force=False):
        if not self.update_tag:
            return False
        if self.is_edit_bm:
            bmesh.update_edit_mesh(self.obj.data, loop_triangles=force, destructive=force)
        else:
            self.bm.to_mesh(self.obj.data)
        return True

    def fake_umesh(self, isl):
        """Need for calculate sub islands"""
        fake = UMesh(self.bm, self.obj, self.is_edit_bm)
        fake.update_tag = self.update_tag
        fake.sync = self.sync
        fake.value = self.value
        fake.aspect = self.aspect
        fake.bm = FakeBMesh(isl)
        return fake

    def free(self, force=False):
        if force or not self.is_edit_bm:
            self.bm.free()

    def mesh_to_bmesh(self):
        bm = bmesh.from_edit_mesh(self.obj.data)
        self.bm = bm
        self.uv = bm.loops.layers.uv.verify()

    def ensure(self, face=True, edge=False, vert=False, force=False):
        if self.is_edit_bm or not force:
            return
        if face:
            self.bm.faces.ensure_lookup_table()
        if edge:
            self.bm.edges.ensure_lookup_table()
        if vert:
            self.bm.verts.ensure_lookup_table()

    def check_uniform_scale(self, report=None, threshold=0.01) -> 'Vector | None':
        _, _, scale = self.obj.matrix_world.decompose()
        if not utils.umath.vec_isclose_to_uniform(scale, threshold):
            if report:
                report({'WARNING'}, f"The '{self.obj.name}' hasn't applied scale: X={scale.x:.4f}, Y={scale.y:.4f}, Z={scale.z:.4f}")
            return scale
        return None

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
    def has_full_selected_uv_faces(self) -> bool:
        if self.sync:
            if self.is_full_face_selected:
                return True
            elif self.is_full_face_deselected:
                return False
            else:
                return all(f.select for f in self.bm.faces)

        if not self.total_face_sel:
            return False

        uv = self.uv
        if bpy.context.tool_settings.uv_select_mode == 'EDGE':
            if self.is_full_face_selected:
                return all(all(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)
            return all(all(crn[uv].select_edge for crn in f.loops) and f.select for f in self.bm.faces)
        else:
            if self.is_full_face_selected:
                return all(all(crn[uv].select for crn in f.loops) for f in self.bm.faces)
            return all(all(crn[uv].select for crn in f.loops) and f.select for f in self.bm.faces)

    def has_selected_uv_faces(self) -> bool:
        if self.sync:
            return bool(self.total_face_sel)
        if not self.total_face_sel:
            return False
        uv = self.uv
        if self.is_full_face_selected:
            if bpy.context.tool_settings.uv_select_mode == 'EDGE':
                return any(all(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)
            return any(all(crn[uv].select for crn in f.loops) for f in self.bm.faces)
        else:
            if bpy.context.tool_settings.uv_select_mode == 'EDGE':
                return any(all(crn[uv].select_edge for crn in f.loops) and f.select for f in self.bm.faces)
            return any(all(crn[uv].select for crn in f.loops) and f.select for f in self.bm.faces)

    def has_visible_uv_faces(self) -> bool:
        if self.total_face_sel:
            return True
        if self.sync:
            return any(not f.hide for f in self.bm.faces)
        return False

    def has_selected_uv_edges(self) -> bool:
        """Warning: Edges might be without corners in sync mode"""
        if self.sync:
            return bool(self.total_edge_sel)
        if not self.total_face_sel:
            return False
        uv = self.uv
        if self.is_full_face_selected:
            return any(any(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)
        return any(f.select and any(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)

    def has_selected_uv_verts(self) -> bool:
        """Warning: Edges might be without corners in sync mode"""
        if self.sync:
            return bool(self.total_vert_sel)
        if not self.total_face_sel:
            return False
        uv = self.uv
        if self.is_full_face_selected:
            return any(any(crn[uv].select for crn in f.loops) for f in self.bm.faces)
        return any(f.select and any(crn[uv].select for crn in f.loops) for f in self.bm.faces)

    @property
    def has_any_selected_crn_non_sync(self):
        if PyBMesh.is_full_face_deselected(self.bm):
            return False

        uv = self.uv
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

        uv = self.uv
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

        uv = self.uv
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
        return pi

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

    def tag_selected_corners(self, both=False, edges_tag=True):
        corners = (_crn for f in self.bm.faces for _crn in f.loops)
        if self.sync:
            if self.is_full_face_selected:
                for crn in corners:
                    crn.tag = True
            else:
                if utils.other.get_select_mode_mesh() == 'FACE':
                    if self.is_full_face_deselected:
                        for crn in corners:
                            crn.tag = False
                        return

                    for f in self.bm.faces:
                        state = f.select
                        for crn in f.loops:
                            crn.tag = state
                else:
                    if edges_tag:  # TODO: Need for loop groups???
                        for f in self.bm.faces:
                            if f.hide:
                                for crn in f.loops:
                                    crn.tag = False
                            else:
                                for crn in f.loops:
                                    crn.tag = crn.edge.select
                    else:
                        for f in self.bm.faces:
                            if f.hide:
                                for crn in f.loops:
                                    crn.tag = False
                            else:
                                for crn in f.loops:
                                    crn.tag = crn.vert.select
        else:
            if self.is_full_face_deselected:
                for crn in corners:
                    crn.tag = False
            else:
                uv = self.uv
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
                    f.tag = f.select
        else:
            if self.is_full_face_deselected:
                self.set_face_tag(False)
            else:
                uv = self.uv
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

    def tag_visible_faces(self):
        if self.sync:
            if self.is_full_face_selected:
                self.set_face_tag()
            else:
                for f in self.bm.faces:
                    f.tag = not f.hide
        else:
            if self.is_full_face_selected:
                self.set_face_tag(True)
            elif self.is_full_face_deselected:
                self.set_face_tag(False)
            else:
                for f in self.bm.faces:
                    f.tag = f.select

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
            if f.select:  # TODO: Check that strange behavior
                for crn in f.loops:
                    crn.tag = state

    def set_corners_tag(self, state=True):
        for f in self.bm.faces:
            for crn in f.loops:
                crn.tag = state

    def set_face_tag(self, state=True):
        for f in self.bm.faces:
            f.tag = state

    def calc_selected_faces(self) -> list[bmesh.types.BMFace] or bmesh.types.BMFaceSeq:
        if self.is_full_face_deselected:
            return []

        if self.is_full_face_selected:
            return self.bm.faces
        return [f for f in self.bm.faces if f.select]

    def mark_seam_tagged_faces(self, additional=False):
        uv = self.uv
        if self.sync:
            for f in self.bm.faces:
                if not f.tag:
                    continue
                for crn in f.loops:
                    shared_crn_ = crn.link_loop_radial_prev
                    if crn == shared_crn_ or not shared_crn_.face.tag:
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == shared_crn_.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn_[uv].uv)
                    if additional:
                        crn.edge.seam |= seam
                    else:
                        crn.edge.seam = seam
        else:
            for f in self.bm.faces:
                for crn in f.loops:
                    shared_crn_ = crn.link_loop_radial_prev
                    if crn == shared_crn_ or not shared_crn_.face.tag:
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == shared_crn_.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn_[uv].uv)
                    if additional:
                        crn.edge.seam |= seam
                    else:
                        crn.edge.seam = seam

    def __hash__(self):
        return hash(self.bm)

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
        self.sync: bool = utils.sync()
        self.elem_mode: typing.Literal['VERTEX', 'EDGE', 'FACE', 'ISLAND'] = \
            utils.get_select_mode_mesh() if self.sync else utils.get_select_mode_uv()
        self.is_edit_mode = bpy.context.mode == 'EDIT_MESH'

    def report(self, info_type={'INFO'}, info="No uv for manipulate"):  # noqa
        if self.report_obj is None:
            print(info_type, info)
            return
        self.report_obj(info_type, info)

    def cancel_with_report(self, info_type: set[str]={'INFO'}, info: str ="No uv for manipulate"): # noqa #pylint: disable=dangerous-default-value
        self._cancel = True
        self.report(info_type, info)
        return {'CANCELLED'}

    def update(self, force=False, info_type={'INFO'}, info="No uv for manipulate"):  # noqa #pylint: disable=dangerous-default-value
        if self._cancel is True:
            return {'CANCELLED'}
        if sum(umesh.update(force=force) for umesh in self.umeshes):
            return {'FINISHED'}
        if info:
            self.report(info_type, info)
        return {'CANCELLED'}

    @property
    def update_tag(self):
        return any(umesh.update_tag for umesh in self)

    @update_tag.setter
    def update_tag(self, value):
        for umesh in self:
            umesh.update_tag = value

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
        for obj in bpy.context.selected_objects[:]:
            obj.select_set(False)

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

    @staticmethod
    def loop_for_object_mode_processing(without_selection=True):
        assert bpy.context.mode == 'EDIT_MESH'
        active = bpy.context.active_object
        selected_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        if without_selection:
            for obj in selected_objects:
                yield obj
        else:
            for obj in reversed(bpy.context.selected_objects):
                obj.select_set(False)

            for obj in selected_objects:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                yield obj
                obj.select_set(False)

            for obj in selected_objects:
                obj.select_set(True)
        if not without_selection:
            bpy.context.view_layer.objects.active = active
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    def set_sync(self, state=True):
        for umesh in self:
            umesh.sync = state
        self.sync = state

    def free(self, force=False):
        """self.umeshes save refs in init in OT classes, so it's necessary to free memory"""
        for umesh in self:
            umesh.free(force)

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
                bmeshes.append(UMesh(bm, obj[0], False))

        return cls(bmeshes)

    def _sel_ob_with_uv(self):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.uv_layers and obj.data.polygons:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.uv_layers and obj.data.polygons:
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, objs[0], False))
        self.umeshes = bmeshes

    @classmethod
    def unselected_with_uv(cls):
        visible_objects = []
        if (area := bpy.context.area).type == 'VIEW_3D' and not area.spaces.active.local_view:
            for obj in bpy.context.view_layer.objects:
                if (not obj.select_get()) and obj.visible_get() and (obj.type == 'MESH') and obj.data.polygons and obj.data.uv_layers:
                    visible_objects.append(obj)
        else:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            spaces = (area.spaces.active for area in utils.get_areas_by_type('VIEW_3D'))
            for obj in bpy.context.view_layer.objects:
                if (not obj.select_get()) and (obj.type == 'MESH') and obj.data.polygons and obj.data.uv_layers:
                    if spaces:
                        if any(obj.evaluated_get(depsgraph).visible_in_viewport_get(space) for space in spaces):
                            visible_objects.append(obj)
                    else:
                        visible_objects.append(obj)

        data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

        for obj in visible_objects:
            data_and_objects[obj.data].append(obj)

        bmeshes = []
        for data, obj in data_and_objects.items():
            bm = bmesh.new()
            bm.from_mesh(data)
            obj.sort(key=lambda a: a.name)
            bmeshes.append(UMesh(bm, obj[0], False))
        return cls(bmeshes)

    @classmethod
    def calc_all_objects(cls):
        bmeshes = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if obj.mode == 'EDIT':
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
                else:
                    bm = bmesh.new()
                    bm.from_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj, False))
        return cls(bmeshes)

    @classmethod
    def calc(cls, report=None):
        # Get umeshes without uv
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.polygons:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.polygons:
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                objs.sort(key=lambda a: a.name)
                bmeshes.append(UMesh(bm, objs[0], False))
        return cls(bmeshes, report=report)

    @classmethod
    def calc_any_unique(cls, report=None):
        # Get unique umeshes without uv
        umeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(obj.data)
                    umeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH':
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                objs.sort(key=lambda a: a.name)
                umeshes.append(UMesh(bm, objs[0], False))
        return cls(umeshes, report=report)

    def filter_selected_faces(self):
        for umesh in reversed(self.umeshes):
            if umesh.is_full_face_deselected:
                self.umeshes.remove(umesh)

    def filter_with_faces(self):
        for umesh in reversed(self.umeshes):
            if not umesh.bm.faces:
                self.umeshes.remove(umesh)

    @property
    def has_selected_uv_faces(self):
        if self.sync:
            return any(umesh.total_face_sel for umesh in self)
        else:
            for umesh in self:
                uv = umesh.uv
                if umesh.total_face_sel:
                    for f in umesh.bm.faces:
                        if all(crn[uv].select for crn in f.loops) and f.select:
                            return True
        return False

    def filtered_by_selected_and_visible_uv_faces(self) -> tuple['UMeshes', 'UMeshes']:
        """Warning: if bmesh has selected faces, non-selected might be without visible faces"""
        selected = []
        visible = []
        for umesh in self:
            if umesh.has_selected_uv_faces():
                selected.append(umesh)
            else:
                visible.append(umesh)
        if not selected:
            for umesh2 in reversed(visible):
                if not umesh2.has_visible_uv_faces():
                    visible.remove(umesh2)

        import copy
        u1 = copy.copy(self)
        u2 = copy.copy(self)
        u1.umeshes = selected
        u2.umeshes = visible
        return u1, u2

    def filtered_by_selected_and_visible_uv_edges(self) -> tuple['UMeshes', 'UMeshes']:
        selected = []
        visible = []
        for umesh in self:
            if umesh.has_selected_uv_edges():
                selected.append(umesh)
            else:
                visible.append(umesh)
        if not selected:
            for umesh2 in reversed(visible):
                if not umesh2.has_visible_uv_faces():
                    visible.remove(umesh2)

        import copy
        u1 = copy.copy(self)
        u2 = copy.copy(self)
        u1.umeshes = selected
        u2.umeshes = visible
        return u1, u2

    def filtered_by_full_selected_and_visible_uv_faces(self) -> tuple['UMeshes', 'UMeshes']:
        """Filter full selected and visible with not full selected"""
        selected = []
        visible = []
        for umesh in self:
            if umesh.has_full_selected_uv_faces:
                selected.append(umesh)
            else:
                if umesh.has_visible_uv_faces():
                    visible.append(umesh)

        import copy
        u1 = copy.copy(self)
        u2 = copy.copy(self)
        u1.umeshes = selected
        u2.umeshes = visible
        return u1, u2

    @property
    def has_update_mesh(self):
        return any(umesh.update_tag for umesh in self)

    def __iter__(self) -> typing.Iterator[UMesh]:
        return iter(self.umeshes)

    def __getitem__(self, item):
        return self.umeshes[item]

    def __len__(self):
        return len(self.umeshes)

    def __bool__(self):
        return bool(self.umeshes)

    def __str__(self):
        return f"UMeshes Count = {len(self.umeshes)}"

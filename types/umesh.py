# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing  # noqa

import bpy
import copy
import bmesh
import mathutils

from collections import defaultdict
from math import pi

from bmesh.types import BMFace, BMEdge, BMLoop

from .. import utils
from ..types import PyBMesh#, AdvIsland

class FakeBMesh:
    def __init__(self, isl):
        self.faces = isl

class UMesh:
    def __init__(self, bm, obj, is_edit_bm=True, verify_uv=True):
        self.bm: bmesh.types.BMesh | FakeBMesh = bm
        self.obj: bpy.types.Object = obj
        self.elem_mode = utils.NoInit()
        # TODO: Remove Vector from annotation (pycharm moment)
        self.uv: bmesh.types.BMLayerItem | mathutils.Vector = bm.loops.layers.uv.verify() if verify_uv else None
        self.is_edit_bm: bool = is_edit_bm
        self.update_tag: bool = True
        self.sync: bool = utils.sync()
        # self.islands_calc_type
        # self.islands_calc_subtype
        self.value: float | int | utils.NoInit = utils.NoInit()  # value for different purposes
        self.other = utils.NoInit()
        self.aspect: float = 1.0
        self.sequence: list[BMFace | BMEdge | BMLoop] | list['AdvIsland'] = []  # noqa

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

    def ensure(self, face=True, edge=False, vert=False):
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
                report({'WARNING'}, f"The {self.obj.name!r} hasn't applied scale: X={scale.x:.4f}, Y={scale.y:.4f}, Z={scale.z:.4f}")
            return scale
        return None

    def check_faces_exist(self, report=None):
        if not self.bm.faces:
            if report:
                report({'WARNING'}, f"Object {self.obj.name} has no faces")
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

    def has_full_selected_uv_faces(self) -> bool:
        if self.sync:
            if self.is_full_face_selected:
                return True
            elif self.is_full_face_deselected:
                return False
            else:
                return all(f.select for f in self.bm.faces if not f.hide)

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

    def has_partial_selected_uv_faces(self):
        if self.sync:
            if self.is_full_face_selected or self.is_full_face_deselected:
                return False
        else:
            if self.is_full_face_deselected:
                return False

        faces = utils.calc_visible_uv_faces(self)
        face_select_get = utils.face_select_get_func(self)
        return not utils.all_equal(faces, face_select_get)

    def has_visible_uv_faces(self) -> bool:
        if self.total_face_sel:
            return True
        if self.sync:
            return any(not f.hide for f in self.bm.faces)
        return False

    def has_selected_uv_edges(self) -> bool:
        if self.sync:
            if not self.total_edge_sel:
                return False
            elif self.total_face_sel:
                return True
            else:
                for f in self.bm.faces:
                    if not f.hide:
                        for e in f.edges:
                            if e.select:
                                return True
                return False
        if not self.total_face_sel:
            return False
        uv = self.uv
        if self.is_full_face_selected:
            return any(any(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)
        return any(f.select and any(crn[uv].select_edge for crn in f.loops) for f in self.bm.faces)

    def has_partial_selected_uv_edges(self):
        if self.sync:
            if self.is_full_face_selected or self.is_full_edge_deselected:
                return False
        else:
            if self.is_full_face_deselected:
                return False

        corners = utils.calc_visible_uv_corners(self)
        edge_select_get = utils.edge_select_get_func(self)
        return not utils.all_equal(corners, edge_select_get)

    def has_partial_selected_3d_edges(self):
        assert self.sync
        if self.is_full_edge_selected or self.is_full_edge_deselected:
            return False
        return not utils.all_equal((e.select for e in self.bm.edges if not e.hide))

    def has_selected_uv_verts(self) -> bool:
        if self.sync:
            if not self.total_vert_sel:
                return False
            elif self.total_face_sel:
                return True
            else:
                for f in self.bm.faces:
                    if not f.hide:
                        for v in f.verts:
                            if v.select:
                                return True
                return False
        if not self.total_face_sel:
            return False
        uv = self.uv
        if self.is_full_face_selected:
            return any(any(crn[uv].select for crn in f.loops) for f in self.bm.faces)
        return any(f.select and any(crn[uv].select for crn in f.loops) for f in self.bm.faces)

    def has_partial_selected_uv_verts(self):
        if self.sync:
            if self.is_full_face_selected or self.is_full_vert_deselected:
                return False
        else:
            if self.is_full_face_deselected:
                return False

        corners = utils.calc_visible_uv_corners(self)
        vert_select_get = utils.vert_select_get_func(self)
        return not utils.all_equal(corners, vert_select_get)

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

    @property
    def has_uv(self):
        return bool(self.bm.loops.layers.uv)

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
                if self.elem_mode == 'FACE':
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

    def verify_uv(self):
        layers_uv = self.bm.loops.layers.uv
        if not layers_uv:
            self.uv = layers_uv.new('UVMap')
        else:
            self.uv = self.bm.loops.layers.uv.verify()

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
        self._elem_mode: typing.Literal['VERT', 'EDGE', 'FACE', 'ISLAND'] = self._elem_mode_init()
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

    @property
    def elem_mode(self):
        return self._elem_mode

    @elem_mode.setter
    def elem_mode(self, mode: typing.Literal['VERT', 'EDGE', 'FACE', 'ISLAND']):
        if self._elem_mode != mode:
            self._elem_mode = mode
            if self.sync:
                utils.set_select_mode_mesh(mode)  # noqa
                for umesh in self:
                    umesh.bm.select_mode = {mode}
            else:
                utils.set_select_mode_uv(mode)
            for umesh in self.umeshes:
                umesh.elem_mode = mode

    def _elem_mode_init(self):
        mode = utils.get_select_mode_mesh() if self.sync else utils.get_select_mode_uv()
        for umesh in self.umeshes:
            umesh.elem_mode = mode
        return mode

    def silent_update(self):
        for umesh in self:
            umesh.update()

    def final(self):
        if self._cancel is True:
            return True
        return any(umesh.update_tag for umesh in self.umeshes)

    def ensure(self, face=True, edge=False, vert=False):
        for umesh in self.umeshes:
            umesh.ensure(face, edge, vert)

    def deselect_all_elem(self):
        for umesh in self:
            if self.sync:
                if self._elem_mode == 'FACE':
                    if umesh.is_full_face_deselected:
                        continue
                    umesh.update_tag = True
                    if umesh.is_full_face_selected:
                        for f in umesh.bm.faces:
                            f.select = False
                    else:
                        for f in umesh.bm.faces:
                            if f.select:
                                f.select = False
                else:
                    if umesh.is_full_vert_deselected:
                        continue
                    umesh.update_tag = True
                    if umesh.is_full_vert_selected:
                        for v in umesh.bm.verts:
                            v.select = False
                    else:
                        for v in umesh.bm.verts:
                            if v.select:
                                v.select = False
            else:
                if selected_corners := utils.calc_selected_uv_vert_corners(umesh):
                    uv = umesh.uv
                    umesh.update_tag = True
                    for crn in selected_corners:
                        crn_uv = crn[uv]
                        crn_uv.select = False
                        crn_uv.select_edge = False

    def verify_uv(self):
        for umesh in self:
            umesh.verify_uv()

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
        self.elem_mode = utils.get_select_mode_mesh() if state else utils.get_select_mode_uv()

    def active_to_first(self):
        active_obj = bpy.context.active_object
        for idx, umesh in enumerate(self.umeshes):
            if umesh.obj == active_obj:
                if idx != 0:
                    self.umeshes[0], self.umeshes[idx] = self.umeshes[idx], self.umeshes[0]
                return True
        return False

    def fix_context(self):
        """If umesh without polygons, then it is not in the list. Set the first umesh with polygons as active,
            so that the context of default operators (bpy.ops.uv) works correctly.
            But it doesn't help when the operator is called via keymap."""
        if self.umeshes:
            active_obj = bpy.context.active_object
            if any(True for umesh in self.umeshes if (act_umesh := umesh).obj == active_obj):
                if not act_umesh.total_corners:
                    bpy.context.view_layer.objects.active = self.umeshes[0].obj
            else:
                bpy.context.view_layer.objects.active = self.umeshes[0].obj

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
            data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

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
                if obj.type == 'MESH' and obj.data.uv_layers:
                    bm = bmesh.from_edit_mesh(obj.data)
                    if bm.faces:
                        bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

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

        data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

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
    def calc_all_objects(cls, verify_uv=True):
        bmeshes = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if obj.mode == 'EDIT':
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj, verify_uv=verify_uv))
                else:
                    bm = bmesh.new()
                    bm.from_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj, False, verify_uv=verify_uv))
        return cls(bmeshes)

    @classmethod
    def calc(cls, report=None, verify_uv=True):
        """ Get umeshes without uv but with faces"""
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(obj.data)
                    if bm.faces:
                        bmeshes.append(UMesh(bm, obj, verify_uv=verify_uv))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.polygons:
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                objs.sort(key=lambda a: a.name)
                bmeshes.append(UMesh(bm, objs[0], False, verify_uv))
        return cls(bmeshes, report=report)

    def calc_aspect_ratio(self, from_mesh):
        if from_mesh:
            for umesh in self:
                umesh.aspect = utils.get_aspect_ratio(umesh)
        else:
            for umesh in self:
                umesh.aspect = utils.get_aspect_ratio()

    @classmethod
    def calc_any_unique(cls, report=None, verify_uv=True):
        """ Get unique umeshes without uv and without faces"""
        umeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(obj.data)
                    umeshes.append(UMesh(bm, obj, verify_uv=verify_uv))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH':
                    data_and_objects[obj.data].append(obj)

            for data, objs in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                objs.sort(key=lambda a: a.name)
                umeshes.append(UMesh(bm, objs[0], False, verify_uv))
        return cls(umeshes, report=report)

    def filter_by_selected_mesh_verts(self):
        for umesh in reversed(self.umeshes):
            if umesh.is_full_vert_deselected:
                self.umeshes.remove(umesh)

    def filter_by_selected_mesh_edges(self):
        for umesh in reversed(self.umeshes):
            if umesh.is_full_edge_deselected:
                self.umeshes.remove(umesh)

    def filter_by_selected_mesh_faces(self):
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

    def filter_by_selected_uv_verts(self) -> None:
        selected = []
        for umesh in self:
            if umesh.has_selected_uv_verts():
                selected.append(umesh)
        self.umeshes = selected

    def filter_by_selected_uv_edges(self) -> None:
        selected = []
        for umesh in self:
            if umesh.has_selected_uv_edges():
                selected.append(umesh)
        self.umeshes = selected

    def filter_by_selected_uv_faces(self) -> None:
        selected = []
        for umesh in self:
            if umesh.has_selected_uv_faces():
                selected.append(umesh)
        self.umeshes = selected

    def filter_by_selected_uv_elem_by_mode(self) -> None:
        if self.elem_mode == 'VERT':
            self.filter_by_selected_uv_verts()
        elif self.elem_mode == 'EDGE':
            self.filter_by_selected_uv_edges()
        else:
            self.filter_by_selected_uv_faces()

    def filter_by_partial_selected_uv_elem_by_mode(self) -> None:
        for umesh in reversed(self):
            if self.elem_mode == 'VERT':
                if not umesh.has_partial_selected_uv_verts():
                    self.umeshes.remove(umesh)
            elif self.elem_mode == 'EDGE':
                if not umesh.has_partial_selected_uv_edges():
                    self.umeshes.remove(umesh)
            else:
                if not umesh.has_partial_selected_uv_faces():
                    self.umeshes.remove(umesh)

    def filter_by_partial_selected_uv_edges(self) -> None:
        for umesh in reversed(self):
            if not umesh.has_partial_selected_uv_edges():
                self.umeshes.remove(umesh)

    def filter_by_partial_selected_3d_edges(self) -> None:
        for umesh in reversed(self):
            if not umesh.has_partial_selected_3d_edges():
                self.umeshes.remove(umesh)

    def filter_by_partial_selected_uv_faces(self) -> None:
        for umesh in reversed(self):
            if not umesh.has_partial_selected_uv_faces():
                self.umeshes.remove(umesh)

    def filter_by_visible_uv_faces(self) -> None:
        for umesh in reversed(self):
            if not umesh.has_visible_uv_faces():
                self.umeshes.remove(umesh)

    def filtered_by_selected_and_visible_uv_verts(self) -> tuple['UMeshes', 'UMeshes']:
        selected = []
        visible = []
        for umesh in self:
            if umesh.has_selected_uv_verts():
                selected.append(umesh)
            else:
                visible.append(umesh)
        if not selected:
            for umesh2 in reversed(visible):
                if not umesh2.has_visible_uv_faces():
                    visible.remove(umesh2)

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

        u1 = copy.copy(self)
        u2 = copy.copy(self)
        u1.umeshes = selected
        u2.umeshes = visible
        return u1, u2

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

        u1 = copy.copy(self)
        u2 = copy.copy(self)
        u1.umeshes = selected
        u2.umeshes = visible
        return u1, u2

    def filtered_by_selected_uv_faces(self):
        selected = []
        unselect_or_invisible = []
        for umesh in self:
            if umesh.has_selected_uv_faces():
                selected.append(umesh)
            else:
                unselect_or_invisible.append(umesh)
        self.umeshes = selected
        other = copy.copy(self)
        other.umeshes = unselect_or_invisible
        return other

    def filtered_by_full_selected_and_visible_uv_faces(self) -> tuple['UMeshes', 'UMeshes']:
        """Filter full selected and visible with not full selected"""
        selected = []
        visible = []
        for umesh in self:
            if umesh.has_full_selected_uv_faces():
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

    def filtered_by_uv_exist(self):
        with_uv_map = []
        without_uv_map = []
        for umesh in self:
            if len(umesh.bm.loops.layers.uv):
                with_uv_map.append(umesh)
            else:
                without_uv_map.append(umesh)

        self.umeshes = with_uv_map
        missing = copy.copy(self)
        missing.umeshes = without_uv_map
        return missing

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

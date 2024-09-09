# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import bmesh
import math
import mathutils
import typing
import enum
import itertools

from mathutils import Vector, Matrix
from mathutils.geometry import intersect_tri_tri_2d as isect_tris_2d
from mathutils.geometry import area_tri

from bmesh.types import BMesh, BMFace, BMLoop, BMLayerItem

from .. import utils
from ..utils import umath, UMesh
from .btypes import PyBMesh
from . import btypes
from. import BBox


class eInfoSelectFaceIsland(enum.IntEnum):
    UNSELECTED = 0
    HALF_SELECTED = 1
    FULL_SELECTED = 2

class SaveTransform:
    def __init__(self, island: 'FaceIsland | AdvIsland'):
        self.island = island
        self.old_crn_pos: list[Vector | float] = []  # need for mix co
        self.rotate = True
        self.is_full_selected = False
        self.target_crn: BMLoop | None = None
        self.old_coords: list[Vector] = [Vector((0, 0)), Vector((0, 0))]
        uv = island.uv_layer

        corners, pinned_corners = self.calc_static_corners(island, uv)
        if corners or pinned_corners:
            corners_iter = itertools.chain(corners, pinned_corners)
            co = next(corners_iter)[uv].uv
            for crn_ in corners_iter:
                if co != crn_[uv].uv:
                    self.rotate = False
                    break
            if self.rotate:
                max_length = -1.0
                max_length_crn = None
                for crn_ in itertools.chain(corners, pinned_corners):
                    if max_length < (new_length := (crn_[uv].uv - crn_.link_loop_next[uv].uv).length_squared):
                        max_length = new_length
                        max_length_crn = crn_

                self.target_crn = max_length_crn  # TODO: Get neutral stretched corner
                self.old_coords = [max_length_crn[uv].uv.copy(), max_length_crn.link_loop_next[uv].uv.copy()]

        else:
            self.is_full_selected = True
            max_uv_area_face = island.calc_max_uv_area_face()
            max_length_crn = utils.calc_max_length_uv_crn(max_uv_area_face.loops, uv)
            max_length_crn[uv].pin_uv = True
            self.target_crn = max_length_crn
            self.old_coords = [max_length_crn[uv].uv.copy(), max_length_crn.link_loop_next[uv].uv.copy()]

        self.bbox = self.island.calc_bbox()

    @staticmethod
    def calc_static_corners(island, uv) -> tuple[list[BMLoop], list[BMLoop]]:
        corners = []
        pinned_corners = []
        if utils.sync():
            if bpy.context.tool_settings.mesh_select_mode[2]:  # FACES
                for f in island:
                    if f.select:
                        for crn in f.loops:
                            if crn[uv].pin_uv:
                                pinned_corners.append(crn)
                    else:
                        for crn in f.loops:
                            crn_uv = crn[uv]
                            if crn_uv.pin_uv:
                                pinned_corners.append(crn)
                            else:
                                # crn_uv.pin_uv = True
                                corners.append(crn)
            elif bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                for f in island:
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        if not crn.edge.select:
                            # crn_uv.pin_uv = True
                            corners.append(crn)
                        elif crn_uv.pin_uv:
                            pinned_corners.append(crn)

            else:  # VERTS
                for f in island:
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        if not crn.vert.select:
                            # crn_uv.pin_uv = True
                            corners.append(crn)
                        elif crn_uv.pin_uv:
                            pinned_corners.append(crn)
        else:
            for f in island:
                for crn in f.loops:
                    crn_uv = crn[uv]
                    if not crn_uv.select:
                        # crn_uv.pin_uv = True
                        corners.append(crn)
                    elif crn_uv.pin_uv:
                        pinned_corners.append(crn)

        return corners, pinned_corners

    def shift(self):
        """A small shift to keep the island from merging."""
        sign_x = hash(self.bbox.width) % 2 == 0
        sign_y = hash(self.bbox.width) % 2 == 0
        x = self.bbox.width * 0.005
        y = self.bbox.height * 0.005
        self.island.move(Vector((x if sign_x else -x, y if sign_y else -y)))

    def inplace(self):
        if not self.rotate:
            return
        uv = self.island.uv_layer

        crn_co = self.target_crn[uv].uv if self.target_crn else Vector((0.0, 0.0))
        crn_next_co = self.target_crn.link_loop_next[uv].uv if self.target_crn else Vector((0.0, 0.0))

        old_dir = self.old_coords[0] - self.old_coords[1]
        new_dir = crn_co - crn_next_co

        if self.bbox.max_length < 2e-05:  # Small and zero area island protection
            new_bbox = self.island.calc_bbox()
            pivot = new_bbox.center
            if new_bbox.max_length != 0:
                self.island.rotate(old_dir.angle_signed(new_dir, 0), pivot)
                scale = 0.15 / new_bbox.max_length
                self.island.scale(Vector((scale, scale)), pivot)  # TODO: Optimize when implement simple scale_with_set_position
            self.island.set_position(self.bbox.center, pivot)
        else:  # TODO: Fix large islands
            if angle := old_dir.angle_signed(new_dir, 0):
                self.island.rotate(-angle, pivot=self.target_crn[uv].uv)
            new_bbox = self.island.calc_bbox()

            if self.bbox.width > self.bbox.height:

                scale = self.bbox.width / new_bbox.width
                self.island.scale(Vector((scale, scale)), new_bbox.center)

                old_center = self.bbox.center
                new_center = new_bbox.center
            else:
                scale = self.bbox.height / new_bbox.height
                self.island.scale(Vector((scale, scale)), new_bbox.center)

                old_center = self.bbox.center
                new_center = new_bbox.center
            self.island.set_position(old_center, new_center)
        if self.is_full_selected:
            self.target_crn[uv].pin_uv = False

    def save_coords(self, axis, mix):
        if mix == 1:
            return
        uv = self.island.uv_layer
        if axis == 'X':
            self.old_crn_pos = [crn[uv].uv.x for f in self.island for crn in f.loops]
        elif axis == 'Y':
            self.old_crn_pos = [crn[uv].uv.y for f in self.island for crn in f.loops]
        else:
            self.old_crn_pos = [crn[uv].uv.copy() for f in self.island for crn in f.loops]

    def apply_saved_coords(self, axis, mix):
        if mix == 1:
            return
        uv = self.island.uv_layer
        corners = (crn[uv].uv for f in self.island for crn in f.loops)
        if mix == 0:
            if axis == 'X':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.x = old_co
            elif axis == 'Y':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.y = old_co
            else:
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.xy = old_co
        else:
            from bl_math import lerp
            if axis == 'X':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.x = lerp(old_co, crn_uv.x, mix)
            elif axis == 'Y':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.y = lerp(old_co, crn_uv.y, mix)
            else:
                from ..utils import vec_lerp
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv[:] = vec_lerp(old_co, crn_uv, mix)

class FaceIsland:
    def __init__(self, faces: list[BMFace], bm: BMesh, uv_layer: BMLayerItem):
        self.faces: list[BMFace] = faces
        self.bm: BMesh = bm
        self.uv_layer: BMLayerItem = uv_layer
        self.value: float | int = -1  # value for different purposes

    def move(self, delta: Vector) -> bool:
        if umath.vec_isclose_to_zero(delta):
            return False

        uv = self.uv_layer
        for face in self.faces:
            for crn in face.loops:
                crn[uv].uv += delta
        return True

    def set_position(self, to: Vector, _from: Vector = None):
        if _from is None:
            _from = self.calc_bbox().min
        return self.move(to - _from)

    def save_transform(self):
        return SaveTransform(self)

    def rotate(self, angle: float, pivot: Vector, aspect: float = 1.0) -> bool:
        """Rotate a list of faces by angle (in radians) around a pivot
        :param angle: Angle in radians
        :param pivot: Pivot
        :param aspect: Aspect Ratio = Width / Height
        """
        if math.isclose(angle, 0, abs_tol=0.0001):
            return False
        rot_matrix = Matrix.Rotation(angle, 2)
        if aspect != 1.0:
            rot_matrix[0][1] = aspect * rot_matrix[0][1]
            rot_matrix[1][0] = rot_matrix[1][0] / aspect

        uv = self.uv_layer
        diff = pivot-(pivot @ rot_matrix)
        for face in self.faces:
            for crn in face.loops:
                crn_uv = crn[uv]
                crn_uv.uv = crn_uv.uv @ rot_matrix + diff  # TODO: Find aspect ratio for Vector.rotate method
        return True

    def rotate_simple(self, angle: float, aspect: float = 1.0) -> bool:
        """Rotate a list of faces by angle (in radians) around a world center"""
        if math.isclose(angle, 0, abs_tol=0.0001):
            return False
        rot_matrix = Matrix.Rotation(-angle, 2)
        if aspect != 1.0:
            rot_matrix[0][1] = aspect * rot_matrix[0][1]
            rot_matrix[1][0] = rot_matrix[1][0] / aspect

        uv = self.uv_layer
        for face in self.faces:
            for crn in face.loops:
                crn_uv = crn[uv]
                crn_uv.uv = crn_uv.uv @ rot_matrix
        return True

    def scale(self, scale: Vector, pivot: Vector) -> bool:
        """Scale a list of faces by pivot"""
        if umath.vec_isclose_to_uniform(scale):
            return False
        diff = pivot - pivot * scale

        uv = self.uv_layer
        for face in self.faces:
            for crn in face.loops:
                crn_co = crn[uv].uv
                crn_co *= scale
                crn_co += diff
        return True

    def scale_simple(self, scale: Vector) -> bool:
        """Scale a list of faces by world center"""
        if umath.vec_isclose_to_uniform(scale):
            return False

        uv = self.uv_layer
        for face in self.faces:
            for crn in face.loops:
                crn[uv].uv *= scale
        return True

    def set_tag(self, tag=True):
        for f in self:
            f.tag = tag

    def set_corners_tag(self, tag=True):
        for f in self:
            for crn in f.loops:
                crn.tag = tag

    def set_selected_crn_edge_tag(self, umesh):
        if umesh.sync:
            for f in self:
                for crn in f.loops:
                    crn.tag = crn.edge.select
        else:
            uv = umesh.uv_layer
            for f in self:
                for crn in f.loops:
                    crn.tag = crn[uv].select_edge

    def iter_corners_by_tag(self):
        return (crn for f in self for crn in f.loops if crn.tag)

    def tag_selected_corner_verts_by_verts(self, umesh):
        corners = (_crn for f in self for _crn in f.loops)

        if umesh.sync:
            if umesh.is_full_vert_selected:
                for crn in corners:
                    crn.tag = True
            else:
                for crn in corners:
                    crn.tag = crn.vert.select
        else:
            uv = self.uv_layer
            for f in self.bm.faces:
                for crn in f.loops:
                    crn.tag = crn[uv].select

    def is_flipped(self) -> bool:
        for f in self.faces:
            area = 0.0
            uvs = [l[self.uv_layer].uv for l in f.loops]
            for i in range(len(uvs)):
                area += uvs[i - 1].cross(uvs[i])
            if area < 0:
                return True
        return False

    def is_full_flipped(self, partial=False) -> bool:
        counter = 0
        for f in self.faces:
            area = 0.0
            uvs = [l[self.uv_layer].uv for l in f.loops]
            for i in range(len(uvs)):
                area += uvs[i - 1].cross(uvs[i])
            if area < 0:
                counter += 1

        if partial:
            if counter != 0 and counter != len(self):
                return True
            return False
        return counter == len(self)

    def calc_bbox(self) -> BBox:
        return BBox.calc_bbox_uv(self.faces, self.uv_layer)

    def calc_convex_points(self):
        points = [l[self.uv_layer].uv for f in self.faces for l in f.loops]  # Warning: points referenced to uv
        return [points[i] for i in mathutils.geometry.convex_hull_2d(points)]

    def __select_force(self, state, sync):
        if sync:
            for face in self.faces:
                face.select = state
                for e in face.edges:
                    e.select = state
                for v in face.verts:
                    v.select = state
        else:
            for face in self.faces:
                for l in face.loops:
                    luv = l[self.uv_layer]
                    luv.select = state
                    luv.select_edge = state

    def _select_ex(self, state, sync, mode):
        if sync:
            if mode == 'FACE':
                for face in self.faces:
                    face.select = state
            elif mode == 'VERTEX':
                for face in self.faces:
                    for v in face.verts:
                        v.select = state
            else:
                for face in self.faces:
                    for e in face.edges:
                        e.select = state
        else:
            uv_layer = self.uv_layer
            if mode == 'VERTEX':
                for face in self.faces:
                    for crn in face.loops:
                        crn[uv_layer].select = state
            else:
                for face in self.faces:
                    for crn in face.loops:
                        crn_uv = crn[uv_layer]
                        crn_uv.select = state
                        crn_uv.select_edge = state

    @property
    def select(self):
        raise NotImplementedError()

    @select.setter
    def select(self, state: bool):
        sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync
        elem_mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()
        self._select_ex(state, sync, elem_mode)

    def select_set(self, mode, sync, force=False):
        if force:
            return self.__select_force(True, sync)
        self._select_ex(True, sync, mode)

    def deselect_set(self, mode, sync, force=False):
        if force:
            return self.__select_force(False, sync)
        self._select_ex(False, sync, mode)

    def __info_vertex_select(self) -> eInfoSelectFaceIsland:
        uv_layer = self.uv_layer
        loops = self[0].loops
        if not sum(l[uv_layer].select for l in loops) == len(loops):
            if any(l[uv_layer].select for face in self for l in face.loops):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for l in face.loops:
                iter_count += 1
                selected_count += l[uv_layer].select
        if selected_count == 0:
            return eInfoSelectFaceIsland.UNSELECTED
        elif selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def __info_crn_select(self) -> eInfoSelectFaceIsland:
        uv_layer = self.uv_layer
        loops = self[0].loops
        if not sum(l[uv_layer].select_edge for l in loops) == len(loops):
            if any(l[uv_layer].select_edge for face in self for l in face.loops):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for l in face.loops:
                iter_count += 1
                selected_count += l[uv_layer].select_edge
        if selected_count == 0:
            return eInfoSelectFaceIsland.UNSELECTED
        elif selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def __info_vertex_select_sync(self) -> eInfoSelectFaceIsland:
        verts: bmesh.types.BMVertSeq = self[0].verts
        if not sum(v.select for v in verts) == len(verts):
            if any(v.select for face in self for v in face.verts):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for v in face.verts:
                iter_count += 1
                selected_count += v.select

        if selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def __info_edge_select_sync(self) -> eInfoSelectFaceIsland:
        edges: bmesh.types.BMEdgeSeq = self[0].edges
        if not sum(e.select for e in edges) == len(edges):
            if any(e.select for face in self for e in face.edges):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for e in face.edges:
                iter_count += 1
                selected_count += e.select

        if selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def __info_face_select_sync(self) -> eInfoSelectFaceIsland:
        if not self[0].select:
            if any(f.select for f in self):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for f in self:
            iter_count += 1
            selected_count += f.select

        if selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def info_select(self, sync=None, mode=None) -> eInfoSelectFaceIsland:
        if sync is None:
            sync = bpy.context.scene.tool_settings.use_uv_select_sync
        if mode is None:
            mode = utils.get_select_mode_mesh() if sync else utils.get_select_mode_uv()
        if not sync:
            if mode == 'VERTEX':
                return self.__info_vertex_select()
            else:
                return self.__info_crn_select()
        else:
            if mode == 'VERTEX':
                return self.__info_vertex_select_sync()
            elif mode == 'EDGE':
                return self.__info_edge_select_sync()
            else:
                return self.__info_face_select_sync()

    def calc_materials(self, umesh: utils.UMesh) -> tuple[str]:
        indexes = set()
        for f in self.faces:
            indexes.add(f.material_index)
        indexes = list(indexes)
        indexes.sort()

        material: list[str] = []
        for idx in indexes:
            if idx < len(umesh.obj.material_slots):
                material.append(umesh.obj.material_slots[idx].name)
            else:
                material.append('')
        return tuple(material)

    def mark_seam(self, additional=False):
        uv = self.uv_layer
        if utils.sync():
            for f in self.faces:
                for crn in f.loops:
                    shared_crn = crn.link_loop_radial_prev
                    if crn == shared_crn and shared_crn.face.select:  # TODO: Test without shared_crn.face.select
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)
                    if additional:
                        crn.edge.seam |= seam
                    else:
                        crn.edge.seam = seam
        else:
            for f in self.faces:
                for crn in f.loops:
                    shared_crn = crn.link_loop_radial_prev
                    if crn == shared_crn and all(_crn[uv].select_edge for _crn in shared_crn.face.loops):  # TODO: Test without second compare
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)
                    if additional:
                        crn.edge.seam |= seam
                    else:
                        crn.edge.seam = seam

    def mark_seam_by_index(self, additional: bool = False):
        # assert (enum.INDEXING in self.tags)  # TODO: Uncomment after implement tags
        index = self.faces[0].index

        for f in self.faces:
            for crn in f.loops:
                shared_crn = crn.link_loop_radial_prev
                if crn == shared_crn:
                    crn.edge.seam = True
                    continue

                if additional:
                    crn.edge.seam |= shared_crn.face.index != index
                else:
                    crn.edge.seam = shared_crn.face.index != index

    def calc_max_uv_area_face(self):
        uv = self.uv_layer
        area = -1.0
        face = None
        for f in self.faces:
            if area < (area_ := utils.calc_face_area_uv(f, uv)):
                area = area_
                face = f
        return face

    def clear(self):
        self.faces = []
        self.bm = None
        self.uv_layer = None

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, idx) -> BMFace:
        return self.faces[idx]

    def __len__(self):
        return len(self.faces)

    def __bool__(self):
        return bool(self.faces)

    def __str__(self):
        return f'Faces count = {len(self.faces)}'

class AdvIslandInfo:
    def __init__(self):
        self.area_uv: float = -1.0
        self.area_3d: float = -1.0
        self.td: float | None = -1.0
        self.edge_length: float | None = -1.0
        self.scale: Vector | None = None
        self.materials: tuple[str] = tuple()

class AdvIsland(FaceIsland):
    def __init__(self, faces: list[BMFace], bm: BMesh, uv_layer: BMLayerItem):
        super().__init__(faces, bm, uv_layer)
        self.tris: list[tuple[BMLoop]] = []
        self.flat_coords = []
        self.convex_coords = []
        self._bbox: BBox | None = None
        self.tag = True
        self._select_state = None
        self.info: AdvIslandInfo | None = None

    def move(self, delta: Vector) -> bool:
        if self._bbox is not None:
            self._bbox.move(delta)
        return super().move(delta)

    def scale(self, scale: Vector, pivot: Vector) -> bool:
        if self._bbox is not None:
            self._bbox.scale(scale, pivot)
        return super().scale(scale, pivot)

    def rotate(self, angle: float, pivot: Vector, aspect: float = 1.0) -> bool:
        self._bbox = None  # TODO: Implement Rotate 90 degrees and aspect ration for bbox
        return super().rotate(angle, pivot, aspect)

    def set_position(self, to: Vector, _from: Vector = None):
        if _from is None:
            _from = self.bbox.min
        return self.move(to - _from)

    def calc_flat_coords(self):
        assert self.tris, 'Calculate tris'

        uv_layer = self.uv_layer
        flat_coords = self.flat_coords
        for t in self.tris:
            flat_coords.extend((t_loop[uv_layer].uv for t_loop in t))

    def is_overlap(self, other: 'AdvIsland'):
        assert (self.flat_coords and other.flat_coords), 'Calculate flat coordinates'
        if not self.bbox.is_isect(other.bbox):
            return False
        for i in range(0, len(self.flat_coords), 3):
            a0, a1, a2 = self.flat_coords[i], self.flat_coords[i + 1], self.flat_coords[i + 2]
            for j in range(0, len(other.flat_coords), 3):
                if isect_tris_2d(a0, a1, a2, other.flat_coords[j], other.flat_coords[j + 1], other.flat_coords[j + 2]):
                    return True
        return False

    def calc_bbox(self) -> BBox:
        if self.convex_coords:
            self._bbox = BBox.calc_bbox(self.convex_coords)
        elif self.flat_coords:
            self._bbox = BBox.calc_bbox(self.flat_coords)
        else:
            self._bbox = BBox.calc_bbox_uv(self.faces, self.uv_layer)
        return self._bbox

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self.calc_bbox()
        return self._bbox

    @property
    def select_state(self):
        if self._select_state is None:
            self._select_state = self.info_select()
        else:
            return self._select_state

    def calc_convex_points(self):
        if self.flat_coords:
            self.convex_coords = [self.flat_coords[i] for i in mathutils.geometry.convex_hull_2d(self.flat_coords)]
        else:
            self.convex_coords = super().calc_convex_points()
        return self.convex_coords

    def calc_area(self):
        area = 0.0
        for i in range(0, len(self.flat_coords), 3):
            area += area_tri(self.flat_coords[i], self.flat_coords[i + 1], self.flat_coords[i + 2])

        if self.info is None:
            self.info = AdvIslandInfo()
        self.info.area_uv = area

    def calc_selected_edge_length(self, selected=True):
        uv = self.uv_layer
        total_length = 0.0
        corners = (_crn for _f in self for _crn in _f.loops)
        if selected:
            if not utils.sync():
                for crn in corners:
                    uv_crn = crn[uv]
                    if uv_crn.select_edge:
                        total_length += (uv_crn.uv - crn.link_loop_next[uv].uv).length
            else:
                for crn in corners:
                    if crn.edge.select:
                        total_length += (crn[uv].uv - crn.link_loop_next[uv].uv).length
        else:
            for crn in corners:
                total_length += (crn[uv].uv - crn.link_loop_next[uv].uv).length

        if self.info is None:
            self.info = AdvIslandInfo()
        self.info.edge_length = total_length

    def calc_materials(self, umesh: utils.UMesh):
        materials = super().calc_materials(umesh)
        if self.info is None:
            self.info = AdvIslandInfo()

        self.info.materials = materials
        return materials

    def calc_sub_islands_all(self, angle: float):
        self.set_tag()
        islands = [AdvIsland(i, self.bm, self.uv_layer) for i in IslandsBase.calc_all_ex(self, self.uv_layer, angle)]
        return AdvIslands(islands, self.bm, self.uv_layer)

    def __str__(self):
        return f'Faces count = {len(self.faces)}, Tris Count = {len(self.tris)}'


class IslandsBase:
    @staticmethod
    def tag_filter_all(umesh: UMesh, tag=True):
        for face in umesh.bm.faces:
            face.tag = tag

    @staticmethod
    def tag_filter_selected(bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.is_full_face_selected(bm):
            if sync:
                for face in bm.faces:
                    face.tag = True
                return
            if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                for face in bm.faces:
                    face.tag = all(l[uv_layer].select for l in face.loops)
            else:
                for face in bm.faces:
                    face.tag = all(l[uv_layer].select_edge for l in face.loops)
            return

        if sync:
            for face in bm.faces:
                face.tag = face.select
            return
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            for face in bm.faces:
                face.tag = all(l[uv_layer].select for l in face.loops) and face.select
        else:
            for face in bm.faces:
                face.tag = all(l[uv_layer].select_edge for l in face.loops) and face.select

    @staticmethod
    def tag_filter_non_selected(umesh: UMesh):
        if umesh.sync:
            if umesh.is_full_face_deselected:
                for face in umesh.bm.faces:
                    face.tag = not face.hide
                return

            for face in umesh.bm.faces:
                face.tag = not (face.select or face.hide)
        else:
            uv = umesh.uv_layer
            if umesh.is_full_face_selected:
                if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                    for face in umesh.bm.faces:
                        face.tag = not all(l[uv].select for l in face.loops)
                else:
                    for face in umesh.bm.faces:
                        face.tag = not all(l[uv].select_edge for l in face.loops)
            else:
                if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                    for face in umesh.bm.faces:
                        face.tag = not all(l[uv].select for l in face.loops) and face.select
                else:
                    for face in umesh.bm.faces:
                        face.tag = not all(l[uv].select_edge for l in face.loops) and face.select

    @staticmethod
    def tag_filter_selected_quad(bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.is_full_face_selected(bm):
            if sync:
                for face in bm.faces:
                    face.tag = len(face.loops) == 4
                return
            if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                for face in bm.faces:
                    corners = face.loops
                    face.tag = all(l[uv_layer].select for l in corners) and len(corners) == 4
            else:
                for face in bm.faces:
                    corners = face.loops
                    face.tag = all(l[uv_layer].select_edge for l in corners) and len(corners) == 4
            return

        if sync:
            for face in bm.faces:
                face.tag = face.select and len(face.loops) == 4
            return
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            for face in bm.faces:
                corners = face.loops
                face.tag = all(l[uv_layer].select for l in corners) and face.select and len(corners) == 4
        else:
            for face in bm.faces:
                corners = face.loops
                face.tag = all(l[uv_layer].select_edge for l in corners) and face.select and len(corners) == 4

    @staticmethod
    def tag_filter_visible(bm: BMesh, sync: bool):
        if sync:
            for face in bm.faces:
                face.tag = not face.hide
        else:
            if PyBMesh.is_full_face_selected(bm):
                for face in bm.faces:
                    face.tag = True
            else:
                for face in bm.faces:
                    face.tag = face.select

    @staticmethod
    def island_filter_is_partial_face_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        if sync:
            select = (face.select for face in island)
        else:
            select = (l[uv_layer].select_edge for face in island for l in face.loops)
        first_check = next(select)
        return any(first_check is not i for i in select)

    @staticmethod
    def island_filter_is_all_face_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        if sync:
            return all(face.select for face in island)
        else:
            return all(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_face_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        if sync:
            return any(face.select for face in island)
        else:
            return any(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_vert_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        if sync:
            return any(v.select for face in island for v in face.verts)
        else:
            return any(l[uv_layer].select for face in island for l in face.loops)

    @staticmethod
    def island_filter_is_any_edge_selected(island: list[BMFace], umesh: UMesh) -> bool:
        if umesh.sync:
            return any(e.select for face in island for e in face.edges)
        else:
            uv = umesh.uv_layer
            return any(crn[uv].select_edge for face in island for crn in face.loops)

    @staticmethod
    def island_filter_is_all_corner_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        assert (sync is False)
        return all(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_corner_selected(island: list[BMFace], uv_layer: BMLayerItem, sync: bool) -> bool:
        assert (sync is False)
        return any(any(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def calc_iter_ex(bm, uv):
        island: list[BMFace] = []

        for face in bm.faces:
            if not face.tag:  # Skip unselected and appended faces
                continue
            face.tag = False  # Tag first element in island (don`t add again)

            parts_of_island = [face]  # Container collector of island elements
            temp = []  # Container for get elements from loop from parts_of_island

            while parts_of_island:  # Blank list == all faces of the island taken
                for f in parts_of_island:
                    for l in f.loops:  # Running through all the neighboring faces
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if l[uv].uv == shared_crn.link_loop_next[uv].uv and l.link_loop_next[uv].uv == shared_crn[uv].uv:
                            temp.append(ff)
                            ff.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_iter_non_manifold_ex(bm, uv):
        island: list[BMFace] = []

        for face in bm.faces:
            if not face.tag:  # Skip unselected and appended faces
                continue
            face.tag = False  # Tag first element in island (don`t add again)

            parts_of_island = [face]  # Container collector of island elements
            temp = []  # Container for get elements from loop from parts_of_island

            while parts_of_island:  # Blank list == all faces of the island taken
                for f in parts_of_island:
                    for l in f.loops:  # Running through all the neighboring faces
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if l[uv].uv == shared_crn.link_loop_next[uv].uv or l.link_loop_next[uv].uv == shared_crn[uv].uv:
                            temp.append(ff)
                            ff.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_with_markseam_iter_ex(bm, uv):
        island: list[BMFace] = []

        for face in bm.faces:
            if not face.tag:
                continue
            face.tag = False

            parts_of_island = [face]
            temp = []

            while parts_of_island:
                for f in parts_of_island:
                    for l in f.loops:
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue
                        if l[uv].uv == shared_crn.link_loop_next[uv].uv and l.link_loop_next[uv].uv == shared_crn[uv].uv:
                            temp.append(ff)
                            ff.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_with_markseam_material_iter_ex(bm, uv):
        island: list[BMFace] = []

        for face in bm.faces:
            if not face.tag:
                continue
            face.tag = False

            parts_of_island = [face]
            temp = []

            while parts_of_island:
                for f in parts_of_island:
                    for l in f.loops:
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue
                        if ff.material_index != f.material_index:  # Skip if other material
                            continue
                        if l[uv].uv == shared_crn.link_loop_next[uv].uv and l.link_loop_next[uv].uv == shared_crn[uv].uv:
                            temp.append(ff)
                            ff.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_all_ex(bm, uv, angle):
        island: list[BMFace] = []

        for face in bm.faces:
            if not face.tag:
                continue
            face.tag = False

            parts_of_island = [face]
            temp = []

            while parts_of_island:
                for f in parts_of_island:
                    for l in f.loops:
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if not l.edge.smooth:  # Skip by sharp
                            continue
                        if l.edge.calc_face_angle() >= angle:  # Skip by angle
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue
                        if ff.material_index != f.material_index:  # Skip if other material
                            continue
                        if l[uv].uv == shared_crn.link_loop_next[uv].uv and l.link_loop_next[uv].uv == shared_crn[uv].uv:
                            temp.append(ff)
                            ff.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

class Islands(IslandsBase):
    island_type = FaceIsland

    def __init__(self, islands, bm, uv_layer):
        self.islands: list[FaceIsland] = islands
        self.bm: BMesh = bm
        self.uv_layer: BMLayerItem = uv_layer
        self.value: float | int = -1  # value for different purposes

    @classmethod
    def calc_selected(cls, umesh: UMesh):
        if umesh.is_full_face_deselected:
            return cls([], None, None)
        cls.tag_filter_selected(umesh.bm, umesh.uv_layer, umesh.sync)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_non_selected(cls, umesh: UMesh):
        if umesh.sync and umesh.is_full_face_selected:
            return cls([], None, None)

        cls.tag_filter_non_selected(umesh)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_selected_with_mark_seam(cls, umesh: UMesh):
        if umesh.is_full_face_deselected:
            return cls([], None, None)
        cls.tag_filter_selected(umesh.bm, umesh.uv_layer, umesh.sync)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_with_markseam_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_selected_quad(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls([], None, None)
        cls.tag_filter_selected_quad(bm, uv_layer, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_full_selected(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls([], None, None)
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_all_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_partial_selected(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls([], None, None)
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_partial_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls([], None, None)
        cls.tag_filter_visible(bm, sync)
        if sync and PyBMesh.is_full_face_selected(bm):
            islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        else:
            islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_any_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended_with_mark_seam(cls, umesh: UMesh):
        if umesh.is_full_face_deselected:
            return cls([], None, None)
        cls.tag_filter_visible(umesh.bm, umesh.sync)
        if umesh.sync and umesh.is_full_face_deselected:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_with_markseam_iter_ex(umesh.bm, umesh.uv_layer)]
        else:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_with_markseam_iter_ex(umesh.bm, umesh.uv_layer)
                       if cls.island_filter_is_any_face_selected(i, umesh.uv_layer, umesh.sync)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_extended_any_elem(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool):
        if not sync:
            if btypes.PyBMesh.is_full_face_deselected(bm):
                return cls([], None, None)
        else:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if btypes.PyBMesh.is_full_face_deselected(bm):
                    return cls([], None, None)
            elif elem_mode == 'VERTEX':
                if btypes.PyBMesh.is_full_vert_deselected(bm):
                    return cls([], None, None)
            else:
                if btypes.PyBMesh.is_full_edge_deselected(bm):
                    return cls([], None, None)

        cls.tag_filter_visible(bm, sync)
        if sync and PyBMesh.is_full_face_selected(bm):
            islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        else:
            islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)
                       if cls.island_filter_is_any_vert_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended_any_elem_with_mark_seam(cls, umesh: UMesh):
        if umesh.sync:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls([], None, None)
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls([], None, None)
            else:
                if umesh.is_full_edge_deselected:
                    return cls([], None, None)
        else:
            if umesh.is_full_face_deselected:
                return cls([], None, None)

        cls.tag_filter_visible(umesh.bm, umesh.sync)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_with_markseam_iter_ex(umesh.bm, umesh.uv_layer)]
        else:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_with_markseam_iter_ex(umesh.bm, umesh.uv_layer)
                       if cls.island_filter_is_any_vert_selected(i, umesh.uv_layer, umesh.sync)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_extended_any_vert_non_manifold(cls, umesh: UMesh):
        """Calc any verts selected islands"""
        if not umesh.sync:
            if umesh.is_full_face_deselected:
                return cls([], None, None)
        else:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls([], None, None)
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls([], None, None)
            else:
                if umesh.is_full_edge_deselected:
                    return cls([], None, None)

        cls.tag_filter_visible(umesh.bm, umesh.sync)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_non_manifold_ex(umesh.bm, umesh.uv_layer)]
        else:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_non_manifold_ex(umesh.bm, umesh.uv_layer)
                       if cls.island_filter_is_any_vert_selected(i, umesh.uv_layer, umesh.sync)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_extended_any_edge_non_manifold(cls, umesh: UMesh):
        """Calc any edges selected islands"""
        if umesh.sync:
            if (elem_mode := utils.get_select_mode_mesh()) == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls([], None, None)
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls([], None, None)
            else:
                if umesh.is_full_edge_deselected:
                    return cls([], None, None)
        else:
            if umesh.is_full_face_deselected:
                return cls([], None, None)

        cls.tag_filter_visible(umesh.bm, umesh.sync)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_non_manifold_ex(umesh.bm, umesh.uv_layer)]
        else:
            islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_non_manifold_ex(umesh.bm, umesh.uv_layer)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_visible_non_manifold(cls, umesh: UMesh):
        cls.tag_filter_visible(umesh.bm, umesh.sync)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_visible(cls, bm: BMesh, uv_layer: BMLayerItem,  sync: bool):
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_visible_b(cls, umesh: UMesh):
        cls.tag_filter_visible(umesh.bm, umesh.sync)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    @classmethod
    def calc_non_selected_extended(cls, bm: BMesh, uv_layer: BMLayerItem,  sync: bool):
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if not cls.island_filter_is_any_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended_all(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool, angle: float):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls([], None, None)
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_all_ex(bm, uv_layer, angle) if cls.island_filter_is_any_vert_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_visible_all(cls, bm: BMesh, uv_layer: BMLayerItem,  sync: bool, angle: float):
        cls.tag_filter_visible(bm, sync)
        islands = [cls.island_type(i, bm, uv_layer) for i in cls.calc_all_ex(bm, uv_layer, angle)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended_or_visible(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool, *, extended) -> 'Islands':
        if extended:
            return cls.calc_extended(bm, uv_layer, sync)
        return cls.calc_visible(bm, uv_layer, sync)

    @classmethod
    def calc_any_extended_or_visible_non_manifold(cls, umesh: UMesh, *, extended) -> 'Islands':
        if extended:
            return cls.calc_extended_any_vert_non_manifold(umesh)
        return cls.calc_visible_non_manifold(umesh)

    @classmethod
    def calc_extended_or_visible_all(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool, angle: float, *, extended) -> 'Islands':
        """All == mark seam, angle, mark sharp, smooth_angle"""
        if extended:
            return cls.calc_extended_all(bm, uv_layer, sync, angle)
        return cls.calc_visible_all(bm, uv_layer, sync, angle)

    @classmethod
    def calc(cls, umesh: UMesh, *, selected) -> 'Islands':
        if selected:
            return cls.calc_selected(umesh)
        return cls.calc_visible_b(umesh)

    @classmethod
    def calc_with_hidden(cls, umesh: UMesh):
        cls.tag_filter_all(umesh)
        islands = [cls.island_type(i, umesh.bm, umesh.uv_layer) for i in cls.calc_iter_ex(umesh.bm, umesh.uv_layer)]
        return cls(islands, umesh.bm, umesh.uv_layer)

    def move(self, delta: Vector) -> bool:
        return bool(sum(island.move(delta) for island in self.islands))

    def set_position(self, to, _from):
        return bool(sum(island.set_position(to, _from) for island in self.islands))

    def scale_simple(self, scale: Vector):
        return bool(sum(island.scale_simple(scale) for island in self.islands))

    def scale(self, scale: Vector, pivot: Vector) -> bool:
        return bool(sum(island.scale(scale, pivot) for island in self.islands))

    def rotate(self, angle: float, pivot: Vector, aspect: float = 1.0) -> bool:
        return bool(sum(island.rotate(angle, pivot, aspect) for island in self.islands))

    def rotate_simple(self, angle: float, aspect: float = 1.0):
        return bool(sum(island.rotate_simple(angle, aspect) for island in self.islands))

    def calc_bbox(self) -> BBox:
        general_bbox = BBox()
        for island in self.islands:
            general_bbox.union(island.calc_bbox())
        return general_bbox

    def indexing(self, force=False):
        if force:
            if sum(len(isl) for isl in self.islands) != len(self.bm.faces):
                for f in self.bm.faces:
                    f.index = -1
            for idx, island in enumerate(self.islands):
                for face in island:
                    face.index = idx
            return

        for idx, island in enumerate(self.islands):
            for face in island:
                face.tag = True
                face.index = idx

    @staticmethod
    def weld_selected(isl_a: FaceIsland | AdvIsland, isl_b: FaceIsland | AdvIsland, selected=True) -> bool:
        """isl_a = target island"""
        assert(isl_a.bm == isl_b.bm)
        sync = utils.sync()
        uv = isl_a.uv_layer
        idx = isl_b[0].loops[0].face.index
        welded = False

        if selected:
            for f in isl_a:
                for crn in f.loops:
                    shared_crn = crn.link_loop_radial_prev
                    if shared_crn.face.index != idx:
                        continue
                    if sync:
                        if crn.edge.select:
                            utils.copy_pos_to_target(crn, uv, idx)
                            welded = True
                    else:
                        if crn[uv].select_edge or shared_crn[uv].select_edge:
                            utils.copy_pos_to_target(crn, uv, idx)
                            welded = True
        else:
            for f in isl_a:
                for crn in f.loops:
                    shared_crn = crn.link_loop_radial_prev
                    if shared_crn.face.index != idx:
                        continue
                    utils.copy_pos_to_target(crn, uv, idx)
                    welded = True

        if welded:
            new_idx = isl_a[0].loops[0].face.index
            for f in isl_b:
                f.index = new_idx
            isl_a.faces.extend(isl_b.faces)
            isl_b.clear()
        return welded

    def faces_iter(self):
        return (f for isl in self for f in isl)

    def __iter__(self) -> typing.Iterator['FaceIsland | AdvIsland']:
        return iter(self.islands)

    def __getitem__(self, idx) -> 'FaceIsland | AdvIsland':
        return self.islands[idx]

    def __bool__(self):
        return bool(self.islands)

    def __len__(self):
        return len(self.islands)

    def __str__(self):
        return f'Islands count = {len(self.islands)}'

class UnionIslands(Islands):
    def __init__(self, islands):
        super().__init__([], None, None)
        self.islands: list[AdvIsland | FaceIsland] = islands
        self.flat_coords = []
        self.convex_coords = []
        self._bbox = None

    def calc_bbox(self, force=True) -> BBox:
        self._bbox = BBox()
        if force:
            for island in self.islands:
                self._bbox.union(island.calc_bbox())
        else:
            for island in self.islands:
                self._bbox.union(island.bbox)
        return self._bbox

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self.calc_bbox()
        return self._bbox

    def calc_convex_points(self):
        if self[0].convex_coords:
            points = []
            for island in self:
                points.extend(island.convex_coords)
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords
        elif self[0].flat_coords:
            points = []
            for island in self:
                points.extend(island.flat_coords)
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords
        else:
            uv_layer = self.uv_layer
            points = [l[uv_layer].uv for island in self for f in island for l in f.loops]  # Warning: points referenced to uv
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords

    @staticmethod
    def calc_overlapped_island_groups(adv_islands: list[AdvIsland]) -> list['UnionIslands']:
        islands_group = []
        union_islands = []
        for island_first in adv_islands:
            if not island_first.tag:
                continue
            island_first.tag = False

            union_islands.append(island_first)
            compare_index = 0
            while True:
                if compare_index > len(union_islands) - 1:
                    islands_group.append(UnionIslands(union_islands))
                    union_islands = []
                    break

                for isl in adv_islands:
                    if not isl.tag:
                        continue
                    if union_islands[compare_index].is_overlap(isl):
                        isl.tag = False
                        union_islands.append(isl)
                compare_index += 1
        return islands_group

    def append(self, island):
        self.islands.append(island)

    def pop(self, island):
        self.islands.pop(island)


class AdvIslands(Islands):
    island_type = AdvIsland

    def __init__(self, islands: list[AdvIsland], bm, uv_layer):
        super().__init__([], bm, uv_layer)
        self.islands: list[AdvIsland] = islands

    def triangulate_islands(self):
        loop_triangles = self.bm.calc_loop_triangles()
        self.indexing()

        islands_of_tris: list[list[tuple[BMLoop]]] = [[] for _ in range(len(self.islands))]
        for tris in loop_triangles:
            face = tris[0].face
            if face.tag:
                islands_of_tris[face.index].append(tris)
        return islands_of_tris

    def calc_tris(self):
        if not self.islands:
            return False
        triangulated_islands = self.triangulate_islands()
        for isl, tria_isl in zip(self.islands, triangulated_islands):
            isl.tris = tria_isl
        return True

    def calc_flat_coords(self):
        for island in self.islands:
            island.calc_flat_coords()

    def calc_area(self):
        for isl in self:
            isl.calc_area()

    def calc_materials(self, umesh: utils.UMesh):
        for isl in self:
            isl.calc_materials(umesh)

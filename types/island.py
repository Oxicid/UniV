# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import collections

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
import numpy as np

from mathutils import Vector, Matrix
from mathutils.geometry import intersect_tri_tri_2d as isect_tris_2d
from mathutils.geometry import area_tri

from bmesh.types import BMFace, BMLoop

from .. import utils
from ..utils import umath
from . import umesh as _umesh
from . import BBox


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
        uv = island.umesh.uv

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
        if island.umesh.sync:
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

    def inplace(self, axis='BOTH'):
        if not self.rotate:
            return
        uv = self.island.umesh.uv

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

            if axis == 'BOTH':
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
            else:
                if axis == 'X':
                    scale = self.bbox.height / new_bbox.height

                    self.island.scale(Vector((scale, scale)), new_bbox.center)

                    old_center = self.bbox.center
                    new_center = new_bbox.center
                else:
                    scale = self.bbox.width / new_bbox.width

                    self.island.scale(Vector((scale, scale)), new_bbox.center)

                    old_center = self.bbox.center
                    new_center = new_bbox.center

                self.island.set_position(old_center, new_center)

        if self.is_full_selected:
            self.target_crn[uv].pin_uv = False

    def save_coords(self, axis, mix):
        if mix == 1:
            return
        uv = self.island.umesh.uv
        if axis == 'X':
            self.old_crn_pos = [crn[uv].uv.x for f in self.island for crn in f.loops]
        elif axis == 'Y':
            self.old_crn_pos = [crn[uv].uv.y for f in self.island for crn in f.loops]
        else:
            self.old_crn_pos = [crn[uv].uv.copy() for f in self.island for crn in f.loops]

    def apply_saved_coords(self, axis, mix):
        uv = self.island.umesh.uv
        corners = (crn[uv].uv for f in self.island for crn in f.loops)

        if axis == 'BOTH':
            if mix == 1:
                return
            if mix == 0:
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.xy = old_co
            else:
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv[:] = old_co.lerp(crn_uv, mix)
            return

        if mix == 1:
            if axis == 'X':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.x = old_co
            else:
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.y = old_co
        else:
            from bl_math import lerp
            if axis == 'X':
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.x = lerp(old_co, crn_uv.x, mix)
            else:
                for crn_uv, old_co in zip(corners, self.old_crn_pos):
                    crn_uv.y = lerp(old_co, crn_uv.y, mix)


class FaceIsland:
    def __init__(self, faces: list[BMFace], umesh: _umesh.UMesh):
        self.faces: list[BMFace] = faces
        self.umesh: _umesh.UMesh = umesh
        self.value: float | int | Vector = -1  # value for different purposes

    def move(self, delta: Vector) -> bool:
        if umath.vec_isclose_to_zero(delta):
            return False

        uv = self.umesh.uv
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

        uv = self.umesh.uv
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

        uv = self.umesh.uv
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

        uv = self.umesh.uv
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

        uv = self.umesh.uv
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
            uv = umesh.uv
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
            uv = self.umesh.uv
            for f in self.umesh.bm.faces:
                for crn in f.loops:
                    crn.tag = crn[uv].select

    def is_flipped(self) -> bool:
        uv = self.umesh.uv
        for f in self.faces:
            area = 0.0
            uvs = [crn[uv].uv for crn in f.loops]
            for i in range(len(uvs)):
                area += uvs[i - 1].cross(uvs[i])
            if area < 0:
                return True
        return False

    def is_full_flipped(self, partial=False) -> bool:
        counter = 0
        uv = self.umesh.uv
        for f in self.faces:
            area = 0.0
            uvs = [crn[uv].uv for crn in f.loops]
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
        return BBox.calc_bbox_uv(self.faces, self.umesh.uv)

    def calc_convex_points(self):
        uv = self.umesh.uv
        points = [crn[uv].uv for f in self.faces for crn in f.loops]  # Warning: points referenced to uv
        return [points[i] for i in mathutils.geometry.convex_hull_2d(points)]

    # def _select_ex(self, state, sync, mode):
    #     if sync:
    #         if mode == 'FACE':
    #             for face in self.faces:
    #                 face.select = state
    #         elif mode == 'VERTEX':
    #             for face in self.faces:
    #                 for v in face.verts:
    #                     v.select = state
    #         else:
    #             for face in self.faces:
    #                 for e in face.edges:
    #                     e.select = state
    #     else:
    #         uv = self.umesh.uv
    #         if mode == 'VERTEX':
    #             for face in self.faces:
    #                 for crn in face.loops:
    #                     crn[uv].select = state
    #         else:
    #             for face in self.faces:
    #                 for crn in face.loops:
    #                     crn_uv = crn[uv]
    #                     crn_uv.select = state
    #                     crn_uv.select_edge = state

    @property
    def select(self):
        raise NotImplementedError()

    @select.setter
    def select(self, state: bool):
        if self.umesh.sync:
            for face in self.faces:
                face.select = state
        else:
            uv = self.umesh.uv
            for face in self.faces:
                for crn in face.loops:
                    luv = crn[uv]
                    luv.select = state
                    luv.select_edge = state

    @property
    def select_all_elem(self):
        raise NotImplementedError()

    @select_all_elem.setter
    def select_all_elem(self, state: bool):
        if self.umesh.sync:
            for face in self.faces:
                face.select = state
                for v in face.verts:
                    v.select = state
                for e in face.edges:
                    e.select = state
        else:
            uv = self.umesh.uv
            for face in self.faces:
                for crn in face.loops:
                    luv = crn[uv]
                    luv.select = state
                    luv.select_edge = state

    @property
    def is_full_face_selected(self):
        if self.umesh.sync:
            return all(f.select for f in self)
        uv = self.umesh.uv
        return all(crn[uv].select for f in self for crn in f.loops)

    @property
    def is_full_face_deselected(self):
        if self.umesh.sync:
            return not any(f.select for f in self)
        uv = self.umesh.uv
        return not any(crn[uv].select for f in self for crn in f.loops)

    @property
    def is_full_vert_deselected(self):
        if self.umesh.sync:
            return not any(v.select for f in self for v in f.verts)
        uv = self.umesh.uv
        return not any(crn[uv].select for f in self for crn in f.loops)

    @property
    def is_full_edge_deselected(self):
        if self.umesh.sync:
            return not any(e.select for f in self for e in f.edges)
        uv = self.umesh.uv
        return not any(crn[uv].select_edge for f in self for crn in f.loops)

    def __info_vertex_select(self) -> eInfoSelectFaceIsland:
        uv = self.umesh.uv
        loops = self[0].loops
        if not sum(crn[uv].select for crn in loops) == len(loops):
            if any(crn[uv].select for face in self for crn in face.loops):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for crn in face.loops:
                iter_count += 1
                selected_count += crn[uv].select
        if selected_count == 0:
            return eInfoSelectFaceIsland.UNSELECTED
        elif selected_count == iter_count:
            return eInfoSelectFaceIsland.FULL_SELECTED
        else:
            return eInfoSelectFaceIsland.HALF_SELECTED

    def __info_crn_select(self) -> eInfoSelectFaceIsland:
        uv = self.umesh.uv
        corners = self[0].loops
        if not sum(crn[uv].select_edge for crn in corners) == len(corners):
            if any(crn[uv].select_edge for face in self for crn in face.loops):
                return eInfoSelectFaceIsland.HALF_SELECTED
            return eInfoSelectFaceIsland.UNSELECTED

        iter_count = 0
        selected_count = 0
        for face in self:
            for crn in face.loops:
                iter_count += 1
                selected_count += crn[uv].select_edge
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

    def calc_materials(self, umesh: _umesh.UMesh) -> tuple[str]:
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
        uv = self.umesh.uv
        if self.umesh.sync:
            for f in self.faces:
                for crn in f.loops:
                    pair = crn.link_loop_radial_prev
                    if crn == pair or pair.face.hide:
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == pair.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == pair[uv].uv)
                    if additional:
                        crn.edge.seam |= seam
                    else:
                        crn.edge.seam = seam
        else:
            for f in self.faces:
                for crn in f.loops:
                    pair = crn.link_loop_radial_prev
                    if crn == pair or not pair.face.select:
                        crn.edge.seam = True
                        continue
                    seam = not (crn[uv].uv == pair.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == pair[uv].uv)
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
        uv = self.umesh.uv
        area = -1.0
        face = None
        for f in self.faces:
            if area < (area_ := utils.calc_face_area_uv(f, uv)):
                area = area_
                face = f
        return face

    def clear(self):
        self.faces = []
        self.umesh = None

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, idx) -> BMFace:
        return self.faces[idx]

    def __len__(self):
        return len(self.faces)

    def __bool__(self):
        return bool(self.faces)

    def __str__(self):
        return f'Face Island. Faces count = {len(self.faces)}'

class AdvIslandInfo:
    def __init__(self):
        self.edge_length: float | None = -1.0
        self.scale: Vector | None = None

class AdvIsland(FaceIsland):
    def __init__(self, faces: list[BMFace] | tuple = (), umesh: _umesh.UMesh | None = None):
        super().__init__(faces, umesh)
        self.tris: list[tuple[BMLoop]] = []
        self.flat_unique_uv_coords: list[Vector] = []
        self.flat_coords: list[Vector] | list[tuple[Vector, Vector, Vector]] = []  # rename to flat_uv_coords
        self.flat_3d_coords: list[Vector] | list[tuple[Vector, Vector, Vector]] = []
        self.is_flat_3d_coords_scaled: bool = False
        self.weights: list[float] = []
        # self.custom_value_2: int | float | Vector = -1
        self.convex_coords = []
        self._bbox: BBox | None = None
        self.tag = True
        self._select_state = None
        self.area_3d: float = -1.0
        self.area_uv: float = -1.0
        self.info: AdvIslandInfo | None = None

    def move(self, delta: Vector) -> bool:
        if self._bbox is not None:
            self._bbox.move(delta)
        return super().move(delta)

    def scale(self, scale: Vector, pivot: Vector) -> bool:
        if self._bbox is not None:
            self._bbox.scale(scale, pivot)
        return super().scale(scale, pivot)

    def set_texel(self, texel: float, texture_size: float | int):
        """Warning: Need calc uv and 3d area"""
        assert self.area_3d != 1.0 and self.area_uv != 1.0, "Need calculate uv and 3d area"
        area_3d = math.sqrt(self.area_3d)
        area_uv = math.sqrt(self.area_uv) * texture_size
        if math.isclose(area_3d, 0.0, abs_tol=1e-6) or math.isclose(area_uv, 0.0, abs_tol=1e-6):
            return None
        scale = (texel / (area_uv / area_3d))
        return self.scale(Vector((scale, scale)), self.bbox.center)

    def rotate(self, angle: float, pivot: Vector, aspect: float = 1.0) -> bool:
        self._bbox = None  # TODO: Implement Rotate 90 degrees and aspect ratio for bbox
        return super().rotate(angle, pivot, aspect)

    def set_position(self, to: Vector, _from: Vector = None):
        if _from is None:
            _from = self.bbox.min
        return self.move(to - _from)

    def calc_flat_coords(self, save_triplet=False):
        assert self.tris, 'Calculate tris'

        uv = self.umesh.uv
        if save_triplet:
            self.flat_coords = [(t[0][uv].uv, t[1][uv].uv, t[2][uv].uv) for t in self.tris]
        else:
            extend = self.flat_coords.extend
            for t in self.tris:
                extend(t_crn[uv].uv for t_crn in t)

    def calc_flat_uv_coords(self, save_triplet_=False):
        self.calc_flat_coords(save_triplet_)

    def calc_flat_unique_uv_coords(self):
        uv = self.umesh.uv
        self.flat_unique_uv_coords = [crn[uv].uv for f in self for crn in f.loops]

    def calc_flat_3d_coords(self, save_triplet=False, scale_=None):
        assert self.tris, 'Calculate tris'
        self.is_flat_3d_coords_scaled = bool(scale_)
        if save_triplet:
            if scale_:
                self.flat_3d_coords = [(t[0].vert.co * scale_, t[1].vert.co * scale_, t[2].vert.co * scale_) for t in self.tris]
            else:
                self.flat_3d_coords = [(t[0].vert.co, t[1].vert.co, t[2].vert.co) for t in self.tris]
        else:
            extend = self.flat_3d_coords.extend
            if scale_:
                for t in self.tris:
                    extend([t_crn.vert.co * scale_ for t_crn in t])
            else:
                for t in self.tris:
                    extend([t_crn.vert.co for t_crn in t])

    def is_overlap(self, other: 'AdvIsland'):
        assert (self.flat_coords and other.flat_coords), 'Calculate flat coordinates'
        if not self.bbox.is_isect(other.bbox):
            return False
        if isinstance(self.flat_coords[0], tuple):
            for a0, a1, a2 in self.flat_coords:
                for b0, b1, b2 in other.flat_coords:
                    if isect_tris_2d(a0, a1, a2, b0, b1, b2):
                        return True
        else:
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
            if isinstance(self.flat_coords[0], tuple):
                self._bbox = BBox.calc_bbox(itertools.chain.from_iterable(self.flat_coords))
            else:
                self._bbox = BBox.calc_bbox(self.flat_coords)
        else:
            self._bbox = BBox.calc_bbox_uv(self.faces, self.umesh.uv)
        return self._bbox

    def has_flip_with_noflip(self):
        from ..utils import is_flipped_uv
        uv = self.umesh.uv
        flip_state = is_flipped_uv(self[-1], uv)
        for f in self:
            if flip_state != is_flipped_uv(f, uv):
                return True
        return False

    def calc_islands_by_flip_with_mark_seam(self) -> 'tuple[AdvIslands, AdvIslands]':
        """Warning: All 'bm.faces' must be untagged"""
        from ..utils import is_flipped_uv
        uv = self.umesh.uv
        no_flipped_faces = []
        flipped_faces = []
        for f in self:
            if is_flipped_uv(f, uv):
                flipped_faces.append(f)
            else:
                f.tag = True
                no_flipped_faces.append(f)

        assert no_flipped_faces and flipped_faces

        fake_umesh = self.umesh.fake_umesh(no_flipped_faces)
        no_flipped_islands = AdvIslands([AdvIsland(i, self.umesh) for i in Islands.calc_with_markseam_iter_ex(fake_umesh)], self.umesh)

        fake_umesh.bm.faces = flipped_faces
        for flipped_f in flipped_faces:
            flipped_f.tag = True
        flipped_islands = AdvIslands([AdvIsland(i, self.umesh) for i in Islands.calc_with_markseam_iter_ex(fake_umesh)], self.umesh)

        return no_flipped_islands, flipped_islands

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

    def calc_area_3d(self, scale=None, areas_to_weight=False):
        area = 0.0
        # self.weights = []
        if self.flat_3d_coords and scale:
            if self.is_flat_3d_coords_scaled:
                scale = None

        weight_append = self.weights.append
        it = self.flat_3d_coords if self.flat_3d_coords else ((crn_a.vert.co, crn_b.vert.co, crn_c.vert.co) for crn_a, crn_b, crn_c in self.tris)
        if areas_to_weight:
            assert self.tris, 'Calculate tris'
            if scale:
                for va, vb, vc in it:
                    ar = area_tri(va * scale, vb * scale, vc * scale)
                    weight_append(ar)
                    area += ar
            else:
                for va, vb, vc in it:
                    ar = area_tri(va, vb, vc)
                    weight_append(ar)
                    area += ar
        elif scale:
            if self.tris:
                for va, vb, vc in it:
                    area += area_tri(va * scale, vb * scale, vc * scale)
            else:
                # newell_cross
                for f in self:
                    n = Vector()
                    corners = f.loops
                    v_prev = corners[-1].vert.co * scale
                    for crn in corners:
                        v_curr = crn.vert.co * scale
                        # inplace optimization ~20%) - n += (v_prev.yzx - v_curr.yzx) * (v_prev.zxy + v_curr.zxy)
                        v_prev_yzx = v_prev.yzx
                        v_prev_zxy = v_prev.zxy

                        v_prev_yzx -= v_curr.yzx
                        v_prev_zxy += v_curr.zxy

                        v_prev_yzx *= v_prev_zxy
                        n += v_prev_yzx

                        v_prev = v_curr

                    area += n.length
                area *= 0.5
        else:
            for f in self:
                area += f.calc_area()

        self.area_3d = area
        return area

    def calc_area_uv(self):
        area = 0.0
        uv = self.umesh.uv
        if self.flat_coords:
            flat_coords = self.flat_coords
            if isinstance(flat_coords[0], tuple):
                for triplet in flat_coords:
                    area += area_tri(*triplet)
            else:
                for i in range(0, len(flat_coords), 3):
                    area += area_tri(flat_coords[i], flat_coords[i + 1], flat_coords[i + 2])
        elif self.tris:
            for crn_a, crn_b, crn_c in self.tris:
                area += area_tri(crn_a[uv].uv, crn_b[uv].uv, crn_c[uv].uv)
        else:
            from ..utils import calc_face_area_uv
            for f in self:
                area += calc_face_area_uv(f, uv)

        self.area_uv = area
        return area

    def calc_selected_edge_length(self, selected=True):
        uv = self.umesh.uv
        total_length = 0.0
        corners = (_crn for _f in self for _crn in _f.loops)
        if selected:
            if not self.umesh.sync:
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

    def calc_materials(self, umesh: _umesh.UMesh):
        materials = super().calc_materials(umesh)
        if self.info is None:
            self.info = AdvIslandInfo()

        self.info.materials = materials
        return materials

    def calc_sub_islands_all(self):
        self.set_tag()
        islands = [AdvIsland(i, self.umesh) for i in IslandsBase.calc_all_ex(self.umesh)]
        return AdvIslands(islands, self.umesh)

    def __str__(self):
        return f'Advanced Island. Faces count = {len(self.faces)}, Tris Count = {len(self.tris)}'


class IslandsBase:
    @staticmethod
    def tag_filter_all(umesh: _umesh.UMesh, tag=True):
        for face in umesh.bm.faces:
            face.tag = tag

    @staticmethod
    def tag_filter_selected(umesh: _umesh.UMesh):
        uv = umesh.uv
        if umesh.is_full_face_selected:
            if umesh.sync:
                for face in umesh.bm.faces:
                    face.tag = True
                return
            if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                for face in umesh.bm.faces:
                    face.tag = all(crn[uv].select for crn in face.loops)
            else:
                for face in umesh.bm.faces:
                    face.tag = all(crn[uv].select_edge for crn in face.loops)
            return

        if umesh.sync:
            for face in umesh.bm.faces:
                face.tag = face.select
            return
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            for face in umesh.bm.faces:
                face.tag = all(crn[uv].select for crn in face.loops) and face.select
        else:
            for face in umesh.bm.faces:
                face.tag = all(crn[uv].select_edge for crn in face.loops) and face.select

    @staticmethod
    def tag_filter_non_selected(umesh: _umesh.UMesh):
        if umesh.sync:
            if umesh.is_full_face_deselected:
                for face in umesh.bm.faces:
                    face.tag = not face.hide
                return

            for face in umesh.bm.faces:
                face.tag = not (face.select or face.hide)
        else:
            uv = umesh.uv
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
    def tag_filter_selected_quad(umesh: _umesh.UMesh):
        uv = umesh.uv
        if umesh.is_full_face_selected:
            if umesh.sync:
                for face in umesh.bm.faces:
                    face.tag = len(face.loops) == 4
                return
            if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
                for face in umesh.bm.faces:
                    corners = face.loops
                    face.tag = all(crn[uv].select for crn in corners) and len(corners) == 4
            else:
                for face in umesh.bm.faces:
                    corners = face.loops
                    face.tag = all(crn[uv].select_edge for crn in corners) and len(corners) == 4
            return

        if umesh.sync:
            for face in umesh.bm.faces:
                face.tag = face.select and len(face.loops) == 4
            return
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            for face in umesh.bm.faces:
                corners = face.loops
                face.tag = all(crn[uv].select for crn in corners) and face.select and len(corners) == 4
        else:
            for face in umesh.bm.faces:
                corners = face.loops
                face.tag = all(crn[uv].select_edge for crn in corners) and face.select and len(corners) == 4

    @staticmethod
    def tag_filter_visible(umesh: _umesh.UMesh):
        if umesh.sync:
            for face in umesh.bm.faces:
                face.tag = not face.hide
        else:
            if umesh.is_full_face_selected:
                for face in umesh.bm.faces:
                    face.tag = True
            else:
                for face in umesh.bm.faces:
                    face.tag = face.select

    @staticmethod
    def island_filter_is_partial_face_selected(island: list[BMFace], umesh: _umesh.UMesh) -> bool:
        if umesh.sync:
            select = (face.select for face in island)
        else:
            uv = umesh.uv
            select = (crn[uv].select_edge for face in island for crn in face.loops)
        first_check = next(select)
        return any(first_check is not i for i in select)

    @staticmethod
    def island_filter_is_all_face_selected(island: list[BMFace], umesh: _umesh.UMesh) -> bool:
        if umesh.sync:
            return all(face.select for face in island)
        else:
            uv = umesh.uv
            return all(all(crn[uv].select_edge for crn in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_face_selected(island: list[BMFace], umesh: _umesh.UMesh) -> bool:
        if umesh.sync:
            return any(face.select for face in island)
        else:
            uv = umesh.uv
            return any(all(crn[uv].select_edge for crn in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_vert_selected(island: list[BMFace], umesh: _umesh.UMesh) -> bool:
        if umesh.sync:
            return any(v.select for face in island for v in face.verts)
        else:
            uv = umesh.uv
            return any(crn[uv].select for face in island for crn in face.loops)

    @staticmethod
    def island_filter_is_any_edge_selected(island: list[BMFace], umesh: _umesh.UMesh) -> bool:
        if umesh.sync:
            return any(e.select for face in island for e in face.edges)
        else:
            uv = umesh.uv
            return any(crn[uv].select_edge for face in island for crn in face.loops)

    @staticmethod
    def calc_iter_ex(umesh: _umesh.UMesh):
        uv = umesh.uv
        island: list[BMFace] = []

        for face in umesh.bm.faces:
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
    def calc_iter_non_manifold_ex(umesh: _umesh.UMesh):
        uv = umesh.uv
        island: list[BMFace] = []

        for face in umesh.bm.faces:
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
    def calc_with_markseam_iter_ex(umesh: _umesh.UMesh):
        uv = umesh.uv
        island: list[BMFace] = []

        for face in umesh.bm.faces:
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
    def calc_with_markseam_material_iter_ex(umesh: _umesh.UMesh):
        uv = umesh.uv
        island: list[BMFace] = []

        for face in umesh.bm.faces:
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
    def calc_all_ex(umesh: _umesh.UMesh):
        uv = umesh.uv
        angle = umesh.value
        island: list[BMFace] = []

        for face in umesh.bm.faces:
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

    def __init__(self, islands=(), umesh: _umesh.UMesh | utils.NoInit = utils.NoInit()):
        self.islands: list[FaceIsland] | tuple = islands
        self.umesh: _umesh.UMesh | utils.NoInit = umesh
        self.value: float | int | Vector = -1  # value for different purposes

    @classmethod
    def calc_selected(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_selected(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_non_selected(cls, umesh: _umesh.UMesh):
        if umesh.sync and umesh.is_full_face_selected:
            return cls()

        cls.tag_filter_non_selected(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_visible_with_mark_seam(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_selected_with_mark_seam(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_selected(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_with_mark_seam(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_deselected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                       if cls.island_filter_is_any_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_selected_quad(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_selected_quad(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_full_selected(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh) if cls.island_filter_is_all_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_partial_selected(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh) if cls.island_filter_is_partial_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended(cls, umesh: _umesh.UMesh):
        if umesh.is_full_face_deselected:
            return cls()
        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh) if cls.island_filter_is_any_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_elem(cls, umesh: _umesh.UMesh):
        if not umesh.sync:
            if umesh.is_full_face_deselected:
                return cls()
        else:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls()
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)
                       if cls.island_filter_is_any_vert_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_elem_with_mark_seam(cls, umesh: _umesh.UMesh):
        if umesh.sync:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls()
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()
        else:
            if umesh.is_full_face_deselected:
                return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                       if cls.island_filter_is_any_vert_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_vert_non_manifold(cls, umesh: _umesh.UMesh):
        """Calc any verts selected islands"""
        if umesh.sync:
            elem_mode = utils.get_select_mode_mesh()
            if elem_mode == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls()
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()
        else:
            if umesh.is_full_face_deselected:
                return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_non_manifold_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_non_manifold_ex(umesh)
                       if cls.island_filter_is_any_vert_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge_non_manifold(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands"""
        if umesh.sync:
            if (elem_mode := utils.get_select_mode_mesh()) == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls()
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()
        else:
            if umesh.is_full_face_deselected:
                return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_non_manifold_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_non_manifold_ex(umesh)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands"""
        if umesh.sync:
            if (elem_mode := utils.get_select_mode_mesh()) == 'FACE':  # TODO: Add elem_mode field to UMesh
                if umesh.is_full_face_deselected:
                    return cls()
            elif elem_mode == 'VERTEX':
                if umesh.is_full_vert_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()
        else:
            if umesh.is_full_face_deselected:
                return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge_with_markseam(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands, with markseam"""
        if umesh.sync:
            if utils.get_select_mode_mesh() == 'FACE':
                if umesh.is_full_face_deselected:
                    return cls()
            else:
                if umesh.is_full_edge_deselected:
                    return cls()
        else:
            if umesh.is_full_face_deselected:
                return cls()

        cls.tag_filter_visible(umesh)
        if umesh.sync and umesh.is_full_face_selected:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        else:
            islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_visible_non_manifold(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_visible(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_non_selected_extended(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh) if not cls.island_filter_is_any_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_all(cls, umesh: _umesh.UMesh):
        if umesh.is_full_vert_deselected:
            return cls()
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_all_ex(umesh) if cls.island_filter_is_any_vert_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_visible_all(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_all_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_or_visible(cls, umesh: _umesh.UMesh, *, extended) -> 'Islands':
        if extended:
            return cls.calc_extended(umesh)
        return cls.calc_visible(umesh)

    @classmethod
    def calc_any_extended_or_visible_non_manifold(cls, umesh: _umesh.UMesh, *, extended) -> 'Islands':
        if extended:
            return cls.calc_extended_any_vert_non_manifold(umesh)
        return cls.calc_visible_non_manifold(umesh)

    @classmethod
    def calc_extended_or_visible_all(cls, umesh: _umesh.UMesh, *, extended) -> 'Islands':
        """All == mark seam, angle, mark sharp, smooth_angle"""
        if extended:
            return cls.calc_extended_all(umesh)
        return cls.calc_visible_all(umesh)

    @classmethod
    def calc(cls, umesh: _umesh.UMesh, *, selected) -> 'Islands':
        if selected:
            return cls.calc_selected(umesh)
        return cls.calc_visible(umesh)

    @classmethod
    def calc_with_hidden(cls, umesh: _umesh.UMesh):
        cls.tag_filter_all(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_with_hidden_with_mark_seam(cls, umesh: _umesh.UMesh):
        cls.tag_filter_all(umesh)
        islands = [cls.island_type(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        return cls(islands, umesh)

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

    def indexing(self, force=True):
        if force:
            if sum(len(isl) for isl in self.islands) != len(self.umesh.bm.faces):
                for f in self.umesh.bm.faces:
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
        assert(isl_a.umesh == isl_b.umesh)
        sync = isl_a.umesh.sync
        uv = isl_a.umesh.uv
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

class UnionIslandsController:
    def __init__(self, islands):
        self._islands = islands

    @property
    def update_tag(self):
        return any(isl.umesh.update_tag for isl in self._islands)

    @update_tag.setter
    def update_tag(self, value):
        for isl in self._islands:
            isl.umesh.update_tag = value

    @property
    def aspect(self):
        import numpy
        return numpy.mean([isl.umesh.aspect for isl in self._islands])

    @property
    def sync(self):
        return self._islands[0].umesh.sync

class UnionIslands(Islands):
    def __init__(self, islands):
        super().__init__([])
        self.islands: list[AdvIsland | FaceIsland] = islands
        self.umesh = UnionIslandsController(islands)
        # self.flat_coords = []
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

    @property
    def area_3d(self) -> float:
        return sum(isl.area_3d for isl in self)

    @property
    def area_uv(self) -> float:
        return sum(isl.area_uv for isl in self)

    def set_texel(self, texel: float, texture_size: float | int):
        """Warning: Need calc uv and 3d area"""
        assert self.islands[0].area_3d != 1.0 and self.islands[0].area_uv != 1.0, "Need calculate uv and 3d area"
        area_3d = math.sqrt(self.area_3d)
        area_uv = math.sqrt(self.area_uv) * texture_size
        if math.isclose(area_3d, 0.0, abs_tol=1e-6) or math.isclose(area_uv, 0.0, abs_tol=1e-6):
            return None
        scale = (texel / (area_uv / area_3d))
        return self.scale(Vector((scale, scale)), self.bbox.center)

    @property
    def flat_3d_coords(self):
        return itertools.chain.from_iterable(isl.flat_3d_coords for isl in self)

    @property
    def flat_uv_coords(self):
        return itertools.chain.from_iterable(isl.flat_coords for isl in self)

    @property
    def flat_coords(self):  # TODO: Remove
        return itertools.chain.from_iterable(isl.flat_coords for isl in self)

    @property
    def weights(self):
        return itertools.chain.from_iterable(isl.weights for isl in self)

    @property
    def flat_unique_uv_coords(self):
        return itertools.chain.from_iterable(isl.flat_unique_uv_coords for isl in self)

    @property
    def select(self):
        raise

    @select.setter
    def select(self, value):
        for isl in self:
            isl.select = value

    def calc_area_uv(self):
        return sum(isl.calc_area_uv() for isl in self)

    def calc_convex_points(self):
        points = []
        if self[0].convex_coords:
            for island in self:
                points.extend(island.convex_coords)
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords
        elif self[0].flat_coords:
            for island in self:
                points.extend(island.flat_coords)
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords
        else:
            for island in self:
                uv = island.umesh.uv
                points.extend([l[uv].uv for f in island for l in f.loops])  # Warning: points referenced to uv
            self.convex_coords = [points[i] for i in mathutils.geometry.convex_hull_2d(points)]
            return self.convex_coords

    @staticmethod
    def calc_overlapped_island_groups(adv_islands: list[AdvIsland], threshold=None) -> list['UnionIslands | AdvIsland']:
        """Warning: Tags should be the default. Optimal Threshold = 0.0005"""
        if not adv_islands:
            return []
        islands_group = []
        union_islands = []
        single_islands = []
        if threshold is not None:
            if threshold == 0:
                threshold_to_precision = 20
            else:
                threshold_to_precision = max(0, int(-math.log10(threshold)))

            class ExactOverlap:
                def __init__(self, island):
                    self.island = island
                    self.coords: np.array = np.array

                def calc_coords(self):
                    uv = self.island.umesh.uv
                    self.coords = np.array([crn[uv].uv for f in self.island for crn in f.loops], dtype='float32')

                def compare(self, other, threshold_):
                    distances = np.linalg.norm(self.coords[:, None] - other.coords, axis=2)
                    a_matches = np.any(distances < threshold_, axis=1)
                    b_matches = np.any(distances < threshold_, axis=0)
                    return np.all(a_matches) and np.all(b_matches)

            # reduce islands by len
            islands_by_len: collections.defaultdict[int | list[AdvIsland]] = collections.defaultdict(list)
            for isl in adv_islands:
                islands_by_len[len(isl)].append(isl)

            islands_by_len_ = islands_by_len.copy()
            for size, list_of_isl in islands_by_len.items():
                if len(list_of_isl) == 1:
                    single_island = islands_by_len_.pop(size)[0]
                    single_islands.append(single_island)

            # reduce islands by bbox
            islands_by_bbox: collections.defaultdict[tuple[float | int] | list[AdvIsland]] = collections.defaultdict(list)
            for size, list_of_isl in islands_by_len_.items():
                for isl in list_of_isl:
                    bbox = isl.bbox
                    bbox_key: list[float | int] = list((round(minmax, threshold_to_precision) for minmax in (bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax)))
                    bbox_key.append(size)
                    islands_by_bbox[tuple(bbox_key)].append(isl)

            islands_by_bbox_ = islands_by_bbox.copy()
            for size, list_of_isl in islands_by_bbox.items():
                if len(list_of_isl) == 1:
                    single_island = islands_by_bbox_.pop(size)[0]
                    single_islands.append(single_island)
            islands_by_len_ = islands_by_bbox_

            # reduce by area_uv
            islands_by_ngons: collections.defaultdict[typing.Any | list[AdvIsland]] = collections.defaultdict(list)
            if adv_islands[0].area_uv != -1.0:
                for list_of_isl in islands_by_len_.values():
                    islands_by_area_uv: collections.defaultdict[float | list[AdvIsland]] = collections.defaultdict(list)
                    for isl in list_of_isl:
                        islands_by_area_uv[round(isl.area_uv, threshold_to_precision)].append(isl)

                    islands_by_area_uv_ = islands_by_area_uv.copy()
                    for area, list_of_isl_by_area in islands_by_area_uv_.items():
                        if len(list_of_isl_by_area) == 1:
                            single_island = islands_by_area_uv_.pop(area)[0]
                            single_islands.append(single_island)
                        else:
                            # reduce by ngons
                            for isl__ in list_of_isl_by_area:
                                ngons_sizes = collections.Counter(len(f.loops) for f in isl__)
                                ngons_sizes = list(ngons_sizes.items())
                                ngons_sizes.sort(key=lambda a: a[0])
                                ngons_sizes.append(area)
                                islands_by_ngons[tuple(ngons_sizes)].append(isl__)
            else:
                # reduce by ngons
                for size, list_of_isl_by_size in islands_by_len_.items():
                    for isl__ in list_of_isl_by_size:
                        ngons_sizes = collections.Counter(len(f.loops) for f in isl__)
                        ngons_sizes = list(ngons_sizes.items())
                        ngons_sizes.sort(key=lambda a: a[0])
                        ngons_sizes.append(size)
                        islands_by_ngons[tuple(ngons_sizes)].append(isl__)

            islands_by_ngons_ = islands_by_ngons.copy()
            for key, list_of_isl in islands_by_ngons.items():
                if len(list_of_isl) == 1:
                    single_island = islands_by_ngons_.pop(key)[0]
                    single_islands.append(single_island)

            for finished_reduced_islands in islands_by_ngons_.values():
                exact_islands = []
                for fin_isl in finished_reduced_islands:
                    exact_isl = ExactOverlap(fin_isl)
                    exact_isl.calc_coords()
                    exact_islands.append(exact_isl)

                for island_first in exact_islands:
                    if not island_first.island.tag:
                        continue
                    island_first.island.tag = False

                    union_islands.append(island_first)
                    compare_index = 0
                    while True:
                        if compare_index > len(union_islands) - 1:
                            if len(union_islands) == 1:
                                single_islands.append(union_islands[0].island)
                            else:
                                islands_group.append(UnionIslands([exact_.island for exact_ in union_islands]))
                            union_islands = []
                            break

                        for isl in exact_islands:
                            if not isl.island.tag:
                                continue
                            if union_islands[compare_index].compare(isl, threshold):
                                isl.island.tag = False
                                union_islands.append(isl)
                        compare_index += 1

        else:
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
        islands_group.extend(single_islands)
        return islands_group

    def append(self, island):
        self.islands.append(island)

    def pop(self, island):
        self.islands.pop(island)


class AdvIslands(Islands):
    island_type = AdvIsland

    def __init__(self, islands: list[AdvIsland] | tuple = (), umesh: _umesh.UMesh | utils.NoInit = utils.NoInit()):
        super().__init__([], umesh)
        self.islands: list[AdvIsland] = islands

    def triangulate_islands(self):
        loop_triangles = self.umesh.bm.calc_loop_triangles()
        self.indexing(force=False)

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

    def calc_flat_coords(self, save_triplet=False):
        for island in self.islands:
            island.calc_flat_coords(save_triplet)

    def calc_flat_uv_coords(self, save_triplet=False):
        self.calc_flat_coords(save_triplet)

    def calc_flat_unique_uv_coords(self):
        for island in self.islands:
            island.calc_flat_unique_uv_coords()

    def calc_flat_3d_coords(self, save_triplet=False, scale=None):
        for island in self.islands:
            island.calc_flat_3d_coords(save_triplet, scale)

    def calc_area_3d(self, scale=None, areas_to_weight=False):
        return sum(isl.calc_area_3d(scale, areas_to_weight) for isl in self)

    def calc_area_uv(self):
        return sum(isl.calc_area_uv() for isl in self)

    def calc_materials(self, umesh: _umesh.UMesh):
        for isl in self:
            isl.calc_materials(umesh)

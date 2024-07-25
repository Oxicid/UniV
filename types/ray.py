"""
Created by Oxicid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import typing
from bmesh.types import BMFace, BMLoop
from mathutils import Vector
from mathutils.kdtree import KDTree
from . import Islands, AdvIslands, LoopGroups
from .. import utils
from ..utils import UMesh, UMeshes
from math import inf
from itertools import chain

class KDData:
    def __init__(self, found, elem, kdmesh):
        self.found: tuple[Vector, int, float] = found
        self.elem: BMFace | BMLoop | None = elem
        self.kdmesh: KDMesh | None | bool = kdmesh

    @property
    def pt(self):
        return self.found[0]

    @property
    def index(self):
        return self.found[1]

    @property
    def distance(self):
        return self.found[2]

    def extract_drag_island(self):
        if len(self.kdmesh.islands) == 1:
            return self.kdmesh.islands.islands.pop()

        self.kdmesh.islands.indexing(force=True)
        if isinstance(self.elem, BMFace):
            ret_isl = self.kdmesh.islands.islands.pop(self.elem.index)
        else:
            ret_isl = self.kdmesh.islands.islands.pop(self.elem.face.index)

        self.kdmesh.faces = []
        self.kdmesh.corners_center = []
        self.kdmesh.calc_all_trees()
        return ret_isl

    def __bool__(self):
        return bool(self.kdmesh)

class KDMesh:
    def __init__(self, umesh, islands=None, loop_groups=None):
        self.umesh: UMesh = umesh
        self.islands: Islands | AdvIslands | None = islands
        self.loop_groups: LoopGroups | None = loop_groups
        self.corners_vert: list[BMLoop] = []
        self.corners_center: list[BMLoop] = []
        self.faces: list[BMFace] = []
        self.kdtree_crn_points: KDTree | None = None
        self.kdtree_crn_center_points: KDTree | None = None
        self.kdtree_face_points: KDTree | None = None

        self.min_res = [Vector((inf, inf, inf)), inf, inf]

    def calc_all_trees(self):
        self.clear_containers()
        for isl in self.islands:
            self.faces.extend(isl)

        if len(self.umesh.bm.faces) == len(self.faces):
            corners_size = self.umesh.total_corners
        else:
            corners_size = sum(len(f.loops) for f in self.faces)
        self.kdtree_crn_points = KDTree(corners_size)
        self.kdtree_crn_center_points = KDTree(corners_size)
        self.kdtree_face_points = KDTree(len(self.faces))

        kd_f_pt_insert = self.kdtree_face_points.insert
        kd_crn_pt_insert = self.kdtree_crn_points.insert
        kd_crn_center_pt_insert = self.kdtree_crn_center_points.insert

        uv = self.islands.uv_layer
        idx_ = 0
        cnr_append = self.corners_vert.append
        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                cnr_append(crn)
                co = crn[uv].uv
                sum_crn += co
                kd_crn_pt_insert(co.to_3d(), idx_)
                kd_crn_center_pt_insert(((co+crn.link_loop_next[uv].uv)/2).to_3d(), idx_)
                idx_ += 1
            kd_f_pt_insert((sum_crn / len(f.loops)).to_3d(), idx)
        self.corners_center = self.corners_vert

        self.kdtree_face_points.balance()
        self.kdtree_crn_points.balance()
        self.kdtree_crn_center_points.balance()

    def calc_all_trees_from_static_corners_by_tag(self):
        self.islands = []
        self.loop_groups = None
        self.clear_containers()
        uv = self.umesh.uv_layer

        corners_vert_append = self.corners_vert.append
        corners_center_append = self.corners_center.append
        faces_append = self.faces.append

        for f in self.umesh.bm.faces:
            full_tagged = True
            for crn in f.loops:
                if crn.tag:
                    corners_vert_append(crn)
                    if crn.link_loop_next.tag:
                        corners_center_append(crn)
                else:
                    full_tagged = False
            if full_tagged:
                faces_append(f)

        self.kdtree_face_points = KDTree(len(self.faces))
        kd_f_pt_insert = self.kdtree_face_points.insert
        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                sum_crn += crn[uv].uv
            kd_f_pt_insert((sum_crn / len(f.loops)).to_3d(), idx)
        self.kdtree_face_points.balance()

        self.kdtree_crn_points = KDTree(len(self.corners_vert))
        kd_crn_pt_insert = self.kdtree_crn_points.insert
        for idx, crn in enumerate(self.corners_vert):
            kd_crn_pt_insert(crn[uv].uv.to_3d(), idx)
        self.kdtree_crn_points.balance()

        self.kdtree_crn_center_points = KDTree(len(self.corners_center))
        kd_crn_center_pt_insert = self.kdtree_crn_center_points.insert
        for idx, crn in enumerate(self.corners_center):
            kd_crn_center_pt_insert(((crn[uv].uv + crn.link_loop_next[uv].uv) / 2).to_3d(), idx)
        self.kdtree_crn_center_points.balance()

    def calc_all_trees_loop_group(self):
        self.clear_containers()
        for lg in self.loop_groups:
            self.corners_center.extend(lg)

        faces_append = self.faces.append
        for f in self.umesh.bm.faces:
            if all(_crn.tag for _crn in f.loops):
                faces_append(f)

        uv = self.loop_groups.umesh.uv_layer

        self.kdtree_crn_points = KDTree(len(self.corners_center) * 2)
        self.kdtree_crn_center_points = KDTree(len(self.corners_center))
        self.kdtree_face_points = KDTree(len(self.faces))

        kd_crn_center_pt_insert = self.kdtree_crn_center_points.insert
        kd_crn_pt_insert = self.kdtree_crn_points.insert
        kd_f_pt_insert = self.kdtree_face_points.insert

        corners_vert_extend = self.corners_vert.extend

        idx_ = 0
        for idx, crn in enumerate(self.corners_center):
            crn_next = crn.link_loop_next
            corners_vert_extend((crn, crn_next))

            co_a = crn[uv].uv.to_3d()
            co_b = crn_next[uv].uv.to_3d()

            kd_crn_pt_insert(co_a, idx_)
            kd_crn_pt_insert(co_b, idx_+1)

            kd_crn_center_pt_insert((co_a+co_b)/2, idx)
            idx_ += 2

        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                sum_crn += crn[uv].uv
            kd_f_pt_insert((sum_crn / len(f.loops)).to_3d(), idx)

        self.kdtree_face_points.balance()
        self.kdtree_crn_points.balance()
        self.kdtree_crn_center_points.balance()

    def calc_face_trees(self):
        for isl in self.islands:
            self.faces.extend(isl)

        self.kdtree_face_points = KDTree(len(self.faces))
        kd_f_pt_insert = self.kdtree_face_points.insert

        uv = self.umesh.uv_layer
        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                sum_crn += crn[uv].uv
            kd_f_pt_insert((sum_crn / len(f.loops)).to_3d(), idx)

        self.kdtree_face_points.balance()

    def find_range(self, co, r):
        res = self.kdtree_face_points.find_range(co, r)
        if res:
            res.extend(self.kdtree_crn_points.find_range(co, r))
        else:
            res = self.kdtree_crn_points.find_range(co, r)
        if res:
            res.extend(self.kdtree_crn_center_points.find_range(co, r))
        else:
            res = self.kdtree_crn_center_points.find_range(co, r)
        return res

    def find_range_vert(self, co, r):
        return self.kdtree_crn_points.find_range(co, r)

    def find_range_crn_center(self, co, r):
        return self.kdtree_crn_center_points.find_range(co, r)

    def find_range_face_center(self, co, r):
        return self.kdtree_face_points.find_range(co, r)

    def clear_containers(self):
        self.corners_center = []
        self.corners_vert = []
        self.faces = []

    def __bool__(self):
        return bool(self.corners_vert or self.corners_center or self.faces)


class KDMeshes:
    def __init__(self, kdmeshes):
        self.kdmeshes: list[KDMesh] = kdmeshes

    @classmethod
    def calc_island_rmeshes(cls, umeshes: UMeshes, extended=False):
        sync = utils.sync()
        rmeshes = []
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                kdmesh = KDMesh(umesh, islands)
                kdmesh.calc_all_trees()
                rmeshes.append(kdmesh)
        cls(rmeshes)

    def find_range(self, co, r) -> typing.Iterator[list[Vector, int, float]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.find_range(co, r))
        return chain.from_iterable(founded)

    def find_range_vert(self, co, r) -> typing.Iterator[list[Vector, int, float]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.find_range_vert(co, r))
        return chain.from_iterable(founded)

    def find_range_crn_center(self, co, r) -> typing.Iterator[list[Vector, int, float]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.find_range_crn_center(co, r))
        return chain.from_iterable(founded)

    def find_range_face_center(self, co, r) -> typing.Iterator[list[Vector, int, float]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.find_range_face_center(co, r))
        return chain.from_iterable(founded)

    @staticmethod
    def range_to_coords(founded) -> list[Vector]:
        return [i[0] for i in founded]

    def __iter__(self) -> typing.Iterator[KDMesh]:
        return iter(self.kdmeshes)

    def __getitem__(self, idx) -> KDMesh:
        return self.kdmeshes[idx]

    def __bool__(self):
        return bool(self.kdmeshes)

    def __len__(self):
        return len(self.kdmeshes)

    def __str__(self):
        return f'KD Meshes count = {len(self.kdmeshes)}'

# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import math

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import typing

from math import inf
from bmesh.types import BMFace, BMLoop
from mathutils import Vector
from mathutils.kdtree import KDTree
from itertools import chain

from . import Islands, FaceIsland, AdvIslands, AdvIsland, UnionIslands, LoopGroups
from . import umesh as _umesh  # noqa: F401 # pylint:disable=unused-import
from .umesh import UMesh, UMeshes
from math import isclose
from ..utils import closest_pt_to_line, point_inside_face

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

        self.kdmesh.islands.indexing()
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

        uv = self.islands.umesh.uv
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
        uv = self.umesh.uv

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

        uv = self.loop_groups.umesh.uv

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

        uv = self.umesh.uv
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
        rmeshes = []
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh, extended=extended):
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

class IslandHit:
    def __init__(self, pt, min_dist=1e200):
        self.island: AdvIsland | FaceIsland | UnionIslands | None = None
        self.point = pt
        self.min_dist = min_dist
        self.crn = None
        self.face = None

    def find_nearest_island(self, island: AdvIsland | FaceIsland | UnionIslands):
        if not isinstance(island, UnionIslands):
            island = (island, )
        pt = self.point
        min_dist = self.min_dist

        zero_pt = Vector((0.0, 0.0))
        for isl in island:
            uv = isl.umesh.uv
            for f in isl:
                face_center = zero_pt.copy()
                corners = f.loops
                v_prev = corners[-1][uv].uv
                for crn in corners:
                    v_curr = crn[uv].uv
                    face_center += v_curr

                    close_pt = closest_pt_to_line(pt, v_prev, v_curr)

                    if isclose((dist := (close_pt-pt).length), min_dist, abs_tol=1e-07):
                        if point_inside_face(pt, f, uv):
                            min_dist = dist
                            # This is necessary for the inequality check to be successful
                            self.min_dist = math.nextafter(self.min_dist, self.min_dist+1)
                    elif dist < min_dist:
                        min_dist = dist
                    v_prev = v_curr

                if (dist := (face_center / len(corners) - pt).length) < min_dist:
                    min_dist = dist

        if self.min_dist != min_dist:
            self.min_dist = min_dist
            if isinstance(island, tuple):
                self.island = island[0]
            else:
                self.island = island

            return True
        return False

    def find_nearest_island_with_face(self, island: AdvIsland | FaceIsland | UnionIslands):
        if not isinstance(island, UnionIslands):
            island = (island, )
        pt = self.point
        min_dist = self.min_dist
        min_face = None

        zero_pt = Vector((0.0, 0.0))
        for isl in island:
            uv = isl.umesh.uv
            for f in isl:
                face_center = zero_pt.copy()
                corners = f.loops
                v_prev = corners[-1][uv].uv
                for crn in corners:
                    v_curr = crn[uv].uv
                    face_center += v_curr

                    close_pt = closest_pt_to_line(pt, v_prev, v_curr)

                    if isclose((dist := (close_pt-pt).length), min_dist, abs_tol=1e-07):
                        if point_inside_face(pt, f, uv):
                            min_dist = dist
                            min_face = f
                            # This is necessary for the inequality check to be successful
                            self.min_dist = math.nextafter(self.min_dist, self.min_dist+1)
                    elif dist < min_dist:
                        min_dist = dist
                        min_face = f
                    v_prev = v_curr

                if (dist := (face_center / len(corners) - pt).length) < min_dist:
                    min_dist = dist
                    min_face = f

        if self.min_dist != min_dist:
            self.face = min_face
            self.min_dist = min_dist
            if isinstance(island, tuple):
                self.island = island[0]
            else:
                self.island = island

            return True
        return False

    def find_nearest_island_by_crn(self, isl: AdvIsland):
        pt = self.point
        min_dist = self.min_dist
        min_crn = None

        uv = isl.umesh.uv
        for f in isl:
            corners = f.loops
            v_prev = corners[-1][uv].uv
            for crn in corners:
                v_curr = crn[uv].uv

                close_pt = closest_pt_to_line(pt, v_prev, v_curr)

                if isclose((dist := (close_pt-pt).length), min_dist, abs_tol=1e-07):
                    if point_inside_face(pt, f, uv):
                        min_dist = dist
                        min_crn = crn
                        self.min_dist = math.nextafter(self.min_dist, self.min_dist+1)
                elif dist < min_dist:
                    min_dist = dist
                    min_crn = crn
                v_prev = v_curr

        if self.min_dist != min_dist:
            self.min_dist = min_dist
            self.island = isl
            self.crn = min_crn.link_loop_prev
            return True
        return False

    def mouse_to_pos(self, event, view):
        self.point = Vector(view.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def __bool__(self):
        return bool(self.island)

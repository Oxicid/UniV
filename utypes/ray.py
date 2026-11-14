# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import bpy
import math
import bmesh
import typing
import mathutils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import numpy as np
from math import inf, isclose
from bmesh.types import BMFace, BMLoop

from itertools import chain
from bpy_extras import view3d_utils

from . import AdvIsland, Islands, MeshIsland, UnionIslands, LoopGroups
from . import umesh as _umesh  # noqa: F401 # pylint:disable=unused-import
from .umesh import UMesh, UMeshes
from .. import utils
from ..utils import point_inside_face, intersect_point_line_segment


class KDData:
    def __init__(
        self, found: tuple[Vector, int, float], elem: BMFace | BMLoop | None, kdmesh: "KDMesh"
    ):
        self.found: tuple[Vector, int, float] = found  # pt, index, distance
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
    def __init__(self, umesh, islands: Islands | None = None, loop_groups=None):
        self.umesh: UMesh = umesh
        self.islands: Islands | list | None = islands
        self.loop_groups: LoopGroups | None = loop_groups
        self.corners_vert: list[BMLoop] = []
        self.corners_center: list[BMLoop] = []
        self.faces: list[BMFace] = []
        self.kdtree_crn_points: mathutils.kdtree.KDTree | None = None
        self.kdtree_crn_center_points: mathutils.kdtree.KDTree | None = None
        self.kdtree_face_points: mathutils.kdtree.KDTree | None = None

        self.min_res = [Vector((inf, inf, inf)), inf, inf]

    def calc_all_trees(self):
        from ..utils import KDTree_from_coords, splitshift
        self.clear_containers()
        for isl in self.islands:
            self.faces.extend(isl)

        uv = self.umesh.uv

        verts_coords = []
        edge_center_coords = []
        face_center_coords = []

        for f in self.faces:
            sum_crn = Vector((0.0, 0.0))
            for curr_uv_co, next_uv_co in splitshift(crn[uv].uv for crn in f.loops):
                verts_coords.append(curr_uv_co)
                edge_center_coords.append((curr_uv_co + next_uv_co) * 0.5)
                sum_crn += curr_uv_co
            face_center_coords.append(sum_crn / len(f.loops))
        self.corners_vert = self.corners_center = [crn for f in self.faces for crn in f.loops]

        self.kdtree_crn_points = KDTree_from_coords(verts_coords, dimensions=2)
        self.kdtree_crn_center_points = KDTree_from_coords(edge_center_coords, dimensions=2)
        self.kdtree_face_points = KDTree_from_coords(face_center_coords, dimensions=2)

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

        from ..utils import face_centroid_uv, KDTree_from_coords

        self.kdtree_crn_points = KDTree_from_coords([crn[uv].uv for crn in self.corners_center], dimensions=2)

        edge_centres = [((crn[uv].uv + crn.link_loop_next[uv].uv) * 0.5) for crn in self.corners_center]
        self.kdtree_crn_center_points = KDTree_from_coords(edge_centres, dimensions=2)
        self.kdtree_face_points = KDTree_from_coords([face_centroid_uv(f, uv) for f in self.faces], dimensions=2)

    def calc_all_trees_loop_group(self):
        self.clear_containers()
        for lg in self.loop_groups:
            self.corners_center.extend(lg)

        self.faces = [f for f in self.umesh.bm.faces if all(_crn.tag for _crn in f.loops)]

        uv = self.loop_groups.umesh.uv

        corners_vert_extend = self.corners_vert.extend

        vertices_coords = []
        edge_center_coords = []

        for crn in self.corners_center:
            crn_next = crn.link_loop_next
            corners_vert_extend((crn, crn_next))

            co_a = crn[uv].uv
            co_b = crn_next[uv].uv
            vertices_coords.append(co_a)
            vertices_coords.append(co_b)

            edge_center_coords.append((co_a + co_b) * 0.5)

        from ..utils import face_centroid_uv, KDTree_from_coords

        self.kdtree_crn_points = KDTree_from_coords(vertices_coords, dimensions=2)
        self.kdtree_crn_center_points = KDTree_from_coords(edge_center_coords, dimensions=2)
        self.kdtree_face_points = KDTree_from_coords([face_centroid_uv(f, uv) for f in self.faces], dimensions=2)

    def find_range(self, co: Vector, r: float):
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

    def clear_containers(self):
        self.corners_center = []
        self.corners_vert = []
        self.faces = []

    def __bool__(self):
        return bool(self.corners_vert or self.corners_center or self.faces)


class KDMeshes:
    def __init__(self, kdmeshes):
        self.kdmeshes: list[KDMesh] = kdmeshes

    def find_range_vert(self, co, r) -> typing.Iterator[list[tuple[Vector, int, float]]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.kdtree_crn_points.find_range(co, r))
        return chain.from_iterable(founded)

    def find_range_crn_center(self, co, r) -> typing.Iterator[list[tuple[Vector, int, float]]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.kdtree_crn_center_points.find_range(co, r))
        return chain.from_iterable(founded)

    def find_range_face_center(self, co, r) -> typing.Iterator[list[tuple[Vector, int, float]]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.kdtree_face_points.find_range(co, r))
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
        return f"KD Meshes count = {len(self.kdmeshes)}."


class TrimKDTree:
    def __init__(self):
        from ..operators.quick_snap import eSnapPointMode

        self.kdtree = utils.KDTree_from_coords([], dimensions=2)
        self.elem_flag: eSnapPointMode = eSnapPointMode.NONE

    def calc(self, flag, bboxes=None):
        self.elem_flag = flag
        if bboxes is None:
            bboxes = utils.get_trim_bboxes()

        coords = []
        for bb in bboxes:
            coords.extend(bb.draw_data_verts())

            coords.append(bb.center)
            coords.extend((bb.left, bb.right, bb.upper, bb.bottom))

        self.kdtree = utils.KDTree_from_coords(coords, dimensions=2)


class IslandHit:
    def __init__(self, pt, min_dist=1e200):
        self.island: AdvIsland | UnionIslands | None = None
        self.point: Vector = pt
        self.min_dist: float = min_dist
        self.crn: BMLoop | None = None
        self.face: BMFace | None = None

    def find_nearest_island(self, island: AdvIsland | UnionIslands):
        if not isinstance(island, UnionIslands):
            island = (island,)
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

                    close_pt, dist = intersect_point_line_segment(pt, v_prev, v_curr)
                    if isclose(dist, min_dist, abs_tol=1e-07):
                        if point_inside_face(pt, f, uv):
                            min_dist = dist
                            # This is necessary for the inequality (in below) check to be successful.
                            self.min_dist = math.nextafter(self.min_dist, self.min_dist + 1)
                    elif dist < min_dist:
                        min_dist = dist
                    v_prev = v_curr

                dist = (face_center / len(corners) - pt).length
                if dist < min_dist:
                    min_dist = dist

        if self.min_dist != min_dist:
            self.min_dist = min_dist
            if isinstance(island, tuple):
                self.island = island[0]
            else:
                self.island = island

            return True
        return False

    def find_nearest_island_with_face(self, island: AdvIsland | UnionIslands):
        if not isinstance(island, UnionIslands):
            island = (island,)
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

                    close_pt, dist = intersect_point_line_segment(pt, v_prev, v_curr)
                    if isclose(dist, min_dist, abs_tol=1e-07):
                        if point_inside_face(pt, f, uv):
                            min_dist = dist
                            min_face = f
                            # This is necessary for the inequality check (in below) to be successful.
                            self.min_dist = math.nextafter(self.min_dist, self.min_dist + 1.0)
                    elif dist < min_dist:
                        min_dist = dist
                        min_face = f
                    v_prev = v_curr

                dist = (face_center / len(corners) - pt).length
                if dist < min_dist:
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

                close_pt, dist = intersect_point_line_segment(pt, v_prev, v_curr)
                if isclose(dist, min_dist, abs_tol=1e-07):
                    if point_inside_face(pt, f, uv):
                        min_dist = dist
                        min_crn = crn
                        self.min_dist = math.nextafter(self.min_dist, self.min_dist + 1)
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

    @staticmethod
    def closest_pt_to_selected_edge(island: AdvIsland, pt) -> float:
        min_dist = math.inf
        uv = island.umesh.uv
        for crn in island.calc_selected_edge_corners_iter():
            _, dist = utils.intersect_point_line_segment(pt, crn[uv].uv, crn.link_loop_next[uv].uv)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def mouse_to_pos(self, event, view):
        self.point = Vector(view.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def __bool__(self):
        return bool(self.island)

    def __str__(self):
        if self.island:
            return f"{self.island}. Distance={self.min_dist:.5}."
        else:
            return f"Island not found. Distance={self.min_dist:.5}."


class CrnEdgeHit:
    def __init__(self, pt, min_dist=1e200):
        self.point = pt
        self.min_dist = min_dist
        self.crn: BMLoop | None = None
        self.face: BMFace | None = None  # use for incref
        self.umesh = None

    def find_nearest_crn_by_visible_faces(self, umesh, use_faces_from_umesh_seq=False):
        from .. import utils
        from math import nextafter, inf
        from ..utils import intersect_point_line_segment

        pt = self.point
        min_dist = self.min_dist
        min_crn = None

        uv = umesh.uv

        if use_faces_from_umesh_seq:
            visible_faces = umesh.sequence
        else:
            visible_faces = utils.calc_visible_uv_faces_iter(umesh)

        for f in visible_faces:
            corners = f.loops
            v_prev = corners[-1][uv].uv
            for crn in corners:
                v_curr = crn[uv].uv

                _, dist = intersect_point_line_segment(pt, v_prev, v_curr)
                # TODO: Prioritize flipped face
                if dist < min_dist:
                    # If the point is inside the face, we add it immediately,
                    # otherwise, we do nextafter and check again for nearest.
                    if point_inside_face(pt, f, uv):
                        min_crn = crn
                        min_dist = dist
                    else:
                        # Adding dist after nextafter is necessary for the next for_each loop
                        # to “hook” another edge (thus avoiding float point errors).
                        dist = nextafter(dist, inf)
                        if dist < min_dist:
                            min_crn = crn
                            min_dist = dist

                v_prev = v_curr

        if min_crn:
            self.crn = min_crn.link_loop_prev
            self.min_dist = min_dist

            radial_prev = self.crn.link_loop_radial_prev
            if utils.is_pair_with_flip(self.crn, radial_prev, umesh.uv) and utils.is_visible_func(umesh.sync)(radial_prev.face):  # fmt: skip
                if point_inside_face(pt, radial_prev.face, uv):
                    self.crn = radial_prev
            else:
                # Prioritize boundary edges where the point is inside the face,
                # otherwise lower the priority to find other boundary edges with the point inside the face.
                if point_inside_face(pt, self.crn.face, uv):
                    self.min_dist = nextafter(min_dist, -inf)
                else:
                    self.min_dist = nextafter(min_dist, inf)

            self.umesh = umesh
            return True
        return False

    def find_nearest_crn_by_corners_from_pick(self, umesh, corners):
        # from .. import utils
        from math import nextafter, inf
        from ..utils import intersect_point_line_segment

        assert self.point is not None
        pt = self.point
        min_dist = self.min_dist
        min_crn = None

        uv = umesh.uv

        for crn in corners:
            _, dist = intersect_point_line_segment(pt, crn[uv].uv, crn.link_loop_next[uv].uv)
            # TODO: Prioritize flipped face
            if dist < min_dist:
                # If the point is inside the face, we add it immediately,
                # otherwise, we do nextafter and check again for nearest.
                if point_inside_face(pt, crn.face, uv):
                    min_crn = crn
                    min_dist = dist
                    # TODO: Add scal from View2D
                else:
                    # Adding dist after nextafter is necessary for the next for_each loop
                    # to “hook” another edge (thus avoiding float point errors).
                    dist = nextafter(dist, inf)
                    if dist < min_dist:
                        min_crn = crn
                        min_dist = dist

        if min_crn:
            self.crn = min_crn
            self.min_dist = min_dist

            # radial_prev = self.crn.link_loop_radial_prev
            # if utils.is_pair_with_flip(self.crn, radial_prev, umesh.uv) and utils.is_visible_func(umesh.sync)(radial_prev.face):  # fmt: skip
            #     if point_inside_face(pt, radial_prev.face, uv):
            #         self.crn = radial_prev
            # else:
            #     # Prioritize boundary edges where the point is inside the face,
            #     # otherwise lower the priority to find other boundary edges with the point inside the face.
            #     if point_inside_face(pt, self.crn.face, uv):
            #         self.min_dist = nextafter(min_dist, -inf)
            #     else:
            #         self.min_dist = nextafter(min_dist, inf)

            self.umesh = umesh
            return True
        return False

    def find_nearest_vert_by_visible_faces(self, umesh, use_faces_from_umesh_seq=False):
        from .. import utils
        from math import nextafter, inf

        pt = self.point
        min_dist = self.min_dist
        min_crn = None

        uv = umesh.uv

        if use_faces_from_umesh_seq:
            visible_faces = umesh.sequence
        else:
            visible_faces = utils.calc_visible_uv_faces_iter(umesh)

        for f in visible_faces:
            for crn in f.loops:
                dist = (crn[uv].uv - pt).length
                if dist < min_dist:
                    # If the point is inside the face, we add it immediately,
                    # otherwise, we do nextafter and check again for nearest.
                    if point_inside_face(pt, f, uv):
                        min_crn = crn
                        min_dist = dist
                    else:
                        # Adding dist after nextafter is necessary for the next for_each loop
                        # to “hook” another edge (thus avoiding float point errors).
                        dist = nextafter(dist, inf)
                        if dist < min_dist:
                            min_crn = crn
                            min_dist = dist

        if min_crn:
            self.crn = min_crn  # .link_loop_prev
            self.min_dist = min_dist

            # radial_prev = self.crn.link_loop_radial_prev
            # if (utils.is_pair_with_flip(self.crn, radial_prev, umesh.uv) and
            #         utils.is_visible_func(umesh.sync)(radial_prev.face)):
            #     if point_inside_face(pt, radial_prev.face, uv):
            #         self.crn = radial_prev
            # else:
            #     # Prioritize boundary edges where the point is inside the face,
            #     # otherwise lower the priority to find other boundary edges with the point inside the face.
            #     if point_inside_face(pt, self.crn.face, uv):
            #         self.min_dist = nextafter(min_dist, -inf)
            #     else:
            #         self.min_dist = nextafter(min_dist, inf)

            self.umesh = umesh
            return True
        return False

    def calc_island_with_seam(self):
        assert self.crn, "Not found picked corner"

        faces: set[BMFace] = {self.crn.face}
        is_boundary = utils.is_boundary_func(self.umesh)

        stack = []
        parts_of_island = [self.crn.face]
        while parts_of_island:
            for f in parts_of_island:
                for crn in f.loops:
                    if is_boundary(crn):
                        continue

                    ff = crn.link_loop_radial_prev.face
                    if ff not in faces:
                        faces.add(ff)
                        stack.append(ff)

            parts_of_island = stack
            stack = []

        return AdvIsland(list(faces), self.umesh), faces

    def calc_island_non_manifold_ignore_seams(self) -> tuple[AdvIsland, set[BMFace]]:
        assert self.crn, "Not found picked corner"

        uv = self.umesh.uv
        island: set[BMFace] = {self.crn.face}
        is_visible = utils.is_visible_func(self.umesh.sync)

        stack = []
        parts_of_island = [self.crn.face]
        while parts_of_island:
            for f in parts_of_island:
                for crn in f.loops:
                    pair_crn = crn.link_loop_radial_prev
                    ff = pair_crn.face
                    if ff in island or not is_visible(ff):
                        continue

                    if (
                        crn[uv].uv == pair_crn.link_loop_next[uv].uv
                        or crn.link_loop_next[uv].uv == pair_crn[uv].uv  # Key moment, `or` instead `and`.
                    ):
                        island.add(ff)
                        stack.append(ff)
            parts_of_island = stack
            stack = []

        return AdvIsland(list(island), self.umesh), island

    def calc_island_non_manifold_with_flip_3d_correction_and_ignore_seams(self) -> tuple[AdvIsland, set[BMFace]]:
        assert self.crn, "Not found picked corner"

        uv = self.umesh.uv
        island: set[BMFace] = {self.crn.face}
        is_visible = utils.is_visible_func(self.umesh.sync)

        stack = []
        parts_of_island = [self.crn.face]
        while parts_of_island:
            for f in parts_of_island:
                for crn in f.loops:
                    pair_crn = crn.link_loop_radial_prev
                    ff = pair_crn.face
                    if ff in island or not is_visible(ff):
                        continue

                    if crn.vert == pair_crn.vert:
                        if (
                            crn[uv].uv == pair_crn[uv].uv
                            or crn.link_loop_next[uv].uv == pair_crn.link_loop_next[uv].uv
                        ):
                            island.add(ff)
                            stack.append(ff)
                    else:
                        if (
                            crn[uv].uv == pair_crn.link_loop_next[uv].uv
                            or crn.link_loop_next[uv].uv == pair_crn[uv].uv
                        ):
                            island.add(ff)
                            stack.append(ff)
            parts_of_island = stack
            stack = []

        return AdvIsland(list(island), self.umesh), island

    def calc_visible_mesh_island(self) -> tuple[MeshIsland, set[BMFace]]:
        assert self.crn, "Not found picked corner"
        island: set[BMFace] = {self.crn.face}
        stack = []
        parts_of_island = [self.crn.face]
        while parts_of_island:
            for f in parts_of_island:
                for crn in f.loops:
                    pair_crn = crn.link_loop_radial_prev
                    ff = pair_crn.face
                    if ff in island or ff.hide or crn.edge.seam:
                        continue

                    island.add(ff)
                    stack.append(ff)
            parts_of_island = stack
            stack = []

        return MeshIsland(list(island), self.umesh), island

    def calc_visible_mesh_island_without_ms(self) -> tuple[MeshIsland, set[BMFace]]:
        assert self.crn, "Not found picked corner"
        island: set[BMFace] = {self.crn.face}
        stack = []
        parts_of_island = [self.crn.face]
        while parts_of_island:
            for f in parts_of_island:
                for crn in f.loops:
                    pair_crn = crn.link_loop_radial_prev
                    ff = pair_crn.face
                    if ff in island or ff.hide:
                        continue

                    island.add(ff)
                    stack.append(ff)
            parts_of_island = stack
            stack = []

        return MeshIsland(list(island), self.umesh), island

    def __bool__(self):
        return bool(self.crn)


class RayCastAndPick(bpy.types.Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos_from_3d: tuple[int, int] | None = None
        self.region: bpy.types.Region | None = None
        self.rv3d: bpy.types.RegionView3D | None = None
        self.ray_origin: Vector | None = None
        self.ray_direction: Vector | None = None

        self.max_distance: float = 0.0
        self.mouse_position: Vector | None = None

    def init_data_for_ray_cast(self, event):
        if bpy.context.area.type == "VIEW_3D":
            self.mouse_pos_from_3d = event.mouse_region_x, event.mouse_region_y
            self.region = bpy.context.region
            self.rv3d = bpy.context.space_data.region_3d
            self.ray_origin = view3d_utils.region_2d_to_origin_3d(
                self.region, self.rv3d, Vector(self.mouse_pos_from_3d)
            )
            self.ray_direction = view3d_utils.region_2d_to_vector_3d(
                self.region, self.rv3d, Vector(self.mouse_pos_from_3d)
            )

    @staticmethod
    def get_bvh_from_polygon(umesh: UMesh) -> tuple[BVHTree, list[BMFace]]:
        faces = []
        faces_append = faces.append
        flat_tris_coords = []
        flat_tris_coords_append = flat_tris_coords.append

        for crn_a, crn_b, crn_c in umesh.bm.calc_loop_triangles():
            face = crn_a.face
            if face.hide:
                continue
            faces_append(face)
            flat_tris_coords_append(crn_a.vert.co)
            flat_tris_coords_append(crn_b.vert.co)
            flat_tris_coords_append(crn_c.vert.co)

        indices = np.arange(len(flat_tris_coords), dtype="uint32").reshape(-1, 3).tolist()
        bvh = BVHTree.FromPolygons(flat_tris_coords, indices, all_triangles=True)
        return bvh, faces

    def ray_cast_umeshes(self, umeshes: "UMeshes"):
        ray_target = self.ray_origin + self.ray_direction
        # from .. import draw
        # draw.LinesDrawSimple3D.max_draw_time = 5

        max_dist = 50_000
        best_length_squared = float("inf")
        umesh: UMesh | None = None
        face_index: int = 0
        for umesh_iter in umeshes:
            world_matrix = umesh_iter.obj.matrix_world
            matrix_inv = world_matrix.inverted()
            ray_origin_obj = matrix_inv @ self.ray_origin
            ray_target_obj = matrix_inv @ ray_target
            ray_direction_obj = ray_target_obj - ray_origin_obj

            bvh: BVHTree = BVHTree.FromBMesh(umesh_iter.bm)
            hit, normal, face_index_, distance = bvh.ray_cast(
                ray_origin_obj, ray_direction_obj, max_dist
            )

            if not hit:
                continue
            # draw.LinesDrawSimple3D.draw_register([world_matrix @ ray_origin_obj, world_matrix @ hit])

            hit_world = world_matrix @ hit
            length_squared = (hit_world - self.ray_origin).length_squared
            if length_squared < best_length_squared:
                umesh_iter.ensure()
                umesh_iter.bm = bmesh.from_edit_mesh(umesh_iter.obj.data)
                # If a face is hidden, the BVH is computed using FromPolygons.
                if umesh_iter.bm.faces[face_index_].hide:
                    bvh, faces = self.get_bvh_from_polygon(umesh_iter)  # slow
                    hit, normal, face_index_, distance = bvh.ray_cast(
                        ray_origin_obj, ray_direction_obj, max_dist
                    )
                    if not hit:
                        continue
                    hit_world = world_matrix @ hit
                    length_squared = (hit_world - self.ray_origin).length_squared
                    if length_squared >= best_length_squared:
                        continue
                    umesh_iter.bm.faces.index_update()
                    face_index_ = faces[face_index_].index

                best_length_squared = length_squared
                umesh = umesh_iter
                face_index = face_index_

        return umesh, face_index

    def ray_cast(self, umeshes: "UMeshes", max_pick_radius: float):
        # TODO: Add raycast by radial patterns
        deps = bpy.context.view_layer.depsgraph
        result, loc, normal, face_index, obj, matrix = bpy.context.scene.ray_cast(
            deps, origin=self.ray_origin, direction=self.ray_direction
        )

        if not (
            result and obj and obj.type == "MESH"
        ):  # TODO: Fix potential non-mesh overlap object
            # TODO: Add raycast, ignoring objects that are not in Edit Mode
            return None

        eval_obj = obj.evaluated_get(deps)
        has_destructive_modifiers = len(obj.data.polygons) != len(eval_obj.data.polygons)
        if obj.mode != "EDIT" or has_destructive_modifiers:
            # Raycast, ignoring objects that are not in Edit Mode
            umesh, face_index = self.ray_cast_umeshes(umeshes)
            if not umesh:
                return None
        else:
            umesh: UMesh = next(u for u in umeshes if u.obj == obj)
        umesh.ensure()
        umesh.bm = bmesh.from_edit_mesh(umesh.obj.data)

        face = umesh.bm.faces[face_index]
        e, dist = utils.find_closest_edge_3d_to_2d(
            self.mouse_pos_from_3d, face, umesh, self.region, self.rv3d
        )
        if dist < max_pick_radius:
            for crn in e.link_loops:
                if crn.face == face:
                    hit = CrnEdgeHit(self.mouse_pos_from_3d)
                    hit.crn = crn
                    hit.umesh = umesh
                    hit.face = crn.face  # incref
                    return hit
        return None

    def store_mouse_pose_on_uv_and_max_distance_if_allowed(self, event):
        if event.value in ("PRESS", "DOUBLE_CLICK"):
            if self.bl_idname.startswith("UV"):
                if bpy.context.area.ui_type == "UV":
                    from .. import preferences

                    self.max_distance = utils.get_max_distance_from_px(
                        preferences.prefs().max_pick_distance, bpy.context.region.view2d
                    )
                    self.mouse_position = Vector(
                        bpy.context.region.view2d.region_to_view(
                            event.mouse_region_x, event.mouse_region_y
                        )
                    )
                    return True
        return False

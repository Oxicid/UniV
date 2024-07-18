# import bpy
# import mathutils
import typing
from bmesh.types import BMFace, BMLoop
from mathutils import Vector
from mathutils.kdtree import KDTree
from . import Islands, AdvIslands
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

    def calc_visible_island(self):
        pass

    def extract_drag_island(self):
        if len(self.kdmesh.islands) == 1:
            return self.kdmesh.islands.islands.pop()

        self.kdmesh.islands.indexing(force=True)
        if isinstance(self.elem, BMFace):
            ret_isl = self.kdmesh.islands.islands.pop(self.elem.index)
        else:
            ret_isl = self.kdmesh.islands.islands.pop(self.elem.face.index)

        self.kdmesh.faces = []
        self.kdmesh.corners = []
        self.kdmesh.calc_all_trees()
        return ret_isl

    def __bool__(self):
        return bool(self.kdmesh)

class KDMesh:
    def __init__(self, umesh, islands):
        self.umesh: UMesh = umesh
        self.islands: Islands | AdvIslands = islands
        self.corners: list[BMLoop] = []
        self.faces: list[BMFace] = []
        self.kdtree_crn_points: KDTree | None = None
        self.kdtree_crn_center_points: KDTree | None = None
        self.kdtree_face_points: KDTree | None = None

        self.min_res = [Vector((inf, inf, inf)), inf, inf]

    def calc_faces(self):
        for isl in self.islands:
            self.faces.extend(isl)
        self.kdtree_face_points = KDTree(len(self.faces))

        kd_insert = self.kdtree_face_points.insert
        uv = self.islands.uv_layer
        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                sum_crn += crn[uv].uv
            kd_insert((sum_crn / len(f.loops)).to_3d(), idx)

        self.kdtree_face_points.balance()

    def calc_all_trees(self):
        for isl in self.islands:
            self.faces.extend(isl)

        corners_size = sum(len(f.loops) for f in self.faces)
        self.kdtree_crn_points = KDTree(corners_size)
        self.kdtree_crn_center_points = KDTree(corners_size)
        self.kdtree_face_points = KDTree(len(self.faces))

        kd_f_pt_insert = self.kdtree_face_points.insert
        kd_crn_pt_insert = self.kdtree_crn_points.insert
        kd_crn_center_pt_insert = self.kdtree_crn_center_points.insert

        uv = self.islands.uv_layer
        idx_ = 0
        cnr_append = self.corners.append
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

        self.kdtree_face_points.balance()
        self.kdtree_crn_points.balance()
        self.kdtree_crn_center_points.balance()

    def calc_face_trees(self):
        for isl in self.islands:
            self.faces.extend(isl)

        self.kdtree_face_points = KDTree(len(self.faces))
        kd_f_pt_insert = self.kdtree_face_points.insert

        uv = self.islands.uv_layer
        for idx, f in enumerate(self.faces):
            sum_crn = Vector((0.0, 0.0))
            for crn in f.loops:
                sum_crn += crn[uv].uv
            kd_f_pt_insert((sum_crn / len(f.loops)).to_3d(), idx)

        self.kdtree_face_points.balance()

    def find_range(self, co, r):
        res = []
        if self.kdtree_face_points:
            res.extend(self.kdtree_face_points.find_range(co, r))
        if self.kdtree_crn_points:
            res.extend(self.kdtree_crn_points.find_range(co, r))
        if self.kdtree_crn_center_points:
            res.extend(self.kdtree_crn_center_points.find_range(co, r))
        return res

    def find_all(self, co, r):
        min_res = [Vector((inf, inf, inf)), inf, inf]
        if min_res_ := self.kdtree_face_points.find(co):
            min_res = min_res_
        if min_res_ := self.kdtree_crn_points.find(co):
            if min_res_[2] < min_res[2]:
                min_res = min_res_
        if min_res_ := self.kdtree_crn_center_points.find(co):
            if min_res_[2] < min_res[2]:
                min_res = min_res_
        if min_res[2] <= r:
            return min_res
        return [Vector((inf, inf, inf)), inf, inf]

    def find_from_all_trees_with_elem(self, co, r) -> tuple[list[Vector, int, float], BMFace | BMLoop | None]:
        founded_elem: BMFace | BMLoop | None = None
        min_res = [Vector((inf, inf, inf)), inf, inf]
        if min_res_ := self.kdtree_face_points.find(co):
            min_res = min_res_
            founded_elem = self.faces[min_res_[1]]
        if min_res_ := self.kdtree_crn_points.find(co):
            if min_res_[2] < min_res[2]:
                min_res = min_res_
                founded_elem = self.corners[min_res_[1]]
        if min_res_ := self.kdtree_crn_center_points.find(co):
            if min_res_[2] < min_res[2]:
                min_res = min_res_
                founded_elem = self.corners[min_res_[1]]
        if min_res[2] <= r:
            return min_res, founded_elem
        return [Vector((inf, inf, inf)), inf, inf], None

    def find_face(self, co, r):
        if res := self.kdtree_face_points.find(co):
            if res[2] <= r:
                return res
        return [Vector((inf, inf, inf)), inf, inf]

    def find_crn(self, co, r):
        if res := self.kdtree_crn_points.find(co):
            if res[2] <= r:
                return res
        return [Vector((inf, inf, inf)), inf, inf]

    def find_crn_center(self, co, r):
        if res := self.kdtree_crn_center_points.find(co):
            if res[2] <= r:
                return res
        return [Vector((inf, inf, inf)), inf, inf]


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

    def find_from_all_trees_with_elem(self, co, r) -> KDData:
        r_pt: list[Vector, int, float] = list((Vector((inf, inf, inf)), inf, inf))
        r_elem: BMFace | BMLoop | None = None
        r_kdmesh: KDMesh | None = None
        for kdmesh in self.kdmeshes:
            pt, elem = kdmesh.find_from_all_trees_with_elem(co, r)
            if r_pt[2] > pt[2]:
                r_pt = pt
                r_elem = elem
                r_kdmesh = kdmesh

        return KDData(r_pt, r_elem, r_kdmesh)

    def find_range(self, co, r) -> typing.Iterator[list[Vector, int, float]]:
        founded = []
        for kdmesh in self.kdmeshes:
            founded.append(kdmesh.find_range(co, r))
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


# radial search pattern to find proximity geometry
# used in conjunction with bvh.ray_cast
def radial_patterns():
    from math import sin, cos, pi
    points = (6,)
    bases = [(r, n) for r, n in enumerate(points, 6)]

    _patterns = []
    for r, n in bases:
        t = ((round(cos(2 * pi / n * x) * r),
              round(sin(2 * pi / n * x) * r)) for x in range(n))
        _patterns.append(tuple(t))
    return tuple(_patterns)


# patterns are fixed per pixels, so store
# on module level for fast retrieval
patterns = radial_patterns()


# def search(rc, rv3d, rw, rh, mx, my):
#     for p in patterns:
#         for x, y in p:
#             ret = ray_cast(rc, rv3d, rw, rh, mx + x * 2, my + y * 2)
#             if ret:
#                 return ret

import bpy
import bmesh
import math
# import mathutils

from mathutils import Vector, Matrix

from bmesh.types import BMesh, BMFace, BMLayerItem
from ..utils import umath
from . import btypes


class IslandsBase:
    @staticmethod
    def tag_filter_any(bm: BMesh, tag=True):
        for face in bm.faces:
            face.tag = tag

    @staticmethod
    def tag_filter_selected(bm: BMesh,
                            uv_layer: BMLayerItem,
                            sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if sync:
            for face in bm.faces:
                face.tag = face.select
        else:
            for face in bm.faces:
                face.tag = all(l[uv_layer].select_edge for l in face.loops) and face.select

    @staticmethod
    def tag_filter_visible(bm: BMesh,
                           sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if sync:
            for face in bm.faces:
                face.tag = not face.hide
        else:
            for face in bm.faces:
                face.tag = not face.hide and face.select

    @staticmethod
    def island_filter_is_partial_face_selected(island: list[BMFace],
                                           uv_layer: BMLayerItem,
                                           sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync) -> bool:
        if sync:
            select = (face.select for face in island)
        else:
            select = (l[uv_layer].select_edge for face in island for l in face.loops)
        first_check = next(select)
        return any(first_check is not i for i in select)

    @staticmethod
    def island_filter_is_all_face_selected(island: list[BMFace],
                                           uv_layer: BMLayerItem,
                                           sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync) -> bool:
        if sync:
            return all(face.select for face in island)
        else:
            return all(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_face_selected(island: list[BMFace],
                                      uv_layer: BMLayerItem,
                                      sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync) -> bool:
        if sync:
            return any(face.select for face in island)
        else:
            return any(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_all_corner_selected(island: list[BMFace],
                                             uv_layer: 'bmesh.types.BMLayerItem',
                                             sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync) -> bool:
        assert (sync is False)
        return all(all(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def island_filter_is_any_corner_selected(island: list[BMFace],
                                             uv_layer: BMLayerItem,
                                             sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync) -> bool:
        assert (sync is False)
        return any(any(l[uv_layer].select_edge for l in face.loops) for face in island)

    @staticmethod
    def calc_iter_ex(bm, uv_layer):
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
                        link_face = l.link_loop_radial_next.face
                        if not link_face.tag:  # Skip appended
                            continue

                        for ll in link_face.loops:
                            if not ll.face.tag:
                                continue
                            # If the coordinates of the vertices of adjacent faces on the uv match,
                            # then this is part of the island, and we append face to the list
                            if ll[uv_layer].uv != l[uv_layer].uv:
                                continue
                            # Skip non-manifold
                            if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
                                    (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
                                temp.append(ll.face)
                                ll.face.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_with_markseam_iter_ex(bm, uv_layer):
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
                        link_face = l.link_loop_radial_next.face
                        if not link_face.tag:
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue

                        for ll in link_face.loops:
                            if not ll.face.tag:
                                continue
                            if ll[uv_layer].uv != l[uv_layer].uv:
                                continue
                            if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
                                    (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
                                temp.append(ll.face)
                                ll.face.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_with_markseam_material_iter_ex(bm, uv_layer):
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
                        link_face = l.link_loop_radial_next.face
                        if not link_face.tag:
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue
                        if link_face.material_index != f.material_index:  # Skip if other material
                            continue

                        for ll in link_face.loops:
                            if not ll.face.tag:
                                continue
                            if ll[uv_layer].uv != l[uv_layer].uv:
                                continue
                            if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
                                    (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
                                temp.append(ll.face)
                                ll.face.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

    @staticmethod
    def calc_with_markseam_material_edgeangle_iter_ex(bm, uv_layer, angle, sharp=True):
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
                        link_face = l.link_loop_radial_next.face
                        if not link_face.tag:
                            continue
                        if sharp and not l.edge.smooth:  # Skip by sharp
                            continue
                        if l.edge.calc_face_angle() >= angle:  # Skip by angle
                            continue
                        if l.edge.seam:  # Skip if seam
                            continue
                        if link_face.material_index != f.material_index:  # Skip if other material
                            continue

                        for ll in link_face.loops:
                            if not ll.face.tag:
                                continue
                            if ll[uv_layer].uv != l[uv_layer].uv:
                                continue
                            if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
                                    (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
                                temp.append(ll.face)
                                ll.face.tag = False

                island.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield island
            island = []

class Islands(IslandsBase):
    def __init__(self, islands, bm, uv_layer):
        self.islands = islands
        self.bm: BMesh = bm
        self.uv_layer: BMLayerItem = uv_layer
        # self.obj: 'bpy.types.Object | None' = None
        # self.mesh: 'bpy.types.Mesh | None' = None

    @classmethod
    def calc_selected(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls(FaceIsland([], bm, uv_layer), bm, uv_layer)
        cls.tag_filter_selected(bm, uv_layer, sync)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_full_selected(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls(FaceIsland([], bm, uv_layer), bm, uv_layer)
        cls.tag_filter_visible(bm, sync)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_all_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_partial_selected(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls(FaceIsland([], bm, uv_layer), bm, uv_layer)
        cls.tag_filter_visible(bm, sync)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_partial_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_extended(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        if btypes.PyBMesh.fields(bm).totfacesel == 0:
            return cls(FaceIsland([], bm, uv_layer), bm, uv_layer)
        cls.tag_filter_visible(bm, sync)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer) if cls.island_filter_is_any_face_selected(i, uv_layer, sync)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_visible(cls, bm: BMesh, uv_layer: BMLayerItem, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        cls.tag_filter_visible(bm, sync)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        return cls(islands, bm, uv_layer)

    @classmethod
    def calc_with_hidden(cls, bm: BMesh, uv_layer: BMLayerItem):
        cls.tag_filter_any(bm)
        islands = [FaceIsland(i, bm, uv_layer) for i in cls.calc_iter_ex(bm, uv_layer)]
        return cls(islands, bm, uv_layer)

    def __iter__(self):
        return iter(self.islands)

    def __getitem__(self, idx):
        return self.islands[idx]

    def __bool__(self):
        return bool(self.islands)

    def __str__(self):
        return f'Islands count = {len(self.islands)}'


class FaceIsland:
    def __init__(self, faces: list[BMFace], bm: BMesh, uv_layer: BMLayerItem):
        self.faces = faces
        self.bm: BMesh = bm
        self.uv_layer: BMLayerItem = uv_layer

    def move(self, uv_layer, delta: Vector):
        if umath.vec_isclose_to_zero(delta):
            return False
        for face in self.faces:
            for loop in face.loops:
                loop[uv_layer].uv += delta
        return True

    def rotate(self, angle: float, pivot: Vector):
        """Rotate a list of faces by angle (in radians) around a pivot"""
        if math.isclose(angle, 0, abs_tol=0.0001):
            return False
        rot_matrix = Matrix.Rotation(angle, 2)
        alpha, beta = rot_matrix.row[0]
        diff = Vector(((1 - alpha) * pivot.x - beta * pivot.y, beta * pivot.x + (1 - alpha) * pivot.y))
        for face in self.faces:
            for loop in face.loops:
                uv = loop[self.uv_layer]
                uv.uv = (uv.uv - diff) @ rot_matrix
        return True

    def rotate_simple(self, angle: float):
        """Rotate a list of faces by angle (in radians) around a world center"""
        if math.isclose(angle, 0, abs_tol=0.0001):
            return False
        rot_matrix = Matrix.Rotation(-angle, 2)
        for face in self.faces:
            for loop in face.loops:
                uv = loop[self.uv_layer]
                uv.uv = uv.uv @ rot_matrix
        return True

    def scale(self, scale: Vector, pivot: Vector):
        """Scale a list of faces by pivot"""
        if umath.vec_isclose_to_uniform(scale):
            return False
        diff = pivot - pivot * scale
        for face in self.faces:
            for loop in face.loops:
                uv = loop[self.uv_layer].uv
                uv.uv = (uv.uv * scale) + diff
        return True

    def scale_simple(self, scale: Vector):
        """Scale a list of faces by world center"""
        if umath.vec_isclose_to_uniform(scale):
            return False
        for face in self.faces:
            for loop in face.loops:
                loop[self.uv_layer].uv *= scale
        return True

    def __select_ex(self, state, force, sync):
        if sync or force:
            for face in self.faces:
                face.select = state
                for e in face.edges:
                    e.select = state
                for v in face.verts:
                    v.select = state
        if not sync or force:
            for face in self.faces:
                for l in face.loops:
                    luv = l[self.uv_layer]
                    luv.select = state
                    luv.select_edge = state

    def select(self, force=False, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        self.__select_ex(True, force, sync)

    def deselect(self, force=False, sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync):
        self.__select_ex(False, force, sync)

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, idx):
        return self.faces[idx]

    def __str__(self):
        return f'Faces count = {len(self.faces)}'

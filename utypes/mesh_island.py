# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import typing  # noqa: F401 # pylint:disable=unused-import
from . import island
from . import umesh as _umesh  # noqa: F401 # pylint:disable=unused-import
from .umesh import UMesh
from bmesh.types import *


class MeshIsland:
    def __init__(self, faces: list[BMFace], umesh: UMesh):
        self.faces: list[BMFace] = faces
        self.umesh: UMesh = umesh
        self.value: float | int = -1  # value for different purposes

    def __select_force(self, state):
        for face in self.faces:
            face.select = state
            for e in face.edges:
                e.select = state
            for v in face.verts:
                v.select = state

    def _select_ex(self, state, mode):
        if mode == 'FACE':
            for face in self.faces:
                face.select = state
        elif mode == 'VERT':
            for face in self.faces:
                for v in face.verts:
                    v.select = state
        else:
            for face in self.faces:
                for e in face.edges:
                    e.select = state

    @property
    def select(self):
        raise NotImplementedError()

    @select.setter
    def select(self, state: bool):
        for face in self.faces:
            face.select = state

    def is_full_face_selected(self):
        return all(f.select for f in self)

    def is_full_face_deselected(self):
        return not any(f.select for f in self)

    @property
    def has_all_face_select(self):
        return all(f.select for f in self)

    @property
    def has_any_elem_select(self):
        if self.umesh.elem_mode == 'FACE':
            return any(f.select for f in self)
        else:
            return any(v.select for f in self for v in f.verts)

    def deselect(self, mode, force=False):
        if force:
            return self.__select_force(False)
        self._select_ex(False, mode)

    def tag_selected_faces(self):
        for f in self:
            f.tag = f.select

    def to_adv_island(self) -> island.AdvIsland:
        adv_isl = island.AdvIsland(self.faces, self.umesh)
        adv_isl.value = self.value
        return adv_isl

    def calc_adv_subislands_with_mark_seam(self):
        assert self.faces
        assert self.umesh.uv
        all_added = set()
        adv_islands: list[island.AdvIsland] = []

        uv = self.umesh.uv
        for first_face in self.faces:
            if first_face in all_added:
                continue
            stack = []
            adv_island: set[BMFace] = {first_face}
            parts_of_island = [first_face]
            while parts_of_island:
                for f in parts_of_island:
                    for crn in f.loops:
                        pair_crn = crn.link_loop_radial_prev
                        ff = pair_crn.face
                        if ff in adv_island or ff.hide or crn.edge.seam:
                            continue
                        if ff in all_added:
                            continue
                        if crn[uv].uv == pair_crn.link_loop_next[uv].uv and \
                                crn.link_loop_next[uv].uv == pair_crn[uv].uv:
                            adv_island.add(ff)
                            stack.append(ff)
                parts_of_island = stack
                stack = []
            all_added.update(adv_island)
            adv_islands.append(island.AdvIsland(list(adv_island), self.umesh))

        return island.AdvIslands(adv_islands, self.umesh)

    def calc_selected_edge_corners_iter(self):
        return (crn for f in self for crn in f.loops if crn.edge.select)

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, idx) -> BMFace:
        return self.faces[idx]

    def __len__(self):
        return len(self.faces)

    def __str__(self):
        return f'Faces count = {len(self.faces)}'


class MeshIslandsBase(island.IslandsBaseTagFilterPre, island.IslandsBaseTagFilterPost):

    @classmethod
    def calc_iter_ex(cls, umesh: UMesh):
        mesh_island: 'list[BMFace]' = []

        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:  # Blank list == all faces of the island taken
                for f in stack:
                    for crn in f.loops:  # Running through all the neighboring faces
                        pair_crn = crn.link_loop_radial_prev
                        link_face = pair_crn.face
                        if not link_face.tag:  # Skip appended
                            continue

                        if pair_crn.vert == crn.vert:  # Skip flipped
                            continue

                        temp.append(link_face)
                        link_face.tag = False

                mesh_island.extend(stack)
                stack = temp
                temp = []

            yield mesh_island
            mesh_island = []

    @classmethod
    def calc_iter_non_manifold_ex(cls, umesh: UMesh):
        mesh_island: 'list[BMFace]' = []

        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:  # Blank list == all faces of the island taken
                for f in stack:
                    for crn in f.loops:  # Running through all the neighboring faces
                        pair_crn = crn.link_loop_radial_prev
                        link_face = pair_crn.face
                        if not link_face.tag:  # Skip appended
                            continue

                        temp.append(link_face)
                        link_face.tag = False

                mesh_island.extend(stack)
                stack = temp
                temp = []

            yield mesh_island
            mesh_island = []

    @classmethod
    def calc_by_material_non_manifold_iter_ex(cls, umesh: UMesh):
        mesh_island: 'list[BMFace]' = []

        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:  # Blank list == all faces of the island taken
                for f in stack:
                    mtl_idx = f.material_index
                    for crn in f.loops:
                        link_face = crn.link_loop_radial_next.face
                        if not link_face.tag or mtl_idx != link_face.material_index:  # Skip appended
                            continue

                        temp.append(link_face)
                        link_face.tag = False

                mesh_island.extend(stack)
                stack = temp
                temp = []

            yield mesh_island
            mesh_island = []

    @classmethod
    def calc_by_sharps_non_manifold_iter_ex(cls, umesh: UMesh):
        mesh_island: 'list[BMFace]' = []

        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:  # Blank list == all faces of the island taken
                for f in stack:
                    for crn in f.loops:
                        link_face = crn.link_loop_radial_next.face
                        if not link_face.tag:  # Skip appended
                            continue

                        if not crn.edge.smooth:
                            continue
                        temp.append(link_face)
                        link_face.tag = False

                mesh_island.extend(stack)
                stack = temp
                temp = []

            yield mesh_island
            mesh_island = []

    @classmethod
    def calc_with_markseam_iter_ex(cls, umesh: UMesh):
        isl: list[BMFace] = []
        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:
                for f in stack:
                    for crn in f.loops:
                        shared_crn = crn.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if crn.edge.seam:  # Skip if seam
                            continue
                        if shared_crn.vert == crn.vert:  # Skip flipped
                            continue

                        temp.append(ff)
                        ff.tag = False

                isl.extend(stack)
                stack = temp
                temp = []

            yield isl
            isl = []

    @classmethod
    def calc_with_markseam_non_manifold_iter_ex(cls, umesh: UMesh):
        isl: list[BMFace] = []
        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            stack = [face]
            temp = []

            while stack:
                for f in stack:
                    for crn in f.loops:
                        shared_crn = crn.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if crn.edge.seam:  # Skip if seam
                            continue

                        temp.append(ff)
                        ff.tag = False

                isl.extend(stack)
                stack = temp
                temp = []

            yield isl
            isl = []


class MeshIslands(MeshIslandsBase):
    def __init__(self, islands, umesh: UMesh):
        self.mesh_islands: list[MeshIsland] = islands
        self.umesh: UMesh = umesh
        self.value: float | int = -1  # value for different purposes
        self.sequence = []

    @classmethod
    def calc_all(cls, umesh: UMesh):
        umesh.set_face_tag()
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    @classmethod
    def calc_with_hidden(cls, umesh: UMesh):
        umesh.set_face_tag()
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    @classmethod
    def calc_visible(cls, umesh: UMesh):
        cls.tag_filter_visible(umesh)
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    @classmethod
    def calc_selected(cls, umesh: UMesh):
        if umesh.is_full_face_deselected:
            return cls([], umesh)
        cls.tag_filter_selected(umesh)
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    @classmethod
    def calc_non_selected(cls, umesh: _umesh.UMesh):
        if umesh.sync and umesh.is_full_face_selected:
            return cls([], umesh)

        cls.tag_filter_non_selected(umesh)
        islands = [MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_selected_with_mark_seam(cls, umesh: UMesh):
        if umesh.is_full_face_deselected:
            return cls([], umesh)
        cls.tag_filter_selected(umesh)
        return cls([MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)], umesh)

    @classmethod
    def calc_visible_with_mark_seam(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_with_mark_seam(cls, umesh: _umesh.UMesh):
        cls.tag_filter_visible(umesh)
        if umesh.is_full_face_selected:
            islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        else:
            islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                       if cls.island_filter_is_any_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands"""
        assert umesh.sync
        if umesh.elem_mode == 'FACE':
            if umesh.is_full_face_deselected:
                return cls([], umesh)
        else:
            if umesh.is_full_edge_deselected:
                return cls([], umesh)

        cls.tag_filter_visible(umesh)
        if umesh.is_full_face_selected:
            islands = [MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)]
        else:
            islands = [MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge_with_markseam(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands, with markseam"""
        assert umesh.sync
        if umesh.elem_mode == 'FACE':
            if umesh.is_full_face_deselected:
                return cls([], umesh)
        else:
            if umesh.is_full_edge_deselected:
                return cls([], umesh)

        cls.tag_filter_visible(umesh)
        if umesh.is_full_face_selected:
            islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        else:
            islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                       if cls.island_filter_is_any_edge_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_partial_selected(cls, umesh: _umesh.UMesh):
        assert umesh.sync
        if umesh.is_full_face_deselected:
            cls([], umesh)
        if umesh.sync:
            if umesh.is_full_face_selected:
                cls([], umesh)
        cls.tag_filter_visible(umesh)
        islands = [MeshIsland(i, umesh) for i in cls.calc_iter_ex(
            umesh) if cls.island_filter_is_partial_face_selected(i, umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_partial_selected_with_mark_seam(cls, umesh: _umesh.UMesh):
        assert umesh.sync
        if umesh.is_full_face_deselected:
            return cls([], umesh)
        if umesh.sync:
            if umesh.is_full_face_selected:
                return cls([], umesh)
        cls.tag_filter_visible(umesh)
        islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)
                   if cls.island_filter_is_partial_face_selected(i, umesh)]
        return cls(islands, umesh)

    def indexing(self):
        if sum(len(isl) for isl in self.mesh_islands) != len(self.umesh.bm.faces):
            for f in self.umesh.bm.faces:
                f.index = -1
        for idx, mesh_island in enumerate(self.mesh_islands):
            for face in mesh_island:
                face.index = idx

    def to_adv_islands(self) -> island.AdvIslands:
        adv_islands = []
        for mesh_isl in self:
            adv_isl = island.AdvIsland(mesh_isl.faces, self.umesh)
            adv_isl.value = mesh_isl.value
            adv_islands.append(adv_isl)
        adv_islands_t = island.AdvIslands(adv_islands, self.umesh)
        adv_islands_t.value = self.value
        return adv_islands_t

    def __iter__(self) -> typing.Iterator[MeshIsland]:
        return iter(self.mesh_islands)

    def __getitem__(self, idx) -> MeshIsland:
        return self.mesh_islands[idx]

    def __bool__(self):
        return bool(self.mesh_islands)

    def __len__(self):
        return len(self.mesh_islands)

    def __str__(self):
        return f'Mesh Islands count = {len(self.mesh_islands)}'

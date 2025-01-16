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
from .. import utils

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
        elif mode == 'VERTEX':
            for face in self.faces:
                for v in face.verts:
                    v.select = state
        else:
            for face in self.faces:
                for e in face.edges:
                    e.select = state

    def select(self, mode, force=False):
        if force:
            return self.__select_force(True)
        self._select_ex(True, mode)

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
                    for l in f.loops:  # Running through all the neighboring faces
                        link_face = l.link_loop_radial_next.face
                        if not link_face.tag:  # Skip appended
                            continue

                        for ll in link_face.loops:
                            if not ll.face.tag:
                                continue

                            if ll.vert != l.vert:
                                continue
                            # Skip non-manifold
                            if (l.link_loop_next.vert == ll.link_loop_prev.vert) or \
                                    (ll.link_loop_next.vert == l.link_loop_prev.vert):
                                temp.append(ll.face)
                                ll.face.tag = False

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
                    for l in f.loops:
                        shared_crn = l.link_loop_radial_prev
                        ff = shared_crn.face
                        if not ff.tag:
                            continue
                        if l.edge.seam:  # Skip if seam
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

    @classmethod
    def calc_all(cls, umesh: UMesh):
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
    def calc_non_selected_with_mark_seam(cls, umesh: _umesh.UMesh):
        if umesh.sync and umesh.is_full_face_selected:
            return cls([], umesh)

        cls.tag_filter_non_selected(umesh)
        islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh)]
        return cls(islands, umesh)

    @classmethod
    def calc_extended_any_edge(cls, umesh: _umesh.UMesh):
        """Calc any edges selected islands"""
        assert umesh.sync
        if utils.get_select_mode_mesh() == 'FACE':
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
        if utils.get_select_mode_mesh() == 'FACE':
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
        islands = [MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh) if cls.island_filter_is_partial_face_selected(i, umesh)]
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
        islands = [MeshIsland(i, umesh) for i in cls.calc_with_markseam_iter_ex(umesh) if cls.island_filter_is_partial_face_selected(i, umesh)]
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

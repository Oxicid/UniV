# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing  # noqa: F401
from . import island
from ..utils import UMesh
from bmesh.types import *

class MeshIsland:
    def __init__(self, faces: list[BMFace], umesh: UMesh):
        self.faces: list[BMFace] = faces
        self.umesh: UMesh = umesh

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

    def to_adv_island(self) -> island.AdvIsland:
        return island.AdvIsland(self.faces, self.umesh.bm, self.umesh.uv_layer)

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, idx) -> BMFace:
        return self.faces[idx]

    def __len__(self):
        return len(self.faces)

    def __str__(self):
        return f'Faces count = {len(self.faces)}'


class MeshIslandsBase:
    @staticmethod
    def tag_filter_visible(umesh: UMesh):
        for face in umesh.bm.faces:
            face.tag = not face.hide

    @classmethod
    def calc_iter_ex(cls, umesh: UMesh):
        mesh_island: 'list[BMFace]' = []

        for face in umesh.bm.faces:
            if not face.tag:
                continue
            face.tag = False

            parts_of_island = [face]
            temp = []

            while parts_of_island:  # Blank list == all faces of the island taken
                for f in parts_of_island:
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

                mesh_island.extend(parts_of_island)
                parts_of_island = temp
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
                        temp.append(ff)
                        ff.tag = False

                isl.extend(parts_of_island)
                parts_of_island = temp
                temp = []

            yield isl
            isl = []


class MeshIslands(MeshIslandsBase):
    def __init__(self, islands, umesh: UMesh):
        self.mesh_islands: list[MeshIsland] = islands
        self.umesh: UMesh = umesh

    @classmethod
    def calc_visible(cls, umesh: UMesh):
        cls.tag_filter_visible(umesh)
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    @classmethod
    def calc_selected(cls, umesh: UMesh):
        umesh.tag_selected_faces()
        return cls([MeshIsland(i, umesh) for i in cls.calc_iter_ex(umesh)], umesh)

    def to_adv_islands(self) -> island.AdvIslands:
        adv_islands = []
        for mesh_isl in self:
            adv_islands.append(island.AdvIsland(mesh_isl.faces, self.umesh.bm, self.umesh.uv_layer))
        return island.AdvIslands(adv_islands, self.umesh.bm, self.umesh.uv_layer)

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

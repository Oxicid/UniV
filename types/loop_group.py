# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import typing
from mathutils import Vector
from collections import defaultdict
from itertools import chain
from bmesh.types import BMLoop
from ..utils import prev_disc, linked_crn_uv_by_crn_tag_unordered_included, linked_crn_uv, vec_isclose_to_zero
from . import umesh as _umesh
from . import bbox
from .. import utils

class LoopGroup:
    def __init__(self, umesh: _umesh.UMesh):
        self.umesh: _umesh.UMesh = umesh
        self.corners: list[BMLoop] = []
        self.tag = True
        self.dirt = False
        self._length_uv: float | None = None
        self._length_3d: float | None = None
        self.weights: list[float] | None = None
        self.chain_linked_corners: list[list[BMLoop]] = []
        self.chain_linked_corners_mask: list[bool] = []

    def is_cyclic_vert(self):
        if len(self.corners) > 1:
            return self.corners[-1].link_loop_next.vert == self.corners[0].vert

    def is_cyclic_crn(self):
        if len(self.corners) > 1:
            return self.corners[-1].link_loop_next == self.corners[0]

    @property
    def is_cyclic(self):
        crn_a = self.corners[0]
        crn_b = self.corners[-1].link_loop_next
        return crn_a.vert == crn_b.vert and crn_a[self.umesh.uv].uv == crn_b[self.umesh.uv].uv

    def calc_loop_group(self, crn):
        crn.tag = False
        group = [crn]
        while True:
            if next_crn := self.next_walk_boundary(group[-1]):
                group.append(next_crn)
            else:
                break
        while True:
            if prev_crn := self.prev_walk_boundary(group[0]):
                group.insert(0, prev_crn)
            else:
                break
        self.corners = group
        return self

    def next_walk_boundary(self, crn):
        crn_next = crn.link_loop_next
        if crn_next.tag:
            crn_next.tag = False
            return crn_next

        uv = self.umesh.uv
        bm_iter = crn_next
        while True:
            if (bm_iter := prev_disc(bm_iter)) == crn_next:
                break
            if bm_iter.tag and crn_next[uv].uv == bm_iter[uv].uv:
                bm_iter.tag = False
                return bm_iter

    def prev_walk_boundary(self, crn):
        crn_prev = crn.link_loop_prev
        if crn_prev.tag:
            crn_prev.tag = False
            return crn_prev

        uv = self.umesh.uv
        bm_iter = crn
        while True:
            if (bm_iter := prev_disc(bm_iter)) == crn:
                break
            if bm_iter.link_loop_prev.tag and crn[uv].uv == bm_iter[uv].uv:
                bm_iter.link_loop_prev.tag = False
                return bm_iter.link_loop_prev

    def is_boundary_sync(self, crn):
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            return True
        if shared_crn.face.hide:  # Change
            return True
        uv = self.umesh.uv
        return not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)

    def is_boundary(self, crn):
        shared_crn = crn.link_loop_radial_prev  # We get a clockwise corner, but linked to the end of the current corner
        if shared_crn == crn:
            return True
        if not shared_crn.face.select:  # Change
            return True
        uv = self.umesh.uv
        return not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)

    def calc_shared_group(self) -> 'typing.Self':
        shared_group = []
        for crn in reversed(self.corners):
            shared_group.append(crn.link_loop_radial_prev)
        lg = LoopGroup(self.umesh)
        lg.corners = shared_group
        return lg

    def boundary_tag_by_face_index(self, crn: BMLoop):
        uv = self.umesh.uv
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return

        # if shared_crn.face.index in (-1, crn.face.index):
        if shared_crn.face.index == -1:
            crn.tag = False
            return

        crn.tag = not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)

    def boundary_tag(self, crn: BMLoop):
        uv = self.umesh.uv
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return
        if not crn[uv].select_edge:
            crn.tag = False
            return
        if not shared_crn.face.select:  # Change
            crn.tag = False
            return
        crn.tag = not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)

    def boundary_tag_sync(self, crn: BMLoop):
        uv = self.umesh.uv
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return
        if not crn.edge.select:
            crn.tag = False
            return
        if shared_crn.face.hide:  # Change
            crn.tag = False
            return
        crn.tag = not (crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv)

    @staticmethod
    def calc_island_index_for_stitch(island) -> defaultdict[int, list[BMLoop]]:
        islands_for_stitch = defaultdict(list)
        for f in island:
            for crn in f.loops:
                if crn.tag:
                    crn.tag = False
                    shared_crn = crn.link_loop_radial_prev
                    islands_for_stitch[shared_crn.face.index].append(crn)
        return islands_for_stitch

    def calc_signed_face_area(self):
        uv = self.umesh.uv
        return sum(utils.calc_signed_face_area_uv(crn.face, uv) for crn in self)

    def calc_signed_corners_area(self):
        uv = self.umesh.uv
        area = 0.0
        first_crn_co = self.corners[-1][uv].uv
        for crn in self.corners:
            next_crn_co = crn[uv].uv
            area += first_crn_co.cross(next_crn_co)
            first_crn_co = next_crn_co
        return area * 0.5

    def tagging(self, island):
        func: typing.Callable = self.boundary_tag_sync if island.umesh.sync else self.boundary_tag
        for f in island:
            for crn in f.loops:
                func(crn)

    def calc_first(self, island, selected=True):
        if selected:
            self.tagging(island)
        else:
            for f__ in island:
                for crn__ in f__.loops:
                    self.boundary_tag_by_face_index(crn__)

        indexes = self.calc_island_index_for_stitch(island)
        for k, corner_edges in indexes.items():
            for _crn in corner_edges:
                _crn.tag = True

            crn_edges = (__crn for __crn in corner_edges if __crn.tag)

            for crn_edge in crn_edges:
                loop_group = self.calc_loop_group(crn_edge)

                yield loop_group

                if loop_group.tag:
                    if len(loop_group) != len(corner_edges):
                        for _crn in corner_edges:
                            _crn.tag = False
                    break

    def set_tag(self, state=True):
        for g in self.corners:
            g.tag = state

    def has_non_sync_crn(self):
        assert utils.sync()
        count_non_shared = 0
        uv = self.umesh.uv
        for crn in self.corners:
            shared_crn = crn.link_loop_radial_prev
            if shared_crn == crn:
                count_non_shared += 1
                continue
            if not shared_crn.tag:
                count_non_shared += 1
                continue
            if crn[uv].uv == shared_crn.link_loop_next[uv].uv and crn.link_loop_next[uv].uv == shared_crn[uv].uv:
                return True
        return count_non_shared == len(self.corners)

    def has_sync_crn(self):
        """ Need tagging and indexing"""
        assert utils.sync()
        for crn in self.corners:
            shared_crn = crn.link_loop_radial_prev
            if shared_crn == crn:
                continue
            elif not shared_crn.tag:
                continue
            elif crn.index == shared_crn.index:
                continue
            # elif crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv or crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv:
            #     continue
            return True
        return False

    def move(self, delta: Vector) -> bool:
        if vec_isclose_to_zero(delta):
            return False
        uv = self.umesh.uv
        for loop in self.corners:
            loop[uv].uv += delta
        return True

    def set_position(self, to: Vector, _from: Vector):
        return self.move(to - _from)

    def calc_bbox(self):
        return bbox.BBox.calc_bbox_uv_corners(self.corners, self.umesh.uv)

    def calc_length_uv(self, aspect: float = 1.0):
        uv = self.umesh.uv
        length = 0.0
        if aspect == 1.0:
            for crn in self:
                length += (crn[uv].uv - crn.link_loop_next[uv].uv).length
        else:
            for crn in self:
                vec = crn[uv].uv - crn.link_loop_next[uv].uv
                vec.x *= aspect
                length += vec.length
        self._length_uv = length
        return length

    def calc_length_3d(self):
        length = 0.0
        for crn in self:
            length += crn.edge.calc_length()
        self._length_3d = length
        return length

    @property
    def length_uv(self):
        if self._length_uv is None:
            return self.calc_length_uv()
        return self._length_uv

    @length_uv.setter
    def length_uv(self, v):
        self._length_uv = v

    @property
    def length_3d(self):
        if self._length_3d is None:
            return self.calc_length_3d()
        return self._length_3d

    @length_3d.setter
    def length_3d(self, v):
        self._length_3d = v

    def get_vector(self):
        uv = self.umesh.uv
        vec = self[-1].link_loop_next[uv].uv - self[0][uv].uv
        if vec == Vector((0.0, 0.0)):
            for crn in self:
                vec = crn.link_loop_next[uv].uv - crn[uv].uv
                if vec != Vector((0.0, 0.0)):
                    return vec
        return vec

    def set_pins(self, state=True):
        assert self.chain_linked_corners
        uv = self.umesh.uv
        for linked_groups in self.chain_linked_corners:
            for crn in linked_groups:
                crn[uv].pin_uv = state

    def set_pins_by_mask(self):
        assert self.chain_linked_corners
        uv = self.umesh.uv
        for linked_groups, state in zip(self.chain_linked_corners, self.chain_linked_corners_mask):
            for crn in linked_groups:
                crn[uv].pin_uv = state

    def calc_chain_linked_corners_mask_from_short_path(self, short_path):
        assert self.chain_linked_corners
        mask = []
        uv = self.umesh.uv
        path_corners = set(short_path)
        for corners in self.chain_linked_corners:
            mask.append(corners[0][uv].pin_uv or any((l_crn in path_corners) for l_crn in corners))
        self.chain_linked_corners_mask = mask

    def calc_chain_linked_corners(self):
        uv = self.umesh.uv
        for crn in chain(self, [self[-1].link_loop_next]):
            linked = linked_crn_uv(crn, uv)
            linked.insert(0, crn)
            self.chain_linked_corners.append(linked)

    def distribute(self, start, end):
        assert self.chain_linked_corners
        uv = self.umesh.uv
        if self.weights is None:
            self.weights = [crn.edge.calc_length() for crn in self]

        points = utils.weighted_linear_space(start, end, self.weights)
        for corners, co in zip(self.chain_linked_corners, points):
            for l_crn in corners:
                l_crn[uv].uv = co

    @classmethod
    def calc_dirt_loop_groups(cls, umesh):
        uv = umesh.uv
        # Tagging
        if utils.sync():
            assert utils.get_select_mode_mesh() != 'FACE'
            umesh.tag_selected_corners()
        else:
            umesh.tag_selected_corners()

        sel_loops = (l for f in umesh.bm.faces for l in f.loops if l.tag)

        groups: list[cls] = []
        for crn_ in sel_loops:
            group = []
            temp_group = [crn_]
            while True:
                temp = []
                for sel in temp_group:
                    it1 = linked_crn_uv_by_crn_tag_unordered_included(sel, uv)
                    it2 = linked_crn_uv_by_crn_tag_unordered_included(sel.link_loop_next, uv)
                    for l in chain(it1, it2):
                        if l.tag:
                            l.tag = False
                            temp.append(l)

                        prev = l.link_loop_prev
                        if prev.tag:
                            prev.tag = False
                            temp.append(prev)
                if not temp:
                    break

                temp_group = temp
                group.extend(temp)
            lg = cls(umesh)
            lg.corners = group
            lg.dirt = True
            groups.append(lg)
        return LoopGroups(groups, umesh)

    def extend_from_linked(self):
        self.set_tag(False)

        if utils.sync():
            # Need tag_visible_corners before use
            move_corners = []
            uv = self.umesh.uv
            for crn in self:
                move_corners.extend(utils.linked_crn_vert_uv_for_transform(crn, uv))
            self.corners.extend(move_corners)
        else:
            move_corners = []
            uv = self.umesh.uv
            for crn in self:
                linked_corners = utils.linked_crn_uv_by_crn_tag_unordered_included(crn, uv)  # TODO: Add linked_crn_uv_by_tag_c by island
                move_corners.extend(linked_corners)
                for crn_ in linked_corners:
                    crn_.tag = False

                linked_corners = utils.linked_crn_uv_by_crn_tag_unordered_included(crn.link_loop_next, uv)
                move_corners.extend(linked_corners)
                for crn_ in linked_corners:
                    crn_.tag = False

            self.corners.extend(move_corners)

    def __iter__(self):
        return iter(self.corners)

    def __getitem__(self, idx) -> BMLoop:
        return self.corners[idx]

    def __len__(self):
        return len(self.corners)

    def __bool__(self):
        return bool(self.corners)

    def __str__(self):
        return f'Corner Edge count = {len(self.corners)}'

class LoopGroups:
    def __init__(self, loop_groups, umesh):
        self.loop_groups: list[LoopGroup] = loop_groups
        self.umesh: _umesh.UMesh | None = umesh
        self.tag = True

    @classmethod
    def calc_by_boundary_crn_tags(cls, isl):
        """Warning: Need uninterrupted tagging by boundary loops"""
        uv = isl.umesh.uv
        loop_groups = []
        for crn in isl.iter_corners_by_tag():
            crn.tag = False
            group = [crn]
            temp_crn: BMLoop | None = crn
            while temp_crn:
                next_crn = temp_crn.link_loop_next
                if next_crn.tag:
                    next_crn.tag = False
                    temp_crn = next_crn
                    group.append(next_crn)
                    continue
                for linked_crn in reversed(linked_crn_uv(next_crn, uv)):
                    if linked_crn.tag:
                        linked_crn.tag = False
                        temp_crn = linked_crn
                        group.append(linked_crn)
                        break
                else:
                    temp_crn = None

            lg = LoopGroup(isl.umesh)
            lg.corners = group
            loop_groups.append(lg)
        return cls(loop_groups, isl.umesh)

    @classmethod
    def calc_by_boundary_crn_tags_v2(cls, isl):
        """Warning: Need tagging by boundary loops"""
        uv = isl.umesh.uv
        loop_groups = []
        for crn in isl.iter_corners_by_tag():
            crn.tag = False
            group = [crn]
            temp_crn: BMLoop | None = crn
            while temp_crn:  # forward
                next_crn = temp_crn.link_loop_next
                if next_crn.tag:
                    next_crn.tag = False
                    temp_crn = next_crn
                    group.append(next_crn)
                    continue

                # TODO: Replace linked_crn_uv with linked_crn_to_vert_pair_iter to avoid non-manifold uv vert links
                for linked_crn in reversed(linked_crn_uv(next_crn, uv)):
                    if linked_crn.tag:
                        linked_crn.tag = False
                        temp_crn = linked_crn
                        group.append(linked_crn)
                        break
                else:
                    temp_crn = None

            temp_crn = crn
            while temp_crn:  # backward
                if temp_crn.link_loop_prev.tag:
                    temp_crn = temp_crn.link_loop_prev
                    temp_crn.tag = False
                    group.insert(0, temp_crn)
                    continue

                for linked_crn in reversed(linked_crn_uv(temp_crn, uv)):
                    linked_crn_prev = linked_crn.link_loop_prev
                    if linked_crn_prev.tag:
                        temp_crn = linked_crn_prev
                        temp_crn.tag = False
                        group.insert(0, temp_crn)
                        break
                else:
                    temp_crn = None

            lg = LoopGroup(isl.umesh)
            lg.corners = group
            loop_groups.append(lg)
        return cls(loop_groups, isl.umesh)

    def indexing(self, _=None):
        for f in self.umesh.bm.faces:
            for crn in f.loops:
                crn.index = -1

        for idx, lg in enumerate(self.loop_groups):
            for crn in lg:
                crn.index = idx

    def set_position(self, to: Vector, _from: Vector):
        return bool(sum(lg.set_position(to, _from) for lg in self.loop_groups))

    def move(self, delta: Vector):
        return bool(sum(lg.move(delta) for lg in self.loop_groups))

    def set_pins(self, state=True):
        for lg in self:
            lg.set_pins(state)

    def set_pins_by_mask(self):
        for lg in self:
            lg.set_pins_by_mask()

    def __iter__(self) -> typing.Iterator[LoopGroup]:
        return iter(self.loop_groups)

    def __getitem__(self, idx) -> LoopGroup:
        return self.loop_groups[idx]

    def __bool__(self):
        return bool(self.loop_groups)

    def __len__(self):
        return len(self.loop_groups)

    def __str__(self):
        return f'Loop Groups count = {len(self.loop_groups)}'

class UnionLoopGroup(LoopGroups):
    def __init__(self, loop_groups: list[LoopGroup]):
        super().__init__(loop_groups, None)

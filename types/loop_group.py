# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import typing
from mathutils import Vector
from collections import defaultdict, deque
from itertools import chain
from bmesh.types import BMLoop
from ..utils import prev_disc, linked_crn_uv_by_crn_tag_unordered_included, linked_crn_uv, vec_isclose_to_zero
from math import pi
from . import umesh as _umesh
from . import bbox
from .. import utils

class LoopGroup:
    def __init__(self, umesh: _umesh.UMesh):
        self.umesh: _umesh.UMesh = umesh
        self.corners: list[BMLoop] = []
        self.tag = True
        self.dirt = False
        self.is_shared = False
        self.is_flipped_3d = False
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

    def calc_shared_group_for_stitch(self) -> 'typing.Self':
        shared_group = []
        is_flipped = self._is_flipped_3d
        if is_flipped:
            for crn in self.corners:
                shared_group.append(crn.link_loop_radial_prev)
        else:
            for crn in self.corners:
                shared_group.append(crn.link_loop_radial_prev.link_loop_next)
        lg = LoopGroup(self.umesh)
        lg.is_shared = True
        lg.is_flipped_3d = is_flipped
        lg.corners = shared_group
        return lg

    def calc_begin_end_pt(self):
        uv = self.umesh.uv
        if self.is_shared:
            if self.is_flipped_3d:
                return self[0][uv].uv, self[-1].link_loop_next[uv].uv
            else:
                return self[0][uv].uv, self[-1].link_loop_prev[uv].uv
        else:
            return self[0][uv].uv, self[-1].link_loop_next[uv].uv

    @property
    def _is_flipped_3d(self):
        assert not self.is_shared
        pair = self[0].link_loop_radial_prev
        return pair.vert == self[0].vert

    def copy_coords_from_ref(self, ref, clean_seams):
        uv = self.umesh.uv
        for ref_crn, trans_crn in zip(ref, self):
            if clean_seams:
                ref_crn.edge.seam = False
            ref_co = ref_crn[uv].uv
            # TODO: Implement linked_crn_to_vert_by_idx_pair_with_seam
            for trans_crn_linked in utils.linked_crn_to_vert_pair_with_seam(trans_crn, uv, self.umesh.sync):
                trans_crn_linked[uv].uv = ref_co
            trans_crn[uv].uv = ref_co

        ref_co = ref[-1].link_loop_next[uv].uv
        end_crn = self[-1].link_loop_next if self.is_flipped_3d else self[-1].link_loop_prev

        for trans_crn_linked in utils.linked_crn_to_vert_pair_with_seam(end_crn, uv, self.umesh.sync):
            trans_crn_linked[uv].uv = ref_co
        end_crn[uv].uv = ref_co

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

class AdvCorner:
    def __init__(self, crn: BMLoop, uv, invert=False):
        self.crn = crn
        self.invert: bool = invert
        self.uv = uv
        self._vec = None
        self._is_pair: bool | None = None

    @property
    def vec(self):
        if not self._vec:
            if self.invert:
                _vec = self.crn.link_loop_prev[self.uv].uv - self.crn[self.uv].uv
            else:
                _vec = self.crn.link_loop_next[self.uv].uv - self.crn[self.uv].uv
            _vec.normalize()
            self._vec = _vec
            return _vec
        return self._vec

    @vec.setter
    def vec(self, v):
        self._vec = v

    def angle(self, other: 'typing.Self', max_angle: float):
        return self.vec.angle(other.vec, max_angle)  # noqa

    @property
    def is_pair(self):
        if self._is_pair is None:
            if self.invert:
                pair = self.crn.link_loop_prev.link_loop_radial_prev
                self._is_pair = utils.is_pair_by_idx(self.crn.link_loop_prev, pair, self.uv)
            else:
                pair = self.crn.link_loop_radial_prev
                self._is_pair = utils.is_pair_by_idx(self.crn, pair, self.uv)
        return self._is_pair

    @is_pair.setter
    def is_pair(self, v: bool):
        self._is_pair = v

    @property
    def vert(self):
        return self.crn.vert

    @property
    def length(self):
        if self.invert:
            return (self.crn[self.uv].uv - self.crn.link_loop_prev[self.uv].uv).length
        return (self.crn[self.uv].uv - self.crn.link_loop_next[self.uv].uv).length

    @property
    def angle_from_cardinal(self):
        vec = self.vec
        card_vec = utils.vec_to_cardinal(vec)
        angle = vec.angle(card_vec, 0)
        return min(angle, pi-angle)

    @property
    def next(self):
        # NOTE: Need face indexing
        if self.invert:
            crn_prev = self.crn.link_loop_prev
            crn_prev_prev = crn_prev.link_loop_prev
            pair_prev_prev = crn_prev_prev.link_loop_radial_prev
            if utils.is_pair_by_idx(crn_prev_prev, pair_prev_prev, self.uv):
                return AdvCorner(pair_prev_prev, self.uv)
            else:
                return AdvCorner(crn_prev, self.uv, invert=True)
        else:
            return AdvCorner(self.crn.link_loop_next, self.uv)

    @property
    def prev(self):
        # NOTE: Need face indexing
        if self.invert:
            pair = self.crn.link_loop_radial_prev
            if utils.is_pair_by_idx(self.crn, pair, self.uv):
                return AdvCorner(pair, self.uv)
            else:
                return AdvCorner(self.crn.link_loop_next, self.uv, invert=True)
        else:
            return AdvCorner(self.crn.link_loop_prev, self.uv)

    def toggle_dir(self):
        if self.is_pair:
            self.crn = self.crn.link_loop_radial_prev
        else:
            if self.invert:
                self.invert = False
                self.crn = self.crn.link_loop_prev
            else:
                self.invert = True
                self.crn = self.crn.link_loop_next
        if self._vec is not None:
            self._vec *= -1
        return self

    @property
    def curr_pt(self):
        return self.crn[self.uv].uv

    @property
    def next_pt(self):
        if self.invert:
            return self.crn.link_loop_prev[self.uv].uv
        else:
            return self.crn.link_loop_next[self.uv].uv

    # def normalize(self):
    #     if self.invert:
    #         if utils.is_pair_by_idx(self.crn, self.crn.link_loop_radial_prev, self.uv):
    #             self.invert = False
    #             self.crn = self.crn.link_loop_radial_prev
    #             return True
    #     return False

    def copy(self):
        adv_crn = AdvCorner(self.crn, self.uv, self.invert)
        adv_crn._is_pair = self.is_pair
        if self._vec:
            adv_crn._vec = self._vec.copy()
        else:
            adv_crn._vec = None
        return adv_crn

    def __hash__(self):
        return hash(self.crn)

    def __eq__(self, other):
        return self.crn == other.crn and self.invert == other.invert


class Segment:
    def __init__(self, seg, umesh):
        self.seg: list[AdvCorner] = seg
        self.umesh = umesh
        self.tag = True

        self.angles_seq: list[float] = []
        self.lengths_seq: list[float] = []
        self.chain_linked_corners: list[list[BMLoop]] = []

        self._length: float | utils.NoInit = utils.NoInit()
        self._weight_angle: float | utils.NoInit = utils.NoInit()
        self.value: float | utils.NoInit = utils.NoInit()

        self.is_start_lock: bool = False
        self.is_end_lock: bool = False

    def calc_chain_linked_corners(self):
        uv = self.umesh.uv
        for adv_crn in self.seg:
            linked = utils.linked_crn_uv_by_idx_unordered_included(adv_crn.crn, uv)
            self.chain_linked_corners.append(linked)

        if self.is_start_lock:
            del self.chain_linked_corners[0]

        if not self.is_end_lock:
            linked = utils.linked_crn_uv_by_idx_unordered_included(self.seg[-1].next.crn, uv)
            self.chain_linked_corners.append(linked)

    def reverse(self):
        self.seg = [crn.toggle_dir() for crn in reversed(self.seg)]
        self.angles_seq.reverse()
        self.lengths_seq.reverse()

        temp_lock = self.is_start_lock
        self.is_start_lock = self.is_end_lock
        self.is_end_lock = temp_lock

    def join_from_end(self, other: 'Segment'):
        assert self.end_vert == other.start_vert
        assert self.end_co == other.start_co
        assert not self.is_end_lock
        assert not other.is_start_lock

        self.seg.extend(other.seg)
        self.angles_seq.extend(other.angles_seq)
        self.lengths_seq.extend(other.lengths_seq)

        self._length += other._length
        self._weight_angle = utils.NoInit()

        self.is_end_lock = other.is_end_lock
        other.tag = False

    def calc_angles_from_card_dir(self):
        for adv_crn in self:
            self.angles_seq.append(adv_crn.angle_from_cardinal)

    def calc_lengths(self):
        for adv_crn in self:
            self.lengths_seq.append(adv_crn.length)

    @property
    def length(self):
        if isinstance(self._length, utils.NoInit):
            if not self.lengths_seq:
                self.calc_lengths()
            self._length = sum(self.lengths_seq)
        return self._length

    @length.setter
    def length(self, value: float):
        self._length = value

    @property
    def weight_angle(self):
        if isinstance(self._weight_angle, utils.NoInit):
            if not self.angles_seq:
                self.calc_angles_from_card_dir()

            if not self.lengths_seq:
                self.calc_lengths()
            try:
                self._weight_angle = np.average(self.angles_seq, weights=self.lengths_seq)
            except ZeroDivisionError:
                self._weight_angle = 0


        return self._weight_angle

    @weight_angle.setter
    def weight_angle(self, value: float):
        self._weight_angle = value

    @property
    def start(self):
        return self.start_vert, self.start_co

    @property
    def end(self):
        return self.end_vert, self.end_co

    @property
    def start_vert(self):
        return self.seg[0].vert

    @property
    def end_vert(self):
        adv_crn = self.seg[-1]
        if adv_crn.invert:
            return adv_crn.crn.link_loop_prev.vert
        else:
            return adv_crn.crn.link_loop_next.vert

    @property
    def start_co(self):
        adv_crn = self.seg[0]
        return adv_crn.crn[adv_crn.uv].uv

    @property
    def end_co(self):
        adv_crn = self.seg[-1]
        if adv_crn.invert:
            return adv_crn.crn.link_loop_prev[adv_crn.uv].uv
        else:
            return adv_crn.crn.link_loop_next[adv_crn.uv].uv

    @property
    def is_circular(self):
        return self.start_vert == self.end_vert and self.start_co == self.end_co

    @staticmethod
    def fix_zero_vec(seg):
        pass

    def break_by_cardinal_dir(self):
        from ..utils import vec_to_cardinal
        if len(self) <= 1:
            return Segments([self], self.umesh)
        seg = list(self.seg)

        break_indexes = []
        if self.is_circular:
            prev_cardinal = vec_to_cardinal(seg[-1].vec)
            for i in range(len(seg)):
                curr_cardinal = vec_to_cardinal(seg[i].vec)
                if curr_cardinal != prev_cardinal:
                    break_indexes.append(i)
                prev_cardinal = curr_cardinal

            slices = []
            for i in range(len(break_indexes)):
                start = break_indexes[i]
                end = break_indexes[(i + 1) % len(break_indexes)]
                if start < end:
                    slices.append(seg[start:end])
                else:
                    slices.append(seg[start:] + seg[:end])
        else:

            for i in range(len(seg)-1):
                prev_cardinal = vec_to_cardinal(seg[i].vec)
                curr_cardinal = vec_to_cardinal(seg[i+1].vec)
                if curr_cardinal != prev_cardinal:
                    break_indexes.append(i+1)

            full_breaks = [0] + break_indexes + [len(seg)]

            slices = [
                seg[full_breaks[i]:full_breaks[i + 1]]
                for i in range(len(full_breaks) - 1)
            ]

        slices = [Segment(seg, self.umesh) for seg in slices]

        if len(slices) > 1:
            if self.is_circular:
                for seg in slices:
                    seg.is_end_lock = True
                    seg.is_start_lock = True
            else:
                last_idx = len(slices) - 1
                for idx, seg in enumerate(slices):
                    if idx != 0:
                        seg.is_start_lock = True

                    if idx != last_idx:
                        seg.is_end_lock = True

        return Segments(slices, self.umesh)

    def __iter__(self):
        return iter(self.seg)

    def __getitem__(self, idx) -> AdvCorner:
        return self.seg[idx]

    def __len__(self):
        return len(self.seg)

    def __bool__(self):
        return bool(self.seg)

    def __str__(self):
        return f'Segment. Adv Corner count = {len(self.seg)}, start lock = {self.is_start_lock}, end lock {self.is_end_lock}'


class Segments:
    def __init__(self, segments, umesh):
        self.segments: typing.Sequence[Segment] | list[Segment] = segments
        self.umesh = umesh

    @classmethod
    def calc_by_tags(cls):
        pass

    @classmethod
    def from_tagged_corners(cls, to_select_corns: list[BMLoop], umesh):
        # NOTE: Need indexing islands and tagged corners

        uv = umesh.uv
        is_pair = utils.is_pair_by_idx
        appended = set()
        segments = []

        for crn in to_select_corns:
            seg = deque()
            first_crn = AdvCorner(crn, uv)
            if first_crn in appended:
                continue

            appended.add(first_crn)
            seg.append(first_crn)

            if first_crn.is_pair:
                pair = AdvCorner(crn.link_loop_radial_prev, uv)
                pair.is_pair = True
                appended.add(pair)
            else:
                pair = AdvCorner(crn.link_loop_next, uv, invert=True)
                pair.is_pair = False
                appended.add(pair)

            # Forward Grow
            while True:
                lead = seg[-1]
                if lead.invert:
                    next_check = lead.crn
                else:
                    next_check = lead.crn.link_loop_next

                linked = utils.linked_crn_uv_by_idx_unordered_included(next_check, uv)

                count = 0
                filtered = []
                for crn_l in linked:
                    # Next grow
                    if crn_l.tag:
                        next_grow = AdvCorner(crn_l, uv)
                        count += 1
                        if next_grow not in appended:
                            next_grow.is_pair = is_pair(crn_l, crn_l.link_loop_radial_prev, uv)
                            filtered.append(next_grow)

                    # Here we skip pair edges, as they are processed in the previous condition.
                    prev = crn_l.link_loop_prev
                    if prev.tag and not is_pair(prev, prev.link_loop_radial_prev, uv):
                        prev_grow = AdvCorner(crn_l, uv, invert=True)
                        count += 1
                        if prev_grow not in appended:
                            filtered.append(prev_grow)

                if count >= 3:
                    break

                if len(filtered) == 1:
                    next_elem = filtered[0]
                    if next_elem in appended:
                        break

                    seg.append(next_elem)

                    # Add possible CrnGrow variants so that we don't go through them again
                    appended.add(next_elem)
                    if next_elem.invert:
                        appended.add(AdvCorner(next_elem.crn.link_loop_prev, uv))
                    else:
                        # The pair doesn't have an invert option, so we do without them
                        if is_pair(next_elem.crn, next_elem.crn.link_loop_radial_prev, uv):
                            appended.add(AdvCorner(next_elem.crn.link_loop_radial_prev, uv))
                        else:
                            appended.add(AdvCorner(next_elem.crn.link_loop_next, uv, invert=True))
                else:
                    break

            # Backward Grow
            while True:
                lead = seg[0]
                next_check = lead.crn
                linked = utils.linked_crn_uv_by_idx_unordered_included(next_check, uv)

                count = 0
                filtered = []
                for crn_l in linked:
                    # Next grow
                    if crn_l.tag:
                        if is_pair(crn_l, crn_l.link_loop_radial_prev, uv):
                            next_grow = AdvCorner(crn_l.link_loop_radial_prev, uv)
                            next_grow.is_pair = True
                        else:
                            next_grow = AdvCorner(crn_l.link_loop_next, uv, invert=True)
                            next_grow.is_pair = False
                        count += 1
                        if next_grow not in appended:
                            filtered.append(next_grow)

                    prev = crn_l.link_loop_prev
                    if prev.tag and not is_pair(prev, prev.link_loop_radial_prev, uv):
                        prev_grow = AdvCorner(prev, uv)
                        prev_grow.is_pair = False
                        count += 1
                        if prev_grow not in appended:
                            filtered.append(prev_grow)

                if count >= 3:
                    break

                if len(filtered) == 1:
                    next_elem = filtered[0]
                    if next_elem in appended:
                        break

                    seg.appendleft(next_elem)
                    appended.add(next_elem)
                    if next_elem.invert:
                        appended.add(AdvCorner(next_elem.crn.link_loop_prev, uv))
                    else:
                        if next_elem.is_pair:
                            appended.add(AdvCorner(next_elem.crn.link_loop_radial_prev, uv))
                        else:
                            appended.add(AdvCorner(next_elem.crn.link_loop_next, uv, invert=True))
                else:
                    break

            segments.append(Segment(seg, umesh))
        return cls(segments, umesh)

    def break_by_cardinal_dir(self):
        segments = []
        for seg in self:
            segments.extend(seg.break_by_cardinal_dir().segments)
        return Segments(segments, self.umesh)

    def __iter__(self) -> typing.Iterator[Segment]:
        return iter(self.segments)

    def __getitem__(self, idx) -> Segment:
        return self.segments[idx]

    def __len__(self):
        return len(self.segments)

    def __bool__(self):
        return bool(self.segments)

    def __str__(self):
        return f'Segments. Segments count = {len(self.segments)}'
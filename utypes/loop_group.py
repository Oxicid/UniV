# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import typing
import numpy as np

from mathutils import Vector
from collections import deque
from itertools import chain
from bmesh.types import BMLoop

from math import pi
from . import umesh as _umesh
from . import bbox
from .. import utils

if typing.TYPE_CHECKING:
    from . import AdvIsland  # noqa


class LoopGroup:
    def __init__(self, umesh: _umesh.UMesh):
        self.umesh: _umesh.UMesh = umesh
        self.corners: list[BMLoop] = []
        self.tag = True
        self.value: typing.Any | utils.NoInit = utils.NoInit()
        self.dirt = False
        self.weights: list[float] | None = None
        self.chain_linked_corners: list[list[BMLoop]] = []

    @property
    def src(self):
        """For search short path"""
        return self.chain_linked_corners[0][0]

    @property
    def dst(self):
        """For search short path"""
        return self.chain_linked_corners[-1][0]

    def set_tag(self, state=True):
        for crn in self.corners:
            crn.tag = state

    def has_sync_crn(self):
        """Need tagging and indexing"""
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
        if utils.vec_isclose_to_zero(delta):
            return False
        uv = self.umesh.uv
        for loop in self.corners:
            loop[uv].uv += delta
        return True

    def set_position(self, to: Vector, _from: Vector):
        return self.move(to - _from)

    def calc_bbox(self):
        return bbox.BBox.calc_bbox_uv_corners(self.corners, self.umesh.uv)

    def calc_chain_linked_corners(self):
        get_linked_inc = utils.get_linked_crn_fn(self.umesh, included=True)  # TODO: Add with_seam ???
        for crn in chain(self, [self[-1].link_loop_next]):
            self.chain_linked_corners.append(get_linked_inc(crn))


    def extend_from_linked(self):
        """That untag linked corners, for avoid duplicates."""

        self.set_tag(False)

        if utils.sync():
            # Need tag_visible_corners before use
            move_corners = []
            uv = self.umesh.uv
            for crn in self:
                move_corners.extend(utils.linked_crn_vert_uv_by_tag_with_untagging_for_transform(crn, uv))
            self.corners.extend(move_corners)
        else:
            move_corners = []
            uv = self.umesh.uv
            for crn in self:
                linked_corners = utils.linked_crn_uv_by_crn_tag_unordered_included(
                    crn, uv
                )  # TODO: Add linked_crn_uv_by_tag_c by island
                move_corners.extend(linked_corners)
                for crn_ in linked_corners:
                    crn_.tag = False

                linked_corners = utils.linked_crn_uv_by_crn_tag_unordered_included(
                    crn.link_loop_next, uv
                )
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
        return f"Corner Edge count = {len(self.corners)}"


class LoopGroups:
    def __init__(self, loop_groups, umesh):
        self.loop_groups: list[LoopGroup] = loop_groups
        self.umesh: _umesh.UMesh | None = umesh
        self.tag = True

    @classmethod
    def calc_dirt_loop_groups_from_selected(cls, umesh):
        uv = umesh.uv
        umesh.tag_selected_corners()
        sel_corners = (crn for f in umesh.bm.faces for crn in f.loops if crn.tag)

        groups: list[LoopGroup] = []
        for crn_ in sel_corners:
            group = []
            temp_group = [crn_]
            while True:
                temp = []
                for sel in temp_group:
                    it1 = utils.linked_crn_uv_unordered_included_with_hidden(sel, uv)
                    it2 = utils.linked_crn_uv_unordered_included_with_hidden(sel.link_loop_next, uv)
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
            lg = LoopGroup(umesh)
            lg.corners = group
            lg.dirt = True
            groups.append(lg)
        return cls(groups, umesh)

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



    def __iter__(self) -> typing.Iterator[LoopGroup]:
        return iter(self.loop_groups)

    def __getitem__(self, idx) -> LoopGroup:
        return self.loop_groups[idx]

    def __bool__(self):
        return bool(self.loop_groups)

    def __len__(self):
        return len(self.loop_groups)

    def __str__(self):
        return f"Loop Groups count = {len(self.loop_groups)}"


class UnionLoopGroup(LoopGroups):
    def __init__(self, loop_groups: list[LoopGroup]):
        super().__init__(loop_groups, None)

class _AdvCorner_Base:
    def __init__(self, crn: BMLoop, umesh: "_umesh.UMesh", is_pair: bool, invert: bool=False):
        self.crn: BMLoop = crn
        self.invert: bool = invert
        self.umesh: "_umesh.UMesh" = umesh
        # self.uv = uv
        self._vec: Vector | None = None
        self.is_pair: bool = is_pair

    @property
    def vec(self):
        """NOTE: Don't forget to sanitize zero-length edges, depending on the context!!!"""
        if not self._vec:
            uv = self.umesh.uv
            if self.invert:
                _vec = self.crn.link_loop_prev[uv].uv - self.crn[uv].uv
            else:
                _vec = self.crn.link_loop_next[uv].uv - self.crn[uv].uv
            _vec *= Vector((1.0, self.umesh.aspect))
            _vec.normalize()
            self._vec = _vec
            return _vec
        return self._vec

    @vec.setter
    def vec(self, v: Vector):
        self._vec = v

    @property
    def length_3d(self):
        if self.invert:
            return (self.crn.vert.co - self.crn.link_loop_prev.vert.co).length
        return (self.crn.vert.co - self.crn.link_loop_next.vert.co).length

    def angle(self, other: "AdvCorner", max_angle: float):
        return self.vec.angle(other.vec, max_angle)  # noqa

    @property
    def vert(self):
        return self.crn.vert

    @property
    def edge(self):
        if self.invert:
            return self.crn.link_loop_prev.edge
        else:
            return self.crn.edge

    @property
    def length_uv(self):
        uv = self.umesh.uv
        if self.invert:
            return (self.crn[uv].uv - self.crn.link_loop_prev[uv].uv).length
        return (self.crn[uv].uv - self.crn.link_loop_next[uv].uv).length

    @property
    def angle_from_cardinal(self):
        vec = self.vec
        card_vec = utils.vec_to_cardinal(vec)
        angle = vec.angle(card_vec, 0)
        return min(angle, pi - angle)

    @property
    def orig_crn(self):
        """ Corner with invert correction."""
        return self.crn.link_loop_prev if self.invert else self.crn

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
            self._vec.negate()
        return self

    def normalize_dir_if_possible(self):
        if self.invert:
            if self.is_pair:
                pass
            else:
                self.invert = False
                self.crn = self.crn.link_loop_prev

    def copy(self):
        adv_crn = type(self)(self.crn, self.umesh, self.is_pair, self.invert)
        if self._vec:
            adv_crn._vec = self._vec.copy()
        else:
            adv_crn._vec = None
        return adv_crn

    def linked_crn_iter(self, is_boundary_) -> tuple[list["AdvCorner | AdvCorner3D"], list["AdvCorner | AdvCorner3D"], bool]:
        """NOTE: For clamp by selected edges need `is_boundary_` for get all edges."""

        T = type(self)
        umesh = self.umesh
        left = []  # CCW
        right = []  # CW

        if self.invert:
            crn = self.crn
            bm_iter = crn
            while True:
                bm_iter_prev = bm_iter.link_loop_prev
                if is_boundary_(bm_iter_prev):  # Skip boundary.
                    if bm_iter_prev.link_loop_next != crn:
                        left.append(T(bm_iter_prev.link_loop_next, umesh, is_pair=False, invert=True))  # Store inverted boundary.

                    # Right. CW.
                    bm_iter = crn
                    while True:
                        if is_boundary_(bm_iter):  # clamp by seam
                            right.append(T(bm_iter, umesh, is_pair=False))
                            return left, right, False

                        right.append(T(bm_iter, umesh, is_pair=True))

                        bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # Get CW corner.

                        # In CW, you also need to check for the initial corner,
                        # since in CCW, the break could have been caused by a seam case.
                        assert bm_iter != crn
                        if bm_iter == crn:
                            return left, right, False

                bm_iter = bm_iter_prev.link_loop_radial_prev  # Wheel CCW.
                left.append(T(bm_iter, umesh, is_pair=True))
                if bm_iter == crn:
                    return left, right, True

        else:
            crn = self.crn
            bm_iter = crn
            while True:
                bm_iter_prev = bm_iter.link_loop_prev
                if is_boundary_(bm_iter_prev):  # Skip boundary.
                    left.append(T(bm_iter, umesh, is_pair=False, invert=True))  # Store inverted boundary.

                    # Right. CW.
                    bm_iter = crn
                    while True:
                        if is_boundary_(bm_iter):  # clamp by seam
                            if bm_iter != crn:  # Clamp on first iter case.
                                right.append(T(bm_iter, umesh, is_pair=False))
                            return left, right, False

                        prev_pair = bm_iter
                        bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # Get CW corner.

                        # In CW, you also need to check for the initial corner,
                        # since in CCW, the break could have been caused by a seam case.
                        if bm_iter == crn:
                            return left, right, False

                        if prev_pair != crn: # Skip first iteration
                            right.append(T(prev_pair, umesh, is_pair=True))
                        else:
                            assert not right

                bm_iter = bm_iter_prev.link_loop_radial_prev  # Wheel CCW.
                if bm_iter == crn:
                    return left, right, True

                left.append(T(bm_iter, umesh, is_pair=True))

    def __hash__(self):
        # NOTE: Don't use *.is_pair in this.
        return hash(self.crn)

    def __eq__(self, other):
        return self.crn == other.crn and self.invert == other.invert



class AdvCorner(_AdvCorner_Base):
    @property
    def curr_pt(self):  # TODO: Replace with curr_uv_pt
        return self.crn[self.umesh.uv].uv

    @property
    def next_pt(self):
        if self.invert:
            return self.crn.link_loop_prev[self.umesh.uv].uv
        else:
            return self.crn.link_loop_next[self.umesh.uv].uv

    @property
    def center_uv(self):
        return self.curr_pt.lerp(self.next_pt, 0.5)

    # def normalize(self):
    #     if self.invert:
    #         if utils.is_pair_by_idx(self.crn, self.crn.link_loop_radial_prev, self.uv):
    #             self.invert = False
    #             self.crn = self.crn.link_loop_radial_prev
    #             return True
    #     return False



class AdvCorner3D(_AdvCorner_Base):
    @property
    def vec(self):
        """NOTE: Don't forget to sanitize zero-length edges, depending on the context!!!"""
        if not self._vec:
            if self.invert:
                _vec = self.crn.link_loop_prev.vert.co - self.crn.vert.co
            else:
                _vec = self.crn.link_loop_next.vert.co - self.crn.vert.co
            _vec.normalize()
            self._vec = _vec
            return _vec
        return self._vec

    @vec.setter
    def vec(self, v):
        self._vec = v

    def __str__(self):
        txt =  f"AdvCorner3D. ID: {id(self.crn)!r} Is Pair: {self.is_pair}."
        if self.invert:
            txt += f" Invert: True"
        return txt
    def __repr__(self):
        return self.__str__()




class Segment:
    def __init__(self, seg, umesh):
        self.seg: list[AdvCorner] = seg
        self.umesh: "_umesh.UMesh" = umesh
        self.tag = True

        self.angles_seq: list[float] = []
        self.lengths_seq: list[float] = []
        self.chain_linked_corners: list[list[BMLoop]] = []

        self._length_uv: float | utils.NoInit = utils.NoInit()
        self._length_3d: float | utils.NoInit = utils.NoInit()
        self._weight_angle: float | utils.NoInit = utils.NoInit()
        self.value: float | utils.NoInit = utils.NoInit()

        self.has_intervals = False
        self.is_start_lock: bool = False
        self.is_end_lock: bool = False

        self.is_shared = False
        self.is_flipped_3d = False

    def to_corners(self):
        return [(adv_crn.crn.link_loop_prev if adv_crn.invert else adv_crn.crn) for adv_crn in self]

    def get_corners_iter(self, included):
        """Get original corners from start to end."""
        for adv_crn in self:
            if adv_crn.invert:
                yield adv_crn.crn.link_loop_prev
            else:
                yield adv_crn.crn

        if included:
            yield self.end_orig_crn

    def mark_seam(self, rewrite_pair_status):
        if rewrite_pair_status:
            for adv_crn in self:
                adv_crn.is_pair = False
                adv_crn.orig_crn.edge.seam = True
        else:
            for adv_crn in self:
                adv_crn.orig_crn.edge.seam = True

    def calc_chain_linked_corners(self):
        get_linked_inc = utils.get_linked_crn_fn(self.umesh, included=True)
        for adv_crn in self.seg:
            if adv_crn.invert:
                crn = adv_crn.crn.link_loop_prev
            else:
                crn = adv_crn.crn
            self.chain_linked_corners.append(get_linked_inc(crn))

        if self.is_start_lock:
            del self.chain_linked_corners[0]

        if not self.is_end_lock:
            last_adv_crn = self.seg[-1]
            if last_adv_crn.invert:
                crn = last_adv_crn.crn
            else:
                crn = last_adv_crn.crn.link_loop_next
            self.chain_linked_corners.append(get_linked_inc(crn))

    def reverse(self):
        self.seg = [crn.toggle_dir() for crn in reversed(self.seg)]
        self.angles_seq.reverse()
        self.lengths_seq.reverse()

        temp_lock = self.is_start_lock
        self.is_start_lock = self.is_end_lock
        self.is_end_lock = temp_lock

    def join_from_end(self, other: "Segment"):
        assert self.end_vert == other.start_vert
        assert self.end_co == other.start_co
        assert not self.is_end_lock
        assert not other.is_start_lock

        self.seg.extend(other.seg)
        self.angles_seq.extend(other.angles_seq)
        self.lengths_seq.extend(other.lengths_seq)

        self._length_uv += other._length_uv
        self._weight_angle = utils.NoInit()

        self.is_end_lock = other.is_end_lock
        other.tag = False

    def calc_angles_from_card_dir(self):
        for adv_crn in self:
            self.angles_seq.append(adv_crn.angle_from_cardinal)

    def calc_lengths_uv(self):
        lengths_seq = self.lengths_seq
        if not lengths_seq:
            for adv_crn in self:
                lengths_seq.append(adv_crn.length_uv)
        return lengths_seq

    # TODO: Delete length
    @property
    def length(self):
        if isinstance(self._length_uv, utils.NoInit):
            if not self.lengths_seq:
                self.calc_lengths_uv()
            self._length_uv = sum(self.lengths_seq)
        return self._length_uv

    @length.setter
    def length(self, value: float):
        self._length_uv = value

    @property
    def length_uv(self):
        if isinstance(self._length_uv, utils.NoInit):
            if not self.lengths_seq:
                self.calc_lengths_uv()  # TODO: Simplify (use straight from AdvCorner)
            self._length_uv = sum(self.lengths_seq)
        return self._length_uv

    @length_uv.setter
    def length_uv(self, value: float):
        self._length_uv = value

    @property
    def length_3d(self):
        if isinstance(self._length_uv, utils.NoInit):
            self._length_uv = sum(adv_crn.length_3d for adv_crn in self)
        return self._length_uv

    def calc_signed_face_area(self):
        uv = self.umesh.uv
        # TODO: Report small areas in stitch
        return sum(utils.calc_signed_face_area_uv(adv_crn.crn.face, uv) for adv_crn in self)

    @property
    def weight_angle(self):
        if isinstance(self._weight_angle, utils.NoInit):
            if not self.angles_seq:
                self.calc_angles_from_card_dir()

            if not self.lengths_seq:
                self.calc_lengths_uv()
            try:
                self._weight_angle = float(np.average(self.angles_seq, weights=self.lengths_seq))
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
            return adv_crn.crn.link_loop_prev.vert  # TODO: Incorrect, need link_loop_next ???
        else:
            return adv_crn.crn.link_loop_next.vert

    @property
    def start_co(self) -> Vector:
        adv_crn = self.seg[0]
        return adv_crn.crn[adv_crn.umesh.uv].uv

    @property
    def end_co(self) -> Vector:
        adv_crn = self.seg[-1]
        if adv_crn.invert:
            return adv_crn.crn.link_loop_prev[adv_crn.umesh.uv].uv
        else:
            return adv_crn.crn.link_loop_next[adv_crn.umesh.uv].uv

    @property
    def start_orig_crn(self) -> BMLoop:
        adv_crn = self.seg[0]
        return adv_crn.crn

    @property
    def end_orig_crn(self) -> BMLoop:
        adv_crn = self.seg[-1]
        if adv_crn.invert:
            return adv_crn.crn.link_loop_prev
        else:
            return adv_crn.crn.link_loop_next

    @property
    def is_circular(self):
        return self.start_vert == self.end_vert and self.start_co == self.end_co

    def is_horizontal_by_bbox(self, aspect_y):
        from . import BBox
        bb = BBox.calc_bbox(adv.curr_pt for adv in self)
        bb.update([self.start_co, self.end_co])
        return bool(bb.width > bb.height*aspect_y)

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

            for i in range(len(seg) - 1):
                prev_cardinal = vec_to_cardinal(seg[i].vec)
                curr_cardinal = vec_to_cardinal(seg[i + 1].vec)
                if curr_cardinal != prev_cardinal:
                    break_indexes.append(i + 1)

            full_breaks = [0] + break_indexes + [len(seg)]

            slices = [seg[full_breaks[i] : full_breaks[i + 1]] for i in range(len(full_breaks) - 1)]

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

    @property
    def _is_flipped_3d(self):
        assert not self.is_shared
        orig_crn = self.start_orig_crn
        return orig_crn.link_loop_radial_prev.vert == orig_crn.vert

    def calc_shared_group_for_stitch(self) -> "Segment":
        shared_group = []
        is_flipped = self._is_flipped_3d

        assert not any(a.invert for a in self)  # TODO: Add to force debug (add force_debug_fn(fn: callable)).
        umesh = self.umesh
        if is_flipped:
            for adv_crn in self.seg:
                new_adv_crn = adv_crn.orig_crn.link_loop_radial_prev
                shared_group.append(AdvCorner(new_adv_crn, umesh, is_pair=False))
        else:
            for adv_crn in self.seg:
                new_adv_crn = adv_crn.orig_crn.link_loop_radial_prev.link_loop_next
                shared_group.append(AdvCorner(new_adv_crn, umesh, False, invert=True,))

        seg = Segment(shared_group, self.umesh)

        seg.is_shared = True
        seg.is_flipped_3d = is_flipped  # TODO: Delete this ???
        seg.corners = shared_group
        return seg

    def copy_coords_from_ref(self, ref: "Segment", clean_seams):
        uv = self.umesh.uv
        get_linked_inc = utils.get_linked_crn_fn(self.umesh, included=True)

        assert len(self) == len(ref)

        for ref_crn, trans_crn in zip(ref, self):
            ref_co = ref_crn.curr_pt

            for trans_crn_linked in get_linked_inc(trans_crn.crn):
                trans_crn_linked[uv].uv = ref_co

        if clean_seams:
            for trans_crn in self:
                trans_crn.edge.seam = False

        ref_end_orig_crn = ref.end_orig_crn
        self_end_orig_crn = self.end_orig_crn
        assert ref_end_orig_crn.vert == self_end_orig_crn.vert

        ref_co = ref_end_orig_crn[uv].uv

        for trans_crn_linked in get_linked_inc(self_end_orig_crn):
            trans_crn_linked[uv].uv = ref_co

    def deepcopy(self):
        return Segment([adv_crn.copy() for adv_crn in self.seg], self.umesh)

    def distribute(self, start, end, uv_length_distribute):
        assert self.chain_linked_corners
        uv = self.umesh.uv
        if uv_length_distribute:
            assert self.lengths_seq
            weights = self.lengths_seq
        else:
            weights = [crn.edge.calc_length() for crn in self]

        points = utils.weighted_linear_space(start, end, weights)
        for corners, co in zip(self.chain_linked_corners, points):
            for l_crn in corners:
                l_crn[uv].uv = co

    def get_corners_for_marking(self):
        # if seg.is_end_lock:
        #     seg.lengths_seq.pop()
        # if not seg.lengths_seq:
        #     continue
        #
        # if seg.is_start_lock:
        #     del seg.lengths_seq[0]
        # if not seg.lengths_seq:
        #     continue

        for adv_crn in self:
            if adv_crn.invert:
                yield adv_crn.crn.link_loop_prev
            else:
                yield adv_crn.crn

            if adv_crn.is_pair:
                yield adv_crn.crn.link_loop_radial_prev

    def calc_bbox(self):
        bb = bbox.BBox.calc_bbox_uv_corners((adv_crn.orig_crn for adv_crn in self), self.umesh.uv)
        bb.update([self.end_co])
        return bb

    def __iter__(self):
        return iter(self.seg)

    def __getitem__(self, idx) -> AdvCorner:
        return self.seg[idx]

    def __len__(self):
        return len(self.seg)

    def __bool__(self):
        return bool(self.seg)

    def __str__(self):
        txt = f"Segment. Adv Corner count = {len(self.seg)}."
        if self.is_start_lock or self.is_end_lock:
            txt += f" Start Lock: {self.is_start_lock}. End Lock: {self.is_end_lock}"
        return txt

    def split_left_and_right(self):
        assert not self.is_start_lock
        assert not self.is_end_lock
        has_intervals_left = False
        has_intervals_right = False
        left = []
        right = []
        umesh = self.umesh
        for adv_crn in self:
            if adv_crn.is_pair:
                assert not adv_crn.invert
                left.append(AdvCorner(adv_crn.crn, umesh, is_pair=False))
                right.append(AdvCorner(adv_crn.crn.link_loop_radial_prev.link_loop_next, umesh, is_pair=False, invert=True))
            else:
                if adv_crn.invert:
                    right.append(adv_crn.copy())
                    has_intervals_left = True
                else:
                    left.append(adv_crn.copy())
                    has_intervals_right = True

        left_seg = Segment(left, umesh)
        right_seg = Segment(right, umesh)
        left_seg.has_intervals = has_intervals_left
        right_seg.has_intervals = has_intervals_right

        return left_seg, right_seg


class Segments:
    def __init__(self, segments, umesh):
        self.segments: typing.Sequence[Segment] | list[Segment] = segments
        self.umesh = umesh

    @classmethod
    def calc_by_tags(cls):
        pass

    @classmethod
    def from_corners(cls, corners: list[BMLoop] | set[BMLoop], umesh: "_umesh.UMesh"):
        # NOTE: Need indexing islands and tagged corners
        is_boundary = utils.is_boundary_func(umesh)
        appended = set()
        segments = []

        get_linked_it_inc = utils.get_linked_crn_iter_fn(umesh, included=True)

        corners = corners if isinstance(corners, set) else set(corners)
        for crn in corners:
            seg = deque()
            first_crn = AdvCorner(crn, umesh, is_pair=not is_boundary(crn))
            if first_crn in appended:
                continue

            appended.add(first_crn)
            seg.append(first_crn)

            if first_crn.is_pair:
                # Avoid adding a pair edge.
                appended.add(AdvCorner(crn.link_loop_radial_prev, umesh, is_pair=True))
            else:
                # Set invert for compare convention.
                # Artificially inverted boundary edges are not included in the final segment.
                # They are needed to prevent reusing the current edge in the opposite direction during a `Backward Grow`.
                appended.add(AdvCorner(crn.link_loop_next, umesh, is_pair=False, invert=True))

            # Forward Grow
            while True:
                lead = seg[-1]
                if lead.invert:
                    next_check = lead.crn
                else:
                    next_check = lead.crn.link_loop_next

                count = 0
                filtered = []
                for crn_l in get_linked_it_inc(next_check):
                    # Next grow
                    if crn_l in corners:
                        next_grow = AdvCorner(crn_l, umesh, is_pair=not is_boundary(crn_l))
                        count += 1
                        if next_grow not in appended:
                            filtered.append(next_grow)

                    # Here we skip pair edges, as they are processed in the previous condition.
                    prev = crn_l.link_loop_prev
                    if prev in corners and is_boundary(prev):  # TODO: Check with is_boundary(crn_l) for avoid bug in multi-face case.
                        prev_grow = AdvCorner(crn_l, umesh, is_pair=False, invert=True)
                        count += 1
                        if prev_grow not in appended:
                            filtered.append(prev_grow)

                if count >= 3:  # Break segment by 3 and more selected edges.
                    break

                if len(filtered) == 1:
                    next_elem = filtered[0]
                    if next_elem in appended:
                        break

                    seg.append(next_elem)

                    # Add possible CrnGrow variants so that we don't go through them again.
                    appended.add(next_elem)
                    # NOTE: is_pair in this case doesn't matter.
                    if next_elem.invert:
                        appended.add(AdvCorner(next_elem.crn.link_loop_prev, umesh, is_pair=False))
                    else:
                        if is_boundary(next_elem.crn):
                            appended.add(AdvCorner(next_elem.crn.link_loop_next, umesh, is_pair=False, invert=True))
                        else:
                            # The pair doesn't have an invert option, so we do without them
                            appended.add(AdvCorner(next_elem.crn.link_loop_radial_prev, umesh, is_pair=True))
                else:
                    break

            # Backward Grow
            while True:
                lead = seg[0]
                next_check = lead.crn

                count = 0
                filtered = []
                for crn_l in get_linked_it_inc(next_check):
                    # Next grow
                    if crn_l in corners:
                        count += 1
                        if is_boundary(crn_l):
                            next_grow = AdvCorner(crn_l.link_loop_next, umesh, is_pair=False, invert=True)
                        else:
                            next_grow = AdvCorner(crn_l.link_loop_radial_prev, umesh, is_pair = True)
                        if next_grow not in appended:
                            filtered.append(next_grow)

                    prev = crn_l.link_loop_prev
                    if prev in corners and is_boundary(prev):
                        prev_grow = AdvCorner(prev, umesh, is_pair = False)
                        count += 1
                        if prev_grow not in appended:
                            filtered.append(prev_grow)

                if count >= 3:  # See above.
                    break

                if len(filtered) == 1:
                    next_elem = filtered[0]
                    if next_elem in appended:
                        break

                    seg.appendleft(next_elem)
                    appended.add(next_elem)
                    # NOTE: is_pair in this case doesn't matter.
                    if next_elem.invert:
                        appended.add(AdvCorner(next_elem.crn.link_loop_prev, umesh, is_pair=False))
                    else:
                        if next_elem.is_pair:
                            appended.add(AdvCorner(next_elem.crn.link_loop_radial_prev, umesh, is_pair=True))
                        else:
                            appended.add(AdvCorner(next_elem.crn.link_loop_next, umesh, is_pair=False, invert=True))
                else:
                    break

            segments.append(Segment(list(seg), umesh))
        return cls(segments, umesh)

    def rotate_longest_to_first(self):
        idx = self.segments.index(max(self, key=lambda s: s.length_uv))
        segments = deque(self.segments)
        segments.rotate(-idx)
        self.segments = list(segments)

    def break_by_cardinal_dir(self):
        segments = []
        for seg in self:
            segments.extend(seg.break_by_cardinal_dir().segments)
        return Segments(segments, self.umesh)

    def draw(self):
        from .. import draw
        draw.SegmentsDrawSimple.draw_register(self)  # noqa

    def __iter__(self) -> typing.Iterator[Segment]:
        return iter(self.segments)

    def __getitem__(self, idx) -> Segment:
        return self.segments[idx]

    def __len__(self):
        return len(self.segments)

    def __bool__(self):
        return bool(self.segments)

    def __str__(self):
        return f"Segments. Segments count = {len(self.segments)}"

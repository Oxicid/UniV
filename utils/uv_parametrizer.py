# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import heapq
import typing
from bmesh.types import BMFace
from mathutils import Vector, Matrix

from math import pi
from bl_math import clamp

from .ubm import polyfill_beautify
from . import bm_select
from .. import utypes
from .umath import LinearSolver


T = typing.TypeVar("T", "PFace", "PEdge", "PVert")
class ParametrizerIt(typing.Generic[T]):
    def __init__(self):
        self.first_item: typing.Optional[T] = None

    def __iter__(self) -> typing.Iterator[T]:
        item = self.first_item
        while item:
            yield item
            item = item.nextlink

    @staticmethod
    def __class_getitem__(item):
        return typing.Annotated[ParametrizerIt, item]


class HeapItem:
    __slots__ = ("angle", "edge", "removed")

    def __init__(self, angle, edge):
        self.angle = angle
        self.edge = edge
        self.removed = False

    def __lt__(self, other):
        return self.angle < other.angle


class UnwrapOptions:
    # Connectivity based on UV coordinates instead of seams. */
    topology_from_uvs: bool = False
    # Also use seams as well as UV coordinates (only valid when `topology_from_uvs` is enabled). */
    topology_from_uvs_use_seams: bool = True
    # Only affect selected faces. */
    only_selected_faces: bool = False

    # Only affect selected UVs.
    # \note Disable this for operations that don't run in the image-window.
    # Unwrapping from the 3D view for example, where only 'only_selected_faces' should be used.

    only_selected_uvs: bool  = False
    # Fill holes to better preserve shape. */
    fill_holes: bool = True
    # Correct for mapped image texture aspect ratio. */
    correct_aspect: bool = True
    # Treat unselected uvs as if they were pinned. */
    pin_unselected: bool = False
    unwrap_along: typing.Literal['UV', 'U', 'V'] = 'UV'

    method: int = 0
    use_slim: bool = False
    use_abf: bool = False
    use_subsurf: bool = False
    use_weights: bool = False

    # slim: ParamSlimOptions = None
    weight_group: str = ''


PVERT_PIN = 1
PVERT_SELECT = 2
PVERT_INTERIOR = 4
PVERT_COLLAPSE = 8
PVERT_SPLIT = 16


# PEdgeFlag
PEDGE_SEAM = 1
PEDGE_VERTEX_SPLIT = 2
PEDGE_PIN = 4
PEDGE_SELECT = 8
PEDGE_DONE = 16
PEDGE_FILLED = 32
PEDGE_COLLAPSE = 64
PEDGE_COLLAPSE_EDGE = 128
PEDGE_COLLAPSE_PAIR = 256

# for flipping faces
PEDGE_VERTEX_FLAGS = PEDGE_PIN

# PFaceFlag
PFACE_CONNECTED = 1
PFACE_FILLED = 2
PFACE_COLLAPSE = 4
PFACE_DONE = 8

def angle_v3v3v3(a: Vector, b: Vector, c: Vector):
    return (b - a).angle(b - c, 0.0)


def fix_large_angle(v_fix: Vector, v1: Vector, v2: Vector, r_fix: float, r_a1: float, r_a2: float) -> tuple[float, float, float]:
    """
    # Angles close to 0 or 180 degrees cause rows filled with zeros in the linear_solver.
    # The matrix will then be rank deficient and / or have poor conditioning.
    # => Reduce the maximum angle to 179 degrees, and spread the remainder to the other angles.
    """
    max_angle: float = math.degrees(179.0)
    fix_amount: float = r_fix - max_angle
    if fix_amount < 0.0:
        # angle is reasonable, i.e. less than 179 degrees.
        return  r_fix, r_a1, r_a2


    # The triangle is probably degenerate, or close to it.
    # Without loss of generality, transform the triangle such that
    # v_fix == {  0, s}, *r_fix = 180 degrees
    # v1    == {-x1, 0}, *r_a1  = 0
    # v2    == { x2, 0}, *r_a2  = 0
    #
    # With `s = 0`, `x1 > 0`, `x2 > 0`
    #
    # Now make `s` a small number and do some math:
    # tan(*r_a1) = s / x1
    # tan(*r_a2) = s / x2
    #
    # Remember that `tan(angle) ~= angle`
    #
    # Rearrange to obtain:
    # *r_a1 = fix_amount * x2 / (x1 + x2)
    # *r_a2 = fix_amount * x1 / (x1 + x2)

    dist_v1: float = (v_fix - v1).length
    dist_v2: float = (v_fix - v2).length
    dist_sum: float = dist_v1 + dist_v2
    weight: float = dist_v2 / dist_sum if (dist_sum > 1e-20) else 0.5

    # Ensure sum of angles in triangle is unchanged.
    r_fix -= fix_amount
    r_a1 += fix_amount * weight
    r_a2 += fix_amount * (1.0 - weight)
    return r_fix, r_a1, r_a2


def p_triangle_angles(v1: Vector, v2: Vector, v3: Vector) -> tuple[float, float, float]:

    a1 = angle_v3v3v3(v3, v1, v2)
    a2 = angle_v3v3v3(v1, v2, v3)
    a3 = angle_v3v3v3(v2, v3, v1)

    # Fix for degenerate geometry e.g. v1 = sum(v2 + v3). See #100874
    a1, a2, a3 = fix_large_angle(v1, v2, v3, a1, a2, a3)
    a2, a3, a1 = fix_large_angle(v2, v3, v1, a2, a3, a1)
    a3, a1, a2 = fix_large_angle(v3, v1, v2, a3, a1, a2)

    # Workaround for degenerate geometry, e.g. v1 == v2 == v3.
    a1 = max(a1, 0.001)
    a2 = max(a2, 0.001)
    a3 = max(a3, 0.001)
    return a1, a2, a3


PHashKey = typing.NewType('PHashKey', int)  # uintptr_t
ParamKey = typing.NewType('ParamKey', int)  # uintptr_t

def PHASH_hash(ph, key):
    return key % ph.cursize


def PHASH_edge(v1: int | PHashKey | ParamKey, v2: int | PHashKey | ParamKey) -> int | PHashKey | ParamKey:
    if v1 < v2:
        return (v1 * 39) ^ (v2 * 31)
    else:
        return (v1 * 31) ^ (v2 * 39)

class PHashLink:
    def __init__(self):
        self.nextlink: PHashLink | None = None  # noqa
        self.key: PHashKey = PHashKey(0)


class PHash:

    PHashSizes: tuple[int, ...] = (
        1,       3,       5,       11,      17,       37,       67,       131,       257,       521,
        1031,    2053,    4099,    8209,    16411,    32771,    65537,    131101,    262147,    524309,
        1048583, 2097169, 4194319, 8388617, 16777259, 33554467, 67108879, 134217757, 268435459,
    )

    def __init__(self, lst: ParametrizerIt, size_hint: int = 1):
        # Pointer of pointer, need for save link for iterate of all elem by *.nextlink
        self.lst: ParametrizerIt[PHashLink | PFace | PEdge | PVert] = lst

        self.size: int = 0
        self.cursize_id: int = 0
        while self.PHashSizes[self.cursize_id] < size_hint:
            self.cursize_id += 1

        self.cursize = self.PHashSizes[self.cursize_id]
        self.buckets: list[PHashLink | None] = [None] * self.cursize

    def insert(self, link: PHashLink | typing.Any):
        size = self.cursize
        hash_value = link.key % size  # PHASH_hash
        lookup: PHashLink = self.buckets[hash_value]

        if lookup is None:
            # insert in front of the list */
            self.buckets[hash_value] = link
            link.nextlink = self.lst.first_item
            self.lst.first_item = link
        else:
            #/* insert after existing element */
            link.nextlink = lookup.nextlink
            lookup.nextlink = link

        self.size += 1

        if self.size > (size * 3):
            first: PHashLink = self.lst.first_item
            self.cursize_id += 1
            self.cursize = PHash.PHashSizes[self.cursize_id]

            self.buckets = [None] * self.cursize
            self.size = 0
            self.lst.first_item = None

            link = first
            while link:
                next_: PHashLink | None = link.nextlink
                self.insert(link)
                link = next_

    def lookup(self, key: PHashKey) -> 'PHashLink | PEdge | typing.Any':
        hash_value = PHASH_hash(self, key)
        link: PHashLink = self.buckets[hash_value]

        while link:
            if link.key == key:
                return link
            if PHASH_hash(self, link.key) != hash_value:
                return None

            link = link.nextlink

        return link

    def next(self, key: PHashKey, link: PHashLink | typing.Any) -> PHashLink | None:
        hash_value = PHASH_hash(self, key)

        link = link.nextlink
        while link:
            if link.key == key:
                return link
            if PHASH_hash(self, link.key) != hash_value:
                return None

            link = link.nextlink

        return link


class PVert:
    def __init__(self):
        self.nextlink: PVert

        self.id: int = 0  # noqa  # ABF/LSCM matrix index

        self.edge: PEdge | None = None
        self.co: Vector = Vector()
        self.uv: Vector = Vector((0.0, 0.0))
        self.flag: int = 0

        self.weight: float = 1.0
        self.on_boundary_flag: bool = False
        # slim_id: int

    @property
    def heaplink(self):
        """Edge collapsing."""
        return self.id

    @heaplink.setter
    def heaplink(self, v):
        self.id = v

    @property
    def key(self):
        """Construct."""
        return self.id

    @key.setter
    def key(self, v):
        self.id = v

    def copy(self):
        nv = PVert()
        nv.co = self.co.copy()
        nv.uv = self.uv.copy()

        nv.id = self.id
        nv.edge = self.edge
        nv.flag = self.flag

        nv.weight = self.weight
        nv.on_boundary_flag = self.on_boundary_flag
        # nv.slim_id = v.slim_id

        return nv

    def load_pin_select_uvs(self):
        n_edges = 0
        n_pins = 0

        self.uv[0] = self.uv[1] = 0.0  # TODO: Remove (check PVert.copy()) ???
        pin_uv = Vector((0.0, 0.0))

        e = self.edge
        while True:
            if e.orig_uv:
                if e.flag & PEDGE_SELECT:
                    self.flag |= PVERT_SELECT

                if e.flag & PEDGE_PIN:
                    pin_uv += e.orig_uv  # * self.aspect_y
                    n_pins += 1
                else:
                    self.uv += e.orig_uv   # * self.aspect_y

                n_edges += 1

            e = e.wheel_edge_next
            if not e or e == self.edge:
                break

        if n_pins > 0:
            self.uv.xy = pin_uv * (1 / n_pins)
            self.flag |= PVERT_PIN
        elif n_edges > 0:
            self.uv.xy *= 1 / n_edges

    @property
    def is_interior(self):
        return bool(self.edge.pair)


class PEdge:
    def __init__(self):
        self.nextlink: PEdge | None = None
        # ABF/LSCM matrix index
        self.id: int = 0  # noqa

        self.vert: PVert | None = None
        self.pair: PEdge | None = None
        self.next: PEdge | None = None
        self.face: PFace | None = None

        self.orig_uv: Vector | None = None
        self.old_uv: Vector | None = None

        self.flag: int = 0

    @property
    def heaplink(self):
        """Edge collapsing."""
        return self.id

    @heaplink.setter
    def heaplink(self, v):
        self.id = v

    @property
    def key(self):
        """Construct."""
        return self.id

    @key.setter
    def key(self, v):
        self.id = v

    @property
    def nextcollapse(self):
        """Simplification."""
        return self.id

    @nextcollapse.setter
    def nextcollapse(self, v):
        self.id = v

    @property
    def length_3d(self):
        return (self.vert.co - self.next.vert.co).length

    @property
    def wheel_edge_next(self) -> 'PEdge':
        return self.next.next.pair

    @property
    def wheel_edge_prev(self) -> 'PEdge | None':
        return self.pair.next if self.pair else None

    @property
    def boundary_edge_next(self: 'PEdge') -> 'PEdge':
        return self.next.vert.edge

    @property
    def boundary_edge_prev(self: 'PEdge') -> 'PEdge':
        we: PEdge = self
        last: PEdge

        while True:
            last = we
            we = we.wheel_edge_next
            if not we or we == self:
                break

        return last.next.next


    def implicit_seam(self, e_pair: 'PEdge') -> bool:
        uv1 = self.orig_uv
        uv2 = self.next.orig_uv

        # Determine the order in which to match UV
        if self.vert.key == e_pair.vert.key:
            uvp1 = e_pair.orig_uv
            uvp2 = e_pair.next.orig_uv
        else:
            uvp1 = e_pair.next.orig_uv
            uvp2 = e_pair.orig_uv

        # Comparison of UV coordinates — too different -> seam
        from . import vec_isclose
        if not vec_isclose(uv1, uvp1):
            self.flag |= PEDGE_SEAM
            e_pair.flag |= PEDGE_SEAM
            return True

        if not vec_isclose(uv2, uvp2):
            self.flag |= PEDGE_SEAM
            e_pair.flag |= PEDGE_SEAM
            return True

        return False

    def edge_connect_pair(self, handle: 'ParamHandleConstruct', stack: 'list[PEdge]') -> bool:
        pair: list[PEdge | None] = [None]

        if not self.pair and self.has_pair(handle, pair):
            pair: PEdge | None = pair[0]
            if self.vert == pair.vert:
                pair.face.flip()

            self.pair = pair
            pair.pair = self

            if not (pair.face.flag & PFACE_CONNECTED):
                stack.append(pair)

        return bool(self.pair)

    def has_pair(self, handle: 'ParamHandleConstruct', r_pair: list) -> bool:
        """r_pair[0] - is imitation pointer of pointer."""
        if self.flag & PEDGE_SEAM:
            return False

        key1 = self.vert.key
        key2 = self.next.vert.key
        key = PHASH_edge(key1, key2)

        pair_e = handle.hash_edges.lookup(key)
        r_pair[0] = None  # TODO: Replace with tuple

        while pair_e:
            if pair_e is not self:
                v1 = pair_e.vert
                v2 = pair_e.next.vert

                if ((v1.key == key1 and v2.key == key2) or
                        (v1.key == key2 and v2.key == key1)):

                    # don't connect seams and t-junctions
                    if ((pair_e.flag & PEDGE_SEAM) or r_pair[0] or
                            (UnwrapOptions.topology_from_uvs and self.implicit_seam(pair_e))):
                        r_pair[0] = None
                        return False

                    r_pair[0] = pair_e

            pair_e = handle.hash_edges.next(key, pair_e)

        if r_pair[0] and self.vert == r_pair[0].vert:
            if r_pair[0].next.pair or r_pair[0].next.next.pair:
                # non-unfoldable, maybe mobius ring or klein bottle
                r_pair[0] = None
                return False

        return r_pair[0] is not None

    def boundary_angle(self) -> float:
        v: PVert = self.vert

        # concave angle check -- could be better
        angle: float = pi

        we: PEdge = v.edge
        while True:
            v1: PVert = we.next.vert
            v2: PVert = we.next.next.vert
            angle -= angle_v3v3v3(v1.co, v.co, v2.co)

            we = we.next.next.pair
            if not (we and (we != v.edge)):
                break

        return angle


class PFace:
    def __init__(self):
        self.nextlink: PFace | None = None
        # ABF/LSCM matrix index
        self.id: int | float = 0  # noqa
        self.edge: PEdge | None = None
        self.flag: int = 0

    @classmethod
    def new(cls):
        f = PFace()

        e1 = PEdge()
        e2 = PEdge()
        e3 = PEdge()

        # set up edges */
        f.edge = e1
        e1.face = e2.face = e3.face = f

        e1.next = e2
        e2.next = e3
        e3.next = e1

        return f

    @property
    def area3d(self) -> float:
        """Edge collapsing."""
        return self.id

    @area3d.setter
    def area3d(self, v: float):
        self.id = v

    @property
    def key(self):
        """Construct."""
        return self.id

    @key.setter
    def key(self, v):
        self.id = v

    @property
    def chart(self) -> int:
        """Construct splitting."""
        return self.id

    @chart.setter
    def chart(self, v: int):
        self.id = v

    def flip(self):
        e1 = self.edge
        e2 = e1.next
        e3 = e2.next

        v1 = e1.vert
        v2 = e2.vert
        v3 = e3.vert

        f1 = e1.flag
        f2 = e2.flag
        f3 = e3.flag

        orig_uv1 = e1.orig_uv
        orig_uv2 = e2.orig_uv
        orig_uv3 = e3.orig_uv

        e1.vert = v2
        e1.next = e3
        e1.orig_uv = orig_uv2
        e1.flag = (f1 & ~PEDGE_VERTEX_FLAGS) | (f2 & PEDGE_VERTEX_FLAGS)

        e2.vert = v3
        e2.next = e1
        e2.orig_uv = orig_uv3
        e2.flag = (f2 & ~PEDGE_VERTEX_FLAGS) | (f3 & PEDGE_VERTEX_FLAGS)

        e3.vert = v1
        e3.next = e2
        e3.orig_uv = orig_uv1
        e3.flag = (f3 & ~PEDGE_VERTEX_FLAGS) | (f1 & PEDGE_VERTEX_FLAGS)

    def backup_uvs(self):

        e1: PEdge = self.edge
        e2: PEdge = e1.next
        e3: PEdge = e2.next

        if e1.orig_uv:
            e1.old_uv = e1.orig_uv.copy()

        if e2.orig_uv:
            e2.old_uv = e2.orig_uv.copy()

        if e3.orig_uv:
            e3.old_uv = e3.orig_uv.copy()

    def calc_angles(self):

        e1: PEdge = self.edge
        e2: PEdge = e1.next
        e3: PEdge = e2.next
        v1: PVert = e1.vert
        v2: PVert = e2.vert
        v3: PVert = e3.vert

        return p_triangle_angles(v1.co, v2.co, v3.co)

    def calc_signed_uv_area(self) -> float:

        e1: PEdge = self.edge
        e2: PEdge = e1.next
        e3: PEdge = e2.next
        v1: PVert = e1.vert
        v2: PVert = e2.vert
        v3: PVert = e3.vert

        return 0.5 * (v2.uv - v1.uv).cross(v3.uv - v1.uv)

    @classmethod
    def add_fill(cls, chart, v1: PVert, v2: PVert, v3: PVert) -> 'PFace':

        f: PFace = cls.new()
        e1: PEdge = f.edge
        e2: PEdge = e1.next
        e3: PEdge = e2.next

        e1.vert = v1
        e2.vert = v2
        e3.vert = v3

        e1.orig_uv = e2.orig_uv = e3.orig_uv = None

        f.nextlink = chart.faces.first_item
        chart.faces.first_item = f
        e1.nextlink = chart.edges.first_item
        chart.edges.first_item = e1
        e2.nextlink = chart.edges.first_item
        chart.edges.first_item = e2
        e3.nextlink = chart.edges.first_item
        chart.edges.first_item = e3

        chart.n_faces += 1
        chart.n_edges += 3

        return f


class PChart:
    def __init__(self):
        self.verts: ParametrizerIt[PVert] = ParametrizerIt()
        self.edges: ParametrizerIt[PEdge] = ParametrizerIt()
        self.faces: ParametrizerIt[PFace] = ParametrizerIt()

        self.n_verts: int = 0
        self.n_edges: int = 0
        self.n_faces: int = 0
        self.n_boundaries: int = 0

        self.collapsed_verts: ParametrizerIt[PVert] = ParametrizerIt()
        self.collapsed_edges: ParametrizerIt[PEdge] = ParametrizerIt()
        self.collapsed_faces: ParametrizerIt[PFace] = ParametrizerIt()

        self.area_uv: float = 0.0
        self.area_3d: float = 0.0

        self.origin: Vector = Vector((0.0, 0.0))

        self.context: LinearSolver | None = None
        self.abf_alpha: list[float]  = [] # list of alpha ???

        self.pin1: PVert | None = None
        self.pin2: PVert | None = None
        self.single_pin: PVert | None = None

        self.has_pins: bool = False
        self.skip_flush: bool = False

    def boundaries(self):
        max_length = -1.0
        outer = None
        self.n_boundaries = 0
        for e in self.edges:
            if e.pair or (e.flag & PEDGE_DONE):
                continue

            self.n_boundaries += 1

            length = 0.0
            be = e
            while True:
                be.flag |= PEDGE_DONE
                length += be.length_3d
                be = be.next.vert.edge  # noqa
                if be == e:
                    break

            if length > max_length:
                outer = e
                max_length = length


        for e in self.edges:
            e.flag &= ~PEDGE_DONE

        return outer

    def ensure_area_uv(self):
        from mathutils.geometry import area_tri
        total_area = 0.0
        for f in self.faces:
            e1 = f.edge
            e2 = e1.next
            e3 = e2.next

            total_area += area_tri(e1.vert.uv, e2.vert.uv, e3.vert.uv)
        self.area_uv = total_area
        return total_area

    def ensure_area_3d(self, store_in_face=False):
        from mathutils.geometry import area_tri
        total_area = 0.0
        if store_in_face:
            for f in self.faces:
                e1 = f.edge
                e2 = e1.next
                e3 = e2.next

                area = area_tri(e1.vert.co, e2.vert.co, e3.vert.co)
                f.id = area  # NOTE: Optimized save to avoid area_3d property use
                total_area += area
        else:
            for f in self.faces:
                e1 = f.edge
                e2 = e1.next
                e3 = e2.next

                total_area += area_tri(e1.vert.co, e2.vert.co, e3.vert.co)
        self.area_3d = total_area
        return total_area

    def split_charts(self, ncharts):
        charts: list[PChart] = [PChart() for _ in range(ncharts)]

        f = self.faces.first_item
        while f:
            e1 = f.edge
            e2 = e1.next
            e3 = e2.next
            next_f = f.nextlink

            n_chart = charts[f.chart]

            # insert face at the beginning of the list of faces
            f.nextlink = n_chart.faces.first_item
            n_chart.faces.first_item = f

            # insert all 3 edges at the beginning of the edges list
            e1.nextlink = n_chart.edges.first_item
            n_chart.edges.first_item = e1

            e2.nextlink = n_chart.edges.first_item
            n_chart.edges.first_item = e2

            e3.nextlink = n_chart.edges.first_item
            n_chart.edges.first_item = e3

            n_chart.n_faces += 1
            n_chart.n_edges += 3

            # split verts
            n_chart.split_vert(e1)
            n_chart.split_vert(e2)
            n_chart.split_vert(e3)

            f = next_f

        return charts

    def split_vert(self, e: PEdge):
        v = e.vert
        copy = True

        if e.flag & PEDGE_PIN:
            self.has_pins = True

        if e.flag & PEDGE_VERTEX_SPLIT:
            return

        # rewind to start
        lastwe = e
        we = e.wheel_edge_prev
        while we and we != e:
            lastwe = we
            we = we.wheel_edge_prev

        # go over all edges in wheel
        we = lastwe
        while we:
            if we.flag & PEDGE_VERTEX_SPLIT:
                break

            we.flag |= PEDGE_VERTEX_SPLIT

            if we == v.edge:
                # found it, no need to copy
                copy = False
                v.nextlink = self.verts.first_item
                self.verts.first_item = v
                self.n_verts += 1
            we = we.wheel_edge_next

        if copy:
            # not found, copying
            v.flag |= PVERT_SPLIT
            v = v.copy()
            v.flag |= PVERT_SPLIT

            v.nextlink = self.verts.first_item
            self.verts.first_item = v
            self.n_verts += 1

            v.edge = lastwe

            we = lastwe
            while True:
                we.vert = v
                we = we.wheel_edge_next
                if not we or we == lastwe:
                    break

    def lscm_begin(self):

        assert self.context is None

        pins: list[PVert] = [v for v in self.verts if v.flag & PVERT_PIN]

    #if 0
        # p_chart_simplify_compute(chart, p_collapse_cost, p_collapse_allowed)
        # p_chart_topological_sanity_check(chart)
    #endif

        if len(pins) == 1:
            self.ensure_area_uv()
            self.single_pin = pins[0]

        if UnwrapOptions.use_abf:
            if not self.abf_solve():
                print("ABF solving failed: falling back to LSCM.")

        # ABF uses these indices for its internal references.
        # Set the indices afterward.
        for idx, v in enumerate(self.verts):
            v.id = idx


        if UnwrapOptions.unwrap_along == 'UV':
            if len(pins) <= 1:
                # No pins, let's find some ourselves.

                outer: PEdge = self.boundaries()

                pin1: list[PVert | None] = [None]
                pin2: list[PVert | None] = [None]
                # Outer can be null with non-finite coordinates.
                if not (outer and self.symmetry_pins(outer, pin1, pin2)):
                    self.extrema_verts(pin1, pin2)


                self.pin1 = pin1[0]
                self.pin2 = pin2[0]

        self.context = LinearSolver.new(2 * self.n_faces, 2 * self.n_verts, least_squares=True)

    def lscm_solve(self) -> bool:
        context: LinearSolver = self.context

        for v in self.verts:
            if v.flag & PVERT_PIN:
                v.load_pin_select_uvs() # Reload for Live Unwrap.

        if self.single_pin:
            # If only one pin, save location as origin.
            self.origin = self.single_pin.uv.copy()
        #
        if self.pin1:
            pin1: PVert = self.pin1
            pin2: PVert = self.pin2
            context.lock_variable(2 * pin1.id, pin1.uv[0])
            context.lock_variable(2 * pin1.id + 1, pin1.uv[1])
            context.lock_variable(2 * pin2.id, pin2.uv[0])
            context.lock_variable(2 * pin2.id + 1, pin2.uv[1])

        else:
            # Set and lock the pins.
            for v in self.verts:
                if v.flag & PVERT_PIN:
                    context.lock_variable(2 * v.id, v.uv[0])
                    context.lock_variable(2 * v.id + 1, v.uv[1])

        # Lock axis
        if UnwrapOptions.unwrap_along == 'V':
            for v in self.verts:
                context.lock_variable(2 * v.id, v.uv[0])
        elif UnwrapOptions.unwrap_along == 'U':
            for v in self.verts:
                context.lock_variable(2 * v.id + 1, v.uv[1])

        # Detect "up" direction based on pinned vertices.
        area_pinned_up: float = 0.0
        area_pinned_down: float = 0.0

        for f in self.faces:
            e1: PEdge = f.edge
            e2: PEdge = e1.next
            e3: PEdge = e2.next
            v1: PVert = e1.vert
            v2: PVert = e2.vert
            v3: PVert = e3.vert

            if (v1.flag & PVERT_PIN) and (v2.flag & PVERT_PIN) and (v3.flag & PVERT_PIN):
                area: float = f.calc_signed_uv_area()

                if area > 0.0:
                    area_pinned_up += area
                else:
                    area_pinned_down -= area

        flip_faces: bool = (area_pinned_down > area_pinned_up)

        # Construct matrix.
        alpha: list[float] = self.abf_alpha

        row: int = 0
        ii = 0
        for f in self.faces:
            e1: PEdge = f.edge
            e2: PEdge = e1.next
            e3: PEdge = e2.next
            v1: PVert = e1.vert
            v2: PVert = e2.vert
            v3: PVert = e3.vert

            if alpha:
                # Use abf angles if present.
                a1 = alpha[ii]
                ii += 1
                a2 = alpha[ii]
                ii += 1
                a3 = alpha[ii]
                ii += 1
            else:
                a1, a2, a3 = f.calc_angles()

            if flip_faces:
                # swap
                a2, a3 = a3, a2
                # e2, e3 = e3, e2
                v2, v3 = v3, v2

            context.matrix_add_angles(row, a1, a2, a3, v1.id, v2.id, v3.id)
            row += 2

        if context.solve():
            for v in self.verts:
                v.uv[0] = context.variable_get(2 * v.id)
                v.uv[1] = context.variable_get(2 * v.id + 1)
            return True

        for v in self.verts:
            v.uv.xy = (0.0, 0.0)

        return False

    def abf_solve(self) -> bool:
        from math import pi
        sys = PAbfSystem()
        limit: float = 1.0 if (self.n_faces > 100) else 0.001
        # lastnorm: float = 1.0 if (chart.n_faces > 100) else 0.001

        for v in self.verts:
            if v.is_interior:
                v.flag |= PVERT_INTERIOR
                v.id = sys.n_interior
                sys.n_interior += 1

            else:
                v.flag &= ~PVERT_INTERIOR

        for f in self.faces:
            e1 = f.edge
            e2 = e1.next
            e3 = e2.next
            f.id = sys.n_faces
            sys.n_faces += 1

            # angle ids are conveniently stored in half edges
            e1.id = sys.n_angles
            sys.n_angles +=1
            e2.id = sys.n_angles
            sys.n_angles +=1
            e3.id = sys.n_angles
            sys.n_angles +=1


        sys.p_abf_setup_system()

        # compute initial angles
        for f in self.faces:
            e1 = f.edge
            e2 = e1.next
            e3 = e2.next
            a1, a2, a3 = f.calc_angles()

            sys.alpha[e1.id] = sys.beta[e1.id] = a1
            sys.alpha[e2.id] = sys.beta[e2.id] = a2
            sys.alpha[e3.id] = sys.beta[e3.id] = a3

            sys.weight[e1.id] = 2.0 / (a1 * a1)
            sys.weight[e2.id] = 2.0 / (a2 * a2)
            sys.weight[e3.id] = 2.0 / (a3 * a3)


        for v in self.verts:
            if v.flag & PVERT_INTERIOR:
                angle_sum: float = 0.0

                e: PEdge = v.edge
                while True:
                    angle_sum += sys.beta[e.id]
                    e = e.next.next.pair
                    if not (e and (e != v.edge)):
                        break

                scale: float = 2.0 * pi / angle_sum if angle_sum else 0.0  # safe divide

                e = v.edge
                while True:
                    sys.beta[e.id] = sys.alpha[e.id] = sys.beta[e.id] * scale
                    e = e.next.next.pair
                    if not (e and (e != v.edge)):
                        break

        if sys.n_interior > 0:
            sys.compute_sines()

            # iteration
            # lastnorm = 1e10 /* UNUSED.

            for i in range(PAbfSystem.ABF_MAX_ITER):
                norm: float = sys.compute_gradient(self)

                # lastnorm = norm /* UNUSED.
                if norm < limit:
                    break


                if not sys.matrix_invert(self):
                    print("UniV: ABF failed to invert matrix")
                    # p_abf_free_system(sys)
                    return False

                sys.compute_sines()

            else:
                print("UniV: ABF maximum iterations reached")
                # p_abf_free_system(sys)
                return False

        self.abf_alpha = sys.alpha
        sys.alpha = []

        return True

    def symmetry_pins(self, outer: PEdge, pin1: list[PVert], pin2: list[PVert]) -> bool:
        max_e1: PEdge | None= None
        max_e2: PEdge | None= None
        cure: PEdge | None = None
        first_e1: PEdge | None = None
        first_e2: PEdge | None = None
        max_len: float = 0.0
        cur_len: float = 0.0
        tot_len: float = 0.0
        first_len: float = 0.0

        # find the longest series of verts split in the chart itself, these are
        # marked during construction
        be: PEdge = outer
        last_be: PEdge = be.boundary_edge_prev
        while True:
            tot_len += be.length_3d
            next_be: PEdge = be.boundary_edge_next

            if (be.vert.flag & PVERT_SPLIT) or (last_be.vert.flag & next_be.vert.flag & PVERT_SPLIT):
                if not cure:
                    if be == outer:
                        first_e1 = be
                    cure = be
                else:
                    cur_len += last_be.length_3d
            elif cure:
                if cur_len > max_len:
                    max_len = cur_len
                    max_e1 = cure
                    max_e2 = last_be
                if first_e1 == cure:
                    first_len = cur_len
                    first_e2 = last_be
                cur_len = 0.0
                cure = None

            last_be = be
            be = next_be
            if be == outer:
                break

        # make sure we also count a series of splits over the starting point
        if cure and (cure != outer):
            first_len += cur_len + be.length_3d

            if first_len > max_len:
                max_len = first_len
                max_e1 = cure
                max_e2 = first_e2

        if not max_e1 or not max_e2 or (max_len < 0.5 * tot_len):
            return False

        # find pin1 in the split vertices
        be1: PEdge = max_e1
        be2: PEdge = max_e2
        len1: float = 0.0
        len2: float = 0.0

        while True:
            if len1 < len2:
                len1 += be1.length_3d
                be1 = be1.boundary_edge_next

            else:
                be2 = be2.boundary_edge_prev
                len2 += be2.length_3d

            if not (be1 != be2):
                break

        pin1[0] = be1.vert

        # find pin2 outside the split vertices
        be1 = max_e1
        be2 = max_e2
        len1 = 0.0
        len2 = 0.0

        while True:
            if len1 < len2:
                be1 = be1.boundary_edge_prev
                len1 += be1.length_3d

            else:
                len2 += be2.length_3d
                be2 = be2.boundary_edge_next

            if not (be1 != be2):
                break

        pin2[0] = be1.vert

        self.pin_positions(pin1, pin2)

        return not pin1[0].co == pin2[0].co

    def pin_positions(self, pin1: list[PVert], pin2: list[PVert]):

        if not pin1[0] or not pin2[0] or pin1[0] == pin2[0]:
            # degenerate case
            f: PFace = self.faces.first_item
            pin1[0] = f.edge.vert
            pin2[0] = f.edge.next.vert

            pin1[0].uv[0] = 0.0
            pin1[0].uv[1] = 0.5
            pin2[0].uv[0] = 1.0
            pin2[0].uv[1] = 0.5
            # raise  # TODO: Test

        else:
            sub = pin1[0].co - pin2[0].co
            sub[0] = abs(sub[0])
            sub[1] = abs(sub[1])
            sub[2] = abs(sub[2])

            if (sub[0] > sub[1]) and (sub[0] > sub[2]):
                dir_x = 0
                dir_y = 1 if (sub[1] > sub[2]) else 2
            elif (sub[1] > sub[0]) and (sub[1] > sub[2]):
                dir_x = 1
                dir_y = 0 if (sub[0] > sub[2]) else 2
            else:
                dir_x = 2
                dir_y = 0 if (sub[0] > sub[1]) else 1


            if dir_x == 2:
                dir_u = 1
                dir_v = 0

            else:
                dir_u = 0
                dir_v = 1

            pin1[0].uv[dir_u] = pin1[0].co[dir_x]
            pin1[0].uv[dir_v] = pin1[0].co[dir_y]
            pin2[0].uv[dir_u] = pin2[0].co[dir_x]
            pin2[0].uv[dir_v] = pin2[0].co[dir_y]

    def extrema_verts(self, pin1: list[PVert], pin2: list[PVert]):
        # find minimum and maximum verts over x/y/z axes
        min_v: Vector = Vector((1e20, 1e20, 1e20))
        max_v: Vector = Vector((-1e20, -1e20, -1e20))

        min_vert: list[PVert] = [None, None, None]
        max_vert: list[PVert] = [None, None, None]

        for v in self.verts:
            for i in range(3):
                if v.co[i] < min_v[i]:
                    min_v[i] = v.co[i]
                    min_vert[i] = v

                if v.co[i] > max_v[i]:
                    max_v[i] = v.co[i]
                    max_vert[i] = v

        # find axes with the longest distance
        dir_: int = 0
        dir_len: float = -1.0

        for i in range(3):
            if max_v[i] - min_v[i] > dir_len:
                dir_ = i
                dir_len = max_v[i] - min_v[i]

        pin1[0] = min_vert[dir_]
        pin2[0] = max_vert[dir_]

        self.pin_positions(pin1, pin2)

    def fill_boundaries(self, outer: PEdge):
        for e in self.edges:
            # e_next = e.nextlink - as yet unused

            if e.pair or (e.flag & PEDGE_FILLED):
                continue

            n_edges: int = 0
            be: PEdge = e
            while True:
                be.flag |= PEDGE_FILLED
                be = be.next.vert.edge
                n_edges += 1
                if not (be != e):
                    break

            if e != outer:
                self.fill_boundary(e, n_edges)

    def fill_boundary(self, be: PEdge, n_edges: int):
        heap = []

        # Initial insertion of all boundary edges
        e: PEdge = be
        while True:
            angle = e.boundary_angle()
            item = HeapItem(angle, e)
            e.heaplink = item
            heapq.heappush(heap, item)

            e = e.boundary_edge_next
            if e == be:
                break

        # Isolated seam case (2 edges)
        if n_edges == 2:
            e = be.next.vert.edge

            e.pair = be
            be.pair = e

            # lazily mark elements as deleted
            e.heaplink.removed = True
            be.heaplink.removed = True
            return

        # General case: fill boundary
        while n_edges > 2:
            # pop with lazy deletion
            while True:
                item = heapq.heappop(heap)
                if not item.removed:
                    break
            e = item.edge

            e1 = e.boundary_edge_prev
            e2 = e.boundary_edge_next

            # remove e1 and e2
            e1.heaplink.removed = True
            e2.heaplink.removed = True

            e.flag |= PEDGE_FILLED
            e1.flag |= PEDGE_FILLED

            f = PFace.add_fill(self, e.vert, e1.vert, e2.vert)
            f.flag |= PFACE_FILLED

            # new edges
            ne = f.edge.next.next
            ne1 = f.edge
            ne2 = f.edge.next

            ne.flag = ne1.flag = ne2.flag = PEDGE_FILLED

            e.pair = ne
            ne.pair = e
            e1.pair = ne1
            ne1.pair = e1

            ne.vert = e2.vert
            ne1.vert = e.vert
            ne2.vert = e1.vert

            if n_edges == 3:
                e2.pair = ne2
                ne2.pair = e2
            else:
                ne2.vert.edge = ne2

                # put ne2 and e2 back into the heap
                it1 = HeapItem(ne2.boundary_angle(), ne2)
                it2 = HeapItem(e2.boundary_angle(), e2)

                ne2.heaplink = it1
                e2.heaplink = it2

                heapq.heappush(heap, it1)
                heapq.heappush(heap, it2)

            n_edges -= 1


class PAbfSystem:
    ABF_MAX_ITER = 20

    def __init__(self):
        # N Interior - is the number of internal vertices for which the system is solved (where the sum of the angles is ≈ 2π).
        self.n_interior: int = 0
        # Total tris.
        self.n_faces: int = 0
        # The total number of corners (3 × n_faces).
        self.n_angles: int = 0

        # Alpha - is the current angle (optimization variable).
        # These are the angles that the algorithm corrects so that the scan turns out to be 'flat'.
        self.alpha: list[float] = []
        # Beta - is the target angle value (from the 3D model). That is, the initial angle between the edges in 3D.
        self.beta: list[float] = []

        # Sine, Cosine - are the precalculated values of the sine/cosine of the angles' alpha.
        self.sine: list[float] = []
        self.cosine: list[float] = []

        # Weight - is the weight of the angle in the objective function
        # (usually depends on the length of the edges and the area of the triangle).
        # More important angles contribute more to optimization.
        self.weight: list[float] = []

        # These are residual vectors for various types of constraints in a system of linear equations

        # bAlpha - residual according to the “angle difference” equation (α - β), i.e., how much the current angles deviate from the target angles.
        self.bAlpha: list[float] = []
        # bTriangle - the residual according to the equations of the sum of angles in a triangle (should be π).
        self.bTriangle: list[float] = []
        # bInterior - the residual according to the equations of the sum of angles around a vertex (should be 2π for interior angles, < 2π for boundary angles).
        self.bInterior: list[float] = []

        # These λ are Lagrange coefficients that ensure the fulfillment of geometric constraints:

        # lambdaTriangle - multipliers for triangle angle sum constraints.
        self.lambdaTriangle: list[float] = []
        # lambdaPlanar - multipliers for planarity constraints (that a triangle can be unfolded in 2D without self-intersection).
        self.lambdaPlanar: list[float] = []
        # lambdaLength - multipliers for edge length matching constraints (so that the sides match when triangles are joined).
        self.lambdaLength: list[float] = []

        # These fields refer to the system of linear equations solved at each iteration step:

        # J2dt - Jacobian (matrix of partial derivatives), with dimensions [n_angles][3], describes the relationship between angle changes and constraints.
        self.J2dt: list[Vector] = []
        # bstar - right-hand side for the reduced system (after eliminating dependencies).
        self.bstar: list[float] = []
        # dstar - result of solving the linear system (changes in variables α, λ, etc.).
        self.dstar: list[float] = []


    def p_abf_setup_system(self):
        # NOTE: Use np.empty
        self.alpha = self.n_angles * [None]
        self.beta = self.n_angles * [None]
        self.sine = self.n_angles * [None]
        self.cosine = self.n_angles * [None]
        self.weight = self.n_angles * [None]

        self.bAlpha = self.n_angles * [None]
        self.bTriangle = self.n_faces * [None]
        self.bInterior = self.n_interior * 2 * [None]

        self.lambdaTriangle = self.n_faces * [0.0]
        self.lambdaPlanar = self.n_interior * [0.0]
        self.lambdaLength = self.n_interior * [0.0]

        self.J2dt = [Vector() for _ in range(self.n_angles)]
        self.bstar = self.n_faces * [None]
        self.dstar = self.n_faces * [None]

        for i in range(self.n_interior):
            self.lambdaLength[i] = 1.0

    def __str__(self):
        return (f"{self.n_interior = }\n {self.n_angles = } \n {self.n_faces = }\n "
                f"{self.alpha = }\n {self.beta = } \n {self.sine = }\n {self.cosine = }\n {self.weight = } "
                f"\n {self.bAlpha = }\n {self.bTriangle = } {self.bInterior = }\n {self.lambdaTriangle = }\n "
                f"{self.lambdaPlanar = }\n {self.lambdaLength = }\n {self.J2dt = }\n {self.bstar = }\n {self.dstar = }")

    def compute_sines(self):
        import numpy as np
        eix = np.exp(1j *  np.array(self.alpha))
        self.sine = eix.imag.tolist()
        self.cosine = eix.real.tolist()

    def compute_sin_product(self, v: PVert, aid: int) -> float:
        sin1 = 1.0
        sin2 = 1.0
        cosine = self.cosine
        sine = self.sine
        e: PEdge = v.edge
        while True:
            e1: PEdge = e.next
            e2: PEdge = e1.next

            if (e1_id := e1.id) == aid:
                # we are computing a derivative for this angle,
                # so we use cos and drop the other part
                sin1 *= cosine[e1_id]
                sin2 = 0.0
            else:
                sin1 *= sine[e1_id]

            if (e2_id := e2.id) == aid:
                # see above
                sin1 = 0.0
                sin2 *= cosine[e2_id]
            else:
                sin2 *= sine[e2_id]

            e = e2.pair
            if not e or e == v.edge:
                break

        return sin1 - sin2

    def compute_grad_alpha(self, f: PFace, e: PEdge) -> float:
        e_id = e.id
        v: PVert = e.vert
        v1: PVert = e.next.vert
        v2: PVert = e.next.next.vert

        deriv: float = (self.alpha[e_id] - self.beta[e_id]) * self.weight[e_id]
        deriv += self.lambdaTriangle[f.id]

        if v.flag & PVERT_INTERIOR:
            deriv += self.lambdaPlanar[v.id]


        if v1.flag & PVERT_INTERIOR:
            product: float = self.compute_sin_product(v1, e_id)
            deriv += self.lambdaLength[v1.id] * product


        if v2.flag & PVERT_INTERIOR:
            product: float = self.compute_sin_product(v2, e_id)
            deriv += self.lambdaLength[v2.id] * product

        return deriv

    def compute_gradient(self, chart: PChart) -> float:
        from math import pi
        norm: float = 0.0

        for f in chart.faces:
            e1: PEdge = f.edge
            e2: PEdge = e1.next
            e3: PEdge = e2.next

            g_alpha1: float = self.compute_grad_alpha(f, e1)
            g_alpha2: float = self.compute_grad_alpha(f, e2)
            g_alpha3: float = self.compute_grad_alpha(f, e3)

            self.bAlpha[e1.id] = -g_alpha1
            self.bAlpha[e2.id] = -g_alpha2
            self.bAlpha[e3.id] = -g_alpha3

            norm += g_alpha1 * g_alpha1 + g_alpha2 * g_alpha2 + g_alpha3 * g_alpha3

            g_triangle: float = self.alpha[e1.id] + self.alpha[e2.id] + self.alpha[e3.id] - pi
            self.bTriangle[f.id] = -g_triangle
            norm += g_triangle * g_triangle


        for v in chart.verts:
            if v.flag & PVERT_INTERIOR:
                g_planar: float = -2 * pi

                e: PEdge = v.edge
                while True:
                    g_planar += self.alpha[e.id]
                    e = e.next.next.pair
                    if not e or e == v.edge:
                        break

                self.bInterior[v.id] = -g_planar
                norm += g_planar * g_planar

                g_length: float = self.compute_sin_product(v, -1)
                self.bInterior[self.n_interior + v.id] = -g_length
                norm += g_length * g_length

        return norm

    def adjust_alpha(self, id_: int, d_lambda1: float, pre: float):
        alpha: float = self.alpha[id_]
        dalpha: float = (self.bAlpha[id_] - d_lambda1)
        alpha += dalpha / self.weight[id_] - pre
        self.alpha[id_] = clamp(alpha, 0.0, pi)


    def matrix_invert(self, chart: PChart) -> bool:

        n_interior: int = self.n_interior
        n_var: int = 2 * n_interior
        context: LinearSolver = LinearSolver.new(0, n_var, 1)

        for i in range(n_var):
            context.right_hand_side_add(i, self.bInterior[i])


        for f in chart.faces:
            beta: Vector = Vector()
            j2: Matrix = Matrix.Identity(3)

            row1: Vector = Vector.Fill(6, 0.0)
            row2: Vector = Vector.Fill(6, 0.0)
            row3: Vector = Vector.Fill(6, 0.0)
            vid: list[int] = [-1] * 6

            e1: PEdge = f.edge
            e2: PEdge = e1.next
            e3: PEdge = e2.next
            v1: PVert = e1.vert
            v2: PVert = e2.vert
            v3: PVert = e3.vert

            wi1: float = 1.0 / self.weight[e1.id]
            wi2: float = 1.0 / self.weight[e2.id]
            wi3: float = 1.0 / self.weight[e3.id]

            # bstar1 = (J1 dInv*bAlpha - bTriangle)
            b: float = self.bAlpha[e1.id] * wi1
            b += self.bAlpha[e2.id] * wi2
            b += self.bAlpha[e3.id] * wi3
            b -= self.bTriangle[f.id]

            # si = J1 d*J1t
            si: float = 1.0 / (wi1 + wi2 + wi3)

            # J1t si*bstar1 - bAlpha
            beta[0] = b * si - self.bAlpha[e1.id]
            beta[1] = b * si - self.bAlpha[e2.id]
            beta[2] = b * si - self.bAlpha[e3.id]

            # use this later for computing other lambda's
            self.bstar[f.id] = b
            self.dstar[f.id] = si

            # set matrix
            W = Matrix([[si] * 3] * 3)

            W[0][0] = si - self.weight[e1.id]
            W[1][1] = si - self.weight[e2.id]
            W[2][2] = si - self.weight[e3.id]

            if v1.flag & PVERT_INTERIOR:
                vid[0] = v1.id
                vid[3] = n_interior + v1.id

                self.J2dt[e1.id][0] = j2[0][0] = 1.0 * wi1
                self.J2dt[e2.id][0] = j2[1][0] = self.compute_sin_product(v1, e2.id) * wi2
                self.J2dt[e3.id][0] = j2[2][0] = self.compute_sin_product(v1, e3.id) * wi3

                context.right_hand_side_add(v1.id, j2[0][0] * beta[0])
                context.right_hand_side_add(n_interior + v1.id, j2[1][0] * beta[1] + j2[2][0] * beta[2])

                row1[0] = j2[0][0] * W[0][0]
                row2[0] = j2[0][0] * W[1][0]
                row3[0] = j2[0][0] * W[2][0]

                row1[3] = j2[1][0] * W[0][1] + j2[2][0] * W[0][2]
                row2[3] = j2[1][0] * W[1][1] + j2[2][0] * W[1][2]
                row3[3] = j2[1][0] * W[2][1] + j2[2][0] * W[2][2]


            if v2.flag & PVERT_INTERIOR:
                vid[1] = v2.id
                vid[4] = n_interior + v2.id

                self.J2dt[e1.id][1] = j2[0][1] = self.compute_sin_product(v2, e1.id) * wi1
                self.J2dt[e2.id][1] = j2[1][1] = 1.0 * wi2
                self.J2dt[e3.id][1] = j2[2][1] = self.compute_sin_product(v2, e3.id) * wi3

                context.right_hand_side_add(v2.id, j2[1][1] * beta[1])
                context.right_hand_side_add(n_interior + v2.id, j2[0][1] * beta[0] + j2[2][1] * beta[2])

                row1[1] = j2[1][1] * W[0][1]
                row2[1] = j2[1][1] * W[1][1]
                row3[1] = j2[1][1] * W[2][1]

                row1[4] = j2[0][1] * W[0][0] + j2[2][1] * W[0][2]
                row2[4] = j2[0][1] * W[1][0] + j2[2][1] * W[1][2]
                row3[4] = j2[0][1] * W[2][0] + j2[2][1] * W[2][2]


            if v3.flag & PVERT_INTERIOR:
                vid[2] = v3.id
                vid[5] = n_interior + v3.id

                self.J2dt[e1.id][2] = j2[0][2] = self.compute_sin_product(v3, e1.id) * wi1
                self.J2dt[e2.id][2] = j2[1][2] = self.compute_sin_product(v3, e2.id) * wi2
                self.J2dt[e3.id][2] = j2[2][2] = 1.0 * wi3

                context.right_hand_side_add(v3.id, j2[2][2] * beta[2])
                context.right_hand_side_add(n_interior + v3.id, j2[0][2] * beta[0] + j2[1][2] * beta[1])

                row1[2] = j2[2][2] * W[0][2]
                row2[2] = j2[2][2] * W[1][2]
                row3[2] = j2[2][2] * W[2][2]

                row1[5] = j2[0][2] * W[0][0] + j2[1][2] * W[0][1]
                row2[5] = j2[0][2] * W[1][0] + j2[1][2] * W[1][1]
                row3[5] = j2[0][2] * W[2][0] + j2[1][2] * W[2][1]


            for i in range(3):
                r: int = vid[i]

                if r == -1:
                    continue

                for j in range(6):
                    c: int = vid[j]

                    if c == -1:
                        continue

                    if i == 0:
                        context.matrix_add(r, c, j2[0][i] * row1[j])
                    else:
                        context.matrix_add(r + n_interior, c, j2[0][i] * row1[j])

                    if i == 1:
                        context.matrix_add(r, c, j2[1][i] * row2[j])
                    else:
                        context.matrix_add(r + n_interior, c, j2[1][i] * row2[j])


                    if i == 2:
                        context.matrix_add(r, c, j2[2][i] * row3[j])
                    else:
                        context.matrix_add(r + n_interior, c, j2[2][i] * row3[j])

        if success := context.solve():
            for f in chart.faces:
                e1: PEdge = f.edge
                e2: PEdge = e1.next
                e3: PEdge = e2.next
                v1: PVert = e1.vert
                v2: PVert = e2.vert
                v3: PVert = e3.vert

                pre: Vector = Vector()

                if v1.flag & PVERT_INTERIOR:
                    x: float = context.variable_get(v1.id)
                    x2: float = context.variable_get(n_interior + v1.id)
                    pre[0] += self.J2dt[e1.id][0] * x
                    pre[1] += self.J2dt[e2.id][0] * x2
                    pre[2] += self.J2dt[e3.id][0] * x2

                if v2.flag & PVERT_INTERIOR:
                    x: float = context.variable_get(v2.id)
                    x2: float = context.variable_get(n_interior + v2.id)
                    pre[0] += self.J2dt[e1.id][1] * x2
                    pre[1] += self.J2dt[e2.id][1] * x
                    pre[2] += self.J2dt[e3.id][1] * x2

                if v3.flag & PVERT_INTERIOR:
                    x: float = context.variable_get(v3.id)
                    x2: float = context.variable_get(n_interior + v3.id)
                    pre[0] += self.J2dt[e1.id][2] * x2
                    pre[1] += self.J2dt[e2.id][2] * x2
                    pre[2] += self.J2dt[e3.id][2] * x

                d_lambda1 = self.dstar[f.id] * (self.bstar[f.id] - sum(pre))
                self.lambdaTriangle[f.id] += d_lambda1

                self.adjust_alpha(e1.id, d_lambda1, pre[0])
                self.adjust_alpha(e2.id, d_lambda1, pre[1])
                self.adjust_alpha(e3.id, d_lambda1, pre[2])


            for i in range(n_interior):
                self.lambdaPlanar[i] += context.variable_get(i)
                self.lambdaLength[i] += context.variable_get(n_interior + i)
        return success


class GeoUVPinIndex:
    def __init__(self, uv_co, reindex):
          self.next: GeoUVPinIndex | None = None
          self.uv: Vector = uv_co
          self.reindex: int = reindex


class ParamHandleConstruct:
    """
     name Chart Construction:

    Faces and seams may only be added between #ParamHandle::ParamHandle() and
    #geometry::uv_parametrizer_construct_end.

    The pointers to `co` and `uv` are stored, rather than being copied. Vertices are implicitly
    created.

    In #geometry::uv_parametrizer_construct_end the mesh will be split up according to the seams.
    The resulting charts must be manifold, connected and open (at least one boundary loop). The
    output will be written to the `uv` pointers.
    """

    PHANDLE_STATE_ALLOCATED = 0
    PHANDLE_STATE_CONSTRUCTED = 1
    PHANDLE_STATE_LSCM = 2
    PHANDLE_STATE_STRETCH = 3

    def __init__(self):
        self.state = ParamHandleConstruct.PHANDLE_STATE_ALLOCATED
        self.construction_chart: PChart = PChart()
        self.hash_verts: PHash = PHash(self.construction_chart.verts)
        self.hash_edges: PHash = PHash(self.construction_chart.edges)
        self.hash_faces: PHash = PHash(self.construction_chart.faces)

        self.pin_hash: dict[int, GeoUVPinIndex] = {}
        self.unique_pin_count: int = 0

        self.charts: list[PChart] = []
        self.ncharts: int = 0

        self.aspect_y = 1.0
        self.blend: float = 0.0

    @classmethod
    def construct_param_handle(cls, isl: 'utypes.AdvIsland'):
        handle = cls()
        umesh = isl.umesh
        handle.aspect_y = umesh.aspect

        # we need the vert indices
        umesh.bm.verts.index_update()

        get_vert_select = bm_select.vert_select_get_func(umesh)
        uv = umesh.uv
        for f in isl:
            handle.uvedit_prepare_pinned_indices(f, uv, get_vert_select)

        for idx, ff in enumerate(isl.faces):
            handle.construct_param_handle_face_add(ff, idx, umesh, get_vert_select)

        handle.construct_param_edge_set_seams(isl)
        handle.uv_parametrizer_construct_end()

        return handle


    def construct_param_handle_face_add(self, f: BMFace, face_index: ParamKey | int, umesh: 'utypes.UMesh', get_vertex_select):
        vkeys: list[ParamKey] = []
        pin: list[bool] = []
        select: list[bool] = []
        coord_3d: list[Vector] = []
        coord_uv: list[Vector] = []
        weight: list[float] = [1.0] * len(f.loops)  # TODO: Implement custom weight

        # let parametrizer split the ngon, it can make better decisions
        # about which split is best for unwrapping than poly-fill. */

        uv = umesh.uv
        for crn in f.loops:
            v = crn.vert
            crn_uv = crn[uv]
            uv_co = crn_uv.uv

            vkeys.append(self.uv_find_pin_index(v.index, uv_co))
            coord_3d.append(v.co)
            coord_uv.append(uv_co)
            pin.append(crn_uv.pin_uv)
            select.append(get_vertex_select(crn))

        if UnwrapOptions.pin_unselected:
            for idx, sel_state in enumerate(select):
                if not sel_state:
                    pin[idx] = True

        self.uv_parametrizer_face_add(face_index, vkeys, coord_3d, coord_uv, weight, pin, select)


    def construct_param_edge_set_seams(self, isl: 'utypes.AdvIsland'):
        """Set seams on UV Parametrizer based on options."""
        if UnwrapOptions.topology_from_uvs and not UnwrapOptions.topology_from_uvs_use_seams:
            return  # Seams are not required with these options.

        uv = isl.umesh.uv
        for crn in isl.corners_iter():
            if not crn.edge.seam:
                # No seam on this edge, nothing to do.
                continue
            #           Pinned vertices might have more than one ParamKey per BMVert.
            #           Check all the BM_LOOPS_OF_EDGE to find all the ParamKeys.

            uv_co = crn[uv].uv
            uv_co_next = crn.link_loop_next[uv].uv

            vkeys = [
                self.uv_find_pin_index(crn.vert.index, uv_co),
                self.uv_find_pin_index(crn.link_loop_next.vert.index, uv_co_next)
            ]

            # Set the seam.
            e: PEdge = self.edge_lookup(vkeys)
            if e:
                e.flag |= PEDGE_SEAM

    def uv_parametrizer_construct_end(self):
        self.ncharts = self.connect_pairs()
        self.charts = self.construction_chart.split_charts(self.ncharts)

        # free
        self.construction_chart = None
        self.hash_verts = None
        self.hash_edges = None
        self.hash_faces = None

        j = 0
        for i in range(self.ncharts):
            chart: PChart = self.charts[i]

            outer: PEdge | None = chart.boundaries()

            if not UnwrapOptions.topology_from_uvs and chart.n_boundaries == 0:
                # UnwrapOptions.count_failed += 1
                continue

            self.charts[j] = chart
            j += 1

            if UnwrapOptions.fill_holes and chart.n_boundaries > 1:
                chart.fill_boundaries(outer)

            for v in chart.verts:
                v.load_pin_select_uvs()

            self.ncharts = j

            self.state = ParamHandleConstruct.PHANDLE_STATE_CONSTRUCTED

    def connect_pairs(self) -> int:
        """Connect pairs, count edges, set vertex-edge pointer to a pair-less edge"""
        stack: list[PEdge] = []

        chart = self.construction_chart
        ncharts = 0

        for f in chart.faces:
            if f.flag & PFACE_CONNECTED:
                continue

            stack.append(f.edge)

            while stack:
                e0 = stack.pop()
                e1 = e0.next
                e2 = e1.next

                f = e0.face
                f.flag |= PFACE_CONNECTED

                f.chart = ncharts  # set index for face

                # Assign verts to charts so we can sort them later.
                if not e0.edge_connect_pair(self, stack):
                    e0.vert.edge = e0
                if not e1.edge_connect_pair(self, stack):
                    e1.vert.edge = e1
                if not e2.edge_connect_pair(self, stack):
                    e2.vert.edge = e2

            ncharts += 1

        assert ncharts == 1
        return ncharts

    def face_add_construct(self, key: ParamKey, vkeys: list[ParamKey], co: list[Vector], uv: list[Vector], weight: list[float],
                           i1: int, i2: int, i3: int, pin: list[bool], select: list[bool]) -> PFace:

        f: PFace = PFace.new()

        e1: PEdge = f.edge
        e2: PEdge = e1.next
        e3: PEdge = e2.next

        weight1 = weight2 = weight3 = 1.0

        if weight:
            weight1 = weight[i1]
            weight2 = weight[i2]
            weight3 = weight[i3]

        e1.vert = self.vert_lookup(vkeys[i1], co[i1], weight1, e1)
        e2.vert = self.vert_lookup(vkeys[i2], co[i2], weight2, e2)
        e3.vert = self.vert_lookup(vkeys[i3], co[i3], weight3, e3)

        e1.orig_uv = uv[i1]
        e2.orig_uv = uv[i2]
        e3.orig_uv = uv[i3]

        if pin:
            if pin[i1]:
                e1.flag |= PEDGE_PIN
            if pin[i2]:
                e2.flag |= PEDGE_PIN
            if pin[i3]:
                e3.flag |= PEDGE_PIN

        if select:
            if select[i1]:
                e1.flag |= PEDGE_SELECT
            if select[i2]:
                e2.flag |= PEDGE_SELECT
            if select[i3]:
                e3.flag |= PEDGE_SELECT

        f.key = key
        self.hash_faces.insert(f)

        e1.key = PHASH_edge(vkeys[i1], vkeys[i2])
        e2.key = PHASH_edge(vkeys[i2], vkeys[i3])
        e3.key = PHASH_edge(vkeys[i3], vkeys[i1])

        self.hash_edges.insert(e1)
        self.hash_edges.insert(e2)
        self.hash_edges.insert(e3)

        return f

    def add_ngon(self, key: ParamKey, vkeys: list[ParamKey], co: list[Vector],
                 uv: list[Vector],  # Output will eventually be written to `uv`
                 weight: list[float], pin: list[bool], select: list[bool]):

        # Beautify helps avoid thin triangles that give numerical problems
        tris = polyfill_beautify(co)

        # Add triangles.
        for (v0, v1, v2) in tris:
            tri_vkeys: list[ParamKey] = [vkeys[v0], vkeys[v1], vkeys[v2]]
            tri_co: list[Vector] = [co[v0], co[v1], co[v2]]
            tri_uv: list[Vector] = [uv[v0], uv[v1], uv[v2]]
            tri_weight = [weight[v0], weight[v1], weight[v2]]
            tri_pin: list[bool] = [pin[v0], pin[v1], pin[v2]]
            tri_select: list[bool] = [select[v0], select[v1], select[v2]]

            self.uv_parametrizer_face_add(key, tri_vkeys, tri_co, tri_uv, tri_weight, tri_pin, tri_select)

    def uv_parametrizer_face_add(self, key: ParamKey, vkeys: list[ParamKey], co: list[Vector],
                                 uv: list[Vector], weight: list[float], pin: list[bool], select: list[bool]):
        """Fix overlap faces after triangulate, if exist"""
        nverts = len(co)
        assert (nverts >= 3)

        if nverts > 3:
            # Protect against (manifold) geometry which has a non-manifold triangulation.
            # See #102543.
            permute: list[int] = list(range(nverts))

            i: int = nverts - 1
            while i >= 0:
                # Just check the "ears" of the n-gon.
                # For quads, this is sufficient.
                # For pentagons and higher, we might miss internal duplicate triangles, but note
                # that such cases are rare if the source geometry is manifold and non-intersecting. */
                pm: int = len(permute)
                assert (pm > 3)
                i0: int = permute[i]
                i1: int = permute[(i + 1) % pm]
                i2: int = permute[(i + 2) % pm]
                if not self.face_exists(vkeys, i0, i1, i2):
                    i -= 1  # All good.
                    continue

                # An existing triangle has already been inserted.
                # As a heuristic, attempt to add the *previous* triangle.
                # NOTE: Should probably call `uv_parametrizer_face_add`
                # instead of `p_face_add_construct`.
                i_prev: int = permute[(i + pm - 1) % pm]
                self.face_add_construct(key, vkeys, co, uv, weight, i_prev, i0, i1, pin, select)

                permute.remove(i)
                if len(permute) == 3:
                    break

            if len(permute) != nverts:
                pm: int = len(permute)
                # Add the remaining `pm-gon` data.
                vkeys_sub: list[ParamKey | int] = [0] * pm
                co_sub: list[Vector | None] = [None] * pm
                uv_sub: list[Vector | None] = [None] * pm
                weight_sub: list[float] = [1.0] * pm
                pin_sub: list[bool] = [False] * pm
                select_sub: list[bool] = [False] * pm

                for i in range(pm):
                    j = permute[i]
                    vkeys_sub[i] = vkeys[j]
                    co_sub[i] = co[j]
                    uv_sub[i] = uv[j]
                    weight_sub[i] = weight[j]

                    pin_sub[i] = pin[j]
                    select_sub[i] = select[j]

                self.add_ngon(key, vkeys_sub, co_sub, uv_sub, weight_sub, pin_sub, select_sub)
                return  # Nothing more to do. */

        # No "ears" have previously been inserted. Continue as normal.
        if nverts > 3:
            # ngon
            self.add_ngon(key, vkeys, co, uv, weight, pin, select)

        elif not self.face_exists(vkeys, 0, 1, 2):
            # triangle
            self.face_add_construct(key, vkeys, co, uv, weight, 0, 1, 2, pin, select)

    def vert_add(self, key: PHashKey, co: Vector, weight: float, e: PEdge) -> PVert:
        """Construction (use only during construction, relies on u.key being set)."""
        v = PVert()
        v.co = co.copy()
        v.weight = weight

        # Sanity check, a single nan/inf point causes the entire result to be invalid.
        # Note that values within the calculation may _become_ non-finite,
        # so the rest of the code still needs to take this possibility into account.
        for i, value in enumerate(v.co):
            if not math.isfinite(value):
                v.co[i] = 0

        v.key = key
        v.edge = e
        v.flag = 0

        # Unused, prevent uninitialized memory access on duplication.
        v.on_boundary_flag = False
        # v.slim_id = 0

        self.hash_verts.insert(v)

        return v

    def vert_lookup(self, key: PHashKey | ParamKey, co: Vector, weight: float, e: PEdge) -> PVert:
        v: PVert = self.hash_verts.lookup(key)
        if v:
            return v
        return self.vert_add(key, co, weight, e)

    def edge_lookup(self, vkeys: list[int | PHashKey | ParamKey]) -> PEdge | None:
        key: PHashKey = PHASH_edge(vkeys[0], vkeys[1])  # noqa

        e: PEdge | None = self.hash_edges.lookup(key)

        while e:
            if (e.vert.key == vkeys[0]) and (e.next.vert.key == vkeys[1]):
                return e
            if (e.vert.key == vkeys[1]) and (e.next.vert.key == vkeys[0]):
                return e

            e = self.hash_edges.next(key, e)

    def face_exists(self, vkeys: list[ParamKey], i1: int, i2: int, i3: int) -> bool:
        key: PHashKey = PHASH_edge(vkeys[i1], vkeys[i2])  # noqa
        e: PEdge | None = self.hash_edges.lookup(key)

        while e:
            if (e.vert.key == vkeys[i1]) and (e.next.vert.key == vkeys[i2]):
                if e.next.next.vert.key == vkeys[i3]:
                    return True

            elif (e.vert.key == vkeys[i2]) and (e.next.vert.key == vkeys[i1]):
                if e.next.next.vert.key == vkeys[i3]:
                    return True

            e = self.hash_edges.next(key, e)

        return False

    # Pins parametrization
    # ================================================================================
    def new_geo_uv_pin_index(self, uv_co: Vector):
        PARAM_KEY_MAX = 1 << 32
        reindex = PARAM_KEY_MAX - self.unique_pin_count
        self.unique_pin_count += 1

        return GeoUVPinIndex(uv_co.copy(), reindex)

    def uv_prepare_pin_index(self, bm_vert_index: int, uv_co: Vector):
        if (pin_uv_list := self.pin_hash.get(bm_vert_index)) is None:
            self.pin_hash[bm_vert_index] = self.new_geo_uv_pin_index(uv_co)
            return

        while True:
            if pin_uv_list.uv == uv_co:
                return

            if not pin_uv_list.next:
                pin_uv_list.next = self.new_geo_uv_pin_index(uv_co)
                return
            pin_uv_list = pin_uv_list.next

    def uv_find_pin_index(self, bm_vert_index: int, uv: Vector) -> ParamKey | int:
        if not self.pin_hash:
            return bm_vert_index  # No verts pinned.

        if (pin_uv_list := self.pin_hash.get(bm_vert_index)) is None:
            return bm_vert_index  # Vert not pinned.

        # At least one of the UVs associated with bm_vert_index is pinned. Find the best one. */

        best_dist_squared: float = (pin_uv_list.uv - uv).length_squared
        best_key = pin_uv_list.reindex

        pin_uv_list = pin_uv_list.next
        while pin_uv_list:
            dist_squared: float = (pin_uv_list.uv-uv).length_squared
            if best_dist_squared > dist_squared:
                best_dist_squared = dist_squared
                best_key = pin_uv_list.reindex
            pin_uv_list = pin_uv_list.next
        return best_key

    def uvedit_prepare_pinned_indices(self, f: BMFace, uv, get_vert_select):
        """Prepare unique indices for each unique pinned UV, even if it shares a BMVert."""
        for crn in f.loops:
            pin = crn[uv].pin_uv
            if UnwrapOptions.pin_unselected and not pin:
                pin = not get_vert_select(crn)

            if pin:
                vert_idx = crn.vert.index
                uv_co = crn[uv].uv
                self.uv_prepare_pin_index(vert_idx, uv_co)


class ParamHandleSolve(ParamHandleConstruct):

    def uv_parametrizer_lscm_begin(self):
        assert (self.state == self.PHANDLE_STATE_CONSTRUCTED);
        self.state = self.PHANDLE_STATE_LSCM

        for chart in self.charts:
            for f in chart.faces:
                f.backup_uvs()

            chart.lscm_begin()

    def uv_parametrizer_lscm_solve(self):

        assert self.state == self.PHANDLE_STATE_LSCM

        for chart in self.charts:
            if not chart.context:
                continue

            if chart.lscm_solve():
                pass
                # if not chart.has_pins:
                #     # Every call to LSCM will eventually call uv_pack, so rotating here might be redundant.
                #     p_chart_rotate_minimum_area(chart)
                #
                # elif chart.single_pin:
                #     p_chart_rotate_fit_aabb(chart)
                #     p_chart_lscm_transform_single_pin(chart)
            else:
                pass
                # count_failed += 1

    def uv_parametrizer_flush(self):
        for chart in self.charts:
            if not chart.skip_flush:
                self.p_flush_uvs(chart)

    def p_flush_uvs(self, chart: PChart):
        blend: float = self.blend
        inv_blend: float = 1.0 - blend
        inv_blend_x: float = inv_blend / self.aspect_y
        for e in chart.edges:
            if e.orig_uv:
                e.orig_uv[0] = blend * e.old_uv[0] + inv_blend_x * e.vert.uv[0]
                e.orig_uv[1] = blend * e.old_uv[1] + inv_blend * e.vert.uv[1]

        # if chart.collapsed_edges:
        #     p_chart_flush_collapsed_uvs(chart)
        #
        #     for e in chart.collapsed_edges:
        #         if e.orig_uv:
        #             e.orig_uv[0] = blend * e.old_uv[0] + inv_blend_x * e.vert.uv[0]
        #             e.orig_uv[1] = blend * e.old_uv[1] + inv_blend * e.vert.uv[1]


class ParamHandle(ParamHandleSolve):
    pass

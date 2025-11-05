import math
import typing

from bmesh.types import BMFace
from mathutils import Vector

from .umath import LinearSolver
from .ubm import polyfill_beautify
from . import bm_select
from .. import utypes

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

    def edge_connect_pair(self, handle: 'ParamHandle', stack: 'list[PEdge]') -> bool:
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

    def has_pair(self, handle: 'ParamHandle', r_pair: list) -> bool:
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

        self.origin: Vector

        self.context: LinearSolver
        self.abf_alpha: float  # list of alpha ???

        self.pin1: PVert
        self.pin2: PVert
        self.single_pin: PVert

        self.has_pins: bool = False
        self.skip_flush: bool

    def boundaries(self):
        max_length = -1.0
        outer = None

        for e in self.edges:
            if e.pair or e.flag & PEDGE_DONE:
                continue

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


class UnwrapOptions:
    # Connectivity based on UV coordinates instead of seams. */
    topology_from_uvs: bool = True
    # Also use seams as well as UV coordinates (only valid when `topology_from_uvs` is enabled). */
    topology_from_uvs_use_seams: bool = True
    # Only affect selected faces. */
    only_selected_faces: bool = True

    # Only affect selected UVs.
    # \note Disable this for operations that don't run in the image-window.
    # Unwrapping from the 3D view for example, where only 'only_selected_faces' should be used.

    only_selected_uvs: bool  = True
    # Fill holes to better preserve shape. */
    fill_holes: bool = False
    # Correct for mapped image texture aspect ratio. */
    correct_aspect: bool = True
    # Treat unselected uvs as if they were pinned. */
    pin_unselected: bool = True

    method: int = 0
    use_slim: bool = False
    use_abf: bool = True
    use_subsurf: bool = False
    use_weights: bool = False

    # slim: ParamSlimOptions = None
    weight_group: str = ''

class GeoUVPinIndex:
    def __init__(self, uv_co, reindex):
          self.next: GeoUVPinIndex | None = None
          self.uv: Vector = uv_co
          self.reindex: int = reindex


class ParamHandle:
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
        self.state = ParamHandle.PHANDLE_STATE_ALLOCATED
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

            _outer: PEdge | None = chart.boundaries()
            if not UnwrapOptions.topology_from_uvs and chart.n_boundaries == 0:
                # UnwrapOptions.count_failed += 1
                continue

            self.charts[j] = chart
            j += 1

            # TODO: Implement fill boundary with weight property for that
            # if UnwrapOptions.fill_holes and chart.n_boundaries > 1:
            # p_chart_fill_boundaries(phandle, chart, outer)

            for v in chart.verts:
                v.load_pin_select_uvs()

            self.ncharts = j

            self.state = ParamHandle.PHANDLE_STATE_CONSTRUCTED

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


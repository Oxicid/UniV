# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa

from bmesh.types import BMesh, BMLoop, BMLayerItem
from collections import deque
from itertools import chain

from .. import utypes
from .bm_tag import is_pair, is_invisible_func

USE_GENERIC_UV_SYNC = hasattr(BMesh, 'uv_select_sync_valid')

def shared_linked_crn_by_idx(crn: BMLoop, uv) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        if crn.link_loop_next[uv].uv == shared[uv].uv and crn[uv].uv == shared.link_loop_next[uv].uv:
            return shared


def shared_linked_crn_to_edge_by_idx(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        return shared

def prev_disc(l: BMLoop) -> BMLoop:
    return l.link_loop_prev.link_loop_radial_prev

def linked_crn_uv(first: BMLoop, uv: BMLayerItem):
    first_vert = first.vert
    first_co = first[uv].uv
    linked = []
    bm_iter = first

    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert:  # Skip boundary or flipped
            bm_iter = first
            linked_cw = []
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                if first_vert != bm_iter.vert:  # Skip boundary or flipped
                    break

                if bm_iter == first:
                    break
                if first_co == bm_iter[uv].uv:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == first:
            break
        if first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


def linked_crn_to_vert_pair_iter(crn: BMLoop, uv, sync):
    """CW corners not reverse"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert
    iterated = False
    bm_iter = crn
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True
        # Finish CCW
        if pair_ccw in (prev_crn, crn) or (first_vert != pair_ccw.vert) or is_invisible(pair_ccw.face) or not is_pair(prev_crn, pair_ccw, uv):
            bm_iter = crn
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if (first_vert != next_crn.vert) or is_invisible(next_crn.face) or not is_pair(bm_iter, pair_cw, uv):
                    break
                yield next_crn
                bm_iter = next_crn
            break
        yield pair_ccw
        bm_iter = pair_ccw


def linked_crn_to_vert_pair(crn: BMLoop, uv, sync: bool):
    """Linked to arg corner by island index with arg corner"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if pair_ccw in (prev_crn, crn) or (first_vert != pair_ccw.vert) or is_invisible(pair_ccw.face) or not is_pair(prev_crn, pair_ccw, uv):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if (first_vert != next_crn.vert) or is_invisible(next_crn.face) or not is_pair(bm_iter, pair_cw, uv):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked


def linked_crn_to_vert_pair_with_seam(crn: BMLoop, uv, sync: bool):
    """Linked to arg corner by island index with arg corner (non-included)"""
    is_invisible = is_invisible_func(sync)
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if (pair_ccw in (prev_crn, crn) or
                    (first_vert != pair_ccw.vert) or
                    pair_ccw.edge.seam or
                    is_invisible(pair_ccw.face) or
                    not is_pair(prev_crn, pair_ccw, uv)
                ):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if ((first_vert != next_crn.vert)
                            or pair_cw.edge.seam
                            or is_invisible(next_crn.face)
                            or not is_pair(bm_iter, pair_cw, uv)
                        ):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked


def linked_crn_to_vert_pair_by_idx_with_seam(crn: BMLoop, uv):
    """Linked to arg corner by island index with arg corner"""
    idx = crn.face.index
    first_vert = crn.vert

    linked = []
    bm_iter = crn
    # Iterated is needed to realize that a full iteration has passed, and there is no need to calculate CW
    iterated = False
    while True:
        prev_crn = bm_iter.link_loop_prev
        pair_ccw = prev_crn.link_loop_radial_prev
        if pair_ccw == crn and iterated:
            break
        iterated = True

        # Finish CCW
        if (pair_ccw in (prev_crn, crn) or
                    (first_vert != pair_ccw.vert) or
                    pair_ccw.edge.seam or
                    pair_ccw.face.index != idx or
                    not is_pair(prev_crn, pair_ccw, uv)
                ):
            bm_iter = crn
            linked_cw = []
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if ((first_vert != next_crn.vert)
                            or pair_cw.edge.seam
                            or next_crn.face.index != idx
                            or not is_pair(bm_iter, pair_cw, uv)
                        ):
                    break
                bm_iter = next_crn
                linked_cw.append(next_crn)
            linked.extend(linked_cw[::-1])
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked


def linked_crn_uv_unordered(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    linked.remove(first)
    return linked


def linked_crn_uv_unordered_included(first: BMLoop, uv: BMLayerItem):
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    return linked


def linked_crn_uv_by_tag_b(first: BMLoop, uv: BMLayerItem):
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


# TODO: Replace with linked unordered (change logic in extend_from_linked)
def linked_crn_vert_uv_for_transform(first, uv):
    # Need tagging. tag == False - not append
    # assert utils.sync()
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.tag:
            continue
        if first_co == bm_iter[uv].uv:
            bm_iter.tag = False
            linked.append(bm_iter)

    next_crn = first.link_loop_next
    next_crn_co = next_crn[uv].uv
    if next_crn.tag:
        next_crn.tag = False
        linked.append(next_crn)

        bm_iter = next_crn
        while True:
            if (bm_iter := prev_disc(bm_iter)) == next_crn:
                break
            if not bm_iter.tag:
                continue
            if next_crn_co == bm_iter[uv].uv:
                bm_iter.tag = False
                linked.append(bm_iter)
    return linked


def linked_crn_uv_by_crn_tag_unordered_included(crn, uv) -> list[BMLoop]:
    """Linked to arg corner by **crn** tag with arg corner and unordered"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.tag and l_crn[uv].uv == first_co]


def linked_crn_uv_by_face_tag_unordered_included(crn, uv) -> list[BMLoop]:
    """Linked to arg corner by **face** tag with arg corner and unordered"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.tag and l_crn[uv].uv == first_co]


def linked_crn_uv_by_face_index(first: BMLoop, uv: BMLayerItem):
    """Included Unordered"""
    face_index = first.face.index
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.face.index == face_index and first[uv].uv == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


def linked_crn_uv_by_idx(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index with arg corner"""
    first_vert = crn.vert
    idx = crn.face.index
    first_co = crn[uv].uv
    linked = []
    bm_iter = crn

    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert:  # Skip boundary or flipped
            bm_iter = crn
            linked_cw = []
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                if first_vert != bm_iter.vert:  # Skip boundary or flipped
                    break

                if bm_iter == crn:
                    break
                if bm_iter.face.index == idx and first_co == bm_iter[uv].uv:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == crn:
            break
        if bm_iter.face.index == idx and first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


def linked_crn_to_vert_by_idx_3d(crn: BMLoop):
    """Linked to arg corner by island index with arg corner"""
    first_vert = crn.vert
    idx = crn.face.index
    linked = []
    bm_iter = crn

    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert:  # Skip boundary or flipped
            bm_iter = crn
            linked_cw = []
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                if first_vert != bm_iter.vert:  # Skip boundary or flipped
                    break

                if bm_iter == crn:
                    break
                if bm_iter.face.index == idx:
                    linked_cw.append(bm_iter)
            linked.extend(linked_cw[::-1])
            break
        if bm_iter == crn:
            break
        if bm_iter.face.index == idx:
            linked.append(bm_iter)
    return linked


def linked_crn_to_vert_3d_iter(crn: BMLoop):
    """Linked to arg corner by visible faces"""
    first_vert = crn.vert
    bm_iter = crn

    while True:
        bm_iter_prev = bm_iter.link_loop_prev
        bm_iter = bm_iter_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert or bm_iter.face.hide:  # Skip boundary or flipped
            bm_iter = crn
            while True:
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                # Skip boundary or flipped or seam clamp or hide clamp
                if first_vert != bm_iter.vert or bm_iter.face.hide:
                    break

                if bm_iter == crn:
                    break
                yield bm_iter
            break  # break first loop
        if bm_iter == crn:
            break
        yield bm_iter


def linked_crn_to_vert_with_seam_3d_iter(crn: BMLoop):
    """Linked to arg corner by visible faces"""
    first_vert = crn.vert
    bm_iter = crn

    while True:
        bm_iter_prev = bm_iter.link_loop_prev
        bm_iter = bm_iter_prev.link_loop_radial_prev  # get ccw corner
        if first_vert != bm_iter.vert or bm_iter_prev.edge.seam or bm_iter.face.hide:  # Skip boundary or flipped
            bm_iter = crn
            while True:
                if bm_iter.edge.seam:  # clamp by seam
                    break
                bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                # Skip boundary or flipped or clamp by hide
                if first_vert != bm_iter.vert or bm_iter.face.hide:
                    break

                if bm_iter == crn:
                    break
                yield bm_iter
            break  # break first loop
        if bm_iter == crn:
            break
        yield bm_iter


def linked_crn_uv_by_idx_unordered(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index without arg corner
    simular - linked_crn_uv_by_island_index_unordered
    """
    first_co = crn[uv].uv
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co]


def linked_crn_uv_by_idx_unordered_included(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index without arg corner
    simular - linked_crn_uv_by_island_index_unordered_included
    """
    first_co = crn[uv].uv
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx and l_crn[uv].uv == first_co]


def linked_crn_uv_by_island_index_unordered_included(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index with arg corner"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx and l_crn[uv].uv == first_co]


def linked_crn_uv_by_island_index_unordered(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index without arg corner"""
    first_co = crn[uv].uv
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co]


def linked_crn_to_vert_by_face_index(crn):
    """Linked to vertex by face index without arg corner"""
    idx = crn.face.index
    linked = deque(l_crn for l_crn in crn.vert.link_loops if l_crn.face.index == idx)
    linked.rotate(-linked.index(crn))
    linked.popleft()
    return linked


def linked_crn_to_vert_by_island_index_unordered(crn):
    """Linked to vertex by island index without arg corner"""
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx]


class ShortPath:

    @staticmethod
    def vert_tag_add_adjacent_uv(heap, l_a: BMLoop, loops_prev: list[BMLoop | None], cost: list[float], uv, prioritize_corners, bound_priority_factor):
        import heapq
        l_a_index = l_a.index
        uv_a = l_a[uv].uv

        # Loop over faces of face, but do so by first looping over loops.
        for l in linked_crn_uv_by_idx_unordered_included(l_a, uv):  # TODO: Add mark seam and bi-direct linked
            #  'l_a' is already tagged, tag all adjacent.

            l.tag = False
            l_b = l.link_loop_next

            while True:
                if l_b.tag:
                    uv_b = l_b[uv].uv
                    # We know 'l_b' is not visited, check it out!
                    l_b_index = l_b.index
                    cost_cut = (uv_a - uv_b).length
                    if l_b in prioritize_corners:
                        cost_cut *= bound_priority_factor
                    cost_new = cost[l_a_index] + cost_cut

                    if cost[l_b_index] > cost_new:
                        cost[l_b_index] = cost_new
                        loops_prev[l_b_index] = l_a
                        heapq.heappush(heap, (cost_new, id(l_b), l_b))

                # This means we only step onto `l->prev` & `l->next`.
                if l_b == l.link_loop_next:
                    l_b = l.link_loop_prev.link_loop_prev
                if (l_b := l_b.link_loop_next) == l:
                    break

    @staticmethod
    def calc_path_uv_vert(isl: 'utypes.AdvIsland',
                          l_src: BMLoop,
                          l_dst: BMLoop,
                          exclude_corners_group: 'list[types.LoopGroup] | tuple',
                          prioritize_corners: set[BMLoop] | tuple = (),
                          bound_priority_factor=0.9) -> list[BMLoop]:
        import heapq
        from collections import deque
        assert l_src.face.index == l_dst.face.index
        path = deque()
        # BM_ELEM_TAG flag is used to store visited edges
        uv = isl.umesh.uv
        heap: list[tuple[float, int, BMLoop]] = []

        # NOTE: would pass BM_EDGE except we are looping over all faces anyway.
        # BM_mesh_elem_index_ensure(bm, BM_LOOP); NOTE: not needed for face tag.
        i = 0
        for f in isl:
            for crn in f.loops:
                crn.tag = True
                crn.index = i
                i += 1

        if exclude_corners_group:
            for exclude_lg in exclude_corners_group:
                for chain_corners in exclude_lg.chain_linked_corners:
                    for crn in chain_corners:
                        crn.tag = False

            # Restore tags that may have been excluded above.
            dst_corners = exclude_corners_group[0].chain_linked_corners[-1]
            src_corners = exclude_corners_group[2].chain_linked_corners[0]
            for crn in chain(dst_corners, src_corners):
                crn.tag = True

        # Allocate.
        loops_prev: list[BMLoop | None] = [None] * i
        cost = [1e100] * i

        # Regular dijkstra the shortest path, but over UV loops instead of vertices.
        heapq.heappush(heap, (0.0, l_src.index, l_src))
        cost[l_src.index] = 0.0

        while heap:
            l = heapq.heappop(heap)[2]
            if (l.vert == l_dst.vert) and l[uv].uv == l_dst[uv].uv:
                # assert l.face.index == l_dst.face.index
                while True:
                    path.appendleft(l)
                    if not (l := loops_prev[l.index]):
                        break
                break

            if l.tag:
                #  Adjacent loops are tagged while stepping to avoid 2x loops.
                l.tag = False
                ShortPath.vert_tag_add_adjacent_uv(heap, l, loops_prev, cost, uv,
                                                   prioritize_corners, bound_priority_factor)

        return list(path)

    @staticmethod
    def path_to_loop_group_for_rect(path, umesh):
        assert path

        uv = umesh.uv
        chain_linked_corners = []
        for crn in path:
            linked = linked_crn_uv_by_idx_unordered(crn, uv)
            linked.insert(0, crn)
            chain_linked_corners.append(linked)

        it = iter(path)
        prev_vert_co = next(it).vert.co

        weights = []
        for crn in it:
            cur_vert_co = crn.vert.co
            length = (prev_vert_co - cur_vert_co).length
            weights.append(length)
            prev_vert_co = cur_vert_co

        lg = utypes.LoopGroup(umesh)
        lg.corners = path
        lg.chain_linked_corners = chain_linked_corners
        lg.dirt = True
        lg.weights = weights
        return lg

    @staticmethod
    def calc_length_3d_and_uv_from_path(path, umesh):  # TODO: Add aspect
        uv = umesh.uv
        it = iter(path)
        first_crn = next(it)

        prev_uv_co = first_crn[uv].uv
        prev_vert_co = first_crn.vert.co

        total_length_3d = 0.0
        total_length_uv = 0.0

        for crn in it:
            cur_uv_co = crn[uv].uv
            cur_vert_co = crn.vert.co
            total_length_3d += (prev_vert_co - cur_vert_co).length
            total_length_uv += (prev_uv_co - cur_uv_co).length

            prev_uv_co = cur_uv_co
            prev_vert_co = cur_vert_co

        return total_length_3d, total_length_uv

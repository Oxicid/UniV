# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy  # noqa

from bmesh.types import BMesh, BMLoop, BMLayerItem
from .bm_tag import is_pair, is_invisible_func

USE_GENERIC_UV_SYNC = hasattr(BMesh, "uv_select_sync_valid")


def shared_linked_crn_by_idx(crn: BMLoop, uv) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        if (
            crn.link_loop_next[uv].uv == shared[uv].uv
            and crn[uv].uv == shared.link_loop_next[uv].uv
        ):
            return shared
    return None


def shared_linked_crn_to_edge_by_idx(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn and crn.face.index == shared.face.index:
        return shared
    return None


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
        if (
            pair_ccw in (prev_crn, crn)
            or (first_vert != pair_ccw.vert)
            or is_invisible(pair_ccw.face)
            or not is_pair(prev_crn, pair_ccw, uv)
        ):
            bm_iter = crn
            while True:
                pair_cw = bm_iter.link_loop_radial_prev
                # Skip flipped and boundary
                if pair_cw == bm_iter:
                    break

                next_crn = pair_cw.link_loop_next
                if next_crn == crn:
                    break

                if (
                    (first_vert != next_crn.vert)
                    or is_invisible(next_crn.face)
                    or not is_pair(bm_iter, pair_cw, uv)
                ):
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
        if (
            pair_ccw in (prev_crn, crn)
            or (first_vert != pair_ccw.vert)
            or is_invisible(pair_ccw.face)
            or not is_pair(prev_crn, pair_ccw, uv)
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

                if (
                    (first_vert != next_crn.vert)
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


def linked_crn_to_vert_pair_with_seam(crn: BMLoop, uv, sync: bool):
    """Linked to arg corner (non-included)"""
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
        if (
            pair_ccw in (prev_crn, crn)
            or (first_vert != pair_ccw.vert)
            or pair_ccw.edge.seam
            or is_invisible(pair_ccw.face)
            or not is_pair(prev_crn, pair_ccw, uv)
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

                if (
                    (first_vert != next_crn.vert)
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


def linked_crn_to_vert_pair_with_seam_included(crn: BMLoop, uv, sync: bool) -> list[BMLoop]:
    linked = linked_crn_to_vert_pair_with_seam(crn, uv, sync)
    linked.insert(0, crn)
    return linked


def linked_crn_to_vert_without_coord_check_with_seam_for_sync_unwrap(crn: BMLoop):
    """Linked to arg corner with_seam without coord check(non-included)
    NOTE: Need for unwrap in vert/edge mode for check
    NOTE: Need actual mark seams for differencing islands by seams
    """
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
                    pair_ccw.face.hide
                ):  # fmt: skip
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
                            or pair_ccw.face.hide
                        ):  # fmt: skip
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
        if (
            pair_ccw in (prev_crn, crn)
            or (first_vert != pair_ccw.vert)
            or pair_ccw.edge.seam
            or pair_ccw.face.index != idx
            or not is_pair(prev_crn, pair_ccw, uv)
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

                if (
                    (first_vert != next_crn.vert)
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


def linked_crn_uv_unordered_included_with_hidden(first: BMLoop, uv: BMLayerItem):
    # TODO: Rename to with hidden
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    return linked


def linked_with_unlinked_crn_uv_to_vert_unordered_included_func(umesh):
    def catcher(uv):
        if umesh.sync:

            def get_linked(first: BMLoop):
                first_co = first[uv].uv
                linked = []
                unlinked = []
                for l_crn in first.vert.link_loops:
                    if not l_crn.face.hide:
                        if l_crn[uv].uv == first_co:
                            linked.append(l_crn)
                        else:
                            unlinked.append(l_crn)
                return linked, unlinked

        else:

            def get_linked(first: BMLoop):
                first_co = first[uv].uv
                linked = []
                unlinked = []
                for l_crn in first.vert.link_loops:
                    if l_crn.face.select:
                        if l_crn[uv].uv == first_co:
                            linked.append(l_crn)
                        else:
                            unlinked.append(l_crn)
                return linked, unlinked

        return get_linked

    return catcher(umesh.uv)


def linked_crn_uv_to_vert_unordered_included_func(umesh):
    def catcher(uv):
        if umesh.sync:

            def get_linked(first: BMLoop):
                first_co = first[uv].uv
                return [
                    l_crn
                    for l_crn in first.vert.link_loops
                    if (not l_crn.face.hide) and l_crn[uv].uv == first_co
                ]

        else:

            def get_linked(first: BMLoop):
                first_co = first[uv].uv
                return [
                    l_crn
                    for l_crn in first.vert.link_loops
                    if l_crn.face.select and l_crn[uv].uv == first_co
                ]

        return get_linked

    return catcher(umesh.uv)


def linked_crn_uv_by_tag_b(first: BMLoop, uv: BMLayerItem):
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        bm_iter = prev_disc(bm_iter)
        if bm_iter == first:
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
        bm_iter = prev_disc(bm_iter)
        if bm_iter == first:
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
            bm_iter = prev_disc(bm_iter)
            if bm_iter == next_crn:
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
    # NOTE: It may return nothing for vertices that have only two edges (vertex with single linked face).
    # TODO: Check this (for 3D edge unwrap, this was not the desired behavior.)

    first_vert = crn.vert
    bm_iter = crn

    while True:
        bm_iter_prev = bm_iter.link_loop_prev
        bm_iter = bm_iter_prev.link_loop_radial_prev  # get ccw corner
        if (
            first_vert != bm_iter.vert or bm_iter_prev.edge.seam or bm_iter.face.hide
        ):  # Skip boundary or flipped
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
    return [
        l_crn
        for l_crn in crn.vert.link_loops
        if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co
    ]


def linked_crn_uv_by_idx_unordered_included(crn: BMLoop, uv: BMLayerItem):
    """Linked to arg corner by island index without arg corner
    simular - linked_crn_uv_by_island_index_unordered_included
    """
    first_co = crn[uv].uv
    idx = crn.face.index
    return [
        l_crn
        for l_crn in crn.vert.link_loops
        if l_crn.face.index == idx and l_crn[uv].uv == first_co
    ]


def linked_crn_uv_by_island_index_unordered_included(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index with arg corner"""
    first_co = crn[uv].uv
    return [
        l_crn
        for l_crn in crn.vert.link_loops
        if l_crn.face.index == idx and l_crn[uv].uv == first_co
    ]


def linked_crn_uv_by_island_index_unordered(crn: BMLoop, uv: BMLayerItem, idx: int):
    """Linked to arg corner by island index without arg corner"""
    first_co = crn[uv].uv
    return [
        l_crn
        for l_crn in crn.vert.link_loops
        if l_crn != crn and l_crn.face.index == idx and l_crn[uv].uv == first_co
    ]


def linked_crn_to_vert_by_idx_without_co_check_unordered(crn):
    """Linked to vertex by face index without arg corner"""
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx]


def linked_crn_to_vert_by_island_index_unordered(crn):
    """Linked to vertex by island index without arg corner"""
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx]


class ShortPath:
    @staticmethod
    def vert_tag_add_adjacent_uv(
        heap,
        l_a: BMLoop,
        loops_prev: list[BMLoop | None],
        cost: list[float],
        umesh,
        prioritize_corners,
        bound_priority_factor,
        visited,
        include_corners,
    ):
        import heapq
        uv = umesh.uv
        l_a_index = l_a.index
        uv_a = l_a[uv].uv

        # Loop over faces of face, but do so by first looping over loops.
        for l in linked_crn_to_vert_pair_with_seam_included(l_a, uv, umesh.sync):
            #  'l_a' is already tagged, tag all adjacent.

            visited.add(l)
            l_b = l.link_loop_next

            while True:
                if l_b in include_corners and l_b not in visited:
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

                l_b = l_b.link_loop_next
                if l_b == l:
                    break

    @staticmethod
    def calc_path_uv_vert(
        l_src: BMLoop,
        l_dst: BMLoop,
        umesh,
        include_corners: set[BMLoop],
        exclude_corners: set[BMLoop],
        prioritize_corners: set[BMLoop] | tuple = (),
        bound_priority_factor=0.9,
    ):
        import heapq
        from collections import deque
        uv = umesh.uv
        path = deque()
        # BM_ELEM_TAG flag is used to store visited edges
        heap: list[tuple[float, int, BMLoop]] = []

        # NOTE: would pass BM_EDGE except we are looping over all faces anyway.
        # BM_mesh_elem_index_ensure(bm, BM_LOOP); NOTE: not needed for face tag.
        prev_size = len(include_corners)
        include_corners = include_corners - exclude_corners
        if exclude_corners:
            assert prev_size != len(include_corners)

        include_corners.update(linked_crn_to_vert_pair_with_seam_included(l_src, umesh.uv, umesh.sync))
        include_corners.update(linked_crn_to_vert_pair_with_seam_included(l_dst, umesh.uv, umesh.sync))

        for idx, crn in enumerate(include_corners):
            crn.index = idx

        visited = set()
        # Allocate.
        loops_prev: list[BMLoop | None] = [None] * len(include_corners)
        cost = [1e100] * len(include_corners)

        # Regular dijkstra the shortest path, but over UV loops instead of vertices.
        heapq.heappush(heap, (0.0, l_src.index, l_src))
        cost[l_src.index] = 0.0

        while heap:
            l = heapq.heappop(heap)[2]
            if (l.vert == l_dst.vert) and l[uv].uv == l_dst[uv].uv:
                # assert l.face.index == l_dst.face.index
                while True:
                    path.appendleft(l)
                    l = loops_prev[l.index]
                    if not l:
                        break
                break

            if l not in visited:
                #  Adjacent loops are tagged while stepping to avoid 2x loops.
                visited.add(l)
                ShortPath.vert_tag_add_adjacent_uv(
                    heap, l, loops_prev, cost, umesh, prioritize_corners, bound_priority_factor, visited, include_corners
                )

        # assert bool(set(linked_crn_to_vert_pair_with_seam_included(l_src, umesh.uv, umesh.sync)) & set(path))
        # assert bool(set(linked_crn_to_vert_pair_with_seam_included(l_src, umesh.uv, umesh.sync)) & set(path))
        return ShortPath._short_pat_to_segment(path, umesh)  # TODO: Fix that.

    @staticmethod
    def _short_pat_to_segment(path, umesh):
        from . import is_boundary_func
        from ..utypes import AdvCorner, Segment

        uv = umesh.uv
        segment = []
        is_boundary = is_boundary_func(umesh)
        for i in range(len(path) - 1):
            linked = linked_crn_to_vert_pair_with_seam_included(path[i], umesh.uv, umesh.sync)
            next_v = path[i+1].vert
            for curr_crn in linked:
                if curr_crn.link_loop_next.vert == next_v:
                    new_adv_crn = AdvCorner(curr_crn, uv)
                    new_adv_crn.is_pair = not is_boundary(curr_crn)
                    segment.append(new_adv_crn)
                    break
            else:
                for curr_crn in linked:
                    prev_crn = curr_crn.link_loop_prev
                    if prev_crn.vert == next_v:
                        if is_boundary(prev_crn):
                            # Set invert for compare convention.
                            boundary = AdvCorner(curr_crn, uv, invert=True)
                            boundary.is_pair = False
                            segment.append(boundary)
                            break
                        else:
                            pair = AdvCorner(prev_crn.link_loop_radial_prev, uv)
                            pair.is_pair = True
                            segment.append(pair)
                            break
                else:
                    # TODO: Add assert for debug
                    print("UniV: Short Path: Missed corner in segment.")
        return Segment(segment, umesh)

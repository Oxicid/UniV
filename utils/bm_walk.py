# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy  # noqa
import heapq
import typing

from bmesh.types import BMesh, BMLoop, BMLayerItem

USE_GENERIC_UV_SYNC = hasattr(BMesh, "uv_select_sync_valid")


# TODO: Implement get_linked_crn_left_right_fn with `is_cyclic` state return.
def get_linked_crn_fn(umesh, *, included=False, with_seam=True, invisible_check=True) -> typing.Callable[[BMLoop], list[BMLoop]]:  # fmt: skip
    def catcher(is_boundary_, included_):
        def get_linked(first_crn: BMLoop):
            linked = [first_crn] if included_ else []
            bm_iter = first_crn
            # No ignore needed to realize that a full iteration has passed, and there is no need to calculate CW
            no_ignore_check = False
            while True:
                # CCW
                prev_crn = bm_iter.link_loop_prev
                pair_ccw = prev_crn.link_loop_radial_prev
                if pair_ccw == first_crn and no_ignore_check:  # End cycle or CW walking.
                    # End cycle.
                    break
                no_ignore_check = True

                # CW
                if is_boundary_(prev_crn):
                    bm_iter = first_crn
                    while True:
                        if is_boundary_(bm_iter):
                            break

                        next_crn = bm_iter.link_loop_radial_prev.link_loop_next
                        if next_crn == first_crn:  # End cycle.
                            break

                        bm_iter = next_crn
                        linked.append(next_crn)
                    break
                bm_iter = pair_ccw
                linked.append(bm_iter)
            return linked
        return get_linked
    from . import is_boundary_func
    return catcher(is_boundary_func(umesh, with_seam=with_seam, invisible_check=invisible_check), included_=included)


def get_linked_crn_iter_fn(umesh, *, included=False, with_seam=True, invisible_check=True) \
        -> typing.Callable[[BMLoop], "typing.Generator[BMLoop]"]:  # fmt: skip
    def catcher(is_boundary_, included_):
        def get_linked(first_crn: BMLoop):
            if included_:
                yield first_crn
            bm_iter = first_crn
            # No ignore needed to realize that a full iteration has passed, and there is no need to calculate CW
            no_ignore_check = False
            while True:
                # CCW
                prev_crn = bm_iter.link_loop_prev
                pair_ccw = prev_crn.link_loop_radial_prev
                if pair_ccw == first_crn and no_ignore_check:  # End cycle or CW walking.
                    # End cycle.
                    break
                no_ignore_check = True

                # CW
                if is_boundary_(prev_crn):
                    bm_iter = first_crn
                    while True:
                        if is_boundary_(bm_iter):
                            break

                        next_crn = bm_iter.link_loop_radial_prev.link_loop_next
                        if next_crn == first_crn:  # End cycle.
                            break

                        bm_iter = next_crn
                        yield next_crn
                    break
                bm_iter = pair_ccw
                yield bm_iter
        return get_linked
    from . import is_boundary_func
    return catcher(is_boundary_func(umesh, with_seam=with_seam, invisible_check=invisible_check), included_=included)


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
                linked.append(next_crn)
            break
        bm_iter = pair_ccw
        linked.append(bm_iter)
    # assert len(linked) == len(set(linked))
    return linked


def linked_crn_uv_unordered_included_with_hidden(first: BMLoop, uv: BMLayerItem):
    # This need for skip checking hidden untagged edges/faces.
    first_co = first[uv].uv
    linked = [l_crn for l_crn in first.vert.link_loops if l_crn[uv].uv == first_co]
    return linked


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
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev
        if bm_iter == first:
            break
        if not bm_iter.tag:
            continue
        if first_co == bm_iter[uv].uv:
            linked.append(bm_iter)
    return linked


# TODO: Replace with linked unordered (change logic in extend_from_linked)
def linked_crn_vert_uv_by_tag_with_untagging_for_transform(first, uv):
    # Need tagging. tag == False - not append
    # assert utils.sync()
    linked = []
    bm_iter = first
    first_co = first[uv].uv
    while True:
        bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev
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
            bm_iter = bm_iter.link_loop_prev.link_loop_radial_prev
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


def get_linked_crn_3d_iter_fn(*, included=False, with_seam=True, invisible_check=True) -> typing.Callable[[BMLoop], "typing.Generator[BMLoop]"]:  # fmt: skip
    # NOTE: It may return nothing for vertices that have only two edges (vertex with single linked face).  # TODO: Check this (for 3D edge unwrap, this was not the desired behavior.)  # fmt: skip

    def catcher(is_boundary_3d_, included_):
        def linked_crn_to_vert_3d_iter_(crn: BMLoop):
            """Linked to arg corner by visible faces"""
            if included_:
                yield crn

            bm_iter = crn
            while True:
                bm_iter_prev = bm_iter.link_loop_prev
                if is_boundary_3d_(bm_iter_prev):  # Skip boundary.
                    bm_iter = crn
                    while True:
                        if is_boundary_3d_(bm_iter):  # clamp by seam
                            return

                        bm_iter = bm_iter.link_loop_radial_next.link_loop_next  # get cw corner
                        if bm_iter == crn:
                            return
                        yield bm_iter

                bm_iter = bm_iter_prev.link_loop_radial_prev
                if bm_iter == crn:
                    return

                yield bm_iter
        return linked_crn_to_vert_3d_iter_

    from . import is_boundary_3d_func
    return catcher(is_boundary_3d_=is_boundary_3d_func(with_seam=with_seam, invisible_check=invisible_check), included_=included)


def linked_crn_to_vert_by_idx_without_co_check_unordered(crn):
    """Linked to vertex by face index without arg corner"""
    idx = crn.face.index
    return [l_crn for l_crn in crn.vert.link_loops if l_crn != crn and l_crn.face.index == idx]


class ShortPath:
    @staticmethod
    def vert_tag_add_adjacent_uv_fn(
        umesh,
        heap_: list[tuple[float, int, BMLoop]],
        loops_prev_: list[BMLoop | None],
        cost_: list[float],
        prioritize_corners_: set[BMLoop],
        bound_priority_factor_: float,
        include_corners_: set[BMLoop],
        get_linked_inc_,
    ) -> typing.Callable[[BMLoop], None]:
        def catcher(uv,
                    heap,
                    loops_prev,
                    cost,
                    prioritize_corners,
                    bound_priority_factor,
                    include_corners,
                    get_linked_inc
                    ):
            def vert_tag_add_adjacent_uv(crn_a: BMLoop):
                l_a_index = crn_a.index
                uv_a = crn_a[uv].uv

                # get_linked_inc = get_linked_crn_iter_fn(umesh, included=True)
                # Loop over faces of face, but do so by first looping over loops.
                for l_crn in get_linked_inc(crn_a):
                    #  'l_a' is already tagged, tag all adjacent.
                    l_crn.tag = False

                    for crn_b in (l_crn.link_loop_next, l_crn.link_loop_prev):
                        if crn_b.tag:
                            if crn_b in include_corners:
                                uv_b = crn_b[uv].uv
                                # We know 'crn_b' is not visited, check it out!
                                l_b_index = crn_b.index
                                cost_cut = (uv_a - uv_b).length

                                # Prioritize.
                                if crn_b in prioritize_corners:
                                    cost_cut *= bound_priority_factor

                                new_cost = cost[l_a_index] + cost_cut
                                if new_cost < cost[l_b_index]:
                                    cost[l_b_index] = new_cost
                                    loops_prev[l_b_index] = crn_a
                                    heapq.heappush(heap, (new_cost, id(crn_b), crn_b))

            return vert_tag_add_adjacent_uv
        return catcher(umesh.uv,
                       heap_,
                       loops_prev_,
                       cost_,
                       prioritize_corners_,
                       bound_priority_factor_,
                       include_corners_,
                       get_linked_inc_)

    @classmethod
    def calc_path_uv_vert(
        cls,
        l_src: BMLoop,
        l_dst: BMLoop,
        umesh,
        include_corners: set[BMLoop],  # NOTE: `set` and `id(crn)` get random priority by session.
        exclude_corners: set[BMLoop],
        prioritize_corners: set[BMLoop] | tuple = (),
        bound_priority_factor=0.9,
    ):
        from heapq import heappop
        from collections import deque
        uv = umesh.uv

        # NOTE: would pass BM_EDGE except we are looping over all faces anyway.
        # BM_mesh_elem_index_ensure(bm, BM_LOOP); NOTE: not needed for face tag.
        prev_size = len(include_corners)
        include_corners = include_corners - exclude_corners

        if exclude_corners:
            # An unlikely case. However, in the case of non-contiguous edges, this possibility cannot be ruled out.
            assert prev_size != len(include_corners)

        get_linked_inc = get_linked_crn_fn(umesh, included=True)
        include_corners.update(get_linked_inc(l_src))
        include_corners.update(get_linked_inc(l_dst))

        for idx, crn in enumerate(include_corners):
            # Index is used for `cost` indexing and for unique number for heapq.
            crn.index = idx
            # Tag flag is used to store visited edges (untagged is visited).
            crn.tag = True

        # Allocate.
        loops_prev: list[BMLoop | None] = [None] * len(include_corners)

        # Regular dijkstra the shortest path, but over UV loops instead of vertices.
        heap: list[tuple[float, int, BMLoop]] = []
        heapq.heappush(heap, (0.0, l_src.index, l_src))

        cost = [1e100] * len(include_corners)
        cost[l_src.index] = 0.0

        vert_tag_add_adjacent_uv = cls.vert_tag_add_adjacent_uv_fn(
            umesh, heap, loops_prev, cost, prioritize_corners, bound_priority_factor, include_corners, get_linked_inc)

        path = deque()
        while heap:
            crn = heappop(heap)[2]
            if (crn.vert == l_dst.vert) and crn[uv].uv == l_dst[uv].uv:
                while True:
                    path.appendleft(crn)
                    crn = loops_prev[crn.index]
                    if not crn:
                        break
                break

            if crn.tag:
                #  Adjacent loops are tagged while stepping to avoid 2x loops.
                crn.tag = False
                vert_tag_add_adjacent_uv(crn)

        # assert bool(set(get_linked(l_src)) & set(path))
        # assert bool(set(get_linked(l_src)) & set(path))
        return cls._short_path_to_segment(path, umesh)

    @staticmethod
    def _short_path_to_segment(path, umesh):
        from . import is_boundary_func
        from ..utypes import AdvCorner, Segment

        segment = []
        is_boundary = is_boundary_func(umesh)
        get_linked_inc = get_linked_crn_fn(umesh, included=True)

        # TODO: Add `splitshift` function.
        for i in range(len(path) - 1):
            next_v = path[i+1].vert

            linked = get_linked_inc(path[i])
            for curr_crn in linked:
                if curr_crn.link_loop_next.vert == next_v:
                    new_adv_crn = AdvCorner(curr_crn, umesh, is_pair=not is_boundary(curr_crn))
                    segment.append(new_adv_crn)
                    break
            else:
                # A likely case. Do not replace `get_linked` with `get_linked_iter`.
                for curr_crn in linked:
                    prev_crn = curr_crn.link_loop_prev
                    if prev_crn.vert == next_v:
                        if is_boundary(prev_crn):
                            # Set invert for compare convention.
                            boundary = AdvCorner(curr_crn, umesh, is_pair=False, invert=True)
                            boundary.is_pair = False
                            segment.append(boundary)
                            break
                        else:
                            pair = AdvCorner(prev_crn.link_loop_radial_prev, umesh, is_pair=True)
                            pair.is_pair = True
                            segment.append(pair)
                            break
                else:
                    # TODO: Add assert for debug
                    print("UniV: Short Path: Missed corner in segment.")
        return Segment(segment, umesh)

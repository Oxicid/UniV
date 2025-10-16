# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import bmesh
import typing

from .. import utypes
from bmesh.types import BMFace, BMLoop

USE_GENERIC_UV_SYNC = hasattr(bmesh.types.BMesh, 'uv_select_sync_valid')

if USE_GENERIC_UV_SYNC:
    def face_select_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
        if umesh.sync and not umesh.sync_valid:
            def select_set(f):
                f.select = True
        else:
            def select_set(f):
                f.select = True
                f.uv_select = True
                for crn in f.loops:
                    crn.uv_select_vert = True
                    crn.uv_select_edge = True
        return select_set
else:
    def face_select_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
        def inner(uv, sync):
            if sync:
                def select_set(f):
                    f.select = True
            else:
                def select_set(f):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        crn_uv.select = True
                        crn_uv.select_edge = True
            return select_set
        return inner(umesh.uv, umesh.sync)

if USE_GENERIC_UV_SYNC:
    def face_deselect_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
        if umesh.sync:
            if not umesh.sync_valid:
                def select_set(f: BMFace):
                    f.select = False
            else:
                if umesh.elem_mode == 'VERT':
                    def select_set(f: BMFace):
                        f.uv_select = False
                        for crn in f.loops:
                            crn.uv_select_vert = False
                            crn.uv_select_edge = False
                            crn.vert.select = any(crn_b.uv_select_vert for crn_b in crn.vert.link_loops if not crn_b.face.hide)

                elif umesh.elem_mode == 'EDGE':
                    def select_set(f: BMFace):
                        f.uv_select = False
                        for crn in f.loops:
                            crn.uv_select_vert = False
                            crn.uv_select_edge = False
                            pair = crn.link_loop_radial_prev
                            if pair.face.hide or not pair.uv_select_edge:
                                crn.edge.select = False
                else:
                    def select_set(f: BMFace):
                        f.select = False
                        f.uv_select = False
                        for crn in f.loops:
                            crn.uv_select_vert = False
                            crn.uv_select_edge = False
        else:
            def select_set(f: BMFace):
                f.uv_select = False
                for crn in f.loops:
                    crn.uv_select_vert = False
                    crn.uv_select_edge = False
        return select_set
else:
    def face_deselect_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
        def inner(uv, sync):
            if sync:
                def select_set(f):
                    f.select = False
            else:
                def select_set(f):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        crn_uv.select = False
                        crn_uv.select_edge = False
            return select_set
        return inner(umesh.uv, umesh.sync)

if USE_GENERIC_UV_SYNC:
    def face_select_set_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace, bool], typing.NoReturn]:  # noqa
        if umesh.sync and not umesh.sync_valid:
            select_set = BMFace.select.__set__
        else:
            if umesh.sync:
                def select_set(f, state):
                    f.select = state
                    f.uv_select = state
                    for crn in f.loops:
                        crn.uv_select_vert = state
                        crn.uv_select_edge = state
            else:
                def select_set(f, state):
                    f.uv_select = state
                    for crn in f.loops:
                        crn.uv_select_vert = state
                        crn.uv_select_edge = state
        return select_set
else:
    def face_select_set_func(umesh: 'types.UMesh') -> typing.Callable[[BMFace, bool], typing.NoReturn]:  # noqa
        def inner(uv, sync):
            if sync:
                select_set = BMFace.select.__set__
            else:
                def select_set(f, state):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        crn_uv.select = state
                        crn_uv.select_edge = state
            return select_set
        return inner(umesh.uv, umesh.sync)


def face_select_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], bool]:
    if umesh.sync and not umesh.sync_valid:
        select_get = BMFace.select.__get__
    else:
        if umesh.sync:
            def select_get(f):
                return (not f.hide) and f.uv_select
        else:
            def select_get(f):
                return (not f.hide) and f.select and f.uv_select
    return select_get

if USE_GENERIC_UV_SYNC:
    def face_select_linked_func(umesh: 'utypes.UMesh', force=False, clamp_by_seams=False) -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
        def inner_catcher(uv, sync_invalid, face_is_invisible):
            if force or clamp_by_seams:
                raise NotImplementedError()

            if sync_invalid:
                select_set = BMFace.select.__set__
            else:
                def select_set(f: BMFace):
                    f.select = True  # useless for no-sync
                    f.uv_select = True
                    for crn in f.loops:
                        vert_co = crn[uv].uv
                        if not crn.uv_select_vert:
                            crn.uv_select_vert = True

                            for linked_crn_ in crn.vert.link_loops:
                                if not face_is_invisible(linked_crn_.face):
                                    if linked_crn_[uv].uv == vert_co:
                                        linked_crn_.uv_select_vert = True

                        if not crn.uv_select_edge:
                            crn.uv_select_edge = True

                            pair_crn = crn.link_loop_radial_prev
                            if not pair_crn.uv_select_edge and not face_is_invisible(pair_crn.face):
                                if crn.link_loop_next[uv].uv == pair_crn[uv].uv and vert_co == pair_crn.link_loop_next[uv].uv:
                                    pair_crn.uv_select_edge = True
            return select_set
        return inner_catcher(umesh.uv, (umesh.sync and not umesh.sync_valid), face_invisible_get_func(umesh))
else:
    def face_select_linked_func(umesh: 'utypes.UMesh', force=False, clamp_by_seams=False) -> typing.Callable[
        [BMFace], typing.NoReturn]:  # noqa
        def inner(uv, sync):
            if force or clamp_by_seams:
                raise NotImplementedError()

            if sync:
                def select_set(f):  # noqa
                    f.select = True
            else:
                def select_set(f):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        vert_co_a = crn_uv.uv

                        if not crn_uv.select:
                            crn_uv.select = True
                            for linked_crn_ in crn.vert.link_loops:
                                if linked_crn_.face.select:
                                    linked_crn_uv_ = linked_crn_[uv]
                                    if linked_crn_uv_.uv == vert_co_a:
                                        linked_crn_uv_.select = True

                        if not crn_uv.select_edge:
                            crn_uv.select_edge = True
                            v2_co = crn.link_loop_next[uv].uv
                            pair_crn = crn.link_loop_radial_prev

                            if pair_crn.face.select and not (pair_crn_uv := pair_crn[uv]).select_edge:
                                if v2_co == pair_crn_uv.uv and vert_co_a == pair_crn.link_loop_next[uv].uv:
                                    pair_crn_uv.select_edge = True
            return select_set

        return inner(umesh.uv, umesh.sync)

def face_visible_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:
    if umesh.sync:
        return lambda f: not f.hide
    else:
        return BMFace.select.__get__


def face_invisible_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMFace], typing.NoReturn]:
    if umesh.sync:
        return BMFace.hide.__get__
    else:
        return lambda f: not f.select


if USE_GENERIC_UV_SYNC:
    # TODO: Add support clamp by seams
    def edge_select_linked_set_func(umesh: 'utypes.UMesh', force=False,
                                    clamp_by_seams=False) -> typing.Callable[[BMLoop, bool], typing.NoReturn]:
        # NOTE: UV_SELECT_FLUSH_MODE_NEEDED and UV_SELECT_SYNC_TO_MESH_NEEDED for deselect
        def inner(uv, sync, sync_invalid, face_is_invisible):
            if sync_invalid:
                def select_set(crn, state):
                    crn.edge.select = state
            else:
                if force or clamp_by_seams:
                    raise NotImplementedError()

                def select_set(crn: BMLoop, state):
                    # crn_uv = crn[uv]
                    # Check vertex select to avoid selected single vert
                    if crn.uv_select_edge == state and crn.uv_select_vert == state:
                        return

                    v1_co = crn[uv].uv
                    v2_co = crn.link_loop_next[uv].uv
                    pair_crn = crn.link_loop_radial_prev
                    if state:
                        # Select pair edge
                        crn.select_edge = True
                        if not face_is_invisible(pair_crn.face) and not pair_crn.uv_select_edge:
                            if v2_co == pair_crn[uv].uv and v1_co == pair_crn.link_loop_next[uv].uv:
                                pair_crn.select_edge = True

                        # Select A
                        for linked_crn_ in crn.vert.link_loops:
                            if not face_is_invisible(linked_crn_.face):
                                if linked_crn_[uv].uv == v1_co:
                                    linked_crn_.uv_select_vert = True
                        # Select B
                        for linked_crn_ in crn.link_loop_next.vert.link_loops:
                            if not face_is_invisible(linked_crn_.face):
                                if linked_crn_[uv].uv == v2_co:
                                    linked_crn_.uv_select_vert = True
                    else:
                        crn.edge.select = False
                        if sync:
                            crn.edge.select = False
                        if not face_is_invisible(pair_crn.face) and pair_crn.uv_select_edge:
                            if v2_co == pair_crn[uv].uv and v1_co == pair_crn.link_loop_next[uv].uv:
                                pair_crn.select_edge = False

                        to_deselect = []
                        has_linked_selected_edge_a = False
                        # When deselecting uv_vert_select, make sure that there are no linked selected edges.
                        # Deselect A
                        for linked_crn_ in crn.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v1_co:
                                    to_deselect.append(linked_crn_uv_)
                                    if linked_crn_uv_.select_edge or linked_crn_.link_loop_prev[uv].select_edge:
                                        has_linked_selected_edge_a = True
                                        break
                        if not has_linked_selected_edge_a:
                            for crn_uv in to_deselect:
                                crn_uv.select = False
                        to_deselect.clear()

                        # Deselect B
                        has_linked_selected_edge_b = False
                        for linked_crn_ in crn.link_loop_next.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v2_co:
                                    to_deselect.append(linked_crn_uv_)
                                    if linked_crn_uv_.select_edge or linked_crn_.link_loop_prev[uv].select_edge:
                                        has_linked_selected_edge_b = True
                                        break
                        if not has_linked_selected_edge_b:
                            for crn_uv in to_deselect:
                                crn_uv.select = False


            return select_set
        return inner(umesh.uv, umesh.sync, (umesh.sync and not umesh.sync_valid), face_invisible_get_func(umesh))
else:
    def edge_select_linked_set_func(umesh: 'utypes.UMesh', force=False,
                                    clamp_by_seams=False) -> typing.Callable[[BMLoop, bool], typing.NoReturn]:
        def inner(uv, sync):
            if sync:
                def select_set(crn, state):
                    crn.edge.select = state
            else:
                if force or clamp_by_seams:
                    raise NotImplementedError()

                def select_set(crn: BMLoop, state):
                    crn_uv = crn[uv]
                    # Check vertex select to avoid selected single vert
                    if crn_uv.select_edge == state and crn_uv.select == state:
                        return

                    v1_co = crn_uv.uv
                    v2_co = crn.link_loop_next[uv].uv
                    pair_crn = crn.link_loop_radial_prev
                    if state:
                        crn_uv.select_edge = True
                        if pair_crn.face.select and not (pair_crn_uv := pair_crn[uv]).select_edge:
                            if v2_co == pair_crn_uv.uv and v1_co == pair_crn.link_loop_next[uv].uv:
                                pair_crn_uv.select_edge = True

                        # Select A
                        for linked_crn_ in crn.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v1_co:
                                    linked_crn_uv_.select = True
                        # Select B
                        for linked_crn_ in crn.link_loop_next.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v2_co:
                                    linked_crn_uv_.select = True
                    else:
                        crn_uv.select_edge = False
                        if pair_crn.face.select and (pair_crn_uv := pair_crn[uv]).select_edge:
                            if v2_co == pair_crn_uv.uv and v1_co == pair_crn.link_loop_next[uv].uv:
                                pair_crn_uv.select_edge = False

                        to_deselect = []
                        has_linked_selected_edge = False
                        # Deselect A
                        for linked_crn_ in crn.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v1_co:
                                    to_deselect.append(linked_crn_uv_)
                                    if linked_crn_uv_.select_edge or linked_crn_.link_loop_prev[uv].select_edge:
                                        has_linked_selected_edge = True
                                        break
                        if not has_linked_selected_edge:
                            for crn_uv in to_deselect:
                                crn_uv.select = False
                        to_deselect.clear()

                        # Deselect B
                        has_linked_selected_edge = False
                        for linked_crn_ in crn.link_loop_next.vert.link_loops:
                            if linked_crn_.face.select:
                                linked_crn_uv_ = linked_crn_[uv]
                                if linked_crn_uv_.uv == v2_co:
                                    to_deselect.append(linked_crn_uv_)
                                    if linked_crn_uv_.select_edge or linked_crn_.link_loop_prev[uv].select_edge:
                                        has_linked_selected_edge = True
                                        break
                        if not has_linked_selected_edge:
                            for crn_uv in to_deselect:
                                crn_uv.select = False
            return select_set

        return inner(umesh.uv, umesh.sync)

if USE_GENERIC_UV_SYNC:
    def edge_select_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMLoop], bool]:
        def inner(sync, sync_valid):
            def select_get(crn):  # noqa
                return crn.uv_select_edge
            if sync and not sync_valid:
                def select_get(crn):  # noqa
                    return crn.edge.select
            return select_get
        return inner(umesh.sync, umesh.sync_valid)
else:
    def edge_select_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMLoop], bool]:
        def inner(uv, sync):
            if sync:
                def select_get(crn):
                    return crn.edge.select
            else:
                def select_get(crn):
                    return crn[uv].select_edge
            return select_get
        return inner(umesh.uv, umesh.sync)

def edge_deselect_safe_3d_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMLoop], None]:
    """In VERTEX sync mode, deselecting an edge also affects neighboring edges - this preserves that behavior, even for unintended edges."""
    assert umesh.sync
    assert umesh.elem_mode in ('VERT', 'EDGE')

    if umesh.elem_mode == 'EDGE':
        def deselect_edge(crn):
            crn.edge.select = False
    else:
        def deselect_edge(crn):
            next_vert = crn.link_loop_next.vert
            more_one_selected_edges_a = sum(e.select for e in crn.vert.link_edges) > 1
            more_one_selected_edges_b = sum(e.select for e in next_vert.link_edges) > 1
            if more_one_selected_edges_a and more_one_selected_edges_b:
                return
            else:
                crn.edge.select = False
                crn.vert.select = more_one_selected_edges_a
                next_vert.select = more_one_selected_edges_b
    return deselect_edge

if USE_GENERIC_UV_SYNC:
    def vert_select_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMLoop], bool]:
        def inner(sync, sync_valid):
            def select_get(crn):  # noqa
                return crn.uv_select_vert
            if sync and not sync_valid:
                def select_get(crn):  # noqa
                    return crn.vert.select
            return select_get
        return inner(umesh.sync, umesh.sync_valid)
else:
    def vert_select_get_func(umesh: 'utypes.UMesh') -> typing.Callable[[BMLoop], bool]:
        def inner(uv, sync):
            if sync:
                def select_get(crn):  # noqa
                    return crn.vert.select
            else:
                def select_get(crn):
                    return crn[uv].select
            return select_get
        return inner(umesh.uv, umesh.sync)

def select_crn_uv_edge_with_shared_by_idx(crn: BMLoop, uv, force=False):
    idx = crn.face.index
    from .bm_tag import shared_is_linked
    from .bm_walk import linked_crn_uv_by_island_index_unordered_included

    if (shared := crn.link_loop_radial_prev) != crn and shared.face.index == idx and shared_is_linked(crn, shared, uv):
        shared[uv].select_edge = True

    if force:
        crn_uv_a = crn[uv]
        crn_uv_a.select_edge = True
        for crn_a in linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
            crn_a[uv].select = True

        crn_uv_next = crn.link_loop_next
        for crn_b in linked_crn_uv_by_island_index_unordered_included(crn_uv_next, uv, idx):
            crn_b[uv].select = True
    else:
        crn_uv_a = crn[uv]
        crn_uv_a.select_edge = True
        if not crn_uv_a.select:
            for crn_a in linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                crn_a[uv].select = True

        crn_uv_next = crn.link_loop_next
        if not crn_uv_next[uv].select:
            for crn_b in linked_crn_uv_by_island_index_unordered_included(crn_uv_next, uv, idx):
                crn_b[uv].select = True


def select_edge_processing(umesh, to_deselect, to_select):
    if USE_GENERIC_UV_SYNC:
        if not umesh.sync_valid:
            umesh.bm.uv_select_sync_from_mesh()
            umesh.sync_valid = True

        if to_deselect:
            umesh.bm.uv_select_foreach_set(False, loop_verts=to_deselect, loop_edges=to_deselect)
            umesh.bm.uv_select_flush(False)
        if to_select:
            umesh.bm.uv_select_foreach_set(True, loop_edges=to_select)
            umesh.bm.uv_select_flush(False)
        if umesh.sync:
            umesh.bm.uv_select_sync_to_mesh()
    else:
        set_edge_select = edge_select_linked_set_func(umesh)
        for crn in to_deselect:
            set_edge_select(crn, False)
        for crn in to_select:
            set_edge_select(crn, True)
        umesh.bm.select_flush(True)
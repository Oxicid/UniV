# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import typing

from bmesh.types import BMFace, BMLoop
# from math import isclose
# from mathutils import Vector
# from mathutils.geometry import area_tri, intersect_point_tri_2d
# from collections import deque
# from itertools import chain

from .. import types



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

def face_select_get_func(umesh: 'types.UMesh') -> typing.Callable[[BMFace], bool]:
    def inner(uv, sync):
        if sync:
            select_get = BMFace.select.__get__
        else:
            if umesh.elem_mode == 'EDGE':
                def select_get(f):
                    for crn in f.loops:
                        if not crn[uv].select_edge:
                            return False
                    return True
            else:
                def select_get(f):
                    for crn in f.loops:
                        if not crn[uv].select:
                            return False
                    return True

        return select_get
    return inner(umesh.uv, umesh.sync)

def face_select_linked_func(umesh: 'types.UMesh', force=False, clamp_by_seams=False) -> typing.Callable[[BMFace], typing.NoReturn]:  # noqa
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

def edge_select_linked_set_func(umesh: 'types.UMesh', force=False, clamp_by_seams=False) -> typing.Callable[[BMLoop, bool], typing.NoReturn]:
    # TODO: Add support clamp by seams
    def inner(uv, sync):
        if sync:
            def select_set(crn, state):  # noqa
                crn.edge.select = state
        else:
            if force or clamp_by_seams:
                raise NotImplementedError()

            def select_set(crn: BMLoop, state):
                crn_uv = crn[uv]
                if crn_uv.select_edge == state and crn_uv.select == state:  # Check vertex select to avoid selected single vert
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
                    if pair_crn.face.select and not (pair_crn_uv := pair_crn[uv]).select_edge:
                        if v2_co == pair_crn_uv.uv and v1_co == pair_crn.link_loop_next[uv].uv:
                            pair_crn_uv.select_edge = False

                    to_deselect = []
                    to_deselect_append = to_deselect.append
                    has_linked_selected_edge = False
                    # Deselect A
                    for linked_crn_ in crn.vert.link_loops:
                        if linked_crn_.face.select:
                            linked_crn_uv_ = linked_crn_[uv]
                            if linked_crn_uv_.uv == v1_co:
                                to_deselect_append(linked_crn_uv_)
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
                                to_deselect_append(linked_crn_uv_)
                                if linked_crn_uv_.select_edge or linked_crn_.link_loop_prev[uv].select_edge:
                                    has_linked_selected_edge = True
                                    break
                    if not has_linked_selected_edge:
                        for crn_uv in to_deselect:
                            crn_uv.select = False
        return select_set
    return inner(umesh.uv, umesh.sync)

def edge_select_get_func(umesh: 'types.UMesh') -> typing.Callable[[BMLoop], bool]:
    def inner(uv, sync):
        if sync:
            def select_get(crn):  # noqa
                return crn.edge.select
        else:
            def select_get(crn):
                return crn[uv].select_edge
        return select_get
    return inner(umesh.uv, umesh.sync)

def vert_select_get_func(umesh: 'types.UMesh') -> typing.Callable[[BMLoop], bool]:
    def inner(uv, sync):
        if sync:
            def select_get(crn):  # noqa
                return crn.vert.select
        else:
            def select_get(crn):
                return crn[uv].select
        return select_get
    return inner(umesh.uv, umesh.sync)
# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import typing

from bmesh.types import BMesh, BMFace, BMLoop, BMLayerItem

USE_GENERIC_UV_SYNC = hasattr(BMesh, 'uv_select_sync_valid')

def shared_crn(crn: BMLoop) -> BMLoop | None:
    shared = crn.link_loop_radial_prev
    if shared != crn:
        return shared


def is_flipped_3d(crn):
    pair = crn.link_loop_radial_prev
    if pair == crn:
        return False
    return pair.vert == crn.vert


def shared_is_linked(crn: BMLoop, _shared_crn: BMLoop, uv: BMLayerItem):
    return crn.link_loop_next[uv].uv == _shared_crn[uv].uv and \
        crn[uv].uv == _shared_crn.link_loop_next[uv].uv


def is_pair(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    return crn.link_loop_next[uv].uv == _rad_prev[uv].uv and \
        crn[uv].uv == _rad_prev.link_loop_next[uv].uv


def is_pair_with_flip(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    if crn.vert == _rad_prev.vert:  # is flipped
        return crn[uv].uv.to_tuple() == _rad_prev[uv].uv.to_tuple() and \
            crn.link_loop_next[uv].uv.to_tuple() == _rad_prev.link_loop_next[uv].uv.to_tuple()
    return crn.link_loop_next[uv].uv.to_tuple() == _rad_prev[uv].uv.to_tuple() and \
        crn[uv].uv.to_tuple() == _rad_prev.link_loop_next[uv].uv.to_tuple()


def has_pair_with_ms(crn: BMLoop, uv: BMLayerItem):
    if crn.edge.seam or crn == (pair := crn.link_loop_radial_prev):
        return False
    if crn.vert == pair.vert:  # avoid flipped 3d
        return False
    return crn.link_loop_next[uv].uv == pair[uv].uv and \
        crn[uv].uv == pair.link_loop_next[uv].uv


def is_pair_by_idx(crn: BMLoop, _rad_prev: BMLoop, uv: BMLayerItem):
    if crn == _rad_prev or crn.face.index != _rad_prev.face.index:
        return False
    return crn.link_loop_next[uv].uv == _rad_prev[uv].uv and \
           crn[uv].uv == _rad_prev.link_loop_next[uv].uv  # noqa

def set_faces_tag(faces, tag=True):
    if tag:  # Constant load optimisation
        for f in faces:
            f.tag = True
    else:
        for f in faces:
            f.tag = False


def is_boundary_non_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(l.face.select)
    if (pair := crn.link_loop_radial_prev) == crn:
        return True
    if not pair.face.select:
        return True
    return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
            crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())


def is_boundary_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(not l.face.hide)
    if (pair := crn.link_loop_radial_prev) == crn:
        return True
    if pair.face.hide:
        return True
    return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
            crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())

def is_boundary_with_flip_check_non_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(l.face.select)
    if (pair := crn.link_loop_radial_prev) == crn:
        return True
    if not pair.face.select:
        return True
    if crn.vert == pair.vert:
        return True
    return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
            crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())


def is_boundary_with_flip_check_sync(crn: BMLoop, uv: BMLayerItem):
    # assert(not l.face.hide)
    if (pair := crn.link_loop_radial_prev) == crn:
        return True
    if pair.face.hide:
        return True
    if crn.vert == pair.vert:
        return True
    return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
            crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())


def is_boundary_func(umesh, with_seam=True, with_flipped_check=True, invisible_check=True) -> typing.Callable[[BMLoop], bool]:
    def catcher(uv: BMLayerItem, is_boundary_):
        if with_seam:
            def is_boundary(crn: BMLoop):
                # assert(l.face.select)
                if crn.edge.seam:
                    return True
                return is_boundary_(crn, uv)
            return is_boundary
        else:
            def is_boundary(crn: BMLoop):
                # assert(l.face.select)
                return is_boundary_(crn, uv)
            return is_boundary


    if invisible_check:
        if umesh.sync:
            if with_flipped_check:
                return catcher(umesh.uv, is_boundary_with_flip_check_sync)
            else:
                return catcher(umesh.uv, is_boundary_sync)
        else:
            if with_flipped_check:
                return catcher(umesh.uv, is_boundary_with_flip_check_non_sync)
            else:
                return catcher(umesh.uv, is_boundary_non_sync)
    else:
        if with_flipped_check:
            def is_boundary_with_flip_no_invisible_check(crn: BMLoop, uv: BMLayerItem):
                if (pair := crn.link_loop_radial_prev) == crn:
                    return True
                if crn.vert == pair.vert:
                    return True
                return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
                        crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())

            return catcher(umesh.uv, is_boundary_with_flip_no_invisible_check)
        else:
            def is_boundary_no_invisible_check(crn: BMLoop, uv: BMLayerItem):
                if (pair := crn.link_loop_radial_prev) == crn:
                    return True
                return (crn[uv].uv.to_tuple() != pair.link_loop_next[uv].uv.to_tuple() or
                        crn.link_loop_next[uv].uv.to_tuple() != pair[uv].uv.to_tuple())

            return catcher(umesh.uv, is_boundary_no_invisible_check)


def is_visible_func(sync: bool):
    if sync:
        return lambda f: not f.hide
    else:
        return BMFace.select.__get__


def is_invisible_func(sync: bool):
    if sync:
        return BMFace.hide.__get__
    else:
        return lambda f: not f.select
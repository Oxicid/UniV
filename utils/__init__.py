# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing  # noqa
import mathutils

import numpy as np  # noqa
from math import pi

from .bench import timer, profile
from .draw import *
from .other import *
from .shapes import *
from .ubm import *
from .umath import *
from .. import types

class NoInit:
    def __getattribute__(self, item):
        raise AttributeError(f'Object not initialized')

    def __bool__(self):
        raise AttributeError(f'Object not initialized')

    def __len__(self):
        raise AttributeError(f'Object not initialized')

def sync():
    return bpy.context.scene.tool_settings.use_uv_select_sync

def calc_avg_normal():
    umeshes = types.UMeshes.sel_ob_with_uv()
    size = sum(len(umesh.bm.faces) for umesh in umeshes)

    normals = np.empty(3 * size).reshape((-1, 3))
    areas = np.empty(size)

    i = 0
    for umesh in umeshes:
        for f in umesh.bm.faces:
            normals[i] = f.normal.to_tuple()
            areas[i] = f.calc_area()
            i += 1

    weighted_normals = normals * areas[:, np.newaxis]
    summed_normals = np.sum(weighted_normals, axis=0)

    return summed_normals / np.linalg.norm(summed_normals)

def find_min_rotate_angle(angle):
    return -(round(angle / (pi / 2)) * (pi / 2) - angle)

def calc_convex_points(points_append):
    return [points_append[i] for i in mathutils.geometry.convex_hull_2d(points_append)]

def calc_min_align_angle(points, aspect=1.0):
    if aspect != 1.0:
        vec_aspect = Vector((aspect, 1.0))
        points = [pt*vec_aspect for pt in points]
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)


def calc_min_align_angle_pt(points):
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)

def get_cursor_location() -> Vector:
    if bpy.context.area.ui_type == 'UV':
        return bpy.context.space_data.cursor_location.copy()
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.ui_type == 'UV':
                return area.spaces.active.cursor_location.copy()

def get_tile_from_cursor() -> Vector:
    if cursor := get_cursor_location():
        return Vector((math.floor(val) for val in cursor))

def set_cursor_location(loc):
    if bpy.context.area.ui_type == 'UV':
        bpy.context.space_data.cursor_location = loc
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.ui_type == 'UV':
                area.spaces.active.cursor_location = loc
                return

def update_area_by_type(area_type: str):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == area_type:
                area.tag_redraw()

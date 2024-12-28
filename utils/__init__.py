# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy # noqa
import sys
import subprocess
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

resolutions = (('256', '256', ''), ('512', '512', ''), ('1024', '1024', ''), ('2048', '2048', ''), ('4096', '4096', ''), ('8192', '8192', ''))
resolution_name_to_value = {'256': 256, '512': 512, '1K': 1024, '2K': 2048, '4K': 4096, '8K': 8192}
resolution_value_to_name = {256: '256', 512: '512', 1024: '1K', 2048: '2K', 4096: '4K', 8192: '8K'}

class NoInit:
    def __getattribute__(self, item):
        raise AttributeError(f'Object not initialized')

    def __bool__(self):
        raise AttributeError(f'Object not initialized')

    def __len__(self):
        raise AttributeError(f'Object not initialized')


class Pip:
    """
    The code was taken and modified from: https://github.com/s-leger/blender-pip/blob/master/blender_pip.py
    Necessary to convert svg to png, used during addon development.
    """
    def __init__(self):
        self._ensure_user_site_package()
        self._ensure_pip()

    @staticmethod
    def _ensure_user_site_package():
        import os
        import site
        site_package = site.getusersitepackages()
        if not os.path.exists(site_package):
            site_package = bpy.utils.user_resource('SCRIPTS', path="site_package", create=True)
            site.addsitedir(site_package)
        if site_package not in sys.path:
            sys.path.append(site_package)

    def _cmd(self, action, options, module):
        if options is not None and "--user" in options:
            self._ensure_user_site_package()

        cmd = [sys.executable, "-m", "pip", action]

        if options is not None:
            cmd.extend(options.split(" "))

        cmd.append(module)
        return self._run(cmd)

    @staticmethod
    def _popen(cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        popen.wait()

    def _run(self, cmd):
        res = False
        status = ""
        for line in self._popen(cmd):
            if "ERROR:" in line:
                status = line.strip()
            if "Error:" in line:
                status = line.strip()
            print(line)
            if "Successfully" in line:
                status = line.strip()
                res = True
        return res, status

    def _ensure_pip(self):
        pip_not_found = False
        try:
            import pip
        except ImportError:
            pip_not_found = True
            pass
        if pip_not_found:
            self._run([sys.executable, "-m", "ensurepip", "--default-pip"])

    @staticmethod
    def upgrade_pip():
        return Pip()._cmd("install", "--upgrade", "pip")

    @staticmethod
    def upgrade(module):
        return Pip()._cmd("install", "--upgrade", module)

    @staticmethod
    def uninstall(module, options=None):
        if options is None or options.strip() == "":
            # force confirm
            options = "-y"
        return Pip()._cmd("uninstall", options, module)

    @staticmethod
    def install(module):
        return Pip()._cmd('install', '-U', module)


class OverlapHelper:
    lock_overlap: bpy.props.BoolProperty(name='Lock Overlaps', default=False)
    lock_overlap_mode: bpy.props.EnumProperty(name='Lock Overlaps Mode', default='ANY', items=(('ANY', 'Any', ''), ('EXACT', 'Exact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.001, min=0.0, soft_min=0.00005, soft_max=0.00999)

    def draw_overlap(self, toggle=True):
        layout = self.layout  # noqa
        if self.lock_overlap:
            if self.lock_overlap_mode == 'EXACT':
                layout.prop(self, 'threshold', slider=True)
            layout.row().prop(self, 'lock_overlap_mode', expand=True)
        layout.prop(self, 'lock_overlap', toggle=toggle)

    def calc_overlapped_island_groups(self, all_islands):
        assert self.lock_overlap, 'Enable Lock Overlap option'
        threshold = None if self.lock_overlap_mode == 'ANY' else self.threshold
        return types.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

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

def get_mouse_pos(context, event):
    return Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))

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

def get_view3d_camera_data(v3d: bpy.types.SpaceView3D, rv3d: bpy.types.RegionView3D):
    #  establish the camera object,
    #  so we can default to view mapping if anything is wrong with it
    if rv3d.view_perspective == 'CAMERA' and v3d.camera and v3d.camera.type == 'CAMERA':
        return v3d.camera.data
    return None

def calc_any_unique_obj() -> list[bpy.types.Object]:
    # Get unique umeshes without uv
    objects = []
    if bpy.context.mode == 'EDIT_MESH':
        for obj in bpy.context.objects_in_mode_unique_data:
            if obj.type == 'MESH':
                objects.append(obj)
    else:
        from collections import defaultdict
        data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                data_and_objects[obj.data].append(obj)

        for data, objs in data_and_objects.items():
            objs.sort(key=lambda a: a.name)
            objects.append(objs[0])
    return objects

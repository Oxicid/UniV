# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import sys
import subprocess
import typing  # noqa

import numpy as np  # noqa
from math import pi

from .bench import timer, profile
from .draw import *
from .other import *
from .shapes import *
from .ubm import *
from .bm_select import *
from .bm_tag import *
from .bm_iterable import *
from .bm_walk import *
from .umath import *
from . import uv_parametrizer
from .. import utypes

resolutions = (('256', '256', ''), ('512', '512', ''), ('1024', '1024', ''),
               ('2048', '2048', ''), ('4096', '4096', ''), ('8192', '8192', ''))
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
    lock_overlap_mode: bpy.props.EnumProperty(
        name='Lock Overlaps Mode', default='ANY', items=(('ANY', 'Any', ''), ('EXACT', 'Exact', '')))
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
        return utypes.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)


class PaddingHelper:
    padding_multiplayer: bpy.props.FloatProperty(
        name='Padding Multiplayer', default=1, min=-32, soft_min=0, soft_max=4, max=32)

    def __init__(self):
        self.padding = 0.0

    def draw_padding(self):
        layout = self.layout  # noqa
        if self.padding_multiplayer:
            from .. import preferences
            settings = preferences.univ_settings()
            layout.separator(factor=0.35)
            layout.label(text=f"Global Texture Size = {min(int(settings.size_x), int(settings.size_y))}")
            layout.label(text=f"Padding = {settings.padding}({int(settings.padding * self.padding_multiplayer)})px")

        layout.prop(self, "padding_multiplayer", slider=True)

    def calc_padding(self):
        from .. import preferences
        settings = preferences.univ_settings()
        self.padding = int(settings.padding * self.padding_multiplayer) / \
            min(int(settings.size_x), int(settings.size_y))

    def report_padding(self):
        if self.padding and (img_size := get_active_image_size()):  # TODO: Get active image size from material id
            from .. import preferences
            settings = preferences.univ_settings()
            if min(int(settings.size_x), int(settings.size_y)) != min(img_size):
                self.report({'WARNING'}, 'Global and Active texture sizes have different values, '  # noqa
                                         'which will result in incorrect padding.')


class ViewBoxSyncBlock:
    def __init__(self, bbox):
        from ..utypes import BBox
        self.view_box: BBox = bbox
        self.has_blocked = False
        self.skip = True

    @classmethod
    def from_area(cls, area):
        if area and area.ui_type == 'UV' and not USE_GENERIC_UV_SYNC:
            reg = area.regions[-1]
            if reg.type == 'WINDOW':
                from ..utypes import BBox
                n_panel_width = next(r.width for r in area.regions if r.type == 'UI')
                tools_width = next(r.width for r in area.regions if r.type == 'TOOLS')

                min_v = Vector(reg.view2d.region_to_view(tools_width, 0))
                max_v = Vector(reg.view2d.region_to_view(reg.width - n_panel_width, reg.height))

                view_rect = BBox.init_from_minmax(min_v, max_v)
                view_rect.scale(0.6)
                return cls(view_rect)

        view_box = cls(None)
        view_box.skip = True
        return view_box

    def draw_if_blocked(self):
        if self.has_blocked:
            update_area_by_type('IMAGE_EDITOR')
            from ..draw import LinesDrawSimple
            LinesDrawSimple.draw_register(self.view_box.draw_data_lines(), (.1,1,1,0.5))

    def skip_from_param(self, umesh: 'utypes.UMesh', select: bool):
        if not USE_GENERIC_UV_SYNC and select and umesh.elem_mode in ('VERT', 'EDGE') and not umesh.sync_valid:
            self.skip = False
        else:
            self.skip = True

    def filter_by_isect_islands(self, islands):
        if not self.skip:
            islands.islands = [isl for isl in islands if self.isect_island(isl)]

    @staticmethod
    def _isl_has_inner_elem(island: 'utypes.FaceIsland | utypes.AdvIsland'):
        uv = island.umesh.uv

        if island.umesh.elem_mode == 'VERT':
            for crn in island.corners_iter():
                if not crn.vert.select:
                    continue
                uv_co = crn[uv].uv
                for l_crn in crn.vert.link_loops:
                    if l_crn.face.hide:
                        continue
                    if l_crn[uv].uv != uv_co:
                        break
                else:
                    return True
        else:
            for crn in island.corners_iter():
                if not crn.edge.select:
                    continue
                pair = crn.link_loop_radial_prev
                if pair == crn or pair.face.hide:
                    # If there is no pair crn, or it is hidden, then this edge is chosen deliberately.
                    return True

                if crn.link_loop_next[uv].uv == pair[uv].uv or \
                    crn[uv].uv == pair.link_loop_next[uv].uv:
                    return True


    def isect_island(self, island):
        """ NOTE: For the intersection check (isect) to work correctly,
        all islands must be sorted by their intersection data first, before applying any transformations."""
        from ..utypes import BBox, FaceIsland, AdvIsland
        if self.skip:
            return True

        view: BBox = self.view_box
        if not island.is_full_face_selected():
            selected_faces = [f for f in island if f.select]
            island = AdvIsland(selected_faces, island.umesh)

        isl_bbox: BBox = island.calc_bbox()

        if not view.isect(isl_bbox):
            if self._isl_has_inner_elem(island):
                return True

            self.has_blocked = True
            return False
        if view.isect_x(isl_bbox.xmin) and view.isect_x(isl_bbox.xmax):
            return True
        if view.isect_y(isl_bbox.ymin) and view.isect_y(isl_bbox.ymax):
            return True

        if type(island) == FaceIsland:
            island = AdvIsland(island.faces, island.umesh)

        if not island.flat_coords:
            island.calc_tris_simple()
            island.calc_flat_coords(save_triplet=True)

        if view.isect_triangles(island.flat_coords):
            return True

        if self._isl_has_inner_elem(island):
            return True
        self.has_blocked = True
        return False

    @staticmethod
    def _lg_has_inner_elem(lg: 'utypes.LoopGroup'):
        uv = lg.umesh.uv

        if lg.umesh.elem_mode == 'VERT':
            for crn in lg:
                assert crn.vert.select
                uv_co = crn[uv].uv
                for l_crn in crn.vert.link_loops:
                    if l_crn.face.hide:
                        continue
                    if l_crn[uv].uv != uv_co:
                        break
                else:
                    return True
        else:
            for crn in lg:
                if not crn.edge.select:
                    continue
                pair = crn.link_loop_radial_prev
                if pair == crn or pair.face.hide:
                    # If there is no pair crn, or it is hidden, then this edge is chosen deliberately.
                    return True

                if crn.link_loop_next[uv].uv == pair[uv].uv or \
                    crn[uv].uv == pair.link_loop_next[uv].uv:
                    return True

    def isect_lg(self, lg):
        from ..utypes import BBox

        if self.skip:
            return True

        view: BBox = self.view_box
        lg_bbox: BBox = lg.calc_bbox()

        if not view.isect(lg_bbox):
            if self._lg_has_inner_elem(lg):
                return True
            self.has_blocked = True
            return False
        elif view.isect_x(lg_bbox.xmin) and view.isect_x(lg_bbox.xmax):
            return True
        elif view.isect_y(lg_bbox.ymin) and view.isect_y(lg_bbox.ymax):
            return True

        if self._lg_has_inner_elem(lg):
            return True

        self.has_blocked = True
        return False

    def has_inner_selection(self, island):
        if self.skip:
            return True
        if not (island.umesh.elem_mode in ('VERT', 'EDGE')):
            return True

        assert island.umesh.sync

        uv = island.umesh.uv
        if island.umesh.elem_mode == 'VERT':
            def vert_has_unpair_select(crn_: BMLoop):
                if not crn_.vert.select:
                    return False
                # If the number of all visible faces and the number of linked corners are the same,
                # then they belong to the same island, and this is not a random selection.
                count_visible_faces = sum(not f_.hide for f_ in crn_.vert.link_faces)
                count_linked_corners = len(linked_crn_to_vert_pair_with_seam(crn_, uv, True)) + 1
                return count_visible_faces == count_linked_corners

            for f in island:
                for crn in f.loops:
                    if vert_has_unpair_select(crn):
                        return True
        else:
            def edge_has_unpair_select(crn_: BMLoop):
                if not crn_.edge.select:
                    return False
                if crn_.edge.is_boundary:
                    return True
                if crn_.edge.seam:
                    return False
                pair = crn_.link_loop_radial_prev
                if pair.face.hide:
                    return True
                return is_pair(crn_, pair, uv)

            for f in island:
                for crn in f.loops:
                    if edge_has_unpair_select(crn):
                        return True
        return False

    def filter_verts(self, corners, uv):
        if self.skip:
            return corners

        xmin = self.view_box.xmin
        xmax = self.view_box.xmax
        ymin = self.view_box.ymin
        ymax = self.view_box.ymax

        filtered_corners = set()
        for crn in corners:
            if crn in filtered_corners:
                continue

            x, y = crn[uv].uv
            if xmin <= x <= xmax and ymin <= y <= ymax:
                filtered_corners.add(crn)
                filtered_corners.update(linked_crn_to_vert_pair_iter(crn, uv, True))
                continue

            # Add outside unpair vertices
            linked = []
            uv_co = crn[uv].uv
            for l_crn in crn.vert.link_loops:
                if l_crn.face.hide:
                    continue
                if uv_co != l_crn[uv].uv:
                    break
                linked.append(crn)
            else:
                filtered_corners.update(linked)

        assert len(filtered_corners) <= len(corners)
        if len(filtered_corners) < len(corners):
            self.has_blocked = True
        return filtered_corners

    def filter_edges(self, corners: list[BMLoop], umesh):
        if self.skip:
            return corners

        uv = umesh.uv
        xmin = self.view_box.xmin
        xmax = self.view_box.xmax
        ymin = self.view_box.ymin
        ymax = self.view_box.ymax
        l1_a, l1_b, l2_a, l2_b, l3_a, l3_b, l4_a, l4_b = self.view_box.draw_data_lines()
        from mathutils.geometry import intersect_line_line_2d

        is_invisible = is_invisible_func(umesh)
        is_boundary = is_boundary_func(umesh, with_seam=False)
        filtered_corners = set()
        for crn in corners:
            if crn in filtered_corners:
                continue

            if not is_boundary(crn):
                filtered_corners.add(crn)
                filtered_corners.add(crn.link_loop_radial_prev)
                continue
            elif crn.edge.is_boundary or is_invisible(crn.link_loop_radial_prev.face):
                filtered_corners.add(crn)
                continue

            pt_1 = crn[uv].uv
            pt_2 = crn.link_loop_next[uv].uv
            if (xmin <= pt_1.x <= xmax and ymin <= pt_1.y <= ymax or
                xmin <= pt_2.x <= xmax and ymin <= pt_2.y <= ymax):
                filtered_corners.add(crn)
                continue

            if (intersect_line_line_2d(pt_1, pt_2, l1_a, l1_b) or
                intersect_line_line_2d(pt_1, pt_2, l2_a, l2_b) or
                intersect_line_line_2d(pt_1, pt_2, l3_a, l3_b) or
                intersect_line_line_2d(pt_1, pt_2, l4_a, l4_b)
            ):
                filtered_corners.add(crn)

        assert len(filtered_corners) <= len(corners)
        if len(filtered_corners) < len(corners):
            self.has_blocked = True
        return filtered_corners


    def flush_if_blocked(self):
        if self.has_blocked:
            from ..draw import LinesDrawSimple
            LinesDrawSimple.draw_register(self.view_box.draw_data_lines(), (0.05, 0.2, 0.9, 0.1))

    def __str__(self):
        return f"View Box={self.view_box}, Skip={self.skip}, Has Blocked={self.has_blocked}"


def set_global_texel(isl: 'utypes.AdvIsland', calc_bbox=True):
    from ..preferences import univ_settings
    if not univ_settings().use_texel:
        return False

    if isl.area_3d == -1.0:
        if isinstance(isl.umesh.value, Vector):
            isl.calc_area_3d(isl.umesh.value)
        else:
            isl.calc_area_3d()

    if isl.area_uv == -1.0:
        isl.calc_area_uv()

    if calc_bbox:
        isl.calc_bbox()

    # TODO: Double scale for small area island
    texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
    res = isl.set_texel(univ_settings().texel_density, texture_size)
    return bool(res)


def sync():
    return bpy.context.scene.tool_settings.use_uv_select_sync


def calc_avg_normal():
    umeshes = utypes.UMeshes.sel_ob_with_uv()
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


def get_cursor_location() -> Vector | None:
    if bpy.context.area.ui_type == 'UV':
        return bpy.context.space_data.cursor_location.copy()
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.ui_type == 'UV':
                return area.spaces.active.cursor_location.copy()


def get_mouse_pos(context, event):
    return Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))


def get_tile_from_cursor() -> Vector | None:
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
        data_and_objects: defaultdict[bpy.types.Mesh, list[bpy.types.Object]] = defaultdict(list)

        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                data_and_objects[obj.data].append(obj)

        for data, objs in data_and_objects.items():
            objs.sort(key=lambda a: a.name)
            objects.append(objs[0])
    return objects


UNIT_CONVERTION = {
    'mm': (0.001, 1000),
    'cm': (0.01, 100),
    'm': (1, 1),
    'km': (1000, 0.001),
    'in': (0.0254, 39.3701),
    'ft': (0.3048, 3.28084),
    'yd': (0.9144, 1.09361),
    'mi': (1609.34, 0.000621371),
}

UNITS = '(mi|mm|cm|m|km|in|ft|yd)'
UNITS_T = typing.Literal['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi',]


def unit_conversion(value: float, from_type: UNITS_T, to_type: UNITS_T) -> float:
    return value * UNIT_CONVERTION[from_type.lower()][0] * UNIT_CONVERTION[to_type.lower()][1]

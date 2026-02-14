# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import re
import bpy  # noqa: F401
import gpu
import json
import math
import bl_math
import numpy as np
import itertools

from bpy.props import *
from pathlib import Path
from bpy.types import Operator
from collections.abc import Callable

from math import pi, sqrt, isclose
from bl_math import clamp
from mathutils import Vector, Matrix

from .. import utils
from .. import utypes
from ..draw import shaders
from ..utypes import (
    UMeshes,
    AdvIslands,
    AdvIsland,
    UnionIslands
)
from ..preferences import prefs, univ_settings


class UNIV_OT_ResetScale(Operator, utils.OverlapHelper):
    bl_idname = "uv.univ_reset_scale"
    bl_label = 'Reset'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Reset the scale of separate UV islands, based on their area in 3D space\n\n" \
        f"Default - Reset islands scale\n" \
        f"Shift - Lock Overlaps"

    shear: BoolProperty(name='Shear', default=True, description='Reduce shear within islands')
    axis: EnumProperty(name='Axis', default='XY', items=(('XY', 'Both', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        self.draw_overlap()
        layout.row(align=True).prop(self, 'axis', expand=True)
        layout.prop(self, 'shear')
        layout.prop(self, 'use_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        if not self.bl_idname.startswith('UV'):
            self.umeshes.set_sync()
            self.umeshes.sync_invalidate()

        all_islands: list[AdvIsland | UnionIslands] = []

        islands_calc_type: Callable[[utypes.UMesh], AdvIslands]
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
            islands_calc_type = AdvIslands.calc_extended_with_mark_seam if selected_umeshes else AdvIslands.calc_visible_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_with_hidden
            for umesh in self.umeshes:
                umesh.ensure(face=True)

        if self.use_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=not self.bl_idname.startswith('UV'))

        for umesh in self.umeshes:
            adv_islands = islands_calc_type(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'
            all_islands.extend(adv_islands)
            adv_islands.calc_tris_simple()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

        if not all_islands:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        if self.lock_overlap:
            all_islands = self.calc_overlapped_island_groups(all_islands)

        for isl in all_islands:
            isl.value = isl.bbox.center  # isl.value == pivot
            # TODO: Find how to calculate the shear for the X axis when aspect != 1 without rotation island
            if self.axis == 'X' and isl.umesh.aspect != 1.0 and self.shear:
                isl.rotate_simple(pi/2, isl.umesh.aspect)
                self.individual_scale(isl, 'Y',  self.shear)
                isl.rotate_simple(-pi/2, isl.umesh.aspect)
                new_center = isl.calc_bbox().center
            else:
                new_center = self.individual_scale(isl, self.axis, self.shear)
            isl.set_position(isl.value, new_center)

        self.umeshes.update(info='All islands were with scaled')

        if not self.umeshes.is_edit_mode:
            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}

    @staticmethod
    def individual_scale(isl: AdvIsland, axis, shear, threshold=1e-8):
        # TODO: The threshold can be made lower if the triangulation (tessellation) is performed using the UV topology.
        from bl_math import clamp
        aspect = isl.umesh.aspect
        clamp_value = aspect * 0.5
        if aspect > 1.0:
            clamp_value = (1 / aspect) * 0.5

        new_center = isl.value.copy()

        transform_acc = Matrix.Identity(2)
        scale_acc = Vector((1.0, 1.0))

        flat_3d_coords = np.array([(pt_a.to_tuple(), pt_b.to_tuple(), pt_c.to_tuple())
                                  for pt_a, pt_b, pt_c in isl.flat_3d_coords], dtype=np.float32)
        vec_ac = flat_3d_coords[:, 0] - flat_3d_coords[:, 2]
        vec_bc = flat_3d_coords[:, 1] - flat_3d_coords[:, 2]

        flat_uv_coords = np.array([(pt_a.to_tuple(), pt_b.to_tuple(), pt_c.to_tuple())
                                  for pt_a, pt_b, pt_c in isl.flat_coords], dtype=np.float32)
        weights = np.array(list(isl.weights) if isinstance(
            isl.weights, itertools.chain) else isl.weights, dtype=np.float32)

        prev_err = float('inf')
        for _ in range(10):
            m00 = flat_uv_coords[:, 0, 0] - flat_uv_coords[:, 2, 0]
            m01 = flat_uv_coords[:, 0, 1] - flat_uv_coords[:, 2, 1]
            m10 = flat_uv_coords[:, 1, 0] - flat_uv_coords[:, 2, 0]
            m11 = flat_uv_coords[:, 1, 1] - flat_uv_coords[:, 2, 1]

            det = m00 * m11 - m01 * m10
            mask = np.abs(det) > threshold

            with np.errstate(divide='ignore', invalid='ignore'):
                inv00, inv01 = m11 / det, -m01 / det
                inv10, inv11 = -m10 / det, m00 / det

                cou = inv00[:, None] * vec_ac + inv01[:, None] * vec_bc
                cov = inv10[:, None] * vec_ac + inv11[:, None] * vec_bc

            w = weights
            if not np.all(mask):
                if not np.any(mask):
                    break
                cou = cou[mask]
                cov = cov[mask]
                w = weights[mask]

            scale_cou = np.sum(utils.np_vec_normalized(cou, keepdims=False) * w)
            scale_cov = np.sum(utils.np_vec_normalized(cov, keepdims=False) * w)
            scale_cross = 0.0
            if shear:
                cou_n = cou / utils.np_vec_normalized(cou)
                cov_n = cov / utils.np_vec_normalized(cov)
                scale_cross = np.sum(utils.np_vec_dot(cou_n, cov_n) * w)

            if scale_cou * scale_cov < 1e-10:
                break

            scale_factor_u = sqrt(scale_cou / scale_cov / aspect)
            if axis != 'XY':
                scale_factor_u **= 2

            # Trade accuracy for performance and for avoid stretches when aspect != 1.0.
            if aspect == 1.0:
                tolerance = 1e-10
            elif aspect > 1.0:
                tolerance = 0.005 * aspect
            else:
                tolerance = 0.0005 * (1 / aspect)

            if shear:
                t = Matrix.Identity(2)
                t[0][0] = scale_factor_u
                t[1][0] = clamp((scale_cross / isl.area_3d) * aspect, -clamp_value, clamp_value)
                t[0][1] = 0
                t[1][1] = 1 / scale_factor_u

                if axis == 'X':
                    t[1][1] = 1
                    temp = t[0][1]
                    t[0][1] = t[1][0]
                    t[1][0] = temp

                elif axis == 'Y':
                    t[0][0] = 1

                err = abs(t[0][0] - 1.0) + abs(t[1][0]) + abs(t[0][1]) + abs(t[1][1] - 1.0)
                if err < tolerance or prev_err < err:
                    break
                prev_err = err

                # Transform
                transform_acc @= t
                flat_uv_coords = flat_uv_coords @ np.array(t, dtype=np.float32)
            else:
                if math.isclose(scale_factor_u, 1.0, abs_tol=tolerance):
                    break
                scale = Vector((scale_factor_u, 1.0/scale_factor_u))
                if axis == 'X':
                    scale.y = 1
                elif axis == 'Y':
                    scale.x = 1

                scale_acc *= scale
                flat_uv_coords *= np.array(scale, dtype=np.float32)

        if shear:
            if transform_acc != Matrix.Identity(2):
                isl.umesh.update_tag = True
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord.xy = uv_coord @ transform_acc
                new_center = new_center @ transform_acc
        else:
            if scale_acc != Vector((1.0, 1.0)):
                isl.umesh.update_tag = True
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord *= scale_acc
                new_center *= scale_acc
        return new_center


class UNIV_OT_ResetScale_VIEW3D(UNIV_OT_ResetScale):
    bl_idname = "mesh.univ_reset_scale"


class UNIV_OT_Normalize_VIEW3D(Operator, utils.OverlapHelper):
    bl_idname = "mesh.univ_normalize"
    bl_label = 'Normalize'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Average the size of separate UV islands, based on their area in 3D space\n\n" \
        f"Default - Average Islands Scale\n" \
        f"Shift - Lock Overlaps"

    shear: BoolProperty(name='Shear', default=False, description='Reduce shear within islands')
    xy_scale: BoolProperty(name='Scale Independently', default=True, description='Scale U and V independently')
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        self.draw_overlap()
        layout.prop(self, 'shear')
        layout.prop(self, 'xy_scale')
        layout.prop(self, 'use_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        is_uv_area = context.area.ui_type == 'UV'
        if not is_uv_area:
            self.umeshes.set_sync(True)

        if self.use_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=not is_uv_area)

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        all_islands: list[AdvIsland | UnionIslands] = []

        islands_calc_type: Callable[[utypes.UMesh], AdvIslands]
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
            # TODO: AdvIslands with FLIPPED_3D
            islands_calc_type = AdvIslands.calc_extended_with_mark_seam if selected_umeshes else AdvIslands.calc_visible_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_with_hidden
            for umesh in self.umeshes:
                umesh.ensure(face=True)

        for umesh in self.umeshes:
            adv_islands = islands_calc_type(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'
            all_islands.extend(adv_islands)
            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

        if not all_islands:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        if self.lock_overlap:
            all_islands = self.calc_overlapped_island_groups(all_islands)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                isl.value = self.individual_scale(isl)

        tot_area_uv, tot_area_3d = self.avg_by_frequencies(all_islands)
        self.normalize(all_islands, tot_area_uv, tot_area_3d)

        self.umeshes.update(info='All islands were normalized')

        if not self.umeshes.is_edit_mode:
            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}

    def individual_scale(self, isl: AdvIsland, threshold=1e-8):
        if not self.shear and not self.xy_scale:
            return isl.value

        if isinstance(isl.value, Vector):
            new_center = isl.value.copy()
        else:
            new_center = Vector((1, 1))

        aspect = isl.umesh.aspect
        transform_acc = Matrix.Identity(2)
        scale_acc = Vector((1.0, 1.0))

        flat_3d_coords = np.array([(pt_a.to_tuple(), pt_b.to_tuple(), pt_c.to_tuple())
                                  for pt_a, pt_b, pt_c in isl.flat_3d_coords], dtype=np.float32)
        vec_ac = flat_3d_coords[:, 0] - flat_3d_coords[:, 2]
        vec_bc = flat_3d_coords[:, 1] - flat_3d_coords[:, 2]
        flat_uv_coords = np.array([(pt_a.to_tuple(), pt_b.to_tuple(), pt_c.to_tuple())
                                  for pt_a, pt_b, pt_c in isl.flat_coords], dtype=np.float32)
        weights = np.array(list(isl.weights) if isinstance(
            isl.weights, itertools.chain) else isl.weights, dtype=np.float32)

        for _ in range(10):
            m00 = flat_uv_coords[:, 0, 0] - flat_uv_coords[:, 2, 0]
            m01 = flat_uv_coords[:, 0, 1] - flat_uv_coords[:, 2, 1]
            m10 = flat_uv_coords[:, 1, 0] - flat_uv_coords[:, 2, 0]
            m11 = flat_uv_coords[:, 1, 1] - flat_uv_coords[:, 2, 1]

            det = m00 * m11 - m01 * m10
            mask = np.abs(det) > threshold

            with np.errstate(divide='ignore', invalid='ignore'):
                inv00, inv01 = m11 / det, -m01 / det
                inv10, inv11 = -m10 / det, m00 / det

                cou = inv00[:, None] * vec_ac + inv01[:, None] * vec_bc
                cov = inv10[:, None] * vec_ac + inv11[:, None] * vec_bc

            w = weights
            if not np.all(mask):
                if not np.any(mask):
                    break
                cou = cou[mask]
                cov = cov[mask]
                w = weights[mask]

            scale_cou = np.sum(utils.np_vec_normalized(cou, keepdims=False) * w)
            scale_cov = np.sum(utils.np_vec_normalized(cov, keepdims=False) * w)
            scale_cross = 0.0
            if self.shear:
                cou_n = cou / utils.np_vec_normalized(cou)
                cov_n = cov / utils.np_vec_normalized(cov)
                scale_cross = np.sum(utils.np_vec_dot(cou_n, cov_n) * w)

            if scale_cou * scale_cov < 1e-10:
                break

            scale_factor_u = sqrt(scale_cou / scale_cov / aspect) if self.xy_scale else 1.0

            tolerance = 1e-5  # Trade accuracy for performance.
            if self.shear:
                t = Matrix.Identity(2)
                t[0][0] = scale_factor_u
                t[1][0] = clamp((scale_cross / isl.area_3d) * aspect, -0.5 * aspect, 0.5 * aspect)
                t[0][1] = 0
                t[1][1] = 1 / scale_factor_u

                err = abs(t[0][0] - 1.0) + abs(t[1][0]) + abs(t[0][1]) + abs(t[1][1] - 1.0)
                if err < tolerance:
                    break

                # Transform
                transform_acc @= t
                flat_uv_coords = flat_uv_coords @ np.array(t, dtype=np.float32)
            else:
                if math.isclose(scale_factor_u, 1.0, abs_tol=tolerance):
                    break
                scale = Vector((scale_factor_u, 1.0/scale_factor_u))
                scale_acc *= scale
                flat_uv_coords *= np.array(scale, dtype=np.float32)

        if self.shear:
            if transform_acc != Matrix.Identity(2):
                isl.umesh.update_tag = True
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord.xy = uv_coord @ transform_acc
                new_center = new_center @ transform_acc
        else:
            if scale_acc != Vector((1.0, 1.0)):
                isl.umesh.update_tag = True
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord *= scale_acc
                new_center *= scale_acc
        return new_center

    def normalize(self, islands: list[AdvIsland], tot_area_uv, tot_area_3d):
        if not self.xy_scale and len(islands) <= 1:
            self.umeshes.cancel_with_report(
                {'WARNING'}, info=f"Islands should be more than 1, given {len(islands)} islands")
            return
        if tot_area_3d == 0.0 or tot_area_uv == 0.0:
            # Prevent divide by zero.
            self.umeshes.cancel_with_report(
                {'WARNING'}, info=f"Cannot normalize islands, total {'UV-area' if tot_area_3d else '3D-area'} of faces is zero")
            return

        tot_fac = tot_area_3d / tot_area_uv

        zero_area_islands = []
        for isl in islands:
            if isclose(isl.area_3d, 0.0, abs_tol=1e-6) or isclose(isl.area_uv, 0.0, abs_tol=1e-6):
                zero_area_islands.append(isl)
                continue

            fac = isl.area_3d / isl.area_uv
            scale = math.sqrt(fac / tot_fac)

            if self.xy_scale or self.shear:
                old_pivot = isl.bbox.center
                new_pivot = isl.value
                new_pivot_with_scale = new_pivot * scale

                diff1 = old_pivot - new_pivot
                diff = (new_pivot - new_pivot_with_scale) + diff1

                if utils.vec_isclose(old_pivot, new_pivot) and math.isclose(scale, 1.0, abs_tol=0.00001):
                    continue

                for crn_co in isl.flat_unique_uv_coords:
                    crn_co *= scale
                    crn_co += diff

                isl.umesh.update_tag = True
            else:
                if math.isclose(scale, 1.0, abs_tol=0.00001):
                    continue
                if isl.scale(Vector((scale, scale)), pivot=isl.calc_bbox().center):
                    isl.umesh.update_tag = True

        if zero_area_islands:
            need_validation = False
            if utils.USE_GENERIC_UV_SYNC:
                if utils.sync() and utils.get_select_mode_mesh() in ('VERT', 'EDGE'):
                    need_validation = True

            for isl in islands:
                if isl not in zero_area_islands:
                    isl.select = False
                    isl.umesh.update_tag = True
            for isl in zero_area_islands:
                if need_validation:
                    isl.umesh.sync_from_mesh_if_needed()
                isl.select = True
                isl.umesh.update_tag = True

            self.report({'WARNING'}, f"Found {len(zero_area_islands)} islands with zero area")

    def avg_by_frequencies(self, all_islands: list[AdvIsland]):
        areas_uv = np.empty(len(all_islands), dtype=float)
        areas_3d = np.empty(len(all_islands), dtype=float)

        for idx, isl in enumerate(all_islands):
            areas_uv[idx] = isl.calc_area_uv()
            areas_3d[idx] = isl.area_3d

        areas = areas_uv if self.bl_idname.startswith('UV') else areas_3d
        median: float = np.median(areas)  # noqa
        min_area = np.amin(areas)
        max_area = np.amax(areas)

        center = (min_area + max_area) / 2
        if center > median:
            diff = bl_math.lerp(median, max_area, 0.15) - median
        else:
            diff = median - bl_math.lerp(median, min_area, 0.15)

        min_clamp = median - diff
        max_clamp = median + diff

        indexes = (areas >= min_clamp) & (areas <= max_clamp)
        total_uv_area = np.sum(areas_uv, where=indexes)
        total_3d_area = np.sum(areas_3d, where=indexes)

        # TODO: Averaging by area_3d to area_uv ratio (by frequency of occurrence of the same values)
        if total_uv_area and total_3d_area:
            return total_uv_area, total_3d_area
        else:
            idx_for_find = math.nextafter(median, max_area)
            idx = UNIV_OT_Normalize_VIEW3D.np_find_nearest(areas, idx_for_find)
            total_uv_area = areas_uv[idx]
            total_3d_area = areas_3d[idx]
            if total_uv_area and total_3d_area:
                return total_uv_area, total_3d_area
            else:
                return np.sum(areas_uv), np.sum(areas_3d)

    @staticmethod
    def np_find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


class UNIV_OT_Normalize(UNIV_OT_Normalize_VIEW3D):
    bl_idname = "uv.univ_normalize"


class UNIV_OT_AdjustScale_VIEW3D(UNIV_OT_Normalize_VIEW3D):
    bl_idname = "mesh.univ_adjust_td"
    bl_label = 'Adjust'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Average the size of separate UV islands from unselected islands or objects, based on their area in 3D space\n\n" \
                     "Default - Average Islands Scale\n" \
                     "Shift - Lock Overlaps"

    def invoke(self, context, event):
        if self.bl_idname.startswith('UV'):
            self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
            self.mouse_pos = utils.get_mouse_pos(context, event)
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None

    def execute(self, context):
        if context.mode == 'EDIT_MESH':
            return self.adjust_edit()
        return self.adjust_object()

    def pick_adjust_edit(self):
        all_islands = []
        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale
            all_islands.extend(adv_islands)

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        for isl in all_islands:
            hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            self.report({'INFO'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        all_islands.remove(hit.island)

        tot_area_uv = tot_area_3d = 0
        for isl in all_islands:
            tot_area_uv += isl.area_uv
            tot_area_3d += isl.area_3d
        all_islands = [hit.island]

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                isl.value = self.individual_scale(isl)

        self.normalize_and_show_adjust_result_info_edit(
            all_islands, tot_area_3d, tot_area_uv, sel='picked', unsel='unpicked')
        return {'FINISHED'}

    def adjust_edit(self):
        all_islands: list[AdvIsland | UnionIslands] = []
        self.umeshes = UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()
            self.umeshes.sync_invalidate()

        if self.use_aspect:
            # TODO: Implement exact aspect for materials (get aspect by face mat id)
            self.umeshes.calc_aspect_ratio(from_mesh=not self.bl_idname.startswith('UV'))

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        if self.max_distance and not selected_umeshes and unselected_umeshes:
            self.umeshes = unselected_umeshes
            return self.pick_adjust_edit()

        self.umeshes = selected_umeshes

        if not self.umeshes:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        tot_area_uv = tot_area_3d = 0
        for umesh in self.umeshes:
            adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

            for isl in adv_islands:
                any_selected = AdvIslands.island_filter_is_any_face_selected(isl, umesh)
                if any_selected:
                    all_islands.append(isl)
                else:
                    tot_area_uv += isl.area_uv
                    tot_area_3d += isl.area_3d

        for umesh in unselected_umeshes:
            if not (faces := utils.calc_visible_uv_faces(umesh)):
                continue
            adv_islands = AdvIsland(faces, umesh)
            tot_area_uv += adv_islands.calc_area_uv()
            tot_area_3d += adv_islands.calc_area_3d(scale=umesh.value)

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                isl.value = self.individual_scale(isl)

        self.normalize_and_show_adjust_result_info_edit(all_islands, tot_area_3d, tot_area_uv)
        return {'FINISHED'}

    def normalize_and_show_adjust_result_info_edit(self, all_islands, tot_area_3d, tot_area_uv, sel='selected', unsel='unselected'):
        info_ = 'All target islands were normalized'
        if isinstance(tot_area_uv, int):
            ret = self.umeshes.update(info=info_)
            if ret == {'FINISHED'}:
                for isl in all_islands:
                    isl.set_position(isl.value, isl.calc_bbox().center)
                self.report({'INFO'}, f'{unsel.capitalize()} islands not found, but {sel} was adjusted')
            else:
                self.report({'WARNING'}, f'{unsel.capitalize()} islands not found')
            return ret

        self.normalize(all_islands, tot_area_uv, tot_area_3d)
        self.umeshes.update(info=info_)

    def adjust_object(self):
        all_islands: list[AdvIsland | UnionIslands] = []
        self.umeshes = UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()
            self.umeshes.sync_invalidate()

        if self.use_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=not self.bl_idname.startswith('UV'))

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        for umesh in (unselected_umeshes := UMeshes.unselected_with_uv()):
            umesh.value = umesh.check_uniform_scale(report=self.report)
        unselected_umeshes.set_sync()
        unselected_umeshes.sync_invalidate()

        tot_area_uv = tot_area_3d = 0
        for umesh in self.umeshes:
            umesh.ensure()
            adv_islands = AdvIslands.calc_with_hidden(umesh)

            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale
            all_islands.extend(adv_islands)

        for umesh in unselected_umeshes:
            adv_islands = AdvIsland(umesh.bm.faces, umesh)  # noqa
            tot_area_uv += adv_islands.calc_area_uv()
            tot_area_3d += adv_islands.calc_area_3d(scale=umesh.value)
            umesh.free()

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                isl.value = self.individual_scale(isl)

        self.umeshes.report_obj = None
        if isinstance(tot_area_uv, int):
            if (ret := self.umeshes.update()) == {'FINISHED'}:
                for isl in all_islands:
                    isl.set_position(isl.value, isl.calc_bbox().center)
                self.report({'INFO'}, f'Unselected objects not found, but selected was adjusted')
            else:
                self.report({'WARNING'}, f"Unselected objects not found")

            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')
            return ret

        self.normalize(all_islands, tot_area_uv, tot_area_3d)
        self.umeshes.report_obj = self.report

        if self.umeshes.has_update_mesh:
            if not unselected_umeshes:
                self.umeshes.update(info=f'Unselected objects not found, but selected was adjusted')
            else:
                self.umeshes.update(info='All target islands were adjusted')
        else:
            self.report({'WARNING'}, f'Unselected objects not found.')

        self.umeshes.free()
        utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}


class UNIV_OT_AdjustScale(UNIV_OT_AdjustScale_VIEW3D):
    bl_idname = "uv.univ_adjust_td"


class UNIV_OT_TexelDensitySet_VIEW3D(Operator):
    bl_idname = "mesh.univ_texel_density_set"
    bl_label = 'Set'
    bl_description = "Set Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    grouping_type: EnumProperty(name='Grouping Type', default='NONE',
                                items=(('NONE', 'None', ''), ('OVERLAP', 'Overlap', ''), ('UNION', 'Union', '')))
    lock_overlap_mode: bpy.props.EnumProperty(name='Lock Overlaps Mode', default='ANY',
                                              items=(('ANY', 'Any', ''), ('EXACT', 'Exact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.001, min=0.0, soft_min=0.00005, soft_max=0.00999)
    td_preset_idx: IntProperty(name='TD Preset Index', default=-1, options={'HIDDEN'})

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        if event.shift:
            self.grouping_type = 'UNION' if event.alt else 'OVERLAP'
        else:
            self.grouping_type = 'NONE'
        return self.execute(context)

    def draw(self, context):
        layout = self.layout  # noqa
        if self.grouping_type == 'OVERLAP':
            if self.lock_overlap_mode == 'EXACT':
                layout.prop(self, 'threshold', slider=True)
            layout.row().prop(self, 'lock_overlap_mode', expand=True)
        layout.row(align=True).prop(self, 'grouping_type', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texel: float = 1.0
        self.texture_size: float = 2048.0
        self.has_selected = True
        self.islands_calc_type: Callable = Callable
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.texel = univ_settings().texel_density
        self.texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2

        if self.td_preset_idx != -1:
            if self.td_preset_idx+1 > len(univ_settings().texels_presets):
                self.report({'ERROR'}, 'Texel Density preset not found')
                return {'FINISHED'}

            td_preset = univ_settings().texels_presets[self.td_preset_idx]
            self.texel = td_preset.texel
            self.texture_size = (int(td_preset.size_x) + int(td_preset.size_y)) / 2

        self.umeshes = UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()
            self.umeshes.sync_invalidate()

        cancel = False
        if not self.umeshes.is_edit_mode:
            if not self.umeshes:
                cancel = True
            else:
                self.has_selected = False
                self.islands_calc_type = AdvIslands.calc_with_hidden_with_mark_seam
                self.umeshes.ensure(True)
        else:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                self.has_selected = True
                self.umeshes = selected_umeshes
                self.islands_calc_type = AdvIslands.calc_extended_with_mark_seam
            elif unselected_umeshes:
                self.has_selected = False
                self.umeshes = unselected_umeshes
                self.islands_calc_type = AdvIslands.calc_visible_with_mark_seam
            else:
                cancel = True

        if cancel:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        all_islands = []
        selected_islands_of_mesh = []
        zero_area_islands = []
        self.umeshes.update_tag = False

        for umesh in self.umeshes:
            if adv_islands := self.islands_calc_type(umesh):  # noqa
                umesh.value = umesh.check_uniform_scale(report=self.report)

                if self.grouping_type != 'NONE':
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_uv_coords(save_triplet=True)
                    all_islands.extend(adv_islands)

                adv_islands.calc_area_uv()
                adv_islands.calc_area_3d(scale=umesh.value)

                if self.grouping_type == 'NONE':
                    for isl in adv_islands:
                        if (status := isl.set_texel(self.texel, self.texture_size)) is None:
                            zero_area_islands.append(isl)
                            continue
                        isl.umesh.update_tag |= status

                if self.has_selected:
                    selected_islands_of_mesh.append(adv_islands)

        if self.grouping_type != 'NONE':
            if self.grouping_type == 'OVERLAP':
                threshold = None if self.lock_overlap_mode == 'ANY' else self.threshold
                groups_of_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
                for isl in groups_of_islands:
                    if (status := isl.set_texel(self.texel, self.texture_size)) is None:
                        zero_area_islands.append(isl)
                        continue
                    isl.umesh.update_tag |= status
            else:
                union_islands = UnionIslands(all_islands)
                status = union_islands.set_texel(self.texel, self.texture_size)
                union_islands.umesh.update_tag = status in (True, None)

                for u_isl in union_islands:
                    area_3d = sqrt(u_isl.area_3d)
                    area_uv = sqrt(u_isl.area_uv) * self.texture_size
                    if isclose(area_3d, 0.0, abs_tol=1e-6) or isclose(area_uv, 0.0, abs_tol=1e-6):
                        zero_area_islands.append(union_islands)

        if zero_area_islands:
            self.report({'WARNING'}, f"Found {len(zero_area_islands)} islands with zero area")
            if self.umeshes.is_edit_mode:
                need_validation = False
                if utils.USE_GENERIC_UV_SYNC:
                    if utils.sync() and utils.get_select_mode_mesh() in ('VERT', 'EDGE'):
                        need_validation = True

                for islands in selected_islands_of_mesh:
                    for isl in islands:
                        isl.select = False
                for isl in zero_area_islands:
                    if need_validation:
                        isl.umesh.sync_from_mesh_if_needed()
                    isl.select = True
            self.umeshes.update_tag = True
            self.umeshes.silent_update()
            if not self.umeshes.is_edit_mode:
                self.umeshes.free()
                utils.update_area_by_type('VIEW_3D')
            return {'FINISHED'}

        if not self.umeshes.is_edit_mode:
            self.umeshes.update(info='All islands adjusted')
            self.umeshes.free()
            if self.umeshes.update_tag:
                utils.update_area_by_type('VIEW_3D')
            return {'FINISHED'}
        self.umeshes.update(info='All islands adjusted')
        return {'FINISHED'}


class UNIV_OT_TexelDensitySet(UNIV_OT_TexelDensitySet_VIEW3D):
    bl_idname = "uv.univ_texel_density_set"


class UNIV_OT_TexelDensityGet_VIEW3D(Operator):
    bl_idname = "mesh.univ_texel_density_get"
    bl_label = 'Get'
    bl_description = "Get Texel Density"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texel: float = 1.0
        self.texture_size: float = 2048.0
        self.has_selected = True
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.texel = univ_settings().texel_density
        self.texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        self.umeshes = UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()
            self.umeshes.sync_invalidate()

        cancel = False
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                self.has_selected = True
                self.umeshes = selected_umeshes
            elif unselected_umeshes:
                self.has_selected = False
                self.umeshes = unselected_umeshes
            else:
                cancel = True
        else:
            if not self.umeshes:
                cancel = True
            else:
                self.has_selected = False

        if cancel:
            self.report({'WARNING'}, 'Faces not found')
            return {'CANCELLED'}

        total_3d_area = 0.0
        total_uv_area = 0.0

        for umesh in self.umeshes:
            if self.umeshes.is_edit_mode:
                faces = utils.calc_uv_faces(umesh, selected=self.has_selected)
            else:
                faces = umesh.bm.faces
            scale = umesh.check_uniform_scale(self.report)
            total_3d_area += utils.calc_total_area_3d(faces, scale)
            total_uv_area += utils.calc_total_area_uv(faces, umesh.uv)

        self.umeshes.free()

        area_3d = sqrt(total_3d_area)
        area_uv = sqrt(total_uv_area) * self.texture_size
        if isclose(area_3d, 0.0, abs_tol=1e-6) or isclose(area_uv, 0.0, abs_tol=1e-6):
            self.report({'WARNING'}, f"All faces has zero area")
            return {'CANCELLED'}
        texel = area_uv / area_3d
        univ_settings().texel_density = bl_math.clamp(texel, 1.0, 100_000.0)
        utils.update_univ_panels()
        return {'FINISHED'}


class UNIV_OT_TexelDensityGet(UNIV_OT_TexelDensityGet_VIEW3D):
    bl_idname = "uv.univ_texel_density_get"


POLIIGON_PHYSICAL_SIZES: dict[int | tuple[float, float]] | dict[int | None] | None = None


class UNIV_OT_TexelDensityFromTexture(Operator):
    bl_idname = "uv.univ_texel_density_from_texture"
    bl_label = 'TD From Texture'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Extracts dimensions from texture name or metadata.\n\n" \
        "Name_30cm_Albedo → 0.3m\n" \
        "Name_2.5Mx2.5M_Albedo → 2.5 x 2.5 m\n" \
        "Supported units: mm, cm, m, km, in, ft, yd, mi\n\n" \
        "Quixel Megascans textures are supported if the original \n" \
        "filenames are intact and the texture path contains the corresponding JSON file. \n\n" \
        "Poliigon textures are supported if the naming convention with the texture ID is preserved." \


    def execute(self, context):
        area = bpy.context.area
        if not area or area.type != 'IMAGE_EDITOR':
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}

        space_data = area.spaces.active
        if not (space_data and space_data.image):
            self.report({'WARNING'}, 'Not found active image')
            return {'CANCELLED'}

        img = space_data.image
        image_width, image_height = img.size
        if not image_height:
            self.report({'WARNING'}, 'Active image not valid')
            return {'CANCELLED'}

        if int(univ_settings().size_x) != image_width or int(univ_settings().size_y) != image_height:
            self.report({'INFO'}, 'Resolution of active texture and resolution of texture in '
                                  'Texel Density do not match. The resolution from Texel Density is used.')

        if size := self.get_physical_size_from_name(img.name):
            self.update_texel_from_size(size)
            return {'FINISHED'}

        if img.name.startswith('Poliigon_'):
            if size := self.get_physical_size_poligon(img.name):
                self.update_texel_from_size(size)
            return {'FINISHED'}

        if not img.packed_file:
            path = Path(img.filepath)
            if size := self.get_physical_size_quixel(path):
                self.update_texel_from_size(size)
                return {'FINISHED'}
        self.report({'WARNING'}, 'Physical size not found in name or metadata')

        return {'FINISHED'}

    @staticmethod
    def update_texel_from_size(size):
        x, y = size
        x_td = int(univ_settings().size_x) / x
        y_td = int(univ_settings().size_y) / y
        univ_settings().texel_density = (x_td + y_td) * 0.5
        utils.update_univ_panels()

    @staticmethod
    def get_physical_size_from_name(name: str):
        pattern = rf'_(\d+(?:\.\d+)?){utils.UNITS}(?:\s*[xх×]\s*(\d+(?:\.\d+)?){utils.UNITS}?)?'
        matches = re.finditer(pattern, name, flags=re.IGNORECASE)
        for m in matches:
            g = m.groups()
            if g[2]:
                unit2 = g[3] if g[3] else g[1]
                x_size = utils.unit_conversion(float(g[0]), g[1], 'm')
                y_size = utils.unit_conversion(float(g[2]), unit2, 'm')
            else:
                x_size = y_size = utils.unit_conversion(float(g[0]), g[1], 'm')
            return x_size, y_size

    @staticmethod
    def get_physical_size_quixel(image_path: Path):
        if not image_path.exists():
            return None

        if not (prefix := image_path.stem.split('_')[0]):
            return None

        quixel_json = image_path.parent / f'{prefix}.json'
        if not quixel_json.exists():
            quixel_json = image_path.parent / f'{prefix}0.json'
            if not quixel_json.exists():
                return None

        with open(quixel_json) as f:
            js = json.load(f)
            for key in js:
                if key != 'physicalSize':
                    continue
                size_info = js[key]

                if size_info is None:
                    return False

                if 'x' in size_info:
                    splitted = size_info.split('x')
                    if len(splitted) != 2:
                        break

                    try:
                        return float(splitted[0]), float(splitted[1])
                    except:  # noqa
                        return False
                break
        return False

    def get_physical_size_poligon(self, name: str):
        import requests  # type: ignore[import-untyped]
        match_poliigon_id = re.search(r'_(\d{4,})_', name)
        if not match_poliigon_id:
            self.report({'WARNING'}, 'Not found id from poliigon texture')
            return

        try:
            poliigon_id: int = int(match_poliigon_id.group(1))

            if POLIIGON_PHYSICAL_SIZES is None:
                self.load_poliigon_physical_size_cache()

            if poliigon_id in POLIIGON_PHYSICAL_SIZES:
                ret = POLIIGON_PHYSICAL_SIZES[poliigon_id]
                if not ret:
                    self.report({'WARNING'}, "Sizes not found")
                return ret

            # TODO: Fix toolbox.get_context
            # p = __import__("poliigon-addon-blender")
            # self = p.toolbox.cTB
            # asset_data = self._asset_index.get_asset(7787)
            # asset_data.specifications.get("physical_size_cm", {})
            # TODO: Implement cache system (and save them in txt)
            url = f"https://www.poliigon.com/texture/.../{poliigon_id}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers, timeout=(3, 10), stream=True)

            if response.status_code == 404:
                self.report({'WARNING'}, f'Not found url {url!r} for get texel density from poliigon metadata')
                return

            # Example: <p data-v-8f651ff6="">2.50 m  x 2.50 m</p>
            pattern = rf'>(\d+(?:\.\d+)?)\s*{utils.UNITS}\s*x\s*(\d+(?:\.\d+)?)\s*{utils.UNITS}<'
            match = re.search(pattern, response.text)
            if match:
                width = match.group(1)
                width_unit = match.group(2)
                height = match.group(3)
                height_unit = match.group(4)
                x_size = utils.unit_conversion(float(width), width_unit, 'm')
                y_size = utils.unit_conversion(float(height), height_unit, 'm')
                size = x_size, y_size
                POLIIGON_PHYSICAL_SIZES[poliigon_id] = size
                return size
            else:
                POLIIGON_PHYSICAL_SIZES[poliigon_id] = None
                self.report({'WARNING'}, "Sizes not found")
        except requests.exceptions.ConnectionError:
            self.report({'WARNING'}, 'Not found internet connection for get texel density from id')
        except requests.exceptions.Timeout:
            self.report({'WARNING'}, 'Server response timeout, try again')

    @staticmethod
    def load_poliigon_physical_size_cache():
        json_path = Path(__file__).parent / 'poliigon_physical_size_cache.json'
        global POLIIGON_PHYSICAL_SIZES
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    POLIIGON_PHYSICAL_SIZES = {int(k): v for k, v in json.load(f).items()}
                except (ValueError, json.JSONDecodeError):
                    POLIIGON_PHYSICAL_SIZES = {}
        else:
            POLIIGON_PHYSICAL_SIZES = {}

    @staticmethod
    def store_poliigon_physical_size_cache():
        json_path = Path(__file__).parent / 'poliigon_physical_size_cache.json'
        global POLIIGON_PHYSICAL_SIZES
        if POLIIGON_PHYSICAL_SIZES:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(POLIIGON_PHYSICAL_SIZES, f, sort_keys=True, indent=4, separators=(',', ': '))  # noqa


class TexelDensity_NameExtract_Test:
    @staticmethod
    def extract_meters_from_name_test():
        texts = ('wood_2mx4m_albedo.png',
                 'brick_1m_diffuse.jpg',
                 'tile_100cmx200cm_normal.png',
                 'metal_0.5mx1m_roughness.tga',
                 'fabric_50kmx50m_albedo.tif',
                 'ground_5.5cmx6.2_albedo.tif',
                 'marble_ric_5in_albedo.tif',
                 'bronze_rich_5ft_albedo.tif',
                 'gold_6ftx6ft_albedo.tif',
                 'paper_dd__8_6mix6mi_albedo.tif')

        for name in texts:
            UNIV_OT_TexelDensityFromTexture.get_physical_size_from_name(name)


class UNIV_OT_TexelDensityFromPhysicalSize(Operator):
    bl_idname = "uv.univ_texel_density_from_physical_size"
    bl_label = 'TD from Phys Size'
    bl_description = "Calculate Texel Density from Physical Texture Size.\n" \
                     " In the Y component, it's not necessary to specify the size if the width and height are equal.\n" \
                     "Formula: TD = Global Texture Size / Physical Texture Size (meters)"

    def execute(self, context):
        size = univ_settings().texture_physical_size.copy()
        if utils.vec_isclose_to_zero(size, abs_tol=0.001):
            self.report({'WARNING'}, 'Physical Size must be a non-zero size')
            return {'CANCELLED'}

        if size[1] == 0.0:
            size[1] = size[0]
        elif size[0] == 0.0:
            size[0] = size[1]
        UNIV_OT_TexelDensityFromTexture.update_texel_from_size(size)
        return {'FINISHED'}


class UNIV_OT_CalcUDIMsFrom_3DArea(Operator):
    bl_idname = "uv.univ_calc_udims_from_3d_area"
    bl_label = 'Calc UDIMs from 3D Area'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Calculates the required UDIMs count coefficient from the 3D area \n" \
        "relative to the global texture resolution and texel size."

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        umeshes = UMeshes(report=self.report)
        if not self.bl_idname.startswith('UV') or not umeshes.is_edit_mode:
            umeshes.set_sync()
            umeshes.sync_invalidate()

        has_selected = False
        if umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                has_selected = True
                umeshes = selected_umeshes
            else:
                umeshes = unselected_umeshes

        if not umeshes:
            self.report({'WARNING'}, 'Faces not found')
            return {'CANCELLED'}

        total_3d_area = 0.0
        for umesh in umeshes:
            if umeshes.is_edit_mode:
                faces = utils.calc_uv_faces(umesh, selected=has_selected)
            else:
                faces = umesh.bm.faces
            scale = umesh.check_uniform_scale()
            total_3d_area += utils.calc_total_area_3d(faces, scale)

        res = self.compute_required_udims(total_3d_area)
        self.report({'INFO'}, f'Average tiles coefficient {res:.1f}')
        umeshes.free()
        return {'FINISHED'}

    @staticmethod
    def compute_required_udims(geom_area):
        texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        texels_per_tile = texture_size ** 2
        texel_area_m2 = 1 / univ_settings().texel_density ** 2
        tile_coverage_m2 = texels_per_tile * texel_area_m2
        return geom_area / tile_coverage_m2 * 1.15


class UNIV_OT_CalcUDIMsFrom_3DArea_VIEW3D(UNIV_OT_CalcUDIMsFrom_3DArea):
    bl_idname = "mesh.univ_calc_udims_from_3d_area"


class UNIV_OT_Calc_UV_Area(Operator):
    bl_idname = "uv.univ_calc_uv_area"
    bl_label = 'Area'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        umeshes = UMeshes(report=self.report)
        if not self.bl_idname.startswith('UV') or not umeshes.is_edit_mode:
            umeshes.set_sync()
            umeshes.sync_invalidate()

        has_selected = False
        if umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                has_selected = True
                umeshes = selected_umeshes
            else:
                umeshes = unselected_umeshes

        if not umeshes:
            self.report({'WARNING'}, 'Faces not found')
            return {'CANCELLED'}

        total_area = 0.0
        for umesh in umeshes:
            if umeshes.is_edit_mode:
                faces = utils.calc_uv_faces(umesh, selected=has_selected)
            else:
                faces = umesh.bm.faces
            total_area += utils.calc_total_area_uv(faces, umesh.uv)

        self.report({'INFO'}, f'UV Area: {total_area:.4f}')
        umeshes.free()
        return {'FINISHED'}


class UNIV_OT_Calc_UV_Area_VIEW3D(UNIV_OT_Calc_UV_Area):
    bl_idname = "mesh.univ_calc_uv_area"


class UNIV_OT_Calc_UV_Coverage(Operator):
    bl_idname = "uv.univ_calc_uv_coverage"
    bl_label = 'Coverage'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Calculates coverage area. Overlaps do not increase the total value. \n\n" \
        "NOTE: The tiles used for coverage calculation are determined \nby vertex and face center inclusion in a tile.\n\n" \
        "For example, a plane scaled 10x will result in 6 tiles, not 100."

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def execute(self, context):
        umeshes = UMeshes(report=self.report)
        if not self.bl_idname.startswith('UV') or not umeshes.is_edit_mode:
            umeshes.set_sync()
            umeshes.sync_invalidate()

        has_selected = False
        if umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                has_selected = True
                umeshes = selected_umeshes
            else:
                umeshes = unselected_umeshes

        if not umeshes:
            self.report({'WARNING'}, 'Faces not found')
            return {'CANCELLED'}

        tiles = set()
        coords = []
        coords_append = coords.append
        for umesh in umeshes:
            uv = umesh.uv
            if umeshes.is_edit_mode:
                islands = AdvIslands.calc_extended_or_visible(umesh, extended=has_selected)
                islands.calc_tris()
                tris_iter = (t for isl in islands for t in isl.tris)
            else:
                tris_iter = umesh.bm.calc_loop_triangles()

            for a, b, c in tris_iter:
                coords_append(a[uv].uv)
                coords_append(b[uv].uv)
                coords_append(c[uv].uv)

        arr = np.empty((len(coords), 2), dtype=np.float32)
        for i, v in enumerate(coords):
            arr[i] = v.to_tuple()  # to_tuple -> x4 performance
        coords = arr

        triplet_coords = coords.reshape(-1, 3, 2)
        centroid = np.mean(triplet_coords, axis=1)

        np_tiles = np.floor(centroid).astype(np.int32)
        tiles.update(((tuple(t) for t in np_tiles)))

        centered = np.nextafter(triplet_coords, centroid.reshape(-1, 1, 2))
        np_tiles = np.floor(centered).astype(np.int32).reshape(-1, 2)

        tiles.update(((tuple(t) for t in np_tiles)))

        if len(tiles) > 200:
            self.report({'WARNING'}, f'Too many tiles ({len(tiles)}) - operation cancelled')
            return {'CANCELLED'}

        from gpu_extras.batch import batch_for_shader
        batch = batch_for_shader(shaders.UNIFORM_COLOR, 'TRIS', {"pos": coords})
        umeshes.free()

        self.draw_coverage(tiles, shaders.UNIFORM_COLOR, batch)

        return {'FINISHED'}

    def draw_coverage(self, tiles, shader, batch):
        size_x = int(univ_settings().size_x)
        size_y = int(univ_settings().size_y)
        offscreen = gpu.types.GPUOffScreen(size_x, size_y)
        offscreen.bind()

        shaders.blend_set_alpha()

        total_coverage = 0
        tiles_with_res = {}
        try:
            for tile in tiles:
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                with gpu.matrix.push_pop():
                    gpu.matrix.load_matrix(self.get_normalize_uvs_matrix(tile))
                    gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                    shader.bind()
                    shader.uniform_float("color", (1, 1, 1, 1))
                    batch.draw(shader)

                pixel_data = fb.read_color(0, 0, size_x, size_y, 4, 0, 'UBYTE')
                pixel_data.dimensions = size_x * size_y * 4

                alpha_channel = np.frombuffer(pixel_data, dtype=np.uint8)[3::4]
                uv_coverage = np.count_nonzero(alpha_channel) / (size_x * size_y)
                if uv_coverage:
                    tiles_with_res[tile] = uv_coverage
                    total_coverage += uv_coverage
        finally:
            offscreen.unbind()
            offscreen.free()

        self.report({'INFO'}, f' Total UV Coverage: {total_coverage:.4f}')
        if len(tiles_with_res) > 1:
            tiles_with_value = list(tiles_with_res.items())
            tiles_with_value.sort(key=lambda tup: tup[0][1], reverse=True)
            tiles_with_value.sort(key=lambda tup: tup[0][0], reverse=True)
            text = []
            for t, v in tiles_with_value:
                first = f"{t[0]}, {t[1]}"
                first += (6 - len(first)) * '  '
                text.append(f"{first} = {v: .5f}")

            from .. import draw
            if not self.bl_idname.startswith('UV'):
                draw.TextDraw.target_area = 'VIEW_3D'

            draw.TextDraw.draw(text)
            draw.TextDraw.max_draw_time = 4
            bpy.context.area.tag_redraw()

        shaders.blend_set_none()

    @staticmethod
    def get_normalize_uvs_matrix(tile):
        """Matrix maps x and y coordinates from [0, 1] to [-1, 1]"""
        matrix = Matrix.Identity(4)
        matrix.col[3][0] = -1 - (tile[0] * 2)
        matrix.col[3][1] = -1 - (tile[1] * 2)
        matrix[0][0] = 2
        matrix[1][1] = 2
        return matrix


class UNIV_OT_Calc_UV_Coverage_VIEW3D(UNIV_OT_Calc_UV_Coverage):
    bl_idname = "mesh.univ_calc_uv_coverage"

# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import enum
import bl_math
import textwrap

from itertools import chain
from mathutils import Vector
from statistics import median_high
from mathutils.geometry import area_tri
from bmesh.types import BMFace
from bpy.props import *
from bpy.types import Operator

from .. import utils
from .. import utypes
from ..utypes import UMeshes


class Inspect(enum.IntFlag):
    Overlap = enum.auto()
    OverlapInexact = enum.auto()
    OverlapSelf = enum.auto()
    OverlapTroubleFace = enum.auto()
    OverlapByMaterial = enum.auto()
    OverlapWithModifier = enum.auto()
    __pass1 = enum.auto()
    __pass2 = enum.auto()

    AllOverlapFlags = (Overlap | OverlapWithModifier | OverlapInexact |
                       OverlapSelf | OverlapTroubleFace | OverlapByMaterial)
    Zero = enum.auto()
    __pass3 = enum.auto()

    OverlapByPadding = enum.auto()
    DoubleVertices3D = enum.auto()
    TileIsect = enum.auto()

    Flipped = enum.auto()
    Flipped3D = enum.auto()
    InsideNormalsOrient = enum.auto()
    Rotated = enum.auto()
    __pass4 = enum.auto()

    NonManifold = enum.auto()
    NonSplitted = enum.auto()

    Over = enum.auto()
    __pass6 = enum.auto()
    AngleStretch = enum.auto()
    __pass7 = enum.auto()

    Concave = enum.auto()
    DeduplicateUVLayers = enum.auto()
    RepairAfterJoin = enum.auto()
    __pass8 = enum.auto()
    IncorrectBMeshTags = enum.auto()
    Other = enum.auto()

    @classmethod
    def default_value_for_settings(cls):
        return cls.Overlap | cls.Zero | cls.Flipped | cls.Over | cls.NonSplitted | cls.Other


class UNIV_OT_Check_Zero(Operator):
    bl_idname = "uv.univ_check_zero"
    bl_label = "Zero"
    bl_description = "Select degenerate UVs (zero area UV triangles)"
    bl_options = {'REGISTER', 'UNDO'}

    precision: FloatProperty(name='Precision', default=1e-5, min=0, soft_max=0.01, step=0.001, precision=6)  # noqa

    def draw(self, context):
        self.layout.prop(self, 'precision', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = self.zero(umeshes, self.precision)
        umeshes.update()

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    @staticmethod
    def zero(umeshes, precision=1e-5):
        sync = umeshes.sync
        tool_settings = bpy.context.scene.tool_settings
        sticky_mode = tool_settings.uv_sticky_select_mode

        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        precision *= 0.0001
        total_counter = 0

        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            is_invisible = utils.face_invisible_get_func(umesh)
            if sticky_mode == 'DISABLED':
                face_select_set = utils.face_select_func(umesh)
            else:
                face_select_set = utils.face_select_linked_func(umesh)

            to_select = set()
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if is_invisible(face) or face in to_select:
                    continue

                area = area_tri(tris_a[uv].uv, tris_b[uv].uv, tris_c[uv].uv)
                if area <= precision:
                    face_select_set(face)
                    to_select.add(face)

            if to_select:
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()

                if sticky_mode == 'DISABLED':
                    set_face_select = utils.face_select_func(umesh)
                else:
                    set_face_select = utils.face_select_linked_func(umesh)

                for f in to_select:
                    set_face_select(f)

            umesh.update_tag |= bool(to_select)
            total_counter += len(to_select)
        return total_counter


class UNIV_OT_Check_Flipped(Operator):
    bl_idname = "uv.univ_check_flipped"
    bl_label = "Flipped"
    bl_description = "Select flipped UV faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        result = self.flipped(umeshes)
        warning_info = self.data_formatting(result)

        umeshes.update()

        if warning_info:
            self.report({'WARNING' if result[0] else 'INFO'}, warning_info)
        else:
            self.report({'INFO'}, 'Flipped faces not found')
        return {'FINISHED'}

    @staticmethod
    def flipped(umeshes):
        sync = umeshes.sync
        tool_settings = bpy.context.scene.tool_settings
        sticky_mode = tool_settings.uv_sticky_select_mode

        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'


        total_counter = 0
        flipped_tris_counter = 0

        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            is_invisible = utils.face_invisible_get_func(umesh)


            to_select = set()
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if is_invisible(face) or face in to_select:
                    continue

                ax, ay = tris_a[uv].uv
                bx, by = tris_b[uv].uv
                cx, cy = tris_c[uv].uv

                # NOTE: The signed area calculated via the cross product
                # has floating-point inaccuracies, so we use the determinant instead.
                signed_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if signed_area < 0.0:
                    if utils.calc_signed_face_area_uv(face, uv) > 3e-08:
                        # TODO: Add flash system for flipped triangles
                        flipped_tris_counter += 1
                        continue
                    to_select.add(face)

            if to_select:
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()

                if sticky_mode == 'DISABLED':
                    set_face_select = utils.face_select_func(umesh)
                else:
                    set_face_select = utils.face_select_linked_func(umesh)

                for f in to_select:
                    set_face_select(f)

            umesh.update_tag |= bool(to_select)
            total_counter += len(to_select)
        return total_counter, flipped_tris_counter

    @staticmethod
    def data_formatting(counters: tuple[int, int]) -> str:
        total_counter, flipped_tris_counter = counters
        r_text = ''
        if total_counter:
            r_text = f'Detected {total_counter} flipped faces'
        elif flipped_tris_counter:
            r_text = (f'No flipped face found, but {flipped_tris_counter} triangles within the face were detected '
                      f'that may become flipped during triangulation.')
        return r_text


class UNIV_OT_Check_Over(Operator):
    bl_idname = 'uv.univ_check_over'
    bl_label = 'Over'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """Selects overstretched edges and overscaled faces.\n
Edge Overstretch is checked individually per face -
edges from other faces are not taken into account.
In other words, stretches are calculated relative to the majority
of edges with a similar coefficient (Coefficient = 3D Edge Length / UV Edge Length).\n
This behavior is preferred, as it allows setting a higher
Overscaled Face Threshold to intentionally exclude uniformly
scaled islands from the calculation, focusing only on actual stretches"""

    edge_over_threshold: FloatProperty(name='Overstretched Edge Threshold', min=0.01, soft_min=0.05, max=2, soft_max=0.5, default=0.2,
                                       description='Selects edge that fall outside this range.')
    face_over_threshold: FloatProperty(name='Overscaled Face Threshold', min=0.01, soft_min=0.05, max=2, soft_max=0.5, default=0.25,
                                       description='Selects faces that fall outside this range.')

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'edge_over_threshold', slider=True)
        layout.prop(self, 'face_over_threshold', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        umeshes.fix_context()
        umeshes.filter_by_visible_uv_faces()
        umeshes.calc_aspect_ratio(from_mesh=False)
        bpy.ops.uv.select_all(action='DESELECT')

        # clamp angle
        result = self.over(umeshes, edge_threshold=self.edge_over_threshold, face_threshold=self.face_over_threshold)

        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'No overscales or overstretches found.')

        return {'FINISHED'}

    @staticmethod
    def overstretched_edges(umesh, faces, edge_threshold):
        uv = umesh.uv
        is_pair = utils.is_pair
        is_visible = utils.is_visible_func(umesh.sync)
        scale_3d = umesh.check_uniform_scale()
        if scale_3d and utils.vec_isclose(scale_3d, scale_3d.zzz):
            # For non-anisotropic scale not need scale correct for calc coef
            scale_3d = None
        aspect_scale = Vector((umesh.aspect, 1))

        to_select = []
        edges_counter = 0
        for f in faces:
            edge_coefficients = []
            if scale_3d:
                for crn in f.loops:
                    edge_length_3d = ((crn.vert.co - crn.link_loop_next.vert.co) * scale_3d).length
                    edge_length_uv = ((crn[uv].uv - crn.link_loop_next[uv].uv) * aspect_scale).length
                    if not edge_length_uv:
                        edge_coefficients.append(0.0)
                    else:
                        edge_coefficients.append(edge_length_3d / edge_length_uv)
            else:
                for crn in f.loops:
                    edge_length_3d = crn.edge.calc_length()
                    edge_length_uv = ((crn[uv].uv - crn.link_loop_next[uv].uv) * aspect_scale).length
                    if not edge_length_uv:
                        edge_coefficients.append(0.0)
                    else:
                        edge_coefficients.append(edge_length_3d / edge_length_uv)

            median = median_high(edge_coefficients)

            if median == 0.0:
                median = max(edge_coefficients)

            low = median * (bl_math.clamp(1 - edge_threshold, 0.01, 1))
            high = median * (1 + edge_threshold)

            local_edges_counter = 0
            for crn, coef in zip(f.loops, edge_coefficients):
                if coef < low or high < coef:
                    to_select.append(crn)
                    pair = crn.link_loop_radial_prev
                    if is_pair(crn, pair, uv) and is_visible(pair.face):
                        local_edges_counter += 1
                    else:
                        local_edges_counter += 2

            if local_edges_counter:
                if local_edges_counter >= 2:
                    local_edges_counter //= 2
                edges_counter += local_edges_counter
        return to_select, edges_counter

    @classmethod
    def over(cls, umeshes: UMeshes, edge_threshold=0.2, face_threshold=0.25, batch_inspect=False):
        edge_threshold *= 2
        face_threshold *= 2
        from ..utils import calc_face_area_uv, calc_face_area_3d
        face_seq_coef_by_mesh: list[tuple[types.UMesh, list[BMFace], list[float]]] = []  # noqa

        for umesh in umeshes:
            uv = umesh.uv

            scale = umesh.check_uniform_scale()
            face_coef_seq: list[float] = []
            face_coef_seq_append = face_coef_seq.append

            # Overscaled Faces:
            if faces := utils.calc_visible_uv_faces(umesh):
                if scale:
                    for f in faces:
                        face_area_3d = calc_face_area_3d(f, scale)
                        face_area_uv = calc_face_area_uv(f, uv)
                        if not face_area_uv:
                            face_coef_seq_append(0.0)
                        else:
                            face_coef_seq_append(face_area_3d / face_area_uv)
                else:
                    for f in faces:
                        face_area_3d = f.calc_area()
                        face_area_uv = calc_face_area_uv(f, uv)
                        if not face_area_uv:
                            face_coef_seq_append(0.0)
                        else:
                            face_coef_seq_append(face_area_3d / face_area_uv)

                face_seq_coef_by_mesh.append((umesh, faces, face_coef_seq))  # noqa

        edges_counter = 0
        faces_counter = 0

        if face_seq_coef_by_mesh:
            median = median_high(chain.from_iterable((face_data[2] for face_data in face_seq_coef_by_mesh)))
            if median == 0.0:
                median = max(chain.from_iterable((face_data[2] for face_data in face_seq_coef_by_mesh)))

            low = median * (bl_math.clamp(1 - face_threshold, 0.01, 1))
            high = median * (1 + face_threshold)

            for umesh, faces, coefficients in face_seq_coef_by_mesh:
                over_faces = []
                for f, coef in zip(faces, coefficients):
                    if coef < low or high < coef:
                        over_faces.append(f)

                if over_faces:
                    if umesh.sync and umesh.elem_mode in ('VERT', 'EDGE'):
                        umesh.sync_from_mesh_if_needed()

                    set_face_select = utils.face_select_linked_func(umesh)
                    for f in over_faces:
                        set_face_select(f)

                to_select_edges, local_edge_counter = cls.overstretched_edges(umesh, faces, edge_threshold)
                umesh.sequence = to_select_edges
                faces_counter += len(over_faces)
                edges_counter += local_edge_counter
                umesh.update_tag |= len(over_faces) or local_edge_counter

        if edges_counter:
            if batch_inspect or faces_counter:
                # Avoid switch elem mode if all edges selected for batch inspect
                if cls.is_full_edge_selected_in_seq(umeshes):
                    for umesh in umeshes:
                        umesh.sequence = []
                    return edges_counter, faces_counter

            if umeshes.elem_mode not in ('EDGE', 'VERT'):
                umeshes.elem_mode = 'EDGE'

            for umesh in umeshes:
                if umesh.sync and umesh.sequence:
                    umesh.sync_from_mesh_if_needed()

                edge_select_set = utils.edge_select_linked_set_func(umesh)
                for edge in umesh.sequence:
                    edge_select_set(edge, True)
                umesh.sequence = []

        return edges_counter, faces_counter

    @staticmethod
    def data_formatting(counters):
        edges_counter, faces_counter = counters
        r_text = ''
        if edges_counter:
            r_text += f'Overstretched Edges - {edges_counter}. '
        if faces_counter:
            r_text += f'Overscaled Faces - {faces_counter}. '

        if r_text:
            r_text = f'Found: {r_text}'
        return r_text

    @staticmethod
    def is_full_edge_selected_in_seq(umeshes):
        if umeshes.elem_mode in ('FACE', 'ISLAND'):
            for umesh in umeshes:
                face_select_get = utils.face_select_get_func(umesh)
                if not all(face_select_get(crn_edge.face) and face_select_get(crn_edge.link_loop_radial_prev.face)
                           for crn_edge in umesh.sequence):
                    return False
        else:
            for umesh in umeshes:
                edge_select_get = utils.edge_select_get_func(umesh)
                if not all(edge_select_get(crn_edge) for crn_edge in umesh.sequence):
                    return False
        return True


class UNIV_OT_Check_Non_Splitted(Operator):
    bl_idname = 'uv.univ_check_non_splitted'
    bl_label = 'Non-Splitted'
    bl_description = "Selects the edges where seams should be marked and unwrapped without connection"
    bl_options = {'REGISTER', 'UNDO'}

    use_auto_smooth: BoolProperty(name='Use Auto Smooth', default=True)
    user_angle: FloatProperty(name='Smooth Angle', default=math.radians(
        66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'use_auto_smooth')
        layout.prop(self, 'user_angle', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()
        umeshes.fix_context()
        bpy.ops.uv.select_all(action='DESELECT')

        # clamp angle
        if self.use_auto_smooth:
            max_angle_from_obj_smooth = max(umesh.smooth_angle for umesh in umeshes)
            self.user_angle = bl_math.clamp(self.user_angle, 0.0, max_angle_from_obj_smooth)

        result = self.non_splitted(umeshes, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def non_splitted(umeshes: UMeshes, use_auto_smooth, user_angle, batch_inspect=False):
        non_seam_counter = 0
        non_manifold_counter = 0
        angle_counter = 0
        sharps_counter = 0
        seam_counter = 0
        mtl_counter = 0

        sync = umeshes.sync
        is_boundary = utils.is_boundary_sync if sync else utils.is_boundary_non_sync

        for umesh in umeshes:
            to_select = set()
            if use_auto_smooth:
                angle = min(umesh.smooth_angle, user_angle)
            else:
                angle = user_angle

            uv = umesh.uv
            for crn in utils.calc_visible_uv_corners(umesh):
                if (pair_crn := crn.link_loop_radial_prev) in to_select:
                    continue

                edge = crn.edge
                if is_boundary(crn, uv):
                    if crn.edge.seam:
                        continue
                    if edge.is_boundary:
                        continue
                    non_seam_counter += 1
                elif not edge.smooth:
                    sharps_counter += 1
                elif (face_angle := edge.calc_face_angle(1000.0)) >= angle:
                    if face_angle == 1000.0:
                        non_manifold_counter += 1  # TODO: Check if batch_inspect disabled, after implement non manifold
                    else:
                        angle_counter += 1
                elif edge.seam:
                    seam_counter += 1
                elif pair_crn.face.material_index != crn.face.material_index:
                    mtl_counter += 1
                else:
                    continue
                to_select.add(crn)

            umesh.sequence = to_select  # noqa
            umesh.update_tag |= bool(to_select)

        if any(umesh.sequence for umesh in umeshes):
            if batch_inspect:
                # Avoid switch elem mode if all edges selected for batch inspect
                if UNIV_OT_Check_Over.is_full_edge_selected_in_seq(umeshes):
                    for umesh in umeshes:
                        umesh.sequence = []
                    return non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

            if umeshes.elem_mode not in ('EDGE', 'VERT'):
                umeshes.elem_mode = 'EDGE'

            for umesh in umeshes:
                edge_select_set = utils.edge_select_linked_set_func(umesh)
                for edge in umesh.sequence:
                    edge_select_set(edge, True)
                umesh.sequence = []
        return non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

    @staticmethod
    def data_formatting(counters):
        non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter = counters
        r_text = ''
        if non_seam_counter:
            r_text += f'Non-Seam - {non_seam_counter}. '
        if non_manifold_counter:
            r_text += f'Non-Manifolds - {non_manifold_counter}. '
        if angle_counter:
            r_text += f'Sharp Angles - {angle_counter}. '
        if sharps_counter:
            r_text += f'Mark Sharps - {sharps_counter}. '
        if seam_counter:
            r_text += f'Mark Seams - {seam_counter}. '
        if mtl_counter:
            r_text += f'Materials - {mtl_counter}. '

        if r_text:
            r_text = f'Found: {sum(counters)} non splitted edges. {r_text}'
        return r_text


class UNIV_OT_Check_Overlap(Operator):
    bl_idname = 'uv.univ_check_overlap'
    bl_label = 'Overlap'
    bl_description = "Select all UV faces which overlap each other.\n" \
                     "Unlike the default operator, this one informs about the number of faces with conflicts"
    bl_options = {'REGISTER', 'UNDO'}

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ALL',
                             items=(('ALL', 'All', ''), ('INEXACT', 'Inexact', '')))
    threshold: FloatProperty(name='Distance', default=0.0008, min=0.0, soft_min=0.00005, soft_max=0.00999)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        if self.check_mode == 'INEXACT':
            layout.prop(self, 'threshold')
        layout.row(align=True).prop(self, 'check_mode', expand=True)

    def execute(self, context):
        umeshes = UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False

        count = self.overlap_check(umeshes, self.check_mode, self.threshold)
        self.report(*self.get_info_from_count(count, self.check_mode))
        umeshes.silent_update()
        return {'FINISHED'}

    @staticmethod
    def get_info_from_count(count, check_mode):
        if count:
            if check_mode == 'INEXACT':
                return {'WARNING'}, f"Found {count} islands with inexact overlap"
            else:
                return {'WARNING'}, f"Found about {count} edges with overlap"
        else:
            if check_mode == 'INEXACT':
                return {'INFO'}, f"Inexact islands with overlap not found"
            else:
                return {'INFO'}, f"Edges with overlap not found"

    @staticmethod
    def overlap_check(umeshes, check_mode, threshold=0.001) -> int:
        if umeshes.sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        bpy.ops.uv.select_overlap()

        count = 0
        if check_mode == 'INEXACT':
            all_islands = []
            for umesh in umeshes:
                adv_islands = utypes.AdvIslands.calc_extended_with_mark_seam(umesh)
                # The following subdivision is needed to ignore the exact self overlaps
                # that are created from the flipped face
                for isl in reversed(adv_islands):
                    if isl.has_flip_with_noflip():
                        adv_islands.islands.remove(isl)
                        noflip, flipped = isl.calc_islands_by_flip_with_mark_seam()
                        adv_islands.islands.extend(noflip)
                        adv_islands.islands.extend(flipped)
                all_islands.extend(adv_islands)

            overlapped = utypes.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
            for isl in overlapped:
                if isinstance(isl, utypes.AdvIsland):
                    count += 1
                else:
                    isl.select = False
                    isl.umesh.update_tag = True
        else:
            for umesh in umeshes:
                if umesh.sync:
                    count += umesh.total_edge_sel
                else:
                    count += len(utils.calc_selected_uv_edge(umesh))

        return count


class UNIV_OT_Check_Other(Operator):
    bl_idname = 'uv.univ_check_other'
    bl_label = 'Other'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """This operator includes a number of small but important checks:\n
1. Verifies that the object contains polygons.
2. Verifies the presence of UV maps.
3. Corrects the operator call context (since many operators fail when the active object lacks polygons or UV maps).
4. Checks for applied or non-uniform object scaling.
5. Ensures a valid aspect ratio.
6. Verifies that the UV Map node in the shader has a valid name.
7. Checks whether the optimal UV smoothing method is selected in the Subdivision Surface modifier."""
# 8. Checks handlers and various flags"""

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ALL',
                             items=(('ALL', 'All', ''), ('INEXACT', 'Inexact', '')))
    threshold: FloatProperty(name='Distance', default=0.0008, min=0.0, soft_min=0.00005, soft_max=0.00999)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):

        layout = self.layout
        col = layout.column()
        global INSPECT_INFO

        if info_list := INSPECT_INFO.get('Other'):
            for check_type, info in info_list:
                box = col.box()
                wrapped_lines = textwrap.wrap(check_type + ': ' + info, width=72)
                for line in wrapped_lines:
                    box.label(text=line)

    def execute(self, context):
        global INSPECT_INFO
        INSPECT_INFO.clear()

        umeshes = UMeshes()
        umeshes.update_tag = False

        if info := self.check_other(umeshes):
            INSPECT_INFO['Other'] = info
            # umeshes.silent_update()
            return context.window_manager.invoke_popup(self, width=420)
        else:
            self.report({'INFO'}, 'Errors not found')
        return {'FINISHED'}

    @staticmethod
    def check_other(umeshes: UMeshes):
        info: list[tuple[str, str]] = []

        # Check context error
        faces = bpy.context.active_object.data.polygons
        uvs = bpy.context.active_object.data.uv_layers
        if not faces or not uvs:
            umeshes.fix_context()
            error_description = 'The active object has '
            if not faces:
                error_description += 'no polygons'
            if not uvs:
                if not faces:
                    error_description += ' and '
                error_description += 'no UV maps'

            error_id = 'Context Error'
            if not bpy.context.active_object.data.polygons or not bpy.context.active_object.data.uv_layers:
                info.append((error_id, error_description))
                return info
            else:
                error_description += '. (Fixed)'
                info.append((error_id, error_description))

        selected_objects = utils.calc_any_unique_obj()

        # Check unapplied scale
        counter = 0
        error_id = 'Unapplied Scales'
        error_description = ''
        for obj in selected_objects:
            _, _, scale = obj.matrix_world.decompose()
            if not utils.umath.vec_isclose_to_uniform(scale, abs_tol=0.01):
                counter += 1
                if counter == 1:
                    error_description = f"The {obj.name!r} hasn't applied scale: X={scale.x:.4f}, Y={scale.y:.4f}, Z={scale.z:.4f}"
        if counter:
            if counter != 1:
                error_description = f'Found {counter} objects without applied scales.'
            info.append((error_id, error_description))

        # Check UV maps exist
        uv_names: set[tuple[str, ...]] = set()
        uv_active: set[tuple[bool, ...]] = set()
        uv_active_render: set[tuple[bool, ...]] = set()
        min_uv_maps = 100
        max_uv_maps = 0
        missed_uvs_counter = 0

        error_id = 'UV Maps'
        for obj in selected_objects:
            uvs = obj.data.uv_layers
            if not len(uvs):
                missed_uvs_counter += 1  # TODO: HP check
                continue
            max_uv_maps = max(max_uv_maps, len(uvs))
            min_uv_maps = min(min_uv_maps, len(uvs))
            uv_names.add(tuple(uv.name for uv in uvs))
            uv_active.add(tuple(uv.active for uv in uvs))
            uv_active_render.add(tuple(uv.active_render for uv in uvs))

        error_description = ''
        if missed_uvs_counter:
            error_description += f'Found {missed_uvs_counter} meshes with missed UV maps. '
        if min_uv_maps != max_uv_maps:
            error_description += 'Meshes have different numbers of UVs'
        else:
            if len(uv_names) >= 2:
                error_description += 'Meshes have different names.'
            if len(uv_active) >= 2 or len(uv_active_render) >= 2:
                error_description += 'Meshes have different '
                if len(uv_active) >= 2:
                    error_description += 'active layers'
                if len(uv_active_render) >= 2:
                    if len(uv_active) >= 2:
                        error_description += 'and different active render layers'
                    else:
                        error_description += 'active render layers'
        if error_description:
            info.append((error_id, error_description))

        # Check missed faces
        counter = 0
        error_id = 'Missed Faces'
        error_description = ''
        for obj in selected_objects:
            if not obj.data.polygons:
                counter += 1
                if counter == 1:
                    error_description += f"{obj.name!r}"
                else:
                    error_description += f", {obj.name!r}"

        if counter:
            info.append((error_id, f"{error_description} meshes hasn't faces."))

        if umeshes.is_edit_mode:
            counter = 0
            mode = utils.get_select_mode_mesh()
            for umesh in umeshes:
                if mode not in umesh.bm.select_mode:
                    counter += 1
                    umesh.bm.select_mode = {mode}
            if counter:
                info.append(('Elem Mode', f"Select Mode at {counter} meshes doesn't match scene with mesh. (Fixed)"))

        # Check Heterogeneous Aspect Ratio
        aspects = {}

        def get_aspects_ratio(u):  # noqa
            if modifiers := [m for m in u.obj.modifiers if m.name.startswith('UniV Checker')]:
                socket = 'Socket_1' if 'Socket_1' in modifiers[0] else 'Input_1'
                if mtl := modifiers[0][socket]:  # noqa
                    for node in mtl.node_tree.nodes:
                        if node.bl_idname == 'ShaderNodeTexImage' and (image := node.image):
                            image_width, image_height = image.size
                            if image_height:
                                aspects[image_width / image_height] = modifiers[0][socket].name
                            else:
                                aspects[1.0] = mtl.name
                            return
                aspects[1.0] = 'Default'

            # Aspect from material
            elif u.obj.material_slots:
                for slot in u.obj.material_slots:  # noqa
                    mtl = slot.material  # noqa
                    if not slot.material:
                        aspects[1.0] = 'Default'
                        continue
                    if getattr(mtl, 'use_nodes', True) and (active_node := mtl.node_tree.nodes.active):
                        if active_node.bl_idname == 'ShaderNodeTexImage' and (image := active_node.image):
                            image_width, image_height = image.size
                            if image_height:
                                aspects[image_width / image_height] = mtl.name
                                continue
                    aspects[1.0] = mtl.name
            else:
                aspects[1.0] = 'Default'

        for umesh in umeshes:
            get_aspects_ratio(umesh)
        if len(aspects) >= 2:
            info.append(('Aspect Ratio', f'Found heterogeneous aspect ratio: ' +
                        ', '.join(f"{v!r}" for v in aspects.values())))

        # Check non valid UV Map node
        error_description = ''
        for umesh in umeshes:
            uv_maps_names = {uv.name for uv in umesh.obj.data.uv_layers}
            uv_maps_names.add('')
            for slot in umesh.obj.material_slots:
                if (mtl := slot.material) and getattr(mtl, 'use_nodes', True):
                    if non_valid_names := {f'{node.uv_map!r}' for node in mtl.node_tree.nodes
                                           if node.type == 'UVMAP' and node.uv_map not in uv_maps_names}:
                        error_description += f'Material {mtl.name!r} in {umesh.obj.name!r} object has non-valid UV Map name: '
                        error_description += ', '.join(non_valid_names) + '.\n'
        if error_description:
            info.append(('UV Map None Name', error_description))

        # Check SubDiv optimal method for UV
        names = []
        for u in umeshes:
            for mod in u.obj.modifiers:
                if isinstance(mod, bpy.types.SubsurfModifier):
                    if mod.uv_smooth in ('PRESERVE_BOUNDARIES', 'NONE'):
                        names.append(repr(u.obj.name))
                        break
        if names:
            error_description = 'The following objects use a non-optimal UV smoothing method in subdivision modifier: ' + \
                ', '.join(names)
            error_description += '. Use Keep Corners or All Method'
            info.append(('UV Smooth', error_description))

        return info


class UNIV_OT_BatchInspectFlags(Operator):
    bl_idname = 'uv.univ_batch_inspect_flags'
    bl_label = 'Flags'
    bl_description = "Inspect Flags"

    flag: IntProperty(name='Flag', default=0, min=0)

    def execute(self, context):
        from ..preferences import univ_settings
        tag = Inspect(self.flag)

        # Turn off the old (all) overlap flag if it doesn't match the new one.
        if tag in Inspect.AllOverlapFlags:
            if tag not in Inspect(univ_settings().batch_inspect_flags):
                univ_settings().batch_inspect_flags &= ~Inspect.AllOverlapFlags

        univ_settings().batch_inspect_flags ^= tag
        return {'FINISHED'}


INSPECT_INFO = {}


class UNIV_OT_BatchInspect(Operator):
    bl_idname = 'uv.univ_batch_inspect'
    bl_label = 'Inspect'
    bl_description = "Batch Inspect by Enabled tags"
    bl_options = {'REGISTER', 'UNDO'}

    inspect_all: BoolProperty(name='Inspect All', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        if 'Hidden' in INSPECT_INFO:
            box = col.box()
            box.label(text='Hidden faces found — the result may be incorrect', icon='INFO')
            col.separator()

        for inspect_flag in ('Overlap', 'Zero', 'Flipped', 'Over', 'Non-Splitted'):
            if info := INSPECT_INFO.get(inspect_flag):
                box = col.box()
                wrapped_lines = textwrap.wrap(inspect_flag + ': ' + info, width=72)
                for line in wrapped_lines:
                    box.label(text=line)

        if info_list := INSPECT_INFO.get('Other'):
            for check_type, info in info_list:
                box = col.box()
                wrapped_lines = textwrap.wrap(check_type + ': ' + info, width=72)
                for line in wrapped_lines:
                    box.label(text=line)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.inspect_all = event.alt
        return self.execute(context)

    def execute(self, context):
        from ..preferences import univ_settings
        flags = Inspect(univ_settings().batch_inspect_flags)
        global INSPECT_INFO
        INSPECT_INFO.clear()
        if not flags:
            self.report({'WARNING'}, 'Not found enabled flags')
            return {'CANCELLED'}

        umeshes = UMeshes()
        umeshes.update_tag = False

        if (Inspect.Other in flags) or self.inspect_all:
            if info := UNIV_OT_Check_Other.check_other(umeshes):
                INSPECT_INFO['Other'] = info
                if not umeshes:
                    return context.window_manager.invoke_popup(self, width=420)

        if not umeshes:
            has_uv_maps = any(obj.data.uv_layers for obj in bpy.context.selected_objects if obj.type == 'MESH')
            self.report({'WARNING'}, 'Not found meshes with polygons' if has_uv_maps else 'Not found meshes with UV maps')
            return {'CANCELLED'}

        if (flags & Inspect.AllOverlapFlags) or self.inspect_all:
            if Inspect.Overlap in flags or (self.inspect_all and not (flags & Inspect.AllOverlapFlags)):
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'ALL'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'ALL')[1]

            elif Inspect.OverlapInexact in flags:
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'INEXACT'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'INEXACT')[1]
        else:
            bpy.ops.uv.select_all(action='DESELECT')

        if Inspect.Zero in flags or self.inspect_all:
            if count := UNIV_OT_Check_Zero.zero(umeshes):
                INSPECT_INFO['Zero'] = f'Detected {count} degenerate triangles'

        if Inspect.Flipped in flags or self.inspect_all:
            result = UNIV_OT_Check_Flipped.flipped(umeshes)
            if info := UNIV_OT_Check_Flipped.data_formatting(result):
                INSPECT_INFO['Flipped'] = info

        if Inspect.Over in flags or self.inspect_all:  # Last check, because it switches elem mode to EDGE.
            result = UNIV_OT_Check_Over.over(umeshes, batch_inspect=True)
            if info := UNIV_OT_Check_Over.data_formatting(result):
                INSPECT_INFO['Over'] = info

        if Inspect.NonSplitted in flags or self.inspect_all:  # Last check, because it switches elem mode to EDGE.
            result = UNIV_OT_Check_Non_Splitted.non_splitted(
                umeshes, use_auto_smooth=True, user_angle=180, batch_inspect=True)
            if info := UNIV_OT_Check_Non_Splitted.data_formatting(result):
                INSPECT_INFO['Non-Splitted'] = info

        has_hidden_faces = False
        for umesh in umeshes:
            if has_hidden_faces:
                break
            if umesh.is_full_face_selected:
                continue

            if umeshes.sync:
                for f in umesh.bm.faces:
                    if f.hide:
                        has_hidden_faces = True
                        break
            else:
                for f in umesh.bm.faces:
                    if not f.select:
                        has_hidden_faces = True
                        break
        if has_hidden_faces:
            if not INSPECT_INFO:
                self.report({'WARNING'}, 'No problems detected, but hidden faces could lead to incorrect results.')
                return {'FINISHED'}
            INSPECT_INFO['Hidden'] = True

        if not INSPECT_INFO:
            self.report({'INFO'}, 'No errors detected.')
        else:
            umeshes.silent_update()
            return context.window_manager.invoke_popup(self, width=420)
        return {'FINISHED'}

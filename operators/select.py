# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math

from math import sqrt, isclose
from bl_math import lerp
from mathutils import Vector
from bpy.props import *
from bpy.types import Operator
from bmesh.types import BMFace
from collections.abc import Callable

from .. import utils
from .. import utypes
from ..preferences import prefs, univ_settings
from ..utypes import Islands, AdvIslands, AdvIsland, BBox, UMeshes, MeshIslands, UnionIslands

from ..utils import (
    face_centroid_uv,
)

class UNIV_OT_SelectLinked(Operator):
    bl_idname = 'uv.univ_select_linked'
    bl_label = 'Linked'
    bl_description = "Select all UV vertices linked to the active UV map"
    bl_options = {'REGISTER', 'UNDO'}

    deselect: bpy.props.BoolProperty(name='Deselect', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.deselect = event.ctrl
        return self.execute(context)

    def execute(self, context):
        umeshes = UMeshes(report=self.report)
        umeshes.fix_context()
        umeshes.update_tag = False

        select_state = not self.deselect
        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        for umesh in umeshes:
            if islands := Islands.calc_partial_selected_by_context(umesh):
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()

                for isl in islands:
                    isl.select = select_state

                if need_sync_validation_check and self.deselect:
                    umesh.bm.uv_select_sync_to_mesh()
                umesh.update_tag = True

        sel_opname = 'select' if select_state else 'deselect'
        umeshes.update(info=f'No found islands for {sel_opname}')
        return {'FINISHED'}


class UNIV_OT_Select_By_Cursor(Operator):
    bl_idname = "uv.univ_select_by_cursor"
    bl_label = "Cursor"
    bl_description = "Select by Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    face_mode: BoolProperty(name='Face Mode', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.face_mode = event.alt

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITIONAL'
        else:
            self.mode = 'SELECT'

        return self.execute(context)

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.report({'INFO'}, f"UV area not found")
            return {'CANCELLED'}

        umeshes = UMeshes(report=self.report)
        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        tile_co = utils.get_tile_from_cursor()
        view_rect = BBox.init_from_minmax(tile_co, tile_co + Vector((1, 1)))
        view_rect.pad(Vector((-2e-08, -2e-08)))

        if self.mode == 'SELECT':
            umeshes.filter_by_visible_uv_faces()
            self.select(umeshes, view_rect, need_sync_validation_check)
        elif self.mode == 'ADDITIONAL':
            umeshes.filter_by_visible_uv_faces()
            self.additional(umeshes, view_rect, need_sync_validation_check)
        else: # self.mode == 'DESELECT':
            if utils.USE_GENERIC_UV_SYNC:
                umeshes.filter_by_selected_uv_verts()
            else:
                umeshes.filter_by_selected_uv_faces()
            self.deselect(umeshes, view_rect, need_sync_validation_check)

        from .. import draw
        lines = view_rect.draw_data_lines()
        draw.LinesDrawSimple.draw_register(lines)

        umeshes.silent_update()

        return {'FINISHED'}

    def select(self, umeshes, view_box: utypes.BBox, need_sync_validation_check):
        for umesh in umeshes:
            to_select = []
            to_deselect = []
            if self.face_mode:
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()
                uv = umesh.uv
                face_select_get = utils.face_select_get_func(umesh)
                has_any_select = utils.has_any_vert_select_func(umesh)

                for f in utils.calc_visible_uv_faces_iter(umesh):
                    if face_centroid_uv(f, uv) in view_box:  # TODO: Add isect by tris
                        if not face_select_get(f):
                            to_select.append(f)
                    elif has_any_select(f):
                        to_deselect.append(f)

                if to_select or to_deselect:
                    if need_sync_validation_check:
                        umesh.sync_from_mesh_if_needed()

                    face_select_set = utils.face_select_set_func(umesh)
                    for f in to_deselect:
                        face_select_set(f, False)
                    for f in to_select:
                        face_select_set(f, True)

                    if need_sync_validation_check and to_deselect:
                        umesh.bm.uv_select_sync_to_mesh()
            else:
                adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords(save_triplet=True)

                for island in adv_islands:
                    if view_box.isect_triangles(island.flat_coords):
                        if not island.is_full_face_selected():
                            to_select.append(island)
                    else:
                        if not island.is_full_deselected_by_context():
                            to_deselect.append(island)

                if to_deselect or to_select:
                    if need_sync_validation_check:
                        umesh.sync_from_mesh_if_needed()

                    for isl in to_deselect:
                        isl.select = False
                    for isl in to_select:
                        isl.select = True
                    if need_sync_validation_check and to_deselect:
                        umesh.bm.uv_select_sync_to_mesh()
            umesh.update_tag = bool(to_select or to_deselect)

    def additional(self, umeshes: UMeshes, view_box, need_sync_validation_check):
        for umesh in umeshes:
            if umesh.has_full_selected_uv_faces():
                continue

            has_update = False
            if self.face_mode:
                uv = umesh.uv
                to_select = []
                for f in utils.calc_unselected_uv_faces_iter(umesh):
                    if face_centroid_uv(f, uv) in view_box:
                        to_select.append(f)

                if to_select:
                    has_update = True
                    if need_sync_validation_check:
                        umesh.sync_from_mesh_if_needed()
                    face_select_set = utils.face_select_set_func(umesh)
                    for f in to_select:
                        face_select_set(f, True)
            else:
                adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords(save_triplet=True)

                for island in adv_islands:
                    has_update = True
                    if island.is_full_face_selected():
                        continue
                    if view_box.isect_triangles(island.flat_coords):
                        if need_sync_validation_check:
                            umesh.sync_from_mesh_if_needed()
                        island.select = True

            umesh.update_tag = has_update

    def deselect(self, umeshes, view_box, need_sync_validation_check):
        has_update = False
        for umesh in umeshes:
            has_any_select = utils.has_any_vert_select_func(umesh)
            if self.face_mode:
                uv = umesh.uv
                to_deselect = []
                for f in utils.calc_visible_uv_faces_iter(umesh):
                    if has_any_select(f) and face_centroid_uv(f, uv) in view_box:
                        to_deselect.append(f)

                if to_deselect:
                    has_update = True
                    if need_sync_validation_check:
                        umesh.sync_from_mesh_if_needed()

                    face_select_set = utils.face_select_set_func(umesh)
                    for f in to_deselect:
                        face_select_set(f, False)
            else:
                adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords(save_triplet=True)

                for island in adv_islands:
                    if island.is_full_deselected_by_context():
                        continue

                    if view_box.isect_triangles(island.flat_coords):
                        has_update = True
                        if need_sync_validation_check:
                            umesh.sync_from_mesh_if_needed()
                        island.select = False

            if has_update and need_sync_validation_check:
                umesh.bm.uv_select_sync_to_mesh()
            umesh.update_tag = has_update


class UNIV_OT_Select_Square_Island(Operator):
    bl_idname = 'uv.univ_select_square_island'
    bl_label = 'Square'
    bl_description = 'Select Square Island'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITIONAL', 'Additional', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    shape: EnumProperty(name='Shape', default='HORIZONTAL', items=(
        ('HORIZONTAL', 'Horizontal', ''),
        ('SQUARE', 'Square', ''),
        ('VERTICAL', 'Vertical', ''),
    ))

    threshold: FloatProperty(name='Square Threshold', default=0.05, min=0, max=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITIONAL'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)

        need_sync_validation_check = False
        if self.umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = self.umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                self.umeshes.elem_mode = 'FACE'

        if self.mode == 'SELECT':
            self.umeshes.filter_by_visible_uv_faces()
            self.select(need_sync_validation_check)
        elif self.mode == 'ADDITIONAL':
            self.umeshes.filter_by_visible_uv_faces()
            self.addition(need_sync_validation_check)
        else: # self.mode == 'DESELECT':
            if utils.USE_GENERIC_UV_SYNC:
                self.umeshes.filter_by_selected_uv_verts()
            else:
                self.umeshes.filter_by_selected_uv_faces()
            self.deselect(need_sync_validation_check)

        self.umeshes.silent_update()

        return {'FINISHED'}

    def select(self, need_sync_validation_check):
        for umesh in self.umeshes:
            if islands := Islands.calc_visible_with_mark_seam(umesh):
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()
                to_select = []
                to_deselect = []

                for island in islands:
                    if self.is_target_island(island):
                        to_select.append(island)
                    else:
                        to_deselect.append(island)

                for isl in to_deselect:
                    isl.select = False
                for isl in to_select:
                    isl.select = True
            umesh.update_tag = bool(islands)

    def addition(self, need_sync_validation_check):
        for umesh in self.umeshes:
            update_tag = False
            if not umesh.has_full_selected_uv_faces():
                for island in Islands.calc_non_full_selected_with_mark_seam(umesh):
                    if self.is_target_island(island):
                        if need_sync_validation_check:
                            umesh.sync_from_mesh_if_needed()
                        island.select = True
                        update_tag = True
            umesh.update_tag = update_tag

    def deselect(self, need_sync_validation_check):
        for umesh in self.umeshes:
            update_tag = False
            for island in Islands.calc_visible_with_mark_seam(umesh):
                if utils.USE_GENERIC_UV_SYNC:
                    if island.is_full_vert_deselected():
                        continue
                else:
                    if island.is_full_face_deselected():
                        continue

                if self.is_target_island(island):
                    if need_sync_validation_check:
                        umesh.sync_from_mesh_if_needed()
                    island.select = False
                    update_tag = True

            if update_tag and need_sync_validation_check:
                umesh.bm.uv_select_sync_to_mesh()
            umesh.update_tag = update_tag

    def is_target_island(self, island):
        percent, close_to_square = self.percent_and_close_to_square(island)
        if self.shape == 'SQUARE':
            return close_to_square
        elif self.shape == 'HORIZONTAL':
            return percent > 0 and not close_to_square
        else:
            return percent < 0 and not close_to_square

    def percent_and_close_to_square(self, island):
        bbox = island.calc_bbox()
        width = bbox.width
        height = bbox.height

        if width == 0 and height == 0:
            width = height = 1
        elif width == 0:
            width = 1e-06
        elif height == 0:
            height = 1e-06

        percent = (width - height) / height
        return percent, math.isclose(percent, 0, abs_tol=self.threshold)


class UNIV_OT_Select_Border(Operator):
    bl_idname = 'uv.univ_select_border'
    bl_label = 'Border'
    bl_description = 'Select border edges'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    border_mode: EnumProperty(name='Border', default='BORDER', items=(
        ('BORDER', 'Border', ''),
        ('BORDER_BETWEEN', 'Border Between', ''),
        ('BORDER_EDGE_BY_ANGLE', 'Border Edge by Angle', ''),
        ('ALL_EDGE_BY_ANGLE', 'All Edge by Angle', ''),
    ))

    edge_dir: EnumProperty(name='Direction', default='HORIZONTAL', items=(
        ('BOTH', 'Both', ''),
        ('HORIZONTAL', 'Horizontal', ''),
        ('VERTICAL', 'Vertical', ''),
    ))

    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)
    angle: FloatProperty(name='Angle', default=math.radians(5), min=0, max=math.radians(45.001), subtype='ANGLE')

    def draw(self, context):
        if self.border_mode in ('BORDER_EDGE_BY_ANGLE', 'ALL_EDGE_BY_ANGLE'):
            row = self.layout.row(align=True)
            row.prop(self, 'edge_dir', expand=True)
            layout = self.layout
            layout.prop(self, 'angle', slider=True)
            layout.prop(self, 'use_correct_aspect')

        col = self.layout.column(align=True)
        col.prop(self, 'border_mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_vec = Vector((1, 0))
        self.y_vec = Vector((0, 1))
        self.angle_45 = math.pi / 4
        self.angle_135 = math.pi * 0.75
        self.edge_orient = self.x_vec
        self.negative_ange = 0
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.border_mode = 'BORDER_BETWEEN' if event.alt else 'BORDER'

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    def execute(self, context):
        if self.border_mode in ('BORDER_EDGE_BY_ANGLE', 'ALL_EDGE_BY_ANGLE'):
            return self.select_edge_by_angle()

        self.umeshes = UMeshes(report=self.report)
        if self.umeshes.elem_mode not in ('EDGE', 'VERT'):
            self.umeshes.elem_mode = 'EDGE'

        self.umeshes.filter_by_visible_uv_faces()

        if self.border_mode == 'BORDER':
            self.select_border()
        else:
            self.select_border_between()
        self.umeshes.silent_update()
        return {'FINISHED'}

    def select_border(self):
        # TODO: Add behavior, border by selected faces
        for umesh in self.umeshes:
            to_select = []
            to_deselect = []
            is_boundary = utils.is_boundary_func(umesh)
            corners = utils.calc_visible_uv_corners_iter(umesh)
            if self.mode == 'SELECT':
                for crn in corners:
                    if is_boundary(crn):
                        to_select.append(crn)
                    else:
                        to_deselect.append(crn)
            elif self.mode == 'DESELECT':
                for crn in corners:
                    if is_boundary(crn):
                        to_deselect.append(crn)
            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if edge_select_get(crn):
                        continue
                    if is_boundary(crn):
                        to_select.append(crn)

            utils.select_edge_processing(umesh, to_deselect, to_select)

    def select_border_between(self):
        for umesh in self.umeshes:
            to_select = []
            to_deselect = []

            islands = Islands.calc_extended_any_elem_with_mark_seam(umesh)
            islands.indexing(force=False)
            if self.mode == 'SELECT':
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn = crn.link_loop_radial_prev
                            if shared_crn.face.tag:  # No match hidden faces
                                if shared_crn.face.index != f.index:
                                    to_select.append(crn)
                                    to_select.append(shared_crn)
                                    continue
                            to_deselect.append(crn)
                            to_deselect.append(shared_crn)

            elif self.mode == 'DESELECT':
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn = crn.link_loop_radial_prev
                            if shared_crn.face.tag:  # No match hidden faces
                                if shared_crn.face.index != f.index:
                                    to_deselect.append(crn)
                                    to_deselect.append(shared_crn)
            else:  # 'ADDITION'
                for island in islands:
                    for f in island:
                        for crn in f.loops:
                            shared_crn = crn.link_loop_radial_prev
                            if shared_crn.face.tag:  # No match hidden faces
                                if shared_crn.face.index != f.index:
                                    to_select.append(crn)
                                    to_select.append(shared_crn)

            umesh.sequence = [set(to_select), set(to_deselect)]

        if not utils.USE_GENERIC_UV_SYNC and  self.mode == 'SELECT':
            # edge_select_set can leave single selected vertices, so we'll deselect everything.
            bpy.ops.uv.select_all(action='DESELECT')
        for umesh in self.umeshes:
            to_select, to_deselect = umesh.sequence
            utils.select_edge_processing(umesh, to_deselect, to_select)

    # Select Edge by Angle
    def select_edge_by_angle(self):
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.elem_mode = 'EDGE'

        if self.use_correct_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=False)

        self.edge_orient = self.x_vec if self.edge_dir == 'HORIZONTAL' else self.y_vec
        self.negative_ange = math.pi - self.angle

        if self.border_mode == 'BORDER_EDGE_BY_ANGLE':
            if self.edge_dir == 'BOTH':
                self.select_both_border()
            else:
                self.select_hv_border()
        else:
            if self.edge_dir == 'BOTH':
                self.select_both()
            else:
                self.select_hv()

        return self.umeshes.update()

    def select_hv(self):
        is_between_angle = self.is_between_angle_fn(self.angle, self.negative_ange)
        for umesh in self.umeshes:
            uv = umesh.uv
            to_select = []
            to_deselect = []

            aspect_for_x = umesh.aspect
            corners = utils.calc_visible_uv_corners_iter(umesh)
            if self.mode == 'SELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x

                    if is_between_angle(vec.angle(self.edge_orient, 0)):
                        to_select.append(crn)
                    else:
                        to_deselect.append(crn)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x

                    if is_between_angle(vec.angle(self.edge_orient, 0)):
                        to_deselect.append(crn)

            else:  # 'ADDITIONAL'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if not edge_select_get(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x

                        if is_between_angle(vec.angle(self.edge_orient, 0)):
                            to_select.append(crn)

            utils.select_edge_processing(umesh, to_deselect, to_select)

    @staticmethod
    def is_between_angle_bidirect_fn(_angle, _neg_angle):
        def inner(_x_angle, _y_angle):
            return _x_angle <= _angle or _x_angle >= _neg_angle or \
            _y_angle <= _angle or _y_angle >= _neg_angle
        return inner

    def select_both(self):
        is_between_angle = self.is_between_angle_bidirect_fn(self.angle, self.negative_ange)
        for umesh in self.umeshes:
            uv = umesh.uv
            to_select = []
            to_deselect = []

            aspect_for_x = umesh.aspect
            corners = utils.calc_visible_uv_corners_iter(umesh)
            if self.mode == 'SELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    if is_between_angle(vec.angle(self.x_vec, 0),
                                        vec.angle(self.y_vec, 0)):
                        to_select.append(crn)
                    else:
                        to_deselect.append(crn)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    vec.x *= aspect_for_x
                    if is_between_angle(vec.angle(self.x_vec, 0),
                                        vec.angle(self.y_vec, 0)):
                        to_deselect.append(crn)

            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if not edge_select_get(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        if is_between_angle(vec.angle(self.x_vec, 0),
                                            vec.angle(self.y_vec, 0)):
                            to_select.append(crn)

            utils.select_edge_processing(umesh, to_deselect, to_select)

    def select_both_border(self):
        is_between_angle = self.is_between_angle_bidirect_fn(self.angle, self.negative_ange)
        for umesh in self.umeshes:
            uv = umesh.uv
            to_select = []
            to_deselect = []

            aspect_for_x = umesh.aspect
            is_boundary = utils.is_boundary_func(umesh)
            corners = utils.calc_visible_uv_corners_iter(umesh)
            if self.mode == 'SELECT':
                for crn in corners:
                    if not is_boundary(crn):
                        to_deselect.append(crn)
                    else:
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        if is_between_angle(vec.angle(self.x_vec, 0),
                                            vec.angle(self.y_vec, 0)):
                            to_select.append(crn)
                        else:
                            to_deselect.append(crn)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    if is_boundary(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        if is_between_angle(vec.angle(self.x_vec, 0),
                                            vec.angle(self.y_vec, 0)):
                            to_deselect.append(crn)

            else:  # 'ADDITION'
                get_edge_select = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if not get_edge_select(crn) and is_boundary(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        if is_between_angle(vec.angle(self.x_vec, 0),
                                            vec.angle(self.y_vec, 0)):
                            to_select.append(crn)

            utils.select_edge_processing(umesh, to_deselect, to_select)

    @staticmethod
    def is_between_angle_fn(_angle, _neg_angle):
        def inner(_a):
            return _a <= _angle or _a >= _neg_angle
        return inner

    def select_hv_border(self):
        is_between_angle = self.is_between_angle_fn(self.angle, self.negative_ange)
        for umesh in self.umeshes:
            uv = umesh.uv
            to_select = []
            to_deselect = []

            aspect_for_x = umesh.aspect
            is_boundary = utils.is_boundary_func(umesh)
            corners = utils.calc_visible_uv_corners_iter(umesh)
            if self.mode == 'SELECT':
                for crn in corners:
                    if not is_boundary(crn):
                        to_deselect.append(crn)
                    else:
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x

                        if is_between_angle(vec.angle(self.edge_orient, 0)):
                            to_select.append(crn)
                        else:
                            to_deselect.append(crn)

            elif self.mode == 'DESELECT':
                for crn in corners:
                    if is_boundary(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x

                        if is_between_angle(vec.angle(self.edge_orient, 0)):
                            to_deselect.append(crn)

            else:  # 'ADDITION'
                edge_select_get = utils.edge_select_get_func(umesh)
                for crn in corners:
                    if not edge_select_get(crn) and is_boundary(crn):
                        vec = crn[uv].uv - crn.link_loop_next[uv].uv
                        vec.x *= aspect_for_x
                        if is_between_angle(vec.angle(self.edge_orient, 0)):
                            to_select.append(crn)

            utils.select_edge_processing(umesh, to_deselect, to_select)


class UNIV_OT_Select_Pick(Operator):
    bl_idname = 'uv.univ_select_pick'
    bl_label = 'Pick Select'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        self.umeshes = UMeshes()
        self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        self.mouse_pos = utils.get_mouse_pos(context, event)
        return self.pick_select()

    def pick_select(self):


        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            if self.select:
                if umesh.has_full_selected_uv_faces():
                    continue
            else:
                if not umesh.has_selected_uv_verts():
                    continue

            for isl in Islands.calc_visible_with_mark_seam(umesh):
                if self.select:  # Skip full selected island
                    if isl.is_full_face_selected():
                        continue
                else:  # Skip full deselected islands
                    if isl.is_full_face_deselected():
                        continue
                hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            return {'CANCELLED'}

        umesh = hit.island.umesh
        if utils.USE_GENERIC_UV_SYNC:
            if not umesh.sync_valid and self.umeshes.elem_mode in ('VERT', 'EDGE'):
                umesh.sync_valid = True
                umesh.bm.uv_select_sync_from_mesh()

        elif self.umeshes.sync and not self.select :
            self.umeshes.elem_mode = 'FACE'

        hit.island.select = self.select
        umesh.update()

        return {'FINISHED'}


class UNIV_OT_SelectLinkedPick_VIEW3D(bpy.types.Macro):
    bl_idname = 'mesh.univ_select_linked_pick'
    bl_label = 'Select Linked Pick'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment


class UNIV_OT_DeselectLinkedPick_VIEW3D(bpy.types.Macro):
    bl_idname = 'mesh.univ_deselect_linked_pick'
    bl_label = 'Deselect Linked Pick'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment


class UNIV_OT_SelectLinked_VIEW3D(Operator):
    bl_idname = 'mesh.univ_select_linked'
    bl_label = 'Select Linked'
    bl_options = {'REGISTER', 'UNDO'}

    select: BoolProperty(name='Select', default=True)
    delimit: EnumProperty(name='Delimit', default=set(),  # noqa
                          items=(('NORMAL', 'Normal', ''),
                                 ('MATERIAL', 'Material', ''),
                                 ('SEAM', 'Seam', ''),
                                 ('SHARP', 'Sharp', ''),
                                 ('UV', 'UVs', '')
                                 ),
                          options={'ENUM_FLAG'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'select')
        self.layout.prop(self, 'delimit', expand=True)

    def execute(self, context):
        if self.select:
            return bpy.ops.mesh.select_linked(delimit=self.delimit)
        else:
            self.deselect()
            return {'FINISHED'}

    def deselect(self):
        # TODO: Add support wire edge
        umeshes = UMeshes.calc_any_unique(verify_uv=False)
        umeshes.set_sync(True)
        umeshes.filter_by_partial_selected_uv_faces()
        umeshes.update_tag = False

        match self.delimit:
            case delimit if "SEAM" in delimit:
                calc_type = MeshIslands.calc_with_markseam_non_manifold_iter_ex
            case delimit if "UV" in delimit:
                calc_type = Islands.calc_with_markseam_iter_ex
            case delimit if "MATERIAL" in delimit:
                calc_type = MeshIslands.calc_by_material_non_manifold_iter_ex
            case delimit if "SHARP" in delimit:
                calc_type = MeshIslands.calc_by_sharps_non_manifold_iter_ex
            case _:
                calc_type = MeshIslands.calc_iter_non_manifold_ex

        for umesh in umeshes:
            if self.delimit == 'UV' and not umesh.obj.data.uv_layers:
                calc_type_ = MeshIslands.calc_with_markseam_non_manifold_iter_ex
            else:
                calc_type_ = calc_type
            Islands.tag_filter_visible(umesh)
            for isl in calc_type_(umesh):
                if not utils.all_equal((f.select for f in isl)):
                    umeshes.elem_mode = 'FACE'
                    for f in isl:
                        f.select = False
                    umesh.update_tag = True
                    umesh.sync_valid = False
        umeshes.update()


# TODO: Grow after 0.3 (within 0.3-1.5 sec) sec and no effect repeat - without seam clamp
class UNIV_OT_Select_Grow_Base(Operator):
    bl_label = 'Grow'
    bl_options = {'REGISTER', 'UNDO'}

    grow: BoolProperty(name='Select', default=True)
    # TODO: Improve clamp
    clamp_on_seam: BoolProperty(name='Clamp on Seam', default=True,
                                description="Edge Grow clamp on edges with seam, but if the original edge has seam, this effect is ignored")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc_islands: Callable = Callable
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)

if utils.USE_GENERIC_UV_SYNC:
    class UNIV_OT_Select_Grow(UNIV_OT_Select_Grow_Base):
        bl_idname = 'uv.univ_select_grow'
        bl_description = "Select more UV vertices connected to initial selection\n\n" \
                         "Default - Grow\n" \
                         "Ctrl or Alt - Shrink\n\n" \
                         "Has [Ctrl + Scroll Up/Down] keymap"

        def execute(self, context):
            self.umeshes = UMeshes()
            self.umeshes.filter_by_partial_selected_uv_elem_by_mode()

            if self.grow:
                return self.grow_select()
            else:
                return self.shrink()

        def is_sticky_off_in_face_mode(self):
            return (self.umeshes.elem_mode == 'FACE' and
                    bpy.context.scene.tool_settings.uv_sticky_select_mode == 'DISABLED')

        def grow_select(self):
            has_update = False
            sync = self.umeshes.sync
            if self.clamp_on_seam:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
            else:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

            for umesh in self.umeshes:
                uv = umesh.uv
                to_select = []
                if self.is_sticky_off_in_face_mode():
                    face_select_get = utils.face_select_get_func(umesh)
                    face_select_set = utils.face_select_set_func(umesh)

                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if any(face_select_get(l_crn.face)
                               for crn in f.loops
                               for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                            to_select.append(f)
                    if to_select:
                        for f in to_select:
                            face_select_set(f, True)
                        umesh.update()
                        has_update = True
                    continue


                vert_select_get = utils.vert_select_get_func(umesh)
                face_select_get = utils.face_select_get_func(umesh)

                if self.umeshes.elem_mode == 'FACE':
                    # To optimize performance, the logic should be split based on whether
                    # there are many selected faces or just a few.
                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if any(face_select_get(l_crn.face)
                               for crn in f.loops if vert_select_get(crn)
                               for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                            to_select.append(f)
                else:
                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if any(vert_select_get(crn) for crn in f.loops):
                            to_select.append(f)

                if to_select:
                    umesh.sync_from_mesh_if_needed()
                    is_boundary = utils.is_boundary_func(umesh, with_seam=self.clamp_on_seam)
                    face_select_set = utils.face_select_set_func(umesh)
                    linked_faces = set()

                    for f in to_select:
                        face_select_set(f, True)
                        for crn in f.loops:
                            if not is_boundary(crn):
                                # The face_select_set func select the 3D edge and vert, so it does not need to be selected.
                                crn.link_loop_radial_prev.uv_select_edge = True
                            for linked_crn in linked_crn_to_vert_pair(crn, uv, sync):
                                linked_crn.uv_select_vert = True
                                linked_faces.add(linked_crn.face)

                    for f in linked_faces:
                        if not f.uv_select:
                            if all(crn.uv_select_vert for crn in f.loops):
                                face_select_set(f, True)

                    umesh.update()
                    has_update = True

            if not has_update:
                self.report({'INFO'}, f'Not found {self.umeshes.elem_mode.lower()} for grow')

            return {'FINISHED'}

        def shrink(self):
            has_update = False
            sync = self.umeshes.sync
            if self.clamp_on_seam:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
            else:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

            for umesh in self.umeshes:
                uv = umesh.uv
                to_deselect = set()
                if self.is_sticky_off_in_face_mode() or (umesh.elem_mode == 'FACE' and not umesh.sync_valid):
                    vert_select_get = utils.vert_select_get_func(umesh)
                    face_select_get = utils.face_select_get_func(umesh)
                    face_select_set = utils.face_select_set_func(umesh)

                    if sync:
                        for f in utils.calc_unselected_uv_faces_iter(umesh):
                            for crn in f.loops:
                                if crn.vert.select:
                                    selected_linked_faces = [l_crn.face for l_crn in linked_crn_to_vert_pair(crn, uv, sync) if face_select_get(l_crn.face)]
                                    if selected_linked_faces:
                                        to_deselect.add(f)
                                        to_deselect.update(selected_linked_faces)
                                    # Force deselect linked faces for rare wrong cases where the vertex is selected but the face is not.
                                    elif not any(face_select_get(ff) for ff in crn.vert.link_faces):
                                        # TODO: Test that
                                        to_deselect.update(crn.vert.link_faces)
                    else:
                        for f in utils.calc_unselected_uv_faces_iter(umesh):
                            for crn in f.loops:
                                linked_corners = linked_crn_to_vert_pair(crn, uv, sync)
                                selected_linked_faces = [l_crn.face for l_crn in linked_corners if face_select_get(l_crn.face)]
                                if selected_linked_faces:
                                    to_deselect.add(f)
                                    to_deselect.update(selected_linked_faces)
                                # Force deselect linked faces for rare wrong cases where the vertex is selected but the face is not.
                                elif vert_select_get(crn):
                                    # TODO: Test that
                                    to_deselect.add(f)
                                    to_deselect.update(l_crn.face for l_crn in linked_corners if not face_select_get(l_crn.face))

                    if to_deselect:
                        for f in to_deselect:
                            face_select_set(f, False)
                        umesh.update()
                        has_update = True
                    continue


                if self.umeshes.elem_mode == 'FACE':
                    # To optimize performance, the logic should be split based on whether
                    # there are many selected faces or just a few.
                    vert_select_get = utils.vert_select_get_func(umesh)
                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        for crn in f.loops:
                            if vert_select_get(crn):
                                to_deselect.add(f)
                                for l_crn in linked_crn_to_vert_pair(crn, uv, sync):
                                    if vert_select_get(l_crn):
                                        to_deselect.add(l_crn.face)
                    if to_deselect:
                        umesh.sync_from_mesh_if_needed()
                        face_select_set = utils.face_select_set_func(umesh)
                        for f in to_deselect:
                            face_select_set(f, False)
                        for f in to_deselect:
                            # Select linked verts.
                            for crn in f.loops:
                                crn.uv_select_vert = any(l_crn.face.uv_select for l_crn in linked_crn_to_vert_pair(crn, uv, sync))

                            # Select edges.
                            for crn in f.loops:
                                crn.uv_select_edge = crn.uv_select_vert and crn.link_loop_next.uv_select_vert
                        umesh.update()
                        has_update = True
                else:
                    for crn in utils.calc_selected_uv_vert_corners_iter(umesh):
                        if crn in to_deselect:
                            continue
                        if not crn.face.uv_select:
                            to_deselect.add(crn)
                            to_deselect.update(linked_crn_to_vert_pair(crn, uv, sync))

                        elif any(not f.uv_select for f in crn.vert.link_faces):  # Avoid using linked_crn_to_vert_pair if all faces selected
                            linked = linked_crn_to_vert_pair(crn, uv, sync)
                            if not all(l_crn.face.uv_select for l_crn in linked):
                                to_deselect.add(crn)
                                to_deselect.update(linked)

                    if to_deselect:
                        # TODO: Use foreach for non-clamped case
                        umesh.sync_from_mesh_if_needed()

                        for crn in to_deselect:
                            crn.uv_select_vert = False
                            crn.uv_select_edge = False
                            crn.link_loop_prev.uv_select_edge = False
                            crn.face.uv_select = False

                        if umesh.elem_mode == 'EDGE':
                            # Deselect verts without linked edges
                            for crn in to_deselect:
                                for crn_f in crn.face.loops:
                                    if not crn_f.uv_select_vert:
                                        continue
                                    if not crn_f.uv_select_edge and not crn_f.link_loop_prev.uv_select_edge:
                                        linked = linked_crn_to_vert_pair(crn, uv, sync)
                                        if not any(l_crn.uv_select_edge or l_crn.link_loop_prev.uv_select_edge for l_crn in linked):
                                            crn_f.uv_select_vert = False
                        if sync:
                            umesh.bm.uv_select_sync_to_mesh()
                        umesh.update()
                        has_update = True
            if not has_update:
                self.report({'INFO'}, f'Not found {self.umeshes.elem_mode.lower()} for shrink')

            return {'FINISHED'}
else:
    class UNIV_OT_Select_Grow(UNIV_OT_Select_Grow_Base):
        bl_idname = 'uv.univ_select_grow'
        bl_description = "Select more UV vertices connected to initial selection\n\n" \
                         "Default - Grow\n" \
                         "Ctrl or Alt - Shrink\n\n" \
                         "Has [Ctrl + Scroll Up/Down] keymap"

        def execute(self, context):
            self.umeshes = UMeshes()
            self.umeshes.filter_by_partial_selected_uv_elem_by_mode()

            # TODO: Implement without island calc (use linked with pair iter and mark seam)
            if self.clamp_on_seam:
                self.calc_islands = Islands.calc_visible_with_mark_seam
            else:
                self.calc_islands = Islands.calc_visible

            if self.grow:
                return self.grow_select()
            else:
                return self.shrink()

        def is_sticky_off_in_face_mode_non_sync(self):
            return (self.umeshes.elem_mode == 'FACE' and not self.umeshes.sync and
                    bpy.context.scene.tool_settings.uv_sticky_select_mode == 'DISABLED')

        def grow_select(self):
            sync = self.umeshes.sync
            if self.clamp_on_seam:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
            else:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

            for umesh in self.umeshes:
                uv = umesh.uv
                islands = self.calc_islands(umesh)
                islands.indexing()
                to_select = []
                if self.is_sticky_off_in_face_mode_non_sync():
                    face_select_get = utils.face_select_get_func(umesh)

                    for f in utils.calc_unselected_uv_faces_iter(umesh):
                        if any(face_select_get(l_crn.face)
                               for crn in f.loops
                               for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                            to_select.append(f)
                else:
                    # TODO: Remove calc island
                    for idx, isl in enumerate(islands):
                        if sync:
                            if self.umeshes.elem_mode == 'FACE':
                                # To optimize performance, the logic should be split based on whether
                                # there are many selected faces or just a few.
                                for f in isl:
                                    if not f.select and any(l_crn.face.select for crn in f.loops if crn.vert.select
                                                            for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx)):
                                        to_select.append(f)
                            else:
                                if umesh.is_full_face_deselected:  # optimize case when only one vertex/edge selected
                                    for f in isl:
                                        if any(v.select for v in f.verts):
                                            to_select.append(f)
                                else:
                                    for f in isl:
                                        if not f.select and self.is_grow_face(f, uv, idx):
                                            to_select.append(f)
                        else:
                            if self.umeshes.elem_mode == 'EDGE':
                                for f in isl:
                                    selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                    if selected_corners and selected_corners != len(f.loops):
                                        to_select.append(f)
                            else:
                                for f in isl:
                                    selected_corners = sum(crn[uv].select for crn in f.loops)
                                    if selected_corners and selected_corners != len(f.loops):
                                        to_select.append(f)
                if to_select:
                    if sync:
                        for f in to_select:
                            f.select = True
                    else:
                        if self.is_sticky_off_in_face_mode_non_sync():
                            face_select_set = utils.face_select_set_func(umesh)
                            for f in to_select:
                                face_select_set(f, True)
                        else:
                            for f in to_select:
                                idx = f.index
                                for crn in f.loops:
                                    if not crn[uv].select:
                                        for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                                            l_crn[uv].select = True
                            for isl in islands:
                                for f in isl:
                                    for crn in f.loops:
                                        if crn[uv].select and crn.link_loop_next[uv].select:
                                            crn[uv].select_edge = True
                    umesh.update()

            return {'FINISHED'}

        def shrink(self):
            sync = self.umeshes.sync
            if self.clamp_on_seam:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair_with_seam
            else:
                linked_crn_to_vert_pair = utils.linked_crn_to_vert_pair

            for umesh in self.umeshes:
                uv = umesh.uv
                islands = self.calc_islands(umesh)
                islands.indexing()
                to_deselect = []
                to_deselect_append = to_deselect.append

                if self.is_sticky_off_in_face_mode_non_sync():
                    face_select_get = utils.face_select_get_func(umesh)
                    for f in utils.calc_selected_uv_faces(umesh):
                        if not all(face_select_get(l_crn.face)
                                   for crn in f.loops
                                   for l_crn in linked_crn_to_vert_pair(crn, uv, sync)):
                            to_deselect_append(f)

                else:
                    for idx, isl in enumerate(islands):
                        if sync:
                            if self.umeshes.elem_mode == 'FACE':
                                for f in isl:
                                    if f.select and any(not l_crn.face.select for crn in f.loops
                                                        for l_crn in utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx)):
                                        to_deselect_append(f)
                            else:
                                if umesh.is_full_face_deselected:
                                    for f in isl:
                                        if not f.select:
                                            if any(v.select for v in f.verts):
                                                to_deselect_append(f)
                                else:
                                    for f in isl:
                                        if not f.select and self.is_shrink_face(f, uv, idx):
                                            to_deselect_append(f)
                        else:
                            if self.umeshes.elem_mode == 'EDGE':
                                for f in isl:
                                    selected_corners = sum(crn[uv].select_edge for crn in f.loops)
                                    if selected_corners and selected_corners != len(f.loops):
                                        to_deselect_append(f)
                            else:
                                for f in isl:
                                    selected_corners = sum(crn[uv].select for crn in f.loops)
                                    if selected_corners and selected_corners != len(f.loops):
                                        to_deselect_append(f)

                if to_deselect:
                    if sync:
                        if self.umeshes.elem_mode == 'FACE':
                            for f in to_deselect:
                                f.select = False
                        else:
                            for f in to_deselect:
                                f.select = False
                                for v in f.verts:
                                    v.select = False
                        umesh.bm.select_flush(False)
                    else:
                        if self.is_sticky_off_in_face_mode_non_sync():
                            face_select_set = utils.face_select_set_func(umesh)
                            for f in to_deselect:
                                face_select_set(f, False)
                        else:
                            # TODO: Implement face linked deselect with pair check and mark seam
                            for f in to_deselect:
                                idx = f.index
                                for crn in f.loops:
                                    if crn[uv].select:
                                        for l_crn in utils.linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                                            l_crn[uv].select = False

                            for isl in islands:
                                for f in isl:
                                    for crn in f.loops:
                                        if not (crn[uv].select and crn.link_loop_next[uv].select):
                                            crn[uv].select_edge = False

                    umesh.update()

            return {'FINISHED'}

        @staticmethod
        def is_grow_face(face: BMFace, uv, idx):
            for crn in face.loops:
                crn_vert = crn.vert
                if not crn_vert.select:
                    continue

                if len(utils.linked_crn_uv_by_island_index_unordered(crn, uv, idx)) + 1 == len(crn_vert.link_loops):
                    return True

                crn_edge = crn.edge

                if crn_edge.is_boundary and crn_edge.select:
                    return True

                if crn_vert.select and crn.link_loop_next.vert.select and not (crn_edge.seam or utils.is_boundary_sync(crn, uv)):
                    return True

            return False

        @staticmethod
        def handle_deselect_vertex(face: BMFace, idx):
            for v in face.verts:
                if v.select:
                    for ff in v.link_faces:
                        if ff.index not in (idx, -1):
                            if ff.select:
                                break
                    else:
                        v.select = False

        def is_shrink_face(self, face: BMFace, _uv, idx):
            has_selected_verts = False  # noqa
            for v in face.verts:
                if v.select:
                    has_selected_verts = True
                    if not v.is_boundary:
                        break
                    for ff in v.link_faces:
                        if ff.index not in (idx, -1):
                            break
            else:
                return True

            if has_selected_verts:
                for crn in face.loops:
                    if crn.vert.select:
                        for ff in crn.vert.link_faces:
                            if ff.index not in (idx, -1):
                                if ff.select:
                                    self.handle_deselect_vertex(face, idx)
                                    return False
                return True
            return False


class UNIV_OT_Select_Grow_VIEW3D(UNIV_OT_Select_Grow_Base):
    bl_idname = 'mesh.univ_select_grow'
    bl_description = "Select more vertices connected to initial selection\n\n" \
                     "Default - Grow\n" \
                     "Ctrl or Alt - Shrink\n\n" \
                     "Has [Ctrl + Scroll Up/Down] keymap"

    def execute(self, context):
        self.umeshes = UMeshes.calc_any_unique(verify_uv=False)

        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()
        if self.grow:
            return self.grow_select()
        else:
            return self.shrink()

    def grow_select(self):
        has_updates = False
        linked_crn_to_vert = utils.linked_crn_to_vert_with_seam_3d_iter if self.clamp_on_seam else utils.linked_crn_to_vert_3d_iter
        if self.umeshes.elem_mode == 'VERT':
            self.umeshes.filter_by_selected_mesh_verts()

            for umesh in self.umeshes:
                if umesh.is_full_vert_selected:
                    continue

                to_select = set()
                for v in umesh.bm.verts:
                    if not v.select:
                        continue
                    if v.is_wire:
                        to_select.update(ee
                                         for ee in v.link_edges
                                         if ee.is_wire and not ee.select and not ee.hide)
                    else:
                        selection_states_from_linked_faces = [f.select for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue

                        elif any(selection_states_from_linked_faces):
                            all_linked_faces_with_select = []
                            all_linked_faces_without_select = []

                            link_corners_to_vert = {crn for crn in v.link_loops if not crn.face.hide}
                            while link_corners_to_vert:
                                crn = link_corners_to_vert.pop()

                                linked_corners = set(crn_ for crn_ in linked_crn_to_vert(crn))
                                link_corners_to_vert -= linked_corners

                                faces = [crn_.face for crn_ in linked_corners]
                                faces.append(crn.face)

                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                            if all_linked_faces_with_select:
                                for faces in all_linked_faces_with_select:
                                    to_select.update(f for f in faces if not f.select)
                            else:
                                for faces in all_linked_faces_without_select:
                                    to_select.update(faces)

                        else:  # extend all visible unselected
                            for f in v.link_faces:
                                if not f.hide:
                                    to_select.add(f)
                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.sync_valid = False
                    umesh.bm.select_flush(True)
                    umesh.update()

        elif self.umeshes.elem_mode == 'EDGE':
            self.umeshes.filter_by_selected_mesh_edges()

            for umesh in self.umeshes:
                if umesh.is_full_edge_selected:
                    continue

                to_select = set()
                for e in umesh.bm.edges:
                    if not e.select:
                        continue
                    if e.is_wire:
                        to_select.update(ee
                                         for v in e.verts for ee in v.link_edges
                                         if ee.is_wire and not ee.select and not ee.hide)
                    else:
                        selection_states_from_linked_faces = [f.select for v in e.verts for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue
                        elif any(selection_states_from_linked_faces):
                            all_linked_faces_with_select = []
                            all_linked_faces_without_select = []

                            for crn in e.link_loops:
                                if crn.face.hide:
                                    continue
                                faces = list(crn_.face for crn_ in linked_crn_to_vert(crn))
                                faces.append(crn.face)

                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                                # Do not combine crn and crn.next in "faces", otherwise grow becomes redundant
                                faces = [crn_.face for crn_ in linked_crn_to_vert(crn.link_loop_next)]
                                if any(f.select for f in faces):
                                    all_linked_faces_with_select.append(faces)
                                else:
                                    all_linked_faces_without_select.append(faces)

                            if all_linked_faces_with_select:
                                for faces in all_linked_faces_with_select:
                                    to_select.update(f for f in faces if not f.select)
                            else:
                                for faces in all_linked_faces_without_select:
                                    to_select.update(faces)

                        else:  # extend all visible unselected
                            for crn in e.link_loops:
                                to_select.update(crn_.face for crn_ in linked_crn_to_vert(crn))
                                to_select.update(crn_.face for crn_ in linked_crn_to_vert(crn.link_loop_next))
                                if not crn.face.hide:
                                    to_select.add(crn.face)
                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.sync_valid = False
                    umesh.bm.select_flush(True)
                    umesh.update()
        else:
            self.umeshes.filter_by_selected_mesh_faces()

            for umesh in self.umeshes:
                if umesh.is_full_face_selected:
                    continue

                to_select = set()
                for f in umesh.bm.faces:
                    if not f.select:
                        continue
                    for crn in f.loops:
                        if all(ff.select for ff in crn.vert.link_faces):  # x2.5 performance
                            continue
                        to_select.update(crn_.face
                                         for crn_ in linked_crn_to_vert(crn)
                                         if not crn_.face.select)

                for f in to_select:
                    f.select = True

                if to_select:
                    has_updates = True
                    umesh.sync_valid = False
                    umesh.update()

        if not has_updates:
            self.report({'INFO'}, 'Not found faces for grow select')
        return {'FINISHED'}

    def shrink(self):
        has_updates = False

        linked_crn_to_vert = utils.linked_crn_to_vert_with_seam_3d_iter if self.clamp_on_seam else utils.linked_crn_to_vert_3d_iter
        if self.umeshes.elem_mode == 'VERT':
            self.umeshes.filter_by_selected_mesh_verts()

            for umesh in self.umeshes:
                if umesh.is_full_vert_selected:
                    continue

                to_deselect = set()
                for v in umesh.bm.verts:
                    if not v.select:
                        continue
                    if v.is_wire:
                        if any(ee.is_wire and not ee.select and not ee.hide for ee in v.link_edges):
                            to_deselect.add(v)
                    else:
                        selection_states_from_linked_faces = [f.select for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue

                        elif any(selection_states_from_linked_faces):
                            link_corners_to_vert = {crn for crn in v.link_loops if not crn.face.hide}
                            while link_corners_to_vert:
                                crn = link_corners_to_vert.pop()

                                linked_corners = set(crn_ for crn_ in linked_crn_to_vert(crn))
                                link_corners_to_vert -= linked_corners

                                if crn.face.select and all(crn_.face.select for crn_ in linked_corners):
                                    break
                            else:  # not break
                                to_deselect.add(v)

                        else:  # shrink all visible unselected
                            to_deselect.add(v)
                for v in to_deselect:
                    v.select = False

                if to_deselect:
                    has_updates = True
                    umesh.bm.select_flush(False)
                    umesh.update()

        elif self.umeshes.elem_mode == 'EDGE':
            self.umeshes.filter_by_selected_mesh_edges()

            for umesh in self.umeshes:
                if umesh.is_full_edge_selected:
                    continue

                to_deselect = set()
                for e in umesh.bm.edges:
                    if not e.select:
                        continue
                    if e.is_wire:
                        if any(ee.is_wire and not ee.select and not ee.hide
                               for v in e.verts for ee in v.link_edges):
                            to_deselect.add(e)
                    else:
                        selection_states_from_linked_faces = [f.select for v in e.verts for f in v.link_faces]
                        if all(selection_states_from_linked_faces):
                            continue
                        elif any(selection_states_from_linked_faces):
                            for crn in e.link_loops:
                                if crn.face.hide:
                                    continue
                                if crn.face.select:
                                    if all(crn_.face.select for crn_ in linked_crn_to_vert(crn)) or \
                                            all(crn_.face.select for crn_ in linked_crn_to_vert(crn.link_loop_next)):
                                        break
                            else:  # not break
                                to_deselect.add(e)
                        else:  # shrink all visible unselected
                            to_deselect.add(e)
                for e in to_deselect:
                    e.select = False

                if to_deselect:
                    has_updates = True
                    umesh.bm.select_flush(False)

                    for e in to_deselect:
                        if e.is_wire:
                            continue
                        for v in e.verts:
                            for ee in v.link_edges:
                                if not ee.select or ee.is_wire:
                                    continue
                                if all(not f.select for f in ee.link_faces):
                                    ee.select = False

                    umesh.update()
        else:
            self.umeshes.filter_by_selected_mesh_faces()

            for umesh in self.umeshes:
                if umesh.is_full_face_selected:
                    continue

                to_deselect = set()
                for f in utils.calc_selected_uv_faces(umesh):
                    for crn in f.loops:
                        if all(ff.select for ff in crn.vert.link_faces):  # x2.5 performance
                            continue
                        if any(not crn_.face.select for crn_ in linked_crn_to_vert(crn)):
                            to_deselect.add(f)
                            break

                for f in to_deselect:
                    f.select = False

                if to_deselect:
                    has_updates = True
                    umesh.update()

        if not has_updates:
            self.report({'INFO'}, 'Not found faces for shrink deselect')
        return {'FINISHED'}


class UNIV_OT_Select_Edge_Grow_Base(Operator):
    bl_label = 'Edge Grow'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Edge Grow/Shrink Select\n\n" \
                     "Default - Grow Select \n" \
                     "Ctrl or Alt - Shrink Select\n\n" \
                     "Has [Alt + Scroll Up/Down] keymap, but it conflicts with the Frame Offset operator"

    clamp_on_seam: BoolProperty(name='Clamp on Seam', default=True,
                                description="Edge Grow clamp on edges with seam, but if the original edge has seam, this effect is ignored")
    grow: BoolProperty(name='Select', default=True, description='Grow/Shrink toggle')
    max_angle: FloatProperty(name='Angle', default=math.radians(20), min=math.radians(1), soft_min=math.radians(5), max=math.radians(90), subtype='ANGLE',
                             description="Max select angle.")
    prioritize_sharps: BoolProperty(name='Prioritize Sharps', default=True,
                                    description='Gives 35% priority to an edge that has a Mark Sharp, works if there are more than 4 linked edges.')
    boundary_by_boundary: BoolProperty(name='Boundary by Boundary', default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calc_islands: Callable = Callable
        self.umeshes: UMeshes | None = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.grow = not (event.ctrl or event.alt)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.grow:
            layout.prop(self, 'prioritize_sharps')
        layout.prop(self, 'boundary_by_boundary')
        layout.prop(self, 'clamp_on_seam')
        layout.prop(self, 'max_angle')
        layout.prop(self, 'grow')


class UNIV_OT_Select_Edge_Grow_VIEW2D(UNIV_OT_Select_Edge_Grow_Base):
    bl_idname = 'uv.univ_select_edge_grow'

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)

        if self.umeshes.elem_mode not in ('VERT', 'EDGE'):
            self.report({'INFO'}, f'Edge Grow not work in "{self.umeshes.elem_mode}" mode, run grow instead')
            return bpy.ops.uv.univ_select_grow(grow=self.grow, clamp_on_seam=self.clamp_on_seam)  # noqa

        if self.clamp_on_seam:
            self.calc_islands = Islands.calc_extended_any_edge_with_markseam
        else:
            self.calc_islands = Islands.calc_extended_any_edge

        self.umeshes.filter_by_selected_uv_edges()
        self.umeshes.update_tag = False

        if self.grow:
            self.grow_select()
            self.umeshes.update(info='Not found edges for grow select')
            return {'FINISHED'}

        self.shrink_select()
        # TODO: Deselect single vert in VERTEX mode when repeat press operator (when prev deselect failed)
        self.umeshes.update(info='Not found edges for shrink select')
        return {'FINISHED'}

    def grow_select(self):
        # TODO: Remove calc islands
        self.umeshes.update_tag = False
        for umesh in self.umeshes:
            islands = self.calc_islands(umesh)
            islands.indexing()
            is_clamped = self.is_clamped_by_selected_and_seams_func(umesh)
            grew = []
            uv = umesh.uv
            for isl in islands:
                for crn in isl.calc_selected_edge_corners_iter():
                    with_seam_clamp = self.clamp_on_seam and crn.edge.seam
                    selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                    if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam_clamp, is_clamped):
                        if not (with_seam_clamp and grow_prev_crn.edge.seam):
                            grew.append(grow_prev_crn)

                    if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam_clamp, is_clamped):
                        if not (with_seam_clamp and grow_next_crn.edge.seam):
                            grew.append(grow_next_crn)

            if grew:
                if utils.USE_GENERIC_UV_SYNC:
                    umesh.sync_from_mesh_if_needed()
                    for grew_crn in grew:
                        utils.select_crn_uv_edge_with_shared_by_idx(grew_crn, uv, force=True)

                    # Select linked faces.
                    if umesh.elem_mode == 'VERT':
                        set_face_select = utils.face_select_set_func(umesh)
                        for grew_crn in grew:
                            for l_crn in utils.linked_crn_uv_by_idx_unordered_included(grew_crn, uv):
                                f = l_crn.face
                                if not f.uv_select:
                                    if all(crn_f.uv_select_vert for crn_f in f.loops):
                                        set_face_select(f, True)
                                    else:
                                        for crn_f in f.loops:
                                            if crn_f.uv_select_vert and crn_f.link_loop_next.uv_select_vert:
                                                crn_f.edge.select = True
                                                crn_f.uv_select_edge = True
                else:
                    if umesh.sync:
                        for grew_crn in grew:
                            grew_crn.edge.select = True
                    else:
                        for grew_crn in grew:
                            utils.select_crn_uv_edge_with_shared_by_idx(grew_crn, uv, force=True)
                umesh.update_tag = True

    def shrink_select(self):
        for umesh in self.umeshes:
            islands = self.calc_islands(umesh)
            islands.indexing()

            is_clamped = self.is_clamped_by_selected_and_seams_func(umesh)
            uv = umesh.uv
            shrink = []
            for isl in islands:
                for crn in isl.calc_selected_edge_corners_iter():
                    with_seam_clamp = self.clamp_on_seam and crn.edge.seam
                    selected_dir = crn.link_loop_next[uv].uv - crn[uv].uv

                    if grow_prev_crn := self.grow_prev(crn, selected_dir, uv, self.max_angle, with_seam_clamp, is_clamped):
                        if not (with_seam_clamp and grow_prev_crn.edge.seam):
                            shrink.append(crn)
                            continue

                    if grow_next_crn := self.grow_next(crn, selected_dir, uv, self.max_angle, with_seam_clamp, is_clamped):
                        if not (with_seam_clamp and grow_next_crn.edge.seam):
                            shrink.append(crn)

            if shrink:
                if utils.USE_GENERIC_UV_SYNC:
                    umesh.sync_from_mesh_if_needed()

                    umesh.bm.uv_select_foreach_set(False, loop_edges=shrink)
                    if umesh.sync:
                        umesh.bm.uv_select_sync_to_mesh()
                    # TODO: Don't forget to use use clam_by_seams
                    # edge_select_set = utils.edge_select_linked_set_func(umesh)
                    # for crn in shrink:
                    #     edge_select_set(crn, False)

                else:
                    if umesh.sync:
                        edge_deselect = utils.edge_deselect_safe_3d_func(umesh)
                        for crn in shrink:
                            edge_deselect(crn)
                        umesh.bm.select_history.validate()  # Active elem validate
                    else:
                        edge_select_set = utils.edge_select_linked_set_func(umesh)
                        for crn in shrink:
                            edge_select_set(crn, False)
                umesh.update_tag = True

    def grow_prev(self, crn, selected_dir, uv, max_angle, with_seam_clamp, is_clamped) -> 'BMLoop | None | False':
        prev_crn = crn.link_loop_prev
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        cur_linked_corners = utils.linked_crn_uv_by_island_index_unordered(crn, uv, crn.face.index)

        if is_clamped(cur_linked_corners, shared, prev_crn, with_seam_clamp):
            return None

        if not len(cur_linked_corners):
            if selected_dir.angle(crn[uv].uv - prev_crn[uv].uv, max_angle) <= max_angle:
                return prev_crn
        elif len(cur_linked_corners) == 3 and len(crn.vert.link_loops) == 4 \
                and shared \
                and len(cur_quad_linked_crn_uv := utils.linked_crn_uv_by_idx(crn, uv)) == 3 \
                and utils.shared_linked_crn_by_idx(cur_quad_linked_crn_uv[1], uv):  # noqa # pylint:disable=used-before-assignment
            return cur_quad_linked_crn_uv[1]
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in cur_linked_corners:
                angle_ = selected_dir.angle(crn_[uv].uv - crn_.link_loop_next[uv].uv, max_angle)

                if self.prioritize_sharps:
                    if not crn.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_by_idx(crn_, uv))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(crn_[uv].uv - prev_crn_[uv].uv, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_by_idx(prev_crn_, uv))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_

            return min_crn
        return False

    def grow_next(self, crn, selected_dir, uv, max_angle, with_seam_clamp, is_clamped) -> 'BMLoop | None | False':
        next_crn = crn.link_loop_next
        shared = utils.shared_linked_crn_by_idx(crn, uv)
        next_linked_corners = utils.linked_crn_uv_by_island_index_unordered(
            crn.link_loop_next, uv, crn.link_loop_next.face.index)

        if is_clamped(next_linked_corners, shared, next_crn, with_seam_clamp):
            return None

        if not len(next_linked_corners):
            if selected_dir.angle(next_crn.link_loop_next[uv].uv - next_crn[uv].uv, max_angle) <= max_angle:
                return next_crn

        elif len(next_linked_corners) == 3 and len(next_crn.vert.link_loops) == 4 \
                and shared \
                and len(next_quad_linked_crn_uv := utils.linked_crn_uv_by_idx(next_crn, uv)) == 3 \
                and utils.shared_linked_crn_by_idx(next_quad_linked_crn_uv[1].link_loop_prev, uv):  # noqa # pylint:disable=used-before-assignment
            return next_quad_linked_crn_uv[1].link_loop_prev
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in next_linked_corners:
                angle_ = selected_dir.angle(crn_.link_loop_next[uv].uv - crn_[uv].uv, max_angle)

                if self.prioritize_sharps:
                    if not crn.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_by_idx(crn_, uv))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(prev_crn_[uv].uv - crn_[uv].uv, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_by_idx(prev_crn_, uv))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_
            return min_crn
        return False

    @staticmethod
    def is_clamped_by_selected_and_seams_func(umesh):
        """Skip if selected or with seam"""
        def catcher(get_edge_select):
            def fn(linked_corners, shared, next_or_prev_crn, with_seam_clamp):
                if get_edge_select(next_or_prev_crn):
                    return True

                if with_seam_clamp:
                    if next_or_prev_crn.edge.seam:
                        return True
                    for crn in linked_corners:
                        if get_edge_select(crn) or crn.edge.seam:
                            return True
                        if (prev_crn := crn.link_loop_prev) != shared:
                            if prev_crn.edge.seam or get_edge_select(prev_crn):
                                return True
                else:
                    for crn in linked_corners:
                        if get_edge_select(crn):
                            return True
                        if (prev_crn := crn.link_loop_prev) != shared:
                            if get_edge_select(prev_crn):
                                return True

            return fn
        return catcher(utils.edge_select_get_func(umesh))


class UNIV_OT_Select_Edge_Grow_VIEW3D(UNIV_OT_Select_Edge_Grow_Base):
    bl_idname = 'mesh.univ_select_edge_grow'

    max_angle: FloatProperty(name='Angle', default=math.radians(40), min=math.radians(1), soft_min=math.radians(5), max=math.radians(90), subtype='ANGLE',
                             description="Max select angle.")

    def execute(self, context):
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)

        if self.umeshes.elem_mode not in ('VERT', 'EDGE'):
            return bpy.ops.mesh.univ_select_grow(grow=self.grow, clamp_on_seam=self.clamp_on_seam)  # noqa

        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()
        if self.clamp_on_seam:
            self.calc_islands = MeshIslands.calc_extended_any_edge_with_markseam
        else:
            self.calc_islands = MeshIslands.calc_extended_any_edge

        self.umeshes.update_tag = False
        if self.grow:
            self.grow_select()
            self.umeshes.update(info='Not found edges for grow select')
            return {'FINISHED'}

        self.shrink_select()
        self.umeshes.update(info='Not found edges for shrink select')
        return {'FINISHED'}

    def grow_select(self):
        for umesh in reversed(self.umeshes):
            if islands := self.calc_islands(umesh):
                islands.indexing()
                grew = []
                for isl in islands:
                    for crn in isl.calc_selected_edge_corners_iter():
                        with_seam = not self.clamp_on_seam or crn.edge.seam
                        selected_dir = crn.link_loop_next.vert.co - crn.vert.co

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam:
                                if grow_prev_crn.edge.seam:
                                    continue
                            grew.append(grow_prev_crn)

                        if grow_next_crn := self.grow_next(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam:
                                if grow_next_crn.edge.seam:
                                    continue
                            grew.append(grow_next_crn)

                for grew_crn in grew:
                    grew_crn.edge.select = True
                if grew:
                    umesh.sync_valid = False
                umesh.update_tag = bool(grew)

    def shrink_select(self):
        for umesh in self.umeshes:
            if islands := self.calc_islands(umesh):
                islands.indexing()
                shrink = []
                for isl in islands:
                    for crn in isl.calc_selected_edge_corners_iter():
                        with_seam = not self.clamp_on_seam or crn.edge.seam
                        selected_dir = crn.link_loop_next.vert.co - crn.vert.co

                        if grow_prev_crn := self.grow_prev(crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam and grow_prev_crn.edge.seam:
                                grow_prev_crn = None

                        if grow_next_crn := self.grow_next(
                                crn, selected_dir, self.max_angle, with_seam, self.is_clamped_by_selected_and_seams):
                            if not with_seam and grow_next_crn.edge.seam:
                                grow_next_crn = None

                        if grow_prev_crn or grow_next_crn:
                            shrink.append(crn)

                if shrink:
                    umesh.sync_valid = False
                    edge_deselect = utils.edge_deselect_safe_3d_func(umesh)
                    for crn in shrink:
                        edge_deselect(crn)
                    umesh.bm.select_history.validate()
                    umesh.update_tag = True

    def grow_prev(self, crn, selected_dir, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        prev_crn = crn.link_loop_prev
        shared = utils.shared_linked_crn_to_edge_by_idx(crn)
        cur_linked_corners = utils.linked_crn_to_vert_by_island_index_unordered(crn)

        if is_clamped(cur_linked_corners, shared, prev_crn, with_seam):
            return None

        if not len(cur_linked_corners):
            if selected_dir.angle(crn.vert.co - prev_crn.vert.co, max_angle) <= max_angle:
                return prev_crn
        elif len(cur_linked_corners) == 3 and len(crn.vert.link_loops) == 4 \
                and shared \
                and len(cur_quad_linked_crn_uv := utils.linked_crn_to_vert_by_idx_3d(crn)) == 3 \
                and utils.shared_linked_crn_to_edge_by_idx(cur_quad_linked_crn_uv[1]):  # noqa # pylint:disable=used-before-assignment
            return cur_quad_linked_crn_uv[1]
        # TODO: Implement border and border with quad
        # elif not shared and len(cur_linked_corners) == 1 \
        # and (shared_prev_crn := utils.shared_linked_crn_to_edge_by_idx(prev_crn))
        else:
            # TODO: Implement angles by normal projection
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in cur_linked_corners:
                angle_ = selected_dir.angle(crn_.vert.co - crn_.link_loop_next.vert.co, max_angle)
                if self.prioritize_sharps:
                    if not crn_.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(crn_))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(crn_.vert.co - prev_crn_.vert.co, max_angle)
                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(prev_crn_))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_

            return min_crn
        return False

    def grow_next(self, crn, selected_dir, max_angle, with_seam, is_clamped) -> 'BMLoop | None | False':
        next_crn = crn.link_loop_next
        shared = utils.shared_linked_crn_to_edge_by_idx(crn)
        next_linked_corners = utils.linked_crn_to_vert_by_island_index_unordered(next_crn)

        if is_clamped(next_linked_corners, shared, next_crn, with_seam):
            return None

        if not len(next_linked_corners):
            if selected_dir.angle(next_crn.link_loop_next.vert.co - next_crn.vert.co, max_angle) <= max_angle:
                return next_crn

        elif len(next_linked_corners) == 3 and len(next_crn.vert.link_loops) == 4 \
                and shared \
                and len(next_quad_linked_crn_uv := utils.linked_crn_to_vert_by_idx_3d(next_crn)) == 3 \
                and utils.shared_linked_crn_to_edge_by_idx(next_quad_linked_crn_uv[1].link_loop_prev):  # noqa # pylint:disable=used-before-assignment
            return next_quad_linked_crn_uv[1].link_loop_prev
        else:
            min_crn = None
            angle = max_angle * 1.0001
            for crn_ in next_linked_corners:
                angle_ = selected_dir.angle(crn_.link_loop_next.vert.co - crn_.vert.co, max_angle)

                if self.prioritize_sharps:
                    if not crn_.edge.smooth and angle_ <= max_angle:
                        angle_ *= 0.65

                if angle_ < angle:
                    if self.boundary_by_boundary:
                        status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(crn_))
                        if bool(shared) is status_grow:
                            angle = angle_
                            min_crn = crn_
                    else:
                        angle = angle_
                        min_crn = crn_

                if (prev_crn_ := crn_.link_loop_prev) != shared:
                    angle_ = selected_dir.angle(prev_crn_.vert.co - crn_.vert.co, max_angle)

                    if self.prioritize_sharps:
                        if not prev_crn_.edge.smooth and angle_ <= max_angle:
                            angle_ *= 0.65

                    if angle_ < angle:
                        if self.boundary_by_boundary:
                            status_grow = bool(utils.shared_linked_crn_to_edge_by_idx(prev_crn_))
                            if bool(shared) is status_grow:
                                angle = angle_
                                min_crn = prev_crn_
                        else:
                            angle = angle_
                            min_crn = prev_crn_
            return min_crn
        return False

    @staticmethod
    def is_clamped_by_selected_and_seams(linked_corners, shared, next_or_prev_crn, with_seam):
        # Skip if selected or with seam
        if next_or_prev_crn.edge.select:
            return True

        if with_seam:
            for crn in linked_corners:
                if crn.edge.select:
                    return True
                if (prev_crn__ := crn.link_loop_prev) != shared:
                    if prev_crn__.edge.select:
                        return True
        else:
            if next_or_prev_crn.edge.seam:
                return True
            for crn in linked_corners:
                crn_edge = crn.edge
                if crn_edge.select or crn_edge.seam:
                    return True
                if (prev_crn__ := crn.link_loop_prev) != shared:
                    prev_crn_edge = prev_crn__.edge
                    if prev_crn_edge.seam or prev_crn_edge.select:
                        return True


class UNIV_OT_SelectTexelDensity_VIEW3D(Operator):
    bl_idname = "mesh.univ_select_texel_density"
    bl_label = 'Select by TD'
    bl_description = "Select by Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    compare_type: EnumProperty(name='Compare Type', default='LESS', items=(
        ('LESS', 'Less', ''),
        ('EQUAL', 'Equal', ''),
        ('GREATER', 'Greater', ''),
    ))
    island_mode: EnumProperty(name='Mode', default='ISLAND', items=(('ISLAND', 'Island', ''), ('FACE', 'Face', '')))

    target_texel: FloatProperty(name='Texel', default=512, min=1, soft_min=32, soft_max=2048, max=10_000)
    threshold: FloatProperty(name='Threshold', default=0.01, min=0, soft_min=0.0001, soft_max=50, max=10_000)

    def draw(self, context):
        layout = self.layout
        row = self.layout.row(align=True)
        row.prop(self, 'mode', expand=True)
        row = self.layout.row(align=True)
        row.prop(self, 'island_mode', expand=True)
        layout.prop(self, 'target_texel', slider=True)
        layout.prop(self, 'threshold', slider=True)
        layout.row(align=True).prop(self, 'compare_type', expand=True)

    def invoke(self, context, event):
        self.target_texel = univ_settings().texel_density

        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        self.island_mode = 'FACE' if event.alt else 'ISLAND'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        umeshes = UMeshes()
        umeshes.update_tag = False
        need_sync_validation_check = False

        if not self.bl_idname.startswith('UV'):
            umeshes.set_sync()
            umeshes.sync_invalidate()
            umeshes.elem_mode = 'FACE'
        else:
            if umeshes.sync:
                if utils.USE_GENERIC_UV_SYNC:
                    need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')

        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        umeshes.filter_by_visible_uv_faces()

        counter = 0
        counter_skipped = 0
        for umesh in umeshes:
            if self.island_mode == 'ISLAND':
                islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            else:
                islands = [AdvIsland([f], umesh) for f in utils.calc_visible_uv_faces(umesh)]

            scale = umesh.check_uniform_scale(self.report)
            to_select = []
            to_deselect = []
            for isl in islands:
                isl.calc_area_3d(scale)
                isl.calc_area_uv()

                area_3d = sqrt(isl.area_3d)
                area_uv = sqrt(isl.area_uv) * texture_size

                texel = area_uv / area_3d if area_3d else 0

                if not (compared_result := isclose(texel, self.target_texel, abs_tol=self.threshold)):
                    if self.compare_type == 'LESS':
                        compared_result = texel < self.target_texel
                    elif self.compare_type == 'GREATER':
                        compared_result = texel > self.target_texel

                if self.mode == 'SELECT':
                    if compared_result:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        to_select.append(isl)
                    elif not isl.is_full_deselected_by_context():
                        to_deselect.append(isl)
                elif self.mode == 'ADDITION':
                    if compared_result:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        to_select.append(isl)
                else:  # self.mode == 'DESELECT':
                    if compared_result:
                        if isl.is_full_face_deselected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        to_deselect.append(isl)

            if to_select or to_deselect:
                umesh.update_tag = True
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()
                elif umesh._sync_invalidate:  # noqa
                    umesh.sync_valid = False

                for isl in to_deselect:
                    isl.select = False

                for isl in to_select:
                    isl.select = True

                if need_sync_validation_check and to_deselect:
                    umesh.bm.uv_select_sync_to_mesh()

        if not umeshes:
            self.report({'WARNING'}, f'{self.island_mode.capitalize() + "s"} not found')
        else:
            if not counter and not counter_skipped:
                self.report({'WARNING'}, f'No found {self.island_mode.capitalize() + "s"} in the specified texel')
        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_SelectTexelDensity(UNIV_OT_SelectTexelDensity_VIEW3D):
    bl_idname = "uv.univ_select_texel_density"


class UNIV_OT_Tests(utils.UNIV_OT_Draw_Test):
    def test_invoke(self, _event):

        # from .. import univ_pro
        umesh = self.umeshes[0]
        uv = umesh.uv

        islands = AdvIslands.calc_visible(umesh)
        islands.indexing()

        if lgs := utypes.LoopGroup.calc_dirt_loop_groups(umesh):
            # umesh.tag_visible_corners()
            # for lg in lgs:
            #     lg.extend_from_linked()

            self.calc_from_corners(lgs, uv)

        umesh.update()


class UNIV_OT_SelectByArea(Operator):
    bl_idname = "uv.univ_select_by_area"
    bl_label = 'Select by Area'
    bl_description = "Select by Area"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    size_mode: EnumProperty(name='Size Mode', default='SMALL', items=(
        ('SMALL', 'Small', ''),
        ('MEDIUM', 'Medium', ''),
        ('LARGE', 'Large', ''),
    ))
    size_type: EnumProperty(name='Size Mode', default='AREA', items=(
        ('AREA', 'Area', ''),
        ('X', 'Size X', ''),
        ('Y', 'Size Y', ''),
    ))

    threshold: FloatProperty(name='Threshold', default=0.005, min=0, soft_min=0.005, max=0.5, subtype='FACTOR')
    lower_slider: FloatProperty(name='Low', default=0.1, min=0, max=0.9, subtype='PERCENTAGE',
                                update=lambda self, _: setattr(self, 'higher_slider', self.lower_slider+0.05) if self.higher_slider-0.05 < self.lower_slider else None)
    higher_slider: FloatProperty(name='High', default=0.8, min=0.1, max=1, subtype='PERCENTAGE',
                                 update=lambda self, _: setattr(self, 'lower_slider', self.higher_slider-0.05) if self.higher_slider-0.05 < self.lower_slider else None)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'mode', expand=True)
        layout.row(align=True).prop(self, 'size_type', expand=True)
        layout.row(align=True).prop(self, 'size_mode', expand=True)

        row = layout.row(align=True)
        row.label(text=f'   Small: {self.lower_slider*100:.2f}%')
        row.label(text=f'  Medium{((self.higher_slider-self.lower_slider)*100):.2f}%')
        row.label(text=f'   Large {(1-self.higher_slider)*100:.2f}%')

        row = layout.row(align=True)
        row.prop(self, 'lower_slider', slider=True)
        row.prop(self, 'higher_slider', slider=True)
        layout.prop(self, 'threshold', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = UMeshes()

        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        if self.mode == 'SELECT':
            umeshes.filter_by_visible_uv_faces()
        elif self.mode == 'ADDITIONAL':
            umeshes.filter_by_visible_uv_faces()
        else: # self.mode == 'DESELECT':
            if utils.USE_GENERIC_UV_SYNC:
                umeshes.filter_by_selected_uv_verts()
            else:
                umeshes.filter_by_selected_uv_faces()

        min_value = float('inf')
        max_value = float('-inf')
        islands_of_mesh = []

        for umesh in umeshes:
            umesh.update_tag = False
            if islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                islands_of_mesh.append(islands)

                if self.size_type == 'AREA':
                    for isl in islands:
                        area = isl.calc_area_uv()
                        isl.value = area
                        min_value = min(area, min_value)
                        max_value = max(area, max_value)
                else:
                    for isl in islands:
                        bbox = isl.calc_bbox()
                        if self.size_type == 'X':
                            size = bbox.width
                        else:
                            size = bbox.height
                        isl.value = size
                        min_value = min(size, min_value)
                        max_value = max(size, max_value)

        if self.size_mode == 'SMALL':
            lower = min_value
            higher = lerp(min_value, max_value, self.lower_slider)
        elif self.size_mode == 'MEDIUM':
            lower = lerp(min_value, max_value, self.lower_slider)
            higher = lerp(min_value, max_value, self.higher_slider)
        else:  # self.size_mode == 'LARGE':
            lower = lerp(min_value, max_value, self.higher_slider)
            higher = max_value
        lower -= lower * self.threshold
        higher += higher * self.threshold

        for islands in islands_of_mesh:
            umesh = islands.umesh
            to_select = []
            to_deselect = []

            for isl in islands:
                if self.mode == 'SELECT':
                    if lower <= isl.value <= higher:
                        if isl.is_full_face_selected():
                            continue
                        to_select.append(isl)
                    else:
                        if isl.is_full_deselected_by_context():
                            continue
                        to_deselect.append(isl)

                elif self.mode == 'ADDITION':
                    if lower <= isl.value <= higher:
                        if isl.is_full_face_selected():
                            continue
                        to_select.append(isl)
                else:  # self.mode == 'DESELECT':
                    if lower <= isl.value <= higher:
                        if isl.is_full_deselected_by_context():
                            continue
                        to_deselect.append(isl)

            islands.umesh.update_tag = (to_select or to_deselect)
            if islands.umesh.update_tag:
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()

                for isl in to_deselect:
                    isl.select = False

                for isl in to_select:
                    isl.select = True

                if need_sync_validation_check and to_deselect:
                    umesh.bm.uv_select_sync_to_mesh()

        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_Stacked(Operator):
    bl_idname = "uv.univ_select_stacked"
    bl_label = 'Stacked'
    bl_description = "Select exact overlapped islands"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))

    threshold: bpy.props.FloatProperty(name='Distance', default=0.001, min=0.0, soft_min=0.00005, soft_max=0.00999)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'mode', expand=True)
        layout.prop(self, 'threshold', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        umeshes = UMeshes()

        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        umeshes.filter_by_visible_uv_faces()

        all_islands = []
        for umesh in umeshes:
            umesh.update_tag = False
            islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            all_islands.extend(islands)

        union_islands = UnionIslands.calc_overlapped_island_groups(all_islands, self.threshold)

        counter = 0
        counter_skipped = 0
        to_select = []
        to_deselect = []

        for union_isl in union_islands:
            if self.mode == 'SELECT':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        to_select.append(isl)
                else:
                    if union_isl.is_full_face_deselected():
                        continue
                    to_deselect.append(union_isl)
            elif self.mode == 'ADDITION':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_selected():
                            counter_skipped += 1
                            continue

                        counter += 1
                        to_select.append(isl)
            else:  # self.mode == 'DESELECT':
                if isinstance(union_isl, UnionIslands):
                    for isl in union_isl:
                        if isl.is_full_face_deselected():
                            counter_skipped += 1
                            continue
                        counter += 1
                        to_deselect.append(union_isl)

            if to_deselect or to_select:
                for isl in to_deselect:
                    if need_sync_validation_check:
                        isl.umesh.sync_from_mesh_if_needed()
                    isl.umesh.update_tag = True
                    isl.select = False

                for isl in to_select:
                    if need_sync_validation_check:
                        isl.umesh.sync_from_mesh_if_needed()
                    isl.umesh.update_tag = True
                    isl.select = True

                # Fix vertex and edge selection after deselect
                if need_sync_validation_check and to_deselect:
                    for umesh in {isl.umesh for isl in to_deselect}:
                        umesh.bm.uv_select_sync_to_mesh()

        if not union_islands:
            self.report({'WARNING'}, f'Islands not found')
        else:
            if not (counter + counter_skipped):
                self.report({'WARNING'}, f'No found stacked islands')
        umeshes.silent_update()
        return {'FINISHED'}

class UNIV_OT_SelectByVertexCount_Base(Operator):
    bl_label = 'Select by Vertex Count'
    bl_description = "Select by Vertex Count"
    bl_options = {'REGISTER', 'UNDO'}

    elem_mode: EnumProperty(name='Elem Mode', default='FACE', items=(
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))

    mode: EnumProperty(name='Select Mode', default='SELECT', items=(
        ('SELECT', 'Select', ''),
        ('ADDITION', 'Addition', ''),
        ('DESELECT', 'Deselect', ''),
    ))
    polygone_type: EnumProperty(name='Polygone Type', default='TRIS', items=(
        ('TRIS', 'Tris', ''),
        ('QUAD', 'Quad', ''),
        ('NGONE', 'N-Gone', ''),
    ))
    use_face_target_size: BoolProperty(name='Use target face size', default=False)
    face_target_size: IntProperty(name='Face Size', min=3, soft_max=32, default=4)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'elem_mode', expand=True)
        layout.row(align=True).prop(self, 'mode', expand=True)
        row = layout.row(align=True)
        row.active = not self.use_face_target_size
        row.prop(self, 'polygone_type', expand=True)

        row = layout.row(align=True)
        row.prop(self, "use_face_target_size", text="")
        row.active = self.use_face_target_size
        row.prop(self, 'face_target_size')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.ctrl:
            self.mode = 'DESELECT'
        elif event.shift:
            self.mode = 'ADDITION'
        else:
            self.mode = 'SELECT'
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def ok_or_report_info(self, counter, counter_without_effect):
        elem_name = self.elem_mode.capitalize() + 's'
        if self.mode == 'SELECT':
            if counter and counter_without_effect:
                print(f"UniV: Select by Vertex Count: "
                      f"Found {counter + counter_without_effect} {elem_name} for select, {counter_without_effect} of them were already selected")
            elif counter and not counter_without_effect:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for select")
            elif not counter and counter_without_effect:
                self.report({'INFO'},
                            f"Found {counter_without_effect} {elem_name} for select, that were all initially selected")
            else:
                self.report({'INFO'}, f"Not found {elem_name} for select")

        elif self.mode == 'DESELECT':
            if counter:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for deselect")
            else:
                self.report({'INFO'}, f"No {elem_name} found to deselect - they may have been unselected initially")
        else:  # self.mode == 'ADDITION':
            if counter:
                print(f"UniV: Select by Vertex Count: Found {counter} {elem_name} for additional select")
            else:
                self.report({'INFO'},
                            f"No {elem_name} found to additional select - they may have been selected initially")

    def get_is_target_face_func(self):
        if self.polygone_type == 'TRIS':
            is_target_polygon = lambda f_: len(f_.loops) == 3
        elif self.polygone_type == 'QUAD':
            is_target_polygon = lambda f_: len(f_.loops) == 4
        else:
            is_target_polygon = lambda f_: len(f_.loops) >= 5
        if self.use_face_target_size:
            is_target_polygon = lambda f_: len(f_.loops) == self.face_target_size
        return is_target_polygon


class UNIV_OT_SelectByVertexCount_VIEW2D(UNIV_OT_SelectByVertexCount_Base):
    bl_idname = "uv.univ_select_by_vertex_count"

    def execute(self, context):
        umeshes = UMeshes()

        need_sync_validation_check = False
        if umeshes.sync:
            if utils.USE_GENERIC_UV_SYNC:
                need_sync_validation_check = umeshes.elem_mode in ('VERT', 'EDGE')
            else:
                umeshes.elem_mode = 'FACE'

        counter = 0
        counter_without_effect = 0
        is_target_face = self.get_is_target_face_func()

        for umesh in umeshes:
            face_select_get = utils.face_select_get_func(umesh)
            has_any_select = utils.has_any_vert_select_func(umesh)

            to_select = []
            to_deselect = []

            if self.mode == 'SELECT':
                if self.elem_mode == 'FACE':
                    for f in utils.calc_visible_uv_faces_iter(umesh):
                        if is_target_face(f):
                            if face_select_get(f):
                                counter_without_effect += 1
                            else:
                                to_select.append(f)
                        elif has_any_select(f):
                            to_deselect.append(f)
                else:
                    for isl in AdvIslands.calc_visible_with_mark_seam(umesh):
                        if all(is_target_face(f) for f in isl):
                            if isl.is_full_face_selected():
                                counter_without_effect += 1
                            else:
                                to_select.append(isl)
                        elif not isl.is_full_deselected_by_context():
                            to_deselect.append(isl)

            elif self.mode == 'ADDITION':
                if self.elem_mode == 'FACE':
                    to_select.extend(f for f in utils.calc_unselected_uv_faces_iter(umesh) if is_target_face(f))
                else:
                    for isl in AdvIslands.calc_visible_with_mark_seam(umesh):
                        if all(is_target_face(f) for f in isl):
                            if not isl.is_full_face_selected():
                                to_select.append(isl)
                            else:
                                counter_without_effect += 1

            else: # self.mode == 'DESELECT':
                if self.elem_mode == 'FACE':
                    for f in utils.calc_selected_uv_faces_iter(umesh):
                        if is_target_face(f):
                            if has_any_select(f):
                                to_deselect.append(f)
                            else:
                                counter_without_effect += 1
                else:
                    for isl in AdvIslands.calc_visible_with_mark_seam(umesh):
                        if all(is_target_face(f) for f in isl):
                            if not isl.is_full_deselected_by_context():
                                to_deselect.append(isl)
                            else:
                                counter_without_effect += 1

            if to_select or to_deselect:
                if need_sync_validation_check:
                    umesh.sync_from_mesh_if_needed()

                # NOTE: Use face_select_set_func after sync validation
                face_select_set = utils.face_select_set_func(umesh)

                if self.elem_mode == 'FACE':
                    for f in to_deselect:
                        face_select_set(f, False)
                    for f in to_select:
                        face_select_set(f, True)
                else:
                    for isl in to_deselect:
                        isl.select = False
                    for isl in to_select:
                        isl.select = True

                if need_sync_validation_check and to_deselect:
                    umesh.bm.uv_select_sync_to_mesh()

                umesh.update()
                counter += len(to_select)
                if self.mode != 'SELECT':
                    counter += len(to_deselect)

        self.ok_or_report_info(counter, counter_without_effect)

        for umesh in umeshes:
            umesh.check_faces_exist(self.report)
        return {'FINISHED'}


class UNIV_OT_SelectByVertexCount_VIEW3D(UNIV_OT_SelectByVertexCount_Base):
    bl_idname = "mesh.univ_select_by_vertex_count"

    elem_mode: EnumProperty(name='Elem Mode', default='ISLAND', items=(
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))

    def execute(self, context):
        umeshes = UMeshes.calc_any_unique(verify_uv=False)
        umeshes.set_sync()
        for umesh in umeshes:
            umesh.sync_valid = False

        if umeshes.sync:
            umeshes.elem_mode = 'FACE'

        counter = 0
        counter_without_effect = 0
        is_target_face = self.get_is_target_face_func()


        for umesh in umeshes:
            local_counter = 0
            if self.mode == 'SELECT':
                has_update = False
                if self.elem_mode == 'FACE':
                    for f in umesh.bm.faces:
                        if f.hide:
                            continue
                        if is_target_face(f):
                            if f.select:
                                counter_without_effect += 1
                            else:
                                local_counter += 1
                                f.select = True
                        else:
                            if not f.select:
                                counter_without_effect += 1
                            else:
                                local_counter += 1
                                f.select = False
                else:
                    for isl in MeshIslands.calc_visible_with_mark_seam(umesh):
                        if all(is_target_face(f) for f in isl):
                            if isl.is_full_face_selected():
                                counter_without_effect += 1
                            else:
                                local_counter += 1
                                isl.select = True
                        elif not isl.is_full_face_deselected():
                            has_update = True
                            isl.select = False

                if has_update or local_counter:
                    umesh.update()
                counter += local_counter
                continue

            elif self.mode == 'ADDITION':
                if self.elem_mode == 'FACE':
                    for f in umesh.bm.faces:
                        if (not f.select and not f.hide) and is_target_face(f):
                            local_counter += 1
                            f.select = True
                else:
                    for isl in MeshIslands.calc_visible_with_mark_seam(umesh):
                        if not isl.is_full_face_selected():
                            if all(is_target_face(f) for f in isl):
                                local_counter += 1
                                isl.select = True

            else: # self.mode == 'DESELECT':
                if self.elem_mode == 'FACE':
                    for f in umesh.bm.faces:
                        if f.select and is_target_face(f):
                            local_counter += 1
                            f.select = False
                else:
                    for isl in MeshIslands.calc_visible_with_mark_seam(umesh):
                        if not isl.is_full_face_deselected():
                            if all(is_target_face(f) for f in isl):
                                local_counter += 1
                                isl.select = False

            if local_counter:
                umesh.update()
                counter += local_counter

        self.ok_or_report_info(counter, counter_without_effect)

        for umesh in umeshes:
            umesh.check_faces_exist(self.report)
        return {'FINISHED'}

class UNIV_OT_SelectMode(Operator):
    bl_idname = "uv.univ_select_mode"
    bl_label = 'Select Mode'
    bl_description = "Set Select Mode with sticky disabled in Face Mode"
    bl_options = {'REGISTER', 'UNDO'}

    type: EnumProperty(name='Type', default='VERTEX', items=(
        ('VERTEX', 'Vertex', ''),
        ('EDGE', 'Edge', ''),
        ('FACE', 'Face', ''),
        ('ISLAND', 'Island', ''),
    ))

    @classmethod
    def poll(cls, context):
        if context.scene.tool_settings.use_uv_select_sync:
            return bpy.ops.mesh.select_mode.poll()  # noqa
        else:
            return bpy.ops.uv.select_mode.poll()  # noqa

    def execute(self, context):
        if utils.sync():
            if self.type == 'ISLAND':
                # TODO: Implement toggle island checkbox for new sync select system
                return {'CANCELLED'}

            update = False
            if bpy.app.version >= (4, 5, 0):
                tool_settings = context.scene.tool_settings
                sticky_mode = tool_settings.uv_sticky_select_mode
                if self.type == 'FACE':
                    if sticky_mode != 'DISABLED':
                        update = True
                        tool_settings.uv_sticky_select_mode = 'DISABLED'
                else:
                    if sticky_mode != 'SHARED_LOCATION':
                        update = True
                        tool_settings.uv_sticky_select_mode = 'SHARED_LOCATION'
            elem_type = self.type
            if elem_type == 'VERTEX':
                elem_type = 'VERT'

            res = bpy.ops.mesh.select_mode(type=elem_type, use_extend=False, use_expand=False)
            if update or res == {'FINISHED'}:
                return {'FINISHED'}
            else:
                return res
        else:
            update = False
            tool_settings = context.scene.tool_settings
            sticky_mode = tool_settings.uv_sticky_select_mode
            if self.type == 'FACE':
                if sticky_mode != 'DISABLED':
                    update = True
                    tool_settings.uv_sticky_select_mode = 'DISABLED'
            else:
                if sticky_mode != 'SHARED_LOCATION':
                    update = True
                    tool_settings.uv_sticky_select_mode = 'SHARED_LOCATION'

            current_mode = utils.get_select_mode_uv()
            if self.type == 'ISLAND' and current_mode == 'FACE':
                # TODO: This behavior needs to be integrated into the Blender source.
                umeshes = UMeshes()
                umeshes.filter_by_selected_uv_faces()
                umeshes.update_tag = False

                for umesh in umeshes:
                    Islands.tag_filter_visible(umesh)
                    select_get = utils.face_select_get_func(umesh)
                    for faces in Islands.calc_with_markseam_iter_ex(umesh):
                        if not utils.all_equal(faces, select_get):
                            utypes.FaceIsland(faces, umesh).select = False
                            umesh.update_tag = True
                update |= umeshes.update_tag
                umeshes.silent_update()

            res = bpy.ops.uv.select_mode(type=self.type)
            if update or res == {'FINISHED'}:
                return {'FINISHED'}
            else:
                return res

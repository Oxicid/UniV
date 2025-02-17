# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import bl_math

from bpy.types import Operator
from bpy.props import *
from .. import utils
from .. import types
from ..types import Islands

class UNIV_OT_Cut_VIEW2D(Operator):
    bl_idname = "uv.univ_cut"
    bl_label = "Cut"
    bl_description = "Cut selected"
    bl_options = {'REGISTER', 'UNDO'}

    addition: BoolProperty(name='Addition', default=True)
    unwrap: EnumProperty(name='Unwrap', default='ANGLE_BASED',
                         items=(
                                ('NONE', 'None', ''),
                                ('ANGLE_BASED', 'Angle Based', ''),
                                ('CONFORMAL', 'Conformal', '')
                            ))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'addition')
        self.layout.column(align=True).prop(self, 'unwrap', expand=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self):
        self.sync = utils.sync()
        self.umeshes: types.UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = types.UMeshes(report=self.report)

        if self.sync:
            self.cut_view_2d_sync()
        else:
            self.cut_view_2d_no_sync()
        if self.unwrap != 'NONE':
            self.unwrap_after_unwrap()
        self.umeshes.update()
        return {'FINISHED'}

    def cut_view_2d_sync(self):
        for umesh in reversed(self.umeshes):
            if umesh.is_full_edge_deselected:
                self.umeshes.umeshes.remove(umesh)
                continue

            uv = umesh.uv
            shared_is_linked = utils.shared_is_linked
            if umesh.is_full_edge_selected:
                for f in umesh.bm.faces:
                    for crn in f.loops:
                        if (_shared_crn := crn.link_loop_radial_prev) == crn:
                            crn.edge.seam = True
                        elif not shared_is_linked(crn, _shared_crn, uv):
                            crn.edge.seam = True
                        elif not self.addition:
                            crn.edge.seam = False
            else:
                for f in umesh.bm.faces:
                    if f.hide:
                        continue
                    for crn in f.loops:
                        if not crn.edge.select:
                            continue
                        elif (_shared_crn := crn.link_loop_radial_prev) == crn:
                            crn.edge.seam = True
                        elif not (_shared_crn.face.select and f.select):
                            crn.edge.seam = True
                        elif not shared_is_linked(crn, _shared_crn, uv):
                            crn.edge.seam = True
                        elif not self.addition:
                            crn.edge.seam = False

    def cut_view_2d_no_sync(self):
        for umesh in self.umeshes:
            if umesh.is_full_face_deselected:
                self.umeshes.umeshes.remove(umesh)
                continue
            umesh.tag_selected_faces()

            uv = umesh.uv
            for f in umesh.bm.faces:
                if not f.select:
                    continue
                for crn in f.loops:
                    if not crn[uv].select_edge or not crn[uv].select:
                        continue
                    elif (_shared_crn := crn.link_loop_radial_prev) == crn:
                        crn.edge.seam = True
                    elif not (_shared_crn.face.tag and f.tag):
                        crn.edge.seam = True
                    elif not utils.shared_is_linked(crn, _shared_crn, uv):
                        crn.edge.seam = True
                    elif not self.addition:
                        crn.edge.seam = False

    def unwrap_after_unwrap(self):
        assert self.unwrap != 'NONE'

        save_transform_islands = []
        for umesh in self.umeshes:
            islands = Islands.calc_selected_with_mark_seam(umesh)
            for isl in islands:
                if any(v.select for f in isl for v in f.verts):
                    save_transform_islands.append(isl.save_transform())

        if save_transform_islands:
            bpy.ops.uv.unwrap(method=self.unwrap)
            for isl in save_transform_islands:
                isl.shift()
                isl.inplace()


class UNIV_OT_Cut_VIEW3D(Operator):
    bl_idname = "mesh.univ_cut"
    bl_label = "Cut"
    bl_description = "Cut selected"
    bl_options = {'REGISTER', 'UNDO'}

    addition: BoolProperty(name='Addition', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self):
        self.sync = utils.sync()
        self.umeshes: types.UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = types.UMeshes(report=self.report)
        self.cut_view_3d()
        self.umeshes.update()
        return {'FINISHED'}

    def cut_view_3d(self):
        for umesh in self.umeshes:
            if umesh.is_full_edge_deselected:
                umesh.update_tag = False
                continue
            elif umesh.is_full_edge_selected:
                for e in umesh.bm.edges:
                    if not e.is_manifold:
                        e.seam = True
                    elif not self.addition:
                        e.seam = False

            for e in umesh.bm.edges:
                if not e.select:
                    continue
                if not e.is_manifold:
                    e.seam = True
                    continue

                sum_select_face = sum(f.select for f in e.link_faces)
                if sum_select_face <= 1:
                    e.seam = True
                elif not self.addition:
                    e.seam = False

class UNIV_OT_Angle(Operator):
    bl_idname = "mesh.univ_angle"
    bl_label = "Angle"
    bl_description = "Seams by angle, sharps, materials, borders"
    bl_options = {'REGISTER', 'UNDO'}

    selected: BoolProperty(name='Selected', default=False)
    addition: BoolProperty(name='Addition', default=True)
    borders: BoolProperty(name='Borders', default=False)
    mtl: BoolProperty(name='Mtl', default=True)
    by_weight: BoolProperty(name='By Weight', default=True)
    by_sharps: BoolProperty(name='By Sharps', default=True)
    seams_to_sharps: BoolProperty(name='Seams to Sharps', default=True)
    obj_smooth: BoolProperty(name='Auto Smooth', default=True)
    angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'selected')
        layout.prop(self, 'addition')
        layout.separator()
        layout.prop(self, 'borders')
        layout.separator()
        layout.prop(self, 'mtl')
        layout.prop(self, 'by_weight')
        layout.prop(self, 'by_sharps')
        layout.prop(self, 'seams_to_sharps')
        layout.prop(self, 'obj_smooth')
        layout.prop(self, 'angle', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self):
        self.sync = utils.sync()
        self.umeshes: types.UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = types.UMeshes(report=self.report)

        # clamp angle
        if self.obj_smooth:
            max_angle_from_obj_smooth = max(umesh.smooth_angle for umesh in self.umeshes)
            self.angle = bl_math.clamp(self.angle, 0.0, max_angle_from_obj_smooth)

        for umesh in self.umeshes:
            umesh.check_uniform_scale(self.report)
            if self.selected and umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue

            if self.obj_smooth:
                angle = min(umesh.smooth_angle, self.angle)
            else:
                angle = self.angle

            if bpy.app.version >= (4, 0, 0):
                bevel_weight_key = umesh.bm.edges.layers.float.get('bevel_weight_edge')
            else:
                bevel_weight_key = umesh.bm.edges.layers.bevel_weight.active
            check_weights = self.by_weight and bevel_weight_key

            if umesh.is_full_face_selected:
                faces = (_f for _f in umesh.bm.faces)
            elif self.selected:
                faces = (_f for _f in umesh.bm.faces if _f.select)
            else:  # visible
                faces = (_f for _f in umesh.bm.faces if not _f.hide)

            for f in faces:
                for crn in f.loops:
                    crn_edge = crn.edge
                    if not crn_edge.is_manifold:  # boundary
                        if self.borders or len(crn_edge.link_faces) > 2:
                            crn_edge.seam = True
                        elif not self.addition:
                            crn_edge.seam = False
                    elif crn_edge.calc_face_angle() >= angle:  # Skip by angle
                        crn_edge.seam = True
                        if self.seams_to_sharps:
                            crn_edge.smooth = False
                    elif self.borders and crn.link_loop_radial_prev.face.hide:
                        crn_edge.seam = True
                    elif self.by_sharps and not crn_edge.smooth:
                        crn_edge.seam = True
                    elif self.mtl and f.material_index != crn.link_loop_radial_prev.face.material_index:
                        crn_edge.seam = True
                    elif check_weights and crn_edge[bevel_weight_key]:
                        crn_edge.seam = True
                    elif not self.addition:
                        crn_edge.seam = False

        self.umeshes.update()
        return {'FINISHED'}

class UNIV_OT_SeamBorder_VIEW3D(Operator):
    bl_idname = "mesh.univ_seam_border"
    bl_label = "Border"
    bl_description = "Seams by borders\n\n" \
                     "Default - Seams by borders\n" \
                     "Shift - Additional\n" \
                     "Alt - All Channels"

    bl_options = {'REGISTER', 'UNDO'}

    all_channels: BoolProperty(name='All Channels', default=False)
    addition: BoolProperty(name='Addition', default=False)
    selected: BoolProperty(name='Selected Islands', default=False)
    mtl: BoolProperty(name='Mtl', default=True)
    by_sharps: BoolProperty(name='By Sharps', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'all_channels')
        layout.prop(self, 'addition')
        layout.prop(self, 'selected')
        layout.prop(self, 'mtl')
        layout.prop(self, 'by_sharps')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        self.all_channels = event.alt
        return self.execute(context)

    def execute(self, context) -> set[str]:
        umeshes = types.UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV'):
            umeshes.set_sync()

        sync = umeshes.sync
        for umesh in reversed(umeshes):
            if self.selected:
                faces = utils.calc_selected_uv_faces(umesh)
            else:
                faces = utils.calc_visible_uv_faces(umesh)

            if not faces:
                umeshes.umeshes.remove(umesh)

            uv = umesh.uv
            is_pair = utils.is_pair
            uv_layers = umesh.obj.data.uv_layers
            if _all_channels := self.all_channels and len(uv_layers) > 1:
                prev_index = uv_layers.active_index
                indexes = list(range(len(uv_layers)))
                del indexes[prev_index]
                indexes.append(prev_index)
                seams = [False] * umesh.total_corners

                corners = [crn for f in faces for crn in f.loops]
                for first_loop_check, active_index in enumerate(indexes):
                    uv_layers.active_index = active_index

                    if first_loop_check == 0:
                        for idx, crn in enumerate(corners):
                            crn_edge = crn.edge
                            pair = crn.link_loop_radial_prev
                            if crn == pair:
                                seams[idx] = True
                            elif not is_pair(crn, pair, uv):
                                seams[idx] = True
                            elif (pair_face := pair.face).hide if sync else not (pair_face := pair.face).select:  # if hidden
                                seams[idx] = True
                            elif self.by_sharps and not crn_edge.smooth:
                                seams[idx] = True
                            elif self.mtl and crn.face.material_index != pair_face.material_index:
                                seams[idx] = True
                            elif crn_edge.seam and self.addition:
                                seams[idx] = True
                    else:
                        for idx, crn in enumerate(corners):
                            if seams[idx]:
                                continue
                            elif not is_pair(crn, crn.link_loop_radial_prev, uv):
                                seams[idx] = True

                for idx, crn in enumerate(corners):
                    crn.edge.seam = seams[idx]

                uv_layers.active_index = prev_index

            else:
                for f in faces:
                    for crn in f.loops:
                        crn_edge = crn.edge
                        pair = crn.link_loop_radial_prev
                        if crn == pair:
                            crn_edge.seam = True
                        elif not is_pair(crn, pair, uv):
                            crn_edge.seam = True
                        elif (pair_face := pair.face).hide if sync else not (pair_face := pair.face).select:  # if hidden
                            crn_edge.seam = True
                        elif self.by_sharps and not crn_edge.smooth:
                            crn_edge.seam = True
                        elif self.mtl and f.material_index != pair_face.material_index:
                            crn_edge.seam = True
                        elif not self.addition:
                            crn_edge.seam = False

        umeshes.update()
        return {'FINISHED'}

class UNIV_OT_SeamBorder(UNIV_OT_SeamBorder_VIEW3D):
    bl_idname = "uv.univ_seam_border"

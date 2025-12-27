# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import bl_math
from mathutils import Vector
from bpy.types import Operator
from bpy.props import *

from .. import utils
from .. import utypes
from ..utypes import UMeshes, AdvIslands
from ..preferences import prefs, univ_settings


class UNIV_OT_Cut_VIEW2D(Operator):
    bl_idname = "uv.univ_cut"
    bl_label = "Cut"
    bl_description = "Cut selected"
    bl_options = {'REGISTER', 'UNDO'}

    addition: BoolProperty(name='Addition', default=True)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)
    unwrap: EnumProperty(name='Unwrap', default='ANGLE_BASED',
                         items=(
                             ('NONE', 'None', ''),
                             ('ANGLE_BASED', 'Hard Surface', ''),
                             ('CONFORMAL', 'Conformal', ''),
                             ('MINIMUM_STRETCH', 'Organic', '')
                         ))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'addition')
        self.layout.prop(self, 'use_correct_aspect')
        self.layout.column(align=True).prop(self, 'unwrap', expand=True)

    def invoke(self, context, event):
        if not (context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV'):
            self.report({'WARNING'}, 'Active area must be UV type')
            return {'CANCELLED'}

        if event.value == 'PRESS':
            self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
            self.mouse_pos = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)

        self.addition = event.shift
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.fix_context()
        if self.unwrap == 'MINIMUM_STRETCH' and bpy.app.version < (4, 3, 0):
            self.unwrap = 'ANGLE_BASED'
            self.report({'WARNING'}, 'Organic Mode is not supported in Blender versions below 4.3')

        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_pos:
            return self.pick_cut()

        self.cut_uv_space()
        if self.unwrap != 'NONE':
            self.unwrap_after_cut()
        self.umeshes.update()

        # Flush System
        from .. import draw
        if not draw.DrawerSeamsProcessing.is_enable():
            visible_umeshes.filter_by_visible_uv_faces()
            self.umeshes.umeshes.extend(visible_umeshes.umeshes.copy())
            coords = draw.mesh_extract.extract_seams_umeshes(self.umeshes)
            draw.LinesDrawSimple.draw_register(coords, draw.DrawerSeamsProcessing.get_color())
        return {'FINISHED'}

    def cut_uv_space(self):
        for umesh in self.umeshes:
            uv = umesh.uv
            face_select_get = utils.face_select_get_func(umesh)
            for crn in utils.calc_selected_uv_edge_iter(umesh):
                pair_crn = crn.link_loop_radial_prev
                if not utils.is_pair(crn, pair_crn, uv):
                    crn.edge.seam = True
                elif not (face_select_get(pair_crn.face) and face_select_get(crn.face)):
                    crn.edge.seam = True
                elif not self.addition:
                    crn.edge.seam = False

    def unwrap_after_cut(self):
        assert self.unwrap != 'NONE'

        save_transform_islands = []
        for umesh in self.umeshes:
            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            islands = AdvIslands.calc_selected_with_mark_seam(umesh)
            for isl in islands:
                isl.apply_aspect_ratio()
                save_transform_islands.append(isl.save_transform(flip_if_needed=True))

        if save_transform_islands:
            bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False)
            for isl in save_transform_islands:
                isl.inplace(flip_if_needed=True)
                isl.island.reset_aspect_ratio()

                if isl.rotate:
                    utils.set_global_texel(isl.island)

    def pick_cut(self):
        hit = utypes.CrnEdgeHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            hit.find_nearest_crn_by_visible_faces(umesh)

        if not hit:
            self.report({'WARNING'}, 'Edge not found within a given radius')
            return {'CANCELLED'}
        else:
            e = hit.crn.edge
            had_seam = e.seam
            if not had_seam:
                hit.crn.edge.seam = True
                hit.umesh.update()

            from .. import draw
            if not draw.DrawerSeamsProcessing.is_enable():
                coords = draw.mesh_extract.extract_seams_umeshes(self.umeshes)
                draw.LinesDrawSimple.draw_register(coords, draw.get_seam_color())
                if coords:
                    bpy.context.area.tag_redraw()
            return {'FINISHED'} if had_seam else {'FINISHED'}


class UNIV_OT_Cut_VIEW3D(Operator, utypes.RayCast):
    bl_idname = "mesh.univ_cut"
    bl_label = "Cut"
    bl_description = "Cut selected"
    bl_options = {'REGISTER', 'UNDO'}

    addition: BoolProperty(name='Addition', default=True)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)
    unwrap: EnumProperty(name='Unwrap', default='ANGLE_BASED',
                         items=(
                             ('NONE', 'None', ''),
                             ('ANGLE_BASED', 'Hard Surface', ''),
                             ('CONFORMAL', 'Conformal', ''),
                             ('MINIMUM_STRETCH', 'Organic', '')
                         ))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout

        layout.prop(univ_settings(), 'use_texel')
        layout.prop(self, 'addition')
        layout.prop(self, 'use_correct_aspect')
        layout.prop(self, 'unwrap')


    def invoke(self, context, event):
        if event.value == 'PRESS':
            self.init_data_for_ray_cast(event)
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = UMeshes.calc(report=self.report, verify_uv=False)
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        selected, visible = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected if selected else visible

        if not self.umeshes:
            return self.umeshes.update(info='No elements for manipulate')

        if not selected and self.mouse_pos_from_3d:
            return self.pick_cut()
        else:
            self.cut_view_3d()
            self.umeshes.update()
            return {'FINISHED'}

    def cut_view_3d(self):
        umeshes_without_uv = []
        save_transform_islands = []
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0
            # TODO: Skip updates, if edges has seams
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

            if not umesh.total_face_sel or self.unwrap == 'NONE':
                continue

            if not len(umesh.bm.loops.layers.uv):
                umeshes_without_uv.append(umesh)
                continue

            umesh.verify_uv()
            islands = utypes.MeshIslands.calc_selected_with_mark_seam(umesh)
            adv_islands = islands.to_adv_islands()
            for isl in adv_islands:
                isl.apply_aspect_ratio()
                save_t = isl.save_transform(flip_if_needed=True)
                save_transform_islands.append(save_t)

        if umeshes_without_uv or save_transform_islands:
            bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False)

            for umesh in self.umeshes:
                umesh.value = umesh.check_uniform_scale(report=self.report)

            for isl in save_transform_islands:
                isl.inplace(flip_if_needed=True)
                isl.island.reset_aspect_ratio()
                if isl.rotate:
                    utils.set_global_texel(isl.island)

            texel = univ_settings().texel_density
            texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2

            for umesh in umeshes_without_uv:
                umesh.verify_uv()
                mesh_islands = utypes.MeshIslands.calc_selected_with_mark_seam(umesh)
                adv_islands = mesh_islands.to_adv_islands()
                adv_islands.calc_area_uv()
                adv_islands.calc_area_3d(scale=umesh.value)

                for isl in adv_islands:
                    if umesh.aspect != 1.0:
                        scale = Vector((1 / umesh.aspect, 1))
                        isl.scale(scale, pivot=isl.bbox.center)

                    # TODO: Check correctness texel density after aspect ratio (check Unwrap and Relax too)
                    if isl.set_texel(texel, texture_size):
                        # zero_area_islands.append(isl)
                        continue

                umesh.update()

    def pick_cut(self):
        if hit := self.ray_cast(prefs().max_pick_distance):
            if hit.crn.edge.seam:
                return {'CANCELLED'}
            hit.crn.edge.seam = True
            hit.umesh.update()
            return {'FINISHED'}
        return {'CANCELLED'}


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = UMeshes(report=self.report)

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
    selected: BoolProperty(name='Selected Faces', default=False)
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
        umeshes = UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV'):
            umeshes.set_sync()
            umeshes.sync_invalidate()

        for umesh in umeshes:
            if self.selected:
                faces = utils.calc_selected_uv_faces_iter(umesh)
            else:
                faces = utils.calc_visible_uv_faces_iter(umesh)

            has_update = False
            is_pair = utils.is_pair
            uv_layers_size = len(umesh.obj.data.uv_layers)
            if _all_channels := self.all_channels and uv_layers_size > 1:
                seams = [False] * umesh.total_corners
                corners = [crn for f in faces for crn in f.loops]

                for layer_idx in range(uv_layers_size):
                    uv = umesh.bm.loops.layers.uv[layer_idx]
                    if layer_idx == 0:
                        umesh.uv = uv
                        is_boundary = utils.is_boundary_func(umesh, with_seam=self.addition)

                        for idx, crn in enumerate(corners):
                            if is_boundary(crn):
                                seams[idx] = True
                            elif self.by_sharps and not crn.edge.smooth:
                                seams[idx] = True
                            elif self.mtl and crn.face.material_index != crn.link_loop_radial_prev.face.material_index:
                                seams[idx] = True
                    else:
                        for idx, crn in enumerate(corners):
                            if seams[idx]:
                                continue
                            elif not is_pair(crn, crn.link_loop_radial_prev, uv):
                                seams[idx] = True

                for idx, crn in enumerate(corners):
                    if crn.edge.seam != seams[idx]:
                        crn.edge.seam = seams[idx]
                        has_update = True

            else:
                is_boundary = utils.is_boundary_func(umesh, with_seam=self.addition)
                for f in faces:
                    for crn in f.loops:
                        crn_edge = crn.edge
                        if (is_boundary(crn) or
                                (self.by_sharps and not crn_edge.smooth) or
                                (self.mtl and f.material_index != crn.link_loop_radial_prev.face.material_index)):
                            if not crn_edge.seam:
                                crn_edge.seam = True
                                has_update = True

                        else:
                            if crn_edge.seam:
                                crn_edge.seam = False
                                has_update = True
            umesh.update_tag = has_update

        if self.bl_idname.startswith('UV'):
            # Flush System
            from .. import draw
            if not draw.DrawerSeamsProcessing.is_enable():
                coords = draw.mesh_extract.extract_seams_umeshes(umeshes)
                draw.LinesDrawSimple.draw_register(coords, draw.DrawerSeamsProcessing.get_color())

        umeshes.silent_update()
        return {'FINISHED'}


class UNIV_OT_SeamBorder(UNIV_OT_SeamBorder_VIEW3D):
    bl_idname = "uv.univ_seam_border"

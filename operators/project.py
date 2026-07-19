# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import bpy
import math

from math import pi, cos, sin
from bl_math import clamp
from .. import utils
from .. import utypes
from ..utypes import BBox, MeshIsland, MeshIslands
from bpy.props import *
from collections.abc import Callable
from mathutils import Vector, Euler, Matrix
from ..preferences import prefs, univ_settings


# noinspection PyTypeHints
class UNIV_OT_Normal(bpy.types.Operator):
    bl_idname = "mesh.univ_normal"
    bl_label = "Normal"
    bl_description = "Projection by faces normal.\n\nShift - Individual"
    bl_options = {'REGISTER', 'UNDO'}

    crop: bpy.props.BoolProperty(name='Crop', default=True,
                                 description='Packs the islands into a base tile, for performance purposes, does so with uncritical inaccuracy')
    orient: bpy.props.BoolProperty(name='Orient 2D', default=True)
    individual: bpy.props.BoolProperty(name='Individual', default=False, description='Individual by Island Meshes')
    mark_seam: bpy.props.BoolProperty(name='Mark Seam', default=True)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True,
                                               description='Gets Aspect Correct from the active image from the shader node editor')

    def draw(self, context):
        col = self.layout.column(align=True)
        if not prefs().use_texel:
            col.prop(self, 'crop')
        col.prop(self, 'orient')
        col.prop(self, 'individual')
        col.prop(self, 'mark_seam')

        col.separator()
        col.prop(prefs(), 'use_texel')
        col.prop(self, 'use_correct_aspect')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.individual = event.shift
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = 'No found faces for manipulate'
        self.has_selected: bool = True
        self.umeshes: utypes.UMeshes | None = None

    def execute(self, context):
        self.umeshes = utypes.UMeshes.calc(self.report, verify_uv=False)
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()
        if self.umeshes.is_edit_mode:
            selected, unselected = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected:
                self.umeshes = selected
                self.has_selected = True
            elif unselected:
                self.umeshes = unselected
                self.has_selected = False
        else:
            self.umeshes.ensure(face=True)  # TODO: Delete ensure?1

        if not self.umeshes:
            return self.umeshes.update(info=self.info)

        self.umeshes.verify_uv()

        if self.use_correct_aspect:
            for umesh in self.umeshes:
                umesh.aspect = utils.get_aspect_ratio(umesh)

        if self.individual:
            self.xyz_to_uv_individual()
        else:
            self.xyz_to_uv()

        if not self.umeshes.is_edit_mode:
            ret = self.umeshes.update(info=self.info)
            self.umeshes.free()
            bpy.context.area.tag_redraw()
            return ret
        return self.umeshes.update(info=self.info)

    def xyz_to_uv(self):
        vector_nor, islands_of_mesh = self.avg_normal_and_calc_faces()
        rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)

        global_bbox = BBox()
        adv_islands_of_mesh = []

        for mesh_islands in islands_of_mesh:
            adv_island = mesh_islands[0].to_adv_island()
            adv_islands_of_mesh.append(adv_island)
            self.project_orient_and_calc_crop_data(adv_island, global_bbox, rot_mtx_from_normal)

        self.crop_islands(adv_islands_of_mesh, global_bbox)

    def xyz_to_uv_individual(self):
        global_bbox = BBox()
        adv_islands_of_mesh = []

        for vector_nor, mesh_islands in self.avg_normal_and_calc_faces_individual():
            adv_island = mesh_islands.to_adv_island()
            adv_islands_of_mesh.append(adv_island)
            rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)
            self.project_orient_and_calc_crop_data(adv_island, global_bbox, rot_mtx_from_normal)

        self.crop_islands(adv_islands_of_mesh, global_bbox)

    def project_orient_and_calc_crop_data(self, adv_island, global_bbox, rot_mtx_from_normal):
        aspect = adv_island.umesh.aspect
        uv = adv_island.umesh.uv

        if aspect >= 1.0:
            aspect_mtx = Matrix.Diagonal((1 / aspect, 1, 1))
        else:
            aspect_mtx = Matrix.Diagonal((1, aspect))
        mtx = aspect_mtx.to_4x4() @ rot_mtx_from_normal @ adv_island.umesh.obj.matrix_world

        if prefs().use_texel:
            td_scale = utils.get_scale_from_texel()
            mtx = mtx @ Matrix.Diagonal([td_scale]*3).to_4x4()

        points = []
        points_append = points.append

        if self.orient or (self.crop and not prefs().use_texel):
            for f in adv_island:
                for crn in f.loops:
                    uv_co = (mtx @ crn.vert.co).to_2d()
                    crn[uv].uv = uv_co
                    points_append(uv_co)
        else:
            for f in adv_island:
                for crn in f.loops:
                    crn[uv].uv = (mtx @ crn.vert.co).to_2d()
            return

        convex_coords = utils.calc_convex_points(points)
        bbox = BBox.calc_bbox(convex_coords)

        if self.orient:
            angle = -utils.calc_min_align_angle(convex_coords, aspect)
            adv_island.rotate(angle, bbox.center, aspect)
            bbox.rotate_expand(angle)

        global_bbox.union(bbox)

    def crop_islands(self, adv_islands_of_mesh: list[utypes.AdvIsland], bbox):
        if not self.crop:
            return

        if prefs().use_texel:
            return

        if self.mark_seam:
            for isl in adv_islands_of_mesh:
                isl.mark_seam()

        pad = utils.get_pad()
        pivot = bbox.left_bottom
        for island in adv_islands_of_mesh:
            if island.umesh.aspect != 1.0:
                if island.umesh.aspect < 1:
                    src_bb = bbox.copy()
                    src_bb.scale(Vector((1, island.umesh.aspect)), pivot=pivot)
                else:
                    src_bb = bbox.copy()
                    src_bb.scale(Vector((1 / island.umesh.aspect, 1)), pivot=pivot)
            else:
                src_bb = bbox

            tar_bb = BBox.from_center((0.5, 0.5))
            scale, delta, pivot = utils.get_transform_from_box(src_bb, tar_bb, axis='XY', pad=pad, use_crop=True)
            island.umesh.update_tag |= island.scale_with_move(scale, delta, pivot)

    def avg_normal_and_calc_faces_individual(self):
        if self.umeshes.is_edit_mode:
            if self.has_selected:
                calc_mesh_isl_obj = MeshIslands.calc_selected
            else:
                calc_mesh_isl_obj = MeshIslands.calc_visible
        else:
            calc_mesh_isl_obj = MeshIslands.calc_all

        for umesh in self.umeshes:
            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            for mesh_isl in calc_mesh_isl_obj(umesh):  # noqa
                weight = Vector()
                for f in mesh_isl:
                    weight += f.normal * f.calc_area()
                weight = mtx @ weight
                yield weight, mesh_isl

    def avg_normal_and_calc_faces(self):
        tot_weight = Vector()
        islands_of_mesh: list[MeshIslands] = []
        for umesh in self.umeshes:
            weight = Vector()
            if not self.umeshes.is_edit_mode:
                faces = umesh.bm.faces
            else:
                if self.has_selected:
                    faces = utils.calc_selected_uv_faces(umesh)
                else:
                    faces = utils.calc_visible_uv_faces(umesh)

            for f in faces:
                weight += f.normal * f.calc_area()

            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            tot_weight += mtx @ weight

            islands_of_mesh.append(MeshIslands([MeshIsland(faces, umesh)], umesh))

        return tot_weight, islands_of_mesh

    @staticmethod
    def calc_rot_mtx_from_normal(normal):
        vector_z = Vector((0.0, 0.0, 1.0))

        # rotate x
        vector_n = Vector((0.0, normal.y, normal.z))
        theta_x = vector_z.angle(vector_n, 0)
        vector_cross = vector_n.cross(vector_z)
        if vector_cross.x < 0:
            theta_x = -theta_x
        eul = Euler((theta_x, 0, 0))
        normal.rotate(eul)

        # rotate y
        vector_n = Vector((normal.x, 0.0, normal.z))
        theta_y = vector_z.angle(vector_n, 0)
        vector_cross = vector_n.cross(vector_z)
        if vector_cross.y < 0:
            theta_y = -theta_y
        eul.y = theta_y

        return eul.to_matrix().to_4x4()

# noinspection PyTypeHints
class UNIV_OT_BoxProject(bpy.types.Operator):
    bl_idname = "mesh.univ_box_project"
    bl_label = "Box"
    bl_description = "Box Projection"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(name='Scale', default=1, soft_min=0.5, soft_max=2)
    scale_individual: bpy.props.FloatVectorProperty(name='Scale Individual', default=(1.0, 1.0, 1.0), soft_min=0.5, soft_max=2)
    rotation: bpy.props.FloatVectorProperty(name='Rotate', subtype='EULER', soft_min=-pi, soft_max=pi)
    move: bpy.props.FloatVectorProperty(name='Move', subtype='XYZ')
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True,
                                               description='Gets Aspect Correct from the active image from the shader node editor')
    avoid_flip: bpy.props.BoolProperty(name='Avoid Flip', default=True)

    def draw(self, context):
        self.layout.prop(self, 'scale', slider=True)
        col = self.layout.column(align=True)
        col.prop(self, 'scale_individual', expand=True, slider=True)
        col.prop(self, 'rotation', expand=True, slider=True)
        col.prop(self, 'move', expand=True)
        col.prop(self, 'avoid_flip')
        col.separator()
        col.prop(prefs(), 'use_texel')
        col.prop(self, 'use_correct_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_edit_mode: bool = bpy.context.mode == 'EDIT_MESH'
        self.has_selected: bool = True
        self.umeshes: utypes.UMeshes | None = None

    def execute(self, context):
        self.umeshes = utypes.UMeshes.calc(self.report, verify_uv=False)
        self.umeshes.set_sync(True)
        if self.is_edit_mode:
            selected, visible = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected:
                self.umeshes = selected
                self.has_selected = True
            elif visible:
                self.umeshes = visible
                self.has_selected = False
            else:
                self.report({'WARNING'}, 'Not found faces for manipulate')
                return {'CANCELLED'}
        self.umeshes.verify_uv()

        self.box()
        for u in self.umeshes:
            u.check_uniform_scale(self.report)

        if not self.is_edit_mode:
            self.umeshes.update('No faces for manipulate')
            self.umeshes.free()
            bpy.context.area.tag_redraw()
        else:
            self.umeshes.update(info='Not selected face')
        return {'FINISHED'}

    def box(self):
        for umesh in self.umeshes:

            mtx_x, mtx_y, mtx_z, r = self.get_box_transforms(umesh)
            if self.is_edit_mode:
                if self.has_selected:
                    faces = utils.calc_selected_uv_faces_iter(umesh)
                else:
                    faces = utils.calc_visible_uv_faces_iter(umesh)
            else:
                faces = umesh.bm.faces

            self.box_ex(faces, mtx_x, mtx_y, mtx_z, r, umesh.uv)

    def get_box_transforms(self, umesh):
        move = Vector(self.move) * -1
        scale = Vector(self.scale_individual) * self.scale
        scale *= utils.get_scale_from_texel()

        mtx_from_prop_x = Matrix.LocRotScale(move, Euler((self.rotation[0], 0, 0)), scale)
        mtx_from_prop_y = Matrix.LocRotScale(move, Euler((0, self.rotation[1], 0)), scale)
        mtx_from_prop_z = Matrix.LocRotScale(move, Euler((0, 0, self.rotation[2])), scale)

        aspect_x_mtx, aspect_y_mtx, aspect_z_mtx = self.get_aspect_matrix(umesh)

        mtx_x = aspect_x_mtx @ umesh.obj.matrix_world @ mtx_from_prop_x
        mtx_y = aspect_y_mtx @ umesh.obj.matrix_world @ mtx_from_prop_y
        mtx_z = aspect_z_mtx @ umesh.obj.matrix_world @ mtx_from_prop_z
        _, r, _ = umesh.obj.matrix_world.decompose()
        # TODO: r.rotate(Euler(self.rotation))
        return mtx_x, mtx_y, mtx_z, r

    def get_aspect_matrix(self, umesh):
        aspect = (utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0)
        if aspect >= 1.0:
            aspect_x_mtx = Matrix.Diagonal((1, 1 / aspect, 1))
            aspect_y_mtx = Matrix.Diagonal((1 / aspect, 1, 1))
            aspect_z_mtx = Matrix.Diagonal((1 / aspect, 1, 1))
        else:
            aspect_x_mtx = Matrix.Diagonal((1, 1, aspect))
            aspect_y_mtx = Matrix.Diagonal((1, 1, aspect))
            aspect_z_mtx = Matrix.Diagonal((1, aspect, 1))
        return aspect_x_mtx.to_4x4(), aspect_y_mtx.to_4x4(), aspect_z_mtx.to_4x4()

    def box_ex(self, faces, mtx_x, mtx_y, mtx_z, r, uv):
        if not self.avoid_flip:
            for f in faces:
                n = f.normal.copy()
                n.rotate(r)
                if abs(n.x) >= abs(n.y) and abs(n.x) >= abs(n.z):  # X
                    for crn in f.loops:
                        crn[uv].uv = (mtx_x @ crn.vert.co).yz
                elif abs(n.y) >= abs(n.x) and abs(n.y) >= abs(n.z):  # Y
                    for crn in f.loops:
                        crn[uv].uv = (mtx_y @ crn.vert.co).xz
                else:  # Z
                    for crn in f.loops:
                        crn[uv].uv = (mtx_z @ crn.vert.co).to_2d()
        else:
            mtx_x_neg = (Matrix.Diagonal(Vector((1, -1, 1))).to_4x4()) @ mtx_x
            mtx_y_neg = (Matrix.Diagonal(Vector((-1, 1, 1))).to_4x4()) @ mtx_y
            mtx_z_neg = (Matrix.Diagonal(Vector((-1, 1, 1))).to_4x4()) @ mtx_z

            vec_x = Vector((1, 0, 0))
            vec_y = Vector((0, 1, 0))
            vec_z = Vector((0, 0, 1))

            angle_80 = math.radians(80)

            for f in faces:
                n = f.normal.copy()
                n.rotate(r)

                # X-axis
                if abs(n.x) >= abs(n.y) and abs(n.x) >= abs(n.z):
                    if n.angle(vec_x, 0) <= angle_80:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_x @ crn.vert.co).yz
                    else:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_x_neg @ crn.vert.co).yz
                # Y-axis
                elif abs(n.y) >= abs(n.x) and abs(n.y) >= abs(n.z):
                    if n.angle(vec_y, 0) <= angle_80:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_y_neg @ crn.vert.co).xz
                    else:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_y @ crn.vert.co).xz
                # Z-axis
                else:
                    if n.angle(vec_z, 0) <= angle_80:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_z @ crn.vert.co).to_2d()
                    else:
                        for crn in f.loops:
                            crn[uv].uv = (mtx_z_neg @ crn.vert.co).to_2d()


class ProjCameraInfo:
    def __init__(self):
        self.cam_angle = 0.0
        self.cam_size = 0.0
        self.aspect = Vector((1, 1))
        self.shift = Vector((0, 0))
        self.rot_mat = Matrix()
        self.cam_inv = Matrix()
        self.do_persp = False
        self.do_pano = False
        self.do_rot_mat = False

    @classmethod
    def uv_project_camera_info(cls, ob: bpy.types.Object, rot_mat, winx, winy):
        uci = cls()
        camera: bpy.types.Camera = ob.data
        uci.do_pano = camera.type == 'PANO'
        uci.do_persp = camera.type == 'PERSP'
        uci.cam_angle = cls.focal_length_to_fov(camera.lens, camera.sensor_width) / 2.0
        uci.cam_size = math.tan(uci.cam_angle) if uci.do_persp else camera.ortho_scale

        cam_inv = ob.matrix_world.normalized()
        res = cam_inv.inverted(None)
        if res is not None:
            uci.cam_inv = res
            # normal projection
            if rot_mat:
                uci.rot_mat = rot_mat.copy()
            uci.do_rot_mat = bool(rot_mat)

            # also make aspect ratio adjustment factors
            if winx > winy:
                uci.aspect.x = 1.0
                uci.aspect.y = winx / winy
            else:
                uci.aspect.x = winy / winx
                uci.aspect.y = 1.0

            # include 0.5f here to move the UVs into the center */
            uci.shift.x = 0.5 - (camera.shift_x * uci.aspect.x)
            uci.shift.y = 0.5 - (camera.shift_y * uci.aspect.y)
            return uci
        else:
            return None

    @staticmethod
    def focal_length_to_fov(focal_length: float, sensor: float):
        return 2.0 * math.atan((sensor / 2.0) / focal_length)

# noinspection PyTypeHints
class UNIV_OT_ViewProject(bpy.types.Operator):
    bl_idname = "mesh.univ_view_project"
    bl_label = "View"
    bl_description = "Projection by View"
    bl_options = {'REGISTER', 'UNDO'}

    camera_bounds: BoolProperty(name='Camera Bounds', default=False)
    use_crop: BoolProperty(name='Crop', default=True, description='Packs the islands into a base tile')
    use_orthographic: BoolProperty(name='Use Orthographic', default=False)
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)

    def draw(self, context):
        layout = self.layout
        if not self.use_orthographic and self.camera:
            layout.prop(self, 'camera_bounds')
        layout.prop(self, 'use_crop')
        layout.prop(self, 'use_orthographic')
        layout.prop(self, 'use_correct_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = 'No found faces for manipulate'
        self.has_selected: bool = True
        self.umeshes: utypes.UMeshes | None = None
        self.region = None
        self.area = None
        self.rv3d = None
        self.v3d = None
        self.faces_calc_type: Callable = Callable
        self.camera = None

    def execute(self, context):
        self.umeshes = utypes.UMeshes.calc(self.report, verify_uv=False)
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        self.area = context.area
        if self.area.type != 'VIEW_3D':
            self.area = utils.get_area_by_type('VIEW_3D')
            if not self.area:
                self.report({'WARNING'}, 'Active area must be 3D View')
                return {'CANCELLED'}

        self.v3d = self.area.spaces.active
        self.region = next(reg for reg in self.area.regions if reg.type == 'WINDOW')
        self.rv3d = self.region.data
        self.camera = utils.get_view3d_camera_data(self.v3d, self.rv3d)  # noqa

        if self.umeshes.is_edit_mode:
            selected, unselected = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected:
                self.umeshes = selected
                self.has_selected = True
                self.faces_calc_type = utils.calc_selected_uv_faces
            elif unselected:
                self.umeshes = unselected
                self.has_selected = False
                self.faces_calc_type = utils.calc_visible_uv_faces
        else:
            self.faces_calc_type = lambda umesh_: umesh_.bm.faces

        if not self.umeshes:
            return self.umeshes.update(info=self.info)

        self.umeshes.verify_uv()

        if self.use_correct_aspect:
            for umesh in self.umeshes:
                umesh.aspect = utils.get_aspect_ratio(umesh)

        self.view_project()

        if not self.umeshes.is_edit_mode:
            ret = self.umeshes.update(info=self.info)
            self.umeshes.free()
            bpy.context.area.tag_redraw()
            return ret
        return self.umeshes.update(info=self.info)

    def view_project(self):
        # objects_pos_avg = Vector()
        # for umesh in self.umeshes:
        #     loc, _, _ = umesh.obj.matrix_world.decompose()
        #     objects_pos_avg += loc
        # objects_pos_offset = -(objects_pos_avg * (1 / len(self.umeshes)))
        # objects_pos_offset.resize(4)
        pointers_to_coords = []
        coords_append = pointers_to_coords.append
        for umesh in self.umeshes:
            uv = umesh.uv
            aspect = utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0
            if self.use_orthographic:
                rot_mat = self.uv_map_rotation_matrix_ex(umesh, aspect=aspect)
                for f in self.faces_calc_type(umesh):
                    for crn in f.loops:
                        # uv_project_from_view_ortho
                        pv = rot_mat @ crn.vert.co  # projected_vertex
                        crn_co = crn[uv].uv
                        crn_co[:] = -pv[0], pv[2]
                        coords_append(crn_co)
            elif self.camera:
                # self.camera_bounds
                r = bpy.context.scene.render
                uci = ProjCameraInfo.uv_project_camera_info(
                    self.v3d.camera,
                    umesh.obj.matrix_world,
                    (r.resolution_x * r.pixel_aspect_x) * 2 if self.camera_bounds else 1.0,
                    (r.resolution_y * r.pixel_aspect_y) * 2 if self.camera_bounds else 1.0)
                if uci:
                    for f in self.faces_calc_type(umesh):
                        for crn in f.loops:
                            crn_co = crn[uv].uv
                            coords_append(crn_co)
                            self.uv_project_from_camera(crn_co, crn.vert.co, uci)
                else:
                    self.report({'WARNING'}, 'Not found camera info')
                    return
            else:

                winx = self.region.width
                winy = self.region.height * aspect
                pers_mat = self.rv3d.perspective_matrix
                rot_mat = umesh.obj.matrix_world.copy()
                for f in self.faces_calc_type(umesh):
                    for crn in f.loops:
                        crn_co = crn[uv].uv
                        coords_append(crn_co)
                        self.uv_project_from_view(crn_co, crn.vert.co, pers_mat, rot_mat, winx, winy)
        if self.use_crop:
            self.crop(pointers_to_coords)

    @staticmethod
    def uv_project_from_view(target: Vector, source: Vector, pers_mat, rot_mat, winx, winy):
        pv4 = source.copy()
        pv4.resize(4)
        pv4[3] = 1.0
        x = 0.0
        y = 0.0

        # rot_mat is the object matrix in this case */
        pv4 = rot_mat @ pv4

        # almost ED_view3d_project_short
        pv4 = pers_mat @ pv4
        if abs(pv4[3]) > 0.00001:  # avoid division by zero
            target[0] = winx / 2.0 + (winx / 2.0) * pv4[0] / pv4[3]
            target[1] = winy / 2.0 + (winy / 2.0) * pv4[1] / pv4[3]

        else:
            # scaling is lost but give a valid result
            target[0] = winx / 2.0 + (winx / 2.0) * pv4[0]
            target[1] = winy / 2.0 + (winy / 2.0) * pv4[1]

        # v3d.pers_mat seems to do this funky scaling
        if winx > winy:
            y = (winx - winy) / 2.0
            winy = winx
        else:
            x = (winy - winx) / 2.0
            winx = winy

        target[0] = (x + target[0]) / winx
        target[1] = (y + target[1]) / winy

    def uv_map_rotation_matrix_ex(self, umesh, up_angle_deg=90.0, side_angle_deg=0.0, radius=1.0, aspect=1.0):
        # get rotation of the current view matrix
        view_matrix = self.rv3d.view_matrix.copy()
        view_matrix[3] = [0] * 4  # but shifting
        # view_matrix[1][1] *= aspect

        # get rotation of the current object matrix
        rot_obj = umesh.obj.matrix_world.copy()
        rot_obj[3] = [0] * 4  # but shifting

        # rot_obj[3] = offset
        rot_obj[3][3] = 0.0

        rot_up = Matrix()
        rot_up.zero()
        rot_side = Matrix()
        rot_side.zero()

        # Compensate front/side.. against opengl x,y,z world definition.
        # This is "a sledgehammer to crack a nut" (overkill), a few plus minus 1 will do here.
        # I wanted to keep the reason here, so we're rotating.
        side_angle = pi * (side_angle_deg + 180.0) / 180.0
        rot_side[0][0] = cos(side_angle)
        rot_side[0][1] = -sin(side_angle)
        rot_side[1][0] = sin(side_angle)
        rot_side[1][1] = cos(side_angle)
        rot_side[2][2] = 1.0

        up_angle = -(pi * up_angle_deg / 180.0)
        rot_up[1][1] = cos(up_angle) / radius
        rot_up[1][2] = -sin(up_angle) / radius
        rot_up[2][1] = sin(up_angle) / radius
        rot_up[2][2] = cos(up_angle) / radius
        rot_up[0][0] = 1.0 / radius

        # Calculate transforms
        if aspect < 1:
            aspect_mtx = Matrix.Diagonal((1, aspect, 1)).to_4x4()
        else:
            aspect_mtx = Matrix.Diagonal((1/aspect, 1, 1)).to_4x4()
        return rot_up  @ aspect_mtx @ rot_side  @ view_matrix   @ rot_obj

    @staticmethod
    def uv_project_from_camera(target: Vector, source: Vector, uci: ProjCameraInfo):
        pv4 = source.copy()
        pv4.resize(4)
        pv4[3] = 1.0

        # rot_mat is the object matrix in this case
        if uci.do_rot_mat:
            pv4 = uci.rot_mat @ pv4  # check with swap matmul
            # pv4 = pv4 @ uci.rot_mat  # check with swap matmul

        # cam_inv is the inverse camera matrix
        pv4 = uci.cam_inv @ pv4
        # pv4 = pv4 @ uci.cam_inv

        if uci.do_pano:
            angle = math.atan2(pv4[0], -pv4[2]) / (pi * 2.0)  # angle around the camera

            if uci.do_persp:
                vec2d = Vector((pv4[0], pv4[2]))  # 2D position from the camera
                target[0] = angle * (pi / uci.cam_angle)
                target[1] = pv4[1] / (vec2d.length * (uci.cam_size * 2.0))
            else:
                target[0] = angle  # no correct method here, just map to  0-1
                target[1] = pv4[1] / uci.cam_size
        else:
            if pv4[2] == 0.0:
                pv4[2] = 0.00001  # don't allow div by 0

            if not uci.do_persp:
                target[:] = pv4.xy / uci.cam_size
            else:
                target[:] = ((-pv4.xy) * ((1.0 / uci.cam_size) / pv4[2])) / 2.0

        target *= uci.aspect
        #  adds camera shift + 0.5
        target += uci.shift

    @staticmethod
    def crop(pointers_to_coords, padding=0.001):
        bbox = BBox.calc_bbox(pointers_to_coords)
        w = bbox.width
        h = bbox.height
        scale_x = ((1.0 - padding) / w) if w else 1
        scale_y = ((1.0 - padding) / h) if h else 1

        scale_x = scale_y = min(scale_x, scale_y)

        scale = Vector((scale_x, scale_y))
        bbox.scale(scale)
        delta = Vector((padding, padding)) / 2 - bbox.min
        pivot = bbox.center

        diff = (pivot - pivot * scale) + delta

        # TODO: Nearest crop (not start)
        for co in pointers_to_coords:
            co *= scale
            co += diff

# noinspection PyTypeHints
class UNIV_OT_SmartProject(bpy.types.Operator):
    bl_idname = 'mesh.univ_smart_project'
    bl_label = 'Smart'
    bl_description = 'Smart Projection'
    bl_options = {'REGISTER', 'UNDO'}

    add_padding: IntProperty(name='Additional Padding', default=0, min=-16, max=16, subtype='PIXEL')
    angle_limit: FloatProperty(name='Angle', default=math.radians(66), min=0, max=pi/2, subtype='ANGLE')


    def invoke(self, context, event):
        settings = univ_settings()
        self.texture_size = min(int(settings.size_x), int(settings.size_y))
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        pad = univ_settings().padding
        layout.label(text=f'Texture Size: {self.texture_size}')
        layout.label(text=f'Padding: {int(clamp(pad + self.add_padding, 0, 100))} ({pad})')
        layout.prop(self, 'add_padding', slider=True)
        layout.prop(self, 'angle_limit', slider=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texture_size = 2048

    def execute(self, context):
        settings = univ_settings()
        kwargs = {
            'angle_limit': self.angle_limit,
            'margin_method': 'FRACTION',
            'island_margin': int(clamp(settings.padding + self.add_padding, 0, 100)) / 2 / self.texture_size,
            'area_weight': 0,
            'correct_aspect': True,
            'scale_to_bounds': False,
        }
        # TODO: Add normalize and correct aspect by modifier
        if context.mode == 'EDIT_MESH':
            umeshes = utypes.UMeshes.calc(self.report, verify_uv=False)
            umeshes.fix_context()
            umeshes.set_sync()
            umeshes.sync_invalidate()

            if not context.active_object:
                return umeshes.update()

            selected, unselected = umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected:
                for umesh in selected:
                    umesh.check_uniform_scale(report=self.report)
                bpy.ops.uv.smart_project(**kwargs)
            elif unselected:
                for umesh in unselected:
                    umesh.check_uniform_scale(report=self.report)
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.smart_project(**kwargs)
                bpy.ops.mesh.select_all(action='DESELECT')
            else:
                self.report({'WARNING'}, 'Not found faces')
                return {'CANCELLED'}
        else:
            meshes = utils.calc_any_unique_obj()
            if not any(obj.data.polygons for obj in meshes):
                self.report({'WARNING'}, 'Not found faces')
                return {'CANCELLED'}

            if not context.active_object or context.active_object.type != 'MESH':
                for obj in meshes:
                    bpy.context.view_layer.objects.active = obj
                    break


            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            for umesh in utypes.UMeshes.calc(self.report, verify_uv=False):
                umesh.check_uniform_scale(report=self.report)

            bpy.ops.mesh.reveal(select=True)
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(**kwargs)
            bpy.ops.object.editmode_toggle()

        return {'FINISHED'}


# noinspection PyTypeHints
class UNIV_OT_Flatten(bpy.types.Operator):
    bl_idname = 'mesh.univ_flatten'
    bl_label = 'Flatten'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Convert 3d coords to 2D from uv map\n\n" \
                     "Context keymaps on button:\n" \
                     "\t\tDefault - Mesh\n" \
                     "\t\tShift - Shape Keys\n" \
                     "\t\tAlt - Modifier" \

    axis: EnumProperty(name="Axis", default='z', items=(
        ('z', 'Bottom', ''),
        ('y', 'Side', ''),
        ('x', 'Front', '')))

    flatten_type: EnumProperty(name="Flatten Type", default='MESH', items=(
                                    ('MESH', 'Mesh', ''),
                                    ('SHAPE_KEY', 'Shape Key', ''),
                                    ('MODIFIER', 'Modifier', '')))
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)
    mix_factor: FloatProperty(name='Mix Factor', default=1, min=0, max=1)
    weld_distance: FloatProperty(name='Weld Distance', default=0.00001, min=0)


    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        if event.alt:
            self.flatten_type = 'MODIFIER'
        elif event.shift:
            self.flatten_type = 'SHAPE_KEY'
        else:
            self.flatten_type = 'MESH'

        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'axis', expand=True)
        layout.prop(self, 'use_correct_aspect')
        if self.flatten_type == 'MODIFIER':
            layout.prop(self, 'mix_factor')
            layout.prop(self, 'weld_distance')
        layout.row(align=True).prop(self, 'flatten_type', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: utypes.UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        import bmesh
        self.umeshes = utypes.UMeshes(report=self.report)
        self.umeshes.fix_context()
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        if not self.umeshes:
            return self.umeshes.update()
        if self.use_correct_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=True)

        if self.umeshes.is_edit_mode:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes
            if not self.umeshes:
                return self.umeshes.update(info='Not found faces for manipulate')

            if self.apply_gn():
                return {'FINISHED'}

            if selected_umeshes:
                for umesh in self.umeshes:
                    uv = umesh.uv
                    split_edges = set()
                    selected_faces = utils.calc_selected_uv_faces(umesh)
                    for f in selected_faces:
                        for crn in f.loops:
                            if not crn.link_loop_radial_prev.face.select or utils.is_boundary_sync(crn, uv):
                                split_edges.add(crn.edge)
                    if split_edges:
                        bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                        for e in split_edges:
                            e.select = True
                        umesh.bm.select_flush(True)
                    if self.flatten_type == 'SHAPE_KEY':
                        continue
                    self.apply_coords(selected_faces, umesh)
                if self.flatten_type == 'SHAPE_KEY':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    for umesh in self.umeshes:
                        self.apply_shape_keys((f for f in umesh.obj.data.polygons if f.select), umesh)
                        umesh.obj.data.update_tag()
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
                    return {'FINISHED'}

            else:
                for umesh in self.umeshes:
                    uv = umesh.uv
                    split_edges = set()
                    visible_faces = utils.calc_visible_uv_faces(umesh)
                    for f in visible_faces:
                        for crn in f.loops:
                            if utils.is_boundary_sync(crn, uv):
                                split_edges.add(crn.edge)
                    bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                    if self.flatten_type == 'SHAPE_KEY':
                        continue
                    self.apply_coords(visible_faces, umesh)

                if self.flatten_type == 'SHAPE_KEY':
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
                    for umesh in self.umeshes:
                        self.apply_shape_keys((f for f in umesh.obj.data.polygons if not f.hide), umesh)
                        umesh.obj.data.update_tag()
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        else:
            if self.apply_gn():
                return {'FINISHED'}

            for umesh in self.umeshes:
                uv = umesh.uv
                split_edges = set()
                for f in umesh.bm.faces:
                    for crn in f.loops:
                        if not utils.is_pair(crn, crn.link_loop_radial_prev, uv):
                            split_edges.add(crn.edge)
                if split_edges:
                    bmesh.ops.split_edges(umesh.bm, edges=list(split_edges))
                    umesh.update()

                if self.flatten_type == 'SHAPE_KEY':
                    umesh.free()
                    self.apply_shape_keys((f for f in umesh.obj.data.polygons), umesh)
                    umesh.obj.data.update_tag()
                else:
                    self.apply_coords(umesh.bm.faces, umesh)
            if self.flatten_type == 'SHAPE_KEY':
                return {'FINISHED'}

        self.umeshes.silent_update()
        self.umeshes.free()
        return {'FINISHED'}

    def apply_coords(self, faces, umesh):
        uv = umesh.uv
        bb3d = utypes.BBox3D.get_from_umesh(umesh)
        bb = bb3d.to_bbox_2d(self.axis)
        max_length = bb.max_length
        if umesh.aspect != 1:
            max_length = self.aspect_to_scale(umesh.aspect).to_2d() * max_length
        delta = Vector((-0.5, -0.5))

        for f in faces:
            for crn in f.loops:
                if self.axis == 'z':
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d()
                elif self.axis == 'y':
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d().zxy
                else:
                    crn.vert.co = ((crn[uv].uv + delta) * max_length).to_3d().xzy

        umesh.bm.normal_update()

    def apply_shape_keys(self, faces, umesh: utypes.UMesh):
        if not umesh.obj.data.shape_keys:
            bb3d = utypes.BBox3D.get_from_umesh(umesh)
        else:
            base_sk_data = umesh.obj.data.shape_keys.key_blocks[0].data
            coords = (base_sk_data[i].co for i in range(len(umesh.obj.data.vertices)))
            bb3d = utypes.BBox3D.calc_bbox(coords)

        bb = bb3d.to_bbox_2d(self.axis)
        max_length = bb.max_length
        if umesh.aspect != 1:
            max_length = self.aspect_to_scale(umesh.aspect).to_2d() * max_length
        delta = Vector((-0.5, -0.5))

        uv_data = umesh.obj.data.uv_layers.active.data

        if not umesh.obj.data.shape_keys:
            umesh.obj.shape_key_add(name="model", from_mix=True)
            sk = umesh.obj.shape_key_add(name="uv", from_mix=True)
        else:
            sk = umesh.obj.data.shape_keys.key_blocks.get('uv')
            if not sk:
                sk = umesh.obj.shape_key_add(name="uv", from_mix=True)

        idx = 0
        for idx, sk_ in enumerate(umesh.obj.data.shape_keys.key_blocks):
            if umesh.obj.data.shape_keys.key_blocks[idx] == sk:
                break

        umesh.obj.active_shape_key_index = idx
        umesh.obj.active_shape_key.value = 1

        corners = umesh.obj.data.loops
        sk_data = sk.data
        for poly in faces:
            for loop_index in poly.loop_indices:
                uv_co = uv_data[loop_index].uv
                vert_index = corners[loop_index].vertex_index

                if self.axis == 'z':
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d()
                elif self.axis == 'y':
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d().zxy
                else:
                    sk_data[vert_index].co = ((uv_co + delta) * max_length).to_3d().xzy

    def apply_gn(self):
        if self.flatten_type == 'MODIFIER':
            if bpy.app.version < (4, 1, 0):
                self.report({'WARNING'}, 'Modifier types is not supported in Blender versions below 4.1')
                return True
            node_group = self.get_flatten_node_group()
            self.create_gn_flatter_modifier(node_group)
            utils.update_area_by_type('VIEW_3D')
            return True
        return False

    def create_gn_flatter_modifier(self, node_group):
        if bpy.app.version >= (5, 2, 0):
            axis = {'z': 'Bottom', 'y': 'Front', 'x': 'Side'}
        else:
            axis = {'z': 2, 'y': 3, 'x': 4}
        for umesh in self.umeshes:
            has_flatten_modifier = False

            for m in umesh.obj.modifiers:
                if not isinstance(m, bpy.types.NodesModifier):
                    continue
                if m.name.startswith('UniV Flatten'):
                    has_flatten_modifier = True
                    if m.node_group != node_group:
                        m.node_group = node_group

                    gn_mod = utils.GN(m)
                    gn_mod['Socket_2'] = umesh.uv.name
                    gn_mod['Socket_3'] = axis[self.axis]
                    gn_mod['Socket_4'] = self.aspect_to_scale(umesh.aspect)
                    gn_mod['Socket_5'] = self.mix_factor
                    umesh.obj.update_tag()
                    break

            if not has_flatten_modifier:
                m = umesh.obj.modifiers.new(name='UniV Flatten', type='NODES')
                m.node_group = node_group
                gn_mod = utils.GN(m)
                gn_mod['Socket_2'] = umesh.uv.name
                gn_mod['Socket_3'] = axis[self.axis]
                gn_mod['Socket_4'] = self.aspect_to_scale(umesh.aspect)
                gn_mod['Socket_5'] = self.mix_factor

    def get_flatten_node_group(self):
        """Get exist flatten node group"""
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Flatten'):
                if self.flatten_node_group_is_changed(ng):
                    print(f"UniV: Flatten: Node Group {ng.name!r} is changed.")
                    if ng.users == 0:
                        bpy.data.node_groups.remove(ng)
                else:
                    return ng
        return self._create_flatten_node_group()

    @staticmethod
    def flatten_node_group_is_changed(ng):
        items = ng.interface.items_tree
        if len(items) != 6:
            return True
        expect_types = (
            'Geometry',
            'Geometry',
            'String',
            'Menu',
            'Vector',
            'Float'
        )
        for str_typ, item in zip(expect_types, items):
            bpy_type = (getattr(bpy.types, 'NodeTreeInterfaceSocket' + str_typ))
            if not isinstance(item.rna_type, bpy_type):
                return True

        for node in ng.nodes:
            if node.type == "GROUP":
                if not node.node_tree:
                    return True
                if node.node_tree.name.startswith(utils.GN_IsUVEdgeBoundary.name):
                    if utils.GN_IsUVEdgeBoundary.is_changed(node.node_tree):
                        print(f"UniV: Flatten: Node Group {node.node_tree.name!r} is changed.")
                        return True

        if len(ng.nodes) != 25:
            return True

        sockets_count = sum(sk.is_linked for n in ng.nodes for sk in n.inputs)
        if sockets_count != 44:
            return True

        all_nodes_types = {'VECT_MATH', 'SET_POSITION', 'SEPXYZ', 'SPLIT_EDGES', 'GROUP_OUTPUT', 'INDEX_SWITCH',
                           'COMBXYZ', 'MATH', 'BOUNDING_BOX', 'MENU_SWITCH', 'MIX', 'SWITCH', 'GROUP', 'GROUP_INPUT',
                           'POSITION', 'INPUT_ATTRIBUTE'}

        if {n.type for n in ng.nodes} != all_nodes_types:
            return True

        return False

    @staticmethod
    def _create_flatten_node_group():
        bb = bpy.data.node_groups.new(type='GeometryNodeTree', name="UniV Flatten")
        # bb = bpy.data.node_groups["UniV Flatten"]
        # bb.nodes.clear()
        # bb.interface.clear()
        # for item in list(bb.interface.items_tree):
        #     bb.interface.remove(item)

        # Interface
        # Socket Geometry
        geometry_socket = bb.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        geometry_socket.attribute_domain = 'POINT'

        # Socket Geometry
        geometry_socket_1 = bb.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        geometry_socket_1.attribute_domain = 'POINT'

        # Socket UV Map
        uv_map_socket = bb.interface.new_socket(name="UV Map", in_out='INPUT', socket_type='NodeSocketString')
        uv_map_socket.default_value = "UVMap"
        uv_map_socket.subtype = 'NONE'
        uv_map_socket.attribute_domain = 'POINT'

        # Socket Axis
        axis_socket = bb.interface.new_socket(name="Axis", in_out='INPUT', socket_type='NodeSocketMenu')
        axis_socket.attribute_domain = 'POINT'
        axis_socket.force_non_field = True

        # Socket Aspect Ratio
        aspect_ratio_socket = bb.interface.new_socket(
            name="Aspect Ratio", in_out='INPUT', socket_type='NodeSocketVector')
        aspect_ratio_socket.default_value = (1.0, 1.0, 0.0)
        aspect_ratio_socket.min_value = 0.01
        aspect_ratio_socket.max_value = 10000
        aspect_ratio_socket.force_non_field = True

        # Socket Factor
        factor_socket = bb.interface.new_socket(name="Factor", in_out='INPUT', socket_type='NodeSocketFloat')
        factor_socket.default_value = 1.0
        factor_socket.min_value = 0.0
        factor_socket.max_value = 1.0
        factor_socket.subtype = 'FACTOR'
        factor_socket.attribute_domain = 'POINT'
        factor_socket.force_non_field = True

        # initialize UniV Flatten nodes
        # node Group Input
        group_input = bb.nodes.new("NodeGroupInput")

        # node Group Output
        group_output = bb.nodes.new("NodeGroupOutput")
        group_output.is_active_output = True

        # Is Boundary
        is_uv_bound = bb.nodes.new("GeometryNodeGroup")
        is_uv_bound.node_tree = utils.GN_IsUVEdgeBoundary.get()
        is_uv_bound.location = (400, -50)

        # node Split Edges
        split_edges = bb.nodes.new("GeometryNodeSplitEdges")
        split_edges.name = "Split Edges"

        # node Set Position
        set_position = bb.nodes.new("GeometryNodeSetPosition")
        set_position.name = "Set Position"

        # node Named Attribute
        named_attribute = bb.nodes.new("GeometryNodeInputNamedAttribute")
        named_attribute.label = "UV Attribute"
        named_attribute.data_type = 'FLOAT_VECTOR'

        # node Bounding Box
        bounding_box = bb.nodes.new("GeometryNodeBoundBox")

        # node Vector Math
        vector_math = bb.nodes.new("ShaderNodeVectorMath")
        vector_math.name = "Vector Math"
        vector_math.operation = 'SUBTRACT'

        # node Separate XYZ
        separate_xyz_widths = bb.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz_widths.label = "BBox Widths"

        # Remap
        remap_to_center = bb.nodes.new("ShaderNodeVectorMath")
        remap_to_center.label = "Remap to centered range"
        remap_to_center.operation = 'SUBTRACT'
        remap_to_center.inputs[1].default_value = (0.5, 0.5, 0.0)

        # Scale UV
        max_length_1 = bb.nodes.new("ShaderNodeMath")
        max_length_1.label = "Max Length"
        max_length_1.operation = 'MAXIMUM'

        max_length_2 = bb.nodes.new("ShaderNodeMath")
        max_length_2.label = "Max Length"
        max_length_2.operation = 'MAXIMUM'

        max_length_3 = bb.nodes.new("ShaderNodeMath")
        max_length_3.label = "Max Length"
        max_length_3.operation = 'MAXIMUM'

        scale_uv = bb.nodes.new("ShaderNodeVectorMath")
        scale_uv.label = "Scale UV"
        scale_uv.operation = 'MULTIPLY'

        # node Menu Switch
        menu_switch = bb.nodes.new("GeometryNodeMenuSwitch")
        menu_switch.label = "Axis Index Menu"
        menu_switch.data_type = 'INT'
        menu_switch.enum_items.clear()

        for idx, item in enumerate(['Bottom', 'Side', 'Front']):
            menu_switch.enum_items.new(item)
            menu_switch.inputs[idx + 1].default_value = idx

        # node Index Switch
        index_switch = bb.nodes.new("GeometryNodeIndexSwitch")
        index_switch.label = "Get scale from bbox"
        index_switch.name = "Index Switch"
        index_switch.data_type = 'FLOAT'
        index_switch.index_switch_items.clear()
        for _ in range(3):
            index_switch.index_switch_items.new()

        # node Index Switch
        index_switch_001 = bb.nodes.new("GeometryNodeIndexSwitch")
        index_switch_001.label = "Switch scale by axis"
        index_switch_001.data_type = 'VECTOR'
        index_switch_001.index_switch_items.clear()
        for _ in range(3):
            index_switch_001.index_switch_items.new()

        # node Side swizzle separate
        separate_xyz_001 = bb.nodes.new("ShaderNodeSeparateXYZ")

        # node Side swizzle combine
        combine_xyz = bb.nodes.new("ShaderNodeCombineXYZ")

        # node Front swizzle separate
        separate_xyz_002 = bb.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz_002.label = "Front swizzle separate"

        # node Front swizzle combine
        combine_xyz_001 = bb.nodes.new("ShaderNodeCombineXYZ")
        combine_xyz_001.label = "Front swizzle combine"

        # node Aspect Ratio Scale
        vector_math_004 = bb.nodes.new("ShaderNodeVectorMath")
        vector_math_004.label = "Aspect Ratio Scale"
        vector_math_004.operation = 'MULTIPLY'

        # node Factor
        mix_factor = bb.nodes.new("ShaderNodeMix")
        mix_factor.label = "Factor"
        mix_factor.blend_type = 'MIX'
        mix_factor.data_type = 'VECTOR'
        mix_factor.factor_mode = 'UNIFORM'

        # node Position
        position = bb.nodes.new("GeometryNodeInputPosition")
        position.name = "Position"

        # node No UVMap case
        switch = bb.nodes.new("GeometryNodeSwitch")
        switch.label = "No UVMap case"
        switch.input_type = 'FLOAT'
        switch.inputs[1].default_value = 0.0

        # Set locations
        group_input.location = (-1465, -10)
        group_output.location = (1350, 30)
        split_edges.location = (685, 1)
        set_position.location = (1138, 28)
        named_attribute.location = (-782, -155)
        bounding_box.location = (-1170, -592)
        vector_math.location = (-978, -587)
        separate_xyz_widths.location = (-785, -590)
        max_length_1.location = (-585, -395)
        remap_to_center.location = (-585, -145)
        scale_uv.location = (42, -192)
        menu_switch.location = (-1170, -380)
        index_switch.location = (-360, -460)
        max_length_2.location = (-586, -560)
        index_switch_001.location = (685, -120)
        separate_xyz_001.location = (268, -250)
        combine_xyz.location = (460, -250)
        max_length_3.location = (-590, -730)
        separate_xyz_002.location = (268, -400)
        combine_xyz_001.location = (460, -400)
        vector_math_004.location = (-165, -460)
        mix_factor.location = (920, -40)
        position.location = (685, -335)
        switch.location = (680, 160)

        # initialize bb links
        new_links = utils.NewLinks(bb)

        new_links(is_uv_bound) >> (split_edges, 1)  # UVMap > Select
        new_links(group_input, 1) >> is_uv_bound  # Attribute Name > Attribute Name
        new_links(group_input) >> split_edges  # Geometry > Mesh
        new_links(split_edges) >> set_position  # Mesh > Geometry
        new_links(group_input) >> bounding_box  # Geometry > Geometry
        new_links(named_attribute) >> remap_to_center  # Attribute > Vector
        new_links(remap_to_center) >> scale_uv  # Vector > Vector
        new_links(vector_math) >> separate_xyz_widths  # Vector > Vector
        new_links(separate_xyz_widths) >> max_length_1  # X > Value
        new_links(separate_xyz_widths, 1) >> (max_length_1, 1)  # Y > Value
        new_links(max_length_1) >> (index_switch, 1)  # Value > 0
        new_links(separate_xyz_widths, 1) >> max_length_2  # Y > Value
        new_links(separate_xyz_widths, 2) >> (max_length_2, 1)  # Z > Value
        new_links(max_length_2) >> (index_switch, 2)  # Value > 1
        new_links(separate_xyz_widths) >> max_length_3  # X > Value
        new_links(separate_xyz_widths, 2) >> (max_length_3, 1)  # z > Value
        new_links(max_length_3) >> (index_switch, 3)  # Value > 2
        new_links(scale_uv) >> (index_switch_001, 1)  # Vector > 0
        new_links(scale_uv) >> separate_xyz_001  # Vector > Vector
        new_links(scale_uv) >> separate_xyz_002  # Vector > Vector

        new_links(bounding_box, 2) >> vector_math  # Max > Vector
        new_links(bounding_box, 1) >> (vector_math, 1)  # Min > Vector
        new_links(group_input, 1) >> named_attribute  # UV Map > Name
        new_links(set_position) >> group_output  # Geometry > Geometry

        new_links(group_input, 2) >> menu_switch  # Axis > Menu
        new_links(menu_switch) >> index_switch  # Output > Index
        new_links(separate_xyz_001, 2) >> combine_xyz  # Z > X
        new_links(separate_xyz_001) >> (combine_xyz, 1)  # X > Y
        new_links(separate_xyz_001, 1) >> (combine_xyz, 2)  # Y > Z
        new_links(combine_xyz) >> (index_switch_001, 2)  # Vector > 1
        new_links(menu_switch) >> index_switch_001  # Output > Index
        new_links(separate_xyz_002) >> combine_xyz_001  # X > X
        new_links(separate_xyz_002, 2) >> (combine_xyz_001, 1)  # Z > Y
        new_links(separate_xyz_002, 1) >> (combine_xyz_001, 2)  # Y > Z
        new_links(combine_xyz_001) >> (index_switch_001, 3)  # Vector > 2
        new_links(index_switch) >> vector_math_004  # Output > Vector
        new_links(vector_math_004) >> (scale_uv, 1)  # Vector > Vector
        new_links(group_input, 3) >> (vector_math_004, 1)  # Aspect Ratio > Vector
        new_links(index_switch_001) >> (mix_factor, 5)  # Output > B
        new_links(position) >> (mix_factor, 4)  # Position > A
        new_links(mix_factor, 1) >> (set_position, 2)  # Result > Position
        new_links(named_attribute, 1) >> switch  # Exists > Switch
        new_links(group_input, 4) >> (switch, 2)  # Factor > True
        new_links(switch) >> mix_factor  # Output > Factor
        return bb

    @staticmethod
    def aspect_to_scale(aspect_y):
        if aspect_y > 1:
            return Vector((aspect_y, 1, 0))
        else:
            return Vector((1, 1/aspect_y, 0))


class UNIV_OT_FlattenCleanup(bpy.types.Operator):
    bl_idname = 'mesh.univ_flatten_clean_up'
    bl_label = 'Flatten'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Remove Flatten modifiers and shape keys and unused nodes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: utypes.UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        self.umeshes = utypes.UMeshes.calc(report=self.report, verify_uv=False)
        self.umeshes.fix_context()
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        removed_sk_counter = 0
        removed_modifiers = 0
        removed_geometry_nodes = 0
        if self.umeshes:
            has_shape_keys = self.has_flatten_shape_keys()
            if has_shape_keys:
                if self.umeshes.is_edit_mode:
                    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

            for umesh in self.umeshes:
                # Remove shape keys
                if umesh.obj.data.shape_keys and umesh.obj.data.shape_keys.key_blocks.get('uv'):
                    if len(umesh.obj.data.shape_keys.key_blocks) == 2:
                        umesh.obj.shape_key_clear()
                        removed_sk_counter += 1
                    else:
                        sk = umesh.obj.data.shape_keys.key_blocks.get('uv')
                        umesh.obj.shape_key_remove(sk)
                        removed_sk_counter += 1

                # Remove modifiers
                for m in reversed(umesh.obj.modifiers):
                    if isinstance(m, bpy.types.NodesModifier) and m.name.startswith('UniV Flatten'):
                        umesh.obj.modifiers.remove(m)
                        removed_modifiers += 1

                umesh.obj.data.update_tag()
                if self.umeshes.is_edit_mode:
                    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

        # Remove geometry nodes
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Flatten'):
                if ng.users == 0:
                    bpy.data.node_groups.remove(ng)
                    removed_geometry_nodes += 1

        info = ''
        if removed_sk_counter:
            info += f"Removed {removed_sk_counter} shape keys. "
        if removed_modifiers:
            info += f"Removed {removed_modifiers} modifiers. "
        if not removed_modifiers and removed_geometry_nodes:
            info += f"Removed {removed_geometry_nodes} node groups."

        if info:
            self.report({'INFO'}, info)

        return {'FINISHED'}

    def has_flatten_shape_keys(self):
        for umesh in self.umeshes:
            if umesh.obj.data.shape_keys:
                if umesh.obj.data.shape_keys.key_blocks.get('uv'):
                    return True
        return False


class UNIV_OT_WrapProject(bpy.types.Operator):
    bl_idname = "mesh.univ_wrap"
    bl_label = "Wrap"
    bl_description = "Swap UV to XYZ coordinates"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        active = context.active_object
        if not active or active.type != "MESH":
            self.report({'WARNING'}, "Not found active mesh object")
            return {"FINISHED"}

        for_wrapping_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH" and obj != active]

        # Check existing flatten.
        if not active.data.shape_keys or len(active.data.shape_keys.key_blocks) <= 1:
            for m in active.modifiers:
                if isinstance(m, bpy.types.NodesModifier) and m.name.startswith('UniV Flatten'):
                    gn_mod = utils.GN(m, print_missed_socket=True)
                    if 'Socket_5' in gn_mod:
                        break
            else:
                for_wrapping_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
                count = self.remove_surface_deform(for_wrapping_objects)
                if count:
                    self.report({"INFO"}, f"Not found shape keys or flatten modifier. "
                                          f"But removed {count!r} surface deform modifiers.")
                else:
                    self.report({"WARNING"}, "Not found shape keys or flatten modifier.")

                return {"CANCELLED"}

        if self.is_toggle(active, for_wrapping_objects):
            return {"FINISHED"}

        if len(for_wrapping_objects) == 0:
            self.report({"WARNING"}, "No meshes found for wrapping")
            return {"FINISHED"}

        for obj in for_wrapping_objects:
            # Delete previous modifiers
            for modifier in reversed(obj.modifiers):
                if modifier.type == 'SURFACE_DEFORM':
                    obj.modifiers.remove(modifier)

            # Add mesh modifier
            modifier_deform = obj.modifiers.new(name="SurfaceDeform", type='SURFACE_DEFORM')
            modifier_deform.target = active

            obj.select_set(state=True, view_layer=None)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.surfacedeform_bind(modifier="SurfaceDeform")


        # Set shape keys to 1.0.
        if active.data.shape_keys and len(active.data.shape_keys.key_blocks) >= 2:
            block = active.data.shape_keys.key_blocks[1]
            block.value = 0.0

        # Set flatten modifier to 0.0.
        for m in active.modifiers:
            if isinstance(m, bpy.types.NodesModifier) and m.name.startswith('UniV Flatten'):
                gn_mod = utils.GN(m, print_missed_socket=True)
                if 'Socket_5' in gn_mod:
                    gn_mod['Socket_5'] = 0.0

        bpy.context.view_layer.objects.active = active
        active.update_tag()
        return {"FINISHED"}

    @classmethod
    def is_toggle(cls, active, for_wrapping_objects):
        is_unflatten = False
        if active.data.shape_keys and len(active.data.shape_keys.key_blocks) >= 2:
            block = active.data.shape_keys.key_blocks[1]
            if block.value == 0.0:
                block.value = 1.0
                is_unflatten = True

        # Set flatten modifier to 1.0.
        for m in active.modifiers:
            if isinstance(m, bpy.types.NodesModifier) and m.name.startswith('UniV Flatten'):
                gn_mod = utils.GN(m, print_missed_socket=True)
                if 'Socket_5' in gn_mod:
                    if gn_mod['Socket_5'] == 0.0:
                        gn_mod['Socket_5'] = 1.0
                        is_unflatten = True

        if is_unflatten:
            active.update_tag()
            cls.remove_surface_deform(for_wrapping_objects)
        return is_unflatten

    @staticmethod
    def remove_surface_deform(for_wrapping_objects):
        count = 0
        for obj in for_wrapping_objects:
            # Delete previous modifiers
            for modifier in reversed(obj.modifiers):
                if modifier.type == 'SURFACE_DEFORM':
                    obj.modifiers.remove(modifier)
                    count += 1
        return count
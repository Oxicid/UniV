# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math

from math import pi, cos, sin
from bl_math import clamp
from . import transform
from .. import utils
from .. import types
from ..types import BBox, MeshIsland, MeshIslands
from bpy.props import *
from collections.abc import Callable
from mathutils import Vector, Euler, Matrix
from ..preferences import univ_settings

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

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.individual = event.shift
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = 'No found faces for manipulate'
        self.has_selected: bool = True
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc(self.report, verify_uv=False)
        self.umeshes.set_sync()
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

        points = []
        points_append = points.append

        if self.orient or self.crop:
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

    def crop_islands(self, adv_islands_of_mesh: list[types.AdvIsland], bbox):
        if not self.crop or not adv_islands_of_mesh:
            return

        if self.mark_seam:
            for isl in adv_islands_of_mesh:
                isl.mark_seam()

        pivot = bbox.left_bottom
        for island in adv_islands_of_mesh:
            if island.umesh.aspect != 1.0:
                if island.umesh.aspect < 1:
                    bbox_ = bbox.copy()
                    bbox_.scale(Vector((1, island.umesh.aspect)), pivot=pivot)
                else:
                    bbox_ = bbox.copy()
                    bbox_.scale(Vector((1/island.umesh.aspect, 1)), pivot=pivot)
            else:
                bbox_ = bbox
            transform.UNIV_OT_Crop.crop_ex('XY', bbox_, inplace=False, islands_of_mesh=[island], offset=Vector((0, 0)), padding=0.001, proportional=True)

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

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):
        self.layout.prop(self, 'scale', slider=True)
        col = self.layout.column(align=True)
        col.prop(self, 'scale_individual', expand=True, slider=True)
        col.prop(self, 'rotation', expand=True, slider=True)
        col.prop(self, 'move', expand=True)
        col.prop(self, 'avoid_flip')
        col.separator()
        col.prop(self, 'use_correct_aspect', toggle=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_edit_mode: bool = bpy.context.mode == 'EDIT_MESH'
        self.has_selected: bool = True
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc(self.report, verify_uv=False)
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
        if (aspect := (utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0)) >= 1.0:
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
        if (res := cam_inv.inverted(None)) is not None:
            uci.cam_inv = res
            # normal projection
            if rot_mat:
                uci.rot_mat = rot_mat.copy()
            uci.do_rot_mat = bool(rot_mat)

            # also make aspect ratio adjustment factors
            if winx > winy:
                uci.aspect.x = 1.0
                uci.aspect.y = winx/winy
            else:
                uci.aspect.x = winy/winx
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


class UNIV_OT_ViewProject(bpy.types.Operator):
    bl_idname = "mesh.univ_view_project"
    bl_label = "View"
    bl_description = "Projection by View"
    bl_options = {'REGISTER', 'UNDO'}

    camera_bounds: BoolProperty(name='Camera Bounds', default=False)
    use_crop: BoolProperty(name='Crop', default=True, description='Packs the islands into a base tile')
    use_orthographic: BoolProperty(name='Use Orthographic', default=False)
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

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
        self.umeshes: types.UMeshes | None = None
        self.region = None
        self.area = None
        self.rv3d = None
        self.v3d = None
        self.faces_calc_type: Callable = Callable
        self.camera = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc(self.report, verify_uv=False)
        self.umeshes.set_sync()

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
                    return {'FINISHED'}
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
        scale_x = ((1.0 - padding) / w) if (w := bbox.width) else 1
        scale_y = ((1.0 - padding) / h) if (h := bbox.height) else 1

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


class UNIV_OT_SmartProject(bpy.types.Operator):
    bl_idname = 'mesh.univ_smart_project'
    bl_label = 'Smart'
    bl_description = 'Smart Projection'
    bl_options = {'REGISTER', 'UNDO'}

    add_padding: IntProperty(name='Additional Padding', default=0, min=-16, max=16, subtype='PIXEL')
    angle_limit: FloatProperty(name='Angle', default=math.radians(66), min=0, max=pi/2, subtype='ANGLE')

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

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
            umeshes = types.UMeshes.calc(self.report, verify_uv=False)
            umeshes.fix_context()
            umeshes.set_sync()

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
            if not any(obj.data.polygons for obj in bpy.context.selected_objects if obj.type == 'MESH'):
                self.report({'WARNING'}, 'Not found faces')
                return {'CANCELLED'}
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)

            for umesh in types.UMeshes.calc(self.report, verify_uv=False):
                umesh.check_uniform_scale(report=self.report)

            umeshes = types.UMeshes.calc(self.report, verify_uv=False)
            umeshes.fix_context()

            bpy.ops.mesh.reveal(select=True)
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(**kwargs)
            bpy.ops.object.editmode_toggle()

        return {'FINISHED'}

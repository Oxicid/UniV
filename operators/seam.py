import bpy
import math
import bl_math

from .. import utils


class UNIV_OT_Cut(bpy.types.Operator):
    bl_idname = "mesh.univ_cut"
    bl_label = "Cut"
    bl_description = "Cut selected"
    bl_options = {'REGISTER', 'UNDO'}

    addition: bpy.props.BoolProperty(name='Addition', default=True)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self):
        self.sync = utils.sync()
        self.umeshes: utils.UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = utils.UMeshes(report=self.report)

        if context.area.ui_type == 'UV':
            if self.sync:
                self.cut_view_2d_sync()
            else:
                self.cut_view_2d_no_sync()
        else:
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

    def cut_view_2d_sync(self):
        for umesh in self.umeshes:
            if umesh.is_full_edge_deselected:
                umesh.update_tag = False
                continue

            uv = umesh.uv_layer
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
                        elif not (_shared_crn.face.select and crn.face.select):
                            crn.edge.seam = True
                        elif not shared_is_linked(crn, _shared_crn, uv):
                            crn.edge.seam = True
                        elif not self.addition:
                            crn.edge.seam = False

    def cut_view_2d_no_sync(self):
        for umesh in self.umeshes:
            if umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue
            umesh.tag_selected_faces()

            uv = umesh.uv_layer
            for f in umesh.bm.faces:
                if not f.select:
                    continue
                for crn in f.loops:
                    if not crn[uv].select_edge or not crn[uv].select:
                        continue
                    elif (_shared_crn := crn.link_loop_radial_prev) == crn:
                        crn.edge.seam = True
                    elif not (_shared_crn.face.tag and crn.face.tag):
                        crn.edge.seam = True
                    elif not utils.shared_is_linked(crn, _shared_crn, uv):
                        crn.edge.seam = True
                    elif not self.addition:
                        crn.edge.seam = False

class UNIV_OT_Angle(bpy.types.Operator):
    bl_idname = "mesh.univ_angle"
    bl_label = "Angle"
    bl_description = "Seams by angle, sharps, materials, borders"
    bl_options = {'REGISTER', 'UNDO'}

    selected: bpy.props.BoolProperty(name='Selected', default=False)
    addition: bpy.props.BoolProperty(name='Addition', default=True)
    borders: bpy.props.BoolProperty(name='Borders', default=False)
    obj_smooth: bpy.props.BoolProperty(name='Obj Smooth', default=True)
    mtl: bpy.props.BoolProperty(name='Mtl', default=True)
    by_sharps: bpy.props.BoolProperty(name='By Sharps', default=True)
    mark_sharps: bpy.props.BoolProperty(name='Mark Sharps', default=True)
    angle: bpy.props.FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'selected')
        layout.prop(self, 'addition')
        layout.prop(self, 'borders')
        layout.prop(self, 'mtl')
        layout.prop(self, 'by_sharps')
        layout.prop(self, 'mark_sharps')
        layout.prop(self, 'obj_smooth')
        layout.prop(self, 'angle', slider=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.addition = event.shift
        return self.execute(context)

    def __init__(self):
        self.sync = utils.sync()
        self.umeshes: utils.UMeshes | None = None

    def execute(self, context) -> set[str]:
        self.umeshes = utils.UMeshes(report=self.report)

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

            if umesh.is_full_face_selected:
                faces = (_f for _f in umesh.bm.faces)
            elif self.selected:
                faces = (_f for _f in umesh.bm.faces if _f.select)
            else:  # visible
                faces = (_f for _f in umesh.bm.faces if not _f.hide)

            for f in faces:
                for crn in f.loops:
                    if not crn.edge.is_manifold:  # boundary
                        if self.borders or len(crn.edge.link_faces) > 2:
                            crn.edge.seam = True
                        elif not self.addition:
                            crn.edge.seam = False
                    elif crn.edge.calc_face_angle() >= angle:  # Skip by angle
                        crn.edge.seam = True
                        if self.mark_sharps:
                            crn.edge.smooth = False
                    elif self.borders and crn.link_loop_radial_prev.face.hide:
                        crn.edge.seam = True
                    elif self.by_sharps and not crn.edge.smooth:
                        crn.edge.seam = True
                    elif self.mtl and crn.face.material_index != crn.link_loop_radial_prev.face.material_index:
                        crn.edge.seam = True
                    elif not self.addition:
                        crn.edge.seam = False

        self.umeshes.update()
        return {'FINISHED'}

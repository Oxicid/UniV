import bpy
import math
import bl_math

from .. import utils


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
        self.umeshes.set_sync()

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
                    if crn == (shared_crn := crn.link_loop_radial_prev):  # boundary
                        if self.borders:
                            crn.edge.seam = True
                        elif not self.addition:
                            crn.edge.seam = False
                    elif crn.edge.calc_face_angle() >= angle:  # Skip by angle
                        crn.edge.seam = True
                        if self.mark_sharps:
                            crn.edge.smooth = False
                    elif self.borders and shared_crn.face.hide:
                        crn.edge.seam = True
                    elif self.by_sharps and not crn.edge.smooth:
                        crn.edge.seam = True
                    elif self.mtl and crn.face.material_index != shared_crn.face.material_index:
                        crn.edge.seam = True
                    elif not self.addition:
                        crn.edge.seam = False

        self.umeshes.update()
        return {'FINISHED'}

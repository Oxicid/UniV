import bpy
import gpu
import blf  # noqa
import typing
import gpu_extras
from math import inf
from ..types import KDMesh, KDMeshes, Islands, UnionIslands, FaceIsland, View2D
from .. import utils
from ..utils import UMeshes
from mathutils import Vector


class UNIV_OT_QuickSnap(bpy.types.Operator):
    bl_idname = "uv.univ_quick_snap"
    bl_label = "Quick Snap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def __init__(self):
        self.sync: bool = utils.sync()
        self.umeshes: UMeshes | None = None
        self.kdmeshes: KDMeshes | None = None
        self.rmesh: KDMesh | None = None
        self.area: bpy.types.Area | None = None
        self.view: bpy.types.View2D | None = None
        self.mouse_position: Vector = Vector((0.0, 0.0, 0.0))
        self.prev_elem_position: Vector = Vector((0.0, 0.0, 0.0))
        self.handler: typing.Any = None
        self.shader: gpu.types.GPUShader | None = None
        self.batch: gpu.types.GPUBatch | None = None
        self.points: list[Vector] = []
        self.nearest_point: list[Vector] = [Vector((inf, inf, inf))]
        self.extended: bool | None = None
        self.last_set_position: Vector = Vector()
        self.radius: float = 0.0
        self.dragged: bool = False
        self.move_object: UnionIslands | None = None

    def modal(self, context, event):
        try:
            return self.modal_ex(context, event)
        except Exception:  # noqa
            import traceback
            traceback.print_exc()
            self.umeshes.silent_update()
            return {'FINISHED'}

    def modal_ex(self, _context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
            return {'PASS_THROUGH'}

        self.calc_radius_and_mouse_position(event)

        if event.type in ('ESC', 'RIGHTMOUSE') or (event.type == 'LEFTMOUSE' and event.value == 'RELEASE'):
            return self.exit()

        if self.dragged:
            if self.mouse_position == self.prev_elem_position.to_3d():
                return {'RUNNING_MODAL'}

            points = self.kdmeshes.find_range(self.mouse_position, self.radius)
            self.points = self.kdmeshes.range_to_coords(points)
            self.refresh_draw_points()

            if not event.ctrl and (kd_data := self.find_nearest_target_pt()):
                self.nearest_point[0] = kd_data.pt

                pos = kd_data.pt.to_2d()
                self.move_object.set_position(pos, self.prev_elem_position)
                self.prev_elem_position = pos
            else:
                self.nearest_point[0] = self.mouse_position

                pos = self.mouse_position.to_2d()
                self.move_object.set_position(pos, self.prev_elem_position)
                self.prev_elem_position = pos
            self.umeshes.silent_update()
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'MOUSEMOVE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:

            points = self.kdmeshes.find_range(self.mouse_position, self.radius)
            self.points = self.kdmeshes.range_to_coords(points)
            self.refresh_draw_points()

            if kd_data := self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius):
                self.nearest_point[0] = kd_data.pt
            else:
                self.nearest_point[0] = Vector((inf, inf, inf))

            self.area.tag_redraw()

        elif event.type == 'LEFTMOUSE':  # and event.value == 'CLICK_DRAG':

            kd_data = self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius)
            if not kd_data:
                self.report({'INFO'}, 'Not found nearest elem')
                return {'PASS_THROUGH'}

            self.dragged = True
            islands = []
            if self.extended:
                for kdmesh in self.kdmeshes:
                    islands.extend(kdmesh.islands)
                self.move_object = UnionIslands(islands)
                self.recalc_island_kd_meshes()
            else:
                self.move_object = kd_data.extract_drag_island()
                if not kd_data.kdmesh.islands:
                    self.kdmeshes.kdmeshes.remove(kd_data.kdmesh)

            self.move_object.set_position(self.mouse_position.to_2d(), kd_data.pt.to_2d())
            self.prev_elem_position = self.mouse_position.to_2d()

            self.calc_update_meshes()
            self.umeshes.silent_update()
            self.area.tag_redraw()

            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.area = context.area
        if self.area.ui_type != 'UV':
            self.report({'INFO'}, 'Area must be UV')
            return {'CANCELLED'}
        self.view = context.region.view2d
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.refresh_draw_points()
        self.register_draw()

        self.umeshes = UMeshes()

        self.calc_island_kdmeshes(extended=True)
        self.extended = True
        if not self.kdmeshes:
            self.calc_island_kdmeshes(extended=False)
            self.extended = False
            if not self.kdmeshes:
                return {'CANCELLED'}

        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def find_nearest_target_pt(self):
        kd_data = self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius)
        m_pos = self.mouse_position.to_2d()
        zoom = View2D.get_zoom(self.view)
        divider = 1/8 if zoom <= 1600 else 1 / 64
        divider = divider if zoom <= 12800 else 1 / 64 / 8
        pos = Vector(utils.round_threshold(v, divider) for v in m_pos)
        dist = (pos - m_pos).length
        if bpy.context.scene.tool_settings.use_snap_uv_grid_absolute \
                and dist <= self.radius and dist < kd_data.distance:
            kd_data.found = (pos, 0, 0.0)
            kd_data.kdmesh = True
        return kd_data

    def register_draw(self):
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(self.univ_quick_snap_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def calc_island_kdmeshes(self, extended=False):
        kdmeshes = []
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, self.sync, extended=extended):
                kdmesh = KDMesh(umesh, islands)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def recalc_island_kd_meshes(self):
        kdmeshes = []
        for umesh in self.umeshes:
            if islands := Islands.calc_non_selected(umesh.bm, umesh.uv_layer, self.sync):
                kdmesh = KDMesh(umesh, islands)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_radius_and_mouse_position(self, event):
        mouse_position = Vector(self.view.region_to_view(event.mouse_region_x, event.mouse_region_y))
        self.radius = (Vector(self.view.region_to_view(event.mouse_region_x + 18, event.mouse_region_y)) - mouse_position).length
        self.mouse_position = mouse_position.to_3d()

    def univ_quick_snap_draw_callback(self):
        gpu.state.blend_set('ALPHA')
        gpu.state.point_size_set(4)
        self.shader.bind()
        self.shader.uniform_float("color", (1, 1, 0, 0.5))
        self.batch.draw(self.shader)

        batch_nearest = gpu_extras.batch.batch_for_shader(self.shader, 'POINTS', {"pos": self.nearest_point})
        self.shader.uniform_float("color", (1, 0.2, 0, 1))
        batch_nearest.draw(self.shader)

        # print(self.area.x)
        # font_id = 0
        # blf.size(font_id, 350)
        # blf.position(font_id, 0, 0, 0)
        # scale = 0.0001
        # blf.color(font_id, 0.8, 0., .0, 1)
        # text = str(self.mouse_position)
        #
        # with gpu.matrix.push_pop():
        #     gpu.matrix.translate(self.mouse_position)
        #     gpu.matrix.scale((scale, scale))
        #     blf.draw(font_id, text)

        self.area.tag_redraw()
        gpu.state.blend_set('NONE')

    def refresh_draw_points(self):
        self.batch = gpu_extras.batch.batch_for_shader(self.shader, 'POINTS', {"pos": self.points})

    def exit(self):
        if not (self.handler is None):
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler, 'WINDOW')
            self.area.tag_redraw()
        return {'FINISHED'}

    def calc_update_meshes(self):
        if isinstance(self.move_object, FaceIsland):
            for umesh in self.umeshes:
                if umesh.bm == self.move_object.bm:
                    self.umeshes.umeshes = [umesh]
                    return

        elif isinstance(self.move_object, UnionIslands):
            bmeshes = {isl.bm for isl in self.move_object}
            umeshes = []
            for umesh in self.umeshes:
                if umesh.bm in bmeshes:
                    umeshes.append(umesh)
            assert umeshes
            self.umeshes.umeshes = umeshes

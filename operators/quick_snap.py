import bpy
import gpu
import blf
import typing
import gpu_extras
from math import inf
from bmesh.types import BMFace

from ..preferences import debug
from ..types import KDMesh, KDData, KDMeshes, Islands, UnionIslands, FaceIsland, View2D, LoopGroup, UnionLoopGroup
from .. import utils
from ..utils import UMeshes
from mathutils import Vector


class UNIV_OT_QuickSnap(bpy.types.Operator):
    bl_idname = "uv.univ_quick_snap"
    bl_label = "Quick Snap"
    bl_options = {'REGISTER', 'UNDO'}

    island_mode: bpy.props.BoolProperty(name='Island Mode', default=True)
    quick_start: bpy.props.BoolProperty(name='Quick Start', default=True)

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
        self.handler_ui: typing.Any = None
        self.shader: gpu.types.GPUShader | None = None
        self.batch: gpu.types.GPUBatch | None = None
        self.points: list[Vector] = []
        self.nearest_point: list[Vector] = [Vector((inf, inf, inf))]
        self.visible: bool | None = None
        self.last_set_position: Vector = Vector()
        self.radius: float = 0.0
        self.dragged: bool = False
        self.island_mode_custom: bool = True  # Need for optimize calculation KDTrees
        self.move_object: UnionIslands | UnionLoopGroup | FaceIsland | None = None

        self._cancel: bool = False
        self.start_pos: Vector | None = None
        self.end_pos: Vector | None = None
        self.first_pic_co: Vector | None = None
        self.axis: str = ''

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
        self.preprocessing()

        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        try:
            return self.modal_ex(context, event)
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            self.umeshes.silent_update()
            self.exit()
            return {'FINISHED'}

    def modal_ex(self, _context, event):
        # print()
        # print(f'{event.type = }')
        # print(f'{event.value = }')
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
            return {'PASS_THROUGH'}

        self.calc_radius_and_mouse_position(event)

        if event.type in ('ESC', 'RIGHTMOUSE'):
            self._cancel = self.dragged
            return self.exit()

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return self.exit()

        if axis_sliding_event := event.type in ('X', 'Y') and event.value == 'PRESS':
            if event.type == 'X':
                if self.axis == 'X':
                    self.axis = ''
                else:
                    self.axis = 'X'
            else:
                if self.axis == 'Y':
                    self.axis = ''
                else:
                    self.axis = 'Y'

        if event.type == 'G' and event.value == 'PRESS':
            bpy.context.scene.tool_settings.use_snap_uv_grid_absolute ^= 1

        if self.dragged:
            if self.mouse_position == self.prev_elem_position.to_3d():
                if not axis_sliding_event:
                    return {'RUNNING_MODAL'}

            points = self.kdmeshes.find_range(self.mouse_position, self.radius)
            self.points = self.kdmeshes.range_to_coords(points)
            self.refresh_draw_points()

            if not event.ctrl and (kd_data := self.find_nearest_target_pt()):
                self.nearest_point[0] = kd_data.pt
                pos = kd_data.pt.to_2d()
            else:
                self.nearest_point[0] = self.mouse_position
                pos = self.mouse_position.to_2d()

            # Sliding by axis
            if self.axis == 'X':
                pos.y = self.first_pic_co.y
            if self.axis == 'Y':
                pos.x = self.first_pic_co.x

            self.move_object.set_position(pos, self.prev_elem_position)
            self.prev_elem_position = pos
            self.umeshes.silent_update()
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'TAB' and event.value == 'PRESS':
            self.island_mode = not self.island_mode
            self.preprocessing()

        if event.type == 'MOUSEMOVE':  # ', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:

            points = self.kdmeshes.find_range(self.mouse_position, self.radius)
            self.points = self.kdmeshes.range_to_coords(points)
            self.refresh_draw_points()

            if kd_data := self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius, self.island_mode_custom):
                self.nearest_point[0] = kd_data.pt
            else:
                self.nearest_point[0] = Vector((inf, inf, inf))

            self.area.tag_redraw()

        elif event.type == 'LEFTMOUSE' or self.quick_start:

            kd_data = self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius, self.island_mode_custom)
            if not kd_data:
                self.quick_start = False
                self.report({'INFO'}, 'Not found nearest elem')
                return {'PASS_THROUGH'}

            self.dragged = True
            if self.island_mode:
                islands = []
                if self.visible:
                    self.move_object = kd_data.extract_drag_island()
                    if not kd_data.kdmesh.islands:
                        self.kdmeshes.kdmeshes.remove(kd_data.kdmesh)
                else:
                    for kdmesh in self.kdmeshes:
                        islands.extend(kdmesh.islands)
                    self.move_object = UnionIslands(islands)
                    self.recalc_island_kd_meshes()
            else:
                self.pick_drag_object(kd_data)

            self.move_object.set_position(self.mouse_position.to_2d(), kd_data.pt.to_2d())
            self.prev_elem_position = self.mouse_position.to_2d()

            self.first_pic_co = kd_data.pt.to_2d()
            self.calc_update_meshes()
            self.umeshes.silent_update()
            self.area.tag_redraw()

            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def preprocessing(self):
        if self.island_mode:
            self.island_mode_custom = True
            self.calc_island_kdmeshes(extended=True)
            self.visible = False
            if not self.kdmeshes:
                self.calc_island_kdmeshes(extended=False)
                self.visible = True
                if not self.kdmeshes:
                    return {'CANCELLED'}
        else:
            if self.sync:
                if bpy.context.tool_settings.mesh_select_mode[2]:  # FACE
                    self.island_mode_custom = True
                    if bool(sum(umesh.total_face_sel for umesh in self.umeshes)):
                        self.visible = False
                        self.calc_elem_kdmeshes_selected()
                    else:
                        self.visible = True
                        self.calc_elem_kdmeshes_visible()
                else:
                    if bool(sum(umesh.total_edge_sel for umesh in self.umeshes)):
                        self.island_mode_custom = False
                        self.visible = False
                        self.calc_elem_kdmeshes_sync()
                    else:
                        self.island_mode_custom = True
                        self.visible = True
                        self.calc_elem_kdmeshes_visible()
            else:
                has_selected = False
                for umesh in self.umeshes:
                    if umesh.has_any_selected_crn_non_sync:
                        has_selected = True
                        break

                self.visible = not has_selected

                if self.visible:
                    self.island_mode_custom = True
                    self.calc_elem_kdmeshes_visible()
                else:
                    self.island_mode_custom = False
                    self.calc_crn_edge_selected_kdmeshes()

    def pick_drag_object(self, kd_data: KDData):
        assert not self.island_mode
        _kdmesh = kd_data.kdmesh
        if self.sync:
            if bpy.context.tool_settings.mesh_select_mode[2]:  # FACE
                if self.visible:
                    # Extract only one face without linked corners, and recalculate the snap points at the current KDMesh
                    _face = kd_data.elem if isinstance(kd_data.elem, BMFace) else kd_data.elem.face
                    self.move_object = FaceIsland([_face], _kdmesh.umesh.bm, _kdmesh.umesh.uv_layer)
                    islands = _kdmesh.islands
                    assert(len(islands) == 1)

                    if isinstance(islands[0].faces, list):
                        islands[0].faces.remove(_face)
                    else:
                        list_island = list(islands[0])
                        list_island.remove(_face)
                        islands[0].faces = list_island

                    if not islands[0]:
                        self.kdmeshes.kdmeshes.remove(_kdmesh)
                    else:
                        _kdmesh.clear_elem()
                        _kdmesh.calc_all_trees()
                else:
                    # Extract only all selected faces without linked, and calculate unselected faces at all KDMeshes
                    move_islands = []
                    for kdmesh in self.kdmeshes:
                        islands = kdmesh.islands
                        assert len(islands) == 1

                        move_islands.append(islands[0])
                    self.move_object = UnionIslands(move_islands)
                    self.calc_elem_kdmeshes_unselected()
            else:
                if self.visible:
                    self.extract_visible_linked_edge_or_face(_kdmesh, kd_data)
                else:
                    # Edge Mode with preserve boundary
                    # If the peak group non has_non_sync_crn, we still add it, but all other non has_non_sync_crn groups do not
                    _crn = kd_data.elem.loops[0] if isinstance(kd_data.elem, BMFace) else kd_data.elem
                    lgs = _kdmesh.loop_groups
                    lgs.indexing()
                    picked_lg = lgs[_crn.index]

                    lgs.loop_groups.remove(picked_lg)
                    self.kdmeshes.kdmeshes.remove(kd_data.kdmesh)

                    all_groups_of_mesh: list[list[LoopGroup]] = [[picked_lg]]

                    kd_data.kdmesh.umesh.tag_selected_corners()
                    for lg in kd_data.kdmesh.loop_groups:
                        if lg.has_non_sync_crn():
                            all_groups_of_mesh[0].append(lg)

                    for kdmesh in self.kdmeshes:
                        all_groups: list[LoopGroup] = []
                        kdmesh.umesh.tag_selected_corners()
                        for lg in kdmesh.loop_groups:
                            if lg.has_non_sync_crn():
                                all_groups.append(lg)
                        if all_groups:
                            all_groups_of_mesh.append(all_groups)

                    move_corners_of_mesh: list[LoopGroup] = []

                    kdmeshes: list[KDMesh] = []

                    for lgs in all_groups_of_mesh:
                        move_corners = []
                        umesh = lgs[0].umesh
                        umesh.tag_visible_corners()
                        for lg in lgs:
                            uv = lg.uv
                            for crn in lg:
                                crn.tag = False
                            for crn in lg:
                                move_corners.extend(utils.linked_crn_vert_uv_for_transform(crn, uv))
                            move_corners.extend(lg)

                        umesh.update_tag = False
                        new_lg = LoopGroup(umesh)

                        new_lg.corners = move_corners

                        umesh.tag_visible_corners()
                        new_lg.set_tag(False)

                        move_corners_of_mesh.append(new_lg)

                        kdmesh = KDMesh(umesh=umesh)
                        kdmesh.calc_all_trees_from_static_corners_by_tag()
                        if kdmesh.corners:
                            kdmeshes.append(kdmesh)

                    for umesh in self.umeshes:
                        if not umesh.update_tag:
                            umesh.update_tag = True
                            continue
                        umesh.tag_visible_corners()

                        kdmesh = KDMesh(umesh=umesh)
                        kdmesh.calc_all_trees_from_static_corners_by_tag()
                        if kdmesh.corners:
                            kdmeshes.append(kdmesh)

                    self.kdmeshes = KDMeshes(kdmeshes)
                    self.move_object = UnionLoopGroup(move_corners_of_mesh)
        else:  # Non-sync
            if self.visible:
                self.extract_visible_linked_edge_or_face(_kdmesh, kd_data)

                kdmeshes = []
                for umesh in self.umeshes:
                    if umesh == _kdmesh.umesh:
                        umesh.update_tag = True
                        continue

                    umesh.tag_visible_corners()

                    kdmesh = KDMesh(umesh=umesh)
                    kdmesh.calc_all_trees_from_static_corners_by_tag()
                    if kdmesh.corners:
                        kdmeshes.append(kdmesh)

                if _kdmesh.corners:
                    kdmeshes.append(_kdmesh)

                self.kdmeshes = KDMeshes(kdmeshes)

            else:
                kdmeshes = []
                move_corners_of_mesh: list[LoopGroup] = []
                for kdmesh in self.kdmeshes:
                    kdmesh.umesh.tag_visible_corners()

                    for f in kdmesh.umesh.bm.faces:
                        for c in f.loops:
                            c.tag = True

                    kdmesh.umesh.update_tag = False

                    move_corners = []
                    uv = kdmesh.umesh.uv_layer
                    for f in kdmesh.umesh.bm.faces:
                        if f.select:
                            for crn in f.loops:
                                crn_uv = crn[uv]
                                if crn_uv.select_edge:
                                    if crn.tag:
                                        crn.tag = False
                                        move_corners.append(crn)
                                    crn_next = crn.link_loop_next
                                    if crn_next.tag:
                                        crn_next.tag = False
                                        move_corners.append(crn_next)
                                elif crn_uv.select:
                                    if crn.tag:
                                        crn.tag = False
                                        move_corners.append(crn)
                    if move_corners:
                        lg = LoopGroup(kdmesh.umesh)
                        lg.corners = move_corners
                        move_corners_of_mesh.append(lg)

                    kdmesh = KDMesh(umesh=kdmesh.umesh)
                    kdmesh.calc_all_trees_from_static_corners_by_tag()
                    if kdmesh.corners:
                        kdmeshes.append(kdmesh)

                for umesh in self.umeshes:
                    if not umesh.update_tag:
                        umesh.update_tag = True
                        continue

                    umesh.tag_visible_corners()

                    kdmesh = KDMesh(umesh=umesh)
                    kdmesh.calc_all_trees_from_static_corners_by_tag()
                    if kdmesh.corners:
                        kdmeshes.append(kdmesh)

                self.move_object = UnionLoopGroup(move_corners_of_mesh)
                self.kdmeshes = KDMeshes(kdmeshes)
            self.island_mode_custom = False

    def extract_visible_linked_edge_or_face(self, _kdmesh, kd_data):
        # Extract only one edge or face with linked corners, and recalculate the snap points at one KDMesh.
        uv = _kdmesh.umesh.uv_layer
        _kdmesh.umesh.tag_visible_corners()
        elem = kd_data.elem
        corners = []
        if isinstance(elem, BMFace):
            for _crn in elem.loops:
                _crn.tag = False
                corners.append(_crn)
                linked_corners = utils.linked_crn_uv_by_tag_b(_crn, uv)
                for linked_crn in linked_corners:
                    linked_crn.tag = False
                corners.extend(linked_corners)
        else:
            elem.tag = False
            corners.append(elem)
            linked_corners = utils.linked_crn_uv_by_tag_b(elem, uv)
            for linked_crn in linked_corners:
                linked_crn.tag = False
            corners.extend(linked_corners)

            next_elem = elem.link_loop_next
            if not utils.vec_isclose(kd_data.pt, elem[uv].uv):  # Edge Mode
                next_elem.tag = False
                corners.append(next_elem)
                linked_corners = utils.linked_crn_uv_by_tag_b(next_elem, uv)
                for linked_crn in linked_corners:
                    linked_crn.tag = False
                corners.extend(linked_corners)

        lg = LoopGroup(umesh=_kdmesh.umesh)
        lg.corners = corners
        self.move_object = lg
        _kdmesh.calc_all_trees_from_static_corners_by_tag()
        if not _kdmesh.corners:
            self.kdmeshes.kdmeshes.remove(_kdmesh)
        self.island_mode_custom = False

    def calc_elem_kdmeshes_sync(self):
        assert self.sync

        kdmeshes = []
        for umesh in self.umeshes:
            groups = LoopGroup.calc_dirt_loop_groups(umesh)
            for _g in groups:
                _g.set_tag()
            kdmesh = KDMesh(umesh, loop_groups=groups)
            kdmesh.calc_all_trees_loop_group()
            kdmeshes.append(kdmesh)

        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_elem_kdmeshes_visible(self):
        assert self.visible

        kdmeshes = []
        for umesh in self.umeshes:
            if faces := utils.calc_visible_uv_faces(umesh.bm, umesh.uv_layer, self.sync):
                f_isl = FaceIsland(faces, umesh.bm, umesh.uv_layer)
                isl = Islands([f_isl], umesh.bm, umesh.uv_layer)
                kdmesh = KDMesh(umesh, isl)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_elem_kdmeshes_selected(self):
        assert self.sync
        assert not self.visible

        kdmeshes = []
        for umesh in self.umeshes:
            if faces := utils.calc_selected_uv_faces(umesh.bm, umesh.uv_layer, self.sync):
                f_isl = FaceIsland(faces, umesh.bm, umesh.uv_layer)
                isl = Islands([f_isl], umesh.bm, umesh.uv_layer)
                kdmesh = KDMesh(umesh, isl)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_elem_kdmeshes_unselected(self):
        assert self.sync
        assert not self.visible

        kdmeshes = []
        for umesh in self.umeshes:
            if faces := utils.calc_unselected_uv_faces(umesh.bm, umesh.uv_layer, self.sync):
                f_isl = FaceIsland(faces, umesh.bm, umesh.uv_layer)
                isl = Islands([f_isl], umesh.bm, umesh.uv_layer)
                kdmesh = KDMesh(umesh, isl)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_crn_edge_selected_kdmeshes(self):
        assert not self.visible
        assert not self.sync
        kdmeshes = []
        for umesh in self.umeshes:
            umesh.tag_selected_corners(both=True)
            kdmesh = KDMesh(umesh)
            kdmesh.calc_all_trees_from_static_corners_by_tag()
            kdmeshes.append(kdmesh)

        self.kdmeshes = KDMeshes(kdmeshes)

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

    def find_nearest_target_pt(self):
        kd_data = self.kdmeshes.find_from_all_trees_with_elem(self.mouse_position, self.radius, self.island_mode_custom)

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
        self.handler_ui = bpy.types.SpaceImageEditor.draw_handler_add(self.univ_quick_snap_ui_draw_callback, (), 'WINDOW', 'POST_PIXEL')
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(self.univ_quick_snap_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def univ_quick_snap_draw_callback(self):
        if bpy.context.area.ui_type != 'UV':
            return
        gpu.state.blend_set('ALPHA')
        gpu.state.point_size_set(4)
        self.shader.bind()
        self.shader.uniform_float("color", (1, 1, 0, 0.5))
        self.batch.draw(self.shader)

        batch_nearest = gpu_extras.batch.batch_for_shader(self.shader, 'POINTS', {"pos": self.nearest_point})
        self.shader.uniform_float("color", (1, 0.2, 0, 1))
        batch_nearest.draw(self.shader)

        self.area.tag_redraw()
        gpu.state.blend_set('NONE')

    def univ_quick_snap_ui_draw_callback(self):
        area = bpy.context.area
        if area.ui_type != 'UV':
            return

        max_dim = 180
        if area.width < max_dim * 2:
            return

        first_col = area.width - max_dim
        second_col = first_col + 40

        gpu.state.blend_set('ALPHA')

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)

        text_y_size = blf.dimensions(0, 'T')[1]
        text_y_size *= 1.75

        blf.position(font_id, first_col, 20, 0)
        blf.draw(font_id, 'Tab')
        blf.position(font_id, second_col, 20, 0)
        blf.draw(font_id, 'Island Mode' if self.island_mode else 'Element Mode')

        blf.position(font_id, first_col, 20 + text_y_size, 0)
        blf.draw(font_id, 'G')
        blf.position(font_id, second_col, 20 + text_y_size, 0)
        blf.draw(font_id, f"Grid: {'Enabled' if bpy.context.scene.tool_settings.use_snap_uv_grid_absolute else 'Disabled'}")

        blf.position(font_id, first_col, 20 + text_y_size*2, 0)
        blf.draw(font_id, 'X, Y')
        blf.position(font_id, second_col, 20 + text_y_size*2, 0)
        blf.draw(font_id, f"Axis: {self.axis if self.axis else 'Both'}")

        gpu.state.blend_set('NONE')

    def refresh_draw_points(self):
        self.batch = gpu_extras.batch.batch_for_shader(self.shader, 'POINTS', {"pos": self.points})

    def exit(self):
        if self._cancel:
            if not (self.start_pos is None or self.end_pos is None or self.move_object is None):
                self.move_object.set_position(self.start_pos, self.end_pos)
                self.umeshes.silent_update()
            else:
                if debug() and self.move_object:
                    self.report({'WARNING'}, 'Failed to cancel the operator')

        if not (self.handler is None):
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler, 'WINDOW')
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler_ui, 'WINDOW')

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.ui_type == 'UV':
                        area.tag_redraw()

        return {'FINISHED'}

    def calc_update_meshes(self):
        if isinstance(self.move_object, FaceIsland):
            co = self.move_object.faces[0].loops[0][self.move_object.uv_layer].uv
            self.start_pos = co.copy()
            self.end_pos = co

            for umesh in self.umeshes:
                if umesh.bm == self.move_object.bm:
                    self.umeshes.umeshes = [umesh]
                    return

        elif isinstance(self.move_object, UnionIslands):
            co = self.move_object.islands[0].faces[0].loops[0][self.move_object.islands[0].uv_layer].uv
            self.start_pos = co.copy()
            self.end_pos = co

            bmeshes = {isl.bm for isl in self.move_object}
            umeshes = []
            for umesh in self.umeshes:
                if umesh.bm in bmeshes:
                    umeshes.append(umesh)
            assert umeshes
            self.umeshes.umeshes = umeshes

        elif isinstance(self.move_object, UnionLoopGroup):
            co = self.move_object.loop_groups[0].corners[0][self.move_object.loop_groups[0].uv].uv
            self.start_pos = co.copy()
            self.end_pos = co

            bmeshes = {lg.umesh.bm for lg in self.move_object}
            umeshes = []
            for umesh in self.umeshes:
                if umesh.bm in bmeshes:
                    umeshes.append(umesh)
            assert umeshes
            self.umeshes.umeshes = umeshes

        elif isinstance(self.move_object, LoopGroup):
            co = self.move_object.corners[0][self.move_object.uv].uv
            self.start_pos = co.copy()
            self.end_pos = co

            for umesh in self.umeshes:
                if umesh == self.move_object.umesh:
                    self.umeshes.umeshes = [umesh]
                    return

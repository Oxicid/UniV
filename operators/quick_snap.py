# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu
import blf
import enum
import typing
from gpu_extras.batch import batch_for_shader

from math import inf
from mathutils import Vector
from bmesh.types import BMFace, BMLoop

from .. import utils
from ..draw import shaders
from ..preferences import debug, prefs
from ..utypes import KDMesh, KDData, KDMeshes, Islands, UnionIslands, FaceIsland, View2D, LoopGroup, UnionLoopGroup, UMeshes


class eSnapPointMode(enum.IntFlag):
    NONE = 1 << 0
    VERTEX = 1 << 1
    EDGE = 1 << 2
    FACE = 1 << 3
    ALL = VERTEX | EDGE | FACE


class SnapMode:
    def __init__(self):
        self.sync: bool = utils.sync()
        self.snap_points_mode: eSnapPointMode = eSnapPointMode.VERTEX

    def snap_mode_init(self):
        self.snap_points_mode = eSnapPointMode.VERTEX
        if prefs().snap_points_default == 'FOLLOW_MODE':
            if self.sync:
                if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                    self.snap_points_mode |= eSnapPointMode.EDGE
                if bpy.context.tool_settings.mesh_select_mode[2]:  # FACE
                    self.snap_points_mode |= eSnapPointMode.FACE
            else:
                uv_mode = utils.get_select_mode_uv()
                if uv_mode in ('FACE', 'ISLAND'):
                    self.snap_points_mode |= eSnapPointMode.FACE
                elif uv_mode == 'EDGE':
                    self.snap_points_mode |= eSnapPointMode.EDGE
        else:
            self.snap_points_mode = eSnapPointMode.ALL

    def snap_mode_update(self, event):
        match event.type:
            case 'ONE':
                if event.shift:
                    self.snap_points_mode ^= eSnapPointMode.VERTEX
                else:
                    self.snap_points_mode = eSnapPointMode.VERTEX
            case 'TWO':
                if event.shift:
                    self.snap_points_mode ^= eSnapPointMode.EDGE
                else:
                    self.snap_points_mode = eSnapPointMode.EDGE
            case 'THREE':
                if event.shift:
                    self.snap_points_mode ^= eSnapPointMode.FACE
                else:
                    self.snap_points_mode = eSnapPointMode.FACE
            case _:
                if event.shift:
                    self.snap_points_mode ^= eSnapPointMode.ALL
                else:
                    self.snap_points_mode = eSnapPointMode.ALL


class QuickSnap_KDMeshes:
    def __init__(self):
        self.sync = True
        self.umeshes = None
        self.kdmeshes = None
        self.visible = None
        self.radius = None
        self.mouse_position = None
        self.snap_points_mode = None
        self.view = None
        self.grid_snap = None

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
            if faces := utils.calc_visible_uv_faces(umesh):
                f_isl = FaceIsland(faces, umesh)
                isl = Islands([f_isl], umesh)
                kdmesh = KDMesh(umesh, isl)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_elem_kdmeshes_selected(self):
        assert self.sync
        assert not self.visible

        kdmeshes = []
        for umesh in self.umeshes:
            if faces := utils.calc_selected_uv_faces(umesh):
                f_isl = FaceIsland(faces, umesh)
                isl = Islands([f_isl], umesh)
                kdmesh = KDMesh(umesh, isl)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def calc_elem_kdmeshes_unselected(self):
        assert self.sync
        assert not self.visible

        kdmeshes = []
        for umesh in self.umeshes:
            if faces := utils.calc_unselected_uv_faces(umesh):
                f_isl = FaceIsland(faces, umesh)
                isl = Islands([f_isl], umesh)
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
            if islands := Islands.calc_extended_or_visible(umesh, extended=extended):
                kdmesh = KDMesh(umesh, islands)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def recalc_island_kd_meshes(self):
        kdmeshes = []
        for umesh in self.umeshes:
            if islands := Islands.calc_non_selected_extended(umesh):
                kdmesh = KDMesh(umesh, islands)
                kdmesh.calc_all_trees()
                kdmeshes.append(kdmesh)
        self.kdmeshes = KDMeshes(kdmeshes)

    def find(self):
        pt: tuple[Vector, int, float] = tuple((Vector((inf, inf, inf)), 0, inf))
        elem: BMFace | BMLoop | None = None
        r_kdmesh: KDMesh | None = None

        r = self.radius
        co = self.mouse_position.to_3d()

        for kdmesh in self.kdmeshes:
            if self.snap_points_mode & eSnapPointMode.VERTEX and (min_res_ := kdmesh.kdtree_crn_points.find(co))[0]:
                if min_res_[2] <= r and min_res_[2] < pt[2]:
                    pt = min_res_
                    elem = kdmesh.corners_vert[min_res_[1]]
                    r_kdmesh = kdmesh

            if self.snap_points_mode & eSnapPointMode.EDGE and (
                    min_res_ := kdmesh.kdtree_crn_center_points.find(co))[0]:
                if min_res_[2] <= r and min_res_[2] < pt[2]:
                    pt = min_res_
                    elem = kdmesh.corners_center[min_res_[1]]
                    r_kdmesh = kdmesh
            if self.snap_points_mode & eSnapPointMode.FACE and (min_res_ := kdmesh.kdtree_face_points.find(co))[0]:
                if min_res_[2] <= r and min_res_[2] < pt[2]:
                    pt = min_res_
                    elem = kdmesh.faces[min_res_[1]]
                    r_kdmesh = kdmesh

        return KDData(pt, elem, r_kdmesh)

    def find_nearest_target_pt(self):
        kd_data = self.find()

        m_pos = self.mouse_position.to_2d()
        zoom = View2D.get_zoom(self.view)
        divider = 1/8 if zoom <= 1600 else 1 / 64
        divider = divider if zoom <= 12800 else 1 / 64 / 8
        pos = Vector(utils.round_threshold(v, divider) for v in m_pos)
        dist = (pos - m_pos).length

        if self.grid_snap \
                and dist <= self.radius and dist < kd_data.distance:
            kd_data.found = (pos, 0, 0.0)
            kd_data.kdmesh = True
        return kd_data

    def find_range(self):
        coords = []
        if eSnapPointMode.VERTEX in self.snap_points_mode:
            res = self.kdmeshes.find_range_vert(self.mouse_position, self.radius)
            coords = self.kdmeshes.range_to_coords(res)
        if eSnapPointMode.EDGE in self.snap_points_mode:
            res = self.kdmeshes.find_range_crn_center(self.mouse_position, self.radius)
            if coords:
                coords.extend(self.kdmeshes.range_to_coords(res))
            else:
                coords = self.kdmeshes.range_to_coords(res)
        if eSnapPointMode.FACE in self.snap_points_mode:
            res = self.kdmeshes.find_range_face_center(self.mouse_position, self.radius)
            if coords:
                coords.extend(self.kdmeshes.range_to_coords(res))
            else:
                coords = self.kdmeshes.range_to_coords(res)
        return coords


class UNIV_OT_QuickSnap(bpy.types.Operator, SnapMode, QuickSnap_KDMeshes):
    bl_idname = "uv.univ_quick_snap"
    bl_label = "Quick Snap"
    bl_options = {'REGISTER', 'UNDO'}

    island_mode: bpy.props.BoolProperty(name='Island Mode', default=True)
    quick_start: bpy.props.BoolProperty(name='Quick Start', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.kdmeshes: KDMeshes | None = None

        self.area: bpy.types.Area | None = None
        self.view: bpy.types.View2D | None = None

        self.dragged: bool = False
        self.first_pick_co: Vector | None = None
        self.mouse_position: Vector = Vector((0.0, 0.0, 0.0))
        self.prev_elem_position: Vector = Vector((0.0, 0.0, 0.0))

        self.handler: typing.Any = None
        self.handler_ui: typing.Any = None
        self.shader: gpu.types.GPUShader | None = None
        self.batch: gpu.types.GPUBatch | None = None
        self.points: list[Vector] = []
        self.nearest_point: list[Vector] = [Vector((inf, inf, inf))]

        self.visible: bool | None = None
        self.radius: float = 0.0
        self.axis: str = ''
        self.grid_snap: bool = False

        self._cancel: bool = False

        self.move_object: UnionIslands | UnionLoopGroup | FaceIsland | None = None

    def invoke(self, context, event):
        self.area = context.area
        if self.area.ui_type != 'UV':
            self.report({'INFO'}, 'Area must be UV')
            return {'CANCELLED'}
        self.view = context.region.view2d
        self.sync = utils.sync()
        self.shader = shaders.POINT_UNIFORM_COLOR
        self.refresh_draw_points()
        self.register_draw()

        self.snap_mode_init()

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

        # TODO: Test
        # if event.type == 'INBETWEEN_MOUSEMOVE':  # fix over move
        #     return {'RUNNING_MODAL'}

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
            self.grid_snap ^= 1
            self.area.tag_redraw()

        if event.value == 'PRESS' and event.type in {'ONE', 'TWO', 'THREE', 'FOUR'}:
            self.snap_mode_update(event)
            self.area.tag_redraw()

        if self.dragged:
            if self.mouse_position == self.prev_elem_position.to_3d():
                if not axis_sliding_event:
                    return {'RUNNING_MODAL'}

            self.points = self.find_range()
            self.refresh_draw_points()

            if not event.ctrl and (kd_data := self.find_nearest_target_pt()):
                self.nearest_point[0] = kd_data.pt
                pos = kd_data.pt.to_2d()
            else:
                self.nearest_point[0] = self.mouse_position
                pos = self.mouse_position.to_2d()

            # Sliding by axis
            if self.axis == 'X':
                pos.y = self.first_pick_co.y
            if self.axis == 'Y':
                pos.x = self.first_pick_co.x

            self.move_object.set_position(pos, self.prev_elem_position)
            self.prev_elem_position = pos
            self.umeshes.silent_update()
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'TAB' and event.value == 'PRESS':
            self.island_mode = not self.island_mode
            self.preprocessing()
            self.area.tag_redraw()

        if event.type == 'MOUSEMOVE' and not self.quick_start:  # ', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:

            self.points = self.find_range()
            self.refresh_draw_points()

            if kd_data := self.find():
                self.nearest_point[0] = kd_data.pt
            else:
                self.nearest_point[0] = Vector((inf, inf, inf))

            self.area.tag_redraw()

        elif event.type == 'LEFTMOUSE' or self.quick_start:

            kd_data = self.find()
            if not kd_data:
                self.quick_start = False
                self.report({'WARNING'}, 'Not found nearest elem')
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

            self.first_pick_co = kd_data.pt.to_2d()
            self.calc_update_meshes()
            self.umeshes.silent_update()
            self.area.tag_redraw()

            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}

    def preprocessing(self):
        if self.island_mode:
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
                    if bool(sum(umesh.total_face_sel for umesh in self.umeshes)):
                        self.visible = False
                        self.calc_elem_kdmeshes_selected()
                    else:
                        self.visible = True
                        self.calc_elem_kdmeshes_visible()
                else:
                    if any(umesh.total_edge_sel for umesh in self.umeshes):
                        self.visible = False
                        self.calc_elem_kdmeshes_sync()
                    else:
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
                    self.calc_elem_kdmeshes_visible()
                else:
                    self.calc_crn_edge_selected_kdmeshes()

    def pick_drag_object(self, kd_data: KDData):
        assert not self.island_mode
        _kdmesh: KDMesh = kd_data.kdmesh
        if self.sync:
            if bpy.context.tool_settings.mesh_select_mode[2]:  # FACE
                if self.visible:
                    # Extract only one face without linked corners, and recalculate the snap
                    # points at the current KDMesh
                    _face = kd_data.elem if isinstance(kd_data.elem, BMFace) else kd_data.elem.face
                    self.move_object = FaceIsland([_face], _kdmesh.umesh)
                    islands = _kdmesh.islands
                    assert (len(islands) == 1)

                    if isinstance(islands[0].faces, list):
                        islands[0].faces.remove(_face)
                    else:
                        list_island = list(islands[0])
                        list_island.remove(_face)
                        islands[0].faces = list_island

                    if not islands[0]:
                        self.kdmeshes.kdmeshes.remove(_kdmesh)
                    else:
                        _kdmesh.clear_containers()
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
                    # If the peak group non has_non_sync_crn, we still add it, but all other
                    # non has_non_sync_crn groups do not
                    _crn = kd_data.elem.loops[0] if isinstance(kd_data.elem, BMFace) else kd_data.elem
                    lgs = _kdmesh.loop_groups
                    lgs.indexing()

                    # Pick can be on link_loop_next, which is only added to kdmesh, not calc_dirt_loop_groups
                    picked_lg = lgs[_crn.link_loop_prev.index if _crn.index == -1 else _crn.index]

                    kdmeshes = []
                    move_corners_of_mesh: list[LoopGroup] = []

                    kd_data.kdmesh.umesh.tag_selected_corners()
                    if picked_lg.has_sync_crn():  # Transform only picked group if it has sync corner in loop_group
                        move_corners_of_mesh.append(picked_lg)
                        for umesh in self.umeshes:
                            umesh.tag_visible_corners()

                        picked_lg.extend_from_linked()  # tags false automatic

                        for umesh in self.umeshes:
                            kdmesh = KDMesh(umesh=umesh)
                            kdmesh.calc_all_trees_from_static_corners_by_tag()
                            if kdmesh:
                                kdmeshes.append(kdmesh)
                    else:
                        for kdmesh in self.kdmeshes:
                            kdmesh.umesh.update_tag = False

                            if kd_data.kdmesh != kdmesh:
                                kdmesh.umesh.tag_selected_corners()
                                kdmesh.loop_groups.indexing()
                            non_sync_lgs = []
                            for lg in kdmesh.loop_groups:
                                if not lg.has_sync_crn():
                                    non_sync_lgs.append(lg)

                            kdmesh.umesh.tag_visible_corners()
                            for non_sync_lg in non_sync_lgs:
                                non_sync_lg.extend_from_linked()  # tags false automatic
                                move_corners_of_mesh.append(non_sync_lg)

                            kdmesh = KDMesh(umesh=kdmesh.umesh)
                            kdmesh.calc_all_trees_from_static_corners_by_tag()
                            if kdmesh:
                                kdmeshes.append(kdmesh)

                        for umesh in self.umeshes:
                            if not umesh.update_tag:
                                umesh.update_tag = True
                                continue
                            kdmesh = KDMesh(umesh=umesh)
                            umesh.tag_visible_corners()
                            kdmesh.calc_all_trees_from_static_corners_by_tag()
                            if kdmesh:
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
                    if kdmesh.corners_center:
                        kdmeshes.append(kdmesh)

                if _kdmesh:
                    kdmeshes.append(_kdmesh)

                self.kdmeshes = KDMeshes(kdmeshes)

            else:
                kdmeshes = []
                move_corners_of_mesh: list[LoopGroup] = []
                for kdmesh in self.kdmeshes:
                    kdmesh.umesh.update_tag = False
                    for f in kdmesh.umesh.bm.faces:
                        for c in f.loops:
                            c.tag = True

                    move_corners = []
                    edge_select_get = utils.edge_select_get_func(kdmesh.umesh)
                    for crn in utils.calc_selected_uv_vert_corners_iter(kdmesh.umesh):
                        if edge_select_get(crn):
                            if crn.tag:
                                crn.tag = False
                                move_corners.append(crn)
                            crn_next = crn.link_loop_next
                            if crn_next.tag:
                                crn_next.tag = False
                                move_corners.append(crn_next)
                        else:  # Select Vertex
                            if crn.tag:
                                crn.tag = False
                                move_corners.append(crn)
                    if move_corners:
                        lg = LoopGroup(kdmesh.umesh)
                        lg.corners = move_corners
                        move_corners_of_mesh.append(lg)

                    kdmesh = KDMesh(umesh=kdmesh.umesh)
                    kdmesh.calc_all_trees_from_static_corners_by_tag()
                    if kdmesh.corners_center:
                        kdmeshes.append(kdmesh)

                for umesh in self.umeshes:
                    if not umesh.update_tag:
                        umesh.update_tag = True
                        continue

                    umesh.tag_visible_corners()

                    kdmesh = KDMesh(umesh=umesh)
                    kdmesh.calc_all_trees_from_static_corners_by_tag()
                    if kdmesh.corners_center:
                        kdmeshes.append(kdmesh)

                self.move_object = UnionLoopGroup(move_corners_of_mesh)
                self.kdmeshes = KDMeshes(kdmeshes)

    def extract_visible_linked_edge_or_face(self, _kdmesh, kd_data):
        # Extract only one edge or face with linked corners, and recalculate the snap points at one KDMesh.
        uv = _kdmesh.umesh.uv
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
        if not _kdmesh:
            self.kdmeshes.kdmeshes.remove(_kdmesh)

    def calc_radius_and_mouse_position(self, event):
        mouse_position = Vector(self.view.region_to_view(event.mouse_region_x, event.mouse_region_y))
        dist = prefs().max_pick_distance // 2 if self.dragged else prefs().max_pick_distance
        self.radius = utils.get_max_distance_from_px(dist, self.view)
        self.mouse_position = mouse_position.to_3d()

    def register_draw(self):
        self.handler_ui = bpy.types.SpaceImageEditor.draw_handler_add(
            self.univ_quick_snap_ui_draw_callback, (), 'WINDOW', 'POST_PIXEL')
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(
            self.univ_quick_snap_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def univ_quick_snap_draw_callback(self):
        if bpy.context.area.ui_type != 'UV':
            return

        shaders.set_point_size(4)
        shaders.blend_set_alpha()

        self.shader.bind()
        self.shader.uniform_float("color", (1, 1, 0, 0.5))
        self.batch.draw(self.shader)

        batch_nearest = batch_for_shader(self.shader, 'POINTS', {"pos": self.nearest_point})
        self.shader.uniform_float("color", (1, 0.2, 0, 1))
        batch_nearest.draw(self.shader)

        self.area.tag_redraw()

        shaders.set_point_size(1)
        shaders.blend_set_none()

    def univ_quick_snap_ui_draw_callback(self):
        area = bpy.context.area
        if area.ui_type != 'UV':
            return

        n_panel_width = next(r.width for r in area.regions if r.type == 'UI')
        max_dim = 240
        if (area.width - n_panel_width) < max_dim or area.height < max_dim:
            return

        first_col = area.width - max_dim - n_panel_width
        second_col = first_col + 90

        gpu.state.blend_set('ALPHA')

        font_id = 0
        utils.blf_size(font_id, 16)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)

        text_y_size = blf.dimensions(0, 'T')[1]
        text_y_size *= 1.75

        blf.position(font_id, first_col, 20, 0)
        blf.color(font_id, 0.75, 0.75, 0.75, 0.85)
        blf.draw(font_id, 'Tab')
        blf.position(font_id, second_col, 20, 0)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)
        blf.draw(font_id, 'Island Mode' if self.island_mode else 'Element Mode')

        blf.position(font_id, first_col, 20 + text_y_size, 0)
        blf.color(font_id, 0.75, 0.75, 0.75, 0.85)
        blf.draw(font_id, 'X, Y')
        blf.position(font_id, second_col, 20 + text_y_size, 0)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)
        blf.draw(font_id, f"Axis: {self.axis if self.axis else 'Both'}")

        blf.position(font_id, first_col, 20 + text_y_size*2, 0)
        blf.color(font_id, 0.75, 0.75, 0.75, 0.85)
        blf.draw(font_id, 'G')
        blf.position(font_id, second_col, 20 + text_y_size*2, 0)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)
        blf.draw(font_id, f"Grid: {'Enabled' if self.grid_snap else 'Disabled'}")

        if (text := self.snap_points_mode.name if self.snap_points_mode else 'NONE') is None:
            text = str(self.snap_points_mode).split('.')[1]
        blf.position(font_id, first_col, 20 + text_y_size*3, 0)
        blf.color(font_id, 0.75, 0.75, 0.75, 0.85)
        blf.draw(font_id, '(Shift) 1-4')
        blf.position(font_id, second_col, 20 + text_y_size*3, 0)
        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)
        blf.draw(font_id, text)

        gpu.state.blend_set('NONE')

    def refresh_draw_points(self):
        self.batch = batch_for_shader(self.shader, 'POINTS', {"pos": self.points})

    def exit(self):
        if self._cancel:
            if not (self.first_pick_co is None or self.prev_elem_position is None or self.move_object is None):
                if self.move_object.move(self.first_pick_co - self.prev_elem_position):
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
            self.umeshes.umeshes = [self.move_object.umesh]

        elif isinstance(self.move_object, UnionIslands):
            assert (umeshes := list({isl.umesh for isl in self.move_object}))
            self.umeshes.umeshes = umeshes

        elif isinstance(self.move_object, UnionLoopGroup):
            assert (umeshes := list({lg.umesh for lg in self.move_object}))
            self.umeshes.umeshes = umeshes

        elif isinstance(self.move_object, LoopGroup):
            self.umeshes.umeshes = [self.move_object.umesh]

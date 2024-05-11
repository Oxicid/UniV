import bpy
import numpy as np

from bpy.types import Operator
from bpy.props import *

from ..types import BBox, Islands, FaceIsland
from .. import utils
from .. import info
from mathutils import Vector


align_align_direction_items = (
    ('UPPER', 'Upper', ''),
    ('BOTTOM', 'Bottom', ''),
    ('LEFT', 'Left', ''),
    ('RIGHT', 'Right', ''),
    ('CENTER', 'Center', ''),
    ('HORIZONTAL', 'Horizontal', ''),
    ('VERTICAL', 'Vertical', ''),
    ('LEFT_UPPER', 'Left upper', ''),
    ('RIGHT_UPPER', 'Right upper', ''),
    ('LEFT_BOTTOM', 'Left bottom', ''),
    ('RIGHT_BOTTOM', 'Right bottom', ''),
)
class UNIV_OT_Align(Operator):
    bl_idname = "uv.univ_align"
    bl_label = "Align"
    bl_description = info.operator.align_event_info
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name="Mode", default='MOVE', items=(
        ('ALIGN', 'Align', ''),
        ('MOVE', 'Move', ''),
        ('ALIGN_CURSOR', 'Move cursor to selected', ''),
        ('ALIGN_TO_CURSOR', 'Align to cursor', ''),
        ('ALIGN_TO_CURSOR_UNION', 'Align to cursor union', ''),
        ('CURSOR_TO_TILE', 'Align cursor to tile', ''),
        ('MOVE_CURSOR', 'Move cursor', ''),
        # ('MOVE_COLLISION', 'Collision move', '')
    ))

    direction: bpy.props.EnumProperty(name="Direction", default='UPPER', items=align_align_direction_items)

    @classmethod
    def poll(cls, context):
        if not bpy.context.active_object:
            return False
        if bpy.context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'ALIGN'
            case True, False, False:
                self.mode = 'ALIGN_TO_CURSOR'
            case True, True, True:
                self.mode = 'ALIGN_TO_CURSOR_UNION'
            case False, False, True:
                self.mode = 'ALIGN_CURSOR'
            case True, False, True:
                self.mode = 'CURSOR_TO_TILE'
            case False, True, True:
                self.mode = 'MOVE_CURSOR'
            case False, True, False:
                self.mode = 'MOVE'
            case _:
                self.report({'INFO'}, f"Event: Ctrl={event.ctrl}, Shift={event.shift}, Alt={event.alt} not implement.\n\n"
                                      f"See all variations:\n\n{info.operator.align_event_info_ex}")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return self.align(self.mode, self.direction, sync=bpy.context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def align(mode, direction, sync, report=None):
        update_obj = utils.UMeshes([])
        umeshes = utils.UMeshes.sel_ob_with_uv()

        match mode:
            case 'ALIGN':
                UNIV_OT_Align.align_ex(direction, sync,  umeshes,  update_obj, selected=True)
                if not update_obj:
                    UNIV_OT_Align.align_ex(direction, sync,  umeshes,  update_obj, selected=False)

            case 'ALIGN_TO_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_to_cursor_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=True)
                if not update_obj:
                    UNIV_OT_Align.move_to_cursor_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=False)

            case 'ALIGN_TO_CURSOR_UNION':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_to_cursor_union_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=True)
                if not update_obj:
                    UNIV_OT_Align.move_to_cursor_union_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=False)

            case 'ALIGN_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                general_bbox = UNIV_OT_Align.align_cursor_ex(umeshes, sync, selected=True)
                if not general_bbox.is_valid:
                    general_bbox = UNIV_OT_Align.align_cursor_ex(umeshes, sync, selected=False)
                if not general_bbox.is_valid:
                    if report:
                        report({'INFO'}, "No elements for manipulate")
                    return {'CANCELLED'}
                UNIV_OT_Align.align_cursor(direction, general_bbox, cursor_loc)
                return {'FINISHED'}

            case 'CURSOR_TO_TILE':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.align_cursor_to_tile(direction, cursor_loc)
                return {'FINISHED'}

            case 'MOVE_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_cursor(direction, cursor_loc)
                return {'FINISHED'}

            case 'MOVE':
                UNIV_OT_Align.move_ex(direction, sync, umeshes, update_obj, selected=True)
                if not update_obj:
                    UNIV_OT_Align.move_ex(direction, sync, umeshes, update_obj, selected=False)

            case _:
                raise NotImplementedError(mode)

        if not update_obj:
            if report:
                report({'INFO'}, "No faces/verts for manipulate")
            return {'CANCELLED'}

        update_obj.update()

        return {'FINISHED'}

    @staticmethod
    def move_to_cursor_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        island_mode = is_island_mode()
        general_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        for umesh in umeshes:
            uv_layer = umesh.bm.loops.layers.uv.verify()
            if island_mode:
                if islands := Islands.calc(umesh.bm, uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        all_groups.append((island, bbox, uv_layer))
                    update_obj.umeshes.append(umesh)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, uv_layer, sync, selected=selected):
                    all_groups.append((corners, uv_layer))
                    update_obj.umeshes.append(umesh)
        if island_mode:
            UNIV_OT_Align.align_islands(all_groups, direction, general_bbox, invert=True)
        else:  # Vertices or Edges UV selection mode
            UNIV_OT_Align.align_corners(all_groups, direction, general_bbox)

    @staticmethod
    def move_to_cursor_union_ex(cursor_loc, direction, umeshes, update_obj, sync, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        target_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        general_bbox = BBox()
        for umesh in umeshes:
            uv_layer = umesh.bm.loops.layers.uv.verify()

            if faces := utils.calc_uv_faces(umesh.bm, uv_layer, sync, selected=selected):
                island = FaceIsland(faces, umesh.bm, uv_layer)
                bbox = island.calc_bbox()
                general_bbox.union(bbox)
                all_groups.append([island, bbox, uv_layer])
                update_obj.umeshes.append(umesh)
        for group in all_groups:
            group[1] = general_bbox
        UNIV_OT_Align.align_islands(all_groups, direction, target_bbox, invert=True)

    @staticmethod
    def align_cursor_ex(umeshes, sync, selected):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        for umesh in umeshes:
            uv_layer = umesh.bm.loops.layers.uv.verify()
            if corners := utils.calc_uv_corners(umesh.bm, uv_layer, sync, selected=selected):
                all_groups.append((corners, uv_layer))
                bbox = BBox.calc_bbox_uv_corners(corners, uv_layer)
                general_bbox.union(bbox)
        return general_bbox

    @staticmethod
    def align_ex(direction, sync, umeshes, update_obj, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        island_mode = is_island_mode()
        for umesh in umeshes:
            uv_layer = umesh.bm.loops.layers.uv.verify()
            if island_mode:
                if islands := Islands.calc(umesh.bm, uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)

                        all_groups.append((island, bbox, uv_layer))
                    update_obj.umeshes.append(umesh)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, uv_layer, sync, selected=selected):
                    bbox = BBox.calc_bbox_uv_corners(corners, uv_layer)
                    general_bbox.union(bbox)

                    all_groups.append((corners, uv_layer))
                    update_obj.umeshes.append(umesh)
        if island_mode:
            UNIV_OT_Align.align_islands(all_groups, direction, general_bbox)
        else:  # Vertices or Edges UV selection mode
            UNIV_OT_Align.align_corners(all_groups, direction, general_bbox)

    @staticmethod
    def move_ex(direction, sync, umeshes, update_obj, selected=True):
        island_mode = is_island_mode()
        for umesh in umeshes:
            uv_layer = umesh.bm.loops.layers.uv.verify()
            if island_mode:
                if islands := Islands.calc(umesh.bm, uv_layer, sync, selected=selected):
                    match direction:
                        case 'CENTER':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta = Vector((0.5, 0.5)) - bbox.center
                                island.move(delta)  # ToDO: Implement update flags for return state move
                        case 'HORIZONTAL':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta_y = 0.5 - bbox.center.y
                                island.move(Vector((0.0, delta_y)))
                        case 'VERTICAL':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta_x = 0.5 - bbox.center.x
                                island.move(Vector((delta_x, 0.0)))
                        case _:
                            move_value = Vector(UNIV_OT_Align.get_move_value(direction))
                            for island in islands:
                                island.move(move_value)
                    update_obj.umeshes.append(umesh)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, uv_layer, sync, selected=selected):
                    match direction:
                        case 'CENTER':
                            for corner in corners:
                                corner[uv_layer].uv = 0.5, 0.5
                        case 'HORIZONTAL':
                            for corner in corners:
                                corner[uv_layer].uv.x = 0.5
                        case 'VERTICAL':
                            for corner in corners:
                                corner[uv_layer].uv.y = 0.5
                        case _:
                            move_value = Vector(UNIV_OT_Align.get_move_value(direction))
                            for corner in corners:
                                corner[uv_layer].uv += move_value
                    update_obj.umeshes.append(umesh)

    @staticmethod
    def align_islands(groups, direction, general_bbox, invert=False):
        for island, bounds, uv_layer in groups:
            center = bounds.center
            match direction:
                case 'UPPER':
                    delta = (0, (general_bbox.min - bounds.min).y) if invert else (0, (general_bbox.max - bounds.max).y)
                case 'BOTTOM':
                    delta = (0, (general_bbox.max - bounds.max).y) if invert else (0, (general_bbox.min - bounds.min).y)
                case 'LEFT':
                    delta = ((general_bbox.max - bounds.max).x, 0) if invert else ((general_bbox.min - bounds.min).x, 0)
                case 'RIGHT':
                    delta = ((general_bbox.min - bounds.min).x, 0) if invert else ((general_bbox.max - bounds.max).x, 0)
                case 'CENTER':
                    delta = general_bbox.center - center
                case 'HORIZONTAL':
                    delta = (0, (general_bbox.center - center).y)
                case 'VERTICAL':
                    delta = (general_bbox.center - center).x, 0
                case 'RIGHT_UPPER':
                    delta = general_bbox.min - bounds.min if invert else general_bbox.max - bounds.max
                case 'LEFT_UPPER':
                    if invert:
                        delta = (general_bbox.max - bounds.max).x, (general_bbox.min - bounds.min).y
                    else:
                        delta = (general_bbox.min - bounds.min).x, (general_bbox.max - bounds.max).y
                case 'LEFT_BOTTOM':
                    delta = general_bbox.max - bounds.max if invert else general_bbox.min - bounds.min
                case 'RIGHT_BOTTOM':
                    if invert:
                        delta = (general_bbox.min - bounds.min).x, (general_bbox.max - bounds.max).y
                    else:
                        delta = (general_bbox.max - bounds.max).x, (general_bbox.min - bounds.min).y
                case _:
                    raise NotImplementedError(direction)
            island.move(Vector(delta))

    @staticmethod
    def align_corners(groups, direction, general_bbox):

        match direction:
            case 'LEFT' | 'RIGHT' | 'VERTICAL':
                if direction == 'LEFT':
                    destination = general_bbox.min.x
                elif direction == 'RIGHT':
                    destination = general_bbox.max.x
                else:
                    destination = general_bbox.center.x

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv[0] = destination
            case 'UPPER' | 'BOTTOM' | 'HORIZONTAL':
                if direction == 'UPPER':
                    destination = general_bbox.max.y
                elif direction == 'BOTTOM':
                    destination = general_bbox.min.y
                else:
                    destination = general_bbox.center.y

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv[1] = destination
            case _:
                if direction == 'CENTER':
                    destination = general_bbox.center
                elif direction == 'LEFT_BOTTOM':
                    destination = general_bbox.left_bottom
                elif direction == 'RIGHT_UPPER':
                    destination = general_bbox.right_upper
                elif direction == 'LEFT_UPPER':
                    destination = general_bbox.left_upper
                elif direction == 'RIGHT_BOTTOM':
                    destination = general_bbox.right_bottom
                else:
                    raise NotImplementedError(direction)

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv = destination

    @staticmethod
    def align_cursor(direction: str, general_bbox, cursor_loc):

        if loc := getattr(general_bbox, direction.lower(), False):
            utils.set_cursor_location(loc)
        elif direction == 'VERTICAL':
            utils.set_cursor_location(Vector((general_bbox.center.x, cursor_loc.y)))
        elif direction == 'HORIZONTAL':
            utils.set_cursor_location(Vector((cursor_loc.x, general_bbox.center.y)))
        else:
            raise NotImplementedError(direction)

    @staticmethod
    def align_cursor_to_tile(direction: str, cursor_loc):
        def pad_floor(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f, f + np.float16(1.0))

        def pad_ceil(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f + np.float16(1.0), f)

        x, y = cursor_loc
        match direction:
            case 'UPPER':
                y = pad_ceil(y)
            case 'BOTTOM':
                y = pad_floor(y)
            case 'LEFT':
                x = pad_floor(x)
            case 'RIGHT':
                x = pad_ceil(x)
            case 'RIGHT_UPPER':
                x = pad_ceil(x)
                y = pad_ceil(y)
            case 'LEFT_UPPER':
                x = pad_floor(x)
                y = pad_ceil(y)
            case 'LEFT_BOTTOM':
                x = pad_floor(x)
                y = pad_floor(y)
            case 'RIGHT_BOTTOM':
                x = pad_ceil(x)
                y = pad_floor(y)
            case 'CENTER':
                x = np.floor(x) + 0.5
                y = np.floor(y) + 0.5
            case 'HORIZONTAL':
                y = np.floor(y) + 0.5
            case 'VERTICAL':
                x = np.floor(x) + 0.5
            case _:
                raise NotImplementedError(direction)
        utils.set_cursor_location((x, y))

    @staticmethod
    def move_cursor(direction: str, cursor_loc):
        match direction:
            case 'CENTER' | 'HORIZONTAL' | 'VERTICAL':
                UNIV_OT_Align.align_cursor_to_tile(direction, cursor_loc)
            case _:
                delta = Vector(UNIV_OT_Align.get_move_value(direction))
                utils.set_cursor_location(cursor_loc+delta)

    @staticmethod
    def get_move_value(direction):
        match direction:
            case 'UPPER':
                return 0, 1
            case 'BOTTOM':
                return 0, -1
            case 'LEFT':
                return -1, 0
            case 'RIGHT':
                return 1, 0
            case 'RIGHT_UPPER':
                return 1, 1
            case 'LEFT_UPPER':
                return -1, 1
            case 'LEFT_BOTTOM':
                return -1, -1
            case 'RIGHT_BOTTOM':
                return 1, -1
            case _:
                raise NotImplementedError(direction)
        return


def is_island_mode():
    scene = bpy.context.scene
    if scene.tool_settings.use_uv_select_sync:
        selection_mode = 'FACE' if scene.tool_settings.mesh_select_mode[2] else 'VERTEX'
    else:
        selection_mode = scene.tool_settings.uv_select_mode
    return selection_mode in ('FACE', 'ISLAND')

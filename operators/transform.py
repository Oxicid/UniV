import bpy
import math
import numpy as np

from bpy.types import Operator
from bpy.props import *

from math import pi
from ..types import BBox, Islands, FaceIsland
from .. import utils
from .. import info
from mathutils import Vector, Matrix


class UNIV_OT_Crop(Operator):
    bl_idname = 'uv.univ_crop'
    bl_label = 'Crop'
    bl_description = info.operator.crop_info
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('TO_CURSOR', 'To cursor', ''),
        ('TO_CURSOR_INDIVIDUAL', 'To cursor individual', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('INPLACE', 'Inplace', ''),
        ('INDIVIDUAL_INPLACE', 'Individual Inplace', ''),
    ))

    axis: bpy.props.EnumProperty(name='Axis', default='XY', items=(('XY', 'XY', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    padding: bpy.props.FloatProperty(name='Padding', description='Padding=1/TextureSize (1/256=0.0039)', default=0, soft_min=0, soft_max=1/256*4, max=0.49)

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
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'TO_CURSOR'
            case True, True, False:
                self.mode = 'TO_CURSOR_INDIVIDUAL'
            case False, True, False:
                self.mode = 'INDIVIDUAL'
            case False, False, True:
                self.mode = 'INPLACE'
            case False, True, True:
                self.mode = 'INDIVIDUAL_INPLACE'
            case _:
                self.report({'INFO'}, f"Event: Ctrl={event.ctrl}, Shift={event.shift}, Alt={event.alt} not implement.\n\n"
                                      f"See all variations:\n\n{self.get_event_info()}")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        return self.crop(self.mode, self.axis, self.padding, proportional=True, sync=sync, report=self.report)

    @staticmethod
    def crop(mode, axis, padding, proportional, sync, report=None):
        update_obj = utils.UMeshes([])
        umeshes = utils.UMeshes.sel_ob_with_uv()
        crop_args = [axis, padding, umeshes, update_obj, proportional, sync]
        match mode:
            case 'DEFAULT':
                UNIV_OT_Crop.crop_default(*crop_args, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_default(*crop_args, extended=False)
            case 'TO_CURSOR':
                if not (offset := utils.get_tile_from_cursor()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Crop.crop_default(*crop_args, offset=offset, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_default(*crop_args, offset=offset, extended=False)
            case 'TO_CURSOR_INDIVIDUAL':
                if not (offset := utils.get_tile_from_cursor()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Crop.crop_individual(*crop_args, offset=offset, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_individual(*crop_args, offset=offset, extended=False)
            case 'INDIVIDUAL':
                UNIV_OT_Crop.crop_individual(*crop_args, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_individual(*crop_args, extended=False)
            case 'INDIVIDUAL_INPLACE':
                UNIV_OT_Crop.crop_individual(*crop_args, inplace=True, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_individual(*crop_args, inplace=True, extended=False)
            case 'INPLACE':
                UNIV_OT_Crop.crop_inplace(*crop_args, extended=True)
                if not update_obj:
                    UNIV_OT_Crop.crop_inplace(*crop_args, extended=False)
            case _:
                raise NotImplementedError(mode)
        if not update_obj:
            if report:
                report({'INFO'}, "No faces/verts for manipulate")
            return {'CANCELLED'}

        update_obj.update()

        return {'FINISHED'}

    @staticmethod
    def crop_default(axis, padding, umeshes, update_obj, proportional, sync, offset=Vector((0, 0)), inplace=False, extended=True):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
                update_obj.umeshes.append(umesh)

        if not update_obj:
            return

        UNIV_OT_Crop.crop_ex(axis, general_bbox, inplace, islands_of_mesh, offset, padding, proportional)

    @staticmethod
    def crop_individual(axis, padding, umeshes, update_obj, proportional, sync, offset=Vector((0, 0)), inplace=False, extended=True):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    UNIV_OT_Crop.crop_ex(axis, island.calc_bbox(), inplace, (island, ), offset, padding, proportional)
                update_obj.umeshes.append(umesh)

    @staticmethod
    def crop_inplace(axis, padding, umeshes, update_obj, proportional, sync, inplace=True, extended=True):
        islands_of_tile: dict[int | list[tuple[FaceIsland | BBox]]] = {}
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    bbox = island.calc_bbox()
                    islands_of_tile.setdefault(bbox.tile_from_center, []).append((island, bbox))
                update_obj.umeshes.append(umesh)

        if not update_obj:
            return

        for tile, islands_and_bboxes in islands_of_tile.items():
            islands = []
            general_bbox = BBox()
            for island, bbox in islands_and_bboxes:
                islands.append(island)
                general_bbox.union(bbox)

            UNIV_OT_Crop.crop_ex(axis, general_bbox, inplace, islands, Vector((0, 0)), padding, proportional)

    @staticmethod
    def crop_ex(axis, bbox, inplace, islands_of_mesh, offset, padding, proportional):
        scale_x = ((1.0 - padding) / w) if (w := bbox.width) else 1
        scale_y = ((1.0 - padding) / h) if (h := bbox.height) else 1
        if proportional:
            if axis == 'XY':
                scale_x = scale_y = min(scale_x, scale_y)
            elif axis == 'X':
                scale_x = scale_y = scale_x
            else:
                scale_x = scale_y = scale_y
        else:
            if axis == 'X':
                scale_y = 1
            if axis == 'Y':
                scale_x = 1

        scale = Vector((scale_x, scale_y))
        bbox.scale(scale)
        delta = Vector((padding, padding)) / 2 - bbox.min + offset
        if axis == 'XY':
            if inplace:
                delta += Vector(math.floor(val) for val in bbox.center)
        elif axis == 'X':
            delta.y = 0
            if inplace:
                delta.x += math.floor(bbox.center_x)
        else:
            delta.x = 0
            if inplace:
                delta.y += math.floor(bbox.center_y)
        for islands in islands_of_mesh:
            islands.scale(scale, bbox.center)
            islands.move(delta)

    @staticmethod
    def get_event_info():
        return info.operator.crop_event_info_ex


class UNIV_OT_Fill(UNIV_OT_Crop):
    bl_idname = 'uv.univ_fill'
    bl_label = 'Fill'
    bl_description = info.operator.fill_info
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        return self.crop(self.mode, self.axis, self.padding, proportional=False, sync=sync, report=self.report)

    @staticmethod
    def get_event_info():
        return info.operator.fill_event_info_ex


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
    bl_idname = 'uv.univ_align'
    bl_label = 'Align'
    bl_description = info.operator.align_info
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
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        all_groups.append((island, bbox, umesh.uv_layer))
                    update_obj.umeshes.append(umesh)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    all_groups.append((corners, umesh.uv_layer))
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
            if faces := utils.calc_uv_faces(umesh.bm, umesh.uv_layer, sync, selected=selected):
                island = FaceIsland(faces, umesh.bm, umesh.uv_layer)
                bbox = island.calc_bbox()
                general_bbox.union(bbox)
                all_groups.append([island, bbox, umesh.uv_layer])
                update_obj.umeshes.append(umesh)
        for group in all_groups:
            group[1] = general_bbox
        UNIV_OT_Align.align_islands(all_groups, direction, target_bbox, invert=True)

    @staticmethod
    def align_cursor_ex(umeshes, sync, selected):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        for umesh in umeshes:
            if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                all_groups.append((corners, umesh.uv_layer))
                bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv_layer)
                general_bbox.union(bbox)
        return general_bbox

    @staticmethod
    def align_ex(direction, sync, umeshes, update_obj, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        island_mode = is_island_mode()
        for umesh in umeshes:
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)

                        all_groups.append((island, bbox, umesh.uv_layer))
                    update_obj.umeshes.append(umesh)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv_layer)
                    general_bbox.union(bbox)

                    all_groups.append((corners, umesh.uv_layer))
                    update_obj.umeshes.append(umesh)
        if island_mode:
            UNIV_OT_Align.align_islands(all_groups, direction, general_bbox)
        else:  # Vertices or Edges UV selection mode
            UNIV_OT_Align.align_corners(all_groups, direction, general_bbox)  # TODO Individual ALign for Vertical and Horizontal or all

    @staticmethod
    def move_ex(direction, sync, umeshes, update_obj, selected=True):
        island_mode = is_island_mode()
        for umesh in umeshes:
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
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
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    match direction:
                        case 'CENTER':
                            for corner in corners:
                                corner[umesh.uv_layer].uv = 0.5, 0.5
                        case 'HORIZONTAL':
                            for corner in corners:
                                corner[umesh.uv_layer].uv.x = 0.5
                        case 'VERTICAL':
                            for corner in corners:
                                corner[umesh.uv_layer].uv.y = 0.5
                        case _:
                            move_value = Vector(UNIV_OT_Align.get_move_value(direction))
                            for corner in corners:
                                corner[umesh.uv_layer].uv += move_value
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

class UNIV_OT_Flip(Operator):
    bl_idname = 'uv.univ_flip'
    bl_label = 'Flip'
    bl_description = 'FlipX and FlipY'
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('BY_CURSOR', 'By cursor', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('FLIPPED', 'Flipped', ''),
        ('FLIPPED_INDIVIDUAL', 'Flipped Individual', ''),
    ))

    axis: bpy.props.EnumProperty(name='Axis', default='X', items=(('X', 'X', ''), ('Y', 'Y', '')))

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
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'BY_CURSOR'
            case False, True, False:
                self.mode = 'INDIVIDUAL'
            case False, False, True:
                self.mode = 'FLIPPED'
            case False, True, True:
                self.mode = 'FLIPPED_INDIVIDUAL'
            case _:
                self.report({'INFO'}, f"Event: Ctrl={event.ctrl}, Shift={event.shift}, Alt={event.alt} not implement.\n\n"
                                      f"See all variations:\n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return self.flip(self.mode, self.axis, sync=bpy.context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def flip(mode, axis, sync, report=None):
        update_obj = utils.UMeshes([])
        umeshes = utils.UMeshes.sel_ob_with_uv()
        flip_args = (axis, sync,  umeshes,  update_obj)

        match mode:
            case 'DEFAULT':
                UNIV_OT_Flip.flip_ex(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Flip.flip_ex(*flip_args, extended=False)

            case 'BY_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Flip.flip_by_cursor(*flip_args, cursor=cursor_loc, extended=True)
                if not update_obj:
                    UNIV_OT_Flip.flip_by_cursor(*flip_args, cursor=cursor_loc, extended=False)

            case 'INDIVIDUAL':
                UNIV_OT_Flip.flip_individual(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Flip.flip_individual(*flip_args, extended=False)

            case 'FLIPPED':
                UNIV_OT_Flip.flip_flipped(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Flip.flip_flipped(*flip_args, extended=False)

            case 'FLIPPED_INDIVIDUAL':
                UNIV_OT_Flip.flip_flipped_individual(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Flip.flip_flipped_individual(*flip_args, extended=False)
            case _:
                raise NotImplementedError(mode)

        if not update_obj.update():
            if report:
                report({'INFO'}, "No faces/verts for manipulate")
            return {'CANCELLED'}

        return {'FINISHED'}

    @staticmethod
    def flip_ex(axis, sync,  umeshes,  update_obj, extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
                update_obj.umeshes.append(umesh)

        if not update_obj:
            return

        pivot = general_bbox.center
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for islands in islands_of_mesh:
            islands.scale(scale=scale, pivot=pivot)

    @staticmethod
    def flip_by_cursor(axis, sync,  umeshes,  update_obj, cursor, extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                islands.scale(scale=scale, pivot=cursor)
                update_obj.umeshes.append(umesh)

    @staticmethod
    def flip_individual(axis, sync,  umeshes,  update_obj, extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    island.scale(scale=scale, pivot=island.calc_bbox().center)
                update_obj.umeshes.append(umesh)

    @staticmethod
    def flip_flipped(axis, sync,  umeshes,  update_obj, extended):
        flipped_islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                flipped_islands = Islands([isl for isl in islands if isl.is_flipped()], umesh.bm, umesh.uv_layer)
                if flipped_islands:
                    general_bbox.union(flipped_islands.calc_bbox())
                    flipped_islands_of_mesh.append(flipped_islands)
                else:
                    umesh.update_flag = False
                update_obj.umeshes.append(umesh)

        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        pivot = general_bbox.center
        for islands in flipped_islands_of_mesh:
            islands.scale(scale, pivot)

    @staticmethod
    def flip_flipped_individual(axis, sync,  umeshes,  update_obj, extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                flipped_islands = [isl for isl in islands if isl.is_flipped()]
                for island in flipped_islands:
                    island.scale(scale=scale, pivot=island.calc_bbox().center)
                umesh.update_flag = bool(flipped_islands)
                update_obj.umeshes.append(umesh)

    @staticmethod
    def get_flip_scale_from_axis(axis):
        return Vector((-1, 1)) if axis == 'X' else Vector((1, -1))


class UNIV_OT_Rotate(Operator):
    bl_idname = 'uv.univ_rotate'
    bl_label = 'Rotate'
    bl_description = 'Rotate CW and Rotate CCW'
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('BY_CURSOR', 'By cursor', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('DOUBLE', 'Double', ''),
        ('DOUBLE_INDIVIDUAL', 'Double Individual', ''),
        ('DOUBLE_BY_CURSOR', 'Double by Cursor', '')
        # ('EXPAND', 'Double by Cursor', '')  # by tile
    ))

    rot_dir: bpy.props.EnumProperty(name='Direction of rotation', default='CW', items=(('CW', 'CW', ''), ('CCW', 'CCW', '')))
    angle: bpy.props.FloatProperty(name='Angle', default=pi*0.5, min=0, max=pi, soft_min=math.radians(5.0), subtype='ANGLE')

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
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'BY_CURSOR'
            case False, True, False:
                self.mode = 'INDIVIDUAL'
            case False, False, True:
                self.mode = 'DOUBLE'
            case False, True, True:
                self.mode = 'DOUBLE_INDIVIDUAL'
            case True, True, True:
                self.mode = 'DOUBLE_BY_CURSOR'
            case _:
                self.report({'INFO'}, f"Event: Ctrl={event.ctrl}, Shift={event.shift}, Alt={event.alt} not implement.\n\n"
                                      f"See all variations:\n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return self.rotate(self.mode, self.angle, self.rot_dir, sync=bpy.context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def rotate(mode, angle, rot_dir, sync, report=None):
        update_obj = utils.UMeshes([])
        umeshes = utils.UMeshes.sel_ob_with_uv()
        if 'DOUBLE' in mode:
            angle *= 2.0
        if rot_dir == 'CCW':
            angle = -angle
        flip_args = (angle, sync,  umeshes,  update_obj)

        match mode:
            case 'DEFAULT' | 'DOUBLE':
                UNIV_OT_Rotate.rotate_ex(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Rotate.rotate_ex(*flip_args, extended=False)

            case 'BY_CURSOR' | 'DOUBLE_BY_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Rotate.rotate_by_cursor(*flip_args, cursor=cursor_loc, extended=True)
                if not update_obj:
                    UNIV_OT_Rotate.rotate_by_cursor(*flip_args, cursor=cursor_loc, extended=False)

            case 'INDIVIDUAL' | 'DOUBLE_INDIVIDUAL':
                UNIV_OT_Rotate.rotate_individual(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Rotate.rotate_individual(*flip_args, extended=False)
            case _:
                raise NotImplementedError(mode)

        if not update_obj.update():
            if report:
                report({'INFO'}, "No faces/verts for manipulate")
            return {'CANCELLED'}

        return {'FINISHED'}

    @staticmethod
    def rotate_ex(angle, sync,  umeshes,  update_obj, extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
                update_obj.umeshes.append(umesh)

        pivot = general_bbox.center
        for islands in islands_of_mesh:
            islands.rotate(angle, pivot=pivot)

    @staticmethod
    def rotate_by_cursor(angle, sync,  umeshes,  update_obj, cursor, extended):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                islands.rotate(angle, pivot=cursor)
                update_obj.umeshes.append(umesh)

    @staticmethod
    def rotate_individual(angle, sync,  umeshes,  update_obj, extended):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    island.rotate(angle, pivot=island.calc_bbox().center)
                update_obj.umeshes.append(umesh)


class UNIV_OT_Sort(Operator):
    bl_idname = 'uv.univ_sort'
    bl_label = 'Sort'
    bl_description = 'Sort'
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('DEFAULT_ALIGN', 'Default Align', ''),
        ('TO_CURSOR', 'To Cursor', ''),
        ('TO_CURSOR_ALIGN', 'To Cursor Align', ''),
        ('OVERLAPPED', 'Overlapped', ''),
        ('OVERLAPPED_TO_CURSOR', 'Overlapped to Cursor', ''),
        ('OVERLAPPED_TO_CURSOR_ALIGN', 'Overlapped to Cursor Align', '')
    ))

    axis: bpy.props.EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    # align: bpy.props.BoolProperty(name='Align', default=False)
    padding: bpy.props.FloatProperty(name='Padding', default=1/2048, min=0, soft_max=0.1,)
    reverse: bpy.props.BoolProperty(name='Reverse', default=True)

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
                self.mode = 'DEFAULT'
            case False, False, True:
                self.mode = 'DEFAULT_ALIGN'
            case True, False, False:
                self.mode = 'TO_CURSOR'
            case True, False, True:
                self.mode = 'TO_CURSOR_ALIGN'
            case False, True, False:
                self.mode = 'OVERLAPPED'
            case False, True, True:
                self.mode = 'OVERLAPPED_TO_CURSOR'
            case True, True, True:
                self.mode = 'OVERLAPPED_TO_CURSOR_ALIGN'
            case _:
                self.report({'INFO'}, f"Event: Ctrl={event.ctrl}, Shift={event.shift}, Alt={event.alt} not implement.\n\n"
                                      f"See all variations:\n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return UNIV_OT_Sort.sort(self.mode, self.axis, self.padding, self.reverse,
                                 sync=bpy.context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def sort(mode, axis, padding, reverse, sync, report=None):
        update_obj = utils.UMeshes([])
        umeshes = utils.UMeshes.sel_ob_with_uv()
        cursor_loc = None
        align = 'ALIGN' in mode
        if 'CURSOR' in mode:
            if not (cursor_loc := utils.get_cursor_location()):
                if report:
                    report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}

        flip_args = (axis, align, padding, reverse, sync,  umeshes,  update_obj, cursor_loc)

        match mode:
            case 'DEFAULT' | 'DEFAULT_ALIGN' | 'TO_CURSOR' | 'TO_CURSOR_ALIGN':
                UNIV_OT_Sort.sort_ex(*flip_args, extended=True)
                if not update_obj:
                    UNIV_OT_Sort.sort_ex(*flip_args, extended=False)
            #
            # case 'OVERLAPPED' | 'OVERLAPPED_ALIGN':
            #     UNIV_OT_Sort.sort_overlapped(*flip_args, extended=True)
            #     if not update_obj:
            #         UNIV_OT_Sort.sort_overlapped(*flip_args, extended=False)
            case _:
                raise NotImplementedError(mode)

        if not update_obj.update():
            if report:
                report({'INFO'}, "No faces/verts for manipulate")
            return {'CANCELLED'}

        return {'FINISHED'}

    @staticmethod
    def sort_ex(axis, align, padding, reverse, sync,  umeshes,  update_obj, cursor=None, extended=True):
        islands_bboxes_points = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    if align:
                        isl_coords = island.calc_corner_points(convex=True)
                        bbox = BBox.calc_bbox(isl_coords)
                        general_bbox.union(bbox)

                        angle = utils.calc_min_align_angle(isl_coords)
                        if not math.isclose(angle, 0, abs_tol=0.0001):
                            island.rotate_simple(angle)
                            bbox = island.calc_bbox()
                    else:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)
                    islands_bboxes_points.append((island, bbox))

                update_obj.umeshes.append(umesh)

        islands_bboxes_points.sort(key=lambda x: x[1].max_length, reverse=reverse)

        if axis == 'AUTO':
            horizontal_sort = general_bbox.width * 2 > general_bbox.height
        else:
            horizontal_sort = axis == 'X'

        margin = general_bbox.min if cursor is None else cursor

        if horizontal_sort:
            for island, bbox in islands_bboxes_points:
                width = bbox.width
                if align and width > bbox.height:
                    width = bbox.height
                    island.rotate(pi*0.5, bbox.center)
                island.set_position(margin)
                margin.x += padding + width
        else:
            for island, bbox in islands_bboxes_points:
                height = bbox.height
                if align and bbox.width < height:
                    height = bbox.width
                    island.rotate(pi*0.5, bbox.center)
                island.set_position(margin)
                margin.y += padding + height

def calc_rotate_simple_bbox(coords, angle):
    rot_matrix = Matrix.Rotation(-angle, 2)
    bbox = BBox()
    # rotated_co = (co @ rot_matrix for co in coords)
    rotated_co = (rot_matrix @ co for co in coords)
    bbox.update(rotated_co)
    return bbox


def is_island_mode():
    scene = bpy.context.scene
    if scene.tool_settings.use_uv_select_sync:
        selection_mode = 'FACE' if scene.tool_settings.mesh_select_mode[2] else 'VERTEX_OR_EDGE'
    else:
        selection_mode = scene.tool_settings.uv_select_mode
    return selection_mode in ('FACE', 'ISLAND')

# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import typing

import bpy
import blf
from mathutils import Vector

def get_aspect_ratio(umesh=None):
    """Aspect Y"""
    if umesh:
        # Aspect from checker
        if modifiers := [m for m in umesh.obj.modifiers if m.name.startswith('UniV Checker')]:
            for node in modifiers[0]["Input_1"].node_tree.nodes:
                if node.bl_idname == 'ShaderNodeTexImage' and (image := node.image):
                    image_width, image_height = image.size
                    if image_height:
                        return image_width / image_height
        # Aspect from material
        elif mtl := umesh.obj.active_material:
            if mtl.use_nodes and (active_node := mtl.node_tree.nodes.active):
                if active_node.bl_idname == 'ShaderNodeTexImage' and (image := active_node.image):
                    image_width, image_height = image.size
                    if image_height:
                        return image_width / image_height
        return 1.0
    # Aspect from active area
    if (area := bpy.context.area) and area.type == 'IMAGE_EDITOR':
        space_data = area.spaces.active
        if space_data and space_data.image:
            image_width, image_height = space_data.image.size
            if image_height:
                return image_width / image_height
    else:
        # Aspect from VIEW3D
        for area in bpy.context.screen.areas:
            if not area.type == 'IMAGE_EDITOR':
                continue
            space_data = area.spaces.active
            if space_data and space_data.image:
                image_width, image_height = space_data.image.size
                if image_height:
                    return image_width / image_height
    return 1.0


def remove_univ_duplicate_modifiers(obj_, modifier_name):
    if obj_.type != 'MESH':
        return
    checker_modifiers_ = []
    for m_ in obj_.modifiers:
        if isinstance(m_, bpy.types.NodesModifier):
            if m_.name.startswith(modifier_name):
                if not m_.show_in_editmode:
                    m_.show_in_editmode = True
                if not m_.show_viewport:
                    m_.show_viewport = True
                checker_modifiers_.append(m_)
    if len(checker_modifiers_) <= 1:
        return

    for m_ in checker_modifiers_[:-1]:
        obj_.modifiers.remove(m_)

    # Move to bottom
    for idx, m_ in enumerate(obj_.modifiers):
        if checker_modifiers_[-1] == m_:
            if len(obj_.modifiers) - 1 != idx:
                obj_.modifiers.move(idx, len(obj_.modifiers))
            return

def is_island_mode():
    scene = bpy.context.scene
    if scene.tool_settings.use_uv_select_sync:
        selection_mode = 'FACE' if scene.tool_settings.mesh_select_mode[2] else 'VERTEX_OR_EDGE'
    else:
        selection_mode = scene.tool_settings.uv_select_mode
    return selection_mode in ('FACE', 'ISLAND')


T_mesh_select_modes = typing.Literal["VERTEX", "EDGE", "FACE"]
def get_select_mode_mesh() -> T_mesh_select_modes:
    if bpy.context.tool_settings.mesh_select_mode[2]:
        return 'FACE'
    elif bpy.context.tool_settings.mesh_select_mode[1]:
        return 'EDGE'
    else:
        return 'VERTEX'

def set_select_mode_mesh(mode: T_mesh_select_modes):
    if get_select_mode_mesh() == mode:
        return
    if mode == 'VERTEX':
        bpy.context.tool_settings.mesh_select_mode[:] = True, False, False
    elif mode == 'EDGE':
        bpy.context.tool_settings.mesh_select_mode[:] = False, True, False
    elif mode == 'FACE':
        bpy.context.tool_settings.mesh_select_mode[:] = False, False, True
    else:
        raise TypeError(f"Mode: '{mode}' not found in ('VERTEX', 'EDGE', 'FACE')")


T_uv_select_modes = typing.Literal['VERTEX', 'EDGE', 'FACE', 'ISLAND']
def get_select_mode_uv() -> T_uv_select_modes:
    return bpy.context.scene.tool_settings.uv_select_mode

def set_select_mode_uv(mode: T_uv_select_modes):
    if get_select_mode_uv() == mode:
        return
    bpy.context.scene.tool_settings.uv_select_mode = mode

def blf_size(font_id, font_size):
    if bpy.app.version > (3, 3):
        blf.size(font_id, font_size)
    else:
        blf.size(font_id, font_size, 72)

def get_max_distance_from_px(px_size: int, view: bpy.types.View2D):
    return (Vector(view.region_to_view(0, 0)) - Vector(view.region_to_view(0, px_size))).length

def get_areas_by_type(area_type: typing.Literal['VIEW_3D', 'IMAGE_EDITOR']):
    return (area for win in bpy.context.window_manager.windows for area in win.screen.areas if area.type == area_type)

def get_area_by_type(area_type: typing.Literal['VIEW_3D', 'IMAGE_EDITOR']):
    for a in get_areas_by_type(area_type):
        return a

def event_to_string(event, text=''):
    if event.ctrl:
        text += 'Ctrl + '
    if event.shift:
        text += 'Shift + '
    if event.alt:
        text += 'Alt + '
    return f'{text} Left Mouse '


def true_groupby(seq):
    """Groups and returns only identical elements"""
    seq = seq.copy()
    sorted_groups = []
    while True:
        if len(seq) <= 1:
            break

        tar_val = seq.pop()
        groups = []
        for i in range(len(seq) - 1, -1, -1):
            v = seq[i]
            if v == tar_val:
                groups.append(v)
                seq.pop(i)
        if groups:
            groups.append(tar_val)
            sorted_groups.append(groups)

    return sorted_groups

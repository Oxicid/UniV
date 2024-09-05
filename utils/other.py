# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import blf

def get_aspect_y(context):
    area = context.area
    if not area:
        return 1.0
    space_data = context.area.spaces.active
    if not space_data:
        return 1.0
    if not space_data.image:
        return 1.0
    image_width = space_data.image.size[0]
    image_height = space_data.image.size[1]
    if image_height:
        return image_width / image_height
    return 1.0

def is_island_mode():
    scene = bpy.context.scene
    if scene.tool_settings.use_uv_select_sync:
        selection_mode = 'FACE' if scene.tool_settings.mesh_select_mode[2] else 'VERTEX_OR_EDGE'
    else:
        selection_mode = scene.tool_settings.uv_select_mode
    return selection_mode in ('FACE', 'ISLAND')

def get_select_mode_mesh() -> str:
    if bpy.context.tool_settings.mesh_select_mode[2]:
        return 'FACE'
    elif bpy.context.tool_settings.mesh_select_mode[1]:
        return 'EDGE'
    else:
        return 'VERTEX'

def set_select_mode_mesh(mode: str):
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


def get_select_mode_uv() -> str:
    return bpy.context.scene.tool_settings.uv_select_mode

def set_select_mode_uv(mode: str):
    if get_select_mode_uv() == mode:
        return
    bpy.context.scene.tool_settings.uv_select_mode = mode

def blf_size(font_id, font_size):
    if bpy.app.version > (3, 3):
        blf.size(font_id, font_size)
    else:
        blf.size(font_id, font_size, 72)

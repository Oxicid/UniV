# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import blf

def get_aspect_ratio(umesh=None):
    """Aspect Y"""
    # Aspect from material
    if umesh and (mtl := umesh.obj.active_material):
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

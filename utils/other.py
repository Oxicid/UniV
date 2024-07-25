"""
Created by Oxicid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import bpy

def is_island_mode():
    scene = bpy.context.scene
    if scene.tool_settings.use_uv_select_sync:
        selection_mode = 'FACE' if scene.tool_settings.mesh_select_mode[2] else 'VERTEX_OR_EDGE'
    else:
        selection_mode = scene.tool_settings.uv_select_mode
    return selection_mode in ('FACE', 'ISLAND')

def get_select_mode_mesh() -> str:
    if bpy.context.tool_settings.mesh_select_mode[0]:
        return 'VERTEX'
    elif bpy.context.tool_settings.mesh_select_mode[1]:
        return 'EDGE'
    else:
        return 'FACE'

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

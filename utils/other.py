# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import bpy
import blf
import typing
import contextlib
from mathutils import Vector
from itertools import groupby


def get_aspect_ratio(umesh=None):
    """Aspect Y. Used for multiply by Y axis."""
    if umesh:
        # Aspect from checker
        if modifiers := [m for m in umesh.obj.modifiers if m.name.startswith('UniV Checker')]:
            socket = 'Socket_1' if 'Socket_1' in modifiers[0] else 'Input_1'

            if mtl := modifiers[0][socket]:
                for node in mtl.node_tree.nodes:
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


def get_active_image_size():
    if (area := bpy.context.area) and area.type == 'IMAGE_EDITOR':
        space_data = area.spaces.active
        if space_data and space_data.image:
            image_width, image_height = space_data.image.size
            if image_height:
                return image_width, image_height


def remove_univ_duplicate_modifiers(obj_, modifier_name, toggle_enable=False):
    if obj_.type != 'MESH':
        return
    checker_modifiers_ = []
    for m_ in obj_.modifiers:
        if isinstance(m_, bpy.types.NodesModifier):
            if m_.name.startswith(modifier_name):
                if not toggle_enable:
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


T_mesh_select_modes = typing.Literal["VERT", "EDGE", "FACE"]


def get_select_mode_mesh() -> T_mesh_select_modes:
    mode = bpy.context.scene.tool_settings.mesh_select_mode
    if mode[0]:
        return 'VERT'
    elif mode[1]:
        return 'EDGE'
    else:
        return 'FACE'


def set_select_mode_mesh(mode: T_mesh_select_modes):
    if get_select_mode_mesh() == mode:
        return
    if mode == 'VERT':
        bpy.context.tool_settings.mesh_select_mode[:] = True, False, False
    elif mode == 'EDGE':
        bpy.context.tool_settings.mesh_select_mode[:] = False, True, False
    elif mode == 'FACE':
        bpy.context.tool_settings.mesh_select_mode[:] = False, False, True
    else:
        raise TypeError(f"Mode: '{mode}' not found in ('VERT', 'EDGE', 'FACE')")


T_uv_select_modes = typing.Literal['VERT', 'EDGE', 'FACE', 'ISLAND']


def get_select_mode_uv() -> T_uv_select_modes:
    if (mode := bpy.context.scene.tool_settings.uv_select_mode) == 'VERTEX':
        return 'VERT'
    return mode  # noqa


def set_select_mode_uv(mode: T_uv_select_modes):
    if get_select_mode_uv() == mode:
        return
    if mode == 'VERT':
        mode = 'VERTEX'
    bpy.context.scene.tool_settings.uv_select_mode = mode


def blf_size(font_id, font_size):
    if bpy.app.version > (3, 3):
        blf.size(font_id, font_size)
    else:
        blf.size(font_id, font_size, 72)


def get_max_distance_from_px(px_size: int, view: bpy.types.View2D):
    return (Vector(view.region_to_view(0, 0)) - Vector(view.region_to_view(0, px_size))).length


def get_areas_by_type(area_type: typing.Literal['VIEW_3D', 'IMAGE_EDITOR'] = 'IMAGE_EDITOR'):
    return (area for win in bpy.context.window_manager.windows for area in win.screen.areas if area.type == area_type)


def get_area_by_type(area_type: typing.Literal['VIEW_3D', 'IMAGE_EDITOR'] = 'IMAGE_EDITOR'):
    for a in get_areas_by_type(area_type):
        return a
    return None


def update_univ_panels():
    import itertools
    for image in itertools.chain(get_areas_by_type('VIEW_3D'), get_areas_by_type()):
        for reg in image.regions:
            if reg.type == 'UI':
                if hasattr(reg, 'active_panel_category'):
                    if reg.active_panel_category == 'UniV':
                        reg.tag_redraw()
                else:
                    reg.tag_redraw()


@contextlib.contextmanager
def operator_context(layout, op_context):
    """Context manager that temporarily overrides the operator context.

    >>> with operator_context(layout, 'INVOKE_REGION_CHANNELS'):
    ...     layout.operator("anim.channels_delete")
    """

    orig_context = layout.operator_context
    layout.operator_context = op_context
    try:
        yield
    finally:
        layout.operator_context = orig_context

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


def all_contiguous_subgroups(seq):
    groups = []
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, n + 1):
            groups.append(seq[i:j])
    return groups


def split_by_similarity(lst, key=None):
    """It differs from Group By in that groups are strictly separated and not reversed.
        true_groupby:        1,0,1,1 -> [1,1,1],[0]
        split_by_similarity: 1,0,1,1 -> [1],[0],[1,1]"""
    if key:
        return [list(group) for _, group in groupby(lst, key=key)]
    else:
        return [list(group) for _, group in groupby(lst)]


def reshape_to_pair(lst: list[Vector]) -> list[tuple[Vector, Vector]]:
    return list(zip(*[iter(lst)] * 2))  # noqa


def load_lib(lib_name: str, root_path=None, lib_ext: typing.Literal['dll', 'so', 'dylib'] | None=None):
    if lib_ext is None:
        import platform
        system = platform.system()
        if system == "Windows":
            lib_ext = "dll"
        elif system == "Darwin":
            lib_ext = "dylib"
        else:
            lib_ext = "so"

    from pathlib import Path
    if root_path is None:
        root_path = Path(__file__).parent.parent.parent
    else:
        root_path = Path(root_path)

    # search 'univ' folder
    univ_dir = None
    for p in root_path.iterdir():
        if p.is_dir():
            name = p.name.lower()
            if name.startswith('univ') and name != 'univ_pro':
                univ_dir = p
                break

    assert univ_dir is not None, f"No directory starting with 'univ' found in {root_path!r}"

    # recursive search lib
    lib_filename = f"{lib_name}.{lib_ext}"
    candidates = list(univ_dir.rglob(lib_filename))

    lib = None
    last_err = None
    from ctypes import CDLL
    for p in candidates:
        try:
            lib = CDLL(str(p))
            break
        except OSError as e:
            last_err = e

    if lib is None:
        raise OSError(f"Could not load {lib_filename}. Tried: {candidates!r}\nLast error: {last_err}")
    return lib
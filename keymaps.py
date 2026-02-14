# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from collections import defaultdict

keys = []
keys_ws = []
keys_areas = ['UV Editor', 'Window', 'Object Mode', 'Mesh']  # TODO: Rename to spaces
keys_areas_workspace = ['3D View Tool: Object, UniV', '3D View Tool: Edit Mesh, UniV']
other_conflict_areas = ['Frames']  # NOTE: not actual after delete keymaps for align?


def add_mesh_keymaps(km, univ_pro):
    # Grow
    kmi = km.keymap_items.new('mesh.univ_select_grow', 'WHEELUPMOUSE', 'PRESS', ctrl=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_grow', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    # Edge grow
    kmi = km.keymap_items.new('mesh.univ_select_edge_grow', 'WHEELUPMOUSE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_edge_grow', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    if univ_pro:
        # Select loop
        kmi = km.keymap_items.new('mesh.univ_select_loop', 'WHEELUPMOUSE', 'PRESS', alt=True)
        keys.append((km, kmi))

        kmi = km.keymap_items.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK')
        keys.append((km, kmi))

        kmi = km.keymap_items.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK', shift=True)
        keys.append((km, kmi))


def add_keymaps():
    global keys

    if not (kc := bpy.context.window_manager.keyconfigs.addon):
        return  # Can be None in background mode.

    try:
        from . import univ_pro
    except ImportError:
        univ_pro = None

    # Object Mode
    km = kc.keymaps.new(name='Object Mode')
    kmi = km.keymap_items.new('object.univ_join', 'J', 'PRESS', ctrl=True)
    keys.append((km, kmi))

    # Pie Menu
    kmi = km.keymap_items.new("wm.call_menu_pie", 'ACCENT_GRAVE', 'PRESS')
    kmi.properties.name = "VIEW3D_MT_PIE_univ_obj"
    keys.append((km, kmi))

    # Mesh
    km = kc.keymaps.new(name='Mesh')

    # Pie Menu
    kmi = km.keymap_items.new("wm.call_menu_pie", 'ACCENT_GRAVE', 'PRESS')
    kmi.properties.name = "VIEW3D_MT_PIE_univ_edit"
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_linked_pick', 'WHEELUPMOUSE', 'PRESS', shift=True)
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_deselect_linked_pick', 'WHEELDOWNMOUSE', 'PRESS', shift=True)
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_linked', 'WHEELUPMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.select = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_linked', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.select = False
    keys.append((km, kmi))

    add_mesh_keymaps(km, univ_pro)

    # Window
    km = kc.keymaps.new(name='Window')

    kmi = km.keymap_items.new('wm.univ_split_uv_toggle', 'T', 'PRESS', shift=True)
    kmi.properties.mode = 'SPLIT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('wm.univ_toggle_panels_by_cursor', 'T', 'PRESS', alt=True)
    keys.append((km, kmi))

    # UV Editor
    km = kc.keymaps.new(name='UV Editor')

    # Pie Menus
    kmi = km.keymap_items.new("wm.call_menu_pie", 'F1', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_inspect"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'ACCENT_GRAVE', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_edit"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'X', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_align"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'D', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_misc"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'Q', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_favorites_edit"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'T', 'PRESS')
    kmi.properties.name = "IMAGE_MT_PIE_univ_transform"
    keys.append((km, kmi))

    # Select
    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELUPMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_pick', 'WHEELUPMOUSE', 'PRESS', shift=True)
    kmi.properties.select = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_pick', 'WHEELDOWNMOUSE', 'PRESS', shift=True)
    kmi.properties.select = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_grow', 'WHEELUPMOUSE', 'PRESS', ctrl=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_grow', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    # Edge Grow (Conflict)
    kmi = km.keymap_items.new('uv.univ_select_edge_grow', 'WHEELUPMOUSE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_edge_grow', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    if univ_pro:
        kmi = km.keymap_items.new('uv.univ_select_loop', 'WHEELUPMOUSE', 'PRESS', alt=True)
        keys.append((km, kmi))

        kmi = km.keymap_items.new('uv.univ_select_similar', 'G', 'PRESS', shift=True)
        keys.append((km, kmi))

    # Flip
    kmi = km.keymap_items.new('uv.univ_flip', 'F', 'PRESS')
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_mode', 'ONE', 'PRESS')
    kmi.properties.type = 'VERTEX'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_mode', 'TWO', 'PRESS')
    kmi.properties.type = 'EDGE'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_mode', 'THREE', 'PRESS')
    kmi.properties.type = 'FACE'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_mode', 'FOUR', 'PRESS')
    kmi.properties.type = 'ISLAND'
    keys.append((km, kmi))

    # Rotate
    # Default. CW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS')
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'DEFAULT'
    keys.append((km, kmi))

    # Default. CCW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'DEFAULT'
    keys.append((km, kmi))

    # Default. CW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True)
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'INDIVIDUAL'
    keys.append((km, kmi))

    # Default. CCW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True, alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'INDIVIDUAL'
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'A', 'PRESS', shift=True)
    kmi.properties.name = "IMAGE_MT_PIE_univ_texel"
    keys.append((km, kmi))

    # Relax
    kmi = km.keymap_items.new('uv.univ_relax', 'R', 'PRESS', alt=True)
    keys.append((km, kmi))

    # Unwrap
    kmi = km.keymap_items.new('uv.univ_unwrap', 'U', 'PRESS')
    if univ_pro:
        kmi.properties.unwrap_along = 'UV'
    keys.append((km, kmi))

    # Pack
    kmi = km.keymap_items.new('uv.univ_pack', 'P', 'PRESS')
    keys.append((km, kmi))

    # Quadrify
    kmi = km.keymap_items.new('uv.univ_quadrify', 'E', 'PRESS')
    keys.append((km, kmi))

    # Straight
    kmi = km.keymap_items.new('uv.univ_straight', 'E', 'PRESS', shift=True)
    keys.append((km, kmi))

    # Weld
    kmi = km.keymap_items.new('uv.univ_weld', 'W', 'PRESS')
    kmi.properties.use_by_distance = False
    keys.append((km, kmi))

    # Stitch
    kmi = km.keymap_items.new('uv.univ_stitch', 'W', 'PRESS', shift=True)
    keys.append((km, kmi))

    # Quick Snap
    kmi = km.keymap_items.new('uv.univ_quick_snap', 'V', 'PRESS')
    kmi.properties.quick_start = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_quick_snap', 'V', 'PRESS', alt=True)
    kmi.properties.quick_start = False
    keys.append((km, kmi))

    # Drag
    if univ_pro:
        kmi = km.keymap_items.new('uv.univ_drag', 'LEFTMOUSE', 'ANY', alt=True)
        keys.append((km, kmi))

    # Cut
    kmi = km.keymap_items.new('uv.univ_cut', 'C', 'PRESS')
    kmi.properties.addition = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_cut', 'C', 'PRESS', shift=True)
    kmi.properties.addition = True
    keys.append((km, kmi))

    # Stack
    kmi = km.keymap_items.new('uv.univ_stack', 'S', 'PRESS', alt=True)
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_symmetrize', 'X', 'PRESS', alt=True)
    keys.append((km, kmi))

    # Orient
    kmi = km.keymap_items.new('uv.univ_orient', 'O', 'PRESS')
    kmi.properties.edge_dir = 'BOTH'
    keys.append((km, kmi))

    # Stretch Toggle
    kmi = km.keymap_items.new('uv.univ_stretch_uv_toggle', 'Z', 'DOUBLE_CLICK')
    kmi.properties.swap = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_stretch_uv_toggle', 'Z', 'CLICK')
    kmi.properties.swap = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_show_modified_uv_edges_toggle', 'Z', 'PRESS', alt=True)
    keys.append((km, kmi))

    # Hide
    kmi = km.keymap_items.new('uv.univ_hide', 'H', 'PRESS')
    kmi.properties.unselected = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_hide', 'H', 'PRESS', shift=True)
    kmi.properties.unselected = True
    keys.append((km, kmi))

    # Set Cursor 2D
    kmi = km.keymap_items.new('uv.univ_set_cursor_2d', 'MIDDLEMOUSE', 'PRESS', ctrl=True, shift=True)
    keys.append((km, kmi))

    # Focus
    kmi = km.keymap_items.new('uv.univ_focus', 'NUMPAD_PERIOD', 'PRESS')
    keys.append((km, kmi))

    for _, kmi in keys:
        kmi.active = False


def add_keymaps_ws():
    global keys_ws
    if not (kc := bpy.context.window_manager.keyconfigs.addon):
        return  # Can be None in background mode.

    try:
        from . import univ_pro
    except ImportError:
        univ_pro = None

    # Workspace keymaps
    def workspace_duplicates(km_ws):
        kmi_ws = km_ws.keymap_items.new("mesh.univ_gravity", 'O', 'PRESS')
        keys_ws.append((km_ws, kmi_ws))

        kmi_ws = km_ws.keymap_items.new("wm.call_menu_pie", 'A', 'PRESS', shift=True)
        kmi_ws.properties.name = "VIEW3D_MT_PIE_univ_texel"
        keys_ws.append((km_ws, kmi_ws))

        kmi_ws = km_ws.keymap_items.new("wm.call_menu_pie", 'Q', 'PRESS', shift=True)
        kmi_ws.properties.name = "VIEW3D_MT_PIE_univ_projection"
        keys_ws.append((km_ws, kmi_ws))

    # Edit Mode
    km = kc.keymaps.new(name='3D View Tool: Edit Mesh, UniV', space_type='VIEW_3D', tool=True)

    kmi = km.keymap_items.new("wm.call_menu_pie", 'D', 'PRESS')
    kmi.properties.name = "VIEW3D_MT_PIE_univ_misc"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("wm.call_menu_pie", 'Q', 'PRESS')
    kmi.properties.name = "VIEW3D_MT_PIE_univ_favorites_edit"
    keys.append((km, kmi))

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG')
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', shift=True)
    kmi.properties.mode = 'ADD'
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', ctrl=True)
    kmi.properties.mode = 'SUB'
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_cut", 'C', 'PRESS')
    kmi.properties.addition = False
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_cut", 'C', 'PRESS', shift=True)
    kmi.properties.addition = True
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_weld", 'W', 'PRESS')
    kmi.properties.use_by_distance = False
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_stitch", 'W', 'PRESS', shift=True)
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_relax", 'R', 'PRESS', alt=True)
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_unwrap", 'U', 'PRESS')
    # if univ_pro:
    #     kmi.properties.unwrap_along = 'UV'
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_stack", 'S', 'PRESS', alt=True)
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_seam_border", 'B', 'PRESS', alt=True)
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("mesh.univ_angle", 'A', 'PRESS', ctrl=True)
    keys_ws.append((km, kmi))

    if univ_pro:
        kmi = km.keymap_items.new('mesh.univ_select_similar', 'G', 'PRESS', shift=True)
        keys.append((km, kmi))

    workspace_duplicates(km)

    # Object Mode
    km = kc.keymaps.new(name='3D View Tool: Object, UniV', space_type='VIEW_3D', tool=True)

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG')
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', shift=True)
    kmi.properties.mode = 'ADD'
    keys_ws.append((km, kmi))

    kmi = km.keymap_items.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', ctrl=True)
    kmi.properties.mode = 'SUB'
    keys_ws.append((km, kmi))

    workspace_duplicates(km)


def remove_keymaps():
    global keys
    import traceback
    from .preferences import debug

    for km, kmi in keys:
        try:
            km.keymap_items.remove(kmi)
        except RuntimeError:
            if debug():
                traceback.print_exc()
    keys.clear()


def remove_keymaps_ws():
    global keys_ws
    import traceback
    from .preferences import debug

    for km, kmi in keys_ws:
        try:
            km.keymap_items.remove(kmi)
        except RuntimeError:
            if debug():
                traceback.print_exc()
    keys_ws.clear()


_EVENT_TYPES = set()
_EVENT_TYPE_MAP = {}
_EVENT_TYPE_MAP_EXTRA = {}


class ConflictFilter:
    def __init__(self):
        self.univ_keys = []
        self.default_keys = []
        self.user_defined = []

    def __str__(self):
        key_name = self.univ_keys[0].to_string()
        return f'{key_name: <30}: UniV - {len(self.univ_keys)}, Blender - {len(self.default_keys)}, User - {len(self.user_defined)}'

    @staticmethod
    def get_conflict_filtered_keymaps(keys_areas_):
        kc = bpy.context.window_manager.keyconfigs.user

        for area in keys_areas_:
            km = kc.keymaps[area]

            conflict_filter = defaultdict(ConflictFilter)
            for kmi in km.keymap_items:
                if ('.univ_' in kmi.idname or
                        'wm.call_menu_pie' == kmi.idname and kmi.name == 'UniV Pie'):
                    keymap_name = kmi.to_string()
                    conflict_filter[keymap_name].univ_keys.append(kmi)

            if not conflict_filter:
                continue

            if area == 'Window':
                areas_ = (area, *other_conflict_areas, '3D View')
            else:
                areas_ = (area, *other_conflict_areas)
            for area1 in areas_:
                km = kc.keymaps[area1]
                for kmi in km.keymap_items:
                    keymap_name = kmi.to_string()
                    if keymap_name in conflict_filter and '.univ_' not in kmi.idname and kmi.name != 'UniV Pie':
                        if kmi.is_user_defined:
                            conflict_filter[keymap_name].user_defined.append((km, kmi))
                        else:
                            conflict_filter[keymap_name].default_keys.append((km, kmi))
            yield area, kc, km, conflict_filter

    @classmethod
    def get_conflict_filtered_keymaps_with_exclude(cls, keys_areas_):
        from .preferences import prefs
        keymap_name_filter = prefs().keymap_name_filter.strip().lower()
        filter_name_fn = cls.filter_by_name

        keymap_key_filter = prefs().keymap_key_filter.strip().lower()
        if keymap_key_filter:
            filter_key_fn = cls.filter_by_key(keymap_key_filter)
        else:
            def filter_key_fn(a): return a  # pycharm warning

        kc = bpy.context.window_manager.keyconfigs.user

        for area in keys_areas_:
            km = kc.keymaps[area]

            conflict_filter = defaultdict(ConflictFilter)
            for kmi in km.keymap_items:
                if ('.univ_' in kmi.idname or
                        'wm.call_menu_pie' == kmi.idname and kmi.name == 'UniV Pie'):
                    # Filter by name and by key
                    if keymap_name_filter and not filter_name_fn(kmi, keymap_name_filter):
                        continue
                    if keymap_key_filter and not filter_key_fn(kmi):
                        continue

                    keymap_name = kmi.to_string()
                    conflict_filter[keymap_name].univ_keys.append(kmi)

            if not conflict_filter:
                continue

            # Check for potential keymap conflicts with addon in other spaces
            if area == 'Window':
                areas_ = (area, *other_conflict_areas, '3D View')
            else:
                areas_ = (area, *other_conflict_areas)
            for area1 in areas_:
                km = kc.keymaps[area1]
                for kmi in km.keymap_items:
                    keymap_name = kmi.to_string()
                    if keymap_name in conflict_filter and '.univ_' not in kmi.idname and kmi.name != 'UniV Pie':
                        # Filter by name and by key
                        if keymap_name_filter and not filter_name_fn(kmi, keymap_name_filter):
                            continue
                        if keymap_key_filter and not filter_key_fn(kmi):
                            continue

                        if kmi.is_user_defined:
                            conflict_filter[keymap_name].user_defined.append((km, kmi))
                        else:
                            conflict_filter[keymap_name].default_keys.append((km, kmi))
            yield area, kc, km, conflict_filter

    @classmethod
    def get_conflict_filtered_keymaps_with_exclude_ws(cls, keys_areas_):
        from .preferences import prefs
        keymap_name_filter = prefs().keymap_name_filter.strip().lower()
        filter_name_fn = cls.filter_by_name

        keymap_key_filter = prefs().keymap_key_filter.strip().lower()
        if keymap_key_filter:
            filter_key_fn = cls.filter_by_key(keymap_key_filter)
        else:
            def filter_key_fn(a): return a  # pycharm warning

        kc = bpy.context.window_manager.keyconfigs.user

        for area in keys_areas_:
            km = kc.keymaps[area]

            conflict_filter = defaultdict(ConflictFilter)
            for kmi in km.keymap_items:
                # Filter by name and by key
                if keymap_name_filter and not filter_name_fn(kmi, keymap_name_filter):
                    continue
                if keymap_key_filter and not filter_key_fn(kmi):
                    continue

                keymap_name = kmi.to_string()
                conflict_filter[keymap_name].univ_keys.append(kmi)

            if not conflict_filter:
                continue
            if area == 'Window':
                areas_ = (area, *other_conflict_areas, '3D View')
            else:
                areas_ = (area, *other_conflict_areas)
            for area1 in areas_:
                km = kc.keymaps[area1]
                for kmi in km.keymap_items:
                    keymap_name = kmi.to_string()
                    if keymap_name in conflict_filter and '.univ_' not in kmi.idname and kmi.name != 'UniV Pie':
                        # Filter by name and by key
                        if keymap_name_filter and not filter_name_fn(kmi, keymap_name_filter):
                            continue
                        if keymap_key_filter and not filter_key_fn(kmi):
                            continue

                        if kmi.is_user_defined:
                            conflict_filter[keymap_name].user_defined.append((km, kmi))
                        else:
                            conflict_filter[keymap_name].default_keys.append((km, kmi))
            yield area, kc, km, conflict_filter

    @staticmethod
    def filter_by_name(kmi, filter_text):
        return (filter_text in kmi.idname.lower() or
                filter_text in kmi.name.lower())

    # rna_keymap_ui.py
    @staticmethod
    def filter_by_key(filter_text):
        if not _EVENT_TYPES:
            enum = bpy.types.Event.bl_rna.properties["type"].enum_items
            _EVENT_TYPES.update(enum.keys())
            _EVENT_TYPE_MAP.update({item.name.replace(" ", "_").upper(): key
                                    for key, item in enum.items()})

            del enum
            _EVENT_TYPE_MAP_EXTRA.update({
                "`": 'ACCENT_GRAVE',
                "*": 'NUMPAD_ASTERIX',
                "/": 'NUMPAD_SLASH',
                '+': 'NUMPAD_PLUS',
                "-": 'NUMPAD_MINUS',
                ".": 'NUMPAD_PERIOD',
                "'": 'QUOTE',
                "RMB": 'RIGHTMOUSE',
                "LMB": 'LEFTMOUSE',
                "MMB": 'MIDDLEMOUSE',
            })
            _EVENT_TYPE_MAP_EXTRA.update({f"{i}": f"NUMPAD_{i}" for i in range(10)})
        # done with once off init

        filter_text_split = filter_text.split()

        # Modifier {kmi.attribute: name} mapping
        key_mod = {
            "ctrl": "ctrl",
            "alt": "alt",
            "shift": "shift",
            "cmd": "oskey",
            "oskey": "oskey",
            "any": "any",
        }
        # KeyMapItem like dict, use for comparing against
        # attr: {states, ...}
        kmi_test_dict = {}
        # Special handling of 'type' using a list if sets,
        # keymap items must match against all.
        kmi_test_type = []

        # initialize? - so if a kmi has a MOD assigned it won't show up.
        # for kv in key_mod.values():
        #     kmi_test_dict[kv] = {False}

        # altname: attr
        for kk, kv in key_mod.items():
            if kk in filter_text_split:
                filter_text_split.remove(kk)
                kmi_test_dict[kv] = {True}

        # what's left should be the event type
        def kmi_type_set_from_string(kmi_type):
            kmi_type = kmi_type.upper()
            kmi_type_set = set()

            if kmi_type in _EVENT_TYPES:
                kmi_type_set.add(kmi_type)

            if not kmi_type_set or len(kmi_type) > 1:
                # replacement table
                for event_type_map in (_EVENT_TYPE_MAP, _EVENT_TYPE_MAP_EXTRA):
                    kmi_type_test = event_type_map.get(kmi_type)
                    if kmi_type_test is not None:
                        kmi_type_set.add(kmi_type_test)
                    else:
                        # print("Unknown Type:", kmi_type_)

                        # Partial match
                        for k, v in event_type_map.items():
                            if (kmi_type in k) or (kmi_type in v):
                                kmi_type_set.add(v)
            return kmi_type_set

        for i, kmi_typ in enumerate(filter_text_split):
            kmi_typ_set = kmi_type_set_from_string(kmi_typ)

            if kmi_typ_set:
                kmi_test_type.append(kmi_typ_set)
        # tiny optimization, sort sets so the smallest is first
        # improve chances of failing early
        kmi_test_type.sort(key=lambda kmi_type_set: len(kmi_type_set))

        # main filter func, runs many times
        def filter_func(kmi):
            for kk_, ki in kmi_test_dict.items():
                val = getattr(kmi, kk_)
                if val not in ki:
                    return False

            # special handling of 'type'
            for ki in kmi_test_type:
                val = kmi.type
                if val == 'NONE' or val not in ki:
                    # exception for 'type'
                    # also inspect 'key_modifier' as a fallback
                    val = kmi.key_modifier
                    if not (val == 'NONE' or val not in ki):
                        continue
                    return False

            return True
        return filter_func


class UNIV_RestoreKeymaps(bpy.types.Operator):
    bl_idname = 'wm.univ_keymaps_config'
    bl_label = 'Keymaps Config'
    bl_description = 'Keymaps Config\n\n' \
                     'Restore - Resets properties and assigned keys, enable keymaps (doesn`t restore deleted keymaps)\n' \
                     'Off/On - Enable/disable keymaps\n' \
                     'Delete User - Remove manually installed UniV keymaps\n' \
                     'Resolve Conflicts - Resolve all conflicts with UniV keymaps (except in cases where the UniV keymap is disabled)'

    mode: bpy.props.EnumProperty(name='Mode', default='RESTORE',
                                 items=(
                                     ('RESTORE', 'Restore', ''),
                                     ('TOGGLE', 'Off/On', ''),
                                     ('DELETE_USER', 'Delete User', ''),
                                     ('RESOLVE_ALL', 'Resolve Conflicts', '')

                                 ))

    def execute(self, context):
        kc = context.window_manager.keyconfigs.user
        counter = 0

        def keymap_items():
            for _area in keys_areas + keys_areas_workspace:
                _km = kc.keymaps[_area]
                for _kmi in _km.keymap_items:
                    if '.univ_' in _kmi.idname:
                        yield _km, _kmi
                    elif 'wm.call_menu_pie' == _kmi.idname and _kmi.name == 'UniV Pie':
                        yield _km, _kmi

        if self.mode == 'DEFAULT':
            for km, kmi in keymap_items():
                if not kmi.is_user_defined:
                    activ_before = kmi.active
                    to_str_before = kmi.to_string()
                    properties_before = [getattr(kmi.properties, str_props)
                                         for str_props in dir(kmi.properties) if not str_props.startswith('__')]

                    km.restore_item_to_default(kmi)

                    if not activ_before:
                        kmi.active = True
                        counter += 1
                        continue
                    else:
                        kmi.active = True

                    if to_str_before != kmi.to_string():
                        counter += 1
                        continue
                    if properties_before != [getattr(kmi.properties, str_props) for str_props in dir(kmi.properties) if not str_props.startswith('__')]:
                        counter += 1

            message = f'Reset to default {counter} addon keymaps' if counter else 'All addon keymaps is default'
        elif self.mode == 'RESOLVE_ALL':

            for area, kc, km, filtered_keymaps in ConflictFilter.get_conflict_filtered_keymaps(
                    keys_areas + keys_areas_workspace):
                for config_filtered in filtered_keymaps.values():
                    if not any(univ_kmi.active for univ_kmi in config_filtered.univ_keys):
                        continue
                    for (_, kmi_) in config_filtered.default_keys:
                        if kmi_.active:
                            counter += 1
                            kmi_.active = False
                    for (_, kmi_) in config_filtered.user_defined:
                        if kmi_.active:
                            counter += 1
                            kmi_.active = False
            message = f'Disabled {counter} keymaps' if counter else 'Not found keymaps with conflicts'

        # elif self.mode == 'RESTORE':
        #     pass
            # for km, kmi in keymap_items():
            #     if not kmi.is_user_defined:
            #         km.keymap_items.remove(kmi)

            # global keys
            # kc = bpy.context.window_manager.keyconfigs.addon
            # new_keys = []
            # for addon_km, addon_kmi in keys:
            #     user_km = kc.keymaps[addon_km.name]
            #     key = user_km, user_km.keymap_items.new_from_item(addon_kmi)
            #     new_keys.append(key)
            #     print(key[1])
            # remove_keymaps()
            # keys.extend(new_keys)
            # remove_keymaps()
            # add_keymaps()

        elif self.mode == 'DELETE_USER':
            for km, kmi in keymap_items():
                if kmi.is_user_defined:
                    counter += 1
                    km.keymap_items.remove(kmi)
            message = f'Deleted {counter} user keymaps' if counter else 'Not found user keymaps'
        else:
            active_states = set()
            for _, kmi in keymap_items():
                active_states.add(kmi.active)

            state = False if (len(active_states) == 2) else (False in active_states)

            if state:
                for _, kmi in keymap_items():
                    if not kmi.active:
                        kmi.active = True
                        counter += 1
                message = f'Enabled {counter} keymaps' if counter else 'Not found keymaps'
            else:
                for _, kmi in keymap_items():
                    if kmi.active:
                        kmi.active = False
                        counter += 1

                message = f'Disable {counter} keymaps' if counter else 'Not found keymaps'

        bpy.context.preferences.is_dirty = True
        self.report({'INFO'}, message)
        return {'FINISHED'}

# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from collections import defaultdict

from importlib.util import find_spec
univ_pro_exist = find_spec(f"{__package__}.univ_pro") is not None
del find_spec

keys = []
keys_areas = ['UV Editor', 'Window', 'Object Mode', 'Mesh']  # TODO: Rename to spaces
keys_areas_workspace = ['3D View Tool: Object, UniV', '3D View Tool: Edit Mesh, UniV']
other_conflict_areas = ['Frames']  # NOTE: not actual after delete keymaps for align?


class UKeymap:
    class UKeymapPropertyController:
        def __init__(self, keymap):
            object.__setattr__(self, "_ZGenerator", keymap)

        def __setattr__(self, name, value):
            if name.startswith("_ZGenerator"):
                object.__setattr__(self, name, value)
                return
            _, kmi = self._ZGenerator.items[-1]  # noqa
            setattr(kmi.properties, name, value)

    def __init__(self):
        self.km = None
        self.items = []
        self.prop = self.UKeymapPropertyController(self)

    def new(self, idname: str, type: str, value: str="PRESS", **kw):  # noqa
        expected = {"any", "shift", "ctrl", "alt", "oskey" "key_modifier", "direction", "repeat", "head"}
        for key in kw:
            if key not in expected:
                raise ValueError(f"Expected {expected!r} keywords, given {key!r}")

        kmi = self.km.keymap_items.new(idname, type, value, **kw)
        self.items.append((self.km, kmi))
        return self

    def new_keymaps(self, kc, name):
        self.km = kc.keymaps.new(name=name)


    @classmethod
    def add_keymaps(cls):
        kc = bpy.context.window_manager.keyconfigs.addon
        if not kc:
            return  # Can be None in background mode.

        km = cls()

        ##################################################
        # Object Mode
        ##################################################
        km.new_keymaps(kc, 'Object Mode')
        km.new('object.univ_join', 'J', ctrl=True)
        if univ_pro_exist:
            km.new('object.univ_isolate', 'NUMPAD_SLASH')
            km.new('object.univ_isolate', 'SLASH')
        # Pie Menu
        km.new("wm.call_menu_pie", 'ACCENT_GRAVE').prop.name = "VIEW3D_MT_PIE_univ_obj"

        ##################################################
        # Mesh
        ##################################################
        km.new_keymaps(kc, name='Mesh')

        # Pie Menu
        km.new("wm.call_menu_pie", 'ACCENT_GRAVE').prop.name = "VIEW3D_MT_PIE_univ_edit"

        if univ_pro_exist:
            km.new('object.univ_isolate', 'NUMPAD_SLASH')
            km.new('object.univ_isolate', 'SLASH')

        ## Selection
        # Select Linked
        km.new('mesh.univ_select_linked_pick', 'WHEELUPMOUSE', shift=True)
        km.new('mesh.univ_deselect_linked_pick', 'WHEELDOWNMOUSE', shift=True)
        km.new('mesh.univ_select_linked', 'WHEELUPMOUSE', ctrl=True, shift=True).prop.select = True
        km.new('mesh.univ_select_linked', 'WHEELDOWNMOUSE', ctrl=True, shift=True).prop.select = False

        km.new("mesh.univ_local_invert_selection", "I", ctrl=True, shift=True)

        cls._add_mesh_keymaps(km)

        ##################################################
        # Window
        ##################################################
        km.new_keymaps(kc, name='Window')
        km.new('wm.univ_split_uv_toggle', 'T', shift=True).prop.mode = 'SPLIT'
        km.new('wm.univ_toggle_panels_by_cursor', 'T', alt=True)

        ##################################################
        # UV Editor
        ##################################################
        km.new_keymaps(kc, name='UV Editor')

        # Pie Menus
        km.new("wm.call_menu_pie", 'F1').prop.name = "IMAGE_MT_PIE_univ_inspect"
        km.new("wm.call_menu_pie", 'ACCENT_GRAVE').prop.name = "IMAGE_MT_PIE_univ_edit"
        km.new("wm.call_menu_pie", 'X').prop.name = "IMAGE_MT_PIE_univ_align"
        km.new("wm.call_menu_pie", 'D').prop.name = "IMAGE_MT_PIE_univ_misc"
        km.new("wm.call_menu_pie", 'Q').prop.name = "IMAGE_MT_PIE_univ_favorites_edit"
        km.new("wm.call_menu_pie", 'T').prop.name = "IMAGE_MT_PIE_univ_transform"
        km.new("wm.call_menu_pie", 'A', shift=True).prop.name = "IMAGE_MT_PIE_univ_texel"

        # Select
        km.new('uv.univ_select_linked', 'WHEELUPMOUSE', ctrl=True, shift=True).prop.deselect = False
        km.new('uv.univ_select_linked', 'WHEELDOWNMOUSE', ctrl=True, shift=True).prop.deselect = True
        km.new('uv.univ_select_pick', 'WHEELUPMOUSE', shift=True).prop.select = True
        km.new('uv.univ_select_pick', 'WHEELDOWNMOUSE', shift=True).prop.select = False
        km.new('uv.univ_select_grow', 'WHEELUPMOUSE', ctrl=True).prop.grow = True
        km.new('uv.univ_select_grow', 'WHEELDOWNMOUSE', ctrl=True).prop.grow = False

        # Edge Grow (Conflict)
        km.new('uv.univ_select_edge_grow', 'WHEELUPMOUSE', ctrl=True, alt=True).prop.grow = True
        km.new('uv.univ_select_edge_grow', 'WHEELDOWNMOUSE', ctrl=True, alt=True).prop.grow = False

        if univ_pro_exist:
            km.new('uv.univ_select_loop', 'WHEELUPMOUSE', alt=True)
            km.new('uv.univ_select_similar', 'G', shift=True)

        # Select Mode.
        km.new('uv.univ_select_mode', 'ONE').prop.type = 'VERTEX'
        km.new('uv.univ_select_mode', 'TWO').prop.type = 'EDGE'
        km.new('uv.univ_select_mode', 'THREE').prop.type = 'FACE'
        km.new('uv.univ_select_mode', 'FOUR').prop.type = 'ISLAND'

        km.new("uv.univ_local_invert_selection", "I", ctrl=True, shift=True)

        # Transform.
        km.new('uv.univ_orient', 'O').prop.edge_dir = 'BOTH'
        km.new('uv.univ_flip', 'F')
        km.new('uv.univ_home', 'G', alt=True)

        kmi = km.new('uv.univ_rotate', 'FIVE')
        kmi.prop.rot_dir = 'CW'
        kmi.prop.mode = 'DEFAULT'

        kmi = km.new('uv.univ_rotate', 'FIVE', alt=True)
        kmi.prop.rot_dir = 'CCW'
        kmi.prop.mode = 'DEFAULT'

        kmi = km.new('uv.univ_rotate', 'FIVE', shift=True)
        kmi.prop.rot_dir = 'CW'
        kmi.prop.mode = 'INDIVIDUAL'

        kmi = km.new('uv.univ_rotate', 'FIVE', shift=True, alt=True)
        kmi.prop.rot_dir = 'CCW'
        kmi.prop.mode = 'INDIVIDUAL'

        # Unfold
        km.new('uv.univ_quadrify', 'E')
        km.new('uv.univ_straight', 'E', shift=True)
        km.new('uv.univ_relax', 'R', alt=True)
        km.new('uv.univ_unwrap', 'U')
        if univ_pro_exist:
            kmi.prop.unwrap_along = 'UV'

        # Misc
        km.new('uv.univ_weld', 'W').prop.use_by_distance = False
        km.new('uv.univ_stitch', 'W', shift=True)
        km.new('uv.univ_stack', 'S', alt=True)
        km.new('uv.univ_symmetrize', 'X', alt=True)

        # Quick Snap
        km.new('uv.univ_quick_snap', 'V').prop.quick_start = True
        km.new('uv.univ_quick_snap', 'V', alt=True).prop.quick_start = False
        if univ_pro_exist:
            # Drag
            km.new('uv.univ_drag', 'LEFTMOUSE', 'ANY', alt=True)
            # Isolate
            km.new('uv.univ_isolate', 'NUMPAD_SLASH')
            km.new('uv.univ_isolate', 'SLASH')

        # Mark
        km.new('uv.univ_cut', 'C').prop.addition = False
        km.new('uv.univ_cut', 'C', shift=True).prop.addition = True
        km.new('uv.univ_pin', 'P')

        # Stretch Toggle
        km.new('uv.univ_stretch_uv_toggle', 'Z', 'DOUBLE_CLICK').prop.swap = True
        km.new('uv.univ_stretch_uv_toggle', 'Z', 'CLICK').prop.swap = False
        km.new('uv.univ_show_modified_uv_edges_toggle', 'Z', alt=True)

        # Other Misc.
        km.new('uv.univ_hide', 'H').prop.unselected = False
        km.new('uv.univ_hide', 'H', shift=True).prop.unselected = True
        km.new('uv.univ_set_cursor_2d', 'MIDDLEMOUSE', ctrl=True, shift=True)
        km.new('uv.univ_focus', 'NUMPAD_PERIOD')

        global keys
        keys.clear()
        keys = km.items

        for _, kmi in keys:
            kmi.active = False

    @staticmethod
    def _add_mesh_keymaps(km):
        # Grow
        km.new('mesh.univ_select_grow', 'WHEELUPMOUSE', ctrl=True).prop.grow = True
        km.new('mesh.univ_select_grow', 'WHEELDOWNMOUSE', ctrl=True).prop.grow = False
        # Edge grow
        km.new('mesh.univ_select_edge_grow', 'WHEELUPMOUSE', ctrl=True, alt=True).prop.grow = True
        km.new('mesh.univ_select_edge_grow', 'WHEELDOWNMOUSE', ctrl=True, alt=True).prop.grow = False

        if univ_pro_exist:
            # Select loop
            km.new('mesh.univ_select_loop', 'WHEELUPMOUSE', alt=True)
            km.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK')
            km.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK', shift=True)

    @staticmethod
    def remove_keymaps():
        global keys
        import traceback

        for km, kmi in reversed(keys):
            try:
                km.keymap_items.remove(kmi)
            except (RuntimeError, UnicodeDecodeError):
                traceback.print_exc()
        keys.clear()

class WSKeymapGenerator:
    class WSKeymapPropertyController:
        def __init__(self, ws_keymap):
            object.__setattr__(self, "_ZGenerator", ws_keymap)

        def __setattr__(self, key, value):
            if key.startswith("_ZGenerator"):
                object.__setattr__(self, key, value)
                return

            last_kmi = self._ZGenerator.items[-1]  # noqa
            misc_attr: dict = last_kmi[2]
            misc_attr.setdefault("properties", []).append((key, value))

    def __init__(self):
        self.items = []
        self.prop = self.WSKeymapPropertyController(self)

    def new(self, idname: str, type: str, value: str = "PRESS", **kw):  # noqa
        expected = {"any", "shift", "ctrl", "alt", "oskey" "key_modifier", "direction", "repeat", "head"}
        for key in kw:
            if key not in expected:
                raise ValueError(f"Expected {expected!r} keywords, given {key!r}")

        self.items.append([idname, {"type": type, "value": value, **kw}, dict()])
        return self

    def to_tuple_ws(self):
        return tuple(tuple(kmi) for kmi in self.items)

    @classmethod
    def add_keymaps_ws_edit(cls):
        try:
            from . import univ_pro
        except ImportError:
            univ_pro = None

        # Edit Mode
        km = WSKeymapGenerator()

        ## Rotate
        kmi = km.new('mesh.univ_rotate', 'FIVE')
        kmi.prop.rot_dir = 'CW'
        kmi.prop.mode = 'DEFAULT'

        # Default. CW. Individual.
        kmi = km.new('mesh.univ_rotate', 'FIVE', shift=True)
        kmi.prop.rot_dir = 'CW'
        kmi.prop.mode = 'INDIVIDUAL'

        # kmi = km.new('uv.univ_flip', 'F')
        # keys_ws.append((km, kmi))

        km.new("wm.call_menu_pie", 'D').prop.name = "VIEW3D_MT_PIE_univ_misc"
        km.new("wm.call_menu_pie", 'Q').prop.name = "VIEW3D_MT_PIE_univ_favorites_edit"

        km.new("mesh.univ_cut", 'C').prop.addition = False
        km.new("mesh.univ_cut", 'C', shift=True).prop.addition = True

        # Unfold
        km.new("mesh.univ_relax", 'R', alt=True)
        km.new("mesh.univ_unwrap", 'U')
        # Misc
        km.new("mesh.univ_weld", 'W').prop.use_by_distance = False
        km.new("mesh.univ_stitch", 'W', shift=True)
        km.new("mesh.univ_stack", 'S', alt=True)
        # Mark
        km.new("mesh.univ_seam_border", 'B', alt=True)
        km.new("mesh.univ_angle", 'A', ctrl=True)

        if univ_pro:
            km.new('mesh.univ_select_similar', 'G', shift=True)
            # Select loop
            km.new('mesh.univ_select_loop', 'WHEELUPMOUSE', alt=True)
            km.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK')
            km.new('mesh.univ_select_loop_pick', 'LEFTMOUSE', 'DOUBLE_CLICK', shift=True)

        cls._workspace_duplicates(km)
        return km.to_tuple_ws()


    @classmethod
    def add_keymaps_ws_object(cls):
        # Object Mode
        km = WSKeymapGenerator()
        cls._workspace_duplicates(km)
        return km.to_tuple_ws()

    @staticmethod
    def _workspace_duplicates(km: "WSKeymapGenerator"):
        km.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG')
        km.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', shift=True).prop.mode = 'ADD'
        km.new("view3d.select_box", 'LEFTMOUSE', 'CLICK_DRAG', ctrl=True).prop.mode = 'SUB'

        km.new("mesh.univ_gravity", 'O')
        km.new("wm.call_menu_pie", 'A', shift=True).prop.name = "VIEW3D_MT_PIE_univ_texel"
        km.new("wm.call_menu_pie", 'Q', shift=True).prop.name = "VIEW3D_MT_PIE_univ_projection"



_EVENT_TYPES = set()
_EVENT_TYPE_MAP = {}
_EVENT_TYPE_MAP_EXTRA = {}


class KeymapFilter:
    def __init__(self):
        self.univ_keys: "list[bpy.types.KeyMapItem]" = []
        self.conflict_keys = []

    def __str__(self):
        key_name = self.univ_keys[0].to_string()  # noqa
        return f'{key_name: <30}: UniV - {len(self.univ_keys)}, Blender - {len(self.conflict_keys)}'

    @classmethod
    def get_sorted(cls, km, keyconfigs, show_only_error, ws_ignore_kmi=""):
        individual_keys: list[KeymapFilter] = []

        for conf_filter in keyconfigs.values():
            if len(conf_filter.univ_keys) == 1:
                individual_keys.append(conf_filter)
            else:
                for keys_ in conf_filter.univ_keys:
                    new_conf_filter = cls()
                    new_conf_filter.univ_keys = [keys_]
                    new_conf_filter.conflict_keys = conf_filter.conflict_keys.copy()
                    new_conf_filter.conflict_keys.extend((km, keys_sub) for keys_sub in conf_filter.univ_keys if keys_sub is not keys_)
                    individual_keys.append(new_conf_filter)

        if show_only_error:
            only_conflict_keys = []
            for filtered in individual_keys:
                univ_kmi = filtered.univ_keys[0]
                if univ_kmi.active:
                    if univ_kmi.idname != ws_ignore_kmi:
                        if any(kmi.active for kmi in filtered.conflict_keys):
                            only_conflict_keys.append(filtered)
            individual_keys = only_conflict_keys

        individual_keys.sort(key=lambda cf: cf.univ_keys[0].name)
        return individual_keys


    @classmethod
    def get_conflict_filtered_keymaps(cls, keys_areas_, *, use_filter=True, is_ws=False):
        if is_ws:
            def is_univ_keymap_item():
                return True
        else:
            def is_univ_keymap_item():
                return (
                        '.univ_' in kmi.idname or
                        (kmi.idname == 'wm.call_menu_pie' and kmi.name == 'UniV Pie')
                )

        kc = bpy.context.window_manager.keyconfigs.user
        is_unmatch_kmi = cls.is_unmatched_kmi_for_filter_fn(use_filter)

        for area in keys_areas_:
            km = kc.keymaps[area]
            addon_kmi = defaultdict(KeymapFilter)

            # Collect UniV keymap items.
            for kmi in km.keymap_items:
                if not is_univ_keymap_item():
                    continue
                if is_unmatch_kmi(kmi):
                    continue

                keymap_name = kmi.to_string()
                addon_kmi[keymap_name].univ_keys.append(kmi)

            addon_kmi_and_conflict_kmi = addon_kmi

            if addon_kmi_and_conflict_kmi:
                # Check for potential keymap conflicts with the addon in other spaces.
                if area == 'Window':
                    all_areas_with_potential_conflicts = (area, *other_conflict_areas, '3D View')
                else:
                    all_areas_with_potential_conflicts = (area, *other_conflict_areas)


                for area_with_potential_conflicts in all_areas_with_potential_conflicts:
                    km_with_potential_conflicts = kc.keymaps[area_with_potential_conflicts]

                    for kmi in km_with_potential_conflicts.keymap_items:
                        keymap_name = kmi.to_string()

                        if keymap_name in addon_kmi_and_conflict_kmi:
                            if is_univ_keymap_item():
                                continue

                            if is_unmatch_kmi(kmi):
                                continue

                            addon_kmi_and_conflict_kmi[keymap_name].conflict_keys.append((km_with_potential_conflicts, kmi))

            yield area, kc, km, addon_kmi_and_conflict_kmi

    @classmethod
    def is_unmatched_kmi_for_filter_fn(cls, use_filter):

        def catcher(name_filter, key_filter, filter_key_fn):
            if name_filter and key_filter:
                def filtered(kmi):
                    if (name_filter in kmi.idname.lower() or name_filter in kmi.name.lower()) and filter_key_fn(kmi):
                        return False
                    return True

            elif name_filter:
                def filtered(kmi):
                    if name_filter in kmi.idname.lower() or name_filter in kmi.name.lower():
                        return False
                    return True
            elif key_filter:
                def filtered(kmi):
                    if filter_key_fn(kmi):
                        return False
                    return True
            else:
                def filtered(_):
                    return False
            return filtered


        km_name_filter = ''
        km_key_filter = ''

        if use_filter:
            from .preferences import prefs
            pref = prefs()
            km_name_filter = pref.km_name_filter.strip().lower()
            km_key_filter = pref.km_key_filter.strip().lower()

        if km_key_filter:
            filter_key_fn_ = cls.filter_by_key(km_key_filter)
        else:
            def filter_key_fn_(a): return a  # pycharm warning

        return catcher(km_name_filter, km_key_filter, filter_key_fn_)

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
                     'Restore - Resets properties and assigned keys, for enabled keymaps (doesn`t restore deleted and user defined keymaps)\n' \
                     'Off/On - Enable/disable keymaps\n' \
                     'Delete User - Remove manually installed UniV keymaps\n' \
                     'Resolve Conflicts - Resolve all conflicts with UniV keymaps (except in cases where the UniV keymap is disabled)'

    # noinspection PyTypeHints
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
            for _area in keys_areas:
                _km = kc.keymaps[_area]
                for _kmi in _km.keymap_items:
                    if '.univ_' in _kmi.idname:
                        yield _km, _kmi
                    elif 'wm.call_menu_pie' == _kmi.idname and _kmi.name == 'UniV Pie':
                        yield _km, _kmi

            for _area in keys_areas_workspace:
                _km = kc.keymaps[_area]
                for _kmi in _km.keymap_items:
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

            for area, kc, km, filtered_keymaps in KeymapFilter.get_conflict_filtered_keymaps(
                    keys_areas + keys_areas_workspace, use_filter=False):
                for config_filtered in filtered_keymaps.values():
                    if not any(univ_kmi.active for univ_kmi in config_filtered.univ_keys):
                        continue
                    for (_, kmi_) in config_filtered.conflict_keys:
                        if kmi_.active:
                            counter += 1
                            kmi_.active = False
            message = f'Disabled {counter} keymaps' if counter else 'Not found keymaps with conflicts'

        elif self.mode == 'RESTORE':
            for km, kmi in keys:
                if (not kmi.is_user_defined) and kmi.is_user_modified:
                    if kmi.active:
                        km.restore_item_to_default(kmi)
                        counter += 1
            message = f'Restored {counter} keymaps' if counter else 'Not found enabled keymaps for restore.'

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

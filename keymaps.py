# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from collections import defaultdict

keys = []
keys_areas = ['UV Editor', 'Window', 'Object Mode', 'Mesh']
other_conflict_areas = ['Frames']


def add_keymaps():
    global keys

    if not (kc := bpy.context.window_manager.keyconfigs.addon):
        from .preferences import debug
        if debug():
            print('UniV: Failed to add keymaps. Result = ', kc)
        return

    ### Object Mode
    km = kc.keymaps.new(name='Object Mode')
    kmi = km.keymap_items.new('object.univ_join', 'J', 'PRESS', ctrl=True)
    keys.append((km, kmi))

    ### Mesh
    km = kc.keymaps.new(name='Mesh')
    kmi = km.keymap_items.new('mesh.univ_select_edge_grow', 'WHEELUPMOUSE', 'PRESS', alt=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('mesh.univ_select_edge_grow', 'WHEELDOWNMOUSE', 'PRESS', alt=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    ### Window
    km = kc.keymaps.new(name='Window')
    kmi = km.keymap_items.new('wm.univ_split_uv_toggle', 'T', 'PRESS', shift=True)
    kmi.properties.mode = 'SPLIT'
    keys.append((km, kmi))

    ### UV Editor
    km = kc.keymaps.new(name='UV Editor')

    kmi = km.keymap_items.new('uv.univ_sync_uv_toggle', 'ACCENT_GRAVE', 'PRESS')
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
    kmi = km.keymap_items.new('uv.univ_select_edge_grow', 'WHEELUPMOUSE', 'PRESS', alt=True)
    kmi.properties.grow = True
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_edge_grow', 'WHEELDOWNMOUSE', 'PRESS', alt=True)
    kmi.properties.grow = False
    keys.append((km, kmi))

    # Rotate
    ## Default. CW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'R', 'DOUBLE_CLICK')  # Work if not selection.
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'DEFAULT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS')
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'DEFAULT'
    keys.append((km, kmi))

    ## Default. CCW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'DEFAULT'
    keys.append((km, kmi))

    ## Default. CW. By Cursor.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', ctrl=True)
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'BY_CURSOR'
    keys.append((km, kmi))

    ## Default. CCW. By Cursor.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'BY_CURSOR'
    keys.append((km, kmi))

    ## Default. CW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True)
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'INDIVIDUAL'
    keys.append((km, kmi))

    ## Default. CCW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True, alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'INDIVIDUAL'
    keys.append((km, kmi))

    # Normalize
    kmi = km.keymap_items.new('uv.univ_normalize', 'A', 'PRESS', shift=True)
    keys.append((km, kmi))

    # Adjust
    kmi = km.keymap_items.new('uv.univ_adjust_td', 'A', 'PRESS', alt=True)
    keys.append((km, kmi))

    # Align operator
    ## Align
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    ## Move
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'INDIVIDUAL_OR_MOVE'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'INDIVIDUAL_OR_MOVE'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'INDIVIDUAL_OR_MOVE'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'INDIVIDUAL_OR_MOVE'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    ## Align Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    ## Align to Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    ## Align to Cursor Union
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    ## Cursor to Tile
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'LEFT'
    keys.append((km, kmi))

    # Relax
    kmi = km.keymap_items.new('uv.univ_relax', 'R', 'PRESS', alt=True)
    keys.append((km, kmi))

    # Unwrap
    kmi = km.keymap_items.new('uv.univ_unwrap', 'U', 'PRESS')
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

    # Orient
    kmi = km.keymap_items.new('uv.univ_orient', 'O', 'PRESS')
    kmi.properties.edge_dir = 'BOTH'
    keys.append((km, kmi))

    for _, kmi in keys:
        kmi.active = False

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


class ConflictFilter:
    def __init__(self):
        self.univ_keys = []
        self.default_keys = []
        self.user_defined = []

    def __str__(self):
        key_name = self.univ_keys[0].to_string()
        return f'{key_name: <30}: UniV - {len(self.univ_keys)}, Blender - {len(self.default_keys)}, User - {len(self.user_defined)}'

    @staticmethod
    def get_conflict_filtered_keymaps():
        kc = bpy.context.window_manager.keyconfigs.user

        for area in keys_areas:
            km = kc.keymaps[area]

            conflict_filter = defaultdict(ConflictFilter)
            for kmi in km.keymap_items:
                if '.univ_' in kmi.idname:
                    keymap_name = kmi.to_string()
                    conflict_filter[keymap_name].univ_keys.append(kmi)

            if area == 'Window':
                areas_ = (area, *other_conflict_areas, '3D View')
            else:
                areas_ = (area, *other_conflict_areas)
            for area1 in areas_:
                km = kc.keymaps[area1]
                for kmi in km.keymap_items:
                    if ((keymap_name := kmi.to_string()) in conflict_filter) and ('.univ_' not in kmi.idname):
                        if kmi.is_user_defined:
                            conflict_filter[keymap_name].user_defined.append((km, kmi))
                        else:
                            conflict_filter[keymap_name].default_keys.append((km, kmi))
            yield area, kc, km, conflict_filter

class UNIV_RestoreKeymaps(bpy.types.Operator):
    bl_idname = 'wm.univ_keymaps_config'
    bl_label = 'Keymaps Config'
    bl_description = 'Keymaps Config\n\n' \
                     'Default - Resets properties and assigned keys, enable keymaps (doesn`t restore deleted keymaps)\n' \
                     'Off/On - Enable/disable keymaps\n' \
                     'Delete User - Remove manually installed UniV keymaps\n' \
                     'Resolve Conflicts - Resolve all conflicts with UniV keymaps (except in cases where the UniV keymap is disabled)'

    mode: bpy.props.EnumProperty(name='Mode', default='DEFAULT',
                                 items=(
                                     ('DEFAULT', 'Default', ''),
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

        if self.mode == 'DEFAULT':
            for km, kmi in keymap_items():
                if not kmi.is_user_defined:
                    activ_before = kmi.active
                    to_str_before = kmi.to_string()
                    properties_before = [getattr(kmi.properties, str_props) for str_props in dir(kmi.properties) if not str_props.startswith('__')]

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

            for area, kc, km, filtered_keymaps in ConflictFilter.get_conflict_filtered_keymaps():
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

        self.report({'INFO'}, message)
        return {'FINISHED'}

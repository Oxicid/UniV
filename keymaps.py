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

keys = []
keys_areas = ['Window', 'UV Editor']

def add_keymaps():
    global keys

    if not (kc := bpy.context.window_manager.keyconfigs.addon):
        from .preferences import debug
        if debug():
            print('Failed to add keymaps. Result = ', kc)
        return

    km = kc.keymaps.new(name='Window')
    kmi = km.keymap_items.new('wm.univ_split_uv_toggle', 'T', 'PRESS', shift=True)
    kmi.properties.mode = 'SPLIT'
    kmi.active = False
    keys.append((km, kmi))
    #
    km = kc.keymaps.new(name='UV Editor')

    kmi = km.keymap_items.new('uv.univ_sync_uv_toggle', 'ACCENT_GRAVE', 'PRESS')
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELUPMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = False
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = True
    kmi.active = False
    keys.append((km, kmi))

    # Rotate
    ## Default. CW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'R', 'DOUBLE_CLICK')  # Work if not selection.
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'DEFAULT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS')
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'DEFAULT'
    kmi.active = False
    keys.append((km, kmi))

    ## Default. CCW.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'DEFAULT'
    kmi.active = False
    keys.append((km, kmi))

    ## Default. CW. By Cursor.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', ctrl=True)
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'BY_CURSOR'
    kmi.active = False
    keys.append((km, kmi))

    ## Default. CCW. By Cursor.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', ctrl=True, alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'BY_CURSOR'
    kmi.active = False
    keys.append((km, kmi))

    ## Default. CW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True)
    kmi.properties.rot_dir = 'CW'
    kmi.properties.mode = 'INDIVIDUAL'
    kmi.active = False
    keys.append((km, kmi))

    ## Default. CCW. Individual.
    kmi = km.keymap_items.new('uv.univ_rotate', 'FIVE', 'PRESS', shift=True, alt=True)
    kmi.properties.rot_dir = 'CCW'
    kmi.properties.mode = 'INDIVIDUAL'
    kmi.active = False
    keys.append((km, kmi))

    # Align operator
    ## Align
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS')
    kmi.properties.mode = 'ALIGN'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Move
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Align Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', alt=True)
    kmi.properties.mode = 'ALIGN_CURSOR'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Align to Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Align to Cursor Union
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True, shift=True, alt=True)
    kmi.properties.mode = 'ALIGN_TO_CURSOR_UNION'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Move Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    ## Cursor to Tile
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'UPPER'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'BOTTOM'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'RIGHT'
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', ctrl=True, alt=True)
    kmi.properties.mode = 'CURSOR_TO_TILE'
    kmi.properties.direction = 'LEFT'
    kmi.active = False
    keys.append((km, kmi))

    # Quad
    kmi = km.keymap_items.new('uv.univ_quad', 'E', 'PRESS')
    kmi.active = False
    keys.append((km, kmi))

    # Straight
    kmi = km.keymap_items.new('uv.univ_straight', 'E', 'PRESS', shift=True)
    kmi.active = False
    keys.append((km, kmi))

    # Weld
    kmi = km.keymap_items.new('uv.univ_weld', 'W', 'PRESS')
    kmi.active = False
    keys.append((km, kmi))

    # Stitch
    kmi = km.keymap_items.new('uv.univ_stitch', 'W', 'PRESS', shift=True)
    kmi.active = False
    keys.append((km, kmi))

    # Quick Snap
    kmi = km.keymap_items.new('uv.univ_quick_snap', 'V', 'PRESS')
    kmi.properties.quick_start = True
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_quick_snap', 'V', 'PRESS', alt=True)
    kmi.properties.quick_start = False
    kmi.active = False
    keys.append((km, kmi))

    # Cut
    kmi = km.keymap_items.new('uv.univ_cut', 'C', 'PRESS')
    kmi.properties.addition = False
    kmi.active = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_cut', 'C', 'PRESS', shift=True)
    kmi.properties.addition = True
    kmi.active = False
    keys.append((km, kmi))

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


class UNIV_RestoreKeymaps(bpy.types.Operator):
    bl_idname = 'wm.univ_keymaps_config'
    bl_label = 'Keymaps Config'

    mode: bpy.props.EnumProperty(name='Mode',
                       default='RESTORE',
                       items=(('RESTORE', 'Restore', 'Resets existing keymaps'),
                              ('DEFAULT', 'Default', 'Resets everything'),
                              ('TOGGLE', 'Off/On', ''),
                              ('DELETE_USER', 'Delete User', 'Remove manually installed keymaps'),
                              )
                       )

    def execute(self, context):
        kc = context.window_manager.keyconfigs.user

        def keymap_items():
            for _area in keys_areas:
                _km = kc.keymaps[_area]
                for _kmi in _km.keymap_items:
                    if '.univ_' in _kmi.idname:
                        yield _km, _kmi

        if self.mode == 'RESTORE':
            for km, kmi in keymap_items():
                if not kmi.is_user_defined:
                    km.restore_item_to_default(kmi)
        elif self.mode == 'DEFAULT':
            pass
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
                    km.keymap_items.remove(kmi)
        else:
            active_states = set()
            for _, kmi in keymap_items():
                active_states.add(kmi.active)

            state = False if (len(active_states) == 2) else (False in active_states)

            for _, kmi in keymap_items():
                kmi.active = state
        return {'FINISHED'}

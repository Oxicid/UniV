import bpy

keys = []

def add_keymaps():
    global keys

    if not (kc := bpy.context.window_manager.keyconfigs.addon):
        return

    km = kc.keymaps.new(name='Window', space_type='EMPTY')
    kmi = km.keymap_items.new('wm.split_uv_toggle', 'T', 'PRESS', shift=True)
    kmi.properties.mode = 'SPLIT'
    keys.append((km, kmi))

    km = kc.keymaps.new(name='UV Editor')
    kmi = km.keymap_items.new('uv.sync_uv_toggle', 'ACCENT_GRAVE', 'PRESS')
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELUPMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = False
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_select_linked', 'WHEELDOWNMOUSE', 'PRESS', ctrl=True, shift=True)
    kmi.properties.deselect = True
    keys.append((km, kmi))

    # Rotate
    ## Default. CW.
    ## DOUBLE_CLICK + R doesn't work, I'll have to set it to 'FIVE'.
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
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', shift=True)
    kmi.properties.mode = 'MOVE'
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

    ## Move Cursor
    kmi = km.keymap_items.new('uv.univ_align', 'UP_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'UPPER'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'DOWN_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'BOTTOM'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'RIGHT_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
    kmi.properties.direction = 'RIGHT'
    keys.append((km, kmi))

    kmi = km.keymap_items.new('uv.univ_align', 'LEFT_ARROW', 'PRESS', shift=True, alt=True)
    kmi.properties.mode = 'MOVE_CURSOR'
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


def remove_keymaps():
    global keys
    import contextlib

    for km, kmi in keys:
        with contextlib.suppress(RuntimeError):
            km.keymap_items.remove(kmi)
    keys.clear()


class UNIV_RestoreKeymaps(bpy.types.Operator):
    bl_idname = 'wm.univ_restore_keymaps'
    bl_label = 'Restore'

    def execute(self, context):
        global keys
        for km, kmi in keys:
            km.restore_item_to_default(kmi)
        remove_keymaps()
        add_keymaps()
        return {'FINISHED'}

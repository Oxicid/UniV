import bpy

keys = []

def add_keymaps():
    global keys

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon

    if kc:
        km = kc.keymaps.new(name='Window', space_type='EMPTY')
        kmi = km.keymap_items.new('wm.split_uv_toggle', 'T', 'PRESS', shift=True)
        kmi.properties.mode = 'SPLIT'
        keys.append((km, kmi))

        km = kc.keymaps.new(name='UV Editor')
        kmi = km.keymap_items.new('uv.sync_uv_toggle', 'ACCENT_GRAVE', 'PRESS', repeat=True)
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
        for km, _ in keys:
            km.restore_to_default()
        remove_keymaps()
        add_keymaps()
        return {'FINISHED'}

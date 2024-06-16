import bpy
import rna_keymap_ui

from . import keymaps
from bpy.props import *

def prefs():
    return bpy.context.preferences.addons[__package__].preferences

def get_keymap_entry_item(km, kmi):
    for i, km_item in enumerate(km.keymap_items):
        if km.keymap_items.keys()[i] == kmi.idname:
            return km_item
    return None

class UNIV_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    tab: EnumProperty(
        items=[
            ('KEYMAPS', 'Keymaps', ''),
        ],
        default='KEYMAPS')

            # ('GENERAL', 'General', ''),  # noqa
            # ('UI', 'UI', ''),  # noqa

    def draw(self, context):
        layout = self.layout

        # row = layout.row()
        # row.prop(self, "tab", expand=True)

        if self.tab == 'KEYMAPS':
            layout.operator('wm.univ_restore_keymaps', text='Restore')
            box = layout.box()
            split = box.split()
            col = split.column()
            kc = context.window_manager.keyconfigs.user

            for km, kmi in keymaps.keys:
                km = kc.keymaps[km.name]
                col.context_pointer_set("keymap", km)
                if _kmi := get_keymap_entry_item(km, kmi):
                    rna_keymap_ui.draw_kmi([], kc, km, get_keymap_entry_item(km, kmi), col, 0)

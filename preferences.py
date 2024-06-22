import bpy
import rna_keymap_ui

from . import keymaps
from bpy.props import *

def prefs():
    return bpy.context.preferences.addons[__package__].preferences

def force_debug():
    return prefs().debug == 'FORCE'

def debug():
    return prefs().debug == 'ENABLED'

class UNIV_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    tab: EnumProperty(
        items=(
            ('GENERAL', 'General', ''),
            ('KEYMAPS', 'Keymaps', ''),
        ),
        default='KEYMAPS')

            # ('UI', 'UI', ''),  # noqa

    debug: EnumProperty(name='Debug',
        items=(
            ('DISABLED', 'Disabled', ''),
            ('ENABLED', 'Enabled', ''),
            ('FORCE', 'Force', ''),
        ),
        default='DISABLED')

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "tab", expand=True)

        if self.tab == 'GENERAL':
            layout.prop(self, "debug", emboss=True)

        if self.tab == 'KEYMAPS':
            layout.operator('wm.univ_restore_keymaps', text='Restore')
            box = layout.box()
            split = box.split()
            col = split.column()

            kc = context.window_manager.keyconfigs.addon

            for km, kmi in keymaps.keys:
                km = km.active()
                col.context_pointer_set("keymap", km)
                rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)

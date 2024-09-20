# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

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

def stable():
    return prefs().mode == 'STABLE'

def experimental():
    return prefs().mode == 'EXPERIMENTAL'


class UNIV_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    tab: EnumProperty(
        items=(
            ('GENERAL', 'General', ''),
            ('KEYMAPS', 'Keymaps', ''),
            ('INFO', 'Info', ''),
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

    mode: EnumProperty(name='Mode',
        items=(
            ('STABLE', 'Stable', ''),
            ('EXTENDED', 'Extended', ''),
            ('EXPERIMENTAL', 'Experimental', ''),
        ),
        default='EXTENDED')

    snap_points_default: EnumProperty(name='Default Snap Points',
        items=(
            ('ALL', 'All', ''),
            ('FOLLOW_MODE', 'Follow Mode', 'Follow the selection mode, VERTEX mode remains always')
        ),
        default='FOLLOW_MODE',
        description='Default Snap Points for QuickSnap')

    show_split_toggle_uv_button: BoolProperty(name='Show Split ToggleUV Button', default=False)

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "tab", expand=True)

        if self.tab == 'GENERAL':
            layout.prop(self, 'debug')
            layout.prop(self, 'mode')
            layout.separator()
            layout.label(text='QuickSnap:')
            layout.prop(self, 'snap_points_default')
            layout.separator()
            layout.prop(self, 'show_split_toggle_uv_button')

        elif self.tab == 'KEYMAPS':
            row = layout.row()
            row.operator('wm.univ_keymaps_config', text='Restore').mode = 'RESTORE'
            # row.operator('wm.univ_keymaps_config', text='Default').mode = 'DEFAULT'
            row.operator('wm.univ_keymaps_config', text='Off/On').mode = 'TOGGLE'
            row.operator('wm.univ_keymaps_config', text='Delete User').mode = 'DELETE_USER'

            layout.label(
                text='To restore deleted keymaps, just reload the addon. But it is better to use the checkboxes to disable them',
                icon='INFO')
            box = layout.box()
            split = box.split()
            col = split.column()

            kc = context.window_manager.keyconfigs.user

            for area in keymaps.keys_areas:
                km = kc.keymaps[area]
                for kmi in km.keymap_items:
                    if '.univ_' in kmi.idname and kmi.idname != 'uv.univ_align':
                        col.context_pointer_set("keymap", km)
                        rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)

            col.separator()
            col.label(text='Keymap at the bottom may have conflicts')
            col.separator()

            km = kc.keymaps['UV Editor']
            for kmi in km.keymap_items:
                if '.univ_' in kmi.idname and kmi.idname == 'uv.univ_align':
                    col.context_pointer_set("keymap", km)
                    rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)

        # elif self.tab == 'INFO':
        else:
            enable = True
            if hasattr(bpy.app, 'online_access'):
                enable = bpy.app.online_access
            row = layout.row(align=True)
            row.enabled = enable
            row.operator("wm.url_open", text="YouTube").url = r"https://www.youtube.com/@oxicid6058"
            row.operator("wm.url_open", text="Discord").url = r"https://discord.gg/SAvEbGTkjR"
            row.operator("wm.url_open", text="GitHub").url = r"https://github.com/Oxicid/UniV"
            row.operator("wm.url_open", text="Blender Market").url = r"https://blendermarket.com/products/univ?search_id=32308413"

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

    snap_points_default: EnumProperty(name='Default Snap Points',
        items=(
            ('ALL', 'All', ''),
            ('FOLLOW_MODE', 'Follow Mode', 'Follow the selection mode, VERTEX mode remains always')
        ),
        default='FOLLOW_MODE',
        description='Default Snap Points for QuickSnap')

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(self, "tab", expand=True)

        if self.tab == 'GENERAL':
            layout.prop(self, "debug")
            layout.separator()
            layout.label(text='QuickSnap:')
            layout.prop(self, "quick_snap_points_default")

        if self.tab == 'KEYMAPS':
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
                    if '.univ_' in kmi.idname:
                        col.context_pointer_set("keymap", km)
                        rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)

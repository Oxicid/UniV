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

    show_stretch: BoolProperty(name='Show Stretch', default=False)
    display_stretch_type: EnumProperty(name='Stretch Type',
        items=(
            ('AREA', 'Area', ''),
            ('ANGLE', 'Angle', '')
        ),
        default='AREA')

    max_pick_distance: IntProperty(name='Max Pick Distance', default=40, min=15, soft_max=100, subtype='PIXEL',
                                   description='Pick Distance for Pick Select, Quick Snap operators'
                                   )

    def draw(self, _context):
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

            row = layout.row()
            row.active = self.show_stretch
            row.prop(self, 'show_stretch')
            row.prop(self, 'display_stretch_type', text='')
            row.separator()

            layout.prop(self, 'max_pick_distance')

        elif self.tab == 'KEYMAPS':
            row = layout.row()
            row.operator('wm.univ_keymaps_config', text='Default').mode = 'DEFAULT'
            row.operator('wm.univ_keymaps_config', text='Off/On').mode = 'TOGGLE'
            row.operator('wm.univ_keymaps_config', text='Delete User').mode = 'DELETE_USER'
            row.operator('wm.univ_keymaps_config', text='Resolve Conflicts').mode = 'RESOLVE_ALL'

            layout.label(
                text='To restore deleted keymaps, just reload the addon. But it is better to use the checkboxes to disable them',
                icon='INFO')

            # TODO: Add 3D View
            for area, kc, km, filtered_keymaps in keymaps.ConflictFilter.get_conflict_filtered_keymaps():
                layout.label(text=area)
                for config_filtered in filtered_keymaps.values():
                    box = layout.box()
                    for univ_kmi in config_filtered.univ_keys:
                        rna_keymap_ui.draw_kmi([], kc, km, univ_kmi, box, 0)
                        any_active = any(univ_kmi.active for univ_kmi in config_filtered.univ_keys)

                        if config_filtered.default_keys:
                            box.label(text='\t\tDefault',
                                      icon='ERROR' if any_active and any(kmi_.active for (_, kmi_) in config_filtered.default_keys) else 'NONE')
                            for (default_km, default_kmi) in config_filtered.default_keys:
                                rna_keymap_ui.draw_kmi([], kc, default_km, default_kmi, box, 1)

                        if config_filtered.user_defined:
                            box.label(text='\t\tUser',
                                      icon='ERROR' if any_active and any(kmi_.active for (_, kmi_) in config_filtered.user_defined) else 'NONE')
                            for (user_km, user_kmi) in config_filtered.user_defined:
                                rna_keymap_ui.draw_kmi([], kc, user_km, user_kmi, box, 1)

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

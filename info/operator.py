# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

align_event_info_ex = \
        "Default - Align faces/verts\n" \
        "Shift - Move faces/verts. Mnemonic - Move(Shift) faces/verts\n" \
        "Ctrl - Align to cursor. Mnemonic - Cursor(Ctrl) faces/verts\n" \
        "Ctrl+Shift+Alt - Align to cursor union\n" \
        "Alt - Align to faces/verts\n\n" \
        "Has [Ctrl | Shift | Alt + Arrows] keymaps, but it conflicts with the \'Frame Jump\' operator"
# "Ctrl+Shift+LMB = Collision move (Not Implement)\n"
align_info = "Align verts, edges, faces, islands and cursor \n\n" + align_event_info_ex

crop_event_info_ex = \
        "Default - Crop faces/verts\n" \
        "Shift - Individual Crop faces/verts.\n" \
        "Ctrl - Crop to cursor.\n" \
        "Ctrl+Shift - Crop to cursor individual\n" \
        "Alt - Inplace Crop\n" \
        "Shift+Alt - Individual Inplace Crop"
crop_info = "Crop islands\n\n" + crop_event_info_ex

fill_event_info_ex = \
        "Default - Fill faces/verts\n" \
        "Shift - Individual fill faces/verts.\n" \
        "Ctrl - Fill to cursor.\n" \
        "Ctrl+Shift - Fill to cursor individual\n" \
        "Alt - Inplace Fill\n" \
        "Shift+Alt - Individual Inplace Fill"
fill_info = "Fill islands\n\n" + fill_event_info_ex

distribution_event_info_ex = \
        "Default - Move island to base tile\n" \
        "Ctrl - Move island to cursor.\n"
distribution_info = "Move island to base tile without changes in the textured object\n\n" + distribution_event_info_ex

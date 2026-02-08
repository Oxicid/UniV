# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

crop_event_info_ex = \
    "Default - Fit faces/verts\n" \
    "Shift - Individual Fit faces/verts.\n" \
    "Ctrl - Fit to cursor.\n" \
    "Ctrl+Shift - Fit to cursor individual\n" \
    "Alt - Inplace Fit\n" \
    "Shift+Alt - Individual Inplace Fit"
crop_info = "Fit islands\n\n" + crop_event_info_ex

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

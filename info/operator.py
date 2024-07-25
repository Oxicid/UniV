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

align_event_info_ex = \
        "LMB - Default. Align faces/verts\n" \
        "Shift+LMB - Move faces/verts. Mnemonic - Move(Shift) faces/verts\n" \
        "Ctrl+LMB - Align to cursor. Mnemonic - Cursor(Ctrl) faces/verts\n" \
        "Ctrl+Shift+Alt+LMB - Align to cursor union\n" \
        "Alt+LMB - Align to faces/verts\n" \
        "Ctrl+Alt+LMB - Cursor to Tile. Mnemonic - Current(Ctrl) tile cursor Align(Alt)\n" \
        "Shift+Alt+LMB - Move Cursor. Mnemonic - Move(Shift) Acros(Alt) cursor"
# "Ctrl+Shift+LMB = Collision move (Not Implement)\n"
align_info = "Align verts, edges, faces, islands and cursor \n\n" + align_event_info_ex

crop_event_info_ex = \
        "LMB - Default. Crop faces/verts\n" \
        "Shift+LMB - Individual Crop faces/verts.\n" \
        "Ctrl+LMB - Crop to cursor.\n" \
        "Ctrl+Shift+LMB - Crop to cursor individual\n" \
        "Alt+LMB - Inplace Crop\n" \
        "Shift+Alt+LMB - Individual Inplace Crop"
crop_info = "Crop islands\n\n" + crop_event_info_ex

fill_event_info_ex = \
        "LMB - Default. Fill faces/verts\n" \
        "Shift+LMB - Individual fill faces/verts.\n" \
        "Ctrl+LMB - Fill to cursor.\n" \
        "Ctrl+Shift+LMB - Fill to cursor individual\n" \
        "Alt+LMB - Inplace Fill\n" \
        "Shift+Alt+LMB - Individual Inplace Fill"
fill_info = "Fill islands\n\n" + fill_event_info_ex


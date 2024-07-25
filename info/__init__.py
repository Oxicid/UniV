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

# Addon created 23 April 2024

from . import operator

def event_to_string(event, text=''):
    if event.ctrl:
        text += 'Ctrl + '
    if event.shift:
        text += 'Shift + '
    if event.alt:
        text += 'Alt + '
    return f'{text} Left Mouse '

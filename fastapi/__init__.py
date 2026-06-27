# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from . import clib # TODO: Determine why this import is used here before replacing it with the proper one.
    from .. import reload
    reload.reload(globals())


import bpy  # noqa: F401
from .clib import *

# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

_needs_reload = "bpy" in locals()


import bpy  # noqa: F401
from .clib import *

if _needs_reload:
    from . import clib
    from .. import reload
    reload.reload(globals())
    del clib
    del reload
# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from . import clib
    from .. import reload
    reload.reload(globals())


import bpy  # noqa: F401
from . import clib

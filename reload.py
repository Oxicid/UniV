# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import inspect
import importlib


def reload(globals_: dict[str, object]):
    """The reload function simplifies addon development by allowing
    the bpy.ops.preferences.addon_disable(addon_enable) operator
    to quickly restart the addon, which simplifies addon development.

    WARNING! Keep in mind that garbage from renamed or deleted classes,
    functions, and variables remains and can potentially cause
    unexpected results, so you should restart blender in strange situations"""
    for name, module in globals_.copy().items():
        if inspect.ismodule(module):
            module_path = getattr(module, '__file__', None)
            if module_path and __package__ in module_path and 'reload' not in module_path:
                globals_[name] = importlib.reload(module)

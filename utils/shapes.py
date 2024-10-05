# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

def radial_patterns():
    from math import sin, cos, pi
    points = (6,)
    bases = [(r, n) for r, n in enumerate(points, 6)]

    _patterns = []
    for r, n in bases:
        t = ((round(cos(2 * pi / n * x) * r),
              round(sin(2 * pi / n * x) * r)) for x in range(n))
        _patterns.append(tuple(t))
    return tuple(_patterns)

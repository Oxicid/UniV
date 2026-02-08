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

def padding_deltas(pad):
    return (
        (-pad, 0), (pad, 0), (0, -pad), (0, pad),
        (-pad, -pad), (pad, pad), (-pad, pad), (pad, -pad)
     )

def grid_points_px(width, height, step=128, center=False, include_for_clip=False):
    off = 0
    if not center:
        if include_for_clip:
            off = step

    start = step // 2
    if not center:
        start = 0

    for y in range(start, height + off, step):
        for x in range(start, width + off, step):
            yield x, y

def grid_points_ndc(width, height, step=128, center=False, include_for_clip=False):
    """ Return Normalized Device Coordinates. """
    from mathutils import Vector

    off = 0
    if not center:
        if include_for_clip:
            off = step

    start = step // 2
    if not center:
        start = 0

    for y in range(start, height + off, step):
        for x in range(start, width + off, step):
            nx = (x / (width  * 0.5)) - 1.0
            ny = (y / (height * 0.5)) - 1.0
            yield Vector((nx, ny))

arrow = (
    (0.0391, 0.0781),
    (0.1641, 0.0781),
    (0.0, 0.281),
    (-0.1641, 0.0781),
    (-0.0391, 0.0781),
    (-0.0391, -0.296875),
    (0.0391, -0.296875),
)
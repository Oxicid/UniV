# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
from random import seed, random

import numpy as np
from bmesh.types import BMLoop
from mathutils import Vector


def color_for_groups(groups) -> list[list[float]]:
    """Return flat colors by group"""
    colors = []
    for i in range(len(groups)):
        seed(i)
        c0 = random()
        seed(hash(c0))
        c1 = random()
        seed(hash(c1))
        c2 = random()
        colors.append((c0, c1, c2))

    return np.repeat(colors, [len(g)*2 for g in groups], axis=0).tolist()  # noqa

def uv_crn_groups_to_lines_with_offset(groups: list[list[BMLoop]], uv, line_offset=0.008):
    edges = []
    for group in groups:
        for crn in group:
            edges.append(crn[uv].uv.copy())
            edges.append(crn.link_loop_next[uv].uv.copy())

    edge_iter = iter(edges)
    for _ in range(int(len(edges) / 2)):
        start_edge = next(edge_iter)
        end_edge = next(edge_iter)

        nx, ny = (end_edge - start_edge)
        n = Vector((-ny, nx))
        n.normalize()
        n *= line_offset

        start_edge += n
        end_edge += n

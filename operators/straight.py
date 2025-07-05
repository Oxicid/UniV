# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The code was taken and modified from the TexTools addon: https://github.com/Oxicid/TexTools-Blender/blob/master/op_island_straighten_edge_loops.py

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from math import copysign
from mathutils import Vector
from collections import defaultdict
from itertools import chain

from .. import types

class UNIV_OT_Straight(bpy.types.Operator):
    bl_idname = "uv.univ_straight"
    bl_label = "Straight"
    bl_description = "Straighten selected edge-chain and relax the rest of the UV Island"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        tool_settings = context.scene.tool_settings
        if tool_settings.use_uv_select_sync:
            return False
        if tool_settings.uv_select_mode not in ('VERTEX', 'EDGE'):
            return False
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        assert (context.area.ui_type == 'UV')

        umeshes = types.UMeshes(report=self.report)
        umeshes.fix_context()
        for umesh in umeshes.loop():
            main(self, umesh)
        return {'FINISHED'}

class StraightIsland:
    pass

def main(self, umesh):
    uv = umesh.uv
    _islands = types.Islands.calc_visible(umesh)
    if not _islands:
        umesh.update_tag = False
        return
    islands = []
    selected_corns_islands = []
    for i in _islands:
        island_of_corns = []
        for f in i:
            corners = f.loops
            selected_loops = [crn for crn in corners if crn[uv].select]
            if len(selected_loops) == len(corners):
                self.report({'INFO'}, "No face should be selected.")
                return
            island_of_corns.extend(selected_loops)
        if not island_of_corns:
            continue
        islands.append(i)
        selected_corns_islands.append(island_of_corns)

    for isl_ in islands:
        isl_.mark_seam()

    if not islands:
        umesh.update_tag = False
        return

    selected_corns_edge = [crn for isl in selected_corns_islands for crn in isl if crn[uv].select_edge]

    for island, selected_corns in zip(islands, selected_corns_islands):
        open_segment = get_loops_segments(self, umesh.uv, selected_corns)
        if not open_segment:
            continue

        straighten(self, umesh.uv, island, open_segment)

    bpy.ops.uv.select_all(action='DESELECT')

    for crn in selected_corns_edge:
        crn[uv].select_edge = True

    for isl in selected_corns_islands:
        for crn in isl:
            crn[uv].select = True

    for isl in _islands:
        for f in isl:
            f.select = True

    umesh.update_tag = True
    umesh.update(force=True)


def straighten(self, uv, island, segment_loops):
    bpy.ops.uv.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='DESELECT')
    for face in island:
        face.select = True
        for loop in face.loops:
            loop[uv].select = True

    bbox = segment_loops[-1][uv].uv - segment_loops[0][uv].uv
    straighten_in_x = True
    sign = copysign(1, bbox.x)
    if abs(bbox.y) >= abs(bbox.x):
        straighten_in_x = False
        sign = copysign(1, bbox.y)

    origin = segment_loops[0][uv].uv
    edge_lengths = []
    length = 0
    newly_pinned = set()

    for i, loop in enumerate(segment_loops):
        if i > 0:
            vect = loop[uv].uv - segment_loops[i - 1][uv].uv
            edge_lengths.append(vect.length)

    for i, loop in enumerate(segment_loops):
        if i == 0:
            if not loop[uv].pin_uv:
                loop[uv].pin_uv = True
                newly_pinned.add(loop)
        else:
            length += edge_lengths[i - 1]
            for nodeLoop in loop.vert.link_loops:
                if nodeLoop[uv].uv == loop[uv].uv:
                    if straighten_in_x:
                        nodeLoop[uv].uv = origin + Vector((sign * length, 0))
                    else:
                        nodeLoop[uv].uv = origin + Vector((0, sign * length))
                    if not nodeLoop[uv].pin_uv:
                        nodeLoop[uv].pin_uv = True
                        newly_pinned.add(nodeLoop)

    try:  # Unwrapping may fail on certain mesh topologies
        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=True, use_subsurf_data=False, margin=0)
    except:  # noqa
        self.report({'ERROR_INVALID_INPUT'}, "Unwrapping failed, unsupported island topology.")
        pass

    for nodeLoop in newly_pinned:
        nodeLoop[uv].pin_uv = False

def get_loops_segments(self, uv, island_loops_dirty):
    island_loops = set()
    island_loops_nexts = set()  # noqa
    processed_edges = set()
    processed_coords = defaultdict(list)
    start_loops = []
    boundary_splitted_edges = {loop.edge for loop in island_loops_dirty if
                               (not loop.edge.is_boundary) and loop[uv].uv != loop.link_loop_radial_next.link_loop_next[
                                   uv].uv}

    for loop in island_loops_dirty:
        if loop.link_loop_next in island_loops_dirty and (loop.edge in boundary_splitted_edges or loop.edge not in processed_edges):
            island_loops.add(loop)
            island_loops_nexts.add(loop.link_loop_next)
            processed_edges.add(loop.edge)

    if not processed_edges:
        self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: no edges selected.")
        return None

    for loop in chain(island_loops, island_loops_nexts):
        processed_coords[loop[uv].uv.copy().freeze()].append(loop)

    for node_loops in processed_coords.values():
        if len(node_loops) > 2:
            self.report({'ERROR_INVALID_INPUT'}, "No forked edge loops should be selected.")
            return None
        elif len(node_loops) == 1:
            start_loops.extend(node_loops)

    if not start_loops:
        self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: closed UV edge loops.")
        return None
    elif len(start_loops) < 2:
        self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: self-intersecting edge loop.")
        return None
    elif len(start_loops) > 2:
        self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: multiple edge loops.")
        return None

    if len(processed_coords.keys()) < 2:
        self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: zero length edges.")
        return None

    elif len(processed_coords.keys()) == 2:
        single_edge_loops = list(chain.from_iterable(processed_coords.values()))
        if len(single_edge_loops) == 2:
            return single_edge_loops
        else:
            self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: zero length or overlapping edges.")
            return None

    else:

        island_nodal_loops = list(chain.from_iterable(processed_coords.values()))

        if start_loops[0] in island_nodal_loops:
            island_nodal_loops.remove(start_loops[0])
        island_nodal_loops.insert(0, start_loops[0])
        if start_loops[1] in island_nodal_loops:
            island_nodal_loops.remove(start_loops[1])
        island_nodal_loops.append(start_loops[1])

        def find_next_loop(loop):

            def get_prev(found_prev):
                if found_prev:
                    for foundLoop in found_prev:
                        if foundLoop[uv].uv == loop.link_loop_prev[uv].uv:
                            segment.append(foundLoop)
                            for anyLoop in found_prev:
                                if anyLoop[uv].uv == loop.link_loop_prev[uv].uv:
                                    island_nodal_loops.remove(anyLoop)
                            return foundLoop, False
                return None, True

            def get_next(found_next):  # noqa
                for foundLoop in found_next:
                    if foundLoop[uv].uv == loop.link_loop_next[uv].uv:
                        segment.append(foundLoop)
                        for anyLoop in found_next:
                            if anyLoop[uv].uv == loop.link_loop_next[uv].uv:
                                island_nodal_loops.remove(anyLoop)
                        return foundLoop, False
                get_prev(set(island_nodal_loops).intersection(loop.link_loop_prev.vert.link_loops))

            found_next = set(island_nodal_loops).intersection(loop.link_loop_next.vert.link_loops)
            if found_next:
                loopNext, end = get_next(found_next)  # noqa
            else:
                loopNext, end = get_prev(set(island_nodal_loops).intersection(loop.link_loop_prev.vert.link_loops))  # noqa

            if end:
                open_segments.append(segment)

            return loopNext, end

        open_segments = []

        while len(island_nodal_loops) > 0:

            loop = island_nodal_loops[0]
            segment = [loop]
            end = False

            island_nodal_loops.pop(0)
            if loop in island_loops:
                if loop.link_loop_next in island_nodal_loops and loop.link_loop_next not in start_loops:
                    island_nodal_loops.remove(loop.link_loop_next)
            elif loop.link_loop_prev in island_nodal_loops and loop.link_loop_prev not in start_loops:
                island_nodal_loops.remove(loop.link_loop_prev)

            while not end:
                loop, end = find_next_loop(loop)

                if not end:
                    if loop.link_loop_next in island_nodal_loops and loop.link_loop_next not in start_loops:
                        island_nodal_loops.remove(loop.link_loop_next)
                    if loop.link_loop_prev in island_nodal_loops and loop.link_loop_prev not in start_loops:
                        island_nodal_loops.remove(loop.link_loop_prev)

                if not island_nodal_loops:
                    open_segments.append(segment)
                    break

        if len(open_segments) > 1:
            self.report({'ERROR_INVALID_INPUT'}, "Invalid selection in an island: multiple edge loops. Working in the longest one.")
            open_segments.sort(key=len, reverse=True)

    return open_segments[0]

/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/** \file
 * \ingroup bke
 *
 * Functions to evaluate mesh data.
 */

#include "MEM_guardedalloc.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BLI_array_utils.hh"
#include "BLI_index_range.hh"
#include "BLI_math_geom.h"
#include "BLI_span.hh"
#include "BLI_utildefines.h"
#include "BLI_virtual_array.hh"

#include "BKE_attribute.hh"
#include "BKE_mesh.hh"

using blender::float3;
using blender::int2;
using blender::MutableSpan;
using blender::OffsetIndices;
using blender::Span;
using blender::VArray;


void BKE_mesh_mdisp_flip(MDisps *md, const bool use_loop_mdisp_flip) {}
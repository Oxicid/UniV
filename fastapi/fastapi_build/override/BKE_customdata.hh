/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "BLI_sys_types.h"
#include "BLI_vector.hh"
#include "BLI_string_ref.hh"

#include "DNA_customdata_types.h"

namespace blender {

struct BMesh;
struct BMFace;
struct CustomData;
struct CustomData_MeshMasks;

using cd_interp = void (*)(const void **sources, const float *weights, int count, void *dest);
using cd_copy = void (*)(const void *source, void *dest, int count);
using cd_set_default_value = void (*)(void *data, int count);
using cd_free = void (*)(void *data, int count);
using cd_validate = bool (*)(void *item, uint totitems, bool do_fixes);


enum eCDAllocType {
  CD_SET_DEFAULT = 2,
  CD_CONSTRUCT = 5,
};

#define UV_PINNED_NAME "pn"
#define BMUVOFFSETS_NONE {-1, -1}

extern const CustomData_MeshMasks CD_MASK_BAREMESH;
extern const CustomData_MeshMasks CD_MASK_BAREMESH_ORIGINDEX;
extern const CustomData_MeshMasks CD_MASK_MESH;
extern const CustomData_MeshMasks CD_MASK_DERIVEDMESH;
extern const CustomData_MeshMasks CD_MASK_BMESH;
extern const CustomData_MeshMasks CD_MASK_EVERYTHING;

#define ORIGINDEX_NONE -1

struct BMUVOffsets {
  int uv;
  int pin;
};


struct BMCustomDataCopyMap {
  struct TrivialCopy {
    int size;
    int src_offset;
    int dst_offset;
  };
  struct Copy {
    cd_copy fn;
    int src_offset;
    int dst_offset;
  };
  struct TrivialDefault {
    int size;
    int dst_offset;
  };
  struct Default {
    cd_set_default_value fn;
    int dst_offset;
  };
  struct Free {
    cd_free fn;
    int dst_offset;
  };
  Vector<TrivialCopy> trivial_copies;
  Vector<Copy> copies;
  Vector<TrivialDefault> trivial_defaults;
  Vector<Default> defaults;
  Vector<Free> free;
};


} // namespace blender
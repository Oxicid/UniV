/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "BLI_sys_types.h"
#include "BLI_vector.hh"
#include "BLI_string_ref.hh"

#include "DNA_customdata_types.h"


namespace blender {

enum eCDAllocType {
  CD_SET_DEFAULT = 2,
  CD_CONSTRUCT = 5,
};

#define UV_PINNED_NAME "pn"
#define BMUVOFFSETS_NONE {-1, -1}
#define ORIGINDEX_NONE -1

struct BMUVOffsets {
  int uv;
  int pin;
};


struct BMCustomDataCopyMap {};


void CustomData_bmesh_free_block(CustomData *data, void **block);
int CustomData_get_offset(const CustomData *data, const eCustomDataType type);
int CustomData_get_offset_named(const CustomData *data, const eCustomDataType type, const StringRef name);
int CustomData_get_layer_index_n(const CustomData *data, const eCustomDataType type, const int n);
int CustomData_get_active_layer(const CustomData *data, const eCustomDataType type);
bool CustomData_data_equals(const eCustomDataType type, const void *data1, const void *data2);
void CustomData_bmesh_copy_block(CustomData &data, void *src_block, void **dst_block);
void *CustomData_bmesh_get(const CustomData *data, void *block, const eCustomDataType type);
void CustomData_bmesh_set_default(CustomData *data, void **block);

} // namespace blender
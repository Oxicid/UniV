/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "BKE_customdata.hh"

namespace blender {

void CustomData_bmesh_free_block(CustomData *data, void **block) {}

int CustomData_get_offset(const CustomData *data, const eCustomDataType type) {return -1;}

int CustomData_get_layer_index_n(const CustomData *data, const eCustomDataType type, const int n) {return -1;}

int CustomData_get_active_layer(const CustomData *data, const eCustomDataType type) {return -1;}


bool CustomData_data_equals(const eCustomDataType type, const void *data1, const void *data2) {return false;}


void CustomData_bmesh_copy_block(CustomData &data, void *src_block, void **dst_block) {}

void *CustomData_bmesh_get(const CustomData *data, void *block, const eCustomDataType type) {nullptr;}

void CustomData_bmesh_set_default(CustomData *data, void **block) {}

} // namespace blender

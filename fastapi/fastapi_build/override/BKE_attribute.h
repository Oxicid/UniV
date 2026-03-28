/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "BLI_string_ref.hh"
#include "DNA_customdata_types.h"
#include "BKE_customdata.hh"

// using namespace blender;

[[nodiscard]] blender::StringRef BKE_uv_map_pin_name_get(blender::StringRef uv_map_name,
                                                         char *buffer);
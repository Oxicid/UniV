/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "BKE_attribute.h"
#include "BLI_string_ref.hh"
#include "BKE_customdata.hh"

#include <fmt/format.h>

using namespace blender;

StringRef BKE_uv_map_pin_name_get(const StringRef uv_map_name, char *buffer)
{
  BLI_assert(strlen(UV_PINNED_NAME) == 2);
  BLI_assert(uv_map_name.size() < MAX_CUSTOMDATA_LAYER_NAME - 4);
  const auto result = fmt::format_to_n(
      buffer, MAX_CUSTOMDATA_LAYER_NAME, ".{}.{}", UV_PINNED_NAME, uv_map_name);
  return StringRef(buffer, result.size);
}


/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "intern/bmesh_private.hh"

namespace blender {
	
int BMO_slot_buffer_len(BMOpSlot slot_args[BMO_OP_MAX_SLOTS], const char *slot_name);

void *BMO_iter_new(BMOIter *iter,
                   BMOpSlot slot_args[BMO_OP_MAX_SLOTS],
                   const char *slot_name,
                   const char restrictmask);
				   
void *BMO_iter_step(BMOIter *iter);

}  // namespace blender
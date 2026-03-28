/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


#include "bmesh.hh"

#define BMO_OP_MAX_SLOTS 21


int BMO_slot_buffer_len(BMOpSlot slot_args[BMO_OP_MAX_SLOTS], const char *slot_name) {return 0;}

void *BMO_iter_new(BMOIter *iter,
                   BMOpSlot slot_args[BMO_OP_MAX_SLOTS],
                   const char *slot_name,
                   const char restrictmask) {return nullptr;}
				   
void *BMO_iter_step(BMOIter *iter) {return nullptr;}
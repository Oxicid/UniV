/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "BLI_sys_types.h"
#include "BLI_vector.hh"
#include "BLI_string_ref.hh"

#include "DNA_customdata_types.h"

struct BMesh;
struct BMFace;

// using namespace blender;
using cd_interp = void (*)(const void **sources, const float *weights, int count, void *dest);
using cd_copy = void (*)(const void *source, void *dest, int count);
using cd_set_default_value = void (*)(void *data, int count);
using cd_free = void (*)(void *data, int count);
using cd_validate = bool (*)(void *item, uint totitems, bool do_fixes);


/** Add/copy/merge allocation types. */
enum eCDAllocType {
  /** Allocate and set to default, which is usually just zeroed memory. */
  CD_SET_DEFAULT = 2,
  /**
   * Default construct new layer values. Does nothing for trivial types. This should be used
   * if all layer values will be set by the caller after creating the layer.
   */
  CD_CONSTRUCT = 5,
};

#define UV_PINNED_NAME "pn"

/** All values reference none layers. */
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
  blender::Vector<TrivialCopy> trivial_copies;
  blender::Vector<Copy> copies;
  blender::Vector<TrivialDefault> trivial_defaults;
  blender::Vector<Default> defaults;
  blender::Vector<Free> free;
};




CustomData CustomData_shallow_copy_remove_non_bmesh_attributes(const CustomData *src,
                                                               eCustomDataMask mask);

void CustomData_init_layout_from(const CustomData *source,
                                 CustomData *dest,
                                 eCustomDataMask mask,
                                 eCDAllocType alloctype,
                                 int totelem);


void CustomData_bmesh_init_pool(CustomData *data, const int totelem, const char htype);

bool CustomData_bmesh_merge_layout(const CustomData *source,
                                   CustomData *dest,
                                   eCustomDataMask mask,
                                   eCDAllocType alloctype,
                                   BMesh *bm,
                                   const char htype);


void *CustomData_add_layer_named(CustomData *data,
                                 const eCustomDataType type,
                                 const eCDAllocType alloctype,
                                 const int totelem,
                                 const blender::StringRef name);

bool CustomData_merge_layout(const CustomData *source,
                             CustomData *dest,
                             eCustomDataMask mask,
                             eCDAllocType alloctype,
                             int totelem);






int CustomData_get_layer_index(const CustomData *data, eCustomDataType type);
int CustomData_get_layer_index_n(const CustomData *data, eCustomDataType type, int n);
void *CustomData_bmesh_get_n(const CustomData *data,
                             void *block,
                             const eCustomDataType type,
                             const int n);
							 
int CustomData_get_offset_named(const CustomData *data,
                                eCustomDataType type,
                                blender::StringRef name);

int CustomData_get_active_layer(const CustomData *data, eCustomDataType type);

int CustomData_get_named_layer_index(const CustomData *data,
                                     const eCustomDataType type,
                                     const blender::StringRef name);

void CustomData_reset(CustomData *data);
void CustomData_free(CustomData *data);
bool CustomData_bmesh_has_free(const CustomData *data);
void CustomData_bmesh_free_block(CustomData *data, void **block);

bool CustomData_data_equals(const eCustomDataType type, const void *data1, const void *data2);


/** Precalculate a map for more efficient copying between custom data formats. */
BMCustomDataCopyMap CustomData_bmesh_copy_map_calc(const CustomData &src,
                                                   const CustomData &dst,
                                                   eCustomDataMask mask_exclude = 0);
												   
void CustomData_bmesh_copy_block(CustomData &dst_data,
                                 const BMCustomDataCopyMap &copy_map,
                                 const void *src_block,
                                 void **dst_block);
								 
void CustomData_bmesh_copy_block(CustomData &data, void *src_block, void **dst_block);

int CustomData_get_active_layer_index(const CustomData *data, const eCustomDataType type);
int CustomData_get_offset(const CustomData *data, const eCustomDataType type);

void *CustomData_bmesh_get(const CustomData *data, void *block, const eCustomDataType type);

void CustomData_bmesh_set_default(CustomData *data, void **block);

void CustomData_bmesh_interp_n(CustomData *data,
                               const void **src_blocks_ofs,
                               const float *weights,
                               int count,
                               void *dst_block_ofs,
                               int n);


void CustomData_copy_elements(const eCustomDataType type,
                              const void *src_data,
                              void *dst_data,
                              const int count);

int CustomData_sizeof(const eCustomDataType type);

// void CustomData_bmesh_interp_n(CustomData *data,
                               // const void **src_blocks_ofs,
                               // const float *weights,
                               // int count,
                               // void *dst_block_ofs,
                               // int n);
int CustomData_number_of_layers(const CustomData *data, const eCustomDataType type);

bool CustomData_free_layer_active(CustomData *data, const eCustomDataType type);

void CustomData_bmesh_set_n(
    CustomData *data, void *block, const eCustomDataType type, const int n, const void *source);

void CustomData_bmesh_interp(
    CustomData *data, const void **src_blocks, const float *weights, int count, void *dst_block);
	
bool CustomData_free_layer_named(CustomData *data, const blender::StringRef name);
void *CustomData_add_layer(CustomData *data,
                           const eCustomDataType type,
                           eCDAllocType alloctype,
                           const int totelem);
						   
const char *CustomData_get_layer_name(const CustomData *data,
                                      const eCustomDataType type,
                                      const int n);
									  
bool CustomData_free_layer(CustomData *data, const eCustomDataType type, const int index);

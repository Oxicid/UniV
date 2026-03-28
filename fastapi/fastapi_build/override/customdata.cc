/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


#pragma once

#include "BLI_utildefines.h"

#include "BKE_customdata.hh"
#include "BLI_string_ref.hh"
#include "bmesh.hh"


const CustomData_MeshMasks CD_MASK_BAREMESH = {
    /*vmask*/ CD_MASK_PROP_FLOAT3,
    /*emask*/ CD_MASK_PROP_INT32_2D,
    /*fmask*/ 0,
    /*pmask*/ 0,
    /*lmask*/ CD_MASK_PROP_INT32,
};
const CustomData_MeshMasks CD_MASK_BAREMESH_ORIGINDEX = {
    /*vmask*/ CD_MASK_PROP_FLOAT3 | CD_MASK_ORIGINDEX,
    /*emask*/ CD_MASK_PROP_INT32_2D | CD_MASK_ORIGINDEX,
    /*fmask*/ 0,
    /*pmask*/ CD_MASK_ORIGINDEX,
    /*lmask*/ CD_MASK_PROP_INT32,
};
const CustomData_MeshMasks CD_MASK_MESH = {
    /*vmask*/ (CD_MASK_PROP_FLOAT3 | CD_MASK_MDEFORMVERT | CD_MASK_MVERT_SKIN | CD_MASK_PROP_ALL),
    /*emask*/
    CD_MASK_PROP_ALL,
    /*fmask*/ 0,
    /*pmask*/
    CD_MASK_PROP_ALL,
    /*lmask*/
    (CD_MASK_MDISPS | CD_MASK_GRID_PAINT_MASK | CD_MASK_PROP_ALL),
};
const CustomData_MeshMasks CD_MASK_DERIVEDMESH = {
    /*vmask*/ (CD_MASK_ORIGINDEX | CD_MASK_MDEFORMVERT | CD_MASK_SHAPEKEY | CD_MASK_MVERT_SKIN |
               CD_MASK_ORCO | CD_MASK_CLOTH_ORCO | CD_MASK_PROP_ALL),
    /*emask*/
    (CD_MASK_ORIGINDEX | CD_MASK_PROP_ALL),
    /*fmask*/ (CD_MASK_ORIGINDEX | CD_MASK_ORIGSPACE),
    /*pmask*/
    (CD_MASK_ORIGINDEX | CD_MASK_PROP_ALL),
    /*lmask*/
    (CD_MASK_ORIGSPACE_MLOOP | CD_MASK_PROP_ALL), /* XXX: MISSING #CD_MASK_MLOOPTANGENT ? */
};
const CustomData_MeshMasks CD_MASK_BMESH = {
    /*vmask*/ (CD_MASK_MDEFORMVERT | CD_MASK_MVERT_SKIN | CD_MASK_SHAPEKEY |
               CD_MASK_SHAPE_KEYINDEX | CD_MASK_PROP_ALL),
    /*emask*/ CD_MASK_PROP_ALL,
    /*fmask*/ 0,
    /*pmask*/
    CD_MASK_PROP_ALL,
    /*lmask*/
    (CD_MASK_MDISPS | CD_MASK_GRID_PAINT_MASK | CD_MASK_PROP_ALL),
};
const CustomData_MeshMasks CD_MASK_EVERYTHING = {
    /*vmask*/ (CD_MASK_BM_ELEM_PYPTR | CD_MASK_ORIGINDEX | CD_MASK_MDEFORMVERT |
               CD_MASK_MVERT_SKIN | CD_MASK_ORCO | CD_MASK_CLOTH_ORCO | CD_MASK_SHAPEKEY |
               CD_MASK_SHAPE_KEYINDEX | CD_MASK_PROP_ALL),
    /*emask*/
    (CD_MASK_BM_ELEM_PYPTR | CD_MASK_ORIGINDEX | CD_MASK_PROP_ALL),
    /*fmask*/
    (CD_MASK_MFACE | CD_MASK_ORIGINDEX | CD_MASK_NORMAL | CD_MASK_MTFACE | CD_MASK_MCOL |
     CD_MASK_ORIGSPACE | CD_MASK_TESSLOOPNORMAL | CD_MASK_PROP_ALL),
    /*pmask*/
    (CD_MASK_BM_ELEM_PYPTR | CD_MASK_ORIGINDEX | CD_MASK_PROP_ALL),
    /*lmask*/
    (CD_MASK_BM_ELEM_PYPTR | CD_MASK_MDISPS | CD_MASK_NORMAL | CD_MASK_MLOOPTANGENT |
     CD_MASK_ORIGSPACE_MLOOP | CD_MASK_GRID_PAINT_MASK | CD_MASK_PROP_ALL),
};



static void CustomData_bmesh_set_default_n(CustomData *data, void **block, const int n)
{
  
}


CustomData CustomData_shallow_copy_remove_non_bmesh_attributes(const CustomData *src,
                                                               const eCustomDataMask mask)
{

  CustomData dst;

  return dst;
}

void CustomData_init_layout_from(const CustomData *source,
                                 CustomData *dest,
                                 eCustomDataMask mask,
                                 eCDAllocType alloctype,
                                 int totelem)
								 {}


bool CustomData_merge_layout(const CustomData *source,
                             CustomData *dest,
                             eCustomDataMask mask,
                             eCDAllocType alloctype,
                             int totelem) {return false;}




void CustomData_bmesh_init_pool(CustomData *data, const int totelem, const char htype) {}

bool CustomData_bmesh_merge_layout(const CustomData *source,
                                   CustomData *dest,
                                   eCustomDataMask mask,
                                   eCDAllocType alloctype,
                                   BMesh *bm,
                                   const char htype)
								   {return false;}
								   
void *CustomData_add_layer_named(CustomData *data,
                                 const eCustomDataType type,
                                 const eCDAllocType alloctype,
                                 const int totelem,
                                 const blender::StringRef name) {return nullptr;}

int CustomData_get_layer_index(const CustomData *data, const eCustomDataType type)
{
  // BLI_assert(customdata_typemap_is_valid(data));
  return data->typemap[type];
}

int CustomData_get_active_layer_index(const CustomData *data, const eCustomDataType type)
{
  const int layer_index = data->typemap[type];
  // BLI_assert(customdata_typemap_is_valid(data));
  return (layer_index != -1) ? layer_index + data->layers[layer_index].active : -1;
}

void CustomData_reset(CustomData *data)
  {   }

void CustomData_free(CustomData *data) {}
bool CustomData_bmesh_has_free(const CustomData *data)
{return false;}
void CustomData_bmesh_free_block(CustomData *data, void **block)
{}

int CustomData_get_offset(const CustomData *data, const eCustomDataType type)
{
  int layer_index = CustomData_get_active_layer_index(data, type);
  if (layer_index == -1) {
    return -1;
  }
  return data->layers[layer_index].offset;
}

int CustomData_get_layer_index_n(const CustomData *data, const eCustomDataType type, const int n)
{
  BLI_assert(n >= 0);
  int i = CustomData_get_layer_index(data, type);

  if (i != -1) {
    /* If the value of n goes past the block of layers of the correct type, return -1. */
    i = (i + n < data->totlayer && data->layers[i + n].type == type) ? (i + n) : (-1);
  }

  return i;
}

void *CustomData_bmesh_get_n(const CustomData *data,
                             void *block,
                             const eCustomDataType type,
                             const int n)
{
  int layer_index = CustomData_get_layer_index(data, type);
  if (layer_index == -1) {
    return nullptr;
  }

  return POINTER_OFFSET(block, data->layers[layer_index + n].offset);
}

int CustomData_get_offset_named(const CustomData *data,
                                const eCustomDataType type,
                                const blender::StringRef name)
{
  int layer_index = CustomData_get_named_layer_index(data, type, name);
  if (layer_index == -1) {
    return -1;
  }

  return data->layers[layer_index].offset;
}

int CustomData_get_active_layer(const CustomData *data, const eCustomDataType type)
{
  const int layer_index = data->typemap[type];
  //BLI_assert(customdata_typemap_is_valid(data));
  return (layer_index != -1) ? data->layers[layer_index].active : -1;
}

int CustomData_get_named_layer_index(const CustomData *data,
                                     const eCustomDataType type,
                                     const blender::StringRef name)
{
  for (int i = 0; i < data->totlayer; i++) {
    if (data->layers[i].type == type) {
      if (data->layers[i].name == name) {
        return i;
      }
    }
  }

  return -1;
}


bool CustomData_data_equals(const eCustomDataType type, const void *data1, const void *data2)
{
  // const LayerTypeInfo *typeInfo = layerType_getInfo(type);

  // if (typeInfo->equal) {
    // return typeInfo->equal(data1, data2);
  // }

  // return !memcmp(data1, data2, typeInfo->size);
  return false;
}


BMCustomDataCopyMap CustomData_bmesh_copy_map_calc(const CustomData &src,
                                                   const CustomData &dst,
                                                   const eCustomDataMask mask_exclude)
{
  BMCustomDataCopyMap map;
  return map;
}

void CustomData_bmesh_copy_block(CustomData &dst_data,
                                 const BMCustomDataCopyMap &copy_map,
                                 const void *src_block,
                                 void **dst_block)
  {   }

void CustomData_bmesh_copy_block(CustomData &data, void *src_block, void **dst_block)
	{   }


void *CustomData_bmesh_get(const CustomData *data, void *block, const eCustomDataType type)
{
  int layer_index = CustomData_get_active_layer_index(data, type);
  if (layer_index == -1) {
    return nullptr;
  }

  return POINTER_OFFSET(block, data->layers[layer_index].offset);
}

void CustomData_bmesh_set_default(CustomData *data, void **block) {}

void CustomData_bmesh_interp_n(CustomData *data,
                               const void **src_blocks_ofs,
                               const float *weights,
                               int count,
                               void *dst_block_ofs,
                               int n) {}


void CustomData_copy_elements(const eCustomDataType type,
                              const void *src_data,
                              void *dst_data,
                              const int count) {}


int CustomData_sizeof(const eCustomDataType type) {return 0;}
// void CustomData_bmesh_interp_n(CustomData *data,
                               // const void **src_blocks_ofs,
                               // const float *weights,
                               // int count,
                               // void *dst_block_ofs,
                               // int n) {}

int CustomData_number_of_layers(const CustomData *data, const eCustomDataType type)
{return 0;}

bool CustomData_free_layer_active(CustomData *data, const eCustomDataType type)
{return false;}

void CustomData_bmesh_set_n(
    CustomData *data, void *block, const eCustomDataType type, const int n, const void *source){}
	
void CustomData_bmesh_interp(
    CustomData *data, const void **src_blocks, const float *weights, int count, void *dst_block) {}
	
bool CustomData_free_layer_named(CustomData *data, const blender::StringRef name) {return false;}

void *CustomData_add_layer(CustomData *data,
                           const eCustomDataType type,
                           eCDAllocType alloctype,
                           const int totelem) {return nullptr;}
						   
const char *CustomData_get_layer_name(const CustomData *data,
                                      const eCustomDataType type,
                                      const int n) {return 0;}
									  
bool CustomData_free_layer(CustomData *data, const eCustomDataType type, const int index) {return false;}
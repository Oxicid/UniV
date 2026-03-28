cmake_minimum_required(VERSION 3.10)
project(univ_fastapi)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(BLENDER_ROOT ${CMAKE_SOURCE_DIR}/..)
set(BLENLIB ${BLENDER_ROOT}/source/blender/blenlib)
set(BMESH_DIR ${BLENDER_ROOT}/source/blender/bmesh)

add_definitions(-DWITH_CLOG_PTHREADS)

add_library(univ_fastapi SHARED
    univ_fastapi.cpp
	
	${CMAKE_SOURCE_DIR}/override/customdata.cc
	${CMAKE_SOURCE_DIR}/override/attribute.cc
	${CMAKE_SOURCE_DIR}/override/bmesh_mesh.cc
	${CMAKE_SOURCE_DIR}/override/bmesh_operators.cc
	${CMAKE_SOURCE_DIR}/override/bmesh_construct.cc
	${CMAKE_SOURCE_DIR}/override/bmesh_interp.cc
	${CMAKE_SOURCE_DIR}/override/bmesh_mods.cc
	${CMAKE_SOURCE_DIR}/override/mesh_evaluate.cc
	
	${BLENDER_ROOT}/extern/fmtlib/src/format.cc
	${BLENDER_ROOT}/extern/wcwidth/wcwidth.c
	
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_core.cc
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_query.cc
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_query_uv.cc
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_iterators.cc
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_marking.cc
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_polygon.cc
	
	${BLENDER_ROOT}/source/blender/bmesh/intern/bmesh_structure.cc
	${BLENDER_ROOT}/source/blender/bmesh/tools/bmesh_beautify.cc
	
    ${BLENDER_ROOT}/intern/eigen/intern/linear_solver.cc
	${BLENDER_ROOT}/intern/utfconv/utfconv.cc
	
    ${BLENDER_ROOT}/source/blender/blenlib/intern/polyfill_2d.cc
    ${BLENDER_ROOT}/source/blender/blenlib/intern/polyfill_2d_beautify.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/task_iterator.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_memarena.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_heap.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_solvers.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_vector.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_rotation_c.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_matrix_c.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_geom.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/string_ref.cc
	
	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn_lockfree_impl.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/leak_detector.cc
	
	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn_guarded_impl.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/memory_usage.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_assert.cc
	
	${BLENDER_ROOT}/intern/eigen/intern/eigenvalues.cc
	${BLENDER_ROOT}/intern/eigen/intern/svd.cc
	${BLENDER_ROOT}/intern/eigen/intern/matrix.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_mempool.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_heap.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_heap_simple.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/stack.cc
	
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_linklist.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_ghash.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_ghash_utils.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/hash_mm2a.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/sort_utils.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/listbase.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/string.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/string_utf8.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/threads.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/time.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/task_pool.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/task_range.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/task_scheduler.cc
	
	${BLENDER_ROOT}/source/blender/blenlib/intern/BLI_kdopbvh.cc

)

target_include_directories(univ_fastapi PRIVATE
	${CMAKE_SOURCE_DIR}/override

    ${BLENDER_ROOT}/lib/windows_x64/pthreads/include
    ${BLENDER_ROOT}/intern/eigen
    ${BLENDER_ROOT}/intern/eigen/intern
    ${BLENDER_ROOT}/extern/eigen3
	${BLENDER_ROOT}/source/blender/blenlib
	
	${BLENDER_ROOT}/source/blender/blenkernel
	${BLENDER_ROOT}/source/blender/bmesh
	${BLENDER_ROOT}/source/blender/makesdna
	${BLENDER_ROOT}/source/blender/memutil 
	${BLENDER_ROOT}/intern/guardedalloc
	${BLENDER_ROOT}/intern/guardedalloc/intern
	${BLENDER_ROOT}/intern/atomic
	
	${BMESH_DIR}/intern
	${BLENDER_ROOT}/extern/fmtlib
	${BLENDER_ROOT}/extern/fmtlib/include
	
	${BLENDER_ROOT}/extern/wcwidth
    ${BLENDER_ROOT}/extern/wcwidth/include
	
	${BLENDER_ROOT}/intern/utfconv
	 
)

set(EXTRA_LIBS "")

if(WIN32)
    list(APPEND EXTRA_LIBS
        ${BLENDER_ROOT}/lib/windows_x64/pthreads/lib/pthreadVC3.lib
    )
endif()

target_link_libraries(univ_fastapi PRIVATE ${EXTRA_LIBS})

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /arch:SSE2 /fp:precise")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX /fp:fast")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -ffast-math")
endif()
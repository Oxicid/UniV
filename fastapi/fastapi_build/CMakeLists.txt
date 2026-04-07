cmake_minimum_required(VERSION 3.10)
project(univ_fastapi)

set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(BLENDER_ROOT ${CMAKE_SOURCE_DIR}/..)
set(BLENLIB ${BLENDER_ROOT}/source/blender/blenlib)
set(BMESH_DIR ${BLENDER_ROOT}/source/blender/bmesh)



if (WIN32)
	set(LIB_BY_OS_FOLDER ${BLENDER_ROOT}/lib/windows_x64)
	if (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
		set(LIB_BY_OS_FOLDER ${BLENDER_ROOT}/lib/windows_arm64)
	endif()
elseif(UNIX)
	set(LIB_BY_OS_FOLDER ${BLENDER_ROOT}/lib/linux_x64)
else()
	set(LIB_BY_OS_FOLDER ${BLENDER_ROOT}/lib/macos_arm64)
endif()


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

	###################
	# Min solver requirements
	${BLENDER_ROOT}/intern/eigen/intern/linear_solver.cc
	${BLENDER_ROOT}/source/blender/blenlib/intern/math_base_inline.cc
	###################


	${BLENDER_ROOT}/intern/eigen/intern/eigenvalues.cc
	${BLENDER_ROOT}/intern/eigen/intern/svd.cc
	${BLENDER_ROOT}/intern/eigen/intern/matrix.cc


	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn_lockfree_impl.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/leak_detector.cc
	
	${BLENDER_ROOT}/intern/guardedalloc/intern/mallocn_guarded_impl.cc
	${BLENDER_ROOT}/intern/guardedalloc/intern/memory_usage.cc
	${BLENDER_ROOT}/intern/utfconv/utfconv.cc

	${BLENDER_ROOT}/extern/wcwidth/wcwidth.c
	
	${BMESH_DIR}/intern/bmesh_core.cc
	${BMESH_DIR}/intern/bmesh_query.cc
	${BMESH_DIR}/intern/bmesh_query_uv.cc
	${BMESH_DIR}/intern/bmesh_iterators.cc
	${BMESH_DIR}/intern/bmesh_marking.cc
	${BMESH_DIR}/intern/bmesh_polygon.cc
	${BMESH_DIR}/intern/bmesh_structure.cc
	${BMESH_DIR}/tools/bmesh_beautify.cc
	
	
	${BLENLIB}/intern/BLI_assert.cc
	
    ${BLENLIB}/intern/polyfill_2d.cc
    ${BLENLIB}/intern/polyfill_2d_beautify.cc
	
	${BLENLIB}/intern/math_solvers.cc
	${BLENLIB}/intern/math_vector.cc
	${BLENLIB}/intern/math_rotation_c.cc
	${BLENLIB}/intern/math_matrix_c.cc
	${BLENLIB}/intern/math_geom.cc
	
	
	${BLENLIB}/intern/BLI_memarena.cc
	${BLENLIB}/intern/BLI_heap.cc
	${BLENLIB}/intern/BLI_mempool.cc
	${BLENLIB}/intern/BLI_heap_simple.cc
	${BLENLIB}/intern/stack.cc
	
	
	${BLENLIB}/intern/BLI_linklist.cc
	${BLENLIB}/intern/BLI_ghash.cc
	${BLENLIB}/intern/BLI_ghash_utils.cc
	${BLENLIB}/intern/hash_mm2a.cc
	${BLENLIB}/intern/sort_utils.cc
	${BLENLIB}/intern/listbase.cc
	${BLENLIB}/intern/string.cc
	
	${BLENLIB}/intern/threads.cc
	${BLENLIB}/intern/time.cc
	${BLENLIB}/intern/task_pool.cc
	${BLENLIB}/intern/task_range.cc
	${BLENLIB}/intern/task_scheduler.cc
	${BLENLIB}/intern/task_iterator.cc
	
	${BLENLIB}/intern/BLI_kdopbvh.cc
)

set(INCLUDE_DIRS "")
if(WIN32)
    list(APPEND INCLUDE_DIRS ${BLENDER_ROOT}/lib/windows_x64/pthreads/include)
endif()

target_include_directories(univ_fastapi PRIVATE
	${BMESH_DIR}
	${INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/override
	
	# Min solver requirements
	${LIB_BY_OS_FOLDER}/eigen/include/eigen3
	${BLENDER_ROOT}/intern/eigen
	${BLENLIB}
	##########
	
	
	${BLENDER_ROOT}/source/blender/blenkernel
	
	${BLENDER_ROOT}/source/blender/makesdna
	${BLENDER_ROOT}/source/blender/memutil 
	${BLENDER_ROOT}/intern/guardedalloc
	${BLENDER_ROOT}/intern/guardedalloc/intern
	${BLENDER_ROOT}/intern/atomic

	${LIB_BY_OS_FOLDER}/fmt/include
    ${BLENDER_ROOT}/extern/wcwidth/include
)


set(EXTRA_LIBS "")

if(WIN32)
    list(APPEND EXTRA_LIBS ${BLENDER_ROOT}/lib/windows_x64/pthreads/lib/pthreadVC3.lib)
    list(APPEND EXTRA_LIBS ${LIB_BY_OS_FOLDER}/fmt/lib/fmt.lib)
else()
    list(APPEND EXTRA_LIBS ${LIB_BY_OS_FOLDER}/fmt/lib/libfmt.a)
endif()

target_link_libraries(univ_fastapi PRIVATE 
	${EXTRA_LIBS}
	)


if(MSVC)
    target_compile_options(univ_fastapi PRIVATE
        /O2        # max optimization
        /Oi        # intrinsic functions
        /Ot        # optimization for speed (not size)
        /GL        # LTO (Whole Program Optimization)
        /arch:AVX  # AVX support (SSE2, AVX, AVX2)
        /fp:fast   # fast float
        /Gy        # functions into separate sections (for the linker)
        /Gw        # data into separate sections
        /GF        # Eliminate Duplicate Strings
        /GR        #
        /MP        # Multiprocessing compilation

    )

    target_link_options(univ_fastapi PRIVATE
        /LTCG      # linker optimization (must be used with /GL)
        /OPT:REF   # remove unused code
        /OPT:ICF   # function deduplication
    )

    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX /fp:fast")
else()
    target_compile_options(univ_fastapi PRIVATE
        -O3                    # max optimization
        -flto
        -mavx                  # AVX support (SSE2, mavx, mavx2)
        -ffast-math
        -ffunction-sections
        -fdata-sections
        -fmerge-all-constants
    )

    target_link_options(univ_fastapi PRIVATE
        -flto
        -Wl,--gc-sections
    )

    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -ffast-math")
endif()


if(MSVC)
	add_definitions(
		/D_USE_MATH_DEFINES
		/DNOMINMAX
	)
endif()

add_definitions(-DWITH_CLOG_PTHREADS)
remove_definitions(-DWITH_OPENCOLORIO)


if(MSVC)
	target_compile_options(univ_fastapi PRIVATE /utf-8)
	
	target_compile_options(univ_fastapi PRIVATE /wd4005)
	target_compile_options(univ_fastapi PRIVATE /wd4018)
	target_compile_options(univ_fastapi PRIVATE /wd4305)
	target_compile_options(univ_fastapi PRIVATE /wd4309)
	target_compile_options(univ_fastapi PRIVATE /wd4200)
	target_compile_options(univ_fastapi PRIVATE /wd4244)
	target_compile_options(univ_fastapi PRIVATE /wd4267)
	target_compile_options(univ_fastapi PRIVATE /wd4717)
	target_compile_options(univ_fastapi PRIVATE /wd4838)
	target_compile_options(univ_fastapi PRIVATE /wd4996)
else()
	target_compile_options(univ_fastapi PRIVATE -w)
	target_compile_options(univ_fastapi PRIVATE -Wno-macro-redefined)
endif()


if(UNIX)
	target_compile_definitions(univ_fastapi PRIVATE HAVE_MALLOC_STATS_H)
endif()

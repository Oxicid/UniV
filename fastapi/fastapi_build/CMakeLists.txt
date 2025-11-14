cmake_minimum_required(VERSION 3.10)
project(univ_fastapi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(univ_fastapi SHARED
    univ_fastapi.cpp
    ${CMAKE_SOURCE_DIR}/../intern/eigen/intern/linear_solver.cc
	# ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/math_vector.cc
    # ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/math_vec.cc
	# ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/memory_utils.cc 
	# ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/math_vec_interp.cc
	# ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/math_interp.cc
	# ${CMAKE_SOURCE_DIR}/../source/blender/blenlib/intern/math_base_safe_inline.cc
)

target_include_directories(univ_fastapi PRIVATE
    ${CMAKE_SOURCE_DIR}/../intern/eigen
    ${CMAKE_SOURCE_DIR}/../intern/eigen/intern
    ${CMAKE_SOURCE_DIR}/../extern/eigen3
	${CMAKE_SOURCE_DIR}/../source/blender/blenlib
	# ${CMAKE_SOURCE_DIR}/../source/blender/makesdna
	# ${CMAKE_SOURCE_DIR}/../source/blender/memutil 
	# ${CMAKE_SOURCE_DIR}/../intern/guardedalloc
	
)


if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /arch:SSE2 /fp:precise")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX /fp:fast")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -ffast-math")
endif()
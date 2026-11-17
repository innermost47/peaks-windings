# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Anthony Charretier
cmake_minimum_required(VERSION 3.16)
project(peaks_windings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PW_NATIVE "Tune the build for this machine (-march=native); the binary may not run elsewhere" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

find_package(blosc CONFIG QUIET)
if(TARGET blosc_shared)
  set(PW_BLOSC blosc_shared)
elseif(TARGET blosc_static)
  set(PW_BLOSC blosc_static)
else()
  find_path(BLOSC_INCLUDE_DIR blosc.h)
  find_library(BLOSC_LIBRARY NAMES blosc libblosc)
  if(NOT BLOSC_INCLUDE_DIR OR NOT BLOSC_LIBRARY)
    message(FATAL_ERROR "c-blosc not found (apt: libblosc-dev, vcpkg: blosc)")
  endif()
  add_library(pw_blosc INTERFACE)
  target_include_directories(pw_blosc INTERFACE ${BLOSC_INCLUDE_DIR})
  target_link_libraries(pw_blosc INTERFACE ${BLOSC_LIBRARY})
  set(PW_BLOSC pw_blosc)
endif()

find_package(zstd CONFIG QUIET)
if(TARGET zstd::libzstd)
  set(PW_ZSTD zstd::libzstd)
elseif(TARGET zstd::libzstd_shared)
  set(PW_ZSTD zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  set(PW_ZSTD zstd::libzstd_static)
else()
  find_path(ZSTD_INCLUDE_DIR zstd.h)
  find_library(ZSTD_LIBRARY NAMES zstd libzstd zstd_static)
  if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    add_library(pw_zstd INTERFACE)
    target_include_directories(pw_zstd INTERFACE ${ZSTD_INCLUDE_DIR})
    target_link_libraries(pw_zstd INTERFACE ${ZSTD_LIBRARY})
    set(PW_ZSTD pw_zstd)
  endif()
endif()

find_package(ZLIB QUIET)

add_library(pw_common STATIC
  src/zarr_reader.cpp
  src/layers.cpp
)
target_include_directories(pw_common PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pw_common PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  Threads::Threads
  PRIVATE
  CURL::libcurl
  ${PW_BLOSC}
)
if(PW_ZSTD)
  target_link_libraries(pw_common PRIVATE ${PW_ZSTD})
  target_compile_definitions(pw_common PRIVATE PW_HAVE_ZSTD)
else()
  message(STATUS "zstd not found: stores compressed with zstd directly will not be readable")
endif()
if(ZLIB_FOUND)
  target_link_libraries(pw_common PRIVATE ZLIB::ZLIB)
  target_compile_definitions(pw_common PRIVATE PW_HAVE_ZLIB)
else()
  message(STATUS "zlib not found: stores compressed with zlib / gzip will not be readable")
endif()

add_executable(peaks_windings src/main.cpp src/windings.cpp)   
add_executable(unroll_layer src/unroll_layer.cpp)           
add_executable(export_3d src/export_3d.cpp)              

set(PW_TARGETS pw_common peaks_windings unroll_layer export_3d)
foreach(target IN LISTS PW_TARGETS)
  if(NOT target STREQUAL "pw_common")
    target_link_libraries(${target} PRIVATE pw_common)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE /W3 /utf-8)
    target_compile_definitions(${target} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    if(PW_NATIVE)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
endforeach()

install(TARGETS peaks_windings unroll_layer export_3d RUNTIME DESTINATION bin)

# Minimum CMake required
cmake_minimum_required(VERSION 3.1)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# Project
project(onnx-xla C CXX)
option(ONNX_XLA_BUILD_BENCHMARKS "Build ONNX_XLA micro-benchmarks" OFF)

option(ONNX_XLA_WERROR "Build with Werror" OFF)
option(ONNX_XLA_COVERAGE "Build with coverage instrumentation" OFF)
option(ONNX_XLA_BUILD_TESTS "Build ONNX_XLA C++ APIs Tests" OFF)

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

set(ONNX_XLA_ROOT ${PROJECT_SOURCE_DIR})

set(CMAKE_MODULE_PATH "")
list(APPEND CMAKE_MODULE_PATH ${ONNX_XLA_ROOT}/cmake/Modules)

if(NOT MSVC)
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  if(ONNX_XLA_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif()
endif()

if(ONNX_XLA_BUILD_TESTS)
	list(APPEND CMAKE_MODULE_PATH ${ONNX_XLA_ROOT}/cmake/external)
	include(googletest)
endif()

if(TARGET protobuf::libprotobuf)
  # Sometimes we need to use protoc compiled for host architecture while
  # linking libprotobuf against target architecture. See
  # https://github.com/caffe2/caffe2/blob/96f35ad75480b25c1a23d6e9e97bccae9f7a7f9c/cmake/ProtoBuf.cmake#L92-L99
  if(EXISTS "${ONNX_XLA_CUSTOM_PROTOC_EXECUTABLE}")
    message(STATUS "Using custom protoc executable")
    set(ONNX_XLA_PROTOC_EXECUTABLE ${ONNX_XLA_CUSTOM_PROTOC_EXECUTABLE})
  else()
    set(ONNX_XLA_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
  endif()
else()
  # Customized version of find Protobuf. We need to avoid situations mentioned
  # in https://github.com/caffe2/caffe2/blob/b7d983f255ef5496474f1ea188edb5e0ac442761/cmake/ProtoBuf.cmake#L82-L92
  # The following section is stolen from cmake/ProtoBuf.cmake in Caffe2
  find_program(Protobuf_PROTOC_EXECUTABLE
    NAMES protoc
    DOC "The Google Protocol Buffers Compiler")

  # Only if protoc was found, seed the include directories and libraries.
  # We assume that protoc is installed at PREFIX/bin.
  # We use get_filename_component to resolve PREFIX.
  if(Protobuf_PROTOC_EXECUTABLE)
    set(ONNX_XLA_PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
    get_filename_component(
      _PROTOBUF_INSTALL_PREFIX
      ${Protobuf_PROTOC_EXECUTABLE}
      DIRECTORY)
    get_filename_component(
      _PROTOBUF_INSTALL_PREFIX
      ${_PROTOBUF_INSTALL_PREFIX}/..
      REALPATH)
    find_library(Protobuf_LIBRARY
      NAMES protobuf
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
      NO_DEFAULT_PATH)
    find_library(Protobuf_PROTOC_LIBRARY
      NAMES protoc
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
      NO_DEFAULT_PATH)
    find_library(Protobuf_LITE_LIBRARY
      NAMES protobuf-lite
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
      NO_DEFAULT_PATH)
    find_path(Protobuf_INCLUDE_DIR
      google/protobuf/service.h
      PATHS ${_PROTOBUF_INSTALL_PREFIX}/include
      NO_DEFAULT_PATH)
    find_package(Protobuf REQUIRED)
  endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ONNX_XLA_NAMESPACE)
  SET(MY_ONNX_XLA_NAMESPACE "-DONNX_XLA_NAMESPACE=${ONNX_XLA_NAMESPACE}")
else()
  SET(ONNX_XLA_NAMESPACE "onnx-xla")
  SET(MY_ONNX_XLA_NAMESPACE "-DONNX_XLA_NAMESPACE=onnx-xla")
endif()
add_definitions(${MY_ONNX_XLA_NAMESPACE})

if(ONNX_XLA_ML)
  add_definitions("-DONNX_XLA_ML=1")
endif()

find_package(Threads)

file(GLOB_RECURSE xla_src
    "${ONNX_XLA_ROOT}/xla_pb/*.h"
    "${ONNX_XLA_ROOT}/xla_pb/*.cc"
    "${ONNX_XLA_ROOT}/test_client.cc"
)
add_executable(xla_o ${xla_src})
target_include_directories(xla_o PUBLIC "${ONNX_XLA_ROOT}/xla_pb/"
                                         "${PROTOBUF_INCLUDE_DIRS}"
                                         "usr/local/include/grpc"
				 	 "usr/local/include/grpc++"
					 "usr/local/include/grpcpp")
if(TARGET protobuf::libprotobuf)
  target_link_libraries(xla_o PUBLIC protobuf::libprotobuf
                                    "/usr/local/lib/libgrpc.so"
				    "/usr/local/lib/libgrpc++.so"
				    ${CMAKE_THREAD_LIBS_INIT})
else()
  target_link_libraries(xla_o PUBLIC ${PROTOBUF_LIBRARIES}
				      "/usr/local/lib/libgrpc.so"
				      "/usr/local/lib/libgrpc++.so"
				      ${CMAKE_THREAD_LIBS_INIT})
endif()

# Export include directories
# set(ONNX_XLA_INCLUDE_DIRS "${ONNX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}")
# get_directory_property(hasParent PARENT_DIRECTORY)
# if(hasParent)
    # set(ONNX_XLA_INCLUDE_DIRS ${ONNX_XLA_INCLUDE_DIRS} PARENT_SCOPE)
# endif()

# ---[ ONNX Interface for Framework Integratin (ONNXIFI)
# add_library(onnxifi INTERFACE)
# target_include_directories(onnxifi INTERFACE onnx)


#add_executable(xla_client "${ONNX_XLA_ROOT}/test_client.cc")
#target_include_directories(xla_client PUBLIC "${ONNX_XLA_ROOT}/xla_pb/"
#                                         "${PROTOBUF_INCLUDE_DIRS}"
#                                         "${gRPC_INSTALL_INCLUDEDIR}")


#if(TARGET protobuf::libprotobuf)
#  target_link_libraries(xla_client PUBLIC protobuf::libprotobuf
#                                         "${gRPC_INSTALL_LIBDIR}"
#					 xla_pb)
#else()
#  target_link_libraries(xla_client PUBLIC ${PROTOBUF_LIBRARIES}
#                                      "${gRPC_INSTALL_LIBDIR}"
#				      xla_pb)
#endif()

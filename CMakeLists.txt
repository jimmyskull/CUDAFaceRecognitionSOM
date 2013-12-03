cmake_minimum_required(VERSION 2.8)
project(FaceSOM C CXX)

set(WITH_CUDA ON)

enable_language(C)
enable_language(CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_C_FLAGS AND NOT CMAKE_CXX_FLAGS)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_COMPILER g++)
    set(CUDA_FLAGS -O3)
    add_definitions(-DDEBUG_VERBOSITY=WARN)
endif(CMAKE_BUILD_TYPE MATCHES Release)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_COMPILER clang++)
    set(CUDA_FLAGS -g -G)
    add_definitions(-DDEBUG_VERBOSITY=TESTING)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")

set(IGNORED_WARNINGS "-Wno-c++98-compat-pedantic -Wno-gnu -Wno-padded")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

cuda_compile(MAIN_O main.cu OPTIONS ${CUDA_FLAGS})
cuda_compile(INIT_O initialize.cu OPTIONS ${CUDA_FLAGS})
cuda_compile(BEST_O find_best.cu OPTIONS ${CUDA_FLAGS})
cuda_compile(UPDATE_O update_neighbors.cu OPTIONS ${CUDA_FLAGS})

cuda_add_executable(FaceSOM  ${MAIN_O} ${INIT_O} ${BEST_O} ${UPDATE_O})

set_target_properties(FaceSOM PROPERTIES COMPILE_FLAGS "-std=c++11")
set_target_properties(FaceSOM PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(FaceSOM ${OpenCV_LIBS} ${CUDA_curand_LIBRARY})

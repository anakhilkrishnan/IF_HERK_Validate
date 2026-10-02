cmake_minimum_required(VERSION 3.20)
project(IF_HERK_Validate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# AMReX: an installed AMReX (-DAMReX_ROOT=...) by default, or fetched and built
# in-tree with -DIFV_FETCH_AMREX=ON. The dimension is fixed at compile time,
# so 1D/2D/3D are separate build directories.
# ---------------------------------------------------------------------------
set(IFV_DIM 3 CACHE STRING "Spatial dimension (1, 2 or 3)")
set_property(CACHE IFV_DIM PROPERTY STRINGS 1 2 3)
option(IFV_FETCH_AMREX "Download and build AMReX as part of this project" OFF)

if(IFV_FETCH_AMREX)
    include(FetchContent)
    set(AMReX_SPACEDIM ${IFV_DIM} CACHE STRING "" FORCE)
    set(AMReX_FORTRAN OFF CACHE BOOL "" FORCE)
    set(AMReX_PARTICLES OFF CACHE BOOL "" FORCE)
    set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(amrex
        GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
        GIT_TAG        development
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(amrex)
    list(APPEND CMAKE_MODULE_PATH "${amrex_SOURCE_DIR}/Tools/CMake")
else()
    find_package(AMReX REQUIRED COMPONENTS ${IFV_DIM}D)
endif()

# setup_target_for_cuda_compilation. An in-tree AMReX (fetched) has already
# defined it while configuring itself -- CMake functions are global. An
# installed AMReX only puts the module on CMAKE_MODULE_PATH, so include it then.
if(NOT COMMAND setup_target_for_cuda_compilation)
    include(AMReXTargetHelpers)
endif()

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
endif()

# mark a target's sources for the active GPU backend (no-op on CPU builds)
function(ifv_setup_gpu tgt)
    if(AMReX_GPU_BACKEND STREQUAL "CUDA")
        setup_target_for_cuda_compilation(${tgt})
    endif()
endfunction()

add_subdirectory(Src/IFHERK_core)
add_subdirectory(Src/ADV_core)
add_subdirectory(Src/ADV_run)

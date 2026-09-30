# Portable integrator core: knows nothing about the problem being integrated.
# LGF_NSE_GPU links this same library once ProjectionWorkspace is ported.
add_library(IFHERK_core_lib STATIC ifherk.cpp)
target_include_directories(IFHERK_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(IFHERK_core_lib PUBLIC AMReX::amrex)
ifv_setup_gpu(IFHERK_core_lib)

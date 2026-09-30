# Problem-specific pieces of the advection-diffusion test bench.
add_library(ADV_core_lib STATIC scalarStepper.cpp)
target_include_directories(ADV_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ADV_core_lib PUBLIC IFHERK_core_lib)
ifv_setup_gpu(ADV_core_lib)

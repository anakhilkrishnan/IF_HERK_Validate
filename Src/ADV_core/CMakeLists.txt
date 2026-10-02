# Problem-specific pieces of the advection-diffusion test bench.
set(ADV_SOURCES scalarStepper.cpp)
if(IFV_DIM GREATER_EQUAL 2)
    # vector Burgers uses the NSE stencils, which assume at least 2D
    list(APPEND ADV_SOURCES vectorStepper.cpp)
endif()

add_library(ADV_core_lib STATIC ${ADV_SOURCES})
target_include_directories(ADV_core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../NSE_shared)   # verbatim copies of LGF_NSE_GPU headers
target_link_libraries(ADV_core_lib PUBLIC IFHERK_core_lib)
ifv_setup_gpu(ADV_core_lib)

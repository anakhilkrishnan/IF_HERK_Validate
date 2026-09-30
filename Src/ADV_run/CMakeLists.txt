add_executable(ADV_run main.cpp)
target_link_libraries(ADV_run PRIVATE ADV_core_lib)
set_target_properties(ADV_run PROPERTIES OUTPUT_NAME "adv${IFV_DIM}d.ex")
ifv_setup_gpu(ADV_run)

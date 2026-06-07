add_executable(streamer_example streamer_example.cu)
target_include_directories(streamer_example
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src/
    ${YAML_CPP_INCLUDE_DIR})
target_compile_options(streamer_example PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-I/usr/include/x86_64-linux-gnu>
  $<$<COMPILE_LANGUAGE:CUDA>:-I${YAML_CPP_INCLUDE_DIR}>)
target_link_libraries(streamer_example
                      ${GEOM_LIB_NAME}
                      yaml-cpp)

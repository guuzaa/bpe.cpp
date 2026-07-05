file(GLOB EXAMPLE_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src ${EXAMPLE_SRCS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE bpe::bpe)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    bpe_enable_warnings(${name})
endforeach()

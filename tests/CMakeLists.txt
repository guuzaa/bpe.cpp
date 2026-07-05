file(GLOB_RECURSE BPE_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(bpe_tests ${BPE_TESTS})
target_link_libraries(bpe_tests PRIVATE
    bpe::bpe
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
# 测试需要访问 src/ 下的内部头(如 models/bpe/word.h)
target_include_directories(bpe_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

bpe_enable_warnings(bpe_tests)

include(GoogleTest)
gtest_discover_tests(bpe_tests)

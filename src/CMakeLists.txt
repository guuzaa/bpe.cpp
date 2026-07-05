# bpe 静态库:实现源码均在 src/,公共头在 include/bpe/
file(GLOB_RECURSE BPE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
add_library(bpe STATIC ${BPE_SOURCES})
add_library(bpe::bpe ALIAS bpe)

target_include_directories(bpe
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bpe
  PUBLIC
    absl::flat_hash_map
    absl::flat_hash_set
    absl::status
    absl::statusor
    absl::strings
  PRIVATE
    bpe::pcre2
    nlohmann_json::nlohmann_json
)

bpe_enable_warnings(bpe)

add_executable(bench_bytes_char EXCLUDE_FROM_ALL bench_bytes_char.cpp)
target_link_libraries(bench_bytes_char PRIVATE
    bpe::bpe
    benchmark::benchmark
)
# 需访问 src/ 下的内部头 utils/bytes_char.h
target_include_directories(bench_bytes_char PRIVATE ${PROJECT_SOURCE_DIR}/src)

bpe_enable_warnings(bench_bytes_char)

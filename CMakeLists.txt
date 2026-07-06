cmake_minimum_required(VERSION 3.21)

project(bpe_cpp LANGUAGES CXX VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

option(BPE_BUILD_TESTS "Build unit tests" ON)
option(BPE_BUILD_BENCH "Build benchmarks" OFF)
option(BPE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ---- 编译选项(仅作用于本项目目标,不影响第三方) --------------------------
function(bpe_enable_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
      -Wconversion -Wsign-conversion -Wold-style-cast
      -Wno-unused-parameter
      -Wno-gcc-compat # 抑制 abseil 头中 enable_if 的 clang 扩展告警
    )
    if(BPE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    if(BPE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()
endfunction()


# ---- 第三方依赖 -------------------------------------------------------------
include(FetchContent)

# abseil-cpp: hashmap / Status / Mutex / strings
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        20240722.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(abseil)

# nlohmann/json: 序列化
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# PCRE2: PreTokenizer 正则(系统库优先,无则源码拉取)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PCRE2 IMPORTED_TARGET QUIET libpcre2-8)
endif()
if(PCRE2_FOUND)
  add_library(bpe::pcre2 ALIAS PkgConfig::PCRE2)
  message(STATUS "PCRE2: system (via pkg-config) ${PCRE2_VERSION}")
else()
  FetchContent_Declare(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG        pcre2-10.43
    GIT_SHALLOW    TRUE
  )
  set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
  set(PCRE2_SUPPORT_JIT     ON CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
  set(PCRE2_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(PCRE2_INSTALL         OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(pcre2)
  add_library(bpe::pcre2 ALIAS pcre2-8-static)
  message(STATUS "PCRE2: vendored (FetchContent)")
endif()

if(BPE_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
  )
  set(gtest_force_shared_rt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# ---- 子目录 -----------------------------------------------------------------
add_subdirectory(src)

if(BPE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(examples)

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE FORMAT_FILES
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*.h
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
    ${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp ${CMAKE_CURRENT_LIST_DIR}/tests/*.h
    ${CMAKE_CURRENT_LIST_DIR}/examples/*.cpp ${CMAKE_CURRENT_LIST_DIR}/examples/*.h
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${FORMAT_FILES}
    COMMENT "Formatting source files"
  )
  add_custom_target(check-format
    COMMAND ${CLANG_FORMAT} --dry-run --Werror ${FORMAT_FILES}
    COMMENT "Checking source formatting"
  )
endif()
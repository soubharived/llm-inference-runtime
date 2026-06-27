cmake_minimum_required(VERSION 3.16)
project(llm_runtime VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")

# ── llama.cpp paths ─────────────────────────────────────────────────
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/llama.cpp")
set(LLAMA_BIN_DIR "${LLAMA_DIR}/build/bin")

# ── Libraries ───────────────────────────────────────────────────────
find_library(LLAMA_LIB
    NAMES llama
    PATHS "${LLAMA_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(GGML_LIB
    NAMES ggml
    PATHS "${LLAMA_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(GGML_BASE_LIB
    NAMES ggml-base
    PATHS "${LLAMA_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(GGML_CPU_LIB
    NAMES ggml-cpu
    PATHS "${LLAMA_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "Found llama:     ${LLAMA_LIB}")
message(STATUS "Found ggml:      ${GGML_LIB}")
message(STATUS "Found ggml-base: ${GGML_BASE_LIB}")
message(STATUS "Found ggml-cpu:  ${GGML_CPU_LIB}")

# ── Runtime sources ─────────────────────────────────────────────────
add_executable(llm_runtime
    runtime/inference.cpp
    runtime/queue.cpp
    runtime/scheduler.cpp
    runtime/batcher.cpp
    runtime/kvcache.cpp
    runtime/server.cpp
)

target_include_directories(llm_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(llm_runtime PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
    pthread
)

# ── RPATH so binary finds .so files at runtime ──────────────────────
set_target_properties(llm_runtime PROPERTIES
    BUILD_RPATH   "${LLAMA_BIN_DIR}"
    INSTALL_RPATH "${LLAMA_BIN_DIR}"
)
cmake_minimum_required(VERSION 3.20)
project(motion_interp C)

set(CMAKE_C_STANDARD 11)
find_package(Vulkan REQUIRED)

# ──────────────────────────────────────────────────────────────────────────────
# GLSL → SPIR-V compilation
# ──────────────────────────────────────────────────────────────────────────────
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

function(compile_shader TARGET_NAME SRC_FILE SPV_NAME)
    set(SPV_FILE ${SHADER_BIN_DIR}/${SPV_NAME}.spv)
    set(EMBED_FILE ${SHADER_BIN_DIR}/${SPV_NAME}_spv.c)

    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLC}
            -fshader-stage=compute
            --target-env=vulkan1.2
            -O
            ${SHADER_SRC_DIR}/${SRC_FILE}
            -o ${SPV_FILE}
        DEPENDS ${SHADER_SRC_DIR}/${SRC_FILE}
        COMMENT "Compiling ${SRC_FILE} → ${SPV_NAME}.spv"
    )

    # Embed SPIR-V as a C uint32_t array for static linking
    add_custom_command(
        OUTPUT ${EMBED_FILE}
        COMMAND ${CMAKE_COMMAND}
            -DSPV_FILE=${SPV_FILE}
            -DOUT_FILE=${EMBED_FILE}
            -DARRAY_NAME=mi_${SPV_NAME}
            -P ${CMAKE_SOURCE_DIR}/cmake/EmbedSpv.cmake
        DEPENDS ${SPV_FILE}
        COMMENT "Embedding ${SPV_NAME}.spv"
    )

    target_sources(${TARGET_NAME} PRIVATE ${EMBED_FILE})
endfunction()

# ──────────────────────────────────────────────────────────────────────────────
# Library
# ──────────────────────────────────────────────────────────────────────────────
add_library(motion_interp STATIC
    src/motion_interp.c
)
target_include_directories(motion_interp PUBLIC include ${SHADER_BIN_DIR})
target_link_libraries(motion_interp PUBLIC Vulkan::Vulkan m)

compile_shader(motion_interp  pyramid_downsample.glsl  pyramid)
compile_shader(motion_interp  mv_upsample.glsl          mv_upsample)
compile_shader(motion_interp  motion_analyze.glsl        analyze)
compile_shader(motion_interp  occlusion_detect.glsl      occlusion)
compile_shader(motion_interp  motion_flow_fps.glsl        flowfps)

# ──────────────────────────────────────────────────────────────────────────────
# Tests / demo
# ──────────────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_executable(test_pipeline tests/test_pipeline.c)
    target_link_libraries(test_pipeline motion_interp)
endif()

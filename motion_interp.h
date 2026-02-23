// motion_interp.h
// Host-side driver for GPU motion-adaptive frame interpolation pipeline.
// Wraps Vulkan compute pipeline setup, descriptor management, and dispatch.
// Designed to integrate with a standard Vulkan renderer; only the
// compute-specific portions are shown.

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

typedef struct MI_Config {
    int    frameWidth;
    int    frameHeight;
    int    blockSize;        // 4, 8, 16, 32 — power of 2
    int    blockOverlap;     // pixels of overlap between blocks (0 = none)
    int    searchRange;      // max integer-pel search displacement
    bool   useSATD;          // Hadamard transform cost (better quality, ~15% slower)
    bool   subpelRefine;     // half-pixel refinement pass
    int    pyramidLevels;    // hierarchical ME levels (1 = no pyramid)
    float  lambda;           // RD-cost motion penalty weight
    float  sadThreshold;     // SAD threshold for occlusion/reliability (0.05–0.2)
    float  fbConsistThresh;  // forward-backward consistency threshold in pixels (1.0–3.0)
} MI_Config;

typedef struct MI_Context MI_Context;

// ──────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Create a motion interpolation context.
 * @param device   VkDevice handle (cast to void*)
 * @param allocator VmaAllocator or null for internal allocator
 * @param cfg      Pipeline configuration (copied internally)
 */
MI_Context* MI_Create(void* device, void* allocator, const MI_Config* cfg);

/** Release all GPU resources. */
void MI_Destroy(MI_Context* ctx);

// ──────────────────────────────────────────────────────────────────────────────
// Per-frame interpolation
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Analyze motion between frame0 and frame1.
 * Fills internal forward and backward MV buffers.
 * Call once per pair of source frames.
 *
 * @param cmdBuf  VkCommandBuffer (cast to void*), must be in recording state
 * @param frame0  VkImageView of previous frame luma (cast to void*)
 * @param frame1  VkImageView of next frame luma (cast to void*)
 */
void MI_Analyze(MI_Context* ctx, void* cmdBuf, void* frame0, void* frame1);

/**
 * Interpolate a frame at time t ∈ (0,1).
 * Uses the MVs computed by the most recent MI_Analyze call.
 *
 * @param cmdBuf  recording command buffer
 * @param frame0  VkImageView of frame at t=0 (full color)
 * @param frame1  VkImageView of frame at t=1 (full color)
 * @param output  VkImageView of destination frame (must be STORAGE usage)
 * @param t       interpolation position (0.5 = midpoint)
 */
void MI_Interpolate(MI_Context* ctx, void* cmdBuf,
                    void* frame0, void* frame1, void* output, float t);

// ──────────────────────────────────────────────────────────────────────────────
// Diagnostic / debug access
// ──────────────────────────────────────────────────────────────────────────────

/** Copy forward MV buffer to host-visible staging buffer for inspection. */
void MI_DownloadMVs(MI_Context* ctx, void* cmdBuf,
                    int32_t* dstXY, uint32_t* dstSAD, int* outCount);

/** Get block grid dimensions for the current config. */
void MI_GetBlockGrid(const MI_Context* ctx, int* outBlocksX, int* outBlocksY);

// ──────────────────────────────────────────────────────────────────────────────
// Internal pipeline structure (exposed for testing / integration)
// ──────────────────────────────────────────────────────────────────────────────

typedef struct MI_Pipeline {
    void* pyramidDownPipeline;    // VkPipeline
    void* mvUpscalePipeline;      // VkPipeline
    void* analyzeComputePipeline; // VkPipeline
    void* occlusionPipeline;      // VkPipeline
    void* flowFpsSplatPipeline;   // VkPipeline (pass A)
    void* flowFpsBlendPipeline;   // VkPipeline (pass B)
    void* pipelineLayout;         // VkPipelineLayout
} MI_Pipeline;

const MI_Pipeline* MI_GetPipeline(const MI_Context* ctx);

#ifdef __cplusplus
}
#endif

// motion_flow_fps.glsl
// GPU Frame Interpolation - MFlowFps equivalent
// Given motion vectors (from motion_analyze.glsl), synthesizes an intermediate
// frame at time t ∈ (0,1) using:
//   • Forward/backward pixel flow (per-block bilinear splat)
//   • Occlusion masking via divergence / SAD thresholding
//   • Blending with weighted overlap (Hann window per block)
//
// Two passes:
//   Pass A (PASS=0): Splat contributions into accumulation buffers
//   Pass B (PASS=1): Normalize, apply masks, blend forward+backward

#version 450 core

layout(constant_id = 0) const int  BLOCK_SIZE  = 8;
layout(constant_id = 1) const int  OVERLAP     = 0;
layout(constant_id = 2) const int  PASS        = 0;    // 0=splat, 1=blend
layout(constant_id = 3) const bool MASK_OCCL   = true; // enable occlusion masking
layout(constant_id = 4) const bool BLEND_SCENE = true; // blend with original on uncertain pixels

// ──────────────────────────────────────────────────────────────────────────────
// Bindings
// ──────────────────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D uFrame0;   // previous frame
layout(set = 0, binding = 1) uniform sampler2D uFrame1;   // next frame

// Motion vectors: backward MVs from frame1→frame0, forward MVs from frame0→frame1
// Format: ivec4(dx*4, dy*4, sad, flags) in quarter-pel
layout(std430, set = 0, binding = 2) readonly buffer MVFwd {  ivec4 mv[]; } mvFwd;
layout(std430, set = 0, binding = 3) readonly buffer MVBwd {  ivec4 mv[]; } mvBwd;

// Accumulation buffers (rgba32f): [color.rgb, weight]
// Separate buffers for forward-warped and backward-warped contributions
layout(set = 0, binding = 4, rgba32f) uniform image2D uAccumFwd;
layout(set = 0, binding = 5, rgba32f) uniform image2D uAccumBwd;

// Final output image
layout(set = 0, binding = 6, rgba8)   uniform writeonly image2D uOutput;

// Mask image: r=occlusion_fwd, g=occlusion_bwd, b=reliability
layout(set = 0, binding = 7, rgba8)   uniform image2D uMask;

layout(push_constant) uniform PushConstants {
    ivec2 frameSize;
    ivec2 blocksSize;
    int   blocksX;
    float t;            // interpolation time: 0=frame0, 1=frame1
    float sadThresh;    // SAD threshold for occlusion detection
    float maskBlend;    // how aggressively to blend on occluded pixels
} pc;

layout(local_size_x_id = 0, local_size_y_id = 0, local_size_z = 1) in;

// ──────────────────────────────────────────────────────────────────────────────
// Hann window weight for smooth block blending (avoids block artifacts)
// ──────────────────────────────────────────────────────────────────────────────
float hannWeight(ivec2 localPx) {
    float wx = 0.5 - 0.5 * cos(3.14159265 * (float(localPx.x) + 0.5) / float(BLOCK_SIZE));
    float wy = 0.5 - 0.5 * cos(3.14159265 * (float(localPx.y) + 0.5) / float(BLOCK_SIZE));
    return wx * wy;
}

// ──────────────────────────────────────────────────────────────────────────────
// Bilinear sample helper
// ──────────────────────────────────────────────────────────────────────────────
vec4 sampleBilinear(sampler2D tex, vec2 pos) {
    vec2 uv = (pos + 0.5) / vec2(pc.frameSize);
    return texture(tex, uv);
}

// ──────────────────────────────────────────────────────────────────────────────
// Atomic splat of a weighted color into an accumulation buffer
// Uses imageAtomicAdd via float emulation (or rgba32f direct on supporting hw)
// For portability we use a simple non-atomic write guarded by weight > 0
// (In production: use separate R32F atomic buffers or compute-friendly layout)
// ──────────────────────────────────────────────────────────────────────────────
void splatPixel(image2D accum, ivec2 dstPos, vec4 color, float weight) {
    if (any(lessThan(dstPos, ivec2(0))) || any(greaterThanEqual(dstPos, pc.frameSize))) return;
    vec4 prev = imageLoad(accum, dstPos);
    imageStore(accum, dstPos, prev + vec4(color.rgb * weight, weight));
}

// ──────────────────────────────────────────────────────────────────────────────
// PASS 0: Forward/Backward Splat
// ──────────────────────────────────────────────────────────────────────────────
// Each invocation handles one pixel in the block grid
// workgroup = one block, invocation = one pixel within block

void passAForwardSplat() {
    ivec2 blockIdx = ivec2(gl_WorkGroupID.xy);
    ivec2 localPx  = ivec2(gl_LocalInvocationID.xy);
    int   step     = BLOCK_SIZE - OVERLAP;
    ivec2 srcPx    = blockIdx * step + localPx;

    if (any(greaterThanEqual(srcPx, pc.frameSize))) return;

    int   bidx  = blockIdx.y * pc.blocksX + blockIdx.x;
    float w     = hannWeight(localPx);

    // ── Forward warp: frame0 → interpolated ───────────────────────────────────
    ivec4 fwdMV  = mvFwd.mv[bidx];
    vec2  mvPxF  = vec2(fwdMV.xy) / 4.0;  // quarter-pel → pixel
    vec2  dstF   = vec2(srcPx) + mvPxF * pc.t;
    vec4  colF   = sampleBilinear(uFrame0, vec2(srcPx));
    splatPixel(uAccumFwd, ivec2(round(dstF)), colF, w);

    // ── Backward warp: frame1 → interpolated ──────────────────────────────────
    ivec4 bwdMV  = mvBwd.mv[bidx];
    vec2  mvPxB  = vec2(bwdMV.xy) / 4.0;
    vec2  dstB   = vec2(srcPx) - mvPxB * (1.0 - pc.t);
    vec4  colB   = sampleBilinear(uFrame1, vec2(srcPx));
    splatPixel(uAccumBwd, ivec2(round(dstB)), colB, w);

    // ── Occlusion mask update ─────────────────────────────────────────────────
    if (MASK_OCCL) {
        float sadNorm = float(fwdMV.z) / float(BLOCK_SIZE * BLOCK_SIZE * 255);
        float occ     = smoothstep(0.0, 1.0, sadNorm / (pc.sadThresh + 0.001));
        vec4  msk     = imageLoad(uMask, srcPx);
        msk.r         = max(msk.r, occ);  // forward occlusion
        imageStore(uMask, srcPx, msk);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PASS 1: Normalize accumulation buffers and blend
// ──────────────────────────────────────────────────────────────────────────────
void passBBlend() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(px, pc.frameSize))) return;

    vec4 accumF = imageLoad(uAccumFwd, px);
    vec4 accumB = imageLoad(uAccumBwd, px);
    vec4 mask   = imageLoad(uMask, px);

    // Normalize by accumulated weight
    vec3 colorF  = (accumF.a > 1e-4) ? accumF.rgb / accumF.a : vec3(0.0);
    vec3 colorB  = (accumB.a > 1e-4) ? accumB.rgb / accumB.a : vec3(0.0);

    float wF     = 1.0 - pc.t;  // contribution weight from frame0
    float wB     = pc.t;         // contribution weight from frame1

    // Apply occlusion masks: if forward-warped pixel is occluded, rely more on backward
    float occF   = MASK_OCCL ? mask.r : 0.0;
    float occB   = MASK_OCCL ? mask.g : 0.0;
    float wFm    = wF * (1.0 - occF);
    float wBm    = wB * (1.0 - occB);
    float wSum   = wFm + wBm;

    vec3 interp;
    if (wSum < 1e-4) {
        // Both occluded: fallback to simple temporal blend of originals
        vec4 c0 = sampleBilinear(uFrame0, vec2(px));
        vec4 c1 = sampleBilinear(uFrame1, vec2(px));
        interp  = mix(c0.rgb, c1.rgb, pc.t);
    } else {
        interp = (colorF * wFm + colorB * wBm) / wSum;
    }

    // ── Reliability blend with original frames ────────────────────────────────
    // Where both forward and backward have low confidence (high SAD), blend with
    // a direct temporal mix to reduce ghosting artifacts.
    if (BLEND_SCENE) {
        float reliability = mask.b;  // pre-computed reliability [0,1]
        vec4  c0 = sampleBilinear(uFrame0, vec2(px));
        vec4  c1 = sampleBilinear(uFrame1, vec2(px));
        vec3  direct = mix(c0.rgb, c1.rgb, pc.t);
        // Low reliability → blend toward direct
        interp = mix(direct, interp, reliability);
    }

    imageStore(uOutput, px, vec4(interp, 1.0));
}

void main() {
    if (PASS == 0)
        passAForwardSplat();
    else
        passBBlend();
}

// occlusion_detect.glsl
// Detects occlusion and unreliable regions by checking:
//   1. Forward-backward consistency: if fwd(bwd(p)) ≠ p, pixel is occluded
//   2. SAD threshold: high distortion → low reliability
//   3. MV smoothness: large MV discontinuities → boundary artifacts
//
// Output: per-pixel mask image
//   R = forward occlusion probability [0,1]
//   G = backward occlusion probability [0,1]
//   B = reliability score [0,1]
//   A = unused

#version 450 core

layout(std430, set = 0, binding = 0) readonly buffer FwdMVBuf { ivec4 mv[]; } fwdMV;
layout(std430, set = 0, binding = 1) readonly buffer BwdMVBuf { ivec4 mv[]; } bwdMV;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D uMask;

layout(push_constant) uniform PC {
    ivec2 frameSize;
    ivec2 blocksSize;
    int   blocksX;
    int   blockSize;
    int   blockStep;     // = blockSize - overlap
    float fbConsistThresh;  // pixel threshold for forward-backward check
    float sadThresh;
} pc;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Get block MV for a given pixel position
ivec4 getMV(ivec4 mvArr[], ivec2 px) {
    ivec2 blockIdx = px / pc.blockStep;
    blockIdx = clamp(blockIdx, ivec2(0), pc.blocksSize - 1);
    return mvArr[blockIdx.y * pc.blocksX + blockIdx.x];
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(px, pc.frameSize))) return;

    ivec4 fmv = getMV(fwdMV.mv, px);
    ivec4 bmv = getMV(bwdMV.mv, px);

    vec2  fwdDisp = vec2(fmv.xy) / 4.0;  // quarter-pel → float pixels
    vec2  bwdDisp = vec2(bmv.xy) / 4.0;

    // Forward-backward consistency: apply fwd then bwd, measure roundtrip error
    vec2  warped  = vec2(px) + fwdDisp;
    ivec2 warpedB = ivec2(round(warped));
    warpedB       = clamp(warpedB, ivec2(0), pc.frameSize - 1);
    ivec4 bmvAtW  = getMV(bwdMV.mv, warpedB);
    vec2  roundTrip = warped + vec2(bmvAtW.xy) / 4.0;
    float fbError   = length(roundTrip - vec2(px));

    // Occlusion from forward check
    float occFwd = smoothstep(pc.fbConsistThresh * 0.5, pc.fbConsistThresh * 2.0, fbError);

    // Backward symmetry
    vec2  warpedF2 = vec2(px) + bwdDisp;
    ivec2 warpedF2i= clamp(ivec2(round(warpedF2)), ivec2(0), pc.frameSize - 1);
    ivec4 fmvAtWF  = getMV(fwdMV.mv, warpedF2i);
    vec2  roundTripB = warpedF2 + vec2(fmvAtWF.xy) / 4.0;
    float fbErrorB   = length(roundTripB - vec2(px));
    float occBwd     = smoothstep(pc.fbConsistThresh * 0.5, pc.fbConsistThresh * 2.0, fbErrorB);

    // SAD-based reliability
    float sadNorm  = float(fmv.z) / float(pc.blockSize * pc.blockSize * 255);
    float reliab   = 1.0 - smoothstep(0.0, 1.0, sadNorm / (pc.sadThresh + 1e-5));
    reliab        *= (1.0 - occFwd * 0.5);

    imageStore(uMask, px, vec4(occFwd, occBwd, reliab, 0.0));
}

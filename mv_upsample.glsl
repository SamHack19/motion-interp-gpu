// mv_upsample.glsl
// Upscales motion vectors from a coarse pyramid level to a finer one.
// Each coarse MV covers 4 fine blocks; we copy + scale by 2 and store
// as predictors for the fine-level exhaustive search.
// Fine-level search uses these as starting candidates instead of (0,0).

#version 450 core

layout(std430, set = 0, binding = 0) readonly  buffer CoarseMV { ivec4 mv[]; } coarse;
layout(std430, set = 0, binding = 1) writeonly buffer FineMV   { ivec4 mv[]; } fine;

layout(push_constant) uniform PC {
    ivec2 coarseBlocks;  // number of blocks at coarse level
    ivec2 fineBlocks;    // number of blocks at fine level
    int   coarseBlocksX;
    int   fineBlocksX;
} pc;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    ivec2 fineBIdx = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(fineBIdx, pc.fineBlocks))) return;

    // Map fine block → coarse block (each coarse covers 2x2 fine blocks)
    ivec2 coarseBIdx = fineBIdx / 2;
    coarseBIdx = clamp(coarseBIdx, ivec2(0), pc.coarseBlocks - 1);

    int coarseLinear = coarseBIdx.y * pc.coarseBlocksX + coarseBIdx.x;
    ivec4 cmv = coarse.mv[coarseLinear];

    // Scale MV by 2 (doubling pixel resolution) – quarter-pel units unchanged in scale factor
    ivec4 upscaled = ivec4(cmv.xy * 2, cmv.z, cmv.w);

    int fineLinear = fineBIdx.y * pc.fineBlocksX + fineBIdx.x;
    fine.mv[fineLinear] = upscaled;
}

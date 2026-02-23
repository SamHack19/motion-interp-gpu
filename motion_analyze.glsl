// motion_analyze.glsl
// GPU Motion Vector Search - MAnalyze equivalent
// Implements hierarchical (multi-scale) block matching with
// diamond search, exhaustive refinement, and sad/satd cost functions.
//
// Dispatch: one workgroup per block in the frame.
//   gl_WorkGroupID.xy = block position in the block grid
//   gl_LocalInvocationID.x = candidate search thread (SEARCH_THREADS)

#version 450 core
#extension GL_KHR_shader_subgroup_arithmetic : require

// ──────────────────────────────────────────────────────────────────────────────
// Tunables (set via specialization constants)
// ──────────────────────────────────────────────────────────────────────────────
layout(constant_id = 0)  const int   BLOCK_SIZE    = 8;    // pixels (must be power-of-2, ≤32)
layout(constant_id = 1)  const int   SEARCH_RANGE  = 16;   // max displacement in pixels
layout(constant_id = 2)  const int   SEARCH_THREADS= 64;   // threads per workgroup
layout(constant_id = 3)  const bool  USE_SATD      = true; // Hadamard vs plain SAD
layout(constant_id = 4)  const bool  SUBPEL        = true; // half-pixel refinement
layout(constant_id = 5)  const int   OVERLAP       = 0;    // block overlap in pixels

// ──────────────────────────────────────────────────────────────────────────────
// Bindings
// ──────────────────────────────────────────────────────────────────────────────
// Luma planes (current/previous frame, possibly mip-mapped for hierarchy)
layout(set = 0, binding = 0) uniform sampler2D uCurrent;   // frame N   (luma)
layout(set = 0, binding = 1) uniform sampler2D uReference; // frame N-1 (luma)

// Output: packed (int16 dx, int16 dy, uint32 sad) per block
// Layout: [blockY * blocksX + blockX]
layout(std430, set = 0, binding = 2) writeonly buffer MVBuffer {
    // x = dx*4 (quarter-pel units), y = dy*4, z = cost, w = flags
    ivec4 mv[];
} outMV;

// Optional: initial predictors from coarser level (hierarchical ME)
layout(std430, set = 0, binding = 3) readonly buffer PredBuffer {
    ivec4 pred[];
} inPred;

// ──────────────────────────────────────────────────────────────────────────────
// Push constants
// ──────────────────────────────────────────────────────────────────────────────
layout(push_constant) uniform PushConstants {
    ivec2 frameSize;     // full-res frame dimensions
    ivec2 blocksSize;    // number of blocks (x,y)
    int   blocksX;
    int   level;         // pyramid level (0 = full-res)
    int   levelScale;    // 1 << level
    float lambda;        // rate-distortion lambda (for motion cost penalty)
    int   predCount;     // number of spatial/temporal predictors
} pc;

// ──────────────────────────────────────────────────────────────────────────────
// Shared memory
// ──────────────────────────────────────────────────────────────────────────────
// Cache the reference search window + current block in LDS
// Reference window: (BLOCK_SIZE + 2*SEARCH_RANGE) ^ 2, padded to 4-byte rows
#define REF_WIN   (BLOCK_SIZE + 2 * SEARCH_RANGE)
// Enough for 48x48 window at most (BLOCK_SIZE=16, RANGE=16)
shared float sRef[REF_WIN * REF_WIN];
shared float sCur[BLOCK_SIZE * BLOCK_SIZE];

// Per-candidate (cost, dx, dy) reduction buffer
shared uint  sBestCost[SEARCH_THREADS];
shared int   sBestDX  [SEARCH_THREADS];
shared int   sBestDY  [SEARCH_THREADS];

layout(local_size_x_id = 2, local_size_y = 1, local_size_z = 1) in;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

float sampleLuma(sampler2D tex, ivec2 pos, ivec2 frameSize) {
    vec2 uv = (vec2(pos) + 0.5) / vec2(frameSize);
    return texture(tex, uv).r;
}

// Load reference window into shared memory (cooperative, all threads participate)
void loadRefWindow(ivec2 blockOrigin) {
    int tid    = int(gl_LocalInvocationID.x);
    int total  = REF_WIN * REF_WIN;
    int stride = SEARCH_THREADS;
    for (int i = tid; i < total; i += stride) {
        int lx = i % REF_WIN;
        int ly = i / REF_WIN;
        ivec2 refPos = blockOrigin - ivec2(SEARCH_RANGE) + ivec2(lx, ly);
        // clamp to frame
        refPos = clamp(refPos, ivec2(0), pc.frameSize - 1);
        sRef[i] = sampleLuma(uReference, refPos, pc.frameSize);
    }
}

// Load current block into shared memory
void loadCurBlock(ivec2 blockOrigin) {
    int tid   = int(gl_LocalInvocationID.x);
    int total = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid; i < total; i += SEARCH_THREADS) {
        int lx = i % BLOCK_SIZE;
        int ly = i / BLOCK_SIZE;
        ivec2 pos = blockOrigin + ivec2(lx, ly);
        pos = clamp(pos, ivec2(0), pc.frameSize - 1);
        sCur[i] = sampleLuma(uCurrent, pos, pc.frameSize);
    }
}

// SAD using shared memory reference window
uint computeSAD(int dx, int dy) {
    // (dx, dy) in pixel units relative to block origin; window origin is -SEARCH_RANGE
    int wx = dx + SEARCH_RANGE;
    int wy = dy + SEARCH_RANGE;
    uint sad = 0u;
    for (int py = 0; py < BLOCK_SIZE; py++) {
        for (int px = 0; px < BLOCK_SIZE; px++) {
            float c = sCur[py * BLOCK_SIZE + px];
            float r = sRef[(wy + py) * REF_WIN + (wx + px)];
            sad += uint(abs(c - r) * 255.0 + 0.5);
        }
    }
    return sad;
}

// 4x4 SATD (Hadamard) sub-block accumulation for 8x8 or 16x16 blocks
float hadamard4(float b[4][4]) {
    float t[4][4];
    // horizontal
    for (int i = 0; i < 4; i++) {
        t[i][0] =  b[i][0] + b[i][1] + b[i][2] + b[i][3];
        t[i][1] =  b[i][0] + b[i][1] - b[i][2] - b[i][3];
        t[i][2] =  b[i][0] - b[i][1] - b[i][2] + b[i][3];
        t[i][3] =  b[i][0] - b[i][1] + b[i][2] - b[i][3];
    }
    float sum = 0.0;
    for (int j = 0; j < 4; j++) {
        float v0 =  t[0][j] + t[1][j] + t[2][j] + t[3][j];
        float v1 =  t[0][j] + t[1][j] - t[2][j] - t[3][j];
        float v2 =  t[0][j] - t[1][j] - t[2][j] + t[3][j];
        float v3 =  t[0][j] - t[1][j] + t[2][j] - t[3][j];
        sum += abs(v0) + abs(v1) + abs(v2) + abs(v3);
    }
    return sum / 2.0;
}

uint computeSATD(int dx, int dy) {
    int wx = dx + SEARCH_RANGE;
    int wy = dy + SEARCH_RANGE;
    uint satd = 0u;
    int nb = BLOCK_SIZE / 4;
    for (int by = 0; by < nb; by++) {
        for (int bx = 0; bx < nb; bx++) {
            float diff[4][4];
            for (int py = 0; py < 4; py++)
                for (int px = 0; px < 4; px++) {
                    float c = sCur[(by*4+py)*BLOCK_SIZE + (bx*4+px)];
                    float r = sRef[(wy+by*4+py)*REF_WIN + (wx+bx*4+px)];
                    diff[py][px] = (c - r) * 255.0;
                }
            satd += uint(hadamard4(diff));
        }
    }
    return satd;
}

uint computeCost(int dx, int dy) {
    uint dist = USE_SATD ? computeSATD(dx, dy) : computeSAD(dx, dy);
    // Motion vector cost penalty (similar to mvtools lambda weighting)
    uint mvcost = uint(pc.lambda * float(dx*dx + dy*dy));
    return dist + mvcost;
}

// ──────────────────────────────────────────────────────────────────────────────
// Diamond search pattern
// ──────────────────────────────────────────────────────────────────────────────
// Returns best (dx, dy) starting from a predictor, integer-pel
ivec2 diamondSearch(ivec2 startMV) {
    // Only thread 0 runs this serially (used for refinement after parallel scan)
    ivec2 best = startMV;
    uint  bestCost = computeCost(best.x, best.y);

    // Large diamond
    const ivec2 largeDiamond[8] = ivec2[8](
        ivec2(0,-2), ivec2(1,-1), ivec2(2,0), ivec2(1,1),
        ivec2(0, 2), ivec2(-1,1), ivec2(-2,0),ivec2(-1,-1)
    );
    bool improved = true;
    int maxIter = 8;
    while (improved && maxIter-- > 0) {
        improved = false;
        for (int k = 0; k < 8; k++) {
            ivec2 cand = best + largeDiamond[k];
            if (any(greaterThan(abs(cand), ivec2(SEARCH_RANGE)))) continue;
            uint c = computeCost(cand.x, cand.y);
            if (c < bestCost) { bestCost = c; best = cand; improved = true; }
        }
    }
    // Small diamond refinement
    const ivec2 smallDiamond[4] = ivec2[4](ivec2(0,-1),ivec2(1,0),ivec2(0,1),ivec2(-1,0));
    improved = true;
    maxIter  = 4;
    while (improved && maxIter-- > 0) {
        improved = false;
        for (int k = 0; k < 4; k++) {
            ivec2 cand = best + smallDiamond[k];
            if (any(greaterThan(abs(cand), ivec2(SEARCH_RANGE)))) continue;
            uint c = computeCost(cand.x, cand.y);
            if (c < bestCost) { bestCost = c; best = cand; improved = true; }
        }
    }
    return best;
}

// ──────────────────────────────────────────────────────────────────────────────
// Half-pixel refinement (bilinear interpolated SAD/SATD)
// ──────────────────────────────────────────────────────────────────────────────
// Uses hardware bilinear via texture sampler at half-pel offsets
// Returns mv in half-pel units (* 2 from integer pel)
ivec2 halfPelRefine(ivec2 intMV) {
    // Evaluate 8 half-pel neighbors around integer MV
    ivec2 bestHp = intMV * 2;
    // Sample with bilinear at half-pel offsets directly from texture
    uint bestCost = 0xFFFFFFFFu;
    ivec2 blockOrigin = ivec2(gl_WorkGroupID.xy) * (BLOCK_SIZE - OVERLAP);

    for (int hdy = -1; hdy <= 1; hdy++) {
        for (int hdx = -1; hdx <= 1; hdx++) {
            vec2 mvF = vec2(intMV) + vec2(hdx, hdy) * 0.5;
            if (any(greaterThan(abs(mvF), vec2(float(SEARCH_RANGE))))) continue;

            uint sad = 0u;
            for (int py = 0; py < BLOCK_SIZE; py++) {
                for (int px = 0; px < BLOCK_SIZE; px++) {
                    vec2 refUV = (vec2(blockOrigin + ivec2(px,py)) + mvF + 0.5) / vec2(pc.frameSize);
                    float r = texture(uReference, refUV).r;
                    float c = sCur[py * BLOCK_SIZE + px];
                    sad += uint(abs(c - r) * 255.0 + 0.5);
                }
            }
            uint mvcost = uint(pc.lambda * dot(mvF, mvF));
            uint cost = sad + mvcost;
            if (cost < bestCost) {
                bestCost = cost;
                bestHp   = ivec2(round(mvF * 2.0));
            }
        }
    }
    return bestHp;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────
void main() {
    int tid = int(gl_LocalInvocationID.x);

    ivec2 blockIdx    = ivec2(gl_WorkGroupID.xy);
    int   step        = BLOCK_SIZE - OVERLAP;
    ivec2 blockOrigin = blockIdx * step;

    if (any(greaterThanEqual(blockOrigin, pc.frameSize))) return;

    // ── 1. Cooperative load of reference window & current block ───────────────
    loadRefWindow(blockOrigin);
    loadCurBlock (blockOrigin);
    barrier();

    // ── 2. Parallel exhaustive search over search window ─────────────────────
    int searchDiam = 2 * SEARCH_RANGE + 1;
    int totalCands = searchDiam * searchDiam;

    uint myBestCost = 0xFFFFFFFFu;
    int  myBestDX   = 0;
    int  myBestDY   = 0;

    for (int cand = tid; cand < totalCands; cand += SEARCH_THREADS) {
        int dx = (cand % searchDiam) - SEARCH_RANGE;
        int dy = (cand / searchDiam) - SEARCH_RANGE;
        uint cost = computeCost(dx, dy);
        if (cost < myBestCost) {
            myBestCost = cost;
            myBestDX   = dx;
            myBestDY   = dy;
        }
    }

    sBestCost[tid] = myBestCost;
    sBestDX  [tid] = myBestDX;
    sBestDY  [tid] = myBestDY;
    barrier();

    // ── 3. Parallel reduction to find global best ─────────────────────────────
    for (int stride = SEARCH_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sBestCost[tid + stride] < sBestCost[tid]) {
                sBestCost[tid] = sBestCost[tid + stride];
                sBestDX  [tid] = sBestDX  [tid + stride];
                sBestDY  [tid] = sBestDY  [tid + stride];
            }
        }
        barrier();
    }

    // ── 4. Thread 0: diamond search refinement + optional half-pel ────────────
    if (tid == 0) {
        ivec2 intMV = diamondSearch(ivec2(sBestDX[0], sBestDY[0]));

        ivec4 result;
        if (SUBPEL) {
            ivec2 hpMV  = halfPelRefine(intMV);         // in half-pel units
            result       = ivec4(hpMV * 2, computeCost(intMV.x, intMV.y), 0); // stored as quarter-pel
        } else {
            result = ivec4(intMV * 4, computeCost(intMV.x, intMV.y), 0);
        }

        int blockLinear = blockIdx.y * pc.blocksX + blockIdx.x;
        outMV.mv[blockLinear] = result;
    }
}

// pyramid_downsample.glsl
// Builds a Gaussian image pyramid for hierarchical motion estimation.
// Each dispatch halves the resolution (13-tap Gaussian or simple box).
// Stored as R8_UNORM luma planes; set N+1 = downsample of set N.

#version 450 core

layout(constant_id = 0) const bool GAUSSIAN = true; // vs box filter

layout(set = 0, binding = 0) uniform sampler2D uSrc;
layout(set = 0, binding = 1, r8)    uniform writeonly image2D uDst;

layout(push_constant) uniform PC {
    ivec2 dstSize;
} pc;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 5-tap Gaussian weights (sum=1)
const float gk[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);

void main() {
    ivec2 dst = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(dst, pc.dstSize))) return;

    vec2  srcSize = vec2(pc.dstSize) * 2.0;
    float val     = 0.0;

    if (GAUSSIAN) {
        // Separable 5-tap Gaussian (samples at src pixel centers)
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                vec2  srcPx = vec2(dst * 2 + ivec2(dx, dy));
                vec2  uv    = (srcPx + 0.5) / srcSize;
                uv          = clamp(uv, 0.0, 1.0);
                val        += texture(uSrc, uv).r * gk[dx+2] * gk[dy+2];
            }
        }
    } else {
        // 2x2 box average
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                vec2 srcPx = vec2(dst * 2 + ivec2(dx, dy));
                vec2 uv    = (srcPx + 0.5) / srcSize;
                val       += texture(uSrc, uv).r * 0.25;
            }
        }
    }

    imageStore(uDst, dst, vec4(val));
}

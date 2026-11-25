// Meteor Shader Pass 2c — Valid-Mask Disocclusion Temporal Reflection Denoiser (0.5x)
module meteor_rt_reflect_temporal;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> rtReflectRaw;
Texture2D<float4> rtReflectHistory;
Texture2D<half4> rtGbufHistory;
Texture2D<float4> sceneDepthTex;
Texture2D<half4> rtGbufCurr;

float4 sanitizeFiniteReflect4(float4 v, float4 fallback) {
    bool bad = !isfinite(v.r) || !isfinite(v.g) || !isfinite(v.b) || !isfinite(v.a);
    return bad ? fallback : v;
}

[shader("fragment")]
float4 fsRtReflectTemporal(float4 pixelPos : SV_Position) : SV_Target0 {
    int2 rawDim = max(metality_tex_size(rtReflectRaw), int2(1, 1));
    int2 rawPx = clamp(int2(pixelPos.xy), int2(0, 0), rawDim - int2(1, 1));
    float2 uv = texelCenterUv(rawPx, rawDim);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    float4 rawCurr = sanitizeFiniteReflect4(rtReflectRaw.Load(int3(rawPx, 0)), float4(0.0f));
    if (rawCurr.a <= 0.01f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float cLuma = max(luminance(rawCurr.rgb), 0.0f);
    float3 curRgb = rawCurr.rgb * (1.0f / (1.0f + cLuma * 0.04f));
    float4 curr = float4(max(curRgb, float3(0.0f)), 1.0f);
    if (isSkyDepth(ds.depth) || (uFrame.tonemapMode & 1u) != 0u) {
        return curr;
    }

    float3 relPos = reconstructRelWorldPos(uFrame, ds.centerUv, ds.depth);
    float4 prevClip = mul(uFrame.prevViewProj, float4(relPos, 1.0f));
    if (prevClip.w <= 1e-4f) {
        return curr;
    }
    float2 prevUv = prevClip.xy / prevClip.w * 0.5f + 0.5f;
    if (prevUv.x < 0.002f || prevUv.x > 0.998f || prevUv.y < 0.002f || prevUv.y > 0.998f) {
        return curr;
    }

    half4 curGb = rtGbufCurr.Load(int3(rawPx, 0));
    float3 curN = float3(curGb.yzw);
    float4 prevGbuf = sanitizeFiniteReflect4(float4(rtGbufHistory.SampleLevel(metality_nearest_clamp, prevUv, 0.0f)), float4(0.0f));
    float depthErr = abs(prevGbuf.x - prevClip.w) / max(prevClip.w, 0.35f);
    float nDot = dot(prevGbuf.yzw, curN);
    bool disoccluded = (prevGbuf.x <= 0.005f) || (depthErr > 0.075f) || (nDot < 0.78f);

    float4 rawHist = sanitizeFiniteReflect4(rtReflectHistory.SampleLevel(metality_nearest_clamp, prevUv, 0.0f), float4(0.0f));
    if (disoccluded || rawHist.a <= 0.5f) {
        return curr;
    }

    float3 mean1 = float3(0.0f);
    float3 mean2 = float3(0.0f);
    float vCnt = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 nPx = clamp(rawPx + int2(dx, dy), int2(0, 0), rawDim - int2(1, 1));
            float4 s = sanitizeFiniteReflect4(rtReflectRaw.Load(int3(nPx, 0)), float4(0.0f));
            if (s.a > 0.5f) {
                float sL = max(luminance(s.rgb), 0.0f);
                float3 sc = max(s.rgb * (1.0f / (1.0f + sL * 0.04f)), float3(0.0f));
                mean1 += sc;
                mean2 += sc * sc;
                vCnt += 1.0f;
            }
        }
    }
    mean1 /= max(vCnt, 1.0f);
    mean2 /= max(vCnt, 1.0f);
    float3 sigma = sqrt(max(mean2 - mean1 * mean1, float3(0.0f)));
    float3 margin = max(sigma * 1.85f, float3(0.015f));
    float3 clampedHist = clamp(max(rawHist.rgb, float3(0.0f)), mean1 - margin, mean1 + margin);

    float motion = length(prevUv - uv) * 24.0f;
    float blendRate = clamp(0.14f + motion * 0.45f, 0.14f, 0.55f);
    return float4(max(lerp(clampedHist, curRgb, blendRate), float3(0.0f)), 1.0f);
}

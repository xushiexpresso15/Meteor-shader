// Meteor Shader Pass 2a — Geometric G-Buffer Disocclusion Temporal Reprojection + 3x3 Variance Clamp (0.5x MRT: GI + Tinted SunVis)
module meteor_rt_temporal;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> rtGiRaw;
Texture2D<float4> rtGiHistory;
Texture2D<half4> rtGbufHistory;
Texture2D<float4> sceneDepthTex;
Texture2D<float4> rtSunRaw;
Texture2D<float4> rtSunHistory;
Texture2D<half4> rtGbufCurr;

struct RtTemporalOut {
    float4 gi : SV_Target0;
    float4 sun : SV_Target1;
};

float4 sanitizeFinite4(float4 v, float4 fallback) {
    bool bad = !isfinite(v.r) || !isfinite(v.g) || !isfinite(v.b) || !isfinite(v.a);
    return bad ? fallback : v;
}

float3 sanitizeFinite3(float3 v, float3 fallback) {
    bool bad = !isfinite(v.r) || !isfinite(v.g) || !isfinite(v.b);
    return bad ? fallback : v;
}

[shader("fragment")]
RtTemporalOut fsRtTemporal(float4 pixelPos : SV_Position) {
    RtTemporalOut o;
    int2 rawDim = max(metality_tex_size(rtGiRaw), int2(1, 1));
    int2 rawPx = clamp(int2(pixelPos.xy), int2(0, 0), rawDim - int2(1, 1));
    float2 uv = texelCenterUv(rawPx, rawDim);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    float4 curr = sanitizeFinite4(rtGiRaw.Load(int3(rawPx, 0)), float4(0.0f, 0.0f, 0.0f, 1.0f));
    float3 currSun = max(sanitizeFinite3(rtSunRaw.Load(int3(rawPx, 0)).rgb, float3(0.0f, 0.0f, 0.0f)), float3(0.0f));
    if (isSkyDepth(ds.depth) || curr.a < -0.5f) {
        o.gi = curr;
        o.sun = float4(currSun, 1.0f);
        return o;
    }

    if ((uFrame.tonemapMode & 1u) != 0u) {
        o.gi = curr;
        o.sun = float4(currSun, 1.0f);
        return o;
    }

    float3 relPos = reconstructRelWorldPos(uFrame, ds.centerUv, ds.depth);
    float4 prevClip = mul(uFrame.prevViewProj, float4(relPos, 1.0f));
    if (prevClip.w <= 1e-4f) {
        o.gi = curr;
        o.sun = float4(currSun, 1.0f);
        return o;
    }
    float2 prevUv = prevClip.xy / prevClip.w * 0.5f + 0.5f;
    if (prevUv.x < 0.002f || prevUv.x > 0.998f || prevUv.y < 0.002f || prevUv.y > 0.998f) {
        o.gi = curr;
        o.sun = float4(currSun, 1.0f);
        return o;
    }

    half4 curGb = rtGbufCurr.Load(int3(rawPx, 0));
    float curZ = float(curGb.x);
    float3 curN = float3(curGb.yzw);
    float4 prevGbuf = sanitizeFinite4(float4(rtGbufHistory.SampleLevel(metality_nearest_clamp, prevUv, 0.0f)), float4(0.0f));
    float depthErr = abs(prevGbuf.x - prevClip.w) / max(prevClip.w, 0.35f);
    float nDot = dot(prevGbuf.yzw, curN);
    bool disoccluded = (prevGbuf.x <= 0.005f) || (depthErr > 0.075f) || (nDot < 0.78f);

    float zNeighborThresh = max(curZ * 0.085f, 0.16f);
    float4 mean1 = float4(0.0f);
    float4 mean2 = float4(0.0f);
    float3 sunM1 = float3(0.0f);
    float3 sunM2 = float3(0.0f);
    float cnt = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 nPx = clamp(rawPx + int2(dx, dy), int2(0, 0), rawDim - int2(1, 1));
            float4 s = sanitizeFinite4(rtGiRaw.Load(int3(nPx, 0)), curr);
            if (s.a < -0.5f) continue;
            if (dx != 0 || dy != 0) {
                half nZ = rtGbufCurr.Load(int3(nPx, 0)).x;
                if (nZ <= 0.005h || abs(float(nZ) - curZ) > zNeighborThresh) continue;
            }
            float3 ss = max(sanitizeFinite3(rtSunRaw.Load(int3(nPx, 0)).rgb, currSun), float3(0.0f));
            mean1 += s;
            mean2 += s * s;
            sunM1 += ss;
            sunM2 += ss * ss;
            cnt += 1.0f;
        }
    }
    float invCnt = 1.0f / max(cnt, 1.0f);
    mean1 *= invCnt;
    mean2 *= invCnt;
    sunM1 *= invCnt;
    sunM2 *= invCnt;

    float4 sigma = sqrt(max(mean2 - mean1 * mean1, float4(0.0f)));
    float3 sunSigma = sqrt(max(sunM2 - sunM1 * sunM1, float3(0.0f)));

    float motion = length(prevUv - uv) * 24.0f;
    bool stationary = (!disoccluded && motion < 0.05f);

    float3 rgbMargin = stationary
        ? max(sigma.rgb * 3.50f, float3(0.18f))
        : max(sigma.rgb * 2.15f, float3(0.025f));
    float aMargin = stationary
        ? max(sigma.a * 3.80f, 0.32f)
        : max(sigma.a * 2.85f, 0.14f);
    float4 boxMin = float4(mean1.rgb - rgbMargin, mean1.a - aMargin);
    float4 boxMax = float4(mean1.rgb + rgbMargin, mean1.a + aMargin);

    float3 sunMargin = stationary
        ? max(sunSigma * 3.80f, float3(0.32f))
        : max(sunSigma * 2.85f, float3(0.14f));
    float3 sunMin = max(sunM1 - sunMargin, float3(0.0f));
    float3 sunMax = sunM1 + sunMargin;

    float4 rawHist = sanitizeFinite4(rtGiHistory.SampleLevel(metality_linear_clamp, prevUv, 0.0f), curr);
    float3 rawSunHist = max(sanitizeFinite3(rtSunHistory.SampleLevel(metality_linear_clamp, prevUv, 0.0f).rgb, currSun), float3(0.0f));

    float4 hist = clamp(rawHist, boxMin, boxMax);
    float3 sunHist = clamp(rawSunHist, sunMin, sunMax);

    float blendRate = disoccluded ? 0.88f : clamp(0.045f + motion * 0.35f, 0.045f, 0.38f);
    o.gi = lerp(hist, curr, blendRate);
    o.sun = float4(max(lerp(sunHist, currSun, blendRate), float3(0.0f)), 1.0f);
    return o;
}

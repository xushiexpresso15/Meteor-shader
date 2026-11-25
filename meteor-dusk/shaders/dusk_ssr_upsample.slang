// Meteor Dusk Pass 4 — Valid-Mask Plane-Distance + Normal Bilateral SSR Upsample (0.5x -> 1.0x)
module meteor_dusk_ssr_upsample;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> duskReflRaw;
Texture2D<float4> sceneDepthTex;

[shader("fragment")]
float4 fsDuskSsrUpsample(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    float centerD = sceneDepthTex.SampleLevel(metality_nearest_clamp, uv, 0.0f).r;
    if (isSkyDepth(centerD)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 hStep = float2(1.5f / max(float(uFrame.width), 1.0f), 1.5f / max(float(uFrame.height), 1.0f));
    float4 centerVal = duskReflRaw.SampleLevel(metality_nearest_clamp, uv, 0.0f);
    float aL = duskReflRaw.SampleLevel(metality_nearest_clamp, clamp(uv - float2(hStep.x, 0.0f), float2(0.001f), float2(0.999f)), 0.0f).a;
    float aR = duskReflRaw.SampleLevel(metality_nearest_clamp, clamp(uv + float2(hStep.x, 0.0f), float2(0.001f), float2(0.999f)), 0.0f).a;
    float aD = duskReflRaw.SampleLevel(metality_nearest_clamp, clamp(uv - float2(0.0f, hStep.y), float2(0.001f), float2(0.999f)), 0.0f).a;
    float aU = duskReflRaw.SampleLevel(metality_nearest_clamp, clamp(uv + float2(0.0f, hStep.y), float2(0.001f), float2(0.999f)), 0.0f).a;
    if (max(centerVal.a, max(max(aL, aR), max(aD, aU))) <= 0.01f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float3 centerRel = reconstructRelWorldPos(uFrame, uv, centerD);
    float centerDist = length(centerRel);
    float3 centerN = reconstructFastNormal(uFrame, uv, centerRel, sceneDepthTex);
    float planeTol = max(0.045f + centerDist * 0.007f, 0.045f);
    float invPlaneTol2 = 1.0f / (planeTol * planeTol);
    float maxDeltaD = max(centerDist * 0.20f, 0.65f);

    bool centerValid = (centerVal.a > 0.01f);
    float3 accum = centerValid ? (max(centerVal.rgb, float3(0.0f)) * centerVal.a * 1.25f) : float3(0.0f);
    float confAccum = centerValid ? (centerVal.a * 1.25f) : 0.0f;
    float wSum = centerValid ? 1.25f : 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 suv = clamp(uv + float2(float(dx), float(dy)) * hStep, float2(0.001f), float2(0.999f));
            float4 sVal = duskReflRaw.SampleLevel(metality_nearest_clamp, suv, 0.0f);
            if (sVal.a > 0.01f) {
                float sd = sceneDepthTex.SampleLevel(metality_nearest_clamp, suv, 0.0f).r;
                if (!isSkyDepth(sd)) {
                    float3 sRel = reconstructRelWorldPos(uFrame, suv, sd);
                    float sDist = length(sRel);
                    float planeDist = abs(dot(sRel - centerRel, centerN));
                    float planeW = exp(-(planeDist * planeDist) * invPlaneTol2);
                    float distGuard = exp(-max(abs(sDist - centerDist) - maxDeltaD, 0.0f) * 3.0f);
                    float w = 0.55f * planeW * distGuard * sVal.a;
                    accum += max(sVal.rgb, float3(0.0f)) * w;
                    confAccum += sVal.a * w;
                    wSum += w;
                }
            }
        }
    }

    if (wSum > 1e-4f) {
        float conf = clamp(confAccum / wSum, 0.0f, 1.0f);
        return float4(max(accum / wSum, float3(0.0f)), conf);
    }
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

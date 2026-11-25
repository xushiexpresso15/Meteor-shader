// Meteor Shader Pass 2d — Final Plane-Distance + Normal Bilateral GI & Soft Shadow Upsample Filter (0.5x -> 1.0x)
module meteor_rt_gi_filter;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> rtGiStep2;
Texture2D<float4> sceneDepthTex;

[shader("fragment")]
float4 fsRtGiFilter(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    float centerD = sceneDepthTex.SampleLevel(metality_nearest_clamp, uv, 0.0f).r;
    if (isSkyDepth(centerD)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float3 centerRel = reconstructRelWorldPos(uFrame, uv, centerD);
    float centerDist = length(centerRel);
    float3 centerN = reconstructFastNormal(uFrame, uv, centerRel, sceneDepthTex);
    float planeTol = max(0.055f + centerDist * 0.008f, 0.055f);
    float invPlaneTol2 = 1.0f / (planeTol * planeTol);
    float maxDeltaD = max(centerDist * 0.25f, 0.80f);

    float2 hStep = float2(1.5f / max(float(uFrame.width), 1.0f), 1.5f / max(float(uFrame.height), 1.0f));
    float4 centerVal = rtGiStep2.SampleLevel(metality_linear_clamp, uv, 0.0f);
    float4 accum = centerVal * 1.10f;
    float wSum = 1.10f;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 suv = clamp(uv + float2(float(dx), float(dy)) * hStep, float2(0.001f), float2(0.999f));
            float sd = sceneDepthTex.SampleLevel(metality_nearest_clamp, suv, 0.0f).r;
            if (!isSkyDepth(sd)) {
                float3 sRel = reconstructRelWorldPos(uFrame, suv, sd);
                float sDist = length(sRel);
                float planeDist = abs(dot(sRel - centerRel, centerN));
                float planeW = exp(-(planeDist * planeDist) * invPlaneTol2);
                float distGuard = exp(-max(abs(sDist - centerDist) - maxDeltaD, 0.0f) * 2.5f);
                float spatialW = (dx == 0 || dy == 0) ? 0.68f : 0.42f;
                float w = spatialW * planeW * distGuard;
                accum += rtGiStep2.SampleLevel(metality_linear_clamp, suv, 0.0f) * w;
                wSum += w;
            }
        }
    }

    return accum / max(wSum, 1e-4f);
}

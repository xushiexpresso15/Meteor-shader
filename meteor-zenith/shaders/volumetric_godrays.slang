// Meteor Shader Pass 3 — Quarter-Res Additive Crepuscular Sunbeams, Cloud Shafts & Gargantua Godrays
module meteor_volumetric_godrays;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> sceneDepthTex;
Texture2D<half4> cloudsQres;

[shader("fragment")]
float4 fsVolumetricGodrays(float4 pixelPos : SV_Position) : SV_Target0 {
    float godrayStrength = clamp(uFrame.userParams.y, 0.0f, 2.0f);
    if (!meteorFeatureGodrays(uFrame) || godrayStrength < 0.01f || uFrame.submergedMode == 2) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    float3 viewDir = reconstructSkyWorldDir(uFrame, uv);

    float3 lightDir = normalize(uFrame.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 shaftTint = lerp(float3(1.06f, 0.90f, 0.64f), float3(2.05f, 0.84f, 0.26f), env.sunsetFactor);
    float activeIntensity = (1.0f - env.rainFactor * 0.78f);

    if (uFrame.dimensionId == 0) {
        if (lightDir.y < -0.08f) {
            lightDir = normalize(uFrame.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
            shaftTint = float3(0.24f, 0.37f, 0.65f);
            activeIntensity *= 0.28f * smoothstep(-0.05f, 0.16f, lightDir.y);
        } else {
            activeIntensity *= smoothstep(-0.08f, 0.06f, lightDir.y) * lerp(0.85f, 1.40f, env.sunsetFactor);
        }
        if (uFrame.submergedMode == 1) {
            shaftTint *= float3(0.45f, 0.92f, 1.05f);
            activeIntensity *= 1.15f;
        }
    } else if (uFrame.dimensionId == 1) {
        lightDir = getMeteorBlackHoleDir();
        shaftTint = float3(0.95f, 0.56f, 1.45f);
        activeIntensity = 0.72f;
    } else {
        lightDir = normalize(float3(0.20f, 0.62f, -0.76f));
        float3 crimsonShaft = float3(1.15f, 0.34f, 0.08f);
        float3 warpedShaft = float3(0.18f, 0.74f, 0.68f);
        float3 soulShaft = float3(0.26f, 0.86f, 0.86f);
        shaftTint = lerp(lerp(crimsonShaft, warpedShaft, env.netherWarpedFactor), soulShaft, env.netherSoulFactor);
        activeIntensity = 0.34f;
    }

    if (activeIntensity <= 0.01f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Early-out BEFORE depth/cloud fetches when looking away from light source
    float facing = dot(viewDir, lightDir);
    if (facing <= 0.20f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float2 lightUv = worldToScreenUv(uFrame, lightDir * 300.0f);
    float2 delta = uv - lightUv;
    float deltaLen = length(delta);
    float offScreenFade = 1.0f - smoothstep(0.75f, 1.75f, deltaLen);
    if (offScreenFade <= 0.01f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // 4-tap 2x2 depth footprint removes 0.25x nearest-depth aliasing on thin glass panes, bars, and silhouettes
    float2 qTexel = float2(1.0f / max(float(uFrame.width), 1.0f), 1.0f / max(float(uFrame.height), 1.0f));
    float2 dOff = qTexel * 0.35f;
    float d0 = sceneDepthTex.SampleLevel(metality_nearest_clamp, clamp(uv + float2(-dOff.x, -dOff.y), 0.001f, 0.999f), 0.0f).r;
    float d1 = sceneDepthTex.SampleLevel(metality_nearest_clamp, clamp(uv + float2( dOff.x, -dOff.y), 0.001f, 0.999f), 0.0f).r;
    float d2 = sceneDepthTex.SampleLevel(metality_nearest_clamp, clamp(uv + float2(-dOff.x,  dOff.y), 0.001f, 0.999f), 0.0f).r;
    float d3 = sceneDepthTex.SampleLevel(metality_nearest_clamp, clamp(uv + float2( dOff.x,  dOff.y), 0.001f, 0.999f), 0.0f).r;
    float skyCount = 0.0f;
    float geomDistSum = 0.0f;
    float geomCount = 0.0f;
    if (isSkyDepth(d0)) { skyCount += 1.0f; } else { geomDistSum += length(reconstructRelWorldPos(uFrame, uv, d0)); geomCount += 1.0f; }
    if (isSkyDepth(d1)) { skyCount += 1.0f; } else { geomDistSum += length(reconstructRelWorldPos(uFrame, uv, d1)); geomCount += 1.0f; }
    if (isSkyDepth(d2)) { skyCount += 1.0f; } else { geomDistSum += length(reconstructRelWorldPos(uFrame, uv, d2)); geomCount += 1.0f; }
    if (isSkyDepth(d3)) { skyCount += 1.0f; } else { geomDistSum += length(reconstructRelWorldPos(uFrame, uv, d3)); geomCount += 1.0f; }
    float skyFrac = skyCount * 0.25f;
    float geomDist = (geomCount > 0.5f) ? (geomDistSum / geomCount) : 280.0f;

    bool useCloudOcc = (uFrame.dimensionId == 0) && meteorFeatureClouds(uFrame);

    // 24-step radial crepuscular ray-march (0.25x res, centered midpoint + subtle isotropic jitter = zero diagonal stripes!)
    int NUM_STEPS = 24;
    float marchFrac = min(0.92f, 0.92f / max(deltaLen, 1.0f));
    float2 stepUv = delta * (marchFrac / float(NUM_STEPS));
    float2 r2Hash = pcg2dHash01(uint2(max(floor(pixelPos.xy), float2(0.0f, 0.0f))));
    float dither = 0.25f + r2Hash.x * 0.50f;
    float2 curUv = uv - stepUv * dither;

    float accumShaft = 0.0f;
    float weightSum = 0.0f;
    float decay = 1.0f;
    for (int i = 0; i < 24; ++i) {
        curUv -= stepUv;
        float2 clamped = clamp(curUv, float2(0.002f, 0.002f), float2(0.998f, 0.998f));
        float d = sceneDepthTex.SampleLevel(metality_nearest_clamp, clamped, 0.0f).r;
        float vis = 0.0f;
        if (isSkyDepth(d)) {
            vis = 1.0f;
            if (useCloudOcc) {
                float cAlpha = float(cloudsQres.SampleLevel(metality_linear_clamp, clamped, 0.0f).a);
                vis = 1.0f - cAlpha * 0.88f;
            }
        }
        accumShaft += vis * decay;
        weightSum += decay;
        decay *= 0.94f;
    }

    float shaftVis = accumShaft / max(weightSum, 1e-4f);
    if (shaftVis <= 0.005f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    shaftVis = pow(saturate(shaftVis), 1.15f);

    float cosPos = saturate(facing);
    float angularPhase = pow(cosPos, 4.0f) * 0.42f + pow(cosPos, 16.0f) * 0.58f;
    if (uFrame.dimensionId == 1) {
        float angToBh = acos(clamp(facing, -1.0f, 1.0f));
        angularPhase *= smoothstep(0.165f, 0.235f, angToBh);
    }

    float centerCloudAlpha = (skyFrac > 0.01f && useCloudOcc)
        ? float(cloudsQres.SampleLevel(metality_linear_clamp, uv, 0.0f).a)
        : 0.0f;
    float skyWeight = lerp(0.20f, 0.72f, saturate(centerCloudAlpha * 1.25f));
    float geomOnlyWeight = smoothstep(3.0f, 26.0f, geomDist) * 0.80f;
    float geomWeight = lerp(geomOnlyWeight, skyWeight, skyFrac);

    float rayEnergy = shaftVis * angularPhase * offScreenFade * activeIntensity * godrayStrength * geomWeight;
    return float4(max(shaftTint * rayEnergy * 0.38f, float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

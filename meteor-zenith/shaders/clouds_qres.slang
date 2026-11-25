// Meteor Shader Shared Pass — Quarter-Res Volumetric Clouds (RGBA16F: rgb=radiance, a=cloudAlpha=1-cloudTrans)
module meteor_clouds_qres;
import metality;

#include "lib/meteor_common.slang"
#include "lib/meteor_sky_end_nether.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> sceneDepthTex;

[shader("fragment")]
float4 fsCloudsQres(float4 pixelPos : SV_Position) : SV_Target0 {
    if (uFrame.dimensionId != 0 || !meteorFeatureClouds(uFrame)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    float3 rayDir = reconstructSkyWorldDir(uFrame, uv);
    if (rayDir.y < 0.015f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Dilated 5-tap sky mask (+/-4 full-res px) so the 5-tap tent upsample never bleeds black halos along silhouettes
    int2 dDim = max(metality_tex_size(sceneDepthTex), int2(1, 1));
    int2 maxD = dDim - int2(1, 1);
    int2 cpx = snapDepthCoord(dDim, uv);
    bool anySky = isSkyDepth(sceneDepthTex.Load(int3(cpx, 0)).r);
    if (!anySky) {
        int2 p0 = clamp(cpx + int2(-4, -4), int2(0, 0), maxD);
        int2 p1 = clamp(cpx + int2( 4, -4), int2(0, 0), maxD);
        int2 p2 = clamp(cpx + int2(-4,  4), int2(0, 0), maxD);
        int2 p3 = clamp(cpx + int2( 4,  4), int2(0, 0), maxD);
        anySky = isSkyDepth(sceneDepthTex.Load(int3(p0, 0)).r)
              || isSkyDepth(sceneDepthTex.Load(int3(p1, 0)).r)
              || isSkyDepth(sceneDepthTex.Load(int3(p2, 0)).r)
              || isSkyDepth(sceneDepthTex.Load(int3(p3, 0)).r);
    }
    if (!anySky) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    uint tier = meteorShadowTier(uFrame);
    int steps = (tier >= 3u) ? 10 : ((tier >= 2u) ? 8 : 6);
    CloudTraceResult c = sampleVolumetricCloudsNAt(rayDir, pixelPos.xy, uFrame, env, steps, tier < 3u);
    float cloudAlpha = saturate(1.0f - c.transmittance);
    return float4(max(c.radiance, float3(0.0f, 0.0f, 0.0f)), cloudAlpha);
}

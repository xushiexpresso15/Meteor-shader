// Meteor Shader Shared Pass — Hillaire (2020) 192x96 Precomputed Sky-View & Sky-Reflection LUT
module meteor_sky_lut;
import metality;

#include "lib/meteor_common.slang"
#include "lib/meteor_sky_end_nether.slang"

ConstantBuffer<MetalityFrame> uFrame;

struct SkyLutOut {
    float4 viewLut : SV_Target0;
    float4 reflLut : SV_Target1;
};

[shader("fragment")]
SkyLutOut fsSkyLut(float4 pixelPos : SV_Position) {
    SkyLutOut o;
    float2 dim = max(float2(float(uFrame.width), float(uFrame.height)), float2(1.0f, 1.0f));
    float2 uv = pixelPos.xy / dim;
    float3 d = skyLutUvToDir(uv);
    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);

    if (uFrame.dimensionId == -1) {
        float3 nether = max(evalNetherSky(d, uFrame, env), float3(0.0f, 0.0f, 0.0f));
        o.viewLut = float4(nether, 1.0f);
        o.reflLut = float4(nether, 1.0f);
        return o;
    }

    if (uFrame.dimensionId == 1) {
        float3 endSky = max(evalEndSkyAndGargantua(d, uFrame), float3(0.0f, 0.0f, 0.0f));
        o.viewLut = float4(endSky, 1.0f);
        o.reflLut = float4(endSky, 1.0f);
        return o;
    }

    // Keep 192x96 RGBA16F Sky-View LUT strictly for smooth low-frequency atmospheric gradients
    // (Milky Way, Aurora, Solar Mie halo, and celestials render crisply at 1.0x with zero zenith pinch/banding)
    float3 sky = evalOverworldSkyBase(d, uFrame, env, false);

    float3 reflSky = sky;
    if (meteorFeatureClouds(uFrame) && d.y >= 0.018f) {
        CloudTraceResult c = sampleVolumetricCloudsN(d, uFrame, env, 6, true);
        reflSky = sky * c.transmittance + c.radiance;
    }

    o.viewLut = float4(max(sky, float3(0.0f, 0.0f, 0.0f)), 1.0f);
    o.reflLut = float4(max(reflSky, float3(0.0f, 0.0f, 0.0f)), 1.0f);
    return o;
}

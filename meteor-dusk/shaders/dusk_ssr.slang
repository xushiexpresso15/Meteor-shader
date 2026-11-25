// Meteor Dusk Pass 3 — Hierarchical Binary-Refined Screen-Space Reflections (SSR) for Water, Metals, Stained Glass & Rain Puddles (0.5x, Zero RT)
module meteor_dusk_ssr;
import metality;

#include "lib/meteor_common.slang"
#include "lib/meteor_sky_end_nether.slang"
#include "lib/meteor_water.slang"

ConstantBuffer<MetalityFrame> uFrame;
StructuredBuffer<uint> voxelGrid;
StructuredBuffer<float4> emissiveLights;
StructuredBuffer<float4> waterRipples;
Texture2D<float4> sceneColorTex;
Texture2D<float4> sceneDepthTex;
Texture2D<float4> duskLightingFiltered;
Texture2D<float4> duskSunFiltered;
Texture2D<half4> skyReflLut;

float4 traceBinaryRefinedSsr(
    MetalityFrame f,
    Texture2D<float4> colorTex,
    Texture2D<float4> depthTex,
    Texture2D<float4> lightingTex,
    Texture2D<float4> sunTex,
    float3 startRel,
    float3 reflDir,
    float3 reflNrm,
    float minPlaneDot,
    float ign,
    int maxSteps,
    int refineSteps,
    float minHitRelY
) {
    float3 reflView = mul(f.viewRot, float4(reflDir, 0.0f)).xyz;
    float backFade = smoothstep(-0.35f, 0.15f, -reflView.z);
    if (backFade <= 0.01f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float3 startView = mul(f.viewRot, float4(startRel, 0.0f)).xyz;
    float3 reflViewNrm = mul(f.viewRot, float4(reflNrm, 0.0f)).xyz;
    bool enforceMinY = (minHitRelY > -1e5f);

    float3 hitUvDepth = float3(-1.0f, -1.0f, 0.0f);
    float hitThickConf = 1.0f;
    bool found = false;
    float prevT = 0.0f;
    float curT = 0.14f + ign * 0.10f;
    int2 depthDim = max(metality_tex_size(depthTex), int2(1, 1));

    for (int i = 0; i < 20; ++i) {
        if (i >= maxSteps) break;
        float3 pView = startView + reflView * curT;
        float4 pClip = mul(f.proj, float4(pView, 1.0f));
        if (pClip.w <= 1e-4f) break;
        float2 pUv = pClip.xy / pClip.w * 0.5f + 0.5f;
        if (pUv.x <= 0.002f || pUv.x >= 0.998f || pUv.y <= 0.002f || pUv.y >= 0.998f) break;

        int2 gPx = snapDepthCoord(depthDim, pUv);
        float gD = depthTex.Load(int3(gPx, 0)).r;
        if (!isSkyDepth(gD)) {
            float3 gView = reconstructViewPos(f, texelCenterUv(gPx, depthDim), gD);
            bool yOk = !enforceMinY || (mul(f.invViewRot, float4(gView, 0.0f)).y >= minHitRelY);
            if (yOk && dot(gView - startView, reflViewNrm) >= minPlaneDot) {
                float depthDiff = (-pView.z) - (-gView.z);
                float stepLen = curT - prevT;
                float thickness = clamp(max(0.55f, stepLen * 1.15f + curT * 0.03f), 0.55f, 2.40f);
                if (depthDiff > 0.015f && depthDiff < thickness) {
                    float lo = prevT;
                    float hi = curT;
                    float2 bestUv = pUv;
                    for (int b = 0; b < 3; ++b) {
                        if (b >= refineSteps) break;
                        float mid = 0.5f * (lo + hi);
                        float3 mView = startView + reflView * mid;
                        float4 mClip = mul(f.proj, float4(mView, 1.0f));
                        if (mClip.w <= 1e-4f) break;
                        float2 mUv = mClip.xy / mClip.w * 0.5f + 0.5f;
                        if (mUv.x <= 0.002f || mUv.x >= 0.998f || mUv.y <= 0.002f || mUv.y >= 0.998f) break;
                        int2 mPx = snapDepthCoord(depthDim, mUv);
                        float mD = depthTex.Load(int3(mPx, 0)).r;
                        if (!isSkyDepth(mD)) {
                            float3 mgView = reconstructViewPos(f, texelCenterUv(mPx, depthDim), mD);
                            float md = (-mView.z) - (-mgView.z);
                            bool myOk = !enforceMinY || (mul(f.invViewRot, float4(mgView, 0.0f)).y >= minHitRelY);
                            if (myOk && dot(mgView - startView, reflViewNrm) >= minPlaneDot && md > 0.01f) {
                                hi = mid;
                                bestUv = mUv;
                            } else {
                                lo = mid;
                            }
                        } else {
                            lo = mid;
                        }
                    }
                    hitUvDepth = float3(bestUv, curT);
                    hitThickConf = 1.0f - smoothstep(thickness * 0.55f, thickness, depthDiff);
                    found = true;
                    break;
                }
            }
        }
        prevT = curT;
        curT += 0.24f + curT * 0.145f;
    }

    if (!found) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 hUv = hitUvDepth.xy;
    float3 hitAlb = srgbToLinear(colorTex.SampleLevel(metality_linear_clamp, hUv, 0.0f).rgb);
    float4 hitLt = lightingTex.SampleLevel(metality_nearest_clamp, hUv, 0.0f);
    float3 hitSunRgb = sunTex.SampleLevel(metality_nearest_clamp, hUv, 0.0f).rgb;
    if (hitLt.a < -0.5f) {
        hitLt = float4(0.0f, 0.0f, 0.0f, 0.0f);
        hitSunRgb = float3(0.0f, 0.0f, 0.0f);
    }
    MeteorEnvFactors env = analyzeMeteorEnv(f);
    float3 directCol = lerp(float3(2.75f, 2.46f, 2.04f), float3(3.55f, 1.44f, 0.48f), env.sunsetFactor);
    directCol = lerp(directCol, float3(0.22f, 0.34f, 0.62f), env.nightFactor);
    directCol = lerp(directCol, float3(0.58f, 0.64f, 0.72f), env.rainFactor * 0.82f);
    if (f.dimensionId == -1) directCol = float3(0.0f);
    else if (f.dimensionId == 1) directCol = float3(1.65f, 1.25f, 2.05f);

    float3 hitRadiance = hitAlb * (directCol * hitSunRgb * 0.75f + hitLt.rgb * 1.15f + float3(0.004f, 0.006f, 0.009f));
    float hitLum = luminance(hitRadiance);
    if (hitLum > 4.0f) {
        hitRadiance *= 4.0f / hitLum;
    }

    float2 edge = abs(hUv * 2.0f - 1.0f);
    float edgeFade = saturate(1.0f - smoothstep(0.76f, 0.99f, max(edge.x, edge.y)));
    float distFade = exp(-hitUvDepth.z * 0.012f);
    float conf = clamp(edgeFade * distFade * hitThickConf * backFade, 0.0f, 1.0f);
    return float4(max(hitRadiance, float3(0.0f)), conf);
}

[shader("fragment")]
float4 fsDuskSsr(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    float depth = ds.depth;
    float3 rawSrgb = sceneColorTex.Load(int3(ds.px, 0)).rgb;
    float3 rawLin = srgbToLinear(rawSrgb);
    bool isSky = isSkyDepth(depth);

    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    float3 rayDir = reconstructSkyWorldDir(uFrame, ds.centerUv);
    float3 camPos = uFrame.camWorldPos.xyz;
    int3 vOrg = uFrame.voxelOrigin.xyz;
    float3 relPos = isSky ? (rayDir * 280.0f) : reconstructRelWorldPos(uFrame, ds.centerUv, depth);
    float3 worldPos = camPos + relPos;
    float terrainT = isSky ? 1e6f : length(relPos);

    float3 surfNormal = isSky ? float3(0.0f, 1.0f, 0.0f) : reconstructFastNormal(uFrame, ds.centerUv, relPos, sceneDepthTex);
    uint activeTag = 0u;
    uint activeSub = 0u;
    if (!isSky && uFrame.voxelOrigin.w != 0) {
        MeteorSurfaceVoxel sv = sampleMeteorSurfaceVoxel(voxelGrid, vOrg, worldPos, rayDir, surfNormal);
        activeTag = sv.activeTag;
        activeSub = sv.activeSub;
    }
    bool isGlassSurface = (!isSky && activeTag == 8u);

    bool isWater = false;
    bool isWaterfallFace = false;
    bool waterDdaBlocked = false;
    float tWater = 1e6f;
    float3 waterGeoNrm = float3(0.0f, 1.0f, 0.0f);
    float2 waterFlowVec = float2(0.0f, 0.0f);
    float seaSurfaceY = uFrame.seaLevel - 0.112f;
    int seaBlockY = int(floor(uFrame.seaLevel)) - 1;
    bool canSampleVoxels = ((uFrame.voxelOrigin.w & 1) != 0);
    bool has3dWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & 2) != 0);
    bool hasWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & (2 | 8)) != 0);

    if (meteorFeatureWater(uFrame) && hasWaterVoxels && uFrame.submergedMode == 0) {
        if (uFrame.dimensionId == 0 && camPos.y > seaSurfaceY + 0.01f && rayDir.y < -0.0005f) {
            float tSea = (seaSurfaceY - camPos.y) / rayDir.y;
            if (tSea > 0.04f && tSea < terrainT - 0.015f) {
                float3 probe = camPos + rayDir * tSea - float3(0.0f, 0.08f, 0.0f);
                int3 vc = int3(floor(probe)) - vOrg;
                if (has3dWaterVoxels && inVoxelBounds512Common(vc)) {
                    uint idx = voxelLinearIndex512Common(vc);
                    uint tag = ((voxelGrid[idx >> 2u] >> ((idx & 3u) * 8u)) >> 4u) & 0xFu;
                    if (tag == 3u) {
                        isWater = true;
                        tWater = tSea;
                    }
                } else if (!isGlassSurface && isFarOrHighSeaWaterAt(voxelGrid, vOrg, probe, seaBlockY)) {
                    isWater = true;
                    tWater = tSea;
                }
            }
        }
        if (!isWater && has3dWaterVoxels && rayOverlapsWaterYBounds(uFrame, camPos.y, worldPos.y, rayDir, isSky)) {
            WaterDdaHit wh = traceWaterMacroDda(voxelGrid, vOrg, camPos, rayDir, 0.04f, isSky ? 224.0f : terrainT, uFrame.voxelOrigin.w);
            waterDdaBlocked = wh.blocked;
            if (wh.hit) {
                isWater = true;
                isWaterfallFace = wh.isVertical;
                tWater = wh.t;
                waterGeoNrm = wh.nrm;
                waterFlowVec = wh.flowVec;
            }
        }
        if (!isWater && !waterDdaBlocked && !isGlassSurface && !isSky && rayDir.y < -0.0005f) {
            int twY = sampleTopWaterWorldY(voxelGrid, vOrg, int2(floor(worldPos.xz)));
            if (twY > -9000 && float(twY) + 0.888f > worldPos.y + 0.05f && camPos.y > float(twY) + 0.90f) {
                float lakeSurfY = float(twY) + 0.888f;
                float tLake = (lakeSurfY - camPos.y) / rayDir.y;
                if (tLake > 0.04f && tLake < terrainT - 0.015f) {
                    float3 lakeHit = camPos + rayDir * tLake;
                    if (!inVoxelBounds512Common(int3(floor(lakeHit)) - vOrg)) {
                        int2 lakeXZ = int2(floor(lakeHit.xz));
                        int twHit = sampleTopWaterWorldY(voxelGrid, vOrg, lakeXZ);
                        int topOpHit = sampleTopOpaqueWorldY(voxelGrid, lakeXZ.x - vOrg.x, lakeXZ.y - vOrg.z);
                        if (twHit > -9000 && abs(twHit - twY) <= 1 && topOpHit <= twHit + 1) {
                            isWater = true;
                            tWater = tLake;
                        }
                    }
                }
            }
        }
    }

    uint tier = meteorShadowTier(uFrame);
    int waterSsrSteps = (tier >= 3u) ? 20 : ((tier >= 2u) ? 16 : 12);
    int pbrSsrSteps = (tier >= 3u) ? 16 : 12;
    float ign = interleavedGradientNoise(pixelPos.xy, 0.0f);
    float waveHeightScale = clamp(uFrame.surfaceParams.x, 0.0f, 2.5f);
    float rippleStrength = clamp(uFrame.surfaceParams.y, 0.0f, 2.5f);

    if (isWater) {
        float3 surfRel = rayDir * tWater;
        float3 surfWorld = camPos + surfRel;
        float ndotvGeo = clamp(abs(dot(-rayDir, waterGeoNrm)), 0.05f, 1.0f);
        float4 waveData = isWaterfallFace
            ? evalMeteorWaterfallWaves((abs(waterGeoNrm.x) > 0.5f) ? surfWorld.zy : surfWorld.xy, uFrame.timeSeconds, tWater, waveHeightScale)
            : ((length(waterFlowVec) > 0.08f)
                ? evalMeteorFlowingWaves(surfWorld.xz, waterFlowVec, uFrame.timeSeconds, tWater, ndotvGeo, waveHeightScale)
                : evalMeteorWaterWaves(surfWorld.xz, uFrame.timeSeconds, tWater, ndotvGeo, waveHeightScale));

        float2 rGrad = float2(0.0f);
        if (!isWaterfallFace && rippleStrength > 0.01f) {
            uint rcCnt = min(uint(waterRipples[0].x + 0.5f), 64u);
            for (uint ri = 0u; ri < rcCnt; ++ri) {
                float4 rp = waterRipples[1u + ri * 2u];
                if (abs(surfRel.y - rp.y) < 1.6f) {
                    rGrad += evalRippleContribution(surfRel.xz, rp, waterRipples[2u + ri * 2u]).xy * rippleStrength;
                }
            }
        }
        float3 waveNrm = isWaterfallFace
            ? normalize(waterGeoNrm + (abs(waterGeoNrm.x) > 0.5f ? float3(0,0,1) : float3(1,0,0)) * waveData.x + float3(0, waveData.y, 0))
            : normalize(waterGeoNrm + float3(waveData.x - rGrad.x, 0.0f, waveData.z - rGrad.y));

        float3 ssrWaveNrm = isWaterfallFace ? waveNrm : normalize(lerp(waterGeoNrm, waveNrm, exp(-tWater * 0.022f)));
        float3 reflDir = reflect(rayDir, ssrWaveNrm);
        if (!isWaterfallFace && reflDir.y < 0.02f) {
            reflDir = normalize(float3(reflDir.x, 0.02f, reflDir.z));
        }
        float minHitRelY = isWaterfallFace ? -1e6f : (surfRel.y - 0.08f);
        bool skipFarOceanSsr = (!isWaterfallFace && tWater > 56.0f && reflDir.y > 0.12f);
        float4 ssrHit = skipFarOceanSsr
            ? float4(0.0f, 0.0f, 0.0f, 0.0f)
            : traceBinaryRefinedSsr(uFrame, sceneColorTex, sceneDepthTex, duskLightingFiltered, duskSunFiltered, surfRel + waterGeoNrm * 0.04f, reflDir, waterGeoNrm, -0.04f, ign, waterSsrSteps, 2, minHitRelY);
        if (ssrHit.a < 0.98f) {
            float surfSkyVis = (uFrame.dimensionId == 0 && canSampleVoxels) ? evalWaterSurfaceSkyVisFast(voxelGrid, vOrg, surfWorld, tWater) : 1.0f;
            if (surfSkyVis > 0.04f) {
                float skyReflW = clamp(surfSkyVis * lerp(0.55f, 1.0f, surfSkyVis), 0.0f, 1.0f);
                float3 skyRefl = float3(sampleSkyLut(skyReflLut, reflDir));
                float3 gatedSky = lerp(float3(0.004f, 0.007f, 0.012f), skyRefl, skyReflW);
                float3 compRgb = lerp(gatedSky, ssrHit.rgb, clamp(ssrHit.a, 0.0f, 1.0f));
                return float4(max(compRgb, float3(0.0f)), max(ssrHit.a, skyReflW));
            }
        }
        return ssrHit;
    }

    if (isSky) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    bool enablePbrRain = meteorFeaturePbrRain(uFrame);
    MeteorPbrMaterial pbr = enablePbrRain ? decodePbrMaterial(activeTag, activeSub, rawSrgb, rawLin) : decodePbrMaterial(0u, 0u, rawSrgb, rawLin);

    bool isRainGround = (enablePbrRain && uFrame.dimensionId == 0 && env.rainFactor > 0.05f && surfNormal.y > 0.78f && !pbr.active && activeTag != 8u);
    float2 rainPw = isRainGround ? evalRainPuddleMask(worldPos.xz, env.rainFactor) : float2(0.0f);
    if (rainPw.y > 0.04f && canSampleVoxels && isColumnOpaqueBlockedFast(voxelGrid, vOrg, worldPos)) {
        rainPw = float2(0.0f);
    }
    bool isRainWet = (rainPw.y > 0.04f);

    bool pbrGlossy = pbr.active && (pbr.roughness <= 0.42f) && (pbr.reflectivity >= 0.12f);
    if (!pbrGlossy && activeTag != 8u && !isRainWet && !(uFrame.dimensionId == 1 && surfNormal.y > 0.72f)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float3 reflNrm = surfNormal;
    if (isRainWet) {
        float2 rip = evalRaindropRippleNormal(worldPos.xz, uFrame.timeSeconds, rainPw.x);
        reflNrm = normalize(surfNormal + float3(-rip.x, 0.0f, -rip.y));
    }
    float3 reflDir = reflect(rayDir, reflNrm);
    return traceBinaryRefinedSsr(uFrame, sceneColorTex, sceneDepthTex, duskLightingFiltered, duskSunFiltered, relPos + surfNormal * 0.05f, reflDir, reflNrm, 0.02f, ign, pbrSsrSteps, 2, -1e6f);
}

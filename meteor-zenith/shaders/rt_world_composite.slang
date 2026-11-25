// Meteor Shader Pass 3 — Unified RT + PBR + 3D Rippled Water + Stained Glass Composite
module meteor_rt_world_composite;
import metality;

#include "lib/meteor_common.slang"
#include "lib/meteor_sky_end_nether.slang"
#include "lib/meteor_water.slang"

ConstantBuffer<MetalityFrame> uFrame;
StructuredBuffer<uint> voxelGrid;
StructuredBuffer<float4> emissiveLights;
StructuredBuffer<float4> waterRipples;
StructuredBuffer<float4> shadowMatrices;
StructuredBuffer<uint> tileLightGrid;
StructuredBuffer<float4> entityShadowCasters;
Texture2D<float4> sceneColorTex;
Texture2D<float4> sceneDepthTex;
Texture2D<float4> rtGiStep2;
Texture2D<float4> rtReflectTemporal;
Texture2D<float4> rtSunStep2;
Texture2D<half4> skyViewLut;
Texture2D<half4> skyReflLut;
Texture2D<half4> cloudsQres;

float4 sampleHalfResBilateralRt(
    Texture2D<float4> halfTex,
    Texture2D<float4> depthTex,
    MetalityFrame f,
    float2 uv,
    float3 centerRelPos,
    float3 surfNormal,
    bool rejectSkyAlpha
) {
    int2 hDim = max(metality_tex_size(halfTex), int2(1, 1));
    int2 dDim = max(metality_tex_size(depthTex), int2(1, 1));
    float2 fCoord = uv * float2(hDim) - 0.5f;
    int2 i0 = int2(floor(fCoord));
    float2 fr = fCoord - float2(i0);

    float centerDist = length(centerRelPos);
    float baseTol = (surfNormal.y > 0.85f) ? 0.068f : 0.030f;
    float planeTol = max(baseTol + centerDist * 0.0045f, baseTol);
    float invPlaneTol2 = 1.0f / (planeTol * planeTol);

    half4 accum = half4(0.0h, 0.0h, 0.0h, 0.0h);
    half wSum = half(0.0h);
    float4 bestTapVal = float4(0.0f, 0.0f, 0.0f, 0.5f);
    float bestPlaneDist = 1e9f;
    bool bestFound = false;
    for (int oy = 0; oy <= 1; ++oy) {
        half wy = half((oy == 0) ? (1.0f - fr.y) : fr.y);
        for (int ox = 0; ox <= 1; ++ox) {
            half wx = half((ox == 0) ? (1.0f - fr.x) : fr.x);
            int2 hp = clamp(i0 + int2(ox, oy), int2(0, 0), hDim - int2(1, 1));
            float4 val = halfTex.Load(int3(hp, 0));
            if (rejectSkyAlpha && val.a < -0.5f) continue;

            int2 dPx = snapDepthCoord(dDim, texelCenterUv(hp, hDim));
            float sd = depthTex.Load(int3(dPx, 0)).r;
            if (!isSkyDepth(sd)) {
                float3 sRel = reconstructRelWorldPos(f, texelCenterUv(dPx, dDim), sd);
                float planeDist = abs(dot(sRel - centerRelPos, surfNormal));
                if (planeDist < bestPlaneDist) {
                    bestPlaneDist = planeDist;
                    bestTapVal = val;
                    bestFound = true;
                }
                half planeW = half(exp(-(planeDist * planeDist) * invPlaneTol2));
                half w = max(wx * wy, half(0.02h)) * planeW;
                accum += half4(val) * w;
                wSum += w;
            }
        }
    }
    if (wSum > half(1e-4h)) {
        return float4(accum / wSum);
    }
    if (bestFound) {
        return bestTapVal;
    }
    int2 nearHp = clamp(int2(floor(uv * float2(hDim))), int2(0, 0), hDim - int2(1, 1));
    float4 fb = halfTex.Load(int3(nearHp, 0));
    if (rejectSkyAlpha && fb.a < -0.5f) {
        return float4(0.0f, 0.0f, 0.0f, 0.5f);
    }
    return fb;
}

struct HalfResPairResultRt {
    float4 a;
    float3 b;
};

HalfResPairResultRt sampleHalfResBilateralPairRt(
    Texture2D<float4> halfTexA,
    Texture2D<float4> halfTexB,
    Texture2D<float4> depthTex,
    MetalityFrame f,
    float2 uv,
    float3 centerRelPos,
    float3 surfNormal
) {
    HalfResPairResultRt pairRes;
    int2 hDim = max(metality_tex_size(halfTexA), int2(1, 1));
    int2 dDim = max(metality_tex_size(depthTex), int2(1, 1));
    float2 fCoord = uv * float2(hDim) - 0.5f;
    int2 i0 = int2(floor(fCoord));
    float2 fr = fCoord - float2(i0);

    float centerDist = length(centerRelPos);
    float baseTol = (surfNormal.y > 0.85f) ? 0.068f : 0.030f;
    float planeTol = max(baseTol + centerDist * 0.0045f, baseTol);
    float invPlaneTol2 = 1.0f / (planeTol * planeTol);

    half4 accumA = half4(0.0h, 0.0h, 0.0h, 0.0h);
    half3 accumB = half3(0.0h, 0.0h, 0.0h);
    half wSumA = half(0.0h);
    half wSumB = half(0.0h);
    float4 bestA = float4(0.0f, 0.0f, 0.0f, 0.5f);
    float3 bestB = float3(0.0f, 0.0f, 0.0f);
    float bestDistA = 1e9f;
    float bestDistB = 1e9f;
    bool foundA = false;
    bool foundB = false;

    for (int oy = 0; oy <= 1; ++oy) {
        half wy = half((oy == 0) ? (1.0f - fr.y) : fr.y);
        for (int ox = 0; ox <= 1; ++ox) {
            half wx = half((ox == 0) ? (1.0f - fr.x) : fr.x);
            int2 hp = clamp(i0 + int2(ox, oy), int2(0, 0), hDim - int2(1, 1));
            int2 dPx = snapDepthCoord(dDim, texelCenterUv(hp, hDim));
            float sd = depthTex.Load(int3(dPx, 0)).r;
            if (isSkyDepth(sd)) continue;

            float3 sRel = reconstructRelWorldPos(f, texelCenterUv(dPx, dDim), sd);
            float planeDist = abs(dot(sRel - centerRelPos, surfNormal));
            half planeW = half(exp(-(planeDist * planeDist) * invPlaneTol2));
            half w = max(wx * wy, half(0.02h)) * planeW;

            float4 valA = halfTexA.Load(int3(hp, 0));
            if (valA.a >= -0.5f) {
                accumA += half4(valA) * w;
                wSumA += w;
                if (planeDist < bestDistA) {
                    bestDistA = planeDist;
                    bestA = valA;
                    foundA = true;
                }
            }
            float3 valB = halfTexB.Load(int3(hp, 0)).rgb;
            accumB += half3(valB) * w;
            wSumB += w;
            if (planeDist < bestDistB) {
                bestDistB = planeDist;
                bestB = valB;
                foundB = true;
            }
        }
    }

    int2 nearHp = clamp(int2(floor(uv * float2(hDim))), int2(0, 0), hDim - int2(1, 1));
    if (wSumA > half(1e-4h)) {
        pairRes.a = float4(accumA / wSumA);
    } else if (foundA) {
        pairRes.a = bestA;
    } else {
        float4 fbA = halfTexA.Load(int3(nearHp, 0));
        pairRes.a = (fbA.a < -0.5f) ? float4(0.0f, 0.0f, 0.0f, 0.5f) : fbA;
    }

    if (wSumB > half(1e-4h)) {
        pairRes.b = float3(accumB / wSumB);
    } else if (foundB) {
        pairRes.b = bestB;
    } else {
        pairRes.b = halfTexB.Load(int3(nearHp, 0)).rgb;
    }
    return pairRes;
}

float3 evalSpecularBrdfGgx(float3 nrm, float3 viewDir, float3 lightDir, float roughness, float3 f0) {
    float ndotl = max(dot(nrm, lightDir), 0.0f);
    float ndotv = max(dot(nrm, viewDir), 1e-4f);
    if (ndotl <= 0.0f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    float3 hSum = viewDir + lightDir;
    float hLen2 = dot(hSum, hSum);
    if (hLen2 <= 1e-6f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    float3 halfV = hSum * rsqrt(hLen2);
    float ndoth = saturate(dot(nrm, halfV));
    float vdoth = saturate(dot(viewDir, halfV));
    float a = clamp(roughness * roughness, 0.0012f, 1.0f);
    float a2 = a * a;
    float denom = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    float dTerm = a2 / max(METEOR_PI * denom * denom, 1e-5f);
    float k = (roughness + 1.0f);
    float kGeo = (k * k) * 0.125f;
    float gV = ndotv / max(ndotv * (1.0f - kGeo) + kGeo, 1e-4f);
    float gL = ndotl / max(ndotl * (1.0f - kGeo) + kGeo, 1e-4f);
    float3 fresnel = f0 + (float3(1.0f, 1.0f, 1.0f) - f0) * pow(saturate(1.0f - vdoth), 5.0f);
    return (dTerm * gV * gL * fresnel) / max(4.0f * ndotv * ndotl, 1e-4f) * ndotl;
}

float3 applyMeteorUnderwaterFog(MetalityFrame f, MeteorEnvFactors env, float3 colorLin, float3 relPos, float3 worldPos, float dist) {
    if (f.submergedMode != 1) {
        return applyUnderwaterOverlay(f, env, colorLin, relPos, worldPos, dist);
    }
    float clarity = clamp(f.userParams.z, 0.4f, 1.8f);
    float densityScale = clamp(1.38f - clarity * 0.32f, 0.80f, 1.55f);
    float3 uwSigma = float3(0.26f, 0.125f, 0.088f) * densityScale;
    float dClamped = clamp(dist, 0.0f, 48.0f);
    float3 uwTrans = exp(-uwSigma * dClamped);
    float depthBelowSea = max((f.seaLevel - 0.111111f) - worldPos.y, 0.0f);
    float depthDark = lerp(1.0f, 0.42f, smoothstep(1.5f, 14.0f, depthBelowSea));
    float3 dayWaterFog = lerp(float3(0.018f, 0.19f, 0.29f), float3(0.036f, 0.14f, 0.22f), env.sunsetFactor) * depthDark;
    float3 waterFog = lerp(dayWaterFog, float3(0.006f, 0.028f, 0.072f), env.nightFactor) * lerp(1.0f, 0.65f, env.rainFactor);
    return colorLin * uwTrans + waterFog * (float3(1.0f, 1.0f, 1.0f) - uwTrans);
}

[shader("fragment")]
float4 fsRtWorldComposite(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    float depth = ds.depth;
    float3 rawSrgb = sceneColorTex.Load(int3(ds.px, 0)).rgb;
    float3 rawLin = srgbToLinear(rawSrgb);
    // Compute hardware quad derivatives in uniform control flow BEFORE any water/sky early returns (P5 Micro-Bevel)
    float rawLumaCenter = luminance(rawSrgb);
    float microGradX = ddx(rawLumaCenter);
    float microGradY = ddy(rawLumaCenter);

    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    float3 rayDir = reconstructSkyWorldDir(uFrame, ds.centerUv);
    float3 camPos = uFrame.camWorldPos.xyz;
    int3 vOrg = uFrame.voxelOrigin.xyz;
    bool isSky = isSkyDepth(depth);

    float3 relPos = isSky ? (rayDir * 280.0f) : reconstructRelWorldPos(uFrame, ds.centerUv, depth);
    float3 worldPos = camPos + relPos;
    float terrainT = isSky ? 1e6f : length(relPos);

    float3 sunDir = normalize(uFrame.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 keyDir = sunDir;
    if (sunDir.y < -0.05f && uFrame.dimensionId == 0) {
        keyDir = normalize(uFrame.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    } else if (uFrame.dimensionId == 1) {
        keyDir = getMeteorBlackHoleDir();
    }

    float3 directCol = lerp(float3(1.66f, 1.56f, 1.40f), float3(2.42f, 1.04f, 0.37f), env.sunsetFactor);
    directCol = lerp(directCol, float3(0.115f, 0.195f, 0.360f), env.nightFactor);
    directCol = lerp(directCol, float3(0.41f, 0.46f, 0.53f), env.rainFactor * 0.82f);
    if (uFrame.dimensionId == 1) {
        directCol = float3(1.65f, 1.25f, 2.05f);
    } else if (uFrame.dimensionId == -1) {
        directCol = float3(0.0f, 0.0f, 0.0f);
    }

    float highlightBoost = clamp(uFrame.lightingParams.x, 0.4f, 2.5f);
    float shadowDepth = clamp(uFrame.lightingParams.y, 0.5f, 2.0f);
    float emissiveScale = clamp(uFrame.lightingParams.z, 0.0f, 3.0f);
    float hazeScale = clamp(uFrame.lightingParams.w, 0.0f, 2.5f);
    float waveHeightScale = clamp(uFrame.surfaceParams.x, 0.0f, 2.5f);
    float rippleStrength = clamp(uFrame.surfaceParams.y, 0.0f, 2.5f);
    float specularStrength = clamp(uFrame.surfaceParams.z, 0.0f, 2.5f);
    float reflectionStrength = clamp(uFrame.surfaceParams.w, 0.0f, 2.5f);
    float claritySetting = clamp(uFrame.userParams.z, 0.4f, 1.8f);

    // -------------------------------------------------------------------------
    // 1. Unified 512x128x512 Analytical + Voxel Water Surface Detection
    //    Runs BEFORE isSkyDepth early-return so FluidRendererMixin-suppressed
    //    oceans, elevated pools, waterfalls, and underwater ceilings all render.
    // -------------------------------------------------------------------------
    bool isWater = false;
    bool underwaterCeiling = false;
    bool isWaterfallFace = false;
    float tWater = 1e6f;
    float3 waterGeoNormal = float3(0.0f, 1.0f, 0.0f);
    float2 waterFlowVec = float2(0.0f, 0.0f);

    float seaSurfaceY = uFrame.seaLevel - 0.111111f;
    int seaBlockY = int(floor(uFrame.seaLevel)) - 1;
    bool canSampleVoxels = ((uFrame.voxelOrigin.w & 1) != 0);
    bool has3dWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & 2) != 0);
    bool hasWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & (2 | 8)) != 0);
    bool hasStainedGlass = canSampleVoxels && ((uFrame.voxelOrigin.w & 4) != 0);
    bool enableShadows = meteorFeatureShadows(uFrame);
    bool enableWater = meteorFeatureWater(uFrame);
    bool enableRefl = meteorFeatureReflections(uFrame);
    bool enableClouds = meteorFeatureClouds(uFrame);
    bool enableGiAo = meteorFeatureGiAo(uFrame);
    bool enablePbrRain = meteorFeaturePbrRain(uFrame);

    // 1a. Underwater looking upward: Snell's Window ceiling before sky/above-water foliage
    if (uFrame.dimensionId != -1 && uFrame.submergedMode == 1 && rayDir.y > 0.005f) {
        float ceilY = seaSurfaceY;
        if (canSampleVoxels) {
            int3 startCell = int3(floor(camPos)) - vOrg;
            for (int upStep = 0; upStep <= 16; ++upStep) {
                int3 probe = int3(startCell.x, startCell.y + upStep, startCell.z);
                int3 probeAbove = int3(startCell.x, startCell.y + upStep + 1, startCell.z);
                if (!inVoxelBounds512Common(probe)) break;
                uint idxP = voxelLinearIndex512Common(probe);
                uint bP = (voxelGrid[idxP >> 2u] >> ((idxP & 3u) * 8u)) & 0xFFu;
                if (((bP >> 4u) & 0xFu) == 3u && inVoxelBounds512Common(probeAbove)) {
                    uint idxA = voxelLinearIndex512Common(probeAbove);
                    uint bA = (voxelGrid[idxA >> 2u] >> ((idxA & 3u) * 8u)) & 0xFFu;
                    if (((bA >> 4u) & 0xFu) != 3u) {
                        ceilY = float(int(floor(camPos.y)) + upStep) + 0.888f;
                        break;
                    }
                }
            }
        }
        if (camPos.y < ceilY - 0.01f) {
            float tCeil = (ceilY - camPos.y) / max(rayDir.y, 1e-4f);
            if (tCeil > 0.02f && tCeil < terrainT - 0.01f && tCeil < 160.0f) {
                isWater = true;
                underwaterCeiling = true;
                tWater = tCeil;
                waterGeoNormal = float3(0.0f, -1.0f, 0.0f);
            }
        }
    }

    // 1b. Above-water view: test primary sea plane, elevated pools, and waterfalls
    if (!isWater && hasWaterVoxels && uFrame.submergedMode == 0) {
        // Primary raster surface classification: stained glass / ice windows must never be overridden by analytical water fallbacks
        bool primaryIsGlass = false;
        if (!isSky && canSampleVoxels) {
            int3 cIn = int3(floor(worldPos - rayDir * 0.04f)) - vOrg;
            int3 cOut = int3(floor(worldPos + rayDir * 0.02f)) - vOrg;
            if (inVoxelBounds512Common(cIn)) {
                uint idxIn = voxelLinearIndex512Common(cIn);
                uint tagIn = ((voxelGrid[idxIn >> 2u] >> ((idxIn & 3u) * 8u)) >> 4u) & 0xFu;
                if (tagIn == 8u) primaryIsGlass = true;
            }
            if (!primaryIsGlass && inVoxelBounds512Common(cOut)) {
                uint idxOut = voxelLinearIndex512Common(cOut);
                uint tagOut = ((voxelGrid[idxOut >> 2u] >> ((idxOut & 3u) * 8u)) >> 4u) & 0xFu;
                if (tagOut == 8u) primaryIsGlass = true;
            }
        }
        bool waterRayBlocked = false;

        if (uFrame.dimensionId == 0 && camPos.y > seaSurfaceY + 0.01f && rayDir.y < -0.0005f && !primaryIsGlass) {
            float tSea = (seaSurfaceY - camPos.y) / rayDir.y;
            if (tSea > 0.04f && tSea < terrainT - 0.004f) {
                float3 seaHitProbe = camPos + rayDir * tSea - float3(0.0f, 0.08f, 0.0f);
                int3 vcSea = int3(floor(seaHitProbe)) - vOrg;
                if (has3dWaterVoxels && inVoxelBounds512Common(vcSea)) {
                    uint idxSea = voxelLinearIndex512Common(vcSea);
                    uint bSea = (voxelGrid[idxSea >> 2u] >> ((idxSea & 3u) * 8u)) & 0xFFu;
                    uint tagSea = (bSea >> 4u) & 0xFu;
                    uint subSea = bSea & 0xFu;
                    if (tagSea == 3u && (subSea == 8u || subSea == 0u || subSea >= 9u)) {
                        int3 vcAbove = vcSea + int3(0, 1, 0);
                        uint tagAbove = 0u;
                        if (inVoxelBounds512Common(vcAbove)) {
                            uint idxA = voxelLinearIndex512Common(vcAbove);
                            tagAbove = ((voxelGrid[idxA >> 2u] >> ((idxA & 3u) * 8u)) >> 4u) & 0xFu;
                        }
                        if (tagAbove != 3u) {
                            isWater = true;
                            tWater = tSea;
                            waterGeoNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                    }
                } else if (isFarOrHighSeaWaterAt(voxelGrid, vOrg, seaHitProbe, seaBlockY)) {
                    isWater = true;
                    tWater = tSea;
                    waterGeoNormal = float3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        // Unified Macro-DDA + Bilinear Sloped Flowing / Waterfall / Deep Pool detection
        bool hasElevatedWaterAboveSea = (uFrame.dimensionId != 0) ||
            ((uint(uFrame.voxelOrigin.w) & 0x80u) == 0u) ||
            ((int((uint(uFrame.voxelOrigin.w) >> 19u) & 0x7FFu) - 512) > seaBlockY);
        bool runWaterDda = !isWater;
        if (isSky && !isWater) {
            WaterYBounds wbSky = decodeWaterYBounds(uFrame.voxelOrigin.w);
            runWaterDda = hasElevatedWaterAboveSea &&
                (!wbSky.valid || (rayDir.y < -0.0005f && camPos.y > wbSky.yMin) || (rayDir.y <= 0.02f && camPos.y < wbSky.yMax));
        }
        if (isWater && hasElevatedWaterAboveSea && camPos.y > float(seaBlockY) + 1.02f) {
            float3 seaHitPos = camPos + rayDir * tWater;
            int tw0 = sampleTopWaterWorldY(voxelGrid, vOrg, int2(floor(camPos.xz)));
            int tw1 = sampleTopWaterWorldY(voxelGrid, vOrg, int2(floor(seaHitPos.xz)));
            int tw2 = sampleTopWaterWorldY(voxelGrid, vOrg, int2(floor(lerp(camPos.xz, seaHitPos.xz, 0.33f))));
            int tw3 = sampleTopWaterWorldY(voxelGrid, vOrg, int2(floor(lerp(camPos.xz, seaHitPos.xz, 0.67f))));
            runWaterDda = (max(max(tw0, tw1), max(tw2, tw3)) > seaBlockY);
        }
        if (runWaterDda && has3dWaterVoxels) {
            float ddaMaxT = isWater
                ? min(tWater, max((float(seaBlockY) + 0.98f - camPos.y) / min(rayDir.y, -0.0005f), 0.04f))
                : (isSky ? 224.0f : terrainT);
            float ddaEndY = isWater ? (camPos.y + rayDir.y * ddaMaxT) : worldPos.y;
            if (ddaMaxT > 0.045f && rayOverlapsWaterYBounds(uFrame, camPos.y, ddaEndY, rayDir, isSky && !isWater)) {
                WaterDdaHit wh = traceWaterMacroDda(voxelGrid, vOrg, camPos, rayDir, 0.04f, ddaMaxT, uFrame.voxelOrigin.w);
                if (wh.hit) {
                    isWater = true;
                    isWaterfallFace = wh.isVertical;
                    tWater = wh.t;
                    waterGeoNormal = wh.nrm;
                    waterFlowVec = wh.flowVec;
                } else if (wh.blocked) {
                    isWater = false;
                    waterRayBlocked = true;
                }
            }
        }

        // Elevated-lake fallback strictly outside the 3D voxel slice and only when no glass/solid blocker sits on the ray
        if (!isWater && !waterRayBlocked && !primaryIsGlass && !isSky && uFrame.dimensionId == 0 && rayDir.y < -0.0005f) {
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
                            waterGeoNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Render 3D Rippled Liquid Water (Surface + Snell's Window + Refraction)
    // -------------------------------------------------------------------------
    if (isWater) {
        if (!enableWater) {
            // Ultra-fast LOW-mode Beer-Lambert water (no waves/refraction/caustics/BVH-refl/point-spec; zero chunk-reload on profile toggle)
            float3 surfRelLow = rayDir * tWater;
            float3 surfWorldLow = camPos + surfRelLow;
            float waterThicknessLow = isSky ? 18.0f : clamp(terrainT - tWater, 0.04f, 48.0f);
            if (underwaterCeiling) {
                float cosULow = clamp(dot(-rayDir, waterGeoNormal), 0.0f, 1.0f);
                float3 refrSkyDirLow = normalize(float3(rayDir.x * 1.25f, max(rayDir.y * 0.85f, 0.04f), rayDir.z * 1.25f));
                float3 aboveBgLow = isSky
                    ? ((uFrame.dimensionId == 0) ? float3(sampleSkyLut(skyViewLut, refrSkyDirLow)) : evalDimensionSkyFast(refrSkyDirLow, uFrame, env, false))
                    : (rawLin * 1.15f);
                float snellWinLow = smoothstep(0.60f, 0.78f, cosULow);
                float3 tirLow = float3(0.016f, 0.135f, 0.215f)
                              * lerp(1.0f, 0.12f, env.nightFactor) * lerp(1.0f, 0.62f, env.rainFactor);
                float3 ceilColLow = lerp(tirLow, aboveBgLow * float3(0.78f, 0.94f, 0.99f), snellWinLow);
                ceilColLow = applyMeteorUnderwaterFog(uFrame, env, ceilColLow, surfRelLow, surfWorldLow, tWater);
                return float4(max(ceilColLow, float3(0.0f, 0.0f, 0.0f)), 1.0f);
            }
            float waterDepthYLow = isSky ? 12.0f : max(surfWorldLow.y - worldPos.y, 0.04f);
            bool caveBlockedLow = (uFrame.dimensionId == 0) && canSampleVoxels && isColumnSkyBlockedFast(voxelGrid, vOrg, surfWorldLow);
            float surfSkyVisLow = caveBlockedLow ? 0.0f : 1.0f;
            float3 openWaterGiLow = lerp(float3(0.16f, 0.24f, 0.42f), float3(0.32f, 0.20f, 0.16f), env.sunsetFactor * 0.65f);
            openWaterGiLow = lerp(openWaterGiLow, float3(0.022f, 0.038f, 0.095f), env.nightFactor);
            openWaterGiLow = lerp(openWaterGiLow, float3(0.10f, 0.13f, 0.18f), env.rainFactor * 0.80f);
            float surfSunVisLow = surfSkyVisLow * smoothstep(-0.02f, 0.12f, keyDir.y);
            float3 waterIllumLow = directCol * surfSunVisLow * (0.40f + 0.60f * max(keyDir.y, 0.0f)) * (1.0f - env.nightFactor * 0.90f)
                                 + openWaterGiLow * surfSkyVisLow * 1.15f + float3(0.008f, 0.012f, 0.018f);
            float wetMaskLow = smoothstep(0.02f, 0.12f, waterDepthYLow);
            float3 seabedColLow = isSky
                ? float3(0.008f, 0.025f, 0.06f)
                : (rawLin * lerp(float3(1.0f, 1.0f, 1.0f), float3(0.74f, 0.77f, 0.80f), wetMaskLow)
                    * (directCol * surfSunVisLow * max(keyDir.y, 0.0f) * (1.0f - env.nightFactor * 0.90f)
                        + openWaterGiLow * surfSkyVisLow * 1.15f + float3(0.008f, 0.012f, 0.018f)));
            float clarityScaleLow = clamp(1.52f - claritySetting * 0.42f, 0.85f, 1.80f);
            float3 sigmaExtLow = float3(0.68f, 0.36f, 0.25f) * clarityScaleLow;
            float effThickLow = isWaterfallFace ? (waterThicknessLow * 0.95f + 0.18f) : lerp(waterDepthYLow, waterThicknessLow, 0.52f);
            float3 transLow = exp(-sigmaExtLow * effThickLow);
            float depthBlendLow = smoothstep(0.25f, 4.20f, isWaterfallFace ? waterThicknessLow : waterDepthYLow);
            float3 waterLightFactorLow = (waterIllumLow * 0.32f + float3(0.008f, 0.015f, 0.025f)) * lerp(1.0f, 0.60f, env.rainFactor);
            float3 waterScatterLow = lerp(float3(0.048f, 0.34f, 0.42f), float3(0.014f, 0.105f, 0.24f), depthBlendLow) * waterLightFactorLow;
            if (isWaterfallFace) {
                float3 aerateLow = float3(0.14f, 0.54f, 0.64f) * (waterIllumLow * 0.35f + float3(0.012f, 0.024f, 0.032f)) * lerp(1.0f, 0.60f, env.rainFactor);
                waterScatterLow = lerp(waterScatterLow, aerateLow, 0.35f);
            }
            waterScatterLow = lerp(waterScatterLow, waterScatterLow * float3(1.22f, 0.80f, 0.58f), env.sunsetFactor * 0.62f);
            float shoreFadeLow = smoothstep(0.02f, 0.35f, waterThicknessLow);
            float deepTurbLow = smoothstep(1.6f, 6.5f, effThickLow);
            float bodyVeilLow = isWaterfallFace
                ? (0.16f + 0.18f * (1.0f - exp(-effThickLow * 0.72f)))
                : ((0.30f * (1.0f - exp(-effThickLow * 0.90f)) + 0.42f * deepTurbLow) * shoreFadeLow);
            float3 bodyColLow = seabedColLow * transLow * (1.0f - bodyVeilLow) + waterScatterLow * (1.0f - transLow * (1.0f - bodyVeilLow));
            float3 reflDirLow = reflect(rayDir, waterGeoNormal);
            if (!isWaterfallFace && reflDirLow.y < 0.02f) {
                reflDirLow = normalize(float3(reflDirLow.x, 0.02f, reflDirLow.z));
            }
            float3 skyReflLow = caveBlockedLow
                ? float3(0.006f, 0.010f, 0.016f)
                : ((uFrame.dimensionId == 0) ? float3(sampleSkyLut(skyViewLut, reflDirLow)) : evalDimensionSkyFast(reflDirLow, uFrame, env, false));
            if (!isWaterfallFace && !caveBlockedLow && uFrame.dimensionId == 0) {
                float grazingHorizonLow = smoothstep(0.14f, 0.01f, reflDirLow.y) * smoothstep(18.0f, 65.0f, tWater);
                skyReflLow *= lerp(float3(1.0f, 1.0f, 1.0f), float3(0.76f, 0.82f, 0.90f), grazingHorizonLow * (1.0f - env.sunsetFactor * 0.65f));
            }
            float cosVLow = saturate(dot(-rayDir, waterGeoNormal));
            float shoreReflFadeLow = isWaterfallFace ? 1.0f : smoothstep(0.01f, 0.18f, waterThicknessLow);
            float fresnelLow = clamp(0.055f + 0.78f * pow(saturate(1.0f - cosVLow), 4.6f), 0.07f, 0.78f) * shoreReflFadeLow;
            float3 waterLitLow = lerp(bodyColLow, skyReflLow, fresnelLow);
            float fogDistLow = max(tWater - 76.0f, 0.0f);
            float fogLow = clamp(1.0f - exp(-fogDistLow * (0.00072f + env.rainFactor * 0.0020f) * hazeScale), 0.0f, 0.72f);
            float horizonFadeLow = smoothstep(0.22f, 0.02f, abs(rayDir.y)) * smoothstep(105.0f, 185.0f, tWater) * 0.96f;
            fogLow = max(fogLow, horizonFadeLow);
            if (fogLow > 1e-4f) {
                float3 airDirLow = normalize(float3(rayDir.x, max(rayDir.y, 0.005f), rayDir.z));
                float3 airColLow = (uFrame.dimensionId == 0)
                    ? (caveBlockedLow ? float3(0.003f, 0.005f, 0.008f) : float3(sampleSkyLut(skyViewLut, airDirLow)))
                    : float3(0.003f, 0.005f, 0.008f);
                waterLitLow = lerp(waterLitLow, airColLow, fogLow);
            }
            return float4(max(waterLitLow, float3(0.0f, 0.0f, 0.0f)), 1.0f);
        }

        float3 surfRel = rayDir * tWater;
        float3 surfWorld = camPos + surfRel;
        float ndotvGeo = clamp(abs(dot(-rayDir, waterGeoNormal)), 0.05f, 1.0f);

        float4 waveData = isWaterfallFace
            ? evalMeteorWaterfallWaves((abs(waterGeoNormal.x) > 0.5f) ? surfWorld.zy : surfWorld.xy, uFrame.timeSeconds, tWater, waveHeightScale)
            : ((length(waterFlowVec) > 0.08f)
                ? evalMeteorFlowingWaves(surfWorld.xz, waterFlowVec, uFrame.timeSeconds, tWater, ndotvGeo, waveHeightScale)
                : evalMeteorWaterWaves(surfWorld.xz, uFrame.timeSeconds, tWater, ndotvGeo, waveHeightScale));

        // Accumulate interactive boat Kelvin V-wakes, oar rings, buoy & splash ripples
        float2 rippleGrad = float2(0.0f, 0.0f);
        float rippleFoam = 0.0f;
        if (!isWaterfallFace && rippleStrength > 0.01f) {
            uint rippleCount = min(uint(waterRipples[0].x + 0.5f), 64u);
            for (uint ri = 0u; ri < rippleCount; ++ri) {
                float4 rPosAge = waterRipples[1u + ri * 2u];
                if (abs(surfRel.y - rPosAge.y) < 1.6f) {
                    float4 rDirAmpKind = waterRipples[2u + ri * 2u];
                    float4 rc = evalRippleContribution(surfRel.xz, rPosAge, rDirAmpKind);
                    rippleGrad += rc.xy * rippleStrength;
                    rippleFoam = max(rippleFoam, rc.w * rippleStrength);
                }
            }
        }

        float3 waveNrm = waveData.xyz;
        if (underwaterCeiling) {
            waveNrm = normalize(float3(waveData.x - rippleGrad.x, -max(abs(waveData.y), 0.22f), waveData.z - rippleGrad.y));
        } else if (isWaterfallFace) {
            float3 wTangent = abs(waterGeoNormal.x) > 0.5f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
            waveNrm = normalize(waterGeoNormal + wTangent * waveData.x + float3(0.0f, waveData.y, 0.0f));
        } else {
            waveNrm = normalize(waterGeoNormal + float3(waveData.x - rippleGrad.x, 0.0f, waveData.z - rippleGrad.y));
        }

        float waterThickness = isSky ? 18.0f : clamp(terrainT - tWater, 0.04f, 48.0f);

        if (underwaterCeiling) {
            float cosU = clamp(dot(-rayDir, waveNrm), 0.0f, 1.0f);
            float3 refrSkyDir = normalize(float3(rayDir.x * 1.25f + waveNrm.x * 0.45f, max(rayDir.y * 0.85f, 0.04f), rayDir.z * 1.25f + waveNrm.z * 0.45f));
            float3 aboveBg = isSky
                ? ((uFrame.dimensionId == 0)
                    ? float3(sampleSkyLut(enableClouds ? skyReflLut : skyViewLut, refrSkyDir))
                    : evalDimensionSkyFast(refrSkyDir, uFrame, env, false))
                : (rawLin * 1.15f);
            float snellWindow = smoothstep(0.60f, 0.78f, cosU);

            // Smooth liquid-mirror Total Internal Reflection sculpted purely by calm 4-octave Simplex waves (zero Voronoi web/zebra lines)
            float3 tirReflDir = reflect(rayDir, waveNrm);
            float horizFactor = pow(clamp(1.0f - abs(tirReflDir.y), 0.0f, 1.0f), 1.35f);
            float2 lightDirXZ = normalize(sunDir.xz + float2(0.18f, 0.12f));
            float2 viewDirXZ = normalize(rayDir.xz + float2(1e-4f, 0.0f));
            float slopeLight = dot(waveNrm.xz, lightDirXZ) * 3.2f + dot(waveNrm.xz, viewDirXZ) * 2.4f;
            float waveSlopeCatch = clamp(0.5f + slopeLight + waveData.w * 4.2f, 0.0f, 1.0f);
            float distFadeU = exp(-tWater * 0.045f);

            float3 deepTir = float3(0.008f, 0.075f, 0.145f);
            float3 brightTir = float3(0.032f, 0.245f, 0.340f);
            float3 tirBase = lerp(deepTir, brightTir, clamp(horizFactor * 0.55f + waveSlopeCatch * 0.50f * distFadeU, 0.0f, 1.0f));
            tirBase = lerp(tirBase, tirBase * float3(1.18f, 0.80f, 0.62f), env.sunsetFactor * 0.55f);
            tirBase *= lerp(1.0f, 0.12f, env.nightFactor) * lerp(1.0f, 0.62f, env.rainFactor);

            float cr = (cosU - 0.67f) * 9.5f;
            float criticalRim = exp(-(cr * cr)) * distFadeU;
            float3 rimTint = lerp(float3(0.28f, 0.66f, 0.78f), float3(0.75f, 0.48f, 0.30f), env.sunsetFactor * 0.65f)
                           * lerp(1.0f, 0.10f, env.nightFactor) * (1.0f - env.rainFactor * 0.65f);
            float3 tirColor = tirBase + rimTint * criticalRim * (0.35f + 0.65f * waveSlopeCatch) * 0.42f;

            float3 transmitted = aboveBg * float3(0.78f, 0.94f, 0.99f);
            float sunThrough = pow(max(dot(refrSkyDir, sunDir), 0.0f), 96.0f) * (1.0f - env.nightFactor) * (1.0f - env.rainFactor * 0.85f);
            transmitted += directCol * sunThrough * 0.45f * highlightBoost;

            float3 ceilCol = lerp(tirColor, transmitted, snellWindow);
            ceilCol = applyMeteorUnderwaterFog(uFrame, env, ceilCol, surfRel, surfWorld, tWater);
            return float4(max(ceilCol, float3(0.0f, 0.0f, 0.0f)), 1.0f);
        }

        // Screen-space seabed refraction (height- & depth-jump-guarded so dry shore/seagrass/banks never bleed into water)
        float2 refrUv = uv;
        float3 refrLin = rawLin;
        float3 refrWorld = worldPos;
        bool refrIsSky = isSky;
        float rawWaterDepthY = isSky ? 12.0f : max(surfWorld.y - worldPos.y, 0.04f);
        if (!isSky && !isWaterfallFace && !(rawWaterDepthY > 12.0f && tWater > 24.0f)) {
            float refrScale = clamp(waterThickness * 0.12f, 0.025f, 0.18f)
                            * smoothstep(0.03f, 0.22f, waterThickness)
                            * clamp(14.0f / max(tWater, 4.0f), 0.30f, 1.0f);
            if (refrScale > 1e-4f) {
                float3 probeRel = lerp(surfRel, relPos, 0.45f) + float3(waveNrm.x, 0.0f, waveNrm.z) * refrScale;
                float2 candUv = worldToScreenUv(uFrame, probeRel);
                if (candUv.x > 0.002f && candUv.x < 0.998f && candUv.y > 0.002f && candUv.y < 0.998f) {
                    float candD = sceneDepthTex.SampleLevel(metality_nearest_clamp, candUv, 0.0f).r;
                    if (!isSkyDepth(candD)) {
                        float3 candRel = reconstructRelWorldPos(uFrame, candUv, candD);
                        float candT = length(candRel);
                        float3 candW = camPos + candRel;
                        if (candW.y < surfWorld.y - 0.01f && abs(candT - terrainT) <= max(0.35f, waterThickness * 0.45f)) {
                            refrUv = candUv;
                            refrLin = srgbToLinear(sceneColorTex.SampleLevel(metality_nearest_clamp, candUv, 0.0f).rgb);
                            refrWorld = candW;
                            refrIsSky = false;
                        }
                    }
                }
            }
        } else if (isWaterfallFace && !isSky) {
            float2 wfUv = clamp(uv + float2(waveData.x, waveData.y) * 0.012f, float2(0.002f), float2(0.998f));
            float wfD = sceneDepthTex.SampleLevel(metality_nearest_clamp, wfUv, 0.0f).r;
            if (!isSkyDepth(wfD)) {
                float3 wfRel = reconstructRelWorldPos(uFrame, wfUv, wfD);
                if (length(wfRel) >= tWater + 0.04f && length(wfRel) >= terrainT - 0.15f) {
                    refrUv = wfUv;
                    refrLin = srgbToLinear(sceneColorTex.SampleLevel(metality_nearest_clamp, wfUv, 0.0f).rgb);
                    refrWorld = camPos + wfRel;
                    refrIsSky = false;
                }
            }
        }

        float refrThickness = refrIsSky ? 18.0f : clamp(length(refrWorld - surfWorld), 0.04f, 48.0f);
        float waterDepthY = refrIsSky ? 12.0f : max(surfWorld.y - refrWorld.y, 0.04f);

        float surfSkyVis = (uFrame.dimensionId == 0 && canSampleVoxels)
            ? evalWaterSurfaceSkyVisFast(voxelGrid, vOrg, surfWorld, tWater)
            : 1.0f;
        float surfSunShadow = (uFrame.dimensionId == 0 && canSampleVoxels)
            ? evalWaterSurfaceSunShadowFast(voxelGrid, vOrg, surfWorld, keyDir, tWater)
            : 1.0f;
        float3 openWaterGi = lerp(float3(0.16f, 0.24f, 0.42f), float3(0.32f, 0.20f, 0.16f), env.sunsetFactor * 0.65f);
        openWaterGi = lerp(openWaterGi, float3(0.022f, 0.038f, 0.095f), env.nightFactor);
        openWaterGi = lerp(openWaterGi, float3(0.10f, 0.13f, 0.18f), env.rainFactor * 0.80f);

        // Shade seabed with RT GI + sun + fast smooth caustics + submerged seagrass wrap lighting
        float3 seabedCol;
        float3 seabedNrm = float3(0.0f, 1.0f, 0.0f);
        float sunVisW = isSky ? 0.65f : 0.0f;
        float3 caveGiW = float3(0.0f, 0.0f, 0.0f);
        if (refrIsSky) {
            seabedCol = float3(0.008f, 0.025f, 0.06f);
        } else if (waterDepthY > 7.5f && surfSkyVis > 0.15f) {
            // P1 Deep-Ocean Seabed Skip: Beer-Lambert extinction hides >97% of deep seabed; avoid normal + bilateral GI loads
            sunVisW = surfSkyVis * surfSunShadow;
            caveGiW = openWaterGi * surfSkyVis;
            float3 seaDirect = directCol * sunVisW * max(keyDir.y, 0.0f) * (1.0f - env.nightFactor * 0.9f);
            seabedCol = refrLin * float3(0.74f, 0.77f, 0.80f) * (seaDirect + caveGiW * 1.15f);
        } else {
            seabedNrm = reconstructFastNormal(uFrame, refrUv, refrWorld - camPos, sceneDepthTex);
            float4 seabedGi = sampleHalfResBilateralRt(rtGiStep2, sceneDepthTex, uFrame, refrUv, refrWorld - camPos, seabedNrm, true);
            sunVisW = clamp(seabedGi.a, 0.0f, 1.0f);
            caveGiW = seabedGi.rgb;
            bool isSubmergedVeg = false;
            // V9b: 2-ALU chroma guard prevents blue SuspendedParticle (0.4, 0.4, 0.7) from triggering submerged vegetation HDR boost
            if (canSampleVoxels && abs(seabedNrm.y) < 0.78f && refrLin.g >= refrLin.b * 0.85f) {
                uint rvIn = sampleVoxelTag(voxelGrid, int3(floor(refrWorld - seabedNrm * 0.04f)) - vOrg);
                uint rvOut = sampleVoxelTag(voxelGrid, int3(floor(refrWorld + seabedNrm * 0.04f)) - vOrg);
                isSubmergedVeg = (rvIn == 3u && rvOut == 3u);
            }
            if (isSubmergedVeg) {
                float seaWrap = 0.45f + 0.55f * abs(dot(seabedNrm, keyDir));
                float vegSunVis = max(sunVisW, surfSkyVis * surfSunShadow * 0.80f);
                float3 seaDirect = directCol * vegSunVis * seaWrap * (1.0f - env.nightFactor * 0.9f);
                float3 vegGi = max(seabedGi.rgb, openWaterGi * surfSkyVis * 0.65f);
                seabedCol = refrLin * (seaDirect + vegGi * 1.15f);
            } else {
                float topFaceAtten = pow(saturate(seabedNrm.y * 0.85f + 0.15f), 1.6f);
                float causticVis = (1.0f - env.nightFactor * 0.92f) * (1.0f - env.rainFactor * 0.78f) * smoothstep(0.08f, 0.34f, sunDir.y) * sunVisW;
                float shallowCausticFade = smoothstep(0.08f, 0.55f, waterDepthY);
                float causticWeight = isWaterfallFace ? 0.0f : (exp(-waterDepthY * 0.20f) * shallowCausticFade * causticVis * topFaceAtten);
                float caustic = 0.0f;
                if (causticWeight > 0.015f && tWater < 48.0f) {
                    float2 cCoord = refrWorld.xz + refrWorld.y * 0.22f + waterDepthY * sunDir.xz * 0.45f;
                    caustic = evalFastWaterCaustic(cCoord + waveNrm.xz * 0.22f, uFrame.timeSeconds) * causticWeight;
                }
                float seaWrap = 0.18f * max(keyDir.y, 0.0f) + 0.82f * max(dot(seabedNrm, keyDir), 0.0f);
                float3 seaDirect = directCol * sunVisW * seaWrap * (1.0f - env.nightFactor * 0.9f);
                float3 bedGi = (seabedNrm.y < 0.35f) ? max(seabedGi.rgb, openWaterGi * surfSkyVis * 0.45f) : seabedGi.rgb;
                // V6a: Physical ~24% wet-seabed darkening under water film + visible sunlit shallow caustics
                float wetMask = smoothstep(0.02f, 0.12f, waterDepthY);
                float3 wetBedAlb = refrLin * lerp(float3(1.0f, 1.0f, 1.0f), float3(0.74f, 0.77f, 0.80f), wetMask);
                seabedCol = wetBedAlb * (seaDirect + bedGi * 1.15f + caustic * directCol * float3(0.45f, 0.92f, 1.0f) * 0.28f * highlightBoost);
            }
        }

        // Physical Beer-Lambert extinction + true RT-lit water scatter & leak-free cave reflections
        if (!refrIsSky && seabedNrm.y > 0.55f && waterThickness < 4.5f) {
            float flatShallowW = smoothstep(0.55f, 0.85f, seabedNrm.y) * (1.0f - smoothstep(1.2f, 4.5f, waterThickness));
            surfSunShadow = lerp(surfSunShadow, min(surfSunShadow, sunVisW), flatShallowW);
        }
        float surfSunVis = surfSkyVis * surfSunShadow * smoothstep(-0.02f, 0.12f, keyDir.y);
        float3 surfGiW = lerp(caveGiW, max(caveGiW, openWaterGi * surfSkyVis), surfSkyVis);

        float3 waterIllum = directCol * surfSunVis * (0.40f + 0.60f * max(keyDir.y, 0.0f)) * (1.0f - env.nightFactor * 0.90f) + surfGiW * 1.15f;
        float clarityScale = clamp(1.52f - claritySetting * 0.42f, 0.85f, 1.80f);
        float3 sigmaExt = float3(0.68f, 0.36f, 0.25f) * clarityScale;
        float effThick = isWaterfallFace ? (waterThickness * 0.95f + 0.18f) : lerp(waterDepthY, refrThickness, 0.52f);
        float3 trans = exp(-sigmaExt * effThick);
        float3 shallowTurq = float3(0.048f, 0.34f, 0.42f);
        float3 deepSapphire = float3(0.014f, 0.105f, 0.24f);
        float depthBlend = smoothstep(0.25f, 4.20f, isWaterfallFace ? waterThickness : waterDepthY);
        float3 waterLightFactor = (waterIllum * 0.32f + float3(0.008f, 0.015f, 0.025f)) * lerp(1.0f, 0.60f, env.rainFactor);
        float3 waterScatter = lerp(shallowTurq, deepSapphire, depthBlend) * waterLightFactor;
        if (isWaterfallFace) {
            float3 aerateCol = float3(0.14f, 0.54f, 0.64f) * (waterIllum * 0.35f + float3(0.012f, 0.024f, 0.032f)) * lerp(1.0f, 0.60f, env.rainFactor);
            waterScatter = lerp(waterScatter, aerateCol, 0.35f);
        }
        waterScatter = lerp(waterScatter, waterScatter * float3(1.22f, 0.80f, 0.58f), env.sunsetFactor * 0.62f);
        float shoreFade = smoothstep(0.02f, 0.35f, waterThickness);
        float deepTurbidity = smoothstep(1.6f, 6.5f, effThick);
        float bodyVeil = isWaterfallFace
            ? (0.16f + 0.18f * (1.0f - exp(-effThick * 0.72f)))
            : ((0.30f * (1.0f - exp(-effThick * 0.90f)) + 0.42f * deepTurbidity) * shoreFade);
        float3 bodyCol = seabedCol * trans * (1.0f - bodyVeil) + waterScatter * (1.0f - trans * (1.0f - bodyVeil));

        // Sky-gated + Hardware BVH reflection over curved wave normal (zero cave sky leak, reflection clouds from Sky-Refl LUT)
        float3 reflDir = reflect(rayDir, waveNrm);
        if (!isWaterfallFace && reflDir.y < 0.02f) {
            reflDir = normalize(float3(reflDir.x, 0.02f, reflDir.z));
        }
        bool caveBlockedW = (uFrame.dimensionId == 0) && canSampleVoxels && (surfSkyVis < 0.04f);
        float4 bvhRefl = enableRefl ? sampleHalfResWaterRefl(rtReflectTemporal, sceneDepthTex, uFrame, ds.centerUv, tWater) : float4(0.0f);
        float3 totalRefl;
        if (bvhRefl.a >= 0.98f) {
            totalRefl = bvhRefl.rgb;
        } else {
            float skyVisW = caveBlockedW ? 0.0f : clamp(surfSkyVis * lerp(0.55f, 1.0f, surfSkyVis), 0.0f, 1.0f);
            float3 skyRefl = caveBlockedW
                ? float3(0.006f, 0.010f, 0.016f)
                : ((uFrame.dimensionId == 0)
                    ? float3(sampleSkyLut(enableClouds ? skyReflLut : skyViewLut, reflDir))
                    : evalDimensionSkyFast(reflDir, uFrame, env, false));
            float3 gatedSkyRefl = lerp(float3(0.006f, 0.010f, 0.016f), skyRefl, skyVisW);
            totalRefl = (bvhRefl.a > 0.05f) ? lerp(gatedSkyRefl, bvhRefl.rgb, clamp(bvhRefl.a, 0.0f, 1.0f)) : gatedSkyRefl;
        }
        // V10: Gentle grazing horizon-reflection balance prevents blown-out pale cyan-white far shoreline strip
        if (!isWaterfallFace && !caveBlockedW && uFrame.dimensionId == 0) {
            float grazingHorizon = smoothstep(0.14f, 0.01f, reflDir.y) * smoothstep(18.0f, 65.0f, tWater);
            totalRefl *= lerp(float3(1.0f, 1.0f, 1.0f), float3(0.76f, 0.82f, 0.90f), grazingHorizon * (1.0f - env.sunsetFactor * 0.65f));
        }

        float cosV = saturate(dot(-rayDir, waveNrm));
        float shoreReflFade = isWaterfallFace ? 1.0f : smoothstep(0.01f, 0.18f, waterThickness);
        float waveTilt = saturate(length(waveNrm.xz) * 3.5f);
        float fresnelMin = lerp(0.075f, 0.13f, waveTilt);
        float fresnel = clamp((0.055f + 0.945f * pow(saturate(1.0f - cosV), 4.6f)) * reflectionStrength, fresnelMin, 0.86f) * shoreReflFade;
        float rainClear = pow(clamp(1.0f - env.rainFactor, 0.0f, 1.0f), 3.0f);
        float glitter = evalGgxSunHighlight(-rayDir, reflDir, waveNrm, keyDir, lerp(0.048f, 0.14f, env.rainFactor)) * rainClear;
        float3 sunGlow = directCol * glitter * highlightBoost * (1.0f - env.nightFactor * 0.85f) * surfSunVis * lerp(0.18f, 1.0f, fresnel);

        // Tile-Culled Voxel-Shadow-Tested Point EmissiveLights GGX Specular on water waves (GGX early-reject BEFORE DDA)
        float3 pointWaterSpec = float3(0.0f, 0.0f, 0.0f);
        if (specularStrength > 0.01f) {
            uint wLightCnt = min(uint(emissiveLights[0].x + 0.5f), 1024u);
            uint tileOffW = (wLightCnt > 0u) ? getTileLightOffset(tileLightGrid, uv) : 0u;
            bool isDayWater = (uFrame.dimensionId == 0 && env.nightFactor < 0.15f);
            uint wScanCap = isDayWater ? 8u : 12u;
            uint wTracedCap = isDayWater ? 4u : 6u;
            uint wLoopCnt = (tileOffW >= 4u) ? min(tileLightGrid[tileOffW], wScanCap) : min(wLightCnt, wScanCap);
            uint wTraced = 0u;
            float3 wDomAxis = snapCardinalAxis(waterGeoNormal);
            float3 lOriginW = surfWorld + (canSampleVoxels ? safeShadowOriginOffset(voxelGrid, vOrg, surfWorld, waterGeoNormal, wDomAxis, -rayDir, tWater) : (wDomAxis * 0.06f));
            for (uint ti = 0u; ti < wLoopCnt && wTraced < wTracedCap; ++ti) {
                uint li = (tileOffW >= 4u) ? tileLightGrid[tileOffW + 1u + ti] : ti;
                if (li >= wLightCnt) continue;
                float4 lp = emissiveLights[1u + li * 2u];
                if (isDayWater && surfSunVis > 0.7f && lp.w < 20.0f) continue;
                float3 toL = lp.xyz - surfRel;
                float d2 = dot(toL, toL);
                float maxR = evalPointLightMaxRange(lp.w);
                if (d2 >= maxR * maxR || d2 < 0.02f) continue;
                float dDist = sqrt(d2);
                float3 lDir = toL / dDist;
                if (dot(waveNrm, lDir) <= 0.0f) continue;
                float rFade = evalPointLightWindow(dDist, maxR);
                float atten = (lp.w / (d2 + 0.85f)) * rFade * specularStrength * highlightBoost * 1.15f;
                if (atten <= 0.0005f) continue;
                float3 specBrdf = evalSpecularBrdfGgx(waveNrm, -rayDir, lDir, 0.09f, float3(0.04f, 0.04f, 0.04f));
                if (max(specBrdf.r, max(specBrdf.g, specBrdf.b)) * atten < 0.008f) continue;
                ++wTraced;
                float3 lVisCol = float3(1.0f, 1.0f, 1.0f);
                if (canSampleVoxels && dDist > 0.35f) {
                    int maxSteps = clamp(int(ceil(dDist * 1.50f)) + 2, 4, 24);
                    lVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                        voxelGrid, vOrg, lOriginW, lDir, dDist, hasStainedGlass, maxSteps);
                }
                if (dot(lVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                float4 lc = emissiveLights[2u + li * 2u];
                pointWaterSpec += specBrdf * (lc.rgb * lVisCol) * atten;
            }
        }

        float shoreFoam = isWaterfallFace
            ? saturate(waveData.z * 0.32f)
            : ((1.0f - smoothstep(0.06f, 0.22f, waterThickness)) * smoothstep(0.01f, 0.06f, waterThickness) * 0.035f * max(0.0f, waveData.w * 2.5f + 0.35f));
        float crestFoam = clamp(rippleFoam * 0.45f + shoreFoam, 0.0f, 0.55f);
        float3 foamCol = lerp(float3(0.42f, 0.52f, 0.60f), float3(0.72f, 0.74f, 0.76f), max(sunDir.y, 0.0f))
                       * clamp(luminance(waterIllum) * 0.45f, 0.02f, 1.15f);

        float3 waterLit = lerp(bodyCol, totalRefl, fresnel) + sunGlow + pointWaterSpec;
        waterLit = lerp(waterLit, foamCol, crestFoam);

        float fogDistW = max(tWater - 76.0f, 0.0f);
        float fogW = clamp(1.0f - exp(-fogDistW * (0.00072f + env.rainFactor * 0.0020f) * hazeScale), 0.0f, 0.72f);
        // V4c: Grazing-gated chunk-edge ocean horizon blend (erases jagged ocean cutoff while keeping Test N steep downward ocean rich blue!)
        float horizonFadeW = smoothstep(0.22f, 0.02f, abs(rayDir.y)) * smoothstep(105.0f, 185.0f, tWater) * 0.96f;
        fogW = max(fogW, horizonFadeW);
        if (fogW > 1e-4f) {
            float caveFogW = caveBlockedW ? 1.0f : (1.0f - smoothstep(0.02f, 0.25f, surfSkyVis));
            float3 airDirW = normalize(float3(rayDir.x, max(rayDir.y, 0.005f), rayDir.z));
            float3 airColW = (uFrame.dimensionId == 0)
                ? lerp(float3(sampleSkyLut(skyViewLut, airDirW)), float3(0.003f, 0.005f, 0.008f), caveFogW)
                : float3(0.003f, 0.005f, 0.008f);
            waterLit = lerp(waterLit, airColW, fogW);
        }
        return float4(max(waterLit, float3(0.0f, 0.0f, 0.0f)), 1.0f);
    }

    // -------------------------------------------------------------------------
    // 3. Sky Background Early Return (192x96 Sky-View LUT + 0.25x Smooth Clouds + 1.0x Crisp Celestials)
    // -------------------------------------------------------------------------
    if (isSky) {
        float3 skyCol;
        if (uFrame.dimensionId == 0) {
            half3 skyBase = sampleSkyLut(skyViewLut, rayDir);
            half4 cld = enableClouds ? sampleCloudsQres(cloudsQres, ds.centerUv) : half4(0.0h, 0.0h, 0.0h, 0.0h);
            half cloudTrans = saturate(half(1.0h) - cld.a);
            float3 softSky = float3(skyBase * cloudTrans + cld.rgb);
            skyCol = evalOverworldCrispCelestials(rayDir, uFrame, env, softSky, float(cloudTrans));
        } else {
            skyCol = evalDimensionSky(rayDir, uFrame, env, false);
        }
        if (uFrame.submergedMode != 0) {
            skyCol = applyMeteorUnderwaterFog(uFrame, env, skyCol, rayDir * 60.0f, camPos + rayDir * 60.0f, 60.0f);
        }
        return float4(max(skyCol, float3(0.0f, 0.0f, 0.0f)), 1.0f);
    }

    // -------------------------------------------------------------------------
    // 4. Geometry Reconstruction + Exact Integer Normals + Hardware-Derivative Micro-Bevel
    // -------------------------------------------------------------------------
    float dist = terrainT;
    float2 texelSz = float2(1.0f, 1.0f) / float2(ds.dim);
    int2 pxL = int2(max(ds.px.x - 1, 0), ds.px.y);
    int2 pxR = int2(min(ds.px.x + 1, ds.dim.x - 1), ds.px.y);
    int2 pxD = int2(ds.px.x, max(ds.px.y - 1, 0));
    int2 pxU = int2(ds.px.x, min(ds.px.y + 1, ds.dim.y - 1));
    float2 uvL = texelCenterUv(pxL, ds.dim);
    float2 uvR = texelCenterUv(pxR, ds.dim);
    float2 uvD = texelCenterUv(pxD, ds.dim);
    float2 uvU = texelCenterUv(pxU, ds.dim);

    float dL = sceneDepthTex.Load(int3(pxL, 0)).r;
    float dR = sceneDepthTex.Load(int3(pxR, 0)).r;
    float dD = sceneDepthTex.Load(int3(pxD, 0)).r;
    float dU = sceneDepthTex.Load(int3(pxU, 0)).r;

    float maxDepthJump = max(max(abs(dL - depth), abs(dR - depth)), max(abs(dD - depth), abs(dU - depth)));
    MeteorNormalDiffs nd = reconstructWorldNormalWithDiffs(uFrame, ds.centerUv, texelSz, relPos, dL, dR, dD, dU, uvL, uvR, uvD, uvU);
    float3 surfNormal = nd.nrm;

    // -------------------------------------------------------------------------
    // 5. Voxel Material Classification (Shared tight surface lookup)
    // -------------------------------------------------------------------------
    uint solidTag = 0u;
    uint spriteTag = 0u;
    uint frontTag = 0u;
    uint upTag = 0u;
    uint activeTag = 0u;
    uint activeSub = 0u;
    uint emitLvl = 0u;

    if (uFrame.voxelOrigin.w != 0) {
        MeteorSurfaceVoxel sv = sampleMeteorSurfaceVoxel(voxelGrid, vOrg, worldPos, rayDir, surfNormal);
        solidTag = sv.solidTag;
        spriteTag = sv.spriteTag;
        frontTag = sv.frontTag;
        upTag = sv.upTag;
        activeTag = sv.activeTag;
        activeSub = sv.activeSub;
        emitLvl = (sv.activeTag >= 14u) ? 0u : sv.emitLvl;
    }
    bool isPassThroughVeg = (activeTag == 14u && activeSub == 15u);

    MeteorPbrMaterial pbr = enablePbrRain ? decodePbrMaterial(activeTag, activeSub, rawSrgb, rawLin) : decodePbrMaterial(0u, 0u, rawSrgb, rawLin);

    // P5: Zero-extra-fetch Micro-Bevel using hardware quad derivatives (ddx/ddy) + reused dx/dy from normal reconstruction
    float3 shadedNormal = surfNormal;
    if (enablePbrRain && dist < 38.0f && activeTag != 8u && !isPassThroughVeg && maxDepthJump < max(depth * 0.015f, 1e-5f)) {
        float3 tanU = nd.dx - surfNormal * dot(nd.dx, surfNormal);
        float3 tanV = nd.dy - surfNormal * dot(nd.dy, surfNormal);
        float lenU = length(tanU);
        float lenV = length(tanV);
        if (lenU > 1e-5f && lenV > 1e-5f) {
            tanU /= lenU;
            tanV /= lenV;
            float bumpGain = pbr.active ? (pbr.bumpScale * 1.25f) : 0.38f;
            float3 deltaN = (tanU * (microGradX * 1.8f) + tanV * (microGradY * 1.8f)) * (bumpGain * (1.0f - smoothstep(3.0f, 36.0f, dist)));
            float deltaLen = length(deltaN);
            if (deltaLen > 0.16f) {
                deltaN *= (0.16f / deltaLen);
            }
            shadedNormal = normalize(surfNormal - deltaN);
        }
    }
    float3 specNormal = normalize(lerp(surfNormal, shadedNormal, 0.25f));

    // -------------------------------------------------------------------------
    // 6. Direct Sunlight (Denoised 0.5x Soft Stained-Glass SunVis) + Bilateral GI/BVH
    // -------------------------------------------------------------------------
    HalfResPairResultRt rtPair = sampleHalfResBilateralPairRt(rtGiStep2, rtSunStep2, sceneDepthTex, uFrame, ds.centerUv, relPos, surfNormal);
    float4 giShadow = rtPair.a;
    float3 sunVisRawRgb = rtPair.b;
    float3 giCol = giShadow.rgb;
    float rtSunVis = clamp(giShadow.a, 0.0f, 1.0f);
    float3 sunVisRgb = max(sunVisRawRgb, float3(0.0f));

    float ndotlKey = isPassThroughVeg
        ? max(abs(dot(shadedNormal, keyDir)) * 0.65f, max(keyDir.y * 0.72f, 0.0f))
        : max(dot(shadedNormal, keyDir), 0.0f);
    if (uFrame.dimensionId != -1 && enableShadows && rtSunVis > 0.01f && ndotlKey > 0.01f && dist < 56.0f) {
        uint entCount = min(uint(entityShadowCasters[0].x + 0.5f), 128u);
        if (entCount > 0u) {
            float entSunVis = evalEntitySunShadowCrisp(entityShadowCasters, relPos, relPos + surfNormal * 0.035f, keyDir, 48.0f);
            rtSunVis *= entSunVis;
            sunVisRgb *= entSunVis;
        }
    }
    float3 directSun = directCol * sunVisRgb * (ndotlKey * highlightBoost);
    if (uFrame.dimensionId == -1) {
        directSun = float3(0.0f, 0.0f, 0.0f);
    }

    // Minimal scotopic fill only; outdoor shadows & caves get real skylight from ray-traced giCol
    bool pixelUnderRoof = (uFrame.dimensionId == 0) && canSampleVoxels && isColumnSkyBlockedFast(voxelGrid, vOrg, worldPos + surfNormal * 0.08f);
    float3 skyAmbient = float3(0.008f, 0.012f, 0.020f) * (1.0f / max(shadowDepth, 0.8f)) * lerp(1.0f, 0.28f, env.nightFactor);
    if (uFrame.dimensionId == -1) {
        float3 crimsonAmb = float3(0.16f, 0.06f, 0.02f);
        float3 warpedAmb = float3(0.04f, 0.15f, 0.14f);
        float3 soulAmb = float3(0.05f, 0.18f, 0.22f);
        skyAmbient = lerp(lerp(crimsonAmb, warpedAmb, env.netherWarpedFactor), soulAmb, env.netherSoulFactor);
    } else if (uFrame.dimensionId == 1) {
        skyAmbient = float3(0.06f, 0.035f, 0.11f);
    }

    float3 diffuseAlbedo = pbr.active ? (rawLin * (1.0f - pbr.metallic * 0.88f)) : rawLin;
    float3 foliageSss = float3(0.0f, 0.0f, 0.0f);
    if ((activeTag == 2u || isPassThroughVeg) && uFrame.dimensionId == 0) {
        float sssGain = isPassThroughVeg ? 0.75f : 1.0f;
        float sssForward = pow(saturate(dot(rayDir, keyDir)), 2.8f);
        if (rtSunVis > 0.01f) {
            foliageSss += diffuseAlbedo * directCol * sunVisRgb
                        * (0.14f + 0.44f * sssForward) * float3(1.12f, 1.24f, 0.62f) * highlightBoost * sssGain;
        }
        if (!pixelUnderRoof) {
            float dayWrap = smoothstep(-0.04f, 0.25f, keyDir.y) * (1.0f - env.nightFactor * 0.92f) * (1.0f - env.rainFactor * 0.55f);
            float3 canopyFill = lerp(float3(0.095f, 0.145f, 0.105f), float3(0.145f, 0.112f, 0.065f), env.sunsetFactor * 0.65f);
            float backWrap = 0.08f + 0.22f * sssForward;
            foliageSss += diffuseAlbedo * (canopyFill * (1.0f - env.nightFactor * 0.88f) + directCol * (backWrap * 0.14f * dayWrap * float3(0.95f, 1.15f, 0.58f))) * sssGain;
        }
    }
    float3 lit = diffuseAlbedo * (directSun + giCol * 1.18f + skyAmbient) + foliageSss;
    if (uFrame.dimensionId == 0 && !pixelUnderRoof && env.nightFactor < 0.85f) {
        float outdoorSheen = smoothstep(0.06f, 0.24f, luminance(giCol)) * (1.0f - env.nightFactor) * 0.010f;
        lit += giCol * outdoorSheen;
    }

    // 6a. Hardware BVH Reflections (Zero sky-leak bilateral upsample & gated by ENABLE_REFLECTIONS)
    float3 viewDir = -rayDir;
    float3 reflDir = reflect(rayDir, specNormal);
    bool needSurfRefl = (activeTag == 8u) || (enablePbrRain && pbr.active) || (enablePbrRain && uFrame.dimensionId == 0 && env.rainFactor > 0.05f && shadedNormal.y > 0.78f) || (uFrame.dimensionId == 1 && shadedNormal.y > 0.72f);
    float4 bvhRefl = (enableRefl && needSurfRefl) ? sampleHalfResBilateralRt(rtReflectTemporal, sceneDepthTex, uFrame, ds.centerUv, relPos, specNormal, false) : float4(0.0f);
    float3 reflRadiance = giCol * 0.35f + float3(0.008f, 0.011f, 0.016f);
    if (bvhRefl.a > 0.05f) {
        reflRadiance = bvhRefl.rgb;
    } else if (enableRefl && needSurfRefl) {
        if (uFrame.dimensionId != 0) {
            reflRadiance = evalDimensionSkyFast(reflDir, uFrame, env, false);
        } else {
            float skyReflMask = max(rtSunVis, smoothstep(0.10f, 0.35f, luminance(giCol)) * (1.0f - env.nightFactor * 0.55f));
            if (skyReflMask > 0.04f && !pixelUnderRoof && reflDir.y > -0.05f) {
                float3 envRefl = float3(sampleSkyLut(enableClouds ? skyReflLut : skyViewLut, reflDir));
                reflRadiance = lerp(reflRadiance, envRefl, smoothstep(0.04f, 0.30f, skyReflMask));
            }
        }
    }

    if (activeTag == 8u) {
        float ndotv = saturate(dot(specNormal, viewDir));
        float glassFresnel = clamp((0.04f + 0.56f * pow(saturate(1.0f - ndotv), 5.0f)) * (enableRefl ? reflectionStrength : 0.25f), 0.02f, 0.58f);
        float3 glassTint = (activeSub >= 2u) ? decodeStainedGlassColor(activeSub) : float3(0.95f, 0.98f, 1.0f);
        float3 glassSpec = evalSpecularBrdfGgx(specNormal, viewDir, keyDir, 0.055f, float3(0.06f, 0.06f, 0.06f)) * directCol * sunVisRgb * specularStrength * highlightBoost;
        lit = lerp(lit * lerp(float3(1.0f), glassTint, 0.25f), reflRadiance * lerp(float3(1.0f), glassTint, 0.16f), glassFresnel) + glassSpec;
    } else if (enablePbrRain && pbr.active) {
        float ndotv = saturate(dot(specNormal, viewDir));
        float roughness = clamp(pbr.roughness, 0.04f, 0.92f);
        float3 baseF0 = lerp(float3(pbr.f0Scalar, pbr.f0Scalar, pbr.f0Scalar), max(rawLin, float3(0.08f)), pbr.metallic);
        float3 fMax = max(float3(1.0f - roughness, 1.0f - roughness, 1.0f - roughness), baseF0);
        float3 schlickF = baseF0 + (fMax - baseF0) * pow(saturate(1.0f - ndotv), 5.0f);
        float smoothFade = (1.0f - roughness) * (1.0f - roughness);
        float reflWeight = enableRefl ? clamp(pbr.reflectivity * reflectionStrength * smoothFade, 0.0f, 0.92f) : 0.0f;
        float3 specReflW = clamp(schlickF * (reflWeight * lerp(1.0f, 1.55f, pbr.metallic)), float3(0.0f), float3(0.95f));

        lit = lit * (float3(1.0f, 1.0f, 1.0f) - specReflW * (1.0f - pbr.metallic)) + reflRadiance * specReflW;
        lit += evalSpecularBrdfGgx(specNormal, viewDir, keyDir, roughness, baseF0) * directCol * sunVisRgb * specularStrength * highlightBoost * 1.65f;

        // Tile-Culled Voxel-Shadow-Tested Point EmissiveLights GGX Specular (GGX early-reject BEFORE DDA)
        uint lightCnt = min(uint(emissiveLights[0].x + 0.5f), 1024u);
        if (specularStrength > 0.01f && lightCnt > 0u) {
            uint tileOffP = getTileLightOffset(tileLightGrid, uv);
            uint loopCntP = (tileOffP >= 4u) ? min(tileLightGrid[tileOffP], 12u) : min(lightCnt, 12u);
            uint tracedP = 0u;
            float specRough = max(roughness, 0.10f);
            float3 domAxisSpec = snapCardinalAxis(specNormal);
            float3 specOff = canSampleVoxels ? safeShadowOriginOffset(voxelGrid, vOrg, worldPos, specNormal, domAxisSpec, viewDir, dist) : (domAxisSpec * 0.06f);
            float3 rayStartRel = relPos + specOff;
            float3 rayStartW = uFrame.camWorldPos.xyz + rayStartRel;
            for (uint ti = 0u; ti < loopCntP && tracedP < 6u; ++ti) {
                uint li = (tileOffP >= 4u) ? tileLightGrid[tileOffP + 1u + ti] : ti;
                if (li >= lightCnt) continue;
                float4 lp = emissiveLights[1u + li * 2u];
                float3 toL = lp.xyz - relPos;
                float d2 = dot(toL, toL);
                float maxR = evalPointLightMaxRange(lp.w);
                if (d2 >= maxR * maxR || d2 < 0.02f) continue;
                float dDist = sqrt(d2);
                float3 lDir = toL / dDist;
                if (dot(specNormal, lDir) <= 0.0f) continue;
                float rFade = evalPointLightWindow(dDist, maxR);
                float atten = (lp.w / (d2 + 0.85f)) * rFade * specularStrength;
                if (atten <= 0.0008f) continue;

                float3 specBrdf = evalSpecularBrdfGgx(specNormal, viewDir, lDir, specRough, baseF0);
                if (max(specBrdf.r, max(specBrdf.g, specBrdf.b)) * atten < 0.008f) continue;

                ++tracedP;
                float3 lVisCol = float3(1.0f, 1.0f, 1.0f);
                float3 toLStart = lp.xyz - rayStartRel;
                float dStart = length(toLStart);
                if (canSampleVoxels && dStart > 0.35f) {
                    int maxSteps = clamp(int(ceil(dStart * 1.50f)) + 2, 4, 24);
                    lVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                        voxelGrid, vOrg, rayStartW, toLStart / dStart, dStart, hasStainedGlass, maxSteps);
                }
                if (dot(lVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;

                float4 lc = emissiveLights[2u + li * 2u];
                lit += specBrdf * (lc.rgb * lVisCol) * atten;
            }
        }
    } else if (enablePbrRain && uFrame.dimensionId == 0 && env.rainFactor > 0.05f && shadedNormal.y > 0.78f) {
        float2 pw = evalRainPuddleMask(worldPos.xz, env.rainFactor);
        if (pw.y > 0.04f && (!canSampleVoxels || !isColumnOpaqueBlockedFast(voxelGrid, vOrg, worldPos))) {
            float2 ripGrad = evalRaindropRippleNormal(worldPos.xz, uFrame.timeSeconds, pw.x);
            float3 puddleNrm = normalize(float3(-ripGrad.x, 1.0f, -ripGrad.y));
            lit *= lerp(1.0f, 0.62f, pw.y * 0.65f + pw.x * 0.35f);
            float ndotvP = saturate(dot(puddleNrm, viewDir));
            float wetMix = 0.28f * pw.y + 0.72f * pw.x;
            float fresP = clamp((0.05f + 0.95f * pow(saturate(1.0f - ndotvP), 4.2f)) * wetMix * (enableRefl ? reflectionStrength : 0.20f), 0.0f, 0.92f);
            lit = lerp(lit, reflRadiance, fresP);

            if (specularStrength > 0.01f) {
                uint pLightCnt = min(uint(emissiveLights[0].x + 0.5f), 1024u);
                uint tileOffR = (pLightCnt > 0u) ? getTileLightOffset(tileLightGrid, uv) : 0u;
                uint pLoopCnt = (tileOffR >= 4u) ? min(tileLightGrid[tileOffR], 12u) : min(pLightCnt, 12u);
                uint pTraced = 0u;
                float pRough = lerp(0.24f, 0.055f, pw.x);
                float3 puddleOff = canSampleVoxels ? safeShadowOriginOffset(voxelGrid, vOrg, worldPos, shadedNormal, float3(0.0f, 1.0f, 0.0f), viewDir, dist) : float3(0.0f, 0.06f, 0.0f);
                float3 rayStartW = worldPos + puddleOff;
                for (uint ti = 0u; ti < pLoopCnt && pTraced < 6u; ++ti) {
                    uint li = (tileOffR >= 4u) ? tileLightGrid[tileOffR + 1u + ti] : ti;
                    if (li >= pLightCnt) continue;
                    float4 lp = emissiveLights[1u + li * 2u];
                    float3 toL = lp.xyz - relPos;
                    float d2 = dot(toL, toL);
                    float maxR = evalPointLightMaxRange(lp.w);
                    if (d2 >= maxR * maxR || d2 < 0.02f) continue;
                    float dDist = sqrt(d2);
                    float3 lDir = toL / dDist;
                    if (dot(puddleNrm, lDir) <= 0.0f) continue;
                    float rFade = evalPointLightWindow(dDist, maxR);
                    float atten = (lp.w / (d2 + 0.85f)) * rFade * (0.25f * pw.y + 0.75f * pw.x) * specularStrength * highlightBoost;
                    if (atten <= 0.0005f) continue;
                    float3 specBrdf = evalSpecularBrdfGgx(puddleNrm, viewDir, lDir, pRough, float3(0.04f, 0.04f, 0.04f));
                    if (max(specBrdf.r, max(specBrdf.g, specBrdf.b)) * atten < 0.008f) continue;
                    ++pTraced;
                    float3 lVisCol = float3(1.0f, 1.0f, 1.0f);
                    if (canSampleVoxels && dDist > 0.35f) {
                        int maxSteps = clamp(int(ceil(dDist * 1.50f)) + 2, 4, 24);
                        lVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                            voxelGrid, vOrg, rayStartW, lDir, dDist, hasStainedGlass, maxSteps);
                    }
                    if (dot(lVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                    float4 lc = emissiveLights[2u + li * 2u];
                    lit += specBrdf * (lc.rgb * lVisCol) * atten;
                }
            }
        }
    } else if (enableRefl && uFrame.dimensionId == 1 && shadedNormal.y > 0.72f) {
        float ndotvE = saturate(dot(shadedNormal, viewDir));
        float fresE = (0.04f + 0.96f * pow(saturate(1.0f - ndotvE), 4.5f)) * 0.28f * reflectionStrength;
        lit = lerp(lit, reflRadiance, clamp(fresE, 0.0f, 0.45f));
    }

    // -------------------------------------------------------------------------
    // 7. Semantic Emission (Lava, Magma, Redstone, Glowing Ores, Froglights, Sea Lantern, End)
    // -------------------------------------------------------------------------
    bool isEmissiveSurface = false;
    uint semSub = (activeTag == 12u) ? activeSub : emitLvl;
    float3 semEmit = (activeTag >= 14u) ? float3(0.0f) : evaluateSemanticEmission(activeTag, semSub, rawSrgb, rawLin, dist, uFrame);
    if (dot(semEmit, float3(1.0f, 1.0f, 1.0f)) > 1e-4f) {
        lit = max(lit, rawLin * 0.25f) + semEmit;
        isEmissiveSurface = true;
    }

    if (uFrame.dimensionId == 1 && !isEmissiveSurface && solidTag == 0u && spriteTag == 0u && dist < 110.0f) {
        if (max(rawSrgb.r, rawSrgb.b) > 0.35f && rawSrgb.b >= rawSrgb.g * 0.85f) {
            lit = rawLin * 4.2f + float3(0.65f, 0.25f, 1.05f) * 1.8f * max(emissiveScale, 0.5f);
            isEmissiveSurface = true;
        }
    }

    // Subtle underwater tip blend strictly for diagonal submerged seagrass/kelp blades poking above seaSurfaceY
    // inside a water voxel (never solid bank walls, waterlogged cardinal blocks, lily pads, or boats)
    if (uFrame.dimensionId == 0 && uFrame.submergedMode == 0 && canSampleVoxels &&
        worldPos.y > seaSurfaceY - 0.02f && worldPos.y <= uFrame.seaLevel + 0.02f &&
        solidTag == 3u && spriteTag == 3u &&
        abs(surfNormal.y) < 0.55f && abs(surfNormal.x) > 0.25f && abs(surfNormal.z) > 0.25f &&
        rawLin.g > rawLin.r * 1.08f && rawLin.g >= rawLin.b * 0.85f) {
        uint tipCellTag = sampleVoxelTag(voxelGrid, int3(floor(worldPos - float3(0.0f, 0.04f, 0.0f))) - vOrg);
        if (tipCellTag == 3u) {
            float3 tipLight = directCol * max(sunVisRgb, float3(0.0f)) * (0.22f * (1.0f - env.nightFactor * 0.90f)) + giCol * 0.65f;
            float3 subTint = float3(0.048f, 0.34f, 0.42f) * tipLight;
            lit = lerp(lit * float3(0.76f, 0.92f, 0.96f), subTint, 0.22f);
        }
    }

    // -------------------------------------------------------------------------
    // 8. Non-Milky Atmospheric Fog + Nether Ash Billboard Fix + Underwater Tint
    // -------------------------------------------------------------------------
    if (uFrame.dimensionId == 0) {
        float fogDist = max(dist - 76.0f, 0.0f);
        if (fogDist > 1e-4f) {
            bool undergroundCave = pixelUnderRoof && (rtSunVis < 0.05f);
            float heightMist = undergroundCave ? 0.0f : exp(-max(worldPos.y - (uFrame.seaLevel - 2.0f), 0.0f) * 0.048f);
            float valleyFog = 1.0f - exp(-fogDist * (0.00072f + 0.0011f * heightMist + env.rainFactor * 0.0024f) * hazeScale);
            float fogW = clamp(valleyFog, 0.0f, 0.72f) * (isEmissiveSurface ? 0.28f : 1.0f);
            if (fogW > 1e-4f) {
                float3 mistTint = float3(0.003f, 0.005f, 0.008f);
                if (!undergroundCave) {
                    float3 aerialCol = float3(sampleSkyLut(skyViewLut, normalize(relPos)));
                    mistTint = lerp(aerialCol, lerp(float3(0.46f, 0.62f, 0.84f), float3(1.08f, 0.52f, 0.24f), env.sunsetFactor), heightMist * 0.28f);
                    mistTint = lerp(mistTint, float3(0.03f, 0.055f, 0.12f), env.nightFactor * 0.85f);
                }
                lit = lerp(lit, mistTint, fogW);
            }
        }
    } else if (uFrame.dimensionId == -1) {
        float fogDist = max(dist - 8.0f, 0.0f);
        float density = lerp(0.0085f, 0.0125f, env.netherSoulFactor);
        float netherFogAmt = 1.0f - exp(-fogDist * density);
        float3 netherHaze = evalNetherSky(normalize(relPos), uFrame, env) * lerp(0.95f, 1.15f, env.netherSoulFactor);
        bool isTinyBillboard = (uFrame.voxelOrigin.w != 0 && dist < 96.0f && solidTag == 0u && spriteTag == 0u && frontTag == 0u && upTag == 0u && maxDepthJump > depth * 0.045f);
        if (isTinyBillboard && !isEmissiveSurface) {
            lit = lerp(max(lit, netherHaze * 0.72f + rawLin * 0.85f), netherHaze, 0.65f);
        } else {
            float fogW = clamp(netherFogAmt, 0.0f, 0.86f) * ((activeTag == 4u || isEmissiveSurface) ? 0.22f : 1.0f);
            lit = lerp(lit, netherHaze, fogW);
        }
    } else {
        float fogDist = max(dist - 18.0f, 0.0f);
        float endFogAmt = 1.0f - exp(-fogDist * 0.0040f);
        float3 endFogCol = float3(0.40f, 0.16f, 0.62f);
        lit = lerp(lit, endFogCol, clamp(endFogAmt, 0.0f, 0.76f) * (isEmissiveSurface ? 0.30f : 1.0f));
    }

    lit = applyMeteorUnderwaterFog(uFrame, env, lit, relPos, worldPos, dist);
    return float4(max(lit, float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

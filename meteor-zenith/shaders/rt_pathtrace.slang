// Meteor Shader Pass 1 — Half-Res Hardware BVH Ray Tracing (3D Liquid Wave + Ripple Reflections, PBR Metal/Glass Reflections, Stained-Glass Tinted NEE, 1-Bounce GI & 512x128x512 Voxel DDA Fallback)
module meteor_rt_pathtrace;
import metality;

#include "lib/meteor_common.slang"
#include "lib/meteor_sky_end_nether.slang"
#include "lib/meteor_water.slang"

ConstantBuffer<MetalityFrame> uFrame;
RaytracingAccelerationStructure sceneTlas;
StructuredBuffer<uint> quadMeta;
StructuredBuffer<uint> instanceMetaOffsets;
StructuredBuffer<uint> voxelGrid;
StructuredBuffer<float4> emissiveLights;
StructuredBuffer<float4> waterRipples;
StructuredBuffer<float4> entityShadowCasters;
StructuredBuffer<uint> tileLightGrid;
Texture2D<float4> sceneColorTex;
Texture2D<float4> sceneDepthTex;
Texture2D<half4> skyViewLut;
Texture2D<half4> skyReflLut;
RWTexture2D<float4> outGiShadow;
RWTexture2D<float4> outReflection;
RWTexture2D<float4> outSunVis;
RWTexture2D<half4> outGbuf;

float3 evalSmoothGlassGain(float3 tint) {
    float minC = min(tint.r, min(tint.g, tint.b));
    float maxC = max(tint.r, max(tint.g, tint.b));
    float sat = saturate((1.0f - minC) * 2.2f + (maxC - minC) * 1.6f);
    float3 rich = pow(max(tint, float3(0.02f, 0.02f, 0.02f)), float3(1.22f, 1.22f, 1.22f)) * 1.28f;
    return lerp(tint, rich, smoothstep(0.03f, 0.24f, sat));
}

float3 decodeRtFaceNormal(uint fdir) {
    if (fdir == 0u) return float3(-1.0f, 0.0f, 0.0f);
    if (fdir == 1u) return float3( 1.0f, 0.0f, 0.0f);
    if (fdir == 2u) return float3( 0.0f,-1.0f, 0.0f);
    if (fdir == 3u) return float3( 0.0f, 1.0f, 0.0f);
    if (fdir == 4u) return float3( 0.0f, 0.0f,-1.0f);
    return float3(0.0f, 0.0f, 1.0f);
}

[shader("compute")]
void csMeteorRayTrace(uint3 tid : SV_DispatchThreadID) {
    uint2 outDim = uint2(max(uFrame.width, 1u), max(uFrame.height, 1u));
    if (tid.x >= outDim.x || tid.y >= outDim.y) {
        return;
    }

    float2 uv = safeScreenUv(float2(tid.xy) + 0.5f, uFrame, 1.0f);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    float depth = ds.depth;
    bool isSky = isSkyDepth(depth);
    float3 skyRayDir = isSky ? reconstructSkyWorldDir(uFrame, ds.centerUv) : float3(0.0f);

    float3 rawSrgb = isSky ? float3(0.0f) : sceneColorTex.Load(int3(ds.px, 0)).rgb;
    float3 relPos = isSky ? (skyRayDir * 240.0f) : reconstructRelWorldPos(uFrame, ds.centerUv, depth);
    float3 worldPos = uFrame.camWorldPos.xyz + relPos;
    float camDist = length(relPos);
    float3 viewDir = normalize(-relPos);
    float3 rayDir = -viewDir;

    int3 vOrg = uFrame.voxelOrigin.xyz;

    // Fast Forward + Analytical Sea/Macro-DDA Water Detection (since Vanilla water mesh is suppressed)
    bool isWater = false;
    bool isWaterfallSurface = false;
    float2 waterFlowVec = float2(0.0f, 0.0f);
    float3 waterSurfWorld = worldPos;
    float3 waterSurfNrm = float3(0.0f, 1.0f, 0.0f);
    float3 nrm = float3(0.0f, 1.0f, 0.0f);
    uint voxelTag = 0u;
    uint voxelSub = 0u;
    uint frontTag = 0u;
    uint upTag = 0u;

    if (!isSky) {
        nrm = reconstructFastNormal(uFrame, ds.centerUv, relPos, sceneDepthTex);

        bool canSampleVoxelsHere = ((uFrame.voxelOrigin.w & 1) != 0);
        if (canSampleVoxelsHere) {
            MeteorSurfaceVoxel sv = sampleMeteorSurfaceVoxel(voxelGrid, vOrg, worldPos, rayDir, nrm);
            voxelTag = sv.activeTag;
            voxelSub = sv.activeSub;
            frontTag = sv.frontTag;
            upTag = sv.upTag;
        }
    }

    bool enableShadows = meteorFeatureShadows(uFrame);
    bool enableRefl = meteorFeatureReflections(uFrame);
    bool enableWater = meteorFeatureWater(uFrame);
    bool enableClouds = meteorFeatureClouds(uFrame);
    bool enableGiAo = meteorFeatureGiAo(uFrame);
    bool enablePbrRain = meteorFeaturePbrRain(uFrame);
    uint tier = meteorShadowTier(uFrame);

    bool canSampleVoxels = ((uFrame.voxelOrigin.w & 1) != 0);
    bool has3dWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & 2) != 0);
    bool hasWaterVoxels = (uFrame.dimensionId != -1) && canSampleVoxels && ((uFrame.voxelOrigin.w & (2 | 8)) != 0);
    bool hasStainedGlass = canSampleVoxels && ((uFrame.voxelOrigin.w & 4) != 0);

    if (enableWater && hasWaterVoxels && uFrame.submergedMode == 0) {
        // Stained glass / ice windows as the primary raster surface must never be overridden by analytical water fallbacks
        bool primaryIsGlass = (!isSky && voxelTag == 8u);
        bool waterRayBlocked = false;

        // 1. Analytical O(1) intersection with Sea-Level Plane (y = seaLevel - 0.112)
        float seaSurfY = uFrame.seaLevel - 0.112f;
        int seaBlockY = int(floor(uFrame.seaLevel)) - 1;
        if (!primaryIsGlass && uFrame.camWorldPos.y > seaSurfY + 0.02f && rayDir.y < -0.0005f) {
            float tSea = (seaSurfY - uFrame.camWorldPos.y) / rayDir.y;
            if (tSea > 0.05f && (isSky || tSea < camDist - 0.02f)) {
                float3 seaHitPos = uFrame.camWorldPos.xyz + rayDir * tSea;
                float3 pBelow = seaHitPos - float3(0.0f, 0.08f, 0.0f);
                int3 cBelow = int3(floor(pBelow)) - vOrg;
                bool hitSea = false;
                if (has3dWaterVoxels && inVoxelBounds512Common(cBelow)) {
                    uint idx = voxelLinearIndex512Common(cBelow);
                    uint tBelow = ((voxelGrid[idx >> 2u] >> ((idx & 3u) * 8u)) >> 4u) & 0xFu;
                    hitSea = (tBelow == 3u);
                } else {
                    hitSea = isFarOrHighSeaWaterAt(voxelGrid, vOrg, seaHitPos, seaBlockY);
                }
                if (hitSea) {
                    isWater = true;
                    waterSurfWorld = seaHitPos;
                    waterSurfNrm = float3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        // 2. Unified Macro-DDA + Bilinear Sloped Flowing / Waterfall / Deep Pool Detection
        if (!isWater && has3dWaterVoxels && rayOverlapsWaterYBounds(uFrame, uFrame.camWorldPos.y, worldPos.y, rayDir, isSky)) {
            WaterDdaHit wh = traceWaterMacroDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz, rayDir, 0.05f, isSky ? 224.0f : min(camDist, 224.0f), uFrame.voxelOrigin.w);
            waterRayBlocked = wh.blocked;
            if (wh.hit) {
                isWater = true;
                isWaterfallSurface = wh.isVertical;
                waterFlowVec = wh.flowVec;
                waterSurfWorld = uFrame.camWorldPos.xyz + rayDir * wh.t;
                waterSurfNrm = wh.nrm;
            }
        }

        // 3. High-altitude elevated pool/river surface fallback via top_water_hmap strictly outside the 3D Y slice
        if (!isWater && !waterRayBlocked && !primaryIsGlass && !isSky && rayDir.y < -0.0005f) {
            int2 hitXZ = int2(floor(worldPos.xz));
            int topWaterY = sampleTopWaterWorldY(voxelGrid, vOrg, hitXZ);
            if (topWaterY > -9000 && float(topWaterY) + 0.888f > worldPos.y + 0.12f) {
                float elevSurfY = float(topWaterY) + 0.888f;
                if (uFrame.camWorldPos.y > elevSurfY + 0.02f) {
                    float tElev = (elevSurfY - uFrame.camWorldPos.y) / rayDir.y;
                    if (tElev > 0.05f && tElev < camDist - 0.02f) {
                        float3 elevHit = uFrame.camWorldPos.xyz + rayDir * tElev;
                        if (!inVoxelBounds512Common(int3(floor(elevHit)) - vOrg)) {
                            int2 eXZ = int2(floor(elevHit.xz));
                            int eWaterY = sampleTopWaterWorldY(voxelGrid, vOrg, eXZ);
                            int eOpaqueY = sampleTopOpaqueWorldY(voxelGrid, eXZ.x - vOrg.x, eXZ.y - vOrg.z);
                            if (abs(eWaterY - topWaterY) <= 1 && eOpaqueY <= eWaterY + 1) {
                                isWater = true;
                                waterSurfWorld = elevHit;
                                waterSurfNrm = float3(0.0f, 1.0f, 0.0f);
                            }
                        }
                    }
                }
            }
        }
    }

    if (isSky && !isWater) {
        outGiShadow.write(float4(0.0f, 0.0f, 0.0f, -2.0f), tid.xy);
        outReflection.write(float4(0.0f, 0.0f, 0.0f, 0.0f), tid.xy);
        outSunVis.write(float4(0.0f, 0.0f, 0.0f, 1.0f), tid.xy);
        outGbuf.write(half4(0.0h, 0.0h, 1.0h, 0.0h), tid.xy);
        return;
    }

    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    float3 sunDir = normalize(uFrame.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 keyDir = sunDir;
    float3 directCol = lerp(float3(1.74f, 1.58f, 1.36f), float3(2.42f, 1.04f, 0.37f), env.sunsetFactor);
    directCol = lerp(directCol, float3(0.115f, 0.195f, 0.360f), env.nightFactor);
    directCol = lerp(directCol, float3(0.41f, 0.46f, 0.53f), env.rainFactor * 0.82f);

    if (sunDir.y < -0.05f && uFrame.dimensionId == 0) {
        keyDir = normalize(uFrame.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    }
    if (uFrame.dimensionId == 1) {
        keyDir = getMeteorBlackHoleDir();
        directCol = float3(1.65f, 1.25f, 2.05f);
    }

    float3 netherAmb = lerp(float3(0.56f, 0.18f, 0.05f), float3(0.72f, 0.12f, 0.08f), env.netherCrimsonFactor);
    netherAmb = lerp(netherAmb, float3(0.16f, 0.54f, 0.58f), env.netherSoulFactor);
    netherAmb = lerp(netherAmb, float3(0.12f, 0.58f, 0.52f), env.netherWarpedFactor);
    float3 skyAmb = (uFrame.dimensionId == 0)
        ? lerp(float3(0.29f, 0.47f, 0.77f), float3(0.45f, 0.23f, 0.42f), env.sunsetFactor) * lerp(1.0f, 0.072f, env.nightFactor)
        : ((uFrame.dimensionId == 1)
            ? float3(0.36f, 0.16f, 0.54f)
            : netherAmb);

    bool isPassThroughVeg = (!isSky && voxelTag == 14u && voxelSub == 15u);
    float3 traceNrm = isPassThroughVeg
        ? normalize(nrm * 0.35f + float3(0.0f, 0.94f, 0.0f))
        : nrm;
    float3 domAxis = snapCardinalAxis(traceNrm);
    float3 safeOriginOff = canSampleVoxels
        ? safeShadowOriginOffset(voxelGrid, vOrg, worldPos, traceNrm, domAxis, viewDir, camDist)
        : (traceNrm * 0.020f);
    float distBias = clamp(0.016f + camDist * 0.0010f, 0.016f, 0.042f);
    float3 traceOrigin = relPos + (canSampleVoxels ? safeOriginOff : (traceNrm * distBias));
    MeteorPbrMaterial pbr = enablePbrRain ? decodePbrMaterial(voxelTag, voxelSub, rawSrgb, srgbToLinear(rawSrgb)) : decodePbrMaterial(0u, 0u, rawSrgb, srgbToLinear(rawSrgb));

    // Spatially blue-noise-like Interleaved Gradient Noise + golden-ratio temporal offset
    float frameMod = uFrame.camWorldPos.w;
    float2 ign = ignPair(float2(tid.xy), frameMod);
    float r0 = ign.x;
    float r1 = ign.y;

    float shadowDepth = clamp(uFrame.lightingParams.y, 0.5f, 2.0f);
    float shadowVis = (uFrame.dimensionId == -1) ? 0.0f : 1.0f;
    float3 sunVisRgb = (uFrame.dimensionId == -1) ? float3(0.0f, 0.0f, 0.0f) : float3(1.0f, 1.0f, 1.0f);
    float skyOpenFrac = 0.0f;
    float3 giAccum = float3(0.0f, 0.0f, 0.0f);
    float3 neeAccum = float3(0.0f, 0.0f, 0.0f);
    float4 reflOut = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Tile-Culled Emissive Point Light Setup (32x32 screen tiles, deterministic normal-aware split-depth Top-63)
    uint lightCount = min(uint(emissiveLights[0].x + 0.5f), 1024u);
    uint tileOff = (lightCount > 0u && !isSky) ? getTileLightOffset(tileLightGrid, uv) : 0u;
    uint tileCnt = (tileOff >= 4u) ? min(tileLightGrid[tileOff], 63u) : min(lightCount, 63u);
    uint maxTracedLights = (tier >= 2u) ? 8u : 6u;
    uint maxLitLights = 6u;

    // Open-ocean deep-water seabed skip (>9m below water surface Beer-Lambert extinguishes terrain bounce/NEE)
    bool deepOceanFloor = isWater && !isWaterfallSurface && !isSky && ((waterSurfWorld.y - worldPos.y) > 9.0f);

    if (uFrame.rtHardwareActive != 0u && uFrame.rtInstanceCount > 0u) {
        intersector<triangle_data, instancing> shadowInter;
        shadowInter.assume_geometry_type(geometry_type::triangle);
        shadowInter.force_opacity(forced_opacity::opaque);
        shadowInter.accept_any_intersection(true);

        intersector<triangle_data, instancing> closestInter;
        closestInter.assume_geometry_type(geometry_type::triangle);
        closestInter.force_opacity(forced_opacity::opaque);

        if (!isSky && deepOceanFloor) {
            shadowVis = (uFrame.dimensionId == -1) ? 0.0f : (smoothstep(0.0f, 0.25f, keyDir.y) * 0.65f);
            sunVisRgb = float3(shadowVis);
            giAccum = skyAmb * 0.28f;
        } else if (!isSky) {
            // 1. Stratified Soft Penumbra Hardware BVH + Articulated Entity Shadows + Jittered Stained-Glass Sun Tint
            if (uFrame.dimensionId != -1) {
                bool isBacklitFoliage = (voxelTag == 2u && dot(nrm, keyDir) <= 0.0f && keyDir.y > 0.02f);
                if (!enableShadows) {
                    shadowVis = smoothstep(0.0f, 0.25f, keyDir.y) * smoothstep(-0.10f, 0.35f, traceNrm.y) * 0.85f;
                    sunVisRgb = float3(shadowVis);
                    if (hasStainedGlass && canSampleVoxels && shadowVis > 0.005f && keyDir.y > 0.02f) {
                        float3 sTint = traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + traceOrigin, keyDir, 36.0f, 36, true);
                        sunVisRgb = evalSmoothGlassGain(sTint) * shadowVis;
                    }
                } else if (dot(nrm, keyDir) <= 0.0f && !isBacklitFoliage && !isPassThroughVeg) {
                    shadowVis = 0.0f;
                    sunVisRgb = float3(0.0f);
                } else {
                    float3 sUp = abs(keyDir.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                    float3 sT = normalize(cross(sUp, keyDir));
                    float3 sB = cross(keyDir, sT);
                    float coneRad = lerp(0.032f, 0.060f, env.rainFactor);
                    float3 sunRayOrigin = isPassThroughVeg
                        ? (traceOrigin + keyDir * 0.18f + float3(0.0f, 0.08f, 0.0f))
                        : (isBacklitFoliage ? (relPos + keyDir * 1.05f) : traceOrigin);
                    float phi = r0 * METEOR_TAU;
                    float rad = coneRad * sqrt(clamp(0.18f + 0.82f * r1, 0.0f, 1.0f));
                    float3 sDir = normalize(keyDir + (sT * cos(phi) + sB * sin(phi)) * rad);
                    // Entity sun shadows are evaluated crisply at full 1.0x resolution in rt_world_composite.slang
                    // (zero 0.5x À-Trous blur halo and zero temporal ghosting on moving players/mobs)
                    bool blocked = false;
                    {
                        ray sray;
                        sray.origin = sunRayOrigin;
                        sray.direction = sDir;
                        sray.min_distance = 0.006f;
                        sray.max_distance = 280.0f;
                        auto shit = shadowInter.intersect(sray, sceneTlas);
                        if (shit.type != intersection_type::none) {
                            blocked = true;
                        }
                    }
                    float visSum = blocked ? 0.0f : 1.0f;
                    float3 sunRgbSum = float3(0.0f);
                    if (!blocked) {
                        if (hasStainedGlass && canSampleVoxels && keyDir.y > 0.02f) {
                            float3 sTint = traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + sunRayOrigin, sDir, 36.0f, 36, true);
                            sunRgbSum = evalSmoothGlassGain(sTint);
                        } else {
                            sunRgbSum = float3(1.0f, 1.0f, 1.0f);
                        }
                    }
                    float backlitAtten = isBacklitFoliage ? 0.55f : 1.0f;
                    shadowVis = visSum * backlitAtten;
                    sunVisRgb = sunRgbSum * backlitAtten;
                }
            } else {
                shadowVis = 0.0f;
                sunVisRgb = float3(0.0f);
            }

            // 2. Tile-Culled NEE Point Light Shadowing (Decoupled cheap geometric scan up to 63, capped BVH/DDA trace & lit budgets)
            uint entTestedLights = 0u;
            uint tracedLights = 0u;
            uint wrapTraced = 0u;
            uint litLights = 0u;
            uint bestDirectLi = 0xFFFFFFFFu;
            float bestDirectW = 0.0f;
            for (uint ti = 0u; ti < tileCnt && tracedLights < maxTracedLights && litLights < maxLitLights; ++ti) {
                uint lIdx = (tileOff >= 4u) ? tileLightGrid[tileOff + 1u + ti] : ti;
                if (lIdx >= lightCount) continue;
                float4 lp = emissiveLights[1u + 2u * lIdx];
                float3 lDiff = lp.xyz - relPos;
                float dist2 = dot(lDiff, lDiff);
                float maxRange = evalPointLightMaxRange(lp.w);
                if (dist2 >= maxRange * maxRange || dist2 < 0.005f) continue;

                float dist = sqrt(dist2);
                float rangeFade = evalPointLightWindow(dist, maxRange);
                float3 lDir = lDiff / dist;
                float geomTerm = evalPointLightGeomTerm(nrm, lDir, dist, voxelTag, voxelSub, isWater);
                float ndL = dot(nrm, lDir);
                float wrapBounce = (dist > 0.65f && ndL > -0.45f)
                    ? (0.16f * smoothstep(-0.45f, 0.15f, ndL) * (1.0f - geomTerm))
                    : 0.0f;
                float effGeom = geomTerm + wrapBounce;
                if (effGeom <= 0.0001f) continue;
                float baseW = (lp.w / (dist2 + 1.15f)) * rangeFade * effGeom;
                if (baseW < 0.0012f) continue;

                // Keep back-facing wrap-only candidates on a tiny separate cap (and skip once already lit) so they never starve front-facing lights or waste cave rays
                bool isWrapOnly = (geomTerm <= 0.0001f);
                if (isWrapOnly) {
                    if (litLights > 0u || wrapTraced >= 1u) continue;
                    ++wrapTraced;
                } else {
                    ++tracedLights;
                }
                float4 lc = emissiveLights[2u + 2u * lIdx];
                bool glassBackLit = (voxelTag == 8u && ndL < -0.05f);
                float3 lShadowOrigin = glassBackLit ? (relPos - domAxis * 0.04f) : (relPos + safeOriginOff);
                float3 toLStart = lp.xyz - lShadowOrigin;
                float dStart = length(toLStart);
                float3 lightCol = lc.rgb;

                if (dStart > 0.35f) {
                    float3 lDirStart = toLStart / dStart;
                    float3 lUp = abs(lDirStart.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                    float3 lT = normalize(cross(lUp, lDirStart));
                    float3 lB = cross(lDirStart, lT);
                    float jPhi = frac(r0 + float(ti + 1u) * 0.381966f) * METEOR_TAU;
                    float jU = frac(r1 + float(ti + 1u) * 0.618034f);
                    float lightRad = clamp(lc.w, 0.18f, 1.60f);
                    float diskR = min(clamp(lightRad * 0.22f, 0.04f, 0.24f) * sqrt(max(jU, 0.04f)), dStart * 0.12f);
                    float3 jitOff = (lT * cos(jPhi) + lB * sin(jPhi)) * diskR;
                    float3 jitTargetRel = lp.xyz + jitOff;
                    if (canSampleVoxels && diskR > 0.015f && isPointBlockedInVoxel(voxelGrid, vOrg, uFrame.camWorldPos.xyz + jitTargetRel)) {
                        jitTargetRel = lp.xyz;
                    }
                    float3 toLJit = jitTargetRel - lShadowOrigin;
                    float dJit = max(length(toLJit), 0.10f);
                    float3 jitDir = toLJit / dJit;

                    bool useHwBvh = (dStart < 8.0f && !isWrapOnly && tracedLights <= 4u) || !canSampleVoxels;
                    if (useHwBvh) {
                        float entTraceLen = max(dJit - ((lc.w > 0.8f) ? 0.65f : 0.20f), 0.0f);
                        float traceLen = max(dJit - 0.08f, 0.0f);
                        float selfMargin = (lc.w > 0.55f) ? max(0.88f, lc.w * 1.10f) : 0.32f;
                        float lVis = 1.0f;
                        float entVis = 1.0f;
                        if (dStart < 10.0f && entTestedLights < 2u && entTraceLen > 0.05f) {
                            ++entTestedLights;
                            entVis = evalEntityShadowSoftVis(entityShadowCasters, relPos, lShadowOrigin, jitDir, lp.xyz, entTraceLen, lc.w);
                        }
                        if (entVis <= 0.01f) {
                            lVis = 0.0f;
                        } else if (traceLen > 0.05f) {
                            float anyHitMax = max(dJit - selfMargin, 0.0f);
                            if (anyHitMax > 0.02f) {
                                ray lAnyRay;
                                lAnyRay.origin = lShadowOrigin;
                                lAnyRay.direction = jitDir;
                                lAnyRay.min_distance = 0.006f;
                                lAnyRay.max_distance = anyHitMax;
                                if (shadowInter.intersect(lAnyRay, sceneTlas).type != intersection_type::none) {
                                    lVis = 0.0f;
                                }
                            }
                            if (lVis > 0.5f && lc.w > 0.55f && dJit < 2.2f && traceLen > max(anyHitMax, 0.006f) + 0.02f) {
                                ray lray;
                                lray.origin = lShadowOrigin;
                                lray.direction = jitDir;
                                lray.min_distance = max(anyHitMax, 0.006f);
                                lray.max_distance = traceLen;
                                auto lhit = closestInter.intersect(lray, sceneTlas);
                                if (lhit.type != intersection_type::none) {
                                    uint lqIdx = instanceMetaOffsets[lhit.instance_id] + (lhit.primitive_id >> 1u);
                                    uint lHitEmit = (quadMeta[lqIdx] >> 3u) & 0xFu;
                                    if (lHitEmit == 0u || (dJit - lhit.distance) > selfMargin) {
                                        lVis = 0.0f;
                                    }
                                }
                            }
                        }
                        if (lVis <= 0.01f) continue;
                        lightCol *= entVis;
                        if (hasStainedGlass && canSampleVoxels) {
                            int gSteps = clamp(int(ceil(dJit * 1.85f)) + 3, 4, 28);
                            lightCol *= traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + lShadowOrigin, jitDir, dJit + 0.08f, gSteps);
                        }
                    } else {
                        int maxSteps = clamp(int(ceil(dJit * 1.65f)) + 3, 4, 36);
                        lightCol *= tracePointLightVoxelVisibilityAndGlassDda(
                            voxelGrid, vOrg, uFrame.camWorldPos.xyz + lShadowOrigin, jitDir, dJit, hasStainedGlass, maxSteps);
                        if (dot(lightCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                    }
                }
                neeAccum += lightCol * baseW;
                if (!isWrapOnly) {
                    ++litLights;
                    if (baseW > bestDirectW) {
                        bestDirectW = baseW;
                        bestDirectLi = lIdx;
                    }
                }
            }
            neeAccum = neeAccum / (float3(1.0f) + max(neeAccum, float3(0.0f)) * 0.065f);

            // 3. Physically Accurate 1-Bounce Colored Global Illumination (Gated by ENABLE_GI_AO)
            if (!enableGiAo) {
                if (!enableShadows) {
                    giAccum = skyAmb * ((uFrame.dimensionId == -1) ? 0.25f : 0.14f) + neeAccum;
                } else {
                    float skyOpenNoGi = (uFrame.dimensionId == 0 && canSampleVoxels)
                        ? evalSmoothSkyOpennessFast(voxelGrid, vOrg, worldPos + safeOriginOff, traceNrm, ign)
                        : ((uFrame.dimensionId == -1) ? 0.0f : 1.0f);
                    float hemiSkyNoGi = clamp(traceNrm.y * 0.26f + 0.74f, 0.44f, 1.0f);
                    if (voxelTag == 2u || isPassThroughVeg) {
                        hemiSkyNoGi = max(hemiSkyNoGi, 0.62f);
                    }
                    float3 skyBaseNoGi = lerp(float3(0.35f, 0.47f, 0.68f), float3(0.47f, 0.31f, 0.26f), env.sunsetFactor * 0.65f);
                    skyBaseNoGi = lerp(skyBaseNoGi, float3(0.021f, 0.038f, 0.094f), env.nightFactor);
                    skyBaseNoGi = lerp(skyBaseNoGi, float3(0.135f, 0.175f, 0.23f), env.rainFactor * 0.82f);
                    float3 ambNoGi = (uFrame.dimensionId == 0)
                        ? (skyBaseNoGi * (hemiSkyNoGi * skyOpenNoGi * 0.84f) + float3(0.004f, 0.006f, 0.009f))
                        : (skyAmb * ((uFrame.dimensionId == -1) ? 0.25f : 0.18f));
                    giAccum = ambNoGi + neeAccum;
                }
            } else {
                float3 nUp = abs(traceNrm.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                float3 nT = normalize(cross(nUp, traceNrm));
                float3 nB = cross(traceNrm, nT);

                float3 giSum = float3(0.0f, 0.0f, 0.0f);
                float skyHits = 0.0f;
                float phi = r0 * METEOR_TAU;
                float cosTheta = sqrt(max(1.0f - r1, 0.01f));
                float sinTheta = sqrt(max(r1, 0.0f));
                float3 giDir = normalize(traceNrm * cosTheta + nT * (cos(phi) * sinTheta) + nB * (sin(phi) * sinTheta));

                ray bray;
                bray.origin = traceOrigin;
                bray.direction = giDir;
                bray.min_distance = 0.006f;
                bray.max_distance = 24.0f;
                auto bhit = closestInter.intersect(bray, sceneTlas);
                if (bhit.type != intersection_type::none && bhit.distance < 0.06f) {
                    uint qIdxPre = instanceMetaOffsets[bhit.instance_id] + (bhit.primitive_id >> 1u);
                    uint metaPre = quadMeta[qIdxPre];
                    if ((metaPre & 0x80u) == 0u && dot(giDir, decodeRtFaceNormal(metaPre & 7u)) > 0.15f) {
                        bray.origin = traceOrigin + giDir * (bhit.distance + 0.025f);
                        bhit = closestInter.intersect(bray, sceneTlas);
                    }
                }

                if (bhit.type != intersection_type::none) {
                    uint qIdx = instanceMetaOffsets[bhit.instance_id] + (bhit.primitive_id >> 1u);
                    uint meta = quadMeta[qIdx];
                    uint fdir = meta & 7u;
                    bool hitIsFoliage = ((meta & 0x80u) != 0u);
                    float hitEmissive = float((meta >> 3u) & 0xFu) / 15.0f;
                    float3 hitAlbedo = float3(float((meta >> 8u) & 0xFFu),
                                              float((meta >> 16u) & 0xFFu),
                                              float((meta >> 24u) & 0xFFu)) / 255.0f * 0.85f + 0.06f;
                    float3 hitNrm = decodeRtFaceNormal(fdir);
                    if (dot(giDir, hitNrm) > 0.0f && !hitIsFoliage) {
                        giSum += float3(0.004f, 0.005f, 0.007f);
                    } else {
                        if (dot(giDir, hitNrm) > 0.0f && hitIsFoliage) {
                            hitNrm = -hitNrm;
                        }
                        float3 hitPos = bray.origin + giDir * bhit.distance + hitNrm * 0.025f;

                        float hitSunVis = (uFrame.dimensionId == -1) ? 0.35f : 0.0f;
                        if (uFrame.dimensionId != -1 && dot(hitNrm, keyDir) > 0.05f) {
                            ray bShadow;
                            bShadow.origin = hitPos;
                            bShadow.direction = keyDir;
                            bShadow.min_distance = 0.006f;
                            bShadow.max_distance = 64.0f;
                            if (shadowInter.intersect(bShadow, sceneTlas).type == intersection_type::none) {
                                hitSunVis = 1.0f;
                            }
                        }

                        // O(1) heightmap sky check on shaded bounce surfaces (sealed caves stay dark, outdoor canopies get skylight)
                        float3 hitAmbCol = (uFrame.dimensionId == -1) ? (skyAmb * 0.28f) : float3(0.008f, 0.011f, 0.016f);
                        bool hitUnderSolidRoof = false;
                        if (uFrame.dimensionId == 0) {
                            if (hitSunVis > 0.5f) {
                                hitAmbCol = skyAmb * 0.52f;
                            } else if (canSampleVoxels) {
                                hitUnderSolidRoof = isColumnSkyBlockedFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos);
                                if (!hitUnderSolidRoof) {
                                    hitAmbCol = skyAmb * 0.44f
                                              + directCol * float3(0.065f, 0.060f, 0.044f) * (1.0f - env.nightFactor * 0.88f) * (1.0f - env.rainFactor * 0.65f);
                                }
                            } else if (bhit.distance < 16.0f) {
                                float3 skyProbeDir = normalize(hitNrm * 0.35f + float3(0.0f, 0.94f, 0.0f));
                                ray skyProbe;
                                skyProbe.origin = hitPos;
                                skyProbe.direction = skyProbeDir;
                                skyProbe.min_distance = 0.015f;
                                skyProbe.max_distance = 44.0f;
                                if (shadowInter.intersect(skyProbe, sceneTlas).type == intersection_type::none) {
                                    hitAmbCol = skyAmb * 0.44f;
                                }
                            }
                        } else if (uFrame.dimensionId == 1) {
                            hitAmbCol = skyAmb * 0.26f;
                        }

                        // Shadow-tested local point-light bounce onto nearby indoor/night walls (top-6 tile + primary visible light, 1 shadow ray)
                        float3 localNeeBounce = float3(0.0f);
                        float bounceProx = 1.0f - smoothstep(6.0f, 14.0f, bhit.distance);
                        if (bounceProx > 0.08f && tileCnt > 0u && (hitUnderSolidRoof || uFrame.dimensionId == -1 || env.nightFactor > 0.25f)) {
                            uint bScan = min(tileCnt, 6u);
                            uint bestLi0 = 0xFFFFFFFFu;
                            uint bestLi1 = 0xFFFFFFFFu;
                            float bestW0 = 0.0f;
                            float bestW1 = 0.0f;
                            bool seenDirect = false;
                            for (uint bti = 0u; bti <= bScan; ++bti) {
                                uint candLi;
                                if (bti < bScan) {
                                    candLi = (tileOff >= 4u) ? tileLightGrid[tileOff + 1u + bti] : bti;
                                    if (candLi == bestDirectLi) seenDirect = true;
                                } else {
                                    if (seenDirect || bestDirectLi == 0xFFFFFFFFu) break;
                                    candLi = bestDirectLi;
                                }
                                if (candLi >= lightCount) continue;
                                float4 clp = emissiveLights[1u + 2u * candLi];
                                float3 cToL = clp.xyz - hitPos;
                                float cd2 = dot(cToL, cToL);
                                float cMaxR = evalPointLightMaxRange(clp.w);
                                if (cd2 >= cMaxR * cMaxR || cd2 < 0.005f) continue;
                                float cd = sqrt(cd2);
                                float cnd = dot(hitNrm, cToL / cd);
                                if (cnd <= 0.001f) continue;
                                float cw = (clp.w / (cd2 + 1.5f)) * cnd * evalPointLightWindow(cd, cMaxR);
                                if (cw > bestW0) {
                                    bestW1 = bestW0;
                                    bestLi1 = bestLi0;
                                    bestW0 = cw;
                                    bestLi0 = candLi;
                                } else if (cw > bestW1) {
                                    bestW1 = cw;
                                    bestLi1 = candLi;
                                }
                            }
                            uint bLi = bestLi0;
                            float stratGain = 1.0f;
                            if (bestLi1 != 0xFFFFFFFFu && bestW1 > 0.0015f) {
                                float sumW = bestW0 + bestW1;
                                float pickProb0 = clamp(bestW0 / max(sumW, 1e-5f), 0.15f, 0.85f);
                                if (r0 > pickProb0) {
                                    bLi = bestLi1;
                                    stratGain = min(1.0f / max(1.0f - pickProb0, 0.15f), 2.2f);
                                } else {
                                    stratGain = min(1.0f / max(pickProb0, 0.15f), 2.2f);
                                }
                            }
                            if (bLi != 0xFFFFFFFFu) {
                                float4 lp0 = emissiveLights[1u + 2u * bLi];
                                float3 toL0 = lp0.xyz - hitPos;
                                float d20 = dot(toL0, toL0);
                                float maxR0 = evalPointLightMaxRange(lp0.w);
                                float d0 = sqrt(max(d20, 0.005f));
                                float3 lDir0 = toL0 / d0;
                                float nd0 = max(dot(hitNrm, lDir0), 0.0f);
                                float rFade0 = evalPointLightWindow(d0, maxR0);
                                float bAtten = (lp0.w / (d20 + 1.5f)) * nd0 * rFade0 * bounceProx * 0.72f * stratGain;
                                if (bAtten > 0.0008f) {
                                    float4 lc0 = emissiveLights[2u + 2u * bLi];
                                    float3 bVisCol = float3(1.0f, 1.0f, 1.0f);
                                    if (canSampleVoxels && isPointBlockedInVoxel(voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos)) {
                                        bVisCol = float3(0.0f, 0.0f, 0.0f);
                                    } else if (canSampleVoxels && d0 > 0.35f) {
                                        int maxSteps = clamp(int(ceil(d0 * 1.65f)) + 3, 4, 24);
                                        bVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                                            voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos, lDir0, d0, hasStainedGlass, maxSteps);
                                    } else if (!canSampleVoxels) {
                                        float bTraceLen = d0 - 0.12f;
                                        if (bTraceLen > 0.06f) {
                                            ray bLightRay;
                                            bLightRay.origin = hitPos;
                                            bLightRay.direction = lDir0;
                                            bLightRay.min_distance = 0.02f;
                                            bLightRay.max_distance = bTraceLen;
                                            auto bLHit = closestInter.intersect(bLightRay, sceneTlas);
                                            if (bLHit.type != intersection_type::none) {
                                                uint bqIdx = instanceMetaOffsets[bLHit.instance_id] + (bLHit.primitive_id >> 1u);
                                                uint bHitEmit = (quadMeta[bqIdx] >> 3u) & 0xFu;
                                                if (bHitEmit == 0u || (d0 - bLHit.distance) > max(0.92f, lc0.w * 1.15f)) {
                                                    bVisCol = float3(0.0f);
                                                }
                                            }
                                        }
                                    }
                                    if (dot(bVisCol, float3(1.0f, 1.0f, 1.0f)) > 0.001f) {
                                        float3 rawBounce = lc0.rgb * bVisCol * bAtten;
                                        localNeeBounce = rawBounce / (float3(1.0f) + max(rawBounce, float3(0.0f)) * 0.35f);
                                    }
                                }
                            }
                        }

                        float ndotk = max(dot(hitNrm, keyDir), 0.0f);
                        float3 bSunTint = float3(1.0f, 1.0f, 1.0f);
                        if (hitSunVis > 0.5f && hasStainedGlass && canSampleVoxels && keyDir.y > 0.02f) {
                            bSunTint = evalSmoothGlassGain(traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos, keyDir, 32.0f, 32, true));
                        }
                        float3 bounceDirect = hitAlbedo * directCol * bSunTint * ndotk * hitSunVis * 1.15f;
                        float3 bounceAmb = hitAlbedo * (hitAmbCol + localNeeBounce);
                        float distFade = 1.0f / (1.0f + bhit.distance * bhit.distance * 0.045f);
                        float3 emitTint = (hitEmissive > 0.85f) ? float3(1.25f, 0.68f, 0.32f) : float3(1.0f);
                        float emitGain = (lightCount > 0u) ? 4.2f : 14.0f;
                        float3 bounceEmit = min(hitAlbedo * emitTint * (hitEmissive * emitGain) * distFade, float3(3.5f, 3.5f, 3.5f));
                        float3 bounceTotal = bounceDirect + bounceAmb + bounceEmit;
                        if (uFrame.dimensionId == 0 && hitIsFoliage && !hitUnderSolidRoof) {
                            float leafSkyW = smoothstep(-0.08f, 0.22f, giDir.y);
                            float3 leafSkyTrans = lerp(skyAmb * 0.22f, float3(sampleSkyLut(skyViewLut, giDir)) * (0.42f / max(shadowDepth, 0.8f)), leafSkyW);
                            bounceTotal = lerp(bounceTotal, max(bounceTotal, leafSkyTrans), 0.35f);
                        }
                        giSum += bounceTotal;
                    }
                } else {
                    skyHits = 1.0f;
                    float3 skyGlassTint = (hasStainedGlass && canSampleVoxels)
                        ? traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + traceOrigin, giDir, 24.0f, 24)
                        : float3(1.0f, 1.0f, 1.0f);
                    float skyGain = (uFrame.dimensionId == 0) ? lerp(0.64f, 0.34f, env.nightFactor) : 0.48f;
                    float3 skySample = float3(sampleSkyLut(skyViewLut, giDir)) * (skyGain / max(shadowDepth, 0.8f));
                    if (uFrame.dimensionId == 0) {
                        float skyW = smoothstep(-0.04f, 0.14f, giDir.y);
                        giSum += lerp(skyAmb * 0.10f, skySample, skyW) * skyGlassTint;
                    } else {
                        giSum += skySample * skyGlassTint;
                    }
                }
                skyOpenFrac = skyHits;
                giAccum = giSum + neeAccum;
            }
        }

        // 4. Reflections (Gated by ENABLE_REFLECTIONS)
        if (enableRefl && isWater) {
            float3 waterSurfRel = waterSurfWorld - uFrame.camWorldPos.xyz;
            float surfDist = length(waterSurfRel);
            float ndv = saturate(dot(viewDir, waterSurfNrm));
            float4 waveEval = isWaterfallSurface
                ? evalMeteorWaterfallWaves((abs(waterSurfNrm.x) > 0.5f) ? waterSurfWorld.zy : waterSurfWorld.xy, uFrame.timeSeconds, surfDist, uFrame.surfaceParams.x)
                : ((length(waterFlowVec) > 0.08f)
                    ? evalMeteorFlowingWaves(waterSurfWorld.xz, waterFlowVec, uFrame.timeSeconds, surfDist, ndv, uFrame.surfaceParams.x)
                    : evalMeteorWaterWaves(waterSurfWorld.xz, uFrame.timeSeconds, surfDist, ndv, uFrame.surfaceParams.x));
            float3 waveNrm = isWaterfallSurface
                ? normalize(waterSurfNrm + (abs(waterSurfNrm.x) > 0.5f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f)) * waveEval.x + float3(0.0f, waveEval.y, 0.0f))
                : normalize(waterSurfNrm + float3(waveEval.x, 0.0f, waveEval.z));

            uint rippleCount = min(uint(waterRipples[0].x + 0.5f), 64u);
            if (!isWaterfallSurface && rippleCount > 0u && uFrame.surfaceParams.y > 0.01f) {
                float gradX = 0.0f;
                float gradZ = 0.0f;
                for (uint ri = 0u; ri < rippleCount; ++ri) {
                    float4 rp = waterRipples[1u + 2u * ri];
                    if (abs(waterSurfRel.y - rp.y) > 1.6f) continue;
                    float4 rd = waterRipples[2u + 2u * ri];
                    float4 rc = evalRippleContribution(waterSurfRel.xz, rp, rd);
                    gradX += rc.x;
                    gradZ += rc.y;
                }
                float rScale = clamp(uFrame.surfaceParams.y, 0.0f, 2.5f);
                waveNrm = normalize(waveNrm + float3(-gradX * rScale, 0.0f, -gradZ * rScale));
            }

            float3 reflDir = reflect(-viewDir, waveNrm);
            if (!isWaterfallSurface && reflDir.y < 0.012f) {
                reflDir = normalize(float3(reflDir.x, max(reflDir.y, 0.015f), reflDir.z));
            }

            bool skipFarOceanRt = (!isWaterfallSurface && surfDist > 64.0f && reflDir.y > 0.14f && canSampleVoxels && !isColumnSkyBlockedFast(voxelGrid, vOrg, waterSurfWorld));
            if (skipFarOceanRt) {
                reflOut = float4(float3(sampleSkyLut(skyReflLut, reflDir)), 1.0f);
            } else {
            ray rray;
            rray.origin = waterSurfRel + waterSurfNrm * 0.08f;
            rray.direction = reflDir;
            rray.min_distance = 0.06f;
            rray.max_distance = 112.0f;
            auto rhit = closestInter.intersect(rray, sceneTlas);

            if (rhit.type != intersection_type::none) {
                uint rqIdx = instanceMetaOffsets[rhit.instance_id] + (rhit.primitive_id >> 1u);
                uint rmeta = quadMeta[rqIdx];
                uint rfdir = rmeta & 7u;
                bool rHitFoliage = ((rmeta & 0x80u) != 0u);
                float rEmit = float((rmeta >> 3u) & 0xFu) / 15.0f;
                float3 rAlbedo = float3(float((rmeta >> 8u) & 0xFFu),
                                        float((rmeta >> 16u) & 0xFFu),
                                        float((rmeta >> 24u) & 0xFFu)) / 255.0f * 0.85f + 0.06f;
                float3 rNrm = decodeRtFaceNormal(rfdir);
                bool rBackFace = (dot(reflDir, rNrm) > 0.0f && !rHitFoliage);
                if (rBackFace) {
                    reflOut = float4(float3(0.006f, 0.008f, 0.012f), 1.0f);
                } else {
                if (dot(reflDir, rNrm) > 0.0f && rHitFoliage) {
                    rNrm = -rNrm;
                }
                float3 rPos = rray.origin + reflDir * rhit.distance + rNrm * 0.05f;
                float rSunVis = 0.0f;
                if (dot(rNrm, keyDir) > 0.05f) {
                    ray rSh;
                    rSh.origin = rPos;
                    rSh.direction = keyDir;
                    rSh.min_distance = 0.05f;
                    rSh.max_distance = 96.0f;
                    if (shadowInter.intersect(rSh, sceneTlas).type == intersection_type::none) {
                        rSunVis = 1.0f;
                    }
                }
                bool rSealed = false;
                if (uFrame.dimensionId == 0 && rSunVis < 0.5f) {
                    if (canSampleVoxels) {
                        rSealed = isColumnSkyBlockedFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + rPos);
                    } else {
                        ray rSkyProbe;
                        rSkyProbe.origin = rPos;
                        rSkyProbe.direction = float3(0.0f, 1.0f, 0.0f);
                        rSkyProbe.min_distance = 0.05f;
                        rSkyProbe.max_distance = 64.0f;
                        rSealed = (shadowInter.intersect(rSkyProbe, sceneTlas).type != intersection_type::none);
                    }
                }
                float3 rAmb = rSealed ? float3(0.012f, 0.016f, 0.024f) : (skyAmb * 0.68f);
                float3 rNeeWater = float3(0.0f);
                uint wLoopLim = (rSealed || env.nightFactor > 0.2f || uFrame.dimensionId == -1) ? min(lightCount, 3u) : 0u;
                for (uint wli = 0u; wli < wLoopLim; ++wli) {
                    float4 lpW = emissiveLights[1u + 2u * wli];
                    float3 toLW = lpW.xyz - rPos;
                    float d2W = dot(toLW, toLW);
                    float maxRW = evalPointLightMaxRange(lpW.w);
                    if (d2W >= maxRW * maxRW || d2W < 0.005f) continue;
                    float dW = sqrt(d2W);
                    float3 lDirW = toLW / dW;
                    float ndW = max(dot(rNrm, lDirW), 0.0f);
                    if (ndW <= 0.001f) continue;
                    float rFadeW = evalPointLightWindow(dW, maxRW);
                    float wAtten = (lpW.w / (d2W + 1.25f)) * ndW * rFadeW * 0.75f;
                    if (wAtten <= 0.0008f) continue;
                    float3 wVisCol = float3(1.0f, 1.0f, 1.0f);
                    if (canSampleVoxels && dW > 0.35f) {
                        int maxSteps = clamp(int(ceil(dW * 1.65f)) + 3, 4, 24);
                        wVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                            voxelGrid, vOrg, uFrame.camWorldPos.xyz + rPos, lDirW, dW, hasStainedGlass, maxSteps);
                    }
                    if (dot(wVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                    float4 lcW = emissiveLights[2u + 2u * wli];
                    rNeeWater += lcW.rgb * wVisCol * wAtten;
                }
                float3 wSunTint = float3(1.0f, 1.0f, 1.0f);
                if (rSunVis > 0.5f && hasStainedGlass && canSampleVoxels && keyDir.y > 0.02f) {
                    wSunTint = evalSmoothGlassGain(traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + rPos, keyDir, 32.0f, 32));
                }
                float3 litHit = rAlbedo * (directCol * wSunTint * max(dot(rNrm, keyDir), 0.0f) * rSunVis + rAmb + rNeeWater)
                              + rAlbedo * (rEmit * 6.5f);
                if (hasStainedGlass && canSampleVoxels) {
                    litHit *= traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + rray.origin, reflDir, min(rhit.distance, 24.0f), 24);
                }
                float fogBlend = 1.0f - exp(-rhit.distance * 0.0042f);
                float3 missBg = rSealed
                    ? float3(0.008f, 0.012f, 0.018f)
                    : float3(sampleSkyLut(skyViewLut, reflDir));
                float3 reflRadiance = lerp(litHit, missBg, clamp(fogBlend, 0.0f, 0.85f));
                reflOut = float4(reflRadiance, 1.0f);
                }
            } else {
                bool caveWater = false;
                if (uFrame.dimensionId == 0) {
                    if (reflDir.y < -0.02f) {
                        caveWater = true;
                    } else if (canSampleVoxels) {
                        caveWater = isColumnSkyBlockedFast(voxelGrid, vOrg, waterSurfWorld + reflDir * 8.0f);
                    } else {
                        ray wSkyProbe;
                        wSkyProbe.origin = rray.origin + reflDir * 10.0f;
                        wSkyProbe.direction = float3(0.0f, 1.0f, 0.0f);
                        wSkyProbe.min_distance = 0.05f;
                        wSkyProbe.max_distance = 64.0f;
                        caveWater = (shadowInter.intersect(wSkyProbe, sceneTlas).type != intersection_type::none);
                    }
                }
                if (caveWater) {
                    reflOut = float4(float3(0.008f, 0.012f, 0.018f), 1.0f);
                } else {
                    float3 wMissGlass = (hasStainedGlass && canSampleVoxels)
                        ? traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + rray.origin, reflDir, 24.0f, 24)
                        : float3(1.0f, 1.0f, 1.0f);
                    reflOut = float4(float3(sampleSkyLut(skyReflLut, reflDir)) * wMissGlass, 1.0f);
                }
            }
            }
        } else if (enableRefl && !isSky) {
            bool isOutdoorRainGround = (enablePbrRain && uFrame.dimensionId == 0 && env.rainFactor > 0.05f && nrm.y > 0.78f && !pbr.active && voxelTag != 8u);
            float2 rainPW = isOutdoorRainGround ? evalRainPuddleMask(worldPos.xz, env.rainFactor) : float2(0.0f, 0.0f);
            if (rainPW.y > 0.04f) {
                if (canSampleVoxels) {
                    if (isColumnOpaqueBlockedFast(voxelGrid, vOrg, worldPos)) rainPW = float2(0.0f, 0.0f);
                } else {
                    ray rRainSky;
                    rRainSky.origin = traceOrigin;
                    rRainSky.direction = float3(0.0f, 1.0f, 0.0f);
                    rRainSky.min_distance = 0.02f;
                    rRainSky.max_distance = 64.0f;
                    if (shadowInter.intersect(rRainSky, sceneTlas).type != intersection_type::none) {
                        rainPW = float2(0.0f, 0.0f);
                    }
                }
            }
            bool isRainWet = (rainPW.y > 0.04f);
            float rough = pbr.active ? pbr.roughness : (isRainWet ? lerp(0.18f, 0.03f, rainPW.x) : 0.08f);
            bool traceSurfRefl = (enablePbrRain && pbr.active && rough <= 0.42f) || isRainWet || (voxelTag == 8u) || (uFrame.dimensionId == 1 && nrm.y > 0.72f);
            if (traceSurfRefl) {
                float3 surfReflNrm = nrm;
                if (isRainWet) {
                    float2 rip = evalRaindropRippleNormal(worldPos.xz, uFrame.timeSeconds, rainPW.x);
                    surfReflNrm = normalize(nrm + float3(-rip.x, 0.0f, -rip.y));
                }
                float3 reflDir = reflect(-viewDir, surfReflNrm);
                if (rough > 0.06f) {
                    float3 rUp = abs(reflDir.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                    float3 rT = normalize(cross(rUp, reflDir));
                    float3 rB = cross(reflDir, rT);
                    float jAng = r0 * METEOR_TAU;
                    float jSpread = rough * rough * 0.65f * sqrt(max(r1, 0.01f));
                    reflDir = normalize(reflDir + (rT * cos(jAng) + rB * sin(jAng)) * jSpread);
                }
                ray rray;
                rray.origin = traceOrigin;
                rray.direction = reflDir;
                rray.min_distance = 0.004f;
                rray.max_distance = 140.0f;
                auto rhit = closestInter.intersect(rray, sceneTlas);
                if (rhit.type != intersection_type::none && rhit.distance < 0.12f) {
                    uint qIdxPre = instanceMetaOffsets[rhit.instance_id] + (rhit.primitive_id >> 1u);
                    uint fdirPre = quadMeta[qIdxPre] & 7u;
                    if (dot(decodeRtFaceNormal(fdirPre), reflDir) > 0.15f) {
                        rray.origin = traceOrigin + reflDir * (rhit.distance + 0.04f);
                        rhit = closestInter.intersect(rray, sceneTlas);
                    }
                }
                if (rhit.type != intersection_type::none) {
                    uint qIdx = instanceMetaOffsets[rhit.instance_id] + (rhit.primitive_id >> 1u);
                    uint meta = quadMeta[qIdx];
                    uint fdir = meta & 7u;
                    bool hitIsFoliageR = ((meta & 0x80u) != 0u);
                    float hitEmissive = float((meta >> 3u) & 0xFu) / 15.0f;
                    float3 hitAlbedo = float3(float((meta >> 8u) & 0xFFu),
                                              float((meta >> 16u) & 0xFFu),
                                              float((meta >> 24u) & 0xFFu)) / 255.0f * 0.85f + 0.06f;
                    float3 hitNrm = decodeRtFaceNormal(fdir);
                    if (dot(reflDir, hitNrm) > 0.0f && !hitIsFoliageR) {
                        reflOut = float4(float3(0.006f, 0.008f, 0.012f), 1.0f);
                    } else {
                    if (dot(reflDir, hitNrm) > 0.0f && hitIsFoliageR) {
                        hitNrm = -hitNrm;
                    }
                    float3 hitPos = rray.origin + reflDir * rhit.distance + hitNrm * 0.03f;
                    float hitSunVis = (uFrame.dimensionId == -1) ? 0.35f : 0.0f;
                    if (uFrame.dimensionId != -1 && dot(hitNrm, keyDir) > 0.05f) {
                        ray rShadow;
                        rShadow.origin = hitPos;
                        rShadow.direction = keyDir;
                        rShadow.min_distance = 0.015f;
                        rShadow.max_distance = 128.0f;
                        if (shadowInter.intersect(rShadow, sceneTlas).type == intersection_type::none) {
                            hitSunVis = 1.0f;
                        }
                    }

                    // Gate reflected wall ambient by actual sky visibility so sealed dark rooms stay dark in mirrors/gems
                    float3 rAmbCol = (uFrame.dimensionId == -1) ? (skyAmb * 0.28f) : float3(0.008f, 0.011f, 0.016f);
                    if (uFrame.dimensionId == 0) {
                        if (hitSunVis > 0.5f) {
                            rAmbCol = skyAmb * 0.50f;
                        } else if (canSampleVoxels) {
                            if (!isColumnSkyBlockedFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos)) {
                                rAmbCol = skyAmb * 0.38f;
                            }
                        } else {
                            ray rSkyProbe;
                            rSkyProbe.origin = hitPos;
                            rSkyProbe.direction = normalize(hitNrm * 0.35f + float3(0.0f, 0.94f, 0.0f));
                            rSkyProbe.min_distance = 0.015f;
                            rSkyProbe.max_distance = 48.0f;
                            if (shadowInter.intersect(rSkyProbe, sceneTlas).type == intersection_type::none) {
                                rAmbCol = skyAmb * 0.38f;
                            }
                        }
                    }

                    // Include nearby wall-occluded point lights on reflected walls
                    float3 rNeeCol = float3(0.0f);
                    uint rLoopLim = min(lightCount, 3u);
                    uint rRtCnt = 0u;
                    for (uint ri = 0u; ri < rLoopLim; ++ri) {
                        float4 lpR = emissiveLights[1u + 2u * ri];
                        float3 toLR = lpR.xyz - hitPos;
                        float d2R = dot(toLR, toLR);
                        float maxRR = evalPointLightMaxRange(lpR.w);
                        if (d2R >= maxRR * maxRR || d2R < 0.005f) continue;
                        float dRDist = sqrt(d2R);
                        float3 lDirR = toLR / dRDist;
                        float ndR = max(dot(hitNrm, lDirR), 0.0f);
                        if (ndR <= 0.001f) continue;
                        float rFadeR = evalPointLightWindow(dRDist, maxRR);
                        float rAtten = (lpR.w / (d2R + 1.25f)) * ndR * rFadeR;
                        if (rAtten <= 0.0008f) continue;

                        float4 lcR = emissiveLights[2u + 2u * ri];
                        float3 rVisCol = float3(1.0f, 1.0f, 1.0f);
                        if (canSampleVoxels && dRDist > 0.35f) {
                            int maxSteps = clamp(int(ceil(dRDist * 1.65f)) + 3, 4, 24);
                            rVisCol = tracePointLightVoxelVisibilityAndGlassDda(
                                voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos, lDirR, dRDist, hasStainedGlass, maxSteps);
                        } else if (!canSampleVoxels) {
                            if (rRtCnt >= 3u) continue;
                            ++rRtCnt;
                            float rTraceLen = dRDist - 0.12f;
                            if (rTraceLen > 0.06f) {
                                ray rLightRay;
                                rLightRay.origin = hitPos;
                                rLightRay.direction = lDirR;
                                rLightRay.min_distance = 0.02f;
                                rLightRay.max_distance = rTraceLen;
                                auto rLHit = closestInter.intersect(rLightRay, sceneTlas);
                                if (rLHit.type != intersection_type::none) {
                                    uint rqIdx = instanceMetaOffsets[rLHit.instance_id] + (rLHit.primitive_id >> 1u);
                                    uint rHitEmit = (quadMeta[rqIdx] >> 3u) & 0xFu;
                                    if (rHitEmit == 0u || (dRDist - rLHit.distance) > max(0.92f, lcR.w * 1.15f)) {
                                        rVisCol = float3(0.0f);
                                    }
                                }
                            }
                        }
                        if (dot(rVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                        rNeeCol += lcR.rgb * rVisCol * rAtten;
                    }

                    float3 rSunTint = float3(1.0f, 1.0f, 1.0f);
                    if (hitSunVis > 0.5f && hasStainedGlass && canSampleVoxels && keyDir.y > 0.02f) {
                        rSunTint = evalSmoothGlassGain(traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + hitPos, keyDir, 32.0f, 32));
                    }
                    float3 litHit = hitAlbedo * (directCol * rSunTint * max(dot(hitNrm, keyDir), 0.0f) * hitSunVis + rAmbCol + rNeeCol)
                                  + hitAlbedo * (hitEmissive * 9.5f);
                    if (hasStainedGlass && canSampleVoxels) {
                        litHit *= traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + traceOrigin, reflDir, min(rhit.distance, 24.0f), 24);
                    }
                    float fogBlend = 1.0f - exp(-max(rhit.distance - 38.0f, 0.0f) * 0.0055f);
                    float3 missBg = (uFrame.dimensionId == 0 && (reflDir.y < -0.02f || (hitSunVis < 0.5f && rAmbCol.b < 0.02f)))
                        ? float3(0.008f, 0.011f, 0.016f)
                        : float3(sampleSkyLut(skyViewLut, reflDir));
                    litHit = lerp(litHit, missBg, clamp(fogBlend, 0.0f, 0.85f));
                    reflOut = float4(litHit, 1.0f);
                    }
                } else {
                    bool sealedMiss = (uFrame.dimensionId == 0 && reflDir.y < -0.02f);
                    if (uFrame.dimensionId == 0 && !sealedMiss) {
                        if (canSampleVoxels) {
                            sealedMiss = isColumnSkyBlockedFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + traceOrigin + reflDir * 8.0f);
                        } else {
                            ray mSkyProbe;
                            mSkyProbe.origin = traceOrigin + reflDir * 10.0f;
                            mSkyProbe.direction = float3(0.0f, 1.0f, 0.0f);
                            mSkyProbe.min_distance = 0.05f;
                            mSkyProbe.max_distance = 64.0f;
                            sealedMiss = (shadowInter.intersect(mSkyProbe, sceneTlas).type != intersection_type::none);
                        }
                    }
                    if (sealedMiss) {
                        reflOut = float4(float3(0.008f, 0.011f, 0.016f), 1.0f);
                    } else {
                        float3 rMissGlass = (hasStainedGlass && canSampleVoxels)
                            ? traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + traceOrigin, reflDir, 24.0f, 24)
                            : float3(1.0f, 1.0f, 1.0f);
                        reflOut = float4(float3(sampleSkyLut(skyReflLut, reflDir)) * rMissGlass, 1.0f);
                    }
                }
            }
        }
    } else if (canSampleVoxels && !isSky) {
        // M1 / M2 512x128x512 Voxel DDA + World-Y Heightmap Fallback (Soft Sun Shadow, Entity Shadows, Voxel-Shadowed NEE Lights & Sky AO)
        bool ddaBacklitLeaf = (voxelTag == 2u && dot(nrm, keyDir) <= 0.0f && keyDir.y > 0.02f);
        if (!enableShadows && uFrame.dimensionId != -1) {
            shadowVis = smoothstep(0.0f, 0.25f, keyDir.y) * smoothstep(-0.10f, 0.35f, traceNrm.y) * 0.85f;
            sunVisRgb = float3(shadowVis);
            if (hasStainedGlass && shadowVis > 0.005f && keyDir.y > 0.02f) {
                sunVisRgb = evalSmoothGlassGain(traceStainedGlassFilterDda(voxelGrid, vOrg, worldPos + safeOriginOff, keyDir, 36.0f, 36, true)) * shadowVis;
            }
        } else if (uFrame.dimensionId != -1 && (dot(nrm, keyDir) > 0.0f || ddaBacklitLeaf || isPassThroughVeg)) {
            float3 sUp = abs(keyDir.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
            float3 sT = normalize(cross(sUp, keyDir));
            float3 sB = cross(keyDir, sT);
            float coneRad = lerp(0.032f, 0.060f, env.rainFactor);
            float phi = r0 * METEOR_TAU;
            float rad = coneRad * sqrt(clamp(0.18f + 0.82f * r1, 0.0f, 1.0f));
            float3 sDir = normalize(keyDir + (sT * cos(phi) + sB * sin(phi)) * rad);

            float3 startW = isPassThroughVeg
                ? (worldPos + safeOriginOff + keyDir * 0.25f + float3(0.0f, 0.08f, 0.0f))
                : (ddaBacklitLeaf ? (worldPos + keyDir * 1.05f) : (worldPos + safeOriginOff));
            // Entity sun shadows are evaluated crisply at full 1.0x resolution in rt_world_composite.slang
            float occluded = 0.0f;
            float tStep = 0.0f;
            for (int st = 1; st <= 32 && occluded < 0.5f; ++st) {
                tStep += 0.80f + float(st) * 0.035f;
                float3 p = startW + sDir * tStep;
                int cx = int(floor(p.x)) - vOrg.x;
                int cy = int(floor(p.y)) - vOrg.y;
                int cz = int(floor(p.z)) - vOrg.z;
                if (cx >= 0 && cx < 512 && cy >= 0 && cy < 128 && cz >= 0 && cz < 512) {
                    uint idx = uint((cy * 512 + cz) * 512 + cx);
                    uint v = (voxelGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
                    uint vTag = (v >> 4u) & 0xFu;
                    if (v != 0u && ((vTag != 3u && (vTag < 8u || vTag == 12u)) ||
                                    (vTag >= 14u && pointInSubBlockVoxel(vTag, v & 0xFu, p - floor(p))))) {
                        occluded = 1.0f;
                        break;
                    }
                }
            }
            float hmVis = (uFrame.dimensionId == 0)
                ? evalHeightmapFarShadow(voxelGrid, vOrg, startW, sDir)
                : 1.0f;
            shadowVis = min(1.0f - occluded, hmVis) * (ddaBacklitLeaf ? 0.55f : 1.0f);
            if (shadowVis > 0.005f && hasStainedGlass && keyDir.y > 0.02f) {
                sunVisRgb = evalSmoothGlassGain(traceStainedGlassFilterDda(voxelGrid, vOrg, startW, sDir, 36.0f, 36, true)) * shadowVis;
            } else {
                sunVisRgb = float3(shadowVis);
            }
        } else {
            shadowVis = 0.0f;
            sunVisRgb = float3(0.0f);
        }
        float hemiOcc = 0.0f;
        for (int vi = 1; vi <= 4; ++vi) {
            float3 p = worldPos + traceNrm * (0.4f + float(vi) * 0.75f);
            int cx = int(floor(p.x)) - vOrg.x;
            int cy = int(floor(p.y)) - vOrg.y;
            int cz = int(floor(p.z)) - vOrg.z;
            if (cx >= 0 && cx < 512 && cy >= 0 && cy < 128 && cz >= 0 && cz < 512) {
                uint idx = uint((cy * 512 + cz) * 512 + cx);
                uint v = (voxelGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
                uint vTag = (v >> 4u) & 0xFu;
                if (v != 0u && ((vTag != 3u && (vTag < 8u || vTag == 12u)) ||
                                (vTag >= 14u && pointInSubBlockVoxel(vTag, v & 0xFu, p - floor(p))))) {
                    hemiOcc += 0.22f;
                }
            }
        }
        if (uFrame.dimensionId == 0) {
            float hmOpen = evalSmoothSkyOpennessFast(voxelGrid, vOrg, worldPos + safeOriginOff, traceNrm, ign);
            hemiOcc = max(hemiOcc, (1.0f - hmOpen) * 0.42f);
        }
        uint ddaEntTestedLights = 0u;
        uint ddaTracedLights = 0u;
        uint ddaWrapTraced = 0u;
        uint ddaLitLights = 0u;
        uint rbLi0 = 0xFFFFFFFFu;
        uint rbLi1 = 0xFFFFFFFFu;
        float rbScore0 = 0.0f;
        float rbScore1 = 0.0f;
        for (uint ti = 0u; ti < tileCnt; ++ti) {
            uint lIdx = (tileOff >= 4u) ? tileLightGrid[tileOff + 1u + ti] : ti;
            if (lIdx >= lightCount) continue;
            float4 lp = emissiveLights[1u + 2u * lIdx];
            float3 lDiff = lp.xyz - relPos;
            float dist2 = dot(lDiff, lDiff);
            float maxRange = evalPointLightMaxRange(lp.w);
            if (dist2 >= maxRange * maxRange || dist2 < 0.005f) continue;
            float dist = sqrt(dist2);
            float rangeFade = evalPointLightWindow(dist, maxRange);
            float3 lDir = lDiff / dist;
            float ndL = dot(nrm, lDir);
            if (ndL <= 0.08f) {
                float rbW = (lp.w / (dist2 + 2.5f)) * rangeFade;
                if (rbW > rbScore0) {
                    rbScore1 = rbScore0;
                    rbLi1 = rbLi0;
                    rbScore0 = rbW;
                    rbLi0 = lIdx;
                } else if (rbW > rbScore1) {
                    rbScore1 = rbW;
                    rbLi1 = lIdx;
                }
            }
            if (ddaTracedLights >= maxTracedLights || ddaLitLights >= maxLitLights) continue;
            float geomTerm = evalPointLightGeomTerm(nrm, lDir, dist, voxelTag, voxelSub, isWater);
            float wrapBounce = (dist > 0.65f && ndL > -0.45f)
                ? (0.16f * smoothstep(-0.45f, 0.15f, ndL) * (1.0f - geomTerm))
                : 0.0f;
            float effGeom = geomTerm + wrapBounce;
            if (effGeom <= 0.0001f) continue;
            float baseW = (lp.w / (dist2 + 1.15f)) * rangeFade * effGeom;
            if (baseW < 0.0012f) continue;

            bool isDdaWrapOnly = (geomTerm <= 0.0001f);
            if (isDdaWrapOnly) {
                if (ddaLitLights > 0u || ddaWrapTraced >= 1u) continue;
                ++ddaWrapTraced;
            } else {
                ++ddaTracedLights;
            }
            float4 lc = emissiveLights[2u + 2u * lIdx];
            bool ddaGlassBackLit = (voxelTag == 8u && ndL < -0.05f);
            float3 ddaRayRelOrigin = ddaGlassBackLit ? (relPos - domAxis * 0.04f) : (relPos + safeOriginOff);
            float3 toLStartD = lp.xyz - ddaRayRelOrigin;
            float dStartD = length(toLStartD);
            float3 lightCol = lc.rgb;
            if (dStartD > 0.35f) {
                float3 lDirStartD = toLStartD / dStartD;
                float traceLen = max(dStartD - ((lc.w > 0.8f) ? 0.65f : 0.20f), 0.0f);
                if (dStartD < 14.0f && ddaEntTestedLights < 3u && traceLen > 0.05f) {
                    ++ddaEntTestedLights;
                    lightCol *= evalEntityShadowSoftVis(entityShadowCasters, relPos, ddaRayRelOrigin, lDirStartD, lp.xyz, traceLen, lc.w);
                    if (dot(lightCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) {
                        continue;
                    }
                }
                int maxSteps = clamp(int(ceil(dStartD * 1.65f)) + 3, 4, 44);
                lightCol *= tracePointLightVoxelVisibilityAndGlassDda(
                    voxelGrid, vOrg, uFrame.camWorldPos.xyz + ddaRayRelOrigin, lDirStartD, dStartD, hasStainedGlass, maxSteps);
            }
            if (dot(lightCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
            neeAccum += lightCol * baseW;
            if (!isDdaWrapOnly) {
                ++ddaLitLights;
            }
        }
        // Leak-proof indoor room-bounce probe for beam undersides / back-facing surfaces (M1/M2 DDA parity with Dusk)
        if (rbLi0 != 0xFFFFFFFFu && !isWater) {
            float3 pMidW = worldPos + nrm * 0.55f;
            float3 pBounceW = worldPos + nrm * 1.15f;
            if (!isPointBlockedInVoxel(voxelGrid, vOrg, pMidW) && !isPointBlockedInVoxel(voxelGrid, vOrg, pBounceW)) {
                for (int rbi = 0; rbi < 2; ++rbi) {
                    uint rLi = (rbi == 0) ? rbLi0 : rbLi1;
                    if (rLi == 0xFFFFFFFFu) break;
                    float4 rlp = emissiveLights[1u + 2u * rLi];
                    float3 rToL = (uFrame.camWorldPos.xyz + rlp.xyz) - pBounceW;
                    float rDist2 = dot(rToL, rToL);
                    float rMaxR = evalPointLightMaxRange(rlp.w);
                    if (rDist2 >= rMaxR * rMaxR || rDist2 < 0.05f) continue;
                    float rDist = sqrt(rDist2);
                    float rWin = evalPointLightWindow(rDist, rMaxR);
                    float rAtten = (rlp.w / (rDist2 + 2.5f)) * 0.16f * rWin;
                    if (rAtten <= 0.0008f) continue;
                    int rSteps = clamp(int(ceil(rDist * 1.50f)) + 2, 4, 24);
                    float3 rVis = tracePointLightVoxelVisibilityAndGlassDda(
                        voxelGrid, vOrg, pBounceW, rToL / rDist, rDist, hasStainedGlass, rSteps);
                    if (dot(rVis, float3(1.0f, 1.0f, 1.0f)) > 0.001f) {
                        float4 rlc = emissiveLights[2u + 2u * rLi];
                        neeAccum += rlc.rgb * rVis * rAtten;
                    }
                }
            }
        }
        neeAccum = neeAccum / (float3(1.0f) + max(neeAccum, float3(0.0f)) * 0.065f);
        giAccum = skyAmb * clamp(0.62f - (enableGiAo ? hemiOcc : 0.15f), 0.03f, 0.62f) + neeAccum;
    }

    float viewZ = isSky ? 0.0f : max(-mul(uFrame.viewRot, float4(relPos, 0.0f)).z, 0.01f);
    outGiShadow.write(float4(max(giAccum, float3(0.0f)), clamp(shadowVis, 0.0f, 1.0f)), tid.xy);
    outReflection.write(max(reflOut, float4(0.0f)), tid.xy);
    outSunVis.write(float4(max(sunVisRgb, float3(0.0f)), 1.0f), tid.xy);
    outGbuf.write(isSky ? half4(0.0h, 0.0h, 1.0h, 0.0h) : half4(half(viewZ), half3(nrm)), tid.xy);
}

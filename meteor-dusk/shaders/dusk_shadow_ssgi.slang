// Meteor Dusk Pass 1 — 2-Cascade Texel-Snapped PCSS + Screen-Space Contact Shadows + Multi-Horizon GTAO/SSGI + Clustered Top-8 Point Lights (0.5x, Zero RT)
module meteor_dusk_shadow_ssgi;
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
DepthTexture2D<float> shadowMapTex;

float2 vogelDiskOffset(int idx, int totalCount, float phi) {
    float r = sqrt((float(idx) + 0.5f) / max(float(totalCount), 1.0f));
    float theta = float(idx) * 2.39996323f + phi;
    return float2(cos(theta), sin(theta)) * r;
}

float3 evalCascadeSamplePos(
    int cascadeIdx,
    float3 relPos,
    float3 surfNormal,
    float3 keyDir,
    bool isPassThroughVeg,
    bool backlitLeaf,
    bool isSubBlockTop
) {
    float3 p;
    if (isPassThroughVeg) {
        float kOff = (cascadeIdx == 0) ? 0.18f : ((cascadeIdx == 1) ? 0.24f : 0.30f);
        float yOff = (cascadeIdx == 0) ? 0.12f : ((cascadeIdx == 1) ? 0.14f : 0.16f);
        p = relPos + keyDir * kOff + float3(0.0f, yOff, 0.0f);
    } else if (backlitLeaf) {
        float kOff = (cascadeIdx == 0) ? 0.90f : ((cascadeIdx == 1) ? 0.95f : 1.00f);
        p = relPos + keyDir * kOff;
    } else {
        float nOff = (cascadeIdx == 0) ? 0.028f : ((cascadeIdx == 1) ? 0.065f : 0.125f);
        float kOff = (cascadeIdx == 0) ? 0.012f : ((cascadeIdx == 1) ? 0.035f : 0.065f);
        p = relPos + surfNormal * nOff + keyDir * kOff;
    }
    if (isSubBlockTop && cascadeIdx < 2) {
        p.y += (cascadeIdx == 0) ? 0.045f : 0.055f;
    }
    return p;
}

float evalCascadeCoverageWeight(
    StructuredBuffer<float4> sMats,
    int cascadeIdx,
    float3 sampleRelPos,
    float innerNdc,
    float outerNdc
) {
    float4 row0 = (cascadeIdx == 0) ? sMats[0] : ((cascadeIdx == 1) ? sMats[4] : sMats[9]);
    float4 row1 = (cascadeIdx == 0) ? sMats[1] : ((cascadeIdx == 1) ? sMats[5] : sMats[10]);
    float4 row2 = (cascadeIdx == 0) ? sMats[2] : ((cascadeIdx == 1) ? sMats[6] : sMats[11]);

    float ndcX = dot(row0, float4(sampleRelPos, 1.0f));
    float ndcY = dot(row1, float4(sampleRelPos, 1.0f));
    float zRef = dot(row2, float4(sampleRelPos, 1.0f));

    float maxNdc = max(abs(ndcX), abs(ndcY));
    float xyW = 1.0f - smoothstep(innerNdc, outerNdc, maxNdc);
    float zW = smoothstep(0.006f, 0.040f, zRef) * (1.0f - smoothstep(0.960f, 0.994f, zRef));
    return xyW * zW;
}

float sampleCascadePcss(
    MetalityFrame f,
    DepthTexture2D<float> shadowAtlas,
    StructuredBuffer<float4> sMats,
    int cascadeIdx,
    float3 sampleRelPos,
    float ndotl,
    float ign,
    int tapCount,
    bool doBlockerSearch
) {
    float4 row0 = (cascadeIdx == 0) ? sMats[0] : ((cascadeIdx == 1) ? sMats[4] : sMats[9]);
    float4 row1 = (cascadeIdx == 0) ? sMats[1] : ((cascadeIdx == 1) ? sMats[5] : sMats[10]);
    float4 row2 = (cascadeIdx == 0) ? sMats[2] : ((cascadeIdx == 1) ? sMats[6] : sMats[11]);
    float invRes = sMats[8].z;
    float uScale = 1.0f / max(sMats[8].w, 1.0f);

    float ndcX = dot(row0, float4(sampleRelPos, 1.0f));
    float ndcY = dot(row1, float4(sampleRelPos, 1.0f));
    float zRef = dot(row2, float4(sampleRelPos, 1.0f));

    if (abs(ndcX) >= 0.992f || abs(ndcY) >= 0.992f || zRef <= 0.002f || zRef >= 0.998f) {
        return 1.0f;
    }

    float uLoc = ndcX * 0.5f + 0.5f;
    float vLoc = 0.5f - ndcY * 0.5f;
    float uBase = float(cascadeIdx) * uScale;
    float pad = max(3.0f * invRes, 0.0012f);
    float cascadeBiasScale = (cascadeIdx == 0) ? 1.0f : ((cascadeIdx == 1) ? 1.35f : 1.75f);
    float zEps = max(0.00024f * (1.0f - saturate(ndotl)) + 0.00010f, 0.00010f) * cascadeBiasScale;
    float phi = ign * METEOR_TAU;
    float rainSoft = 1.0f + saturate(f.rainStrength) * 0.55f;
    float halfM = (cascadeIdx == 0)
        ? max(sMats[8].x, 20.0f)
        : ((cascadeIdx == 1) ? max(sMats[8].y, 80.0f) : max(sMats[13].x, 160.0f));
    float invWorldSpan = 1.0f / (2.0f * halfM);

    float minRadius = max(((cascadeIdx == 0) ? 0.11f : ((cascadeIdx == 1) ? 0.22f : 0.34f)) * rainSoft * invWorldSpan, 1.8f * invRes);
    float maxRadius = min(((cascadeIdx == 0) ? 0.64f : ((cascadeIdx == 1) ? 1.00f : 1.30f)) * rainSoft * invWorldSpan, 14.0f * invRes);
    float filterRadius = clamp(((cascadeIdx == 0) ? 0.18f : ((cascadeIdx == 1) ? 0.32f : 0.52f)) * rainSoft * invWorldSpan, minRadius, maxRadius);
    int effTaps = tapCount;

    if (doBlockerSearch) {
        int2 atlasDim = max(metality_tex_size(shadowAtlas), int2(1, 1));
        int resPx = max(atlasDim.y, 1);
        int xOff = cascadeIdx * resPx;
        float searchRadius = max(((cascadeIdx == 0) ? 0.72f : 0.98f) * rainSoft * invWorldSpan, 2.8f * invRes);
        float blockerAcc = 0.0f;
        float blockerCnt = 0.0f;

        int2 cPx = int2(xOff + clamp(int(uLoc * float(resPx)), 0, resPx - 1), clamp(int(vLoc * float(resPx)), 0, resPx - 1));
        float cz = shadowAtlas.Load(int3(cPx, 0)).r;
        if (cz + zEps < zRef) {
            blockerAcc += (zRef - cz);
            blockerCnt += 1.0f;
        }

        for (int bi = 0; bi < 4; ++bi) {
            float2 vo = vogelDiskOffset(bi, 4, phi + 0.35f) * searchRadius;
            float cu = clamp(uLoc + vo.x, pad, 1.0f - pad);
            float cv = clamp(vLoc + vo.y, pad, 1.0f - pad);
            int2 bPx = int2(xOff + clamp(int(cu * float(resPx)), 0, resPx - 1), clamp(int(cv * float(resPx)), 0, resPx - 1));
            float bz = shadowAtlas.Load(int3(bPx, 0)).r;
            if (bz + zEps < zRef) {
                blockerAcc += (zRef - bz);
                blockerCnt += 1.0f;
            }
        }
        if (blockerCnt < 0.5f) {
            filterRadius = max(minRadius, filterRadius * 0.45f);
            effTaps = min(tapCount, 4);
        } else {
            float zSpanM = (cascadeIdx == 0)
                ? (2.0f * max(sMats[13].y, 160.0f))
                : ((cascadeIdx == 1) ? (2.0f * max(sMats[13].z, 240.0f)) : (2.0f * max(sMats[13].w, 320.0f)));
            float blockerDistM = (blockerAcc / blockerCnt) * zSpanM;
            float penumbraM = (cascadeIdx == 0)
                ? clamp(0.11f + blockerDistM * 0.044f, 0.11f, 0.64f)
                : clamp(0.22f + blockerDistM * 0.042f, 0.22f, 1.00f);
            filterRadius = clamp(penumbraM * rainSoft * invWorldSpan, minRadius, maxRadius);
        }
    }

    // Pure hardware 2x2 bilinear sample_compare PCF with IGN Vogel rotation
    float visSum = 0.0f;
    for (int i = 0; i < 6; ++i) {
        if (i >= effTaps) break;
        float2 vo = vogelDiskOffset(i, effTaps, phi) * filterRadius;
        float uTap = clamp(uLoc + vo.x, pad, 1.0f - pad);
        float vTap = clamp(vLoc + vo.y, pad, 1.0f - pad);
        float cmpRef = saturate(zRef - (zEps + length(vo) * 0.00025f));
        visSum += metality_shadow_cmp_sample(shadowAtlas, float2(uBase + uTap * uScale, vTap), cmpRef);
    }
    return visSum / max(float(effTaps), 1.0f);
}

float evalScreenSpaceContactShadow(
    MetalityFrame f,
    Texture2D<float4> depthTex,
    float3 originRel,
    float3 dir,
    float dist,
    float ign,
    int stepCap
) {
    if (dist > 28.0f) {
        return 1.0f;
    }
    float distFade = 1.0f - smoothstep(16.0f, 28.0f, dist);
    float maxLen = clamp(dist * 0.06f, 0.25f, 0.90f);
    float occ = 0.0f;
    for (int s = 0; s < 4; ++s) {
        if (s >= stepCap) break;
        float fracStep = (float(s) + 0.25f + ign * 0.65f) / float(stepCap);
        float tStep = maxLen * fracStep;
        float3 pRel = originRel + dir * tStep;
        float3 pView = mul(f.viewRot, float4(pRel, 0.0f)).xyz;
        float4 pClip = mul(f.proj, float4(pView, 1.0f));
        if (pClip.w <= 1e-4f) break;
        float2 pUv = pClip.xy / pClip.w * 0.5f + 0.5f;
        if (pUv.x < 0.002f || pUv.x > 0.998f || pUv.y < 0.002f || pUv.y > 0.998f) break;
        MeteorDepthSample gs = loadDepthAtUv(depthTex, pUv);
        if (!isSkyDepth(gs.depth)) {
            float3 gRel = reconstructRelWorldPos(f, gs.centerUv, gs.depth);
            float3 gView = mul(f.viewRot, float4(gRel, 0.0f)).xyz;
            float diff = (-pView.z) - (-gView.z);
            float thick = 0.16f + tStep * 0.16f;
            if (diff > 0.025f && diff < thick * 1.35f) {
                float hitW = (1.0f - smoothstep(0.025f, thick * 1.35f, diff)) * (1.0f - fracStep * 0.58f);
                occ = max(occ, hitW);
            }
        }
    }
    return 1.0f - clamp(occ * distFade, 0.0f, 0.55f);
}

// Compact under-roof sun-leak guard (0 outdoor cost; preserves glass-window & open-porch PCSS penumbras)
bool isUnderRoofSunLeakBlocked(StructuredBuffer<uint> vGrid, int3 vOrg, float3 startWorld, float3 sunDir) {
    float3 pos = startWorld - float3(vOrg);
    int3 cell = int3(floor(pos));
    int3 startCell = cell;
    float3 invDir = float3(
        abs(sunDir.x) > 1e-5f ? 1.0f / sunDir.x : 1e5f,
        abs(sunDir.y) > 1e-5f ? 1.0f / sunDir.y : 1e5f,
        abs(sunDir.z) > 1e-5f ? 1.0f / sunDir.z : 1e5f
    );
    int3 stepV = int3(sunDir.x >= 0.0f ? 1 : -1, sunDir.y >= 0.0f ? 1 : -1, sunDir.z >= 0.0f ? 1 : -1);
    float3 tMax = (float3(cell) + float3(stepV.x > 0 ? 1.0f : 0.0f, stepV.y > 0 ? 1.0f : 0.0f, stepV.z > 0 ? 1.0f : 0.0f) - pos) * invDir;
    float3 tDelta = abs(invDir);
    float dist = 0.0f;
    int hitAxis = 1;
    for (int iter = 0; iter < 16; ++iter) {
        if (!inVoxelBounds512Common(cell) || dist > 16.0f) {
            return false;
        }
        uint idx = voxelLinearIndex512Common(cell);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        if (b != 0u) {
            uint tag = (b >> 4u) & 0xFu;
            if (tag == 8u) {
                return false;
            }
            if (isOpaqueShadowVoxelCommon(tag) && any(cell != startCell)) {
                for (int ni = 0; ni < 4; ++ni) {
                    int sgn = ((ni & 1) == 0) ? 1 : -1;
                    int3 nOff = (ni < 2)
                        ? ((hitAxis == 0) ? int3(0, sgn, 0) : int3(sgn, 0, 0))
                        : ((hitAxis == 2) ? int3(0, sgn, 0) : int3(0, 0, sgn));
                    int3 nC = cell + nOff;
                    if (!inVoxelBounds512Common(nC)) return false;
                    uint nIdx = voxelLinearIndex512Common(nC);
                    uint nTag = ((vGrid[nIdx >> 2u] >> ((nIdx & 3u) * 8u)) >> 4u) & 0xFu;
                    if (nTag == 8u) return false;
                    if (!isOpaqueShadowVoxelCommon(nTag)) {
                        int topSolidY = sampleTopSolidWorldY(vGrid, nC.x, nC.z);
                        if (topSolidY > -9000 && topSolidY <= nC.y + vOrg.y + 1) return false;
                    }
                }
                return true;
            }
        }
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                dist = tMax.x; tMax.x += tDelta.x; cell.x += stepV.x; hitAxis = 0;
            } else {
                dist = tMax.z; tMax.z += tDelta.z; cell.z += stepV.z; hitAxis = 2;
            }
        } else {
            if (tMax.y < tMax.z) {
                dist = tMax.y; tMax.y += tDelta.y; cell.y += stepV.y; hitAxis = 1;
            } else {
                dist = tMax.z; tMax.z += tDelta.z; cell.z += stepV.z; hitAxis = 2;
            }
        }
    }
    return false;
}

bool isSoftShadowCellOpenForPeek(StructuredBuffer<uint> vGrid, int3 cell, int3 lightCell) {
    if (!inVoxelBounds512Common(cell) || all(cell == lightCell)) {
        return true;
    }
    uint idx = voxelLinearIndex512Common(cell);
    uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    if (b == 0u) return true;
    uint tag = (b >> 4u) & 0xFu;
    if (isOpaqueShadowVoxelCommon(tag) || tag == 15u) return false;
    if (tag == 14u) {
        uint sub = b & 0xFu;
        return (sub == 0u || sub >= 14u);
    }
    return true;
}

// Ultra-lean C1-continuous analytical soft-penumbra voxel DDA (0 stack arrays, 0 air-step overhead, 0 extra loop registers, leak-proof walls/doors)
float3 tracePointLightSoftVoxelDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 startWorld,
    float3 rayDir,
    float totalDist,
    float penumbraRad,
    bool hasStainedGlass,
    int maxSteps,
    bool doSoftEdges
) {
    float3 pos = startWorld - float3(vOrg);
    int3 cell = int3(floor(pos));
    int3 startCell = cell;
    int3 lightCell = int3(floor(pos + rayDir * totalDist));

    float3 invDir = float3(
        abs(rayDir.x) > 1e-5f ? 1.0f / rayDir.x : 1e5f,
        abs(rayDir.y) > 1e-5f ? 1.0f / rayDir.y : 1e5f,
        abs(rayDir.z) > 1e-5f ? 1.0f / rayDir.z : 1e5f
    );
    int3 stepV = int3(rayDir.x >= 0.0f ? 1 : -1, rayDir.y >= 0.0f ? 1 : -1, rayDir.z >= 0.0f ? 1 : -1);
    float3 tMax = (float3(cell) + float3(stepV.x > 0 ? 1.0f : 0.0f, stepV.y > 0 ? 1.0f : 0.0f, stepV.z > 0 ? 1.0f : 0.0f) - pos) * invDir;
    float3 tDelta = abs(invDir);

    float3 optDepth = float3(0.0f, 0.0f, 0.0f);
    bool hitGlass = false;
    float dist = 0.0f;
    float stopDist = max(totalDist + 0.08f, 0.10f);
    float occStopDist = max(totalDist - 0.10f, 0.0f);
    float softVis = 1.0f;
    bool alreadyPeeked = false;
    bool budgetExhausted = false;

    for (int iter = 0; iter < 64; ++iter) {
        if (iter >= maxSteps) {
            budgetExhausted = (dist < occStopDist);
            break;
        }
        if (!inVoxelBounds512Common(cell)) {
            break;
        }
        float nextDist = min(tMax.x, min(tMax.y, tMax.z));
        uint idx = voxelLinearIndex512Common(cell);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        if (b != 0u) {
            uint tag = (b >> 4u) & 0xFu;
            uint sub = b & 0xFu;
            if (hasStainedGlass && tag == 8u && sub >= 2u) {
                float cellDt = max(min(nextDist, stopDist) - dist, 0.0f);
                float maxAxisDir = max(abs(rayDir.x), max(abs(rayDir.y), abs(rayDir.z)));
                optDepth += -log(max(decodeStainedGlassColor(sub), float3(0.05f, 0.05f, 0.05f))) * (cellDt * maxAxisDir);
                hitGlass = true;
            } else if (tag >= 14u) {
                if (any(cell != lightCell) &&
                    intersectSubBlockVoxel(tag, sub, cell, pos, rayDir, occStopDist, all(cell == startCell))) {
                    return float3(0.0f, 0.0f, 0.0f);
                }
            } else if (isOpaqueShadowVoxelCommon(tag) && dist < occStopDist &&
                       any(cell != startCell) && any(cell != lightCell)) {
                int3 dLight = abs(cell - lightCell);
                if (!(tag == 4u && sub >= 10u && dLight.x <= 4 && dLight.z <= 4 && dLight.y <= 10)) {
                    if (!doSoftEdges || alreadyPeeked) {
                        return float3(0.0f, 0.0f, 0.0f);
                    }
                    float3 p0 = pos + rayDir * dist - float3(cell);
                    float3 p1 = pos + rayDir * nextDist - float3(cell);
                    float3 penP = float3(1.0f) - min(p0, p1);
                    float3 penN = max(p0, p1);
                    float3 pen = min(penP, penN);
                    float invTot = 1.0f / max(totalDist, 0.25f);
                    float wPenMax = clamp(penumbraRad * 1.55f * (min(nextDist, occStopDist) * invTot), 0.03f, 0.45f);
                    if (min(pen.x, min(pen.y, pen.z)) >= wPenMax) {
                        return float3(0.0f, 0.0f, 0.0f);
                    }
                    float3 s2 = max(float3(1.0f) - rayDir * rayDir, float3(0.0f));
                    float3 invSin = float3(
                        s2.x >= 0.0064f ? rsqrt(s2.x) : 1e5f,
                        s2.y >= 0.0064f ? rsqrt(s2.y) : 1e5f,
                        s2.z >= 0.0064f ? rsqrt(s2.z) : 1e5f
                    );
                    float3 dPerp = max(pen, float3(0.0f)) * invSin;
                    if (min(dPerp.x, min(dPerp.y, dPerp.z)) >= wPenMax) {
                        return float3(0.0f, 0.0f, 0.0f);
                    }

                    float bestVis = 0.0f;
                    float3 bestRawOut = float3(0.0f, 0.0f, 0.0f);
                    float bestT = dist;
                    for (int ax = 0; ax < 3; ++ax) {
                        float dp = (ax == 0) ? dPerp.x : ((ax == 1) ? dPerp.y : dPerp.z);
                        if (dp >= wPenMax) continue;
                        bool isPos = (ax == 0) ? (penP.x <= penN.x) : ((ax == 1) ? (penP.y <= penN.y) : (penP.z <= penN.z));
                        float rdA = (ax == 0) ? rayDir.x : ((ax == 1) ? rayDir.y : rayDir.z);
                        float tD = ((isPos ? rdA : -rdA) < 0.0f) ? nextDist : dist;
                        float wPen = clamp(penumbraRad * 1.55f * (min(tD, occStopDist) * invTot), 0.03f, 0.45f);
                        if (dp >= wPen) continue;
                        int dSide = isPos ? 1 : -1;
                        int3 nOff = (ax == 0) ? int3(dSide, 0, 0) : ((ax == 1) ? int3(0, dSide, 0) : int3(0, 0, dSide));
                        if (isSoftShadowCellOpenForPeek(vGrid, cell + nOff, lightCell)) {
                            float vA = 1.0f - smoothstep(0.0f, wPen, dp);
                            if (vA > bestVis) {
                                bestVis = vA;
                                bestRawOut = float3(nOff);
                                bestT = tD;
                            }
                        }
                    }

                    if (bestVis <= 0.008f) {
                        return float3(0.0f, 0.0f, 0.0f);
                    }
                    softVis = bestVis;
                    float3 perpV = bestRawOut - rayDir * dot(bestRawOut, rayDir);
                    float perpLen2 = dot(perpV, perpV);
                    float3 perpOut = (perpLen2 > 1e-6f) ? (perpV * rsqrt(perpLen2)) : bestRawOut;

                    float3 lightPosGrid = pos + rayDir * totalDist;
                    float effRim = penumbraRad;
                    if (!isSoftShadowCellOpenForPeek(vGrid, int3(floor(lightPosGrid + perpOut * effRim)), lightCell)) {
                        effRim *= 0.45f;
                        if (!isSoftShadowCellOpenForPeek(vGrid, int3(floor(lightPosGrid + perpOut * effRim)), lightCell)) {
                            return float3(0.0f, 0.0f, 0.0f);
                        }
                    }
                    float3 rimLightGrid = lightPosGrid + perpOut * effRim;
                    float3 toRim = rimLightGrid - pos;
                    float rimTotalDist = max(length(toRim), 0.10f);
                    rayDir = toRim / rimTotalDist;
                    float tJump = min(bestT, rimTotalDist);
                    pos = pos + rayDir * tJump + perpOut * 0.015f;
                    float remDist = max(rimTotalDist - tJump, 0.02f);

                    alreadyPeeked = true;
                    cell = int3(floor(pos));
                    startCell = int3(-999999, -999999, -999999);
                    lightCell = int3(floor(rimLightGrid));
                    invDir = float3(
                        abs(rayDir.x) > 1e-5f ? 1.0f / rayDir.x : 1e5f,
                        abs(rayDir.y) > 1e-5f ? 1.0f / rayDir.y : 1e5f,
                        abs(rayDir.z) > 1e-5f ? 1.0f / rayDir.z : 1e5f
                    );
                    stepV = int3(rayDir.x >= 0.0f ? 1 : -1, rayDir.y >= 0.0f ? 1 : -1, rayDir.z >= 0.0f ? 1 : -1);
                    tMax = (float3(cell) + float3(stepV.x > 0 ? 1.0f : 0.0f, stepV.y > 0 ? 1.0f : 0.0f, stepV.z > 0 ? 1.0f : 0.0f) - pos) * invDir;
                    tDelta = abs(invDir);
                    dist = 0.0f;
                    stopDist = max(remDist + 0.08f, 0.10f);
                    occStopDist = max(remDist - 0.10f, 0.0f);
                    continue;
                }
            }
        }

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                dist = tMax.x; tMax.x += tDelta.x; cell.x += stepV.x;
            } else {
                dist = tMax.z; tMax.z += tDelta.z; cell.z += stepV.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                dist = tMax.y; tMax.y += tDelta.y; cell.y += stepV.y;
            } else {
                dist = tMax.z; tMax.z += tDelta.z; cell.z += stepV.z;
            }
        }
        if (dist >= stopDist) {
            break;
        }
    }
    if (budgetExhausted || (dist < occStopDist && inVoxelBounds512Common(cell))) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return (hitGlass ? exp(-optDepth) : float3(1.0f, 1.0f, 1.0f)) * softVis;
}

struct DuskShadowSsgiOut {
    float4 col0 : SV_Target0;
    float4 col1 : SV_Target1;
    float4 col2 : SV_Target2;
};

[shader("fragment")]
DuskShadowSsgiOut fsDuskShadowSsgi(float4 pixelPos : SV_Position) {
    DuskShadowSsgiOut outVal;
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    if (isSkyDepth(ds.depth)) {
        outVal.col0 = float4(0.0f, 0.0f, 0.0f, -2.0f);
        outVal.col1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
        outVal.col2 = float4(0.0f, 0.0f, 1.0f, 0.0f);
        return outVal;
    }

    float3 relPos = reconstructRelWorldPos(uFrame, ds.centerUv, ds.depth);
    float dist = length(relPos);
    float3 viewDir = (dist > 1e-4f) ? (-relPos / dist) : float3(0.0f, 0.0f, 1.0f);
    int3 vOrg = uFrame.voxelOrigin.xyz;
    float3 surfNormal = reconstructFastNormal(uFrame, ds.centerUv, relPos, sceneDepthTex);
    bool canSampleVoxels = (uFrame.voxelOrigin.w & 1) != 0;

    uint activeTag = 0u;
    uint activeSub = 0u;
    uint emitLvl = 0u;
    if (canSampleVoxels) {
        MeteorSurfaceVoxel sv = sampleMeteorSurfaceVoxel(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos, normalize(relPos), surfNormal);
        activeTag = sv.activeTag;
        activeSub = sv.activeSub;
        emitLvl = sv.emitLvl;
    }
    bool isPassThroughVeg = (activeTag == 14u && activeSub == 15u);
    float3 originNrm = isPassThroughVeg
        ? normalize(surfNormal * 0.35f + float3(0.0f, 0.94f, 0.0f))
        : surfNormal;
    float3 domAxis = snapCardinalAxis(originNrm);
    float3 safeOriginOff = canSampleVoxels
        ? safeShadowOriginOffset(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos, originNrm, domAxis, viewDir, dist)
        : (originNrm * 0.015f);
    bool isEmissiveBlock = (activeTag == 4u || activeTag == 5u || activeTag == 6u || activeTag == 7u ||
                            activeTag == 9u || activeTag == 10u || activeTag == 11u || activeTag == 13u ||
                            (emitLvl >= 12u && activeTag != 3u && activeTag != 8u && activeTag != 12u && activeTag < 14u));

    float3 sunDir = normalize(uFrame.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 keyDir = sunDir;
    if (sunDir.y < -0.05f && uFrame.dimensionId == 0) {
        keyDir = normalize(uFrame.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    } else if (uFrame.dimensionId == 1) {
        keyDir = getMeteorBlackHoleDir();
    }

    bool enableShadows = meteorFeatureShadows(uFrame);
    bool enableGiAo = meteorFeatureGiAo(uFrame);
    uint tier = meteorShadowTier(uFrame);
    float2 rn = bayer4x4Pair(pixelPos.xy);

    // -------------------------------------------------------------------------
    // 1. Hardware sample_compare 3-Cascade PCSS (Single-Call Loop) + Far Heightmap Fallback + SSCS
    // -------------------------------------------------------------------------
    float sunVis = (uFrame.dimensionId == -1) ? 0.0f : 1.0f;
    float rasterCascadeConf = 0.0f;
    float ndotlKey = max(dot(surfNormal, keyDir), 0.0f);
    bool allowSunDirect = (ndotlKey > 0.002f) || isEmissiveBlock || (activeTag == 2u) || (activeTag == 8u) || (activeTag == 3u) || isPassThroughVeg;
    bool underSolidRoof = (uFrame.dimensionId == 0) && canSampleVoxels && isColumnSkyBlockedFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos + safeOriginOff);

    if (uFrame.dimensionId != -1) {
        float4 cParams = shadowMatrices[8];
        if (!enableShadows || cParams.w <= 0.5f) {
            sunVis = allowSunDirect ? (smoothstep(0.0f, 0.25f, keyDir.y) * smoothstep(-0.10f, 0.35f, originNrm.y) * 0.85f) : 0.0f;
            if (isEmissiveBlock) sunVis = max(sunVis, 0.92f);
        } else if (!allowSunDirect) {
            sunVis = 0.0f;
        } else {
            bool hasC1 = (cParams.w > 1.5f) && (cParams.y > 1.0f);
            bool hasC2 = (cParams.w > 2.5f) && (shadowMatrices[13].x > 1.0f);
            bool backlitLeaf = (activeTag == 2u && ndotlKey <= 0.02f && keyDir.y > 0.02f);
            bool isSubBlockTop = (activeTag >= 14u && activeSub != 15u && surfNormal.y > 0.7f);

            float w0 = evalCascadeCoverageWeight(shadowMatrices, 0, evalCascadeSamplePos(0, relPos, surfNormal, keyDir, isPassThroughVeg, backlitLeaf, isSubBlockTop), 0.78f, 0.95f);
            float w1 = hasC1 ? evalCascadeCoverageWeight(shadowMatrices, 1, evalCascadeSamplePos(1, relPos, surfNormal, keyDir, isPassThroughVeg, backlitLeaf, isSubBlockTop), 0.80f, 0.96f) : 0.0f;
            float w2 = hasC2 ? evalCascadeCoverageWeight(shadowMatrices, 2, evalCascadeSamplePos(2, relPos, surfNormal, keyDir, isPassThroughVeg, backlitLeaf, isSubBlockTop), 0.82f, 0.96f) : 0.0f;
            rasterCascadeConf = max(w0, max(w1, w2));

            float cascadeVis = 1.0f;
            if (w0 < 0.99f && w1 < 0.99f && w2 < 0.99f && uFrame.dimensionId == 0 && canSampleVoxels) {
                cascadeVis = evalHeightmapFarShadow(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos + safeOriginOff, keyDir);
            }

            int pcssTaps = (tier >= 2u) ? 6 : 4;
            for (int ci = 2; ci >= 0; --ci) {
                float wCi = (ci == 2) ? w2 : ((ci == 1) ? w1 : w0);
                if (wCi <= 0.005f) continue;
                if (ci == 2 && (w0 >= 0.99f || w1 >= 0.99f)) continue;
                if (ci == 1 && w0 >= 0.99f) continue;
                int cTaps = (ci == 0) ? pcssTaps : ((ci == 1) ? max(pcssTaps - 2, 4) : 4);
                bool cBlocker = (ci == 0) ? (tier >= 1u) : ((ci == 1) ? (tier >= 2u) : false);
                float3 posCi = evalCascadeSamplePos(ci, relPos, surfNormal, keyDir, isPassThroughVeg, backlitLeaf, isSubBlockTop);
                float vCi = sampleCascadePcss(uFrame, shadowMapTex, shadowMatrices, ci, posCi, ndotlKey, rn.x, cTaps, cBlocker);
                cascadeVis = lerp(cascadeVis, vCi, wCi);
            }

            float sscsVis = 1.0f;
            if (tier >= 1u && !isPassThroughVeg && ndotlKey > 0.08f && cascadeVis >= 0.05f && dist < 28.0f) {
                float rawSscs = evalScreenSpaceContactShadow(uFrame, sceneDepthTex, relPos + surfNormal * 0.045f, keyDir, dist, rn.y, 4);
                sscsVis = lerp(1.0f, rawSscs, smoothstep(0.08f, 0.35f, ndotlKey));
            }
            sunVis = cascadeVis * lerp(1.0f, sscsVis, 0.35f);
            if (underSolidRoof && sunVis > 0.01f && keyDir.y > 0.02f && !isEmissiveBlock &&
                isUnderRoofSunLeakBlocked(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos + safeOriginOff, keyDir)) {
                sunVis = 0.0f;
            }
            if (backlitLeaf) {
                sunVis *= 0.55f;
            }
            if (isEmissiveBlock) {
                sunVis = max(sunVis, 0.92f);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Symmetric 2x2-Bayer Multi-Horizon GTAO + Concave-Gated SSGI (Zero Blotches)
    // -------------------------------------------------------------------------
    float gtaoVis = 1.0f;
    float3 ssgiBounce = float3(0.0f, 0.0f, 0.0f);
    if (enableGiAo) {
        int aoDirs = (tier >= 2u) ? 4 : 2;
        float aoOcc = 0.0f;
        float baseRad = clamp(0.38f / max(dist, 1.5f), 0.008f, 0.055f);
        float2 screenRadius = float2(
            baseRad,
            baseRad * (float(uFrame.width) / max(float(uFrame.height), 1.0f))
        );
        uint2 qPx = uint2(max(floor(pixelPos.xy), float2(0.0f, 0.0f)));
        uint qM = (((qPx.x ^ qPx.y) & 1u) << 1u) | (qPx.y & 1u);
        float bayer2 = (float(qM) + 0.5f) * 0.25f;
        float sectorSpan = (aoDirs == 4) ? (METEOR_PI * 0.5f) : METEOR_PI;
        float rotPhi = bayer2 * sectorSpan;
        float rJit = (float(qM ^ 2u) * 0.25f - 0.375f) * 0.16f;
        for (int d = 0; d < 4; ++d) {
            if (d >= aoDirs) break;
            float ang = rotPhi + float(d) * sectorSpan;
            float2 dir2 = float2(cos(ang), sin(ang));
            float maxH = 0.0f;
            float3 dirBounce = float3(0.0f, 0.0f, 0.0f);
            for (int step = 1; step <= 2; ++step) {
                float sFrac = clamp((float(step) == 1.0f ? 0.38f : 0.86f) + rJit, 0.20f, 1.0f);
                float2 suv = uv + dir2 * (screenRadius * sFrac);
                if (suv.x < 0.002f || suv.x > 0.998f || suv.y < 0.002f || suv.y > 0.998f) continue;
                MeteorDepthSample ns = loadDepthAtUv(sceneDepthTex, suv);
                if (!isSkyDepth(ns.depth)) {
                    float3 sRel = reconstructRelWorldPos(uFrame, ns.centerUv, ns.depth);
                    float3 diffV = sRel - relPos;
                    float d2 = dot(diffV, diffV);
                    if (d2 > 0.002f && d2 < 18.0f) {
                        float dLen = sqrt(d2);
                        float3 wDir = diffV / dLen;
                        float cosUp = dot(surfNormal, wDir);
                        float viewDepthJump = abs(length(sRel) - dist);
                        float cutoutGuard = (viewDepthJump > max(0.85f, dist * 0.14f)) ? 0.15f : 1.0f;
                        float rangeFalloff = clamp(1.0f - dLen / 4.0f, 0.0f, 1.0f) * cutoutGuard;
                        float occ = saturate(cosUp - 0.14f) * rangeFalloff;
                        maxH = max(maxH, occ);

                        if (cosUp > 0.08f) {
                            float3 sRaw = sceneColorTex.Load(int3(ns.px, 0)).rgb;
                            float3 sCol = sRaw * sRaw;
                            float lLum = luminance(sCol);
                            float lavaChroma = saturate((sRaw.r - sRaw.b) * 1.4f) * smoothstep(0.60f, 0.92f, sRaw.r);
                            float stableLum = lerp(lLum, min(lLum, 0.35f), lavaChroma * 0.75f);
                            float3 stableCol = (lLum > 1e-4f) ? (sCol * (stableLum / lLum)) : sCol;
                            float bounceEnergy = clamp(stableLum * 1.15f, 0.0f, 0.85f) * saturate(cosUp - 0.08f) * rangeFalloff / (1.0f + d2 * 0.45f);
                            dirBounce += stableCol * bounceEnergy;
                        }
                    }
                }
            }
            aoOcc += maxH;
            ssgiBounce += dirBounce * 0.5f;
        }
        if (activeTag == 8u || isPassThroughVeg) {
            aoOcc *= 0.25f;
        } else if (activeTag == 2u) {
            aoOcc *= 0.35f;
        }
        gtaoVis = clamp(1.0f - (aoOcc / float(aoDirs)) * 0.44f * clamp(uFrame.lightingParams.y, 0.5f, 2.0f), 0.68f, 1.0f);
        ssgiBounce = ssgiBounce / float(aoDirs);
    }

    // -------------------------------------------------------------------------
    // 3. Smooth L2-Cached 512 KB World-Y Heightmap Sky Openness
    // -------------------------------------------------------------------------
    float skyOpen = (uFrame.dimensionId == 0 && canSampleVoxels)
        ? evalSmoothSkyOpennessFast(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos + safeOriginOff, originNrm, rn, sunDir.y > -0.15f)
        : ((uFrame.dimensionId == -1) ? 0.0f : 1.0f);
    if (uFrame.dimensionId == 0 && enableShadows && (ndotlKey > 0.04f || isPassThroughVeg) && originNrm.y > -0.15f && !underSolidRoof) {
        skyOpen = max(skyOpen, sunVis * lerp(skyOpen, 0.88f, rasterCascadeConf));
    }

    // -------------------------------------------------------------------------
    // 4. Tile-Culled Point Lights + C1-Continuous Leak-Proof Room-Bounce Probe
    // -------------------------------------------------------------------------
    bool hasStainedGlass = (uFrame.voxelOrigin.w & 4) != 0;
    uint lightCount = min(uint(emissiveLights[0].x + 0.5f), 1024u);
    uint tileOff = (lightCount > 0u) ? getTileLightOffset(tileLightGrid, uv) : 0u;
    uint scanCap = (tileOff >= 4u) ? min(tileLightGrid[tileOff], 63u) : min(lightCount, 63u);
    uint maxTraced = (tier >= 2u) ? 10u : 6u;
    uint tracedLights = 0u;
    uint litLights = 0u;
    float3 pointLightAccum = float3(0.0f, 0.0f, 0.0f);
    uint entCasterCount = enableShadows ? min(uint(entityShadowCasters[0].x + 0.5f), 256u) : 0u;
    uint entTestedLights = 0u;

    float nightFactorFast = 1.0f - smoothstep(-0.22f, 0.03f, sunDir.y);
    float roomBounceEnvMask = (uFrame.dimensionId != 0)
        ? 1.0f
        : max(1.0f - smoothstep(0.10f, 0.40f, skyOpen), smoothstep(0.10f, 0.35f, nightFactorFast));
    bool wantRoomBounce = canSampleVoxels && !isEmissiveBlock && (roomBounceEnvMask > 0.01f);
    uint bIdx0 = 0xFFFFFFFFu;
    uint bIdx1 = 0xFFFFFFFFu;
    float bScore0 = 0.0f;
    float bScore1 = 0.0f;
    float bNeed0 = 0.0f;
    float bNeed1 = 0.0f;

    for (uint ti = 0u; ti < scanCap; ++ti) {
        if (tracedLights >= maxTraced || litLights >= 6u) break;
        uint li = (tileOff >= 4u) ? tileLightGrid[tileOff + 1u + ti] : ti;
        if (li >= lightCount) continue;
        float4 lp = emissiveLights[1u + 2u * li];
        float3 toL = lp.xyz - relPos;
        float dist2 = dot(toL, toL);
        float maxR = evalPointLightMaxRange(lp.w);
        if (dist2 >= maxR * maxR || dist2 < 0.005f) continue;

        float dDist = sqrt(dist2);
        float win = evalPointLightWindow(dDist, maxR);
        float3 lDir = toL / dDist;
        float geomTerm = evalPointLightGeomTerm(surfNormal, lDir, dDist, activeTag, activeSub, activeTag == 3u);
        float ndL = dot(surfNormal, lDir);
        float bounceDistMask = smoothstep(0.35f, 0.90f, dDist);

        // Route back-facing lights to leak-proof roomBounce (zero glowing edge rims on dark cave overhangs)
        bool isBackFacing = (!isPassThroughVeg && activeTag != 8u && activeTag != 3u && ndL <= 0.005f);
        if (geomTerm <= 0.0001f || isBackFacing) {
            if (wantRoomBounce && bounceDistMask > 0.01f) {
                float bCand = (lp.w / (dist2 + 2.6f)) * win * bounceDistMask;
                if (bCand > bScore1 && bCand > 0.0005f) {
                    if (bCand > bScore0) {
                        bScore1 = bScore0; bIdx1 = bIdx0; bNeed1 = bNeed0;
                        bScore0 = bCand;   bIdx0 = li;    bNeed0 = bounceDistMask;
                    } else {
                        bScore1 = bCand;   bIdx1 = li;    bNeed1 = bounceDistMask;
                    }
                }
            }
            continue;
        }

        float4 lc = emissiveLights[2u + 2u * li];
        float lightRad = clamp(lc.w, 0.18f, 1.60f);
        float frontWrap = smoothstep(0.35f, 0.95f, dDist) * 0.28f * smoothstep(0.005f, 0.24f, ndL) * (1.0f - geomTerm);
        float atten = (lp.w * 1.18f / (dist2 + (1.95f + lightRad * 0.75f))) * (geomTerm + frontWrap) * win;
        float frontLitNeed = 0.45f * (1.0f - smoothstep(0.0f, 0.24f, ndL)) * bounceDistMask;
        float frontShadowNeed = lerp(0.36f, 1.0f, 1.0f - smoothstep(0.0f, 0.06f, ndL)) * bounceDistMask;

        if (atten < 0.0002f) {
            if (wantRoomBounce && frontLitNeed > 0.005f) {
                float bCand = (lp.w / (dist2 + 2.6f)) * win * frontLitNeed;
                if (bCand > bScore1 && bCand > 0.0005f) {
                    if (bCand > bScore0) {
                        bScore1 = bScore0; bIdx1 = bIdx0; bNeed1 = bNeed0;
                        bScore0 = bCand;   bIdx0 = li;    bNeed0 = frontLitNeed;
                    } else {
                        bScore1 = bCand;   bIdx1 = li;    bNeed1 = frontLitNeed;
                    }
                }
            }
            continue;
        }

        ++tracedLights;

        bool glassBackLit = (activeTag == 8u && ndL < -0.05f);
        float3 lShadowOriginRel = glassBackLit ? (relPos - domAxis * 0.06f) : (relPos + safeOriginOff);
        float3 toLStart = lp.xyz - lShadowOriginRel;
        float dStart = length(toLStart);
        float3 lVisCol = float3(1.0f, 1.0f, 1.0f);
        float voxelVisFrac = 1.0f;

        if (enableShadows && !isEmissiveBlock && dStart > 0.35f) {
            float3 lDirStart = toLStart / dStart;
            if (canSampleVoxels) {
                float l1Steps = abs(toLStart.x) + abs(toLStart.y) + abs(toLStart.z);
                int maxSteps = clamp(int(ceil(l1Steps)) + 5, 6, 62);
                float softFade = (1.0f - smoothstep(8.5f, 13.0f, dStart)) * smoothstep(0.006f, 0.018f, atten);
                float penumbraRad = clamp(0.28f + lightRad * 0.24f, 0.32f, 0.68f) * softFade;
                bool doSoft = (tracedLights <= 3u) && (penumbraRad > 0.03f) && !glassBackLit;
                lVisCol = tracePointLightSoftVoxelDda(
                    voxelGrid, vOrg, uFrame.camWorldPos.xyz + lShadowOriginRel,
                    lDirStart, dStart, penumbraRad, hasStainedGlass, maxSteps, doSoft);
                voxelVisFrac = saturate(max(lVisCol.r, max(lVisCol.g, lVisCol.b)));
            }

            if (wantRoomBounce && bounceDistMask > 0.01f) {
                float bNeed = lerp(frontShadowNeed, frontLitNeed, voxelVisFrac);
                if (bNeed > 0.005f) {
                    float bCand = (lp.w / (dist2 + 2.6f)) * win * bNeed;
                    if (bCand > bScore1 && bCand > 0.0005f) {
                        if (bCand > bScore0) {
                            bScore1 = bScore0; bIdx1 = bIdx0; bNeed1 = bNeed0;
                            bScore0 = bCand;   bIdx0 = li;    bNeed0 = bNeed;
                        } else {
                            bScore1 = bCand;   bIdx1 = li;    bNeed1 = bNeed;
                        }
                    }
                }
            }
            if (voxelVisFrac <= 0.001f) continue;

            if (entCasterCount > 0u && dStart < 14.0f && atten > 0.0025f && entTestedLights < 3u) {
                ++entTestedLights;
                float entTraceLen = max(dStart - ((lightRad > 0.8f) ? 0.55f : 0.18f), 0.0f);
                if (entTraceLen > 0.05f) {
                    float entVis = evalEntityShadowSoftVis(
                        entityShadowCasters, relPos, lShadowOriginRel, lDirStart, lp.xyz, entTraceLen, lightRad);
                    lVisCol *= entVis;
                }
            }
        } else if (wantRoomBounce && frontLitNeed > 0.005f) {
            float bCand = (lp.w / (dist2 + 2.6f)) * win * frontLitNeed;
            if (bCand > bScore1 && bCand > 0.0005f) {
                if (bCand > bScore0) {
                    bScore1 = bScore0; bIdx1 = bIdx0; bNeed1 = bNeed0;
                    bScore0 = bCand;   bIdx0 = li;    bNeed0 = frontLitNeed;
                } else {
                    bScore1 = bCand;   bIdx1 = li;    bNeed1 = frontLitNeed;
                }
            }
        }
        if (dot(lVisCol, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;

        pointLightAccum += lc.rgb * lVisCol * atten;
        ++litLights;
    }

    // Leak-proof voxel-verified room-bounce probe (max 2 probes, C1-continuous across grazing & soft-shadow boundaries)
    float3 roomBounce = float3(0.0f, 0.0f, 0.0f);
    float bounceDirectAtten = 1.0f / (1.0f + max(luminance(pointLightAccum), 0.0f) * 1.15f);
    if (wantRoomBounce && bScore0 > 0.0f && bounceDirectAtten > 0.005f) {
        float3 worldPosLocal = uFrame.camWorldPos.xyz + relPos;
        bool clear035 = !isPointBlockedInVoxel(voxelGrid, vOrg, worldPosLocal + domAxis * 0.35f);
        bool clear055 = clear035 && !isPointBlockedInVoxel(voxelGrid, vOrg, worldPosLocal + domAxis * 0.55f);
        if (clear055) {
            bool clear115 = !isPointBlockedInVoxel(voxelGrid, vOrg, worldPosLocal + domAxis * 0.85f)
                         && !isPointBlockedInVoxel(voxelGrid, vOrg, worldPosLocal + domAxis * 1.15f);
            float bouncePush = clear115 ? 1.15f : 0.55f;
            float3 bounceOriginRel = relPos + domAxis * bouncePush;
            float3 bounceOriginW = uFrame.camWorldPos.xyz + bounceOriginRel;
            for (int bi = 0; bi < 2; ++bi) {
                uint bli = (bi == 0) ? bIdx0 : bIdx1;
                float bNeed = (bi == 0) ? bNeed0 : bNeed1;
                if (bli >= lightCount || bNeed <= 0.005f) continue;
                float4 blp = emissiveLights[1u + 2u * bli];
                uint lCellTag = sampleVoxelTag(voxelGrid, int3(floor(uFrame.camWorldPos.xyz + blp.xyz)) - vOrg);
                if (lCellTag == 1u || lCellTag == 12u || lCellTag == 2u) continue;
                float3 bToL = blp.xyz - bounceOriginRel;
                float bDist2 = dot(bToL, bToL);
                float bMaxR = evalPointLightMaxRange(blp.w);
                if (bDist2 >= bMaxR * bMaxR || bDist2 < 0.08f) continue;
                float bDist = sqrt(bDist2);
                float3 bDir = bToL / bDist;
                float3 bSurfToL = blp.xyz - relPos;
                float bSurfNdL = dot(domAxis, bSurfToL * rsqrt(max(dot(bSurfToL, bSurfToL), 1e-4f)));
                float bWrap = smoothstep(-0.85f, 0.05f, bSurfNdL);
                if (bWrap <= 0.002f) continue;
                float bWin = evalPointLightWindow(bDist, bMaxR);
                float bAtten = (blp.w / (bDist2 + 2.6f)) * bWin * (0.18f * bWrap * bNeed * bounceDirectAtten * roomBounceEnvMask);
                if (bAtten < 0.0002f) continue;
                float3 bVis = float3(1.0f, 1.0f, 1.0f);
                if (enableShadows) {
                    float bL1Steps = abs(bToL.x) + abs(bToL.y) + abs(bToL.z);
                    int bSteps = clamp(int(ceil(bL1Steps)) + 3, 4, 48);
                    bVis = tracePointLightVoxelVisibilityAndGlassDda(
                        voxelGrid, vOrg, bounceOriginW, bDir, bDist, hasStainedGlass, bSteps);
                }
                if (dot(bVis, float3(1.0f, 1.0f, 1.0f)) <= 0.001f) continue;
                float4 blc = emissiveLights[2u + 2u * bli];
                roomBounce += blc.rgb * bVis * bAtten;
            }
        }
    }

    float3 pointLightTotal = pointLightAccum + roomBounce;
    pointLightTotal = pointLightTotal / (float3(1.0f) + max(pointLightTotal, float3(0.0f)) * 0.11f);

    // -------------------------------------------------------------------------
    // 4b. Indirect Sky/Sun/Indoor SSGI Combine + Stained-Glass Sun Visibility
    // -------------------------------------------------------------------------
    MeteorEnvFactors env = analyzeMeteorEnv(uFrame);
    float3 directCol = lerp(float3(1.74f, 1.58f, 1.36f), float3(2.42f, 1.04f, 0.37f), env.sunsetFactor);
    directCol = lerp(directCol, float3(0.115f, 0.195f, 0.360f), env.nightFactor);
    directCol = lerp(directCol, float3(0.41f, 0.46f, 0.53f), env.rainFactor * 0.82f);

    float hemiSky = clamp(originNrm.y * 0.26f + 0.74f, 0.44f, 1.0f);
    if (activeTag == 2u || isPassThroughVeg) {
        hemiSky = max(hemiSky, 0.62f);
    }
    float3 skyBaseCol = lerp(float3(0.35f, 0.47f, 0.68f), float3(0.47f, 0.31f, 0.26f), env.sunsetFactor * 0.65f);
    skyBaseCol = lerp(skyBaseCol, float3(0.021f, 0.038f, 0.094f), env.nightFactor);
    skyBaseCol = lerp(skyBaseCol, float3(0.135f, 0.175f, 0.23f), env.rainFactor * 0.82f);

    float3 indoorSsgiIllum = min(pointLightTotal, float3(0.85f, 0.85f, 0.85f)) * 0.45f;
    float3 indirectGi = float3(0.0f, 0.0f, 0.0f);
    if (uFrame.dimensionId == 0) {
        float skyCurve = pow(max(clamp(skyOpen, 0.0f, 1.0f), 1e-4f), 0.80f) * smoothstep(0.02f, 0.10f, skyOpen);
        float sunBounceStrength = skyCurve * (1.0f - env.nightFactor * 0.88f) * (1.0f - env.rainFactor * 0.65f);
        float shadowWarmAtten = lerp(0.80f, 1.0f, clamp(sunVis, 0.0f, 1.0f));
        float3 warmBounce = directCol * float3(0.16f, 0.145f, 0.115f) * (sunBounceStrength * 0.48f * shadowWarmAtten) * (0.48f + 0.52f * (1.0f - abs(surfNormal.y) * 0.35f));
        float3 localSsgiIllum = directCol * (0.55f * skyCurve) + skyBaseCol * (0.28f * skyCurve) + indoorSsgiIllum;
        indirectGi = skyBaseCol * (hemiSky * skyCurve * 0.90f) + warmBounce + ssgiBounce * localSsgiIllum + float3(0.004f, 0.006f, 0.009f);
    } else if (uFrame.dimensionId == -1) {
        float3 netherAmb = lerp(float3(0.22f, 0.09f, 0.04f), float3(0.30f, 0.05f, 0.04f), env.netherCrimsonFactor);
        netherAmb = lerp(netherAmb, float3(0.06f, 0.22f, 0.28f), env.netherSoulFactor);
        netherAmb = lerp(netherAmb, float3(0.05f, 0.24f, 0.22f), env.netherWarpedFactor);
        indirectGi = netherAmb * hemiSky + ssgiBounce * (float3(0.35f, 0.35f, 0.35f) + indoorSsgiIllum * 0.8f);
    } else {
        indirectGi = float3(0.08f, 0.045f, 0.14f) * hemiSky + ssgiBounce * (float3(0.25f, 0.25f, 0.25f) + indoorSsgiIllum * 0.8f);
    }
    indirectGi *= gtaoVis;

    // -------------------------------------------------------------------------
    // 4b. Half-Res Frame-Aware Stained-Glass Sun Visibility (MRT SV_Target1)
    // -------------------------------------------------------------------------
    float clampedSunVis = clamp(sunVis, 0.0f, 1.0f);
    float3 sunVisRgb = float3(clampedSunVis, clampedSunVis, clampedSunVis);
    if (uFrame.dimensionId == 0 && canSampleVoxels && hasStainedGlass && clampedSunVis > 0.005f && keyDir.y > 0.02f) {
        float3 tint = traceStainedGlassFilterDda(voxelGrid, vOrg, uFrame.camWorldPos.xyz + relPos + safeOriginOff, keyDir, 36.0f, 32, true);
        float minTint = min(tint.r, min(tint.g, tint.b));
        float tintMask = 1.0f - smoothstep(0.88f, 0.99f, minTint);
        float3 boostedTint = pow(max(tint, float3(0.02f)), float3(1.35f)) * 1.35f;
        float3 glassFilterGain = lerp(float3(1.0f), boostedTint, tintMask);
        sunVisRgb = glassFilterGain * clampedSunVis;
        indirectGi *= lerp(float3(1.0f), glassFilterGain, 0.35f * tintMask * smoothstep(0.01f, 0.25f, clampedSunVis));
    }

    float viewZ = max(-mul(uFrame.viewRot, float4(relPos, 0.0f)).z, 0.01f);
    float3 totalIndirect = indirectGi + pointLightTotal * 0.82f;
    outVal.col0 = float4(max(totalIndirect, float3(0.0f, 0.0f, 0.0f)), clampedSunVis);
    outVal.col1 = float4(max(sunVisRgb, float3(0.0f, 0.0f, 0.0f)), 1.0f);
    outVal.col2 = float4(viewZ, surfNormal.x, surfNormal.y, surfNormal.z);
    return outVal;
}

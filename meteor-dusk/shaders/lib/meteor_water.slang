// Meteor Shader Water — Dual-Surface Volumetric Water, Beer-Lambert Caustics & 3D BVH Reflections

bool isFarOceanTint(float3 rawSrgb) {
    return rawSrgb.b > rawSrgb.r * 1.06f && rawSrgb.g > rawSrgb.r * 0.92f && rawSrgb.b > 0.10f;
}

float evaluateWaterMask(float3 rawSrgb, float3 worldPos, float3 nrm, MetalityFrame f) {
    if (f.dimensionId != 0 || f.submergedMode == 1) {
        return 0.0f;
    }
    float flatTop = smoothstep(0.78f, 0.94f, nrm.y);
    float sideFlow = smoothstep(0.18f, 0.55f, rawSrgb.b - rawSrgb.r) * smoothstep(0.04f, 0.22f, rawSrgb.b - rawSrgb.g * 0.90f);
    float nearSea = 1.0f - smoothstep(0.45f, 1.35f, abs(worldPos.y - (f.seaLevel - 0.12f)));
    float blueRatio = smoothstep(1.04f, 1.28f, rawSrgb.b / max(rawSrgb.r, 0.04f));
    float greenOK = smoothstep(0.84f, 1.02f, rawSrgb.b / max(rawSrgb.g, 0.05f));
    float topMask = flatTop * blueRatio * greenOK * max(nearSea, smoothstep(0.08f, 0.22f, rawSrgb.b - rawSrgb.r));
    return clamp(max(topMask, sideFlow * 0.75f), 0.0f, 1.0f);
}

half2 simplexGrad2D_fast_h(int2 ic) {
    uint h = (uint(ic.x) * 1664525u) ^ (uint(ic.y) * 1013904223u);
    h = ((h >> 16u) ^ h) * 0x45d9f3bu;
    h = (h >> 16u) & 7u;
    half sx = (h == 0u || h == 1u || h == 7u) ? half(1.0h) : ((h == 3u || h == 4u || h == 5u) ? half(-1.0h) : half(0.0h));
    half sy = (h == 1u || h == 2u || h == 3u) ? half(1.0h) : ((h == 5u || h == 6u || h == 7u) ? half(-1.0h) : half(0.0h));
    half sc = ((h & 1u) != 0u) ? half(0.7071h) : half(1.0h);
    return half2(sx, sy) * sc;
}

half3 snoise_fast_h(float2 p) {
    float K1 = 0.366025404f;
    float K2 = 0.211324865f;
    float2 i = floor(p + (p.x + p.y) * K1);
    int2 ic = int2(i);
    half2 a = half2(p - i + (i.x + i.y) * K2);
    int ox = (a.x >= a.y) ? 1 : 0;
    int2 io = int2(ox, 1 - ox);
    half2 o = half2(half(ox), half(1 - ox));
    half2 b = a - o + half(0.2113249h);
    half2 c = a - half(0.5773503h);

    half3 h = max(half(0.5h) - half3(dot(a, a), dot(b, b), dot(c, c)), half3(0.0h, 0.0h, 0.0h));
    half3 h2 = h * h;
    half3 h3 = h2 * h;
    half3 h4 = h2 * h2;

    half2 ga = simplexGrad2D_fast_h(ic);
    half2 gb = simplexGrad2D_fast_h(ic + io);
    half2 gc = simplexGrad2D_fast_h(ic + int2(1, 1));

    half va = dot(ga, a);
    half vb = dot(gb, b);
    half vc = dot(gc, c);

    half val = half(70.0h) * (h4.x * va + h4.y * vb + h4.z * vc);
    half2 grad = half(70.0h) * (
        h4.x * ga - half(8.0h) * h3.x * va * a +
        h4.y * gb - half(8.0h) * h3.y * vb * b +
        h4.z * gc - half(8.0h) * h3.z * vc * c
    );
    return half3(val, grad.x, grad.y);
}

float3 snoise2DGrad(float2 p) {
    return float3(snoise_fast_h(p));
}

float4 evalMeteorWaterWaves(float2 xz, float timeSec, float dist, float ndotv, float waveHeightScale) {
    // Unrolled zero-array branchless FP16 4-octave Simplex FBM with distance octave LOD (<18/42/80m)
    // and dual-band attenuation: low-freq swells keep >=0.14 floor at ocean/horizon distance (fixes flat mirror ocean & far strip)
    // while high-freq ripples aggressively attenuate with distance & grazing angle (zero aliasing/shimmer).
    half2 gradLow = half2(0.0h, 0.0h);
    half hLow = half(0.0h);
    half2 gradHigh = half2(0.0h, 0.0h);
    half hHigh = half(0.0h);

    // Octave 0: 0 deg, sc=(0.30, 0.42), amp=0.38, always active
    {
        float2 q0 = float2(xz.x * 0.30f, xz.y * 0.42f) + timeSec * float2(0.16f, 0.26f) + float2(3.1f, 7.4f);
        half3 n0 = snoise_fast_h(q0);
        gradLow += half2(n0.y * half(0.30h), n0.z * half(0.42h)) * half(0.38h);
        hLow += n0.x * half(0.38h);
    }
    // Octave 1: 58 deg, sc=(0.58, 0.80), amp=0.28, active < 80m
    if (dist < 80.0f) {
        float2 q1 = float2((0.5299f * xz.x - 0.8480f * xz.y) * 0.58f,
                           (0.8480f * xz.x + 0.5299f * xz.y) * 0.80f)
                  + timeSec * float2(-0.18f, 0.34f) + float2(29.7f, 11.3f);
        half3 n1 = snoise_fast_h(q1);
        half wa1 = half(0.28f * exp(-dist * 0.010f) * saturate((80.0f - dist) * 0.06f));
        gradLow += half2(n1.y * half(0.3073h) + n1.z * half(0.6784h),
                         n1.y * half(-0.4918h) + n1.z * half(0.4239h)) * wa1;
        hLow += n1.x * wa1;
    }
    // Octave 2: 117 deg, sc=(1.12, 1.54), amp=0.20, active < 42m
    if (dist < 42.0f) {
        float2 q2 = float2((-0.4540f * xz.x - 0.8910f * xz.y) * 1.12f,
                           ( 0.8910f * xz.x - 0.4540f * xz.y) * 1.54f)
                  + timeSec * float2(0.24f, -0.32f) + float2(53.2f, 41.9f);
        half3 n2 = snoise_fast_h(q2);
        half wa2 = half(0.20f * exp(-dist * 0.026f) * saturate((42.0f - dist) * 0.12f));
        gradHigh += half2(n2.y * half(-0.5085h) + n2.z * half(1.3721h),
                          n2.y * half(-0.9979h) + n2.z * half(-0.6992h)) * wa2;
        hHigh += n2.x * wa2;
    }
    // Octave 3: 163 deg, sc=(2.15, 2.95), amp=0.14, active < 18m
    if (dist < 18.0f) {
        float2 q3 = float2((-0.9563f * xz.x - 0.2924f * xz.y) * 2.15f,
                           ( 0.2924f * xz.x - 0.9563f * xz.y) * 2.95f)
                  + timeSec * float2(-0.30f, 0.44f) + float2(89.5f, 17.8f);
        half3 n3 = snoise_fast_h(q3);
        half wa3 = half(0.14f * exp(-dist * 0.052f) * saturate((18.0f - dist) * 0.25f));
        gradHigh += half2(n3.y * half(-2.0560h) + n3.z * half(0.8626h),
                          n3.y * half(-0.6287h) + n3.z * half(-2.8211h)) * wa3;
        hHigh += n3.x * wa3;
    }

    float baseStrength = 0.088f * clamp(waveHeightScale, 0.25f, 2.5f);
    float ndv = saturate(ndotv);
    float attenLow = clamp(exp(-dist * 0.011f) * pow(ndv, 0.55f), 0.14f, 1.0f);
    float attenHigh = clamp(exp(-dist * 0.024f) * pow(ndv, 0.85f), 0.0f, 1.0f);
    float sLow = baseStrength * attenLow;
    float sHigh = baseStrength * attenHigh;

    float2 finalGrad = float2(gradLow) * sLow + float2(gradHigh) * sHigh;
    float finalH = float(hLow) * sLow + float(hHigh) * sHigh;
    float3 waveNrm = normalize(float3(-finalGrad.x, 1.0f, -finalGrad.y));
    return float4(waveNrm, finalH);
}

float evalFastWaterCaustic(float2 p, float t) {
    float2 q1 = float2(p.x * 0.92f - p.y * 0.39f, p.x * 0.39f + p.y * 0.92f) * 1.35f + float2(t * 0.24f, -t * 0.18f);
    float2 q2 = float2(p.x * 0.54f + p.y * 0.84f, -p.x * 0.84f + p.y * 0.54f) * 2.05f + float2(-t * 0.21f, t * 0.27f);
    half n1 = snoise_fast_h(q1).x;
    half n2 = snoise_fast_h(q2 + float(n1) * 0.28f).x;
    half r1 = half(1.0h) - abs(n1);
    half r2 = half(1.0h) - abs(n2);
    return pow(saturate(float(r1 * r2) * 1.15f), 3.2f);
}

float4 evalMeteorWaterfallWaves(float2 faceUV, float timeSec, float dist, float waveHeightScale) {
    half2 gradUV = half2(0.0h, 0.0h);
    half h = half(0.0h);
    {
        float2 q0 = float2(faceUV.x * 0.85f, (faceUV.y + timeSec * 1.85f) * 0.28f) + float2(11.4f, 5.2f);
        half3 n0 = snoise_fast_h(q0);
        gradUV += half2(n0.y * half(0.85h), n0.z * half(0.28h)) * half(0.58h);
        h += n0.x * half(0.58h);
    }
    {
        float2 q1 = float2(faceUV.x * 1.75f, (faceUV.y + timeSec * 2.65f) * 0.58f) + float2(37.9f, 19.7f);
        half3 n1 = snoise_fast_h(q1);
        gradUV += half2(n1.y * half(1.75h), n1.z * half(0.58h)) * half(0.28h);
        h += n1.x * half(0.28h);
    }
    if (dist < 42.0f) {
        float2 q2 = float2(faceUV.x * 3.40f, (faceUV.y + timeSec * 3.55f) * 1.15f) + float2(61.3f, 43.1f);
        half3 n2 = snoise_fast_h(q2);
        gradUV += half2(n2.y * half(3.40h), n2.z * half(1.15h)) * half(0.12h);
        h += n2.x * half(0.12h);
    }
    float strength = 0.11f * clamp(waveHeightScale, 0.25f, 2.5f) * exp(-dist * 0.012f);
    float hf = float(h);
    float foamStreak = saturate(smoothstep(0.24f, 0.78f, hf * 0.80f + 0.24f));
    return float4(clamp(float2(gradUV) * strength, float2(-0.42f), float2(0.42f)), foamStreak, hf * strength);
}

float4 evalMeteorFlowingWaves(float2 xz, float2 flowVec, float timeSec, float dist, float ndotv, float waveHeightScale) {
    float flowLen = length(flowVec);
    float2 flowDir = (flowLen > 1e-4f) ? (flowVec / flowLen) : float2(0.0f, 1.0f);
    float2 flowAdvect = -flowDir * (timeSec * clamp(1.2f + flowLen * 1.4f, 0.8f, 2.6f));
    return evalMeteorWaterWaves(xz + flowAdvect, timeSec, dist, ndotv, waveHeightScale * 1.15f);
}

float4 evalRippleContribution(float2 surfRelXZ, float4 rPosAge, float4 rDirAmpKind) {
    float age = rPosAge.w;
    if (age < 0.0f || age > 2.8f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float fadeIn = smoothstep(0.0f, 0.08f, age);
    float amp = rDirAmpKind.z;
    if (rDirAmpKind.w < 0.5f) {
        // Huygens circular wavelet / splash / swimming / wading / oar ring packet
        float2 d = surfRelXZ - rPosAge.xz;
        float r = length(d);
        if (r > 6.5f) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float frontR = 0.22f + age * 2.15f;
        float w = 0.38f + age * 0.26f;
        float dr = r - frontR;
        if (abs(dr) > w * 2.4f) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float2 dirR = (r > 1e-4f) ? (d / r) : float2(1.0f, 0.0f);
        float env = fadeIn * exp(-(dr * dr) / (w * w)) * exp(-age * 1.15f) / (1.0f + r * 0.85f);
        float phase = dr * 7.2f;
        float s = sin(phase);
        float c = cos(phase);
        float h = env * s * amp * 0.13f;
        float dhdr = env * (c * 7.2f - s * (2.0f * dr / (w * w))) * amp * 0.13f;
        float foam = saturate(env * max(s, 0.0f) * max(amp - 0.35f, 0.0f) * exp(-age * 2.40f) * 0.14f);
        return float4(dirR.x * dhdr, dirR.y * dhdr, h, foam);
    } else {
        // Compact curved boat hull bow/stern wake (never emitted for slight player steps)
        float2 fwd = normalize(rDirAmpKind.xy + float2(1e-5f, 0.0f));
        float2 right = float2(-fwd.y, fwd.x);
        float2 rel = surfRelXZ - (rPosAge.xz + fwd * (age * 0.45f));
        float along = dot(rel, fwd);
        float across = dot(rel, right);
        float dBack = max(0.0f, -along + 0.30f);
        float armCenter = 0.32f + dBack * 0.36f + age * 0.85f;
        float armW = 0.36f + age * 0.22f;
        float absAcross = abs(across);
        float dArm = absAcross - armCenter;
        if (along > 1.0f || dBack > 4.2f || abs(dArm) > armW * 2.4f) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float backMask = exp(-max(along, 0.0f) * 3.5f) * exp(-dBack * 0.58f);
        float env = fadeIn * backMask * exp(-(dArm * dArm) / (armW * armW)) * exp(-age * 1.25f);
        float phase = dArm * 6.8f;
        float s = sin(phase);
        float c = cos(phase);
        float sgnAcross = (across >= 0.0f) ? 1.0f : -1.0f;
        float h = env * s * amp * 0.14f;
        float dh_dAcross = env * (c * 6.8f - s * (2.0f * dArm / (armW * armW))) * sgnAcross * amp * 0.14f;
        float dh_dAlong = dh_dAcross * sgnAcross * 0.34f;
        float2 gradXZ = right * dh_dAcross + fwd * dh_dAlong;
        float foam = saturate(env * max(s, 0.0f) * amp * exp(-age * 2.00f) * 0.22f);
        return float4(gradXZ.x, gradXZ.y, h, foam);
    }
}

float3 getMeteorWaterNormalAtten(float2 xz, float timeSec, float dist, float ndotv) {
    return evalMeteorWaterWaves(xz, timeSec, dist, ndotv, 1.15f).xyz;
}

float3 getMeteorWaterNormal(float2 xz, float timeSec) {
    return getMeteorWaterNormalAtten(xz, timeSec, 8.0f, 0.8f);
}

float evalGgxSunHighlight(float3 viewDir, float3 reflDir, float3 waveNrm, float3 sunDir, float roughness) {
    float3 hSum = sunDir + viewDir;
    float hLen2 = dot(hSum, hSum);
    float3 H = (hLen2 > 1e-6f) ? (hSum * rsqrt(hLen2)) : waveNrm;
    float NdotH = saturate(dot(waveNrm, H));
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    float ggx = a2 / max(METEOR_PI * denom * denom, 1e-4f);
    float beam = pow(saturate(dot(reflDir, sunDir)), 480.0f) * 1.4f;
    return clamp(ggx * 0.055f + beam * 0.12f, 0.0f, 2.2f);
}

float3 shadeMeteorWaterFallback(
    MetalityFrame f,
    MeteorEnvFactors env,
    float3 rawSrgb,
    float3 rawLinear,
    float3 relPos,
    float3 worldPos,
    float3 geomNrm,
    float3 externalReflCol,
    float externalReflWeight
) {
    float3 viewDir = normalize(-relPos);
    float ndv = saturate(dot(viewDir, geomNrm));
    float3 waveNrm = lerp(geomNrm, getMeteorWaterNormalAtten(worldPos.xz, f.timeSeconds, length(relPos), ndv), clamp(geomNrm.y, 0.35f, 1.0f));
    float3 reflDir = reflect(-viewDir, waveNrm);
    if (reflDir.y < 0.015f) {
        reflDir = normalize(float3(reflDir.x, max(reflDir.y, 0.02f), reflDir.z));
    }

    float rawLuma = luminance(rawSrgb);
    float shallowEstimate = smoothstep(0.20f, 0.65f, rawLuma) * smoothstep(0.05f, 0.25f, rawSrgb.g - rawSrgb.b * 0.35f + 0.18f);
    float clarity = clamp(f.userParams.z, 0.4f, 1.8f);
    float shallowTurquoise = clamp(shallowEstimate * clarity, 0.0f, 1.0f);

    float causticA = voronoiCaustic(worldPos.xz * 1.45f + waveNrm.xz * 0.55f, f.timeSeconds);
    float causticB = voronoiCaustic(worldPos.xz * 2.30f - float2(f.timeSeconds * 0.16f, f.timeSeconds * 0.11f), f.timeSeconds * 0.85f);
    float caustics = (causticA * 0.65f + causticB * 0.45f) * (0.25f + 0.75f * shallowTurquoise) * exp(-length(relPos) * 0.045f);

    float3 shallowBedTint = lerp(float3(0.03f, 0.36f, 0.48f), float3(0.07f, 0.72f, 0.70f), shallowTurquoise);
    float3 deepBodyTint = float3(0.010f, 0.075f, 0.22f);
    float3 sunColor = lerp(float3(1.25f, 1.18f, 1.05f), float3(1.80f, 0.85f, 0.38f), env.sunsetFactor);
    sunColor = lerp(sunColor, float3(0.16f, 0.24f, 0.42f), env.nightFactor);

    float3 sunDir = normalize(f.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float ambientScale = lerp(lerp(1.05f, 0.42f, env.sunsetFactor), 0.045f, env.nightFactor)
                       * (1.0f - env.rainFactor * 0.55f);
    float3 refractedBody = lerp(deepBodyTint, shallowBedTint, 0.30f + 0.70f * shallowTurquoise)
                         * (0.62f + 0.52f * luminance(rawLinear)) * ambientScale;
    float causticVis = (1.0f - env.nightFactor * 0.92f) * (1.0f - env.rainFactor * 0.78f)
                     * smoothstep(0.12f, 0.36f, sunDir.y);
    refractedBody += sunColor * float3(0.42f, 0.92f, 1.00f) * caustics * 0.22f * causticVis;

    float3 analyticRefl = evalDimensionSky(reflDir, f, env, true);
    float3 reflRadiance = lerp(analyticRefl, externalReflCol, clamp(externalReflWeight, 0.0f, 0.95f));

    float sunVis = smoothstep(-0.08f, 0.06f, sunDir.y) * (1.0f - env.rainFactor * 0.85f);
    float glitter = evalGgxSunHighlight(viewDir, reflDir, waveNrm, sunDir, lerp(0.060f, 0.14f, env.rainFactor));
    float3 specColor = lerp(float3(2.2f, 1.9f, 1.5f), float3(2.8f, 1.15f, 0.44f), env.sunsetFactor);
    reflRadiance += specColor * glitter * sunVis;

    float cosV = saturate(dot(viewDir, waveNrm));
    float fresnel = clamp(0.025f + 0.975f * pow(saturate(1.0f - cosV), 4.6f), 0.04f, 0.96f);
    fresnel = lerp(fresnel, clamp(fresnel * 1.18f + 0.06f, 0.0f, 0.97f), env.sunsetFactor);

    return lerp(refractedBody, reflRadiance, fresnel);
}

float3 applyUnderwaterOverlay(MetalityFrame f, MeteorEnvFactors env, float3 colorLin, float3 relPos, float3 worldPos, float dist) {
    if (f.submergedMode == 1) {
        float absorb = 1.0f - exp(-dist * 0.056f);
        float3 dayWaterFog = lerp(float3(0.024f, 0.28f, 0.39f), float3(0.048f, 0.20f, 0.29f), env.sunsetFactor);
        float3 waterFog = lerp(dayWaterFog, float3(0.008f, 0.038f, 0.095f), env.nightFactor);
        return lerp(colorLin, waterFog, clamp(absorb, 0.0f, 0.92f));
    }
    if (f.submergedMode == 2) {
        float absorb = 1.0f - exp(-dist * 0.24f);
        float3 lavaFog = float3(1.65f, 0.40f, 0.06f);
        return lerp(colorLin, lavaFog, clamp(absorb, 0.0f, 0.96f));
    }
    return colorLin;
}

struct WaterDdaHit {
    bool hit;
    bool blocked;
    float t;
    float3 nrm;
    bool isVertical;
    float2 flowVec;
};

bool inWaterVoxelBounds512(int3 c) {
    return (c.x >= 0 && c.x < 512 && c.y >= 0 && c.y < 128 && c.z >= 0 && c.z < 512);
}

uint waterVoxelIdx512(int3 c) {
    return uint((c.y * 512 + c.z) * 512 + c.x);
}

bool isWaterBlockingVoxel(uint tag) {
    return (tag >= 1u && tag <= 7u && tag != 3u) || (tag == 8u) || (tag == 12u);
}

bool isWaterRetainingVoxel(uint tag, uint sub) {
    return isWaterBlockingVoxel(tag) || (tag == 15u) || (tag == 14u && sub >= 1u && sub <= 13u);
}

float sampleWaterCellHeight(StructuredBuffer<uint> vGrid, int3 cell) {
    if (!inWaterVoxelBounds512(cell)) {
        return -1.0f;
    }
    uint idx = waterVoxelIdx512(cell);
    uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    uint tag = (b >> 4u) & 0xFu;
    uint sub = b & 0xFu;
    if (isWaterRetainingVoxel(tag, sub)) {
        return -2.0f;
    }
    if (tag == 3u) {
        int3 cAbove = cell + int3(0, 1, 0);
        if (inWaterVoxelBounds512(cAbove)) {
            uint idxA = waterVoxelIdx512(cAbove);
            uint tagA = ((vGrid[idxA >> 2u] >> ((idxA & 3u) * 8u)) >> 4u) & 0xFu;
            if (tagA == 3u) {
                return 1.0f;
            }
        }
        if (sub >= 9u) {
            return 1.0f;
        }
        if (sub == 8u || sub == 0u) {
            return 0.888889f;
        }
        return float(sub) / 9.0f;
    }
    return -1.0f;
}

float2 accumWaterCornerSample(float h, float2 sw) {
    if (h >= 0.8f) {
        return sw + float2(h * 10.0f, 10.0f);
    } else if (h >= 0.0f) {
        return sw + float2(h, 1.0f);
    } else if (h > -1.5f) {
        return sw + float2(0.0f, 1.0f);
    }
    return sw;
}

float evalWaterCornerHeight(StructuredBuffer<uint> vGrid, int3 c0, int dx, int dz) {
    float hSelf = sampleWaterCellHeight(vGrid, c0);
    if (hSelf >= 0.999f) {
        return 1.0f;
    }
    int sx = (dx == 0) ? -1 : 1;
    int sz = (dz == 0) ? -1 : 1;
    float hX = sampleWaterCellHeight(vGrid, c0 + int3(sx, 0, 0));
    float hZ = sampleWaterCellHeight(vGrid, c0 + int3(0, 0, sz));
    if (hX >= 0.999f || hZ >= 0.999f) {
        return 1.0f;
    }

    float2 sw = float2(0.0f, 0.0f);
    sw = accumWaterCornerSample(hSelf, sw);
    sw = accumWaterCornerSample(hX, sw);
    sw = accumWaterCornerSample(hZ, sw);
    if (hX > 0.0f || hZ > 0.0f) {
        float hD = sampleWaterCellHeight(vGrid, c0 + int3(sx, 0, sz));
        if (hD >= 0.999f) {
            return 1.0f;
        }
        sw = accumWaterCornerSample(hD, sw);
    }
    if (sw.y <= 1e-4f) {
        return max(hSelf, 0.111f);
    }
    return clamp(sw.x / sw.y, 0.06f, 1.0f);
}

// Returns: 0 = flat water neighbor, 1 = breaks flat, 2 = non-water retaining block (triggers diag check)
int classifyWaterFlatNeighbor(StructuredBuffer<uint> vGrid, int3 nc) {
    if (!inWaterVoxelBounds512(nc)) {
        return 0;
    }
    uint ni = waterVoxelIdx512(nc);
    uint nb = (vGrid[ni >> 2u] >> ((ni & 3u) * 8u)) & 0xFFu;
    uint nTag = (nb >> 4u) & 0xFu;
    uint nSub = nb & 0xFu;
    if ((nTag == 3u && nSub != 8u && nSub != 0u) ||
        (nTag != 3u && !isWaterRetainingVoxel(nTag, nSub))) {
        return 1;
    }
    if (nTag != 3u) {
        return 2;
    }
    int3 nca = nc + int3(0, 1, 0);
    if (inWaterVoxelBounds512(nca)) {
        uint nai = waterVoxelIdx512(nca);
        if ((((vGrid[nai >> 2u] >> ((nai & 3u) * 8u)) >> 4u) & 0xFu) == 3u) {
            return 1;
        }
    }
    return 0;
}

WaterDdaHit traceWaterVoxelFine(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 camPos,
    float3 rayDir,
    float tStart,
    float tEnd,
    float tTerrainMax,
    int maxSteps
) {
    WaterDdaHit res;
    res.hit = false;
    res.blocked = false;
    res.t = min(tEnd, tTerrainMax);
    res.nrm = float3(0.0f, 1.0f, 0.0f);
    res.isVertical = false;
    res.flowVec = float2(0.0f, 0.0f);

    float tLimit = min(tEnd, tTerrainMax);
    if (tLimit <= tStart + 1e-4f) {
        return res;
    }

    float3 pos = (camPos + rayDir * (tStart + 1e-4f)) - float3(vOrg);
    int3 cell = int3(floor(pos));
    float3 invDir = float3(
        abs(rayDir.x) > 1e-5f ? 1.0f / rayDir.x : 1e5f,
        abs(rayDir.y) > 1e-5f ? 1.0f / rayDir.y : 1e5f,
        abs(rayDir.z) > 1e-5f ? 1.0f / rayDir.z : 1e5f
    );
    int3 stepV = int3(
        rayDir.x >= 0.0f ? 1 : -1,
        rayDir.y >= 0.0f ? 1 : -1,
        rayDir.z >= 0.0f ? 1 : -1
    );
    float3 tAxisMax = float3(
        ((float(cell.x) + (stepV.x > 0 ? 1.0f : 0.0f)) - pos.x) * invDir.x,
        ((float(cell.y) + (stepV.y > 0 ? 1.0f : 0.0f)) - pos.y) * invDir.y,
        ((float(cell.z) + (stepV.z > 0 ? 1.0f : 0.0f)) - pos.z) * invDir.z
    );
    float3 tDelta = abs(invDir);

    float3 hitNrm = -rayDir;
    float curT = 0.0f;
    float baseT = tStart + 1e-4f;
    bool prevWasWater = false;

    for (int iter = 0; iter < 32; ++iter) {
        if (iter >= maxSteps || (baseT + curT) >= tLimit) {
            break;
        }
        float tEntry = max(baseT + curT, tStart);
        float tExit = min(baseT + min(tAxisMax.x, min(tAxisMax.y, tAxisMax.z)), tLimit);

        if (inWaterVoxelBounds512(cell)) {
            uint idx = waterVoxelIdx512(cell);
            uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
            uint tag = (b >> 4u) & 0xFu;
            uint sub = b & 0xFu;

            if (isWaterBlockingVoxel(tag)) {
                res.blocked = true;
                return res;
            }

            if (tag == 3u) {
                if (iter == 0) {
                    float3 localF = pos - float3(cell) - 0.5f;
                    if (abs(localF.y) >= max(abs(localF.x), abs(localF.z))) {
                        hitNrm = float3(0.0f, rayDir.y < 0.0f ? 1.0f : -1.0f, 0.0f);
                    } else if (abs(localF.x) > abs(localF.z)) {
                        hitNrm = float3(rayDir.x < 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
                    } else {
                        hitNrm = float3(0.0f, 0.0f, rayDir.z < 0.0f ? 1.0f : -1.0f);
                    }
                }

                bool aboveIsWater = false;
                int3 cAbove = cell + int3(0, 1, 0);
                if (inWaterVoxelBounds512(cAbove)) {
                    uint idxA = waterVoxelIdx512(cAbove);
                    uint tagA = ((vGrid[idxA >> 2u] >> ((idxA & 3u) * 8u)) >> 4u) & 0xFu;
                    aboveIsWater = (tagA == 3u);
                }

                if (sub >= 9u || aboveIsWater) {
                    if (sub != 10u || abs(hitNrm.y) > 0.5f) {
                        res.hit = true;
                        res.t = tEntry;
                        res.nrm = normalize(hitNrm);
                        res.isVertical = abs(res.nrm.y) < 0.5f;
                        return res;
                    }
                } else {
                    float h00 = 0.888889f;
                    float h10 = 0.888889f;
                    float h01 = 0.888889f;
                    float h11 = 0.888889f;
                    bool isStillFlat = (sub == 8u || sub == 0u);
                    if (isStillFlat) {
                        int c0 = classifyWaterFlatNeighbor(vGrid, cell + int3(-1, 0,  0));
                        int c1 = (c0 != 1) ? classifyWaterFlatNeighbor(vGrid, cell + int3( 1, 0,  0)) : 1;
                        int c2 = (c1 != 1) ? classifyWaterFlatNeighbor(vGrid, cell + int3( 0, 0, -1)) : 1;
                        int c3 = (c2 != 1) ? classifyWaterFlatNeighbor(vGrid, cell + int3( 0, 0,  1)) : 1;
                        if (c0 == 1 || c1 == 1 || c2 == 1 || c3 == 1) {
                            isStillFlat = false;
                        } else if (c0 == 2 || c1 == 2 || c2 == 2 || c3 == 2) {
                            if (classifyWaterFlatNeighbor(vGrid, cell + int3(-1, 0, -1)) == 1 ||
                                classifyWaterFlatNeighbor(vGrid, cell + int3( 1, 0, -1)) == 1 ||
                                classifyWaterFlatNeighbor(vGrid, cell + int3(-1, 0,  1)) == 1 ||
                                classifyWaterFlatNeighbor(vGrid, cell + int3( 1, 0,  1)) == 1) {
                                isStillFlat = false;
                            }
                        }
                    }
                    if (!isStillFlat) {
                        h00 = evalWaterCornerHeight(vGrid, cell, 0, 0);
                        h10 = evalWaterCornerHeight(vGrid, cell, 1, 0);
                        h01 = evalWaterCornerHeight(vGrid, cell, 0, 1);
                        h11 = evalWaterCornerHeight(vGrid, cell, 1, 1);
                    }

                    float3 cellRelCam = camPos - float3(vOrg + cell);
                    float3 p0Raw = cellRelCam + rayDir * tEntry;
                    float3 p1Raw = cellRelCam + rayDir * tExit;
                    float2 p0xz = clamp(p0Raw.xz, float2(0.0f), float2(1.0f));
                    float2 p1xz = clamp(p1Raw.xz, float2(0.0f), float2(1.0f));
                    float s0 = lerp(lerp(h00, h10, p0xz.x), lerp(h01, h11, p0xz.x), p0xz.y);
                    float s1 = lerp(lerp(h00, h10, p1xz.x), lerp(h01, h11, p1xz.x), p1xz.y);

                    if (p0Raw.y <= s0 + 0.002f) {
                        float dhdx = lerp(h10 - h00, h11 - h01, p0xz.y);
                        float dhdz = lerp(h01 - h00, h11 - h10, p0xz.x);
                        bool allowVert = (!prevWasWater) && (abs(hitNrm.y) < 0.5f);
                        res.hit = true;
                        res.t = tEntry;
                        res.nrm = allowVert ? normalize(hitNrm) : normalize(float3(-dhdx, 1.0f, -dhdz));
                        res.isVertical = allowVert;
                        res.flowVec = float2(-dhdx, -dhdz);
                        return res;
                    } else if (p1Raw.y <= s1 + 0.002f) {
                        float ta = tEntry;
                        float tb = tExit;
                        float fa = p0Raw.y - s0;
                        float fb = p1Raw.y - s1;
                        for (int ri = 0; ri < 2; ++ri) {
                            float frac = clamp(fa / max(fa - fb, 1e-5f), 0.0f, 1.0f);
                            float tm = lerp(ta, tb, frac);
                            float3 pm = cellRelCam + rayDir * tm;
                            float2 pmxz = clamp(pm.xz, float2(0.0f), float2(1.0f));
                            float sm = lerp(lerp(h00, h10, pmxz.x), lerp(h01, h11, pmxz.x), pmxz.y);
                            float fm = pm.y - sm;
                            if (fm > 0.0f) {
                                ta = tm;
                                fa = fm;
                            } else {
                                tb = tm;
                                fb = fm;
                            }
                        }
                        float tHit = lerp(ta, tb, clamp(fa / max(fa - fb, 1e-5f), 0.0f, 1.0f));
                        if (tHit <= tTerrainMax - 0.003f) {
                            float3 ph = cellRelCam + rayDir * tHit;
                            float2 phxz = clamp(ph.xz, float2(0.0f), float2(1.0f));
                            float dhdx = lerp(h10 - h00, h11 - h01, phxz.y);
                            float dhdz = lerp(h01 - h00, h11 - h10, phxz.x);
                            res.hit = true;
                            res.t = tHit;
                            res.nrm = normalize(float3(-dhdx, 1.0f, -dhdz));
                            res.isVertical = false;
                            res.flowVec = float2(-dhdx, -dhdz);
                            return res;
                        }
                    }
                }
                prevWasWater = true;
            } else {
                prevWasWater = false;
            }
        } else {
            prevWasWater = false;
        }

        if (tAxisMax.x < tAxisMax.y) {
            if (tAxisMax.x < tAxisMax.z) {
                curT = tAxisMax.x;
                tAxisMax.x += tDelta.x;
                cell.x += stepV.x;
                hitNrm = float3(-float(stepV.x), 0.0f, 0.0f);
            } else {
                curT = tAxisMax.z;
                tAxisMax.z += tDelta.z;
                cell.z += stepV.z;
                hitNrm = float3(0.0f, 0.0f, -float(stepV.z));
            }
        } else {
            if (tAxisMax.y < tAxisMax.z) {
                curT = tAxisMax.y;
                tAxisMax.y += tDelta.y;
                cell.y += stepV.y;
                hitNrm = float3(0.0f, -float(stepV.y), 0.0f);
            } else {
                curT = tAxisMax.z;
                tAxisMax.z += tDelta.z;
                cell.z += stepV.z;
                hitNrm = float3(0.0f, 0.0f, -float(stepV.z));
            }
        }
    }
    return res;
}

struct WaterYBounds {
    bool valid;
    float yMin;
    float yMax;
};

WaterYBounds decodeWaterYBounds(int voxelFlags) {
    WaterYBounds b;
    uint vf = uint(voxelFlags);
    if ((vf & 0x80u) == 0u) {
        b.valid = false;
        b.yMin = -1e6f;
        b.yMax = 1e6f;
        return b;
    }
    b.valid = true;
    b.yMin = float((vf >> 8u) & 0x7FFu) - 512.0f - 1.0f;
    b.yMax = float((vf >> 19u) & 0x7FFu) - 512.0f + 1.5f;
    return b;
}

bool rayOverlapsWaterYBounds(MetalityFrame f, float startY, float endY, float3 rayDir, bool isSky) {
    uint vf = uint(f.voxelOrigin.w);
    if ((vf & 0x80u) == 0u) {
        return true;
    }
    float yMin = float((vf >> 8u) & 0x7FFu) - 512.0f - 1.0f;
    float yMax = float((vf >> 19u) & 0x7FFu) - 512.0f + 1.5f;
    if (startY > yMax && rayDir.y >= -1e-5f) {
        return false;
    }
    if (startY < yMin && rayDir.y <= 1e-5f) {
        return false;
    }
    if (!isSky) {
        float rMin = min(startY, endY);
        float rMax = max(startY, endY);
        if (rMax < yMin || rMin > yMax) {
            return false;
        }
    }
    return true;
}

WaterDdaHit traceWaterMacroDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 camPos,
    float3 rayDir,
    float tStart,
    float tMaxDist,
    int voxelFlags
) {
    WaterDdaHit res;
    res.hit = false;
    res.blocked = false;
    res.t = tMaxDist;
    res.nrm = float3(0.0f, 1.0f, 0.0f);
    res.isVertical = false;
    res.flowVec = float2(0.0f, 0.0f);

    float tMax = min(tMaxDist, 224.0f);
    if (tMax <= tStart + 1e-3f) {
        return res;
    }

    float3 boxMin = float3(vOrg);
    float3 boxMax = boxMin + float3(512.0f, 128.0f, 512.0f);
    WaterYBounds wb = decodeWaterYBounds(voxelFlags);
    if (wb.valid) {
        boxMin.y = max(boxMin.y, wb.yMin);
        boxMax.y = min(boxMax.y, wb.yMax);
        if (boxMax.y <= boxMin.y + 1e-3f) {
            return res;
        }
    }
    float3 invDir = float3(
        abs(rayDir.x) > 1e-5f ? 1.0f / rayDir.x : (rayDir.x >= 0.0f ? 1e5f : -1e5f),
        abs(rayDir.y) > 1e-5f ? 1.0f / rayDir.y : (rayDir.y >= 0.0f ? 1e5f : -1e5f),
        abs(rayDir.z) > 1e-5f ? 1.0f / rayDir.z : (rayDir.z >= 0.0f ? 1e5f : -1e5f)
    );
    float3 t0 = (boxMin - camPos) * invDir;
    float3 t1 = (boxMax - camPos) * invDir;
    float3 tNear3 = min(t0, t1);
    float3 tFar3 = max(t0, t1);
    float tBoxEnter = max(tNear3.x, max(tNear3.y, tNear3.z));
    float tBoxExit = min(tFar3.x, min(tFar3.y, tFar3.z));
    if (tBoxExit <= max(tBoxEnter, tStart) + 1e-3f || tBoxEnter >= tMax) {
        return res;
    }

    float tMarchStart = max(tStart, tBoxEnter + 0.01f);
    tMax = min(tMax, tBoxExit);
    if (tMax <= tMarchStart + 1e-3f) {
        return res;
    }

    float3 mPos = ((camPos + rayDir * tMarchStart) - float3(vOrg)) * 0.125f;
    int3 mCell = clamp(int3(floor(mPos)), int3(0, 0, 0), int3(63, 15, 63));
    int3 stepM = int3(
        rayDir.x >= 0.0f ? 1 : -1,
        rayDir.y >= 0.0f ? 1 : -1,
        rayDir.z >= 0.0f ? 1 : -1
    );
    float3 tMaxM = float3(
        ((float(mCell.x) + (stepM.x > 0 ? 1.0f : 0.0f)) - mPos.x) * 8.0f * invDir.x,
        ((float(mCell.y) + (stepM.y > 0 ? 1.0f : 0.0f)) - mPos.y) * 8.0f * invDir.y,
        ((float(mCell.z) + (stepM.z > 0 ? 1.0f : 0.0f)) - mPos.z) * 8.0f * invDir.z
    );
    float3 tDeltaM = abs(invDir) * 8.0f;

    float curT = 0.0f;
    for (int mi = 0; mi < 40; ++mi) {
        float tMacroEntry = tMarchStart + curT;
        if (tMacroEntry >= tMax) {
            break;
        }
        float tMacroExit = tMarchStart + min(tMaxM.x, min(tMaxM.y, tMaxM.z));

        if (mCell.x >= 0 && mCell.x < 64 && mCell.y >= 0 && mCell.y < 16 && mCell.z >= 0 && mCell.z < 64) {
            uint bitIdx = uint((mCell.y * 64 + mCell.z) * 64 + mCell.x);
            uint macroWord = vGrid[8388608u + (bitIdx >> 5u)];
            if ((macroWord & (1u << (bitIdx & 31u))) != 0u) {
                WaterDdaHit fine = traceWaterVoxelFine(
                    vGrid,
                    vOrg,
                    camPos,
                    rayDir,
                    max(tMacroEntry - 0.002f, tMarchStart),
                    min(tMacroExit + 0.002f, tMax),
                    tMax,
                    28
                );
                if (fine.hit || fine.blocked) {
                    return fine;
                }
            }
        } else {
            break;
        }

        if (tMaxM.x < tMaxM.y) {
            if (tMaxM.x < tMaxM.z) {
                curT = tMaxM.x;
                tMaxM.x += tDeltaM.x;
                mCell.x += stepM.x;
            } else {
                curT = tMaxM.z;
                tMaxM.z += tDeltaM.z;
                mCell.z += stepM.z;
            }
        } else {
            if (tMaxM.y < tMaxM.z) {
                curT = tMaxM.y;
                tMaxM.y += tDeltaM.y;
                mCell.y += stepM.y;
            } else {
                curT = tMaxM.z;
                tMaxM.z += tDeltaM.z;
                mCell.z += stepM.z;
            }
        }
    }
    return res;
}

WaterDdaHit traceWaterMacroDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 camPos,
    float3 rayDir,
    float tStart,
    float tMaxDist
) {
    return traceWaterMacroDda(vGrid, vOrg, camPos, rayDir, tStart, tMaxDist, 0);
}

float2 evalRainPuddleMask(float2 worldXZ, float rainFactor) {
    float rainGain = smoothstep(0.05f, 0.35f, rainFactor);
    if (rainGain <= 0.001f) {
        return float2(0.0f, 0.0f);
    }
    float pNoise = fbm2D(worldXZ * 0.16f) * 0.65f + fbm2D(worldXZ * 0.48f + float2(11.3f, 7.7f)) * 0.35f;
    float puddleMask = smoothstep(0.39f, 0.50f, pNoise) * rainGain;
    float wetMask = smoothstep(0.28f, 0.42f, pNoise) * rainGain;
    return float2(puddleMask, wetMask);
}

float2 evalRaindropRippleNormal(float2 worldXZ, float timeSec, float puddleStrength) {
    if (puddleStrength <= 0.01f) {
        return float2(0.0f, 0.0f);
    }
    float2 grad = float2(0.0f, 0.0f);
    for (int layer = 0; layer < 2; ++layer) {
        float sc = (layer == 0) ? 0.85f : 1.45f;
        float sp = (layer == 0) ? 1.35f : 1.75f;
        float2 p = worldXZ * sc;
        float2 baseCell = floor(p);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float2 cell = baseCell + float2(float(dx), float(dy));
                float2 rnd = hash22(cell + float2(float(layer) * 37.0f, float(layer) * 17.0f));
                float tCycle = frac(timeSec * sp + hash12(cell + float2(19.3f + float(layer) * 41.0f, 5.7f)));
                float2 center = cell + float2(0.16f, 0.16f) + rnd * 0.68f;
                float2 d = p - center;
                float dist = length(d);
                if (dist > 0.005f && dist < 0.78f) {
                    float rad = tCycle * 0.72f;
                    float dr = dist - rad;
                    float env = exp(-abs(dr) * 10.5f) * (1.0f - tCycle) * smoothstep(0.0f, 0.08f, tCycle) * (1.0f - smoothstep(0.18f, 0.76f, dist));
                    float dWave = cos(dr * 30.0f) * env;
                    grad += (d / dist) * dWave * (layer == 0 ? 0.145f : 0.095f);
                }
            }
        }
    }
    return grad * clamp(puddleStrength, 0.0f, 1.5f);
}

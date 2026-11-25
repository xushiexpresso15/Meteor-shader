// Meteor Shader Common Library — Math, Noise, Reversed-Z Reconstruction & AgX Punchy

static const float METEOR_PI = 3.14159265f;
static const float METEOR_TAU = 6.28318530f;

float3 getMeteorBlackHoleDir() {
    return normalize(float3(-0.56f, 0.34f, -0.75f));
}

float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 srgbToLinear(float3 c) {
    float3 x = max(c, float3(1e-6f, 1e-6f, 1e-6f));
    return pow(x, float3(2.2f, 2.2f, 2.2f));
}

float3 linearToSrgb(float3 c) {
    float3 x = max(c, float3(1e-6f, 1e-6f, 1e-6f));
    return pow(x, float3(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f));
}

float2 safeScreenUv(float2 pixelPos, MetalityFrame f, float scaleFactor) {
    float2 dim = max(float2(1.0f, 1.0f), round(float2(float(f.width), float(f.height)) * scaleFactor));
    return clamp(pixelPos / dim, float2(0.0005f, 0.0005f), float2(0.9995f, 0.9995f));
}

bool isSkyDepth(float d) {
    return d <= 1e-6f;
}

bool meteorFeatureShadows(MetalityFrame f) { return (f.tonemapMode & 0x200u) != 0u; }
bool meteorFeatureReflections(MetalityFrame f) { return (f.tonemapMode & 0x400u) != 0u; }
bool meteorFeatureWater(MetalityFrame f) { return (f.tonemapMode & 0x800u) != 0u; }
bool meteorFeatureClouds(MetalityFrame f) { return (f.tonemapMode & 0x1000u) != 0u; }
bool meteorFeatureGodrays(MetalityFrame f) { return (f.tonemapMode & 0x2000u) != 0u; }
bool meteorFeatureBloom(MetalityFrame f) { return (f.tonemapMode & 0x4000u) != 0u; }
bool meteorFeatureGiAo(MetalityFrame f) { return (f.tonemapMode & 0x8000u) != 0u; }
bool meteorFeaturePbrRain(MetalityFrame f) { return (f.tonemapMode & 0x10000u) != 0u; }
uint meteorShadowTier(MetalityFrame f) { return (f.tonemapMode >> 17) & 3u; }

struct MeteorDepthSample {
    float depth;
    int2 px;
    int2 dim;
    float2 centerUv;
};

int2 snapDepthCoord(int2 depthDim, float2 uv) {
    int2 dim = max(depthDim, int2(1, 1));
    return clamp(int2(floor(uv * float2(dim))), int2(0, 0), dim - int2(1, 1));
}

float2 texelCenterUv(int2 px, int2 dim) {
    return (float2(px) + 0.5f) / float2(max(dim, int2(1, 1)));
}

MeteorDepthSample loadDepthAtUv(Texture2D<float4> depthTex, float2 uv) {
    MeteorDepthSample s;
    s.dim = max(metality_tex_size(depthTex), int2(1, 1));
    s.px = snapDepthCoord(s.dim, uv);
    s.centerUv = texelCenterUv(s.px, s.dim);
    s.depth = depthTex.Load(int3(s.px, 0)).r;
    return s;
}

float3 snapCardinalAxis(float3 n) {
    float3 absN = abs(n);
    if (absN.y >= absN.x && absN.y >= absN.z) {
        return float3(0.0f, n.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
    } else if (absN.x >= absN.z) {
        return float3(n.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
    }
    return float3(0.0f, 0.0f, n.z >= 0.0f ? 1.0f : -1.0f);
}

float hash12(float2 p) {
    float3 p3 = frac(float3(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

float2 hash22(float2 p) {
    float3 p3 = frac(float3(p.xyx) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float hash13(float3 p3) {
    p3 = frac(p3 * 0.1031f);
    p3 += dot(p3, p3.zyx + 31.32f);
    return frac((p3.x + p3.y) * p3.z);
}

float interleavedGradientNoise(float2 pixelPos, float frameIdx) {
    float2 p = pixelPos + float2(5.588238f, 5.588238f) * frameIdx;
    return frac(52.9829189f * frac(dot(p, float2(0.06711056f, 0.00583715f))));
}

float2 pcg2dHash01(uint2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> uint2(16u, 16u));
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> uint2(16u, 16u));
    return float2(v & uint2(0x00FFFFFFu, 0x00FFFFFFu)) * (1.0f / 16777216.0f);
}

float2 ignPair(float2 pixelPos, float frameIdx) {
    float2 p = floor(max(pixelPos, float2(0.0f, 0.0f)));
    float2 base = float2(
        interleavedGradientNoise(p, 0.0f),
        interleavedGradientNoise(float2(p.y, p.x) + float2(17.0f, 31.0f), 0.0f)
    );
    return frac(base + float2(0.7548776662f, 0.5698402910f) * frameIdx);
}

// Zero-array bit-exact 4x4 Hammersley-Bayer ordered pair (0.0000 correlation, exact 0.000000 residual after step=1,2 1-2-1 À-Trous)
float2 bayer4x4Pair(float2 pixelPos) {
    uint2 p = uint2(max(floor(pixelPos), float2(0.0f, 0.0f)));
    uint x = p.x & 3u;
    uint y = p.y & 3u;
    uint m = (((x ^ y) & 1u) << 3u) | ((y & 1u) << 2u) | ((x ^ y) & 2u) | ((y >> 1u) & 1u);
    uint r = (((m & 1u) << 3u) | ((m & 2u) << 1u) | ((m & 4u) >> 1u) | ((m & 8u) >> 3u)) ^ 5u;
    return (float2(float(m), float(r)) + 0.5f) * (1.0f / 16.0f);
}

float noise2D(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0f - 2.0f * f);
    float a = hash12(i + float2(0.0f, 0.0f));
    float b = hash12(i + float2(1.0f, 0.0f));
    float c = hash12(i + float2(0.0f, 1.0f));
    float d = hash12(i + float2(1.0f, 1.0f));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float noise3D(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0f - 2.0f * f);
    float n000 = hash13(i + float3(0.0f, 0.0f, 0.0f));
    float n100 = hash13(i + float3(1.0f, 0.0f, 0.0f));
    float n010 = hash13(i + float3(0.0f, 1.0f, 0.0f));
    float n110 = hash13(i + float3(1.0f, 1.0f, 0.0f));
    float n001 = hash13(i + float3(0.0f, 0.0f, 1.0f));
    float n101 = hash13(i + float3(1.0f, 0.0f, 1.0f));
    float n011 = hash13(i + float3(0.0f, 1.0f, 1.0f));
    float n111 = hash13(i + float3(1.0f, 1.0f, 1.0f));
    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float fbm2D(float2 p) {
    float sum = 0.0f;
    float amp = 0.5f;
    for (int i = 0; i < 4; ++i) {
        sum += amp * noise2D(p);
        p = float2(p.x * 1.97f - p.y * 0.42f, p.x * 0.42f + p.y * 1.97f);
        amp *= 0.5f;
    }
    return sum;
}

float fbm3D(float3 p) {
    float sum = 0.0f;
    float amp = 0.52f;
    for (int i = 0; i < 3; ++i) {
        sum += amp * noise3D(p);
        p = p * 2.04f + float3(13.7f, 7.1f, 19.3f);
        amp *= 0.48f;
    }
    return sum;
}

float voronoiCaustic(float2 p, float t) {
    p += 0.22f * float2(sin(p.y * 1.7f + t * 1.35f), cos(p.x * 1.6f - t * 1.15f));
    float2 n = floor(p);
    float2 f = frac(p);
    float d1 = 8.0f;
    float d2 = 8.0f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 g = float2(float(i), float(j));
            float2 rnd = hash22(n + g);
            float2 o = 0.5f + 0.42f * sin(t * 1.35f + METEOR_TAU * rnd);
            float2 r = g + o - f;
            float d = dot(r, r);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    float edge = clamp(sqrt(d2) - sqrt(d1), 0.0f, 1.0f);
    return pow(1.0f - saturate(edge * 3.8f), 3.4f);
}

float3 reconstructViewPos(MetalityFrame f, float2 uv, float depth) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, max(depth, 1e-5f), 1.0f);
    float4 viewH = mul(f.invProj, clip);
    float w = abs(viewH.w) > 1e-6f ? viewH.w : (viewH.w >= 0.0f ? 1e-6f : -1e-6f);
    return viewH.xyz / w;
}

float3 reconstructRelWorldPos(MetalityFrame f, float2 uv, float depth) {
    float3 viewPos = reconstructViewPos(f, uv, depth);
    return mul(f.invViewRot, float4(viewPos, 0.0f)).xyz;
}

float3 reconstructSkyWorldDir(MetalityFrame f, float2 uv) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, 0.002f, 1.0f);
    float4 viewH = mul(f.invProj, clip);
    float w = abs(viewH.w) > 1e-6f ? viewH.w : (viewH.w >= 0.0f ? 1e-6f : -1e-6f);
    float3 viewDir = normalize(viewH.xyz / w);
    return normalize(mul(f.invViewRot, float4(viewDir, 0.0f)).xyz);
}

struct MeteorNormalDiffs {
    float3 nrm;
    float3 dx;
    float3 dy;
};

float evalNormalCandidateScore(float3 a, float3 b, float3 viewRay) {
    float3 cn = cross(a, b);
    float cLen2 = dot(cn, cn);
    if (cLen2 <= 1e-16f) {
        return -1.0f;
    }
    cn *= rsqrt(cLen2);
    return max(abs(cn.x), max(abs(cn.y), abs(cn.z))) + 0.35f * abs(dot(cn, viewRay));
}

MeteorNormalDiffs reconstructWorldNormalWithDiffs(
    MetalityFrame f,
    float2 uv,
    float2 texel,
    float3 centerRelPos,
    float dL,
    float dR,
    float dD,
    float dU,
    float2 uvL,
    float2 uvR,
    float2 uvD,
    float2 uvU
) {
    MeteorNormalDiffs res;
    float3 pL = isSkyDepth(dL) ? centerRelPos : reconstructRelWorldPos(f, uvL, dL);
    float3 pR = isSkyDepth(dR) ? centerRelPos : reconstructRelWorldPos(f, uvR, dR);
    float3 pD = isSkyDepth(dD) ? centerRelPos : reconstructRelWorldPos(f, uvD, dD);
    float3 pU = isSkyDepth(dU) ? centerRelPos : reconstructRelWorldPos(f, uvU, dU);

    bool missL = isSkyDepth(dL) || abs(uv.x - uvL.x) < texel.x * 0.25f;
    bool missR = isSkyDepth(dR) || abs(uvR.x - uv.x) < texel.x * 0.25f;
    bool missD = isSkyDepth(dD) || abs(uv.y - uvD.y) < texel.y * 0.25f;
    bool missU = isSkyDepth(dU) || abs(uvU.y - uv.y) < texel.y * 0.25f;

    float3 dx1 = pR - centerRelPos;
    float3 dx2 = centerRelPos - pL;
    float centerDist = length(centerRelPos);
    float3 viewRay = (centerDist > 1e-5f) ? (-centerRelPos / centerDist) : float3(0.0f, 0.0f, 1.0f);
    float d1x = dot(dx1, viewRay);
    float d2x = dot(dx2, viewRay);
    bool useRight = !missR && (missL || abs(d1x) <= abs(d2x));
    float3 dx = (missR && !missL) ? dx2 : (useRight ? dx1 : dx2);
    if (!missL && !missR && d1x * d2x < 0.0f && abs(d1x - d2x) < 0.08f) {
        dx = 0.5f * (dx1 + dx2);
    }

    float3 dy1 = pU - centerRelPos;
    float3 dy2 = centerRelPos - pD;
    float d1y = dot(dy1, viewRay);
    float d2y = dot(dy2, viewRay);
    bool useUp = !missU && (missD || abs(d1y) <= abs(d2y));
    float3 dy = (missU && !missD) ? dy2 : (useUp ? dy1 : dy2);
    if (!missD && !missU && d1y * d2y < 0.0f && abs(d1y - d2y) < 0.08f) {
        dy = 0.5f * (dy1 + dy2);
    }

    res.dx = dx;
    res.dy = dy;

    float silJump = max(0.50f, centerDist * 0.14f);
    bool steepX = (missL || abs(d2x) > silJump) && (missR || abs(d1x) > silJump);
    bool steepY = (missD || abs(d2y) > silJump) && (missU || abs(d1y) > silJump);
    if (steepX || steepY) {
        res.nrm = viewRay;
        return res;
    }

    // Sub-block flat cutout decal step recovery (zero stack arrays!)
    bool decalStepX = !missL && !missR && abs(d1x - d2x) > 0.001f && abs(d1x - d2x) < 0.08f;
    bool decalStepY = !missD && !missU && abs(d1y - d2y) > 0.001f && abs(d1y - d2y) < 0.08f;
    if (decalStepX || decalStepY) {
        float3 refDy = decalStepY ? (0.5f * (dy1 + dy2)) : dy;
        if (decalStepX) {
            float3 dxAvg = 0.5f * (dx1 + dx2);
            float sc0 = evalNormalCandidateScore(dx1, refDy, viewRay);
            float sc1 = evalNormalCandidateScore(dx2, refDy, viewRay);
            float sc2 = evalNormalCandidateScore(dxAvg, refDy, viewRay);
            if (sc0 >= sc1 && sc0 >= sc2 && sc0 > -0.5f) dx = dx1;
            else if (sc1 >= sc2 && sc1 > -0.5f) dx = dx2;
            else if (sc2 > -0.5f) dx = dxAvg;
        }
        if (decalStepY) {
            float3 dyAvg = 0.5f * (dy1 + dy2);
            float sc0 = evalNormalCandidateScore(dx, dy1, viewRay);
            float sc1 = evalNormalCandidateScore(dx, dy2, viewRay);
            float sc2 = evalNormalCandidateScore(dx, dyAvg, viewRay);
            if (sc0 >= sc1 && sc0 >= sc2 && sc0 > -0.5f) dy = dy1;
            else if (sc1 >= sc2 && sc1 > -0.5f) dy = dy2;
            else if (sc2 > -0.5f) dy = dyAvg;
        }
        res.dx = dx;
        res.dy = dy;
    }

    float lenDx2 = dot(dx, dx);
    float lenDy2 = dot(dy, dy);
    if (lenDx2 < 1e-18f || lenDy2 < 1e-18f) {
        res.nrm = viewRay;
        return res;
    }
    float3 uDx = dx * rsqrt(lenDx2);
    float3 uDy = dy * rsqrt(lenDy2);
    float3 n = cross(uDx, uDy);
    float len2 = dot(n, n);
    if (len2 < 1e-8f) {
        res.nrm = viewRay;
        return res;
    }
    n *= rsqrt(len2);
    if (dot(n, viewRay) < 0.0f) {
        n = -n;
    }
    res.nrm = n;
    return res;
}

float3 reconstructWorldNormalFromDepths(
    MetalityFrame f,
    float2 uv,
    float2 texel,
    float3 centerRelPos,
    float dL,
    float dR,
    float dD,
    float dU,
    float2 uvL,
    float2 uvR,
    float2 uvD,
    float2 uvU
) {
    return reconstructWorldNormalWithDiffs(f, uv, texel, centerRelPos, dL, dR, dD, dU, uvL, uvR, uvD, uvU).nrm;
}

float3 reconstructFastNormal(
    MetalityFrame f,
    float2 uv,
    float3 centerRelPos,
    Texture2D<float4> depthTex
) {
    int2 depthDim = max(metality_tex_size(depthTex), int2(1, 1));
    int2 px = snapDepthCoord(depthDim, uv);
    float2 centerUv = texelCenterUv(px, depthDim);
    float2 depthTexel = float2(1.0f, 1.0f) / float2(depthDim);
    int2 pxL = int2(max(px.x - 1, 0), px.y);
    int2 pxR = int2(min(px.x + 1, depthDim.x - 1), px.y);
    int2 pxD = int2(px.x, max(px.y - 1, 0));
    int2 pxU = int2(px.x, min(px.y + 1, depthDim.y - 1));
    float dL = depthTex.Load(int3(pxL, 0)).r;
    float dR = depthTex.Load(int3(pxR, 0)).r;
    float dD = depthTex.Load(int3(pxD, 0)).r;
    float dU = depthTex.Load(int3(pxU, 0)).r;
    float2 uvL = texelCenterUv(pxL, depthDim);
    float2 uvR = texelCenterUv(pxR, depthDim);
    float2 uvD = texelCenterUv(pxD, depthDim);
    float2 uvU = texelCenterUv(pxU, depthDim);
    return reconstructWorldNormalFromDepths(f, centerUv, depthTexel, centerRelPos, dL, dR, dD, dU, uvL, uvR, uvD, uvU);
}

float3 reconstructSimpleNormal(
    MetalityFrame f,
    int2 px,
    int2 depthDim,
    float dC,
    Texture2D<float4> depthTex
) {
    int2 maxC = max(depthDim - int2(1, 1), int2(0, 0));
    int2 cpx = clamp(px, int2(0, 0), maxC);
    float3 pC = reconstructViewPos(f, texelCenterUv(cpx, depthDim), dC);
    if (isSkyDepth(dC)) {
        return normalize(mul(f.invViewRot, float4(normalize(-pC), 0.0f)).xyz);
    }
    int2 pxL = int2(max(cpx.x - 1, 0), cpx.y);
    int2 pxR = int2(min(cpx.x + 1, maxC.x), cpx.y);
    int2 pxD = int2(cpx.x, max(cpx.y - 1, 0));
    int2 pxU = int2(cpx.x, min(cpx.y + 1, maxC.y));
    float dL = depthTex.Load(int3(pxL, 0)).r;
    float dR = depthTex.Load(int3(pxR, 0)).r;
    float dD = depthTex.Load(int3(pxD, 0)).r;
    float dU = depthTex.Load(int3(pxU, 0)).r;
    bool hasL = (cpx.x > 0) && !isSkyDepth(dL);
    bool hasR = (cpx.x < maxC.x) && !isSkyDepth(dR);
    bool hasD = (cpx.y > 0) && !isSkyDepth(dD);
    bool hasU = (cpx.y < maxC.y) && !isSkyDepth(dU);
    if ((!hasL && !hasR) || (!hasD && !hasU)) {
        return normalize(mul(f.invViewRot, float4(normalize(-pC), 0.0f)).xyz);
    }
    bool useL = hasL && (!hasR || abs(dL - dC) < abs(dR - dC));
    bool useD = hasD && (!hasU || abs(dD - dC) < abs(dU - dC));
    int2 pxX = useL ? pxL : pxR;
    int2 pxY = useD ? pxD : pxU;
    float dX = useL ? dL : dR;
    float dY = useD ? dD : dU;
    float3 dxV = useL ? (pC - reconstructViewPos(f, texelCenterUv(pxX, depthDim), dX))
                      : (reconstructViewPos(f, texelCenterUv(pxX, depthDim), dX) - pC);
    float3 dyV = useD ? (pC - reconstructViewPos(f, texelCenterUv(pxY, depthDim), dY))
                      : (reconstructViewPos(f, texelCenterUv(pxY, depthDim), dY) - pC);
    float lenDx2 = dot(dxV, dxV);
    float lenDy2 = dot(dyV, dyV);
    if (lenDx2 < 1e-18f || lenDy2 < 1e-18f) {
        return normalize(mul(f.invViewRot, float4(normalize(-pC), 0.0f)).xyz);
    }
    float3 nV = cross(dxV * rsqrt(lenDx2), dyV * rsqrt(lenDy2));
    float len2 = dot(nV, nV);
    if (len2 < 1e-8f) {
        return normalize(mul(f.invViewRot, float4(normalize(-pC), 0.0f)).xyz);
    }
    nV *= rsqrt(len2);
    if (dot(nV, -pC) < 0.0f) {
        nV = -nV;
    }
    return normalize(mul(f.invViewRot, float4(nV, 0.0f)).xyz);
}

float3 reconstructSimpleNormal(
    MetalityFrame f,
    int2 px,
    int2 depthDim,
    Texture2D<float4> depthTex
) {
    int2 maxC = max(depthDim - int2(1, 1), int2(0, 0));
    int2 cpx = clamp(px, int2(0, 0), maxC);
    float dC = depthTex.Load(int3(cpx, 0)).r;
    return reconstructSimpleNormal(f, cpx, depthDim, dC, depthTex);
}

float2 worldToScreenUv(MetalityFrame f, float3 relWorldPos) {
    float3 viewPos = mul(f.viewRot, float4(relWorldPos, 0.0f)).xyz;
    float4 clip = mul(f.proj, float4(viewPos, 1.0f));
    if (clip.w <= 1e-4f) {
        return float2(-2.0f, -2.0f);
    }
    return clip.xy / clip.w * 0.5f + 0.5f;
}

struct MeteorEnvFactors {
    float dayFactor;
    float sunsetFactor;
    float nightFactor;
    float rainFactor;
    float netherSoulFactor;
    float netherCrimsonFactor;
    float netherWarpedFactor;
};

MeteorEnvFactors analyzeMeteorEnv(MetalityFrame f) {
    MeteorEnvFactors e;
    float3 sunDir = normalize(f.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float sunY = sunDir.y;
    e.dayFactor = smoothstep(0.04f, 0.34f, sunY);
    e.nightFactor = 1.0f - smoothstep(-0.22f, 0.03f, sunY);
    e.sunsetFactor = saturate(1.0f - abs(sunY - 0.04f) * 5.2f) * (1.0f - 0.65f * e.nightFactor);
    e.rainFactor = saturate(f.rainStrength);

    float3 fog = f.fogColor.rgb;
    float cyanDominance = (fog.b + fog.g * 0.85f) - fog.r * 1.95f;
    e.netherSoulFactor = smoothstep(-0.02f, 0.14f, cyanDominance);
    float purpleDominance = min(fog.r, fog.b) - fog.g * 1.35f;
    e.netherWarpedFactor = saturate(smoothstep(0.01f, 0.10f, purpleDominance) * (1.0f - e.netherSoulFactor));
    e.netherCrimsonFactor = saturate(1.0f - max(e.netherSoulFactor, e.netherWarpedFactor));
    return e;
}

struct MeteorPbrMaterial {
    float metallic;
    float roughness;
    float f0Scalar;
    float bumpScale;
    float reflectivity;
    bool active;
};

float3 decodeStainedGlassColor(uint sub) {
    if (sub == 2u) return float3(0.95f, 0.14f, 0.12f);
    if (sub == 3u) return float3(1.00f, 0.54f, 0.12f);
    if (sub == 4u) return float3(1.00f, 0.92f, 0.18f);
    if (sub == 5u) return float3(0.42f, 0.96f, 0.18f);
    if (sub == 6u) return float3(0.16f, 0.78f, 0.22f);
    if (sub == 7u) return float3(0.12f, 0.88f, 0.96f);
    if (sub == 8u) return float3(0.36f, 0.78f, 1.00f);
    if (sub == 9u) return float3(0.18f, 0.36f, 1.00f);
    if (sub == 10u) return float3(0.68f, 0.24f, 0.98f);
    if (sub == 11u) return float3(0.94f, 0.30f, 0.86f);
    if (sub == 12u) return float3(1.00f, 0.58f, 0.76f);
    if (sub == 13u) return float3(0.96f, 0.97f, 1.00f);
    if (sub == 14u) return float3(0.36f, 0.38f, 0.44f);
    if (sub == 15u) return float3(0.66f, 0.42f, 0.24f);
    return float3(0.92f, 0.96f, 1.00f);
}

struct MeteorSurfaceVoxel {
    uint activeTag;
    uint activeSub;
    uint emitLvl;
    uint solidTag;
    uint spriteTag;
    uint frontTag;
    uint upTag;
};

bool inVoxelBounds512Common(int3 c) {
    return (c.x >= 0 && c.x < 512 && c.y >= 0 && c.y < 128 && c.z >= 0 && c.z < 512);
}

uint voxelLinearIndex512Common(int3 c) {
    return uint((c.y * 512 + c.z) * 512 + c.x);
}

int2 sampleTopSolidAndOpaqueWorldY(StructuredBuffer<uint> vGrid, int gx, int gz) {
    if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) {
        return int2(-10000, -10000);
    }
    uint col = uint(gz) * 512u + uint(gx);
    uint word = vGrid[8421376u + (col >> 1u)];
    uint raw16 = (word >> ((col & 1u) * 16u)) & 0xFFFFu;
    uint solidRaw = raw16 & 0x0FFFu;
    if (solidRaw == 0u) {
        return int2(-10000, -10000);
    }
    int solidY = int(solidRaw) - 1024;
    int leafOff = int((raw16 >> 12u) & 0xFu) * 2;
    return int2(solidY, solidY + leafOff);
}

int sampleTopSolidWorldY(StructuredBuffer<uint> vGrid, int gx, int gz) {
    return sampleTopSolidAndOpaqueWorldY(vGrid, gx, gz).x;
}

int sampleTopSolidWorldY(StructuredBuffer<uint> vGrid, int3 vOrg, int wx, int wz) {
    return sampleTopSolidAndOpaqueWorldY(vGrid, wx - vOrg.x, wz - vOrg.z).x;
}

int sampleTopOpaqueWorldY(StructuredBuffer<uint> vGrid, int gx, int gz) {
    return sampleTopSolidAndOpaqueWorldY(vGrid, gx, gz).y;
}

int sampleTopOpaqueWorldY(StructuredBuffer<uint> vGrid, int3 vOrg, int wx, int wz) {
    return sampleTopSolidAndOpaqueWorldY(vGrid, wx - vOrg.x, wz - vOrg.z).y;
}

uint sampleTopOpaqueY(StructuredBuffer<uint> vGrid, int gx, int gz) {
    int wy = sampleTopOpaqueWorldY(vGrid, gx, gz);
    return (wy > -9000) ? uint(clamp(wy, 0, 255)) : 0u;
}

bool isColumnSkyBlockedFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 worldPos) {
    int3 p = int3(floor(worldPos)) - vOrg;
    if (p.x < 0 || p.x >= 512 || p.z < 0 || p.z >= 512) {
        return false;
    }
    int topSolidY = sampleTopSolidWorldY(vGrid, p.x, p.z);
    return (topSolidY > -9000) && (float(topSolidY) > worldPos.y + 0.35f);
}

bool isColumnOpaqueBlockedFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 worldPos) {
    int3 p = int3(floor(worldPos)) - vOrg;
    if (p.x < 0 || p.x >= 512 || p.z < 0 || p.z >= 512) {
        return false;
    }
    int topOpaqueY = sampleTopOpaqueWorldY(vGrid, p.x, p.z);
    return (topOpaqueY > -9000) && (topOpaqueY > int(floor(worldPos.y + 0.85f)));
}

float evalHeightmapFarShadow(StructuredBuffer<uint> vGrid, int3 vOrg, float3 worldPos, float3 sunDir) {
    if (sunDir.y <= 0.005f) {
        return 0.0f;
    }
    float2 gridCenterXZ = float2(float(vOrg.x) + 256.0f, float(vOrg.z) + 256.0f);
    float2 boxDelta = abs(worldPos.xz - gridCenterXZ);
    float boxEdgeDist = max(boxDelta.x, boxDelta.y);
    if (boxEdgeDist >= 252.0f) {
        return 1.0f;
    }
    float boundaryFade = 1.0f - smoothstep(210.0f, 252.0f, boxEdgeDist);

    float occ = 0.0f;
    int gx0 = int(floor(worldPos.x)) - vOrg.x;
    int gz0 = int(floor(worldPos.z)) - vOrg.z;
    if (gx0 >= 0 && gx0 < 512 && gz0 >= 0 && gz0 < 512) {
        int top0 = sampleTopOpaqueWorldY(vGrid, gx0, gz0);
        if (top0 > -9000) {
            float overheadM = float(top0) - worldPos.y;
            if (overheadM > 10.0f) {
                occ = max(occ, smoothstep(10.0f, 16.0f, overheadM));
            }
        }
    }

    float t = 1.2f;
    for (int i = 0; i < 10; ++i) {
        if (occ >= 0.999f) {
            break;
        }
        float3 p = worldPos + sunDir * t;
        int gx = int(floor(p.x)) - vOrg.x;
        int gz = int(floor(p.z)) - vOrg.z;
        if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) {
            break;
        }
        int topY = sampleTopOpaqueWorldY(vGrid, gx, gz);
        if (topY > -9000) {
            float hDiff = float(topY) - p.y;
            float penumbra = 0.85f + t * 0.045f;
            float stepOcc = smoothstep(0.25f, penumbra, hDiff);
            occ = max(occ, stepOcc);
        }
        t = t * 1.62f + 1.5f;
    }
    float shadowVis = 1.0f - clamp(occ, 0.0f, 1.0f);
    return lerp(1.0f, shadowVis, boundaryFade);
}

bool sampleSeaWaterMask1024(StructuredBuffer<uint> vGrid, int3 vOrg, int2 worldXZ) {
    int sx = worldXZ.x - (vOrg.x - 256);
    int sz = worldXZ.y - (vOrg.z - 256);
    if (sx < 0 || sx >= 1024 || sz < 0 || sz >= 1024) {
        return false;
    }
    uint bitIdx = uint(sz) * 1024u + uint(sx);
    uint word = vGrid[8552448u + (bitIdx >> 5u)];
    return (word & (1u << (bitIdx & 31u))) != 0u;
}

int sampleTopWaterWorldY(StructuredBuffer<uint> vGrid, int3 vOrg, int2 worldXZ) {
    int gx = worldXZ.x - vOrg.x;
    int gz = worldXZ.y - vOrg.z;
    if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) {
        return -10000;
    }
    uint col = uint(gz) * 512u + uint(gx);
    uint word = vGrid[8585216u + (col >> 1u)];
    uint raw16 = (word >> ((col & 1u) * 16u)) & 0xFFFFu;
    if (raw16 == 0u) {
        return -10000;
    }
    return int(raw16) - 1024;
}

bool isFarOrHighSeaWaterAt(StructuredBuffer<uint> vGrid, int3 vOrg, float3 hitWorld, int seaBlockY) {
    int2 wxz = int2(floor(hitWorld.xz));
    int gx = wxz.x - vOrg.x;
    int gz = wxz.y - vOrg.z;
    if (gx >= 0 && gx < 512 && gz >= 0 && gz < 512) {
        int topWaterY = sampleTopWaterWorldY(vGrid, vOrg, wxz);
        int topOpaqueY = sampleTopOpaqueWorldY(vGrid, gx, gz);
        if (topWaterY > -9000 && abs(topWaterY - seaBlockY) <= 2 && topOpaqueY <= topWaterY + 1) {
            return true;
        }
        if (sampleSeaWaterMask1024(vGrid, vOrg, wxz) && topOpaqueY <= seaBlockY + 1) {
            return true;
        }
        return false;
    }
    return sampleSeaWaterMask1024(vGrid, vOrg, wxz);
}

float evalWaterZenithTapOpen(StructuredBuffer<uint> vGrid, int gx, int gz, float surfY) {
    int2 ty = sampleTopSolidAndOpaqueWorldY(vGrid, clamp(gx, 0, 511), clamp(gz, 0, 511));
    float solidDiff = (ty.x > -9000) ? max(float(ty.x) - surfY, 0.0f) : 0.0f;
    float opaqueDiff = (ty.y > -9000) ? max(float(ty.y) - surfY, 0.0f) : 0.0f;
    float solidOpen = 1.0f - smoothstep(0.4f, 4.2f, solidDiff);
    float opaqueOpen = 1.0f - smoothstep(0.4f, 4.2f, opaqueDiff);
    // Cave solid roof drops to 0.0; overhanging outdoor foliage leaves soft canopy shade (~0.26)
    return lerp(solidOpen, opaqueOpen, 0.74f);
}

float sampleBilinearWaterZenithOpen(StructuredBuffer<uint> vGrid, float2 gridXZ, float surfY) {
    float2 gp = gridXZ - float2(0.5f, 0.5f);
    int2 g0 = int2(floor(gp));
    float2 fr = gp - float2(g0);
    float v00 = evalWaterZenithTapOpen(vGrid, g0.x,     g0.y,     surfY);
    float v10 = evalWaterZenithTapOpen(vGrid, g0.x + 1, g0.y,     surfY);
    float v01 = evalWaterZenithTapOpen(vGrid, g0.x,     g0.y + 1, surfY);
    float v11 = evalWaterZenithTapOpen(vGrid, g0.x + 1, g0.y + 1, surfY);
    return lerp(lerp(v00, v10, fr.x), lerp(v01, v11, fr.x), fr.y);
}

float evalWaterHorizonTapOpen(StructuredBuffer<uint> vGrid, int gx, int gz, float surfY, float invR) {
    int topY = sampleTopOpaqueWorldY(vGrid, clamp(gx, 0, 511), clamp(gz, 0, 511));
    if (topY <= -9000) {
        return 1.0f;
    }
    float diff = max(float(topY) - surfY, 0.0f);
    float slope = diff * invR;
    float sinElev = slope * rsqrt(1.0f + slope * slope);
    return saturate(1.0f - sinElev);
}

float sampleBilinearWaterHorizonOpen(StructuredBuffer<uint> vGrid, float2 gridXZ, float surfY, float invR) {
    float2 gp = gridXZ - float2(0.5f, 0.5f);
    int2 g0 = int2(floor(gp));
    float2 fr = gp - float2(g0);
    float v00 = evalWaterHorizonTapOpen(vGrid, g0.x,     g0.y,     surfY, invR);
    float v10 = evalWaterHorizonTapOpen(vGrid, g0.x + 1, g0.y,     surfY, invR);
    float v01 = evalWaterHorizonTapOpen(vGrid, g0.x,     g0.y + 1, surfY, invR);
    float v11 = evalWaterHorizonTapOpen(vGrid, g0.x + 1, g0.y + 1, surfY, invR);
    return lerp(lerp(v00, v10, fr.x), lerp(v01, v11, fr.x), fr.y);
}

float evalWaterSurfaceSkyVisFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 surfWorld, float tWater) {
    float2 gridXZ = surfWorld.xz - float2(float(vOrg.x), float(vOrg.z));
    if (gridXZ.x < 1.0f || gridXZ.x >= 510.0f || gridXZ.y < 1.0f || gridXZ.y >= 510.0f) {
        return 1.0f;
    }
    float surfY = surfWorld.y + 0.15f;
    if (tWater >= 64.0f) {
        int gx = clamp(int(floor(gridXZ.x)), 0, 511);
        int gz = clamp(int(floor(gridXZ.y)), 0, 511);
        float zVis = evalWaterZenithTapOpen(vGrid, gx, gz, surfY);
        float farOpen = 0.25f * (
            evalWaterHorizonTapOpen(vGrid, gx + 5, gz + 5, surfY, 0.142857f) +
            evalWaterHorizonTapOpen(vGrid, gx - 5, gz + 5, surfY, 0.142857f) +
            evalWaterHorizonTapOpen(vGrid, gx + 5, gz - 5, surfY, 0.142857f) +
            evalWaterHorizonTapOpen(vGrid, gx - 5, gz - 5, surfY, 0.142857f)
        );
        return clamp(zVis * (0.28f + 0.72f * farOpen), 0.0f, 1.0f);
    }
    float zenithVis = sampleBilinearWaterZenithOpen(vGrid, gridXZ, surfY);
    int gx = int(floor(gridXZ.x));
    int gz = int(floor(gridXZ.y));
    float nearOpen = 0.25f * (
        evalWaterHorizonTapOpen(vGrid, gx + 3, gz,     surfY, 0.333333f) +
        evalWaterHorizonTapOpen(vGrid, gx - 3, gz,     surfY, 0.333333f) +
        evalWaterHorizonTapOpen(vGrid, gx,     gz + 3, surfY, 0.333333f) +
        evalWaterHorizonTapOpen(vGrid, gx,     gz - 3, surfY, 0.333333f)
    );
    float farOpen = 0.25f * (
        evalWaterHorizonTapOpen(vGrid, gx + 5, gz + 5, surfY, 0.142857f) +
        evalWaterHorizonTapOpen(vGrid, gx - 5, gz + 5, surfY, 0.142857f) +
        evalWaterHorizonTapOpen(vGrid, gx + 5, gz - 5, surfY, 0.142857f) +
        evalWaterHorizonTapOpen(vGrid, gx - 5, gz - 5, surfY, 0.142857f)
    );
    float horizonOpen = nearOpen * 0.55f + farOpen * 0.45f;
    return clamp(zenithVis * (0.28f + 0.72f * horizonOpen), 0.0f, 1.0f);
}

float evalWaterSurfaceSkyVis(StructuredBuffer<uint> vGrid, int3 vOrg, float3 surfWorld) {
    return evalWaterSurfaceSkyVisFast(vGrid, vOrg, surfWorld, 8.0f);
}

float evalWaterSunOccTap(StructuredBuffer<uint> vGrid, int gx, int gz, float rayY, float penumbra) {
    if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) {
        return 0.0f;
    }
    int topY = sampleTopOpaqueWorldY(vGrid, gx, gz);
    if (topY <= -9000) {
        return 0.0f;
    }
    return smoothstep(0.15f, penumbra, float(topY) - rayY);
}

float evalWaterSurfaceSunShadowFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 surfWorld, float3 keyDir, float tWater) {
    if (keyDir.y <= 0.005f) {
        return 0.0f;
    }
    float2 baseGrid = surfWorld.xz - float2(float(vOrg.x), float(vOrg.z));
    if (baseGrid.x < 1.0f || baseGrid.x >= 510.0f || baseGrid.y < 1.0f || baseGrid.y >= 510.0f) {
        return 1.0f;
    }
    float occ = 0.0f;
    if (tWater > 72.0f) {
        float t = 2.0f;
        for (int i = 0; i < 4; ++i) {
            if (occ >= 0.995f) break;
            float3 p = surfWorld + keyDir * t;
            int gx = int(floor(p.x)) - vOrg.x;
            int gz = int(floor(p.z)) - vOrg.z;
            if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) break;
            occ = max(occ, evalWaterSunOccTap(vGrid, gx, gz, p.y, 0.75f + t * 0.045f));
            t = t * 2.15f + 1.5f;
        }
        return 1.0f - clamp(occ, 0.0f, 1.0f);
    }
    float t = 0.9f;
    bool nearBilinear = (tWater < 28.0f);
    for (int i = 0; i < 6; ++i) {
        if (occ >= 0.995f) break;
        float3 p = surfWorld + keyDir * t;
        float penumbra = 0.75f + t * 0.045f;
        if (nearBilinear && i < 2) {
            float2 gp = p.xz - float2(float(vOrg.x), float(vOrg.z)) - float2(0.5f, 0.5f);
            int2 g0 = int2(floor(gp));
            if (g0.x < 0 || g0.x >= 511 || g0.y < 0 || g0.y >= 511) break;
            float2 fr = gp - float2(g0);
            float o00 = evalWaterSunOccTap(vGrid, g0.x,     g0.y,     p.y, penumbra);
            float o10 = evalWaterSunOccTap(vGrid, g0.x + 1, g0.y,     p.y, penumbra);
            float o01 = evalWaterSunOccTap(vGrid, g0.x,     g0.y + 1, p.y, penumbra);
            float o11 = evalWaterSunOccTap(vGrid, g0.x + 1, g0.y + 1, p.y, penumbra);
            occ = max(occ, lerp(lerp(o00, o10, fr.x), lerp(o01, o11, fr.x), fr.y));
        } else {
            int gx = int(floor(p.x)) - vOrg.x;
            int gz = int(floor(p.z)) - vOrg.z;
            if (gx < 0 || gx >= 512 || gz < 0 || gz >= 512) break;
            occ = max(occ, evalWaterSunOccTap(vGrid, gx, gz, p.y, penumbra));
        }
        t = t * 1.80f + 0.65f;
    }
    return 1.0f - clamp(occ, 0.0f, 1.0f);
}

float evalWaterSurfaceSunShadow(StructuredBuffer<uint> vGrid, int3 vOrg, float3 surfWorld, float3 keyDir) {
    return evalWaterSurfaceSunShadowFast(vGrid, vOrg, surfWorld, keyDir, 8.0f);
}

float4 sampleHalfResWaterRefl(
    Texture2D<float4> halfTex,
    Texture2D<float4> fullDepthTex,
    MetalityFrame f,
    float2 uv,
    float tWater
) {
    int2 hDim = max(metality_tex_size(halfTex), int2(1, 1));
    int2 dDim = max(metality_tex_size(fullDepthTex), int2(1, 1));
    float2 fCoord = uv * float2(hDim) - 0.5f;
    int2 i0 = int2(floor(fCoord));
    float2 fr = fCoord - float2(i0);

    float4 clipRay = float4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, 0.5f, 1.0f);
    float4 vRayH = mul(f.invProj, clipRay);
    float vRayScale = length(vRayH.xyz) / max(abs(vRayH.z), 1e-6f);

    float4 accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wSum = 0.0f;
    for (int oy = 0; oy <= 1; ++oy) {
        float wy = (oy == 0) ? (1.0f - fr.y) : fr.y;
        for (int ox = 0; ox <= 1; ++ox) {
            float wx = (ox == 0) ? (1.0f - fr.x) : fr.x;
            int2 hp = clamp(i0 + int2(ox, oy), int2(0, 0), hDim - int2(1, 1));
            float4 val = halfTex.Load(int3(hp, 0));
            if (val.a < -0.5f) continue;

            float2 hUv = texelCenterUv(hp, hDim);
            int2 dPx = snapDepthCoord(dDim, hUv);
            float sd = fullDepthTex.Load(int3(dPx, 0)).r;
            bool behindWater = isSkyDepth(sd);
            if (!behindWater) {
                float sdz = max(sd, 1e-5f);
                float sw = f.invProj[3][2] * sdz + f.invProj[3][3];
                float sViewZ = abs((f.invProj[2][2] * sdz + f.invProj[2][3]) / (abs(sw) > 1e-6f ? sw : 1e-6f));
                behindWater = (sViewZ * vRayScale > tWater - 0.15f);
            }
            if (behindWater) {
                float w = max(wx * wy, 0.02f);
                accum += val * w;
                wSum += w;
            }
        }
    }
    if (wSum > 1e-4f) {
        return accum / wSum;
    }
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

bool isOpaqueShadowVoxelCommon(uint tag) {
    return (tag >= 1u && tag <= 7u && tag != 3u) || (tag == 12u);
}

uint sampleVoxelTag(StructuredBuffer<uint> vGrid, int3 c) {
    if (!inVoxelBounds512Common(c)) {
        return 0u;
    }
    uint idx = voxelLinearIndex512Common(c);
    uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    return (b >> 4u) & 0xFu;
}

bool pointInLocalAabbPad(float3 p, float3 bMin, float3 bMax, float pad) {
    return (p.x >= bMin.x - pad && p.x <= bMax.x + pad &&
            p.y >= bMin.y - pad && p.y <= bMax.y + pad &&
            p.z >= bMin.z - pad && p.z <= bMax.z + pad);
}

bool pointInSubBlockVoxel(uint tag, uint sub, float3 localP) {
    if (tag < 14u) {
        return false;
    }
    float pad = 0.006f;
    if (tag == 14u) {
        if (sub == 0u || sub >= 14u) {
            return false;
        }
        bool isInset = (sub >= 12u);
        uint maxYPack = (sub < 8u) ? 0x7FFF2FFFu : 0xFFFD5BEFu;
        float3 bMin = float3(
            (sub == 2u) ? 0.8125f : (isInset ? 0.0625f : 0.0f),
            (sub == 4u) ? 0.8125f : ((sub == 8u) ? 0.5f : 0.0f),
            (sub == 6u) ? 0.8125f : (isInset ? 0.0625f : 0.0f)
        );
        float3 bMax = float3(
            (sub == 1u) ? 0.1875f : (isInset ? 0.9375f : 1.0f),
            float(((maxYPack >> ((sub & 7u) * 4u)) & 0xFu) + 1u) * 0.0625f,
            (sub == 5u) ? 0.1875f : (isInset ? 0.9375f : 1.0f)
        );
        return pointInLocalAabbPad(localP, bMin, bMax, pad);
    }
    if (tag == 15u) {
        bool topBase = ((sub & 4u) != 0u);
        float3 aMin = float3(0.0f, topBase ? 0.5f : 0.0f, 0.0f);
        float3 aMax = float3(1.0f, topBase ? 1.0f : 0.5f, 1.0f);
        if (pointInLocalAabbPad(localP, aMin, aMax, pad)) {
            return true;
        }
        uint dir = sub & 3u;
        bool isOuter = ((sub & 8u) != 0u);
        float3 sMin = float3(
            isOuter ? (((dir & 1u) != 0u) ? 0.5f : 0.0f) : ((dir == 1u) ? 0.5f : 0.0f),
            topBase ? 0.0f : 0.5f,
            isOuter ? (((dir & 2u) != 0u) ? 0.5f : 0.0f) : ((dir == 3u) ? 0.5f : 0.0f)
        );
        float3 sMax = float3(
            isOuter ? (((dir & 1u) != 0u) ? 1.0f : 0.5f) : ((dir == 0u) ? 0.5f : 1.0f),
            topBase ? 0.5f : 1.0f,
            isOuter ? (((dir & 2u) != 0u) ? 1.0f : 0.5f) : ((dir == 2u) ? 0.5f : 1.0f)
        );
        return pointInLocalAabbPad(localP, sMin, sMax, pad);
    }
    return false;
}

bool isPointBlockedInVoxel(StructuredBuffer<uint> vGrid, int3 vOrg, float3 worldPos) {
    float3 fl = floor(worldPos);
    int3 c = int3(fl) - vOrg;
    if (!inVoxelBounds512Common(c)) {
        return false;
    }
    uint idx = voxelLinearIndex512Common(c);
    uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
    if (b == 0u) {
        return false;
    }
    uint tag = (b >> 4u) & 0xFu;
    if (isOpaqueShadowVoxelCommon(tag)) {
        return true;
    }
    if (tag >= 14u) {
        uint sub = b & 0xFu;
        return pointInSubBlockVoxel(tag, sub, worldPos - fl);
    }
    return false;
}

bool rayHitsLocalAabb(float3 o, float3 invDir, float3 bMin, float3 bMax, float maxT, bool isStartCell) {
    float3 t0 = (bMin - o) * invDir;
    float3 t1 = (bMax - o) * invDir;
    float3 tLo = min(t0, t1);
    float3 tHi = max(t0, t1);
    float tNear = max(tLo.x, max(tLo.y, tLo.z));
    float tFar = min(tHi.x, min(tHi.y, tHi.z));
    if (isStartCell) {
        return (tNear >= 0.001f) && (tFar >= tNear + 0.012f) && (tNear < maxT);
    }
    return (tFar >= max(tNear, 0.002f)) && (tNear < maxT);
}

bool intersectSubBlockVoxel(uint tag, uint sub, int3 cell, float3 pos, float3 rayDir, float occStopDist, bool isStartCell) {
    if (tag < 14u) {
        return false;
    }
    if (tag == 14u && (sub == 0u || sub >= 14u)) {
        return false;
    }
    float3 o = pos - float3(cell);
    float3 invDir = float3(
        abs(rayDir.x) > 1e-5f ? 1.0f / rayDir.x : 1e5f,
        abs(rayDir.y) > 1e-5f ? 1.0f / rayDir.y : 1e5f,
        abs(rayDir.z) > 1e-5f ? 1.0f / rayDir.z : 1e5f
    );
    if (tag == 14u) {
        bool isInset = (sub >= 12u);
        uint maxYPack = (sub < 8u) ? 0x7FFF2FFFu : 0xFFFD5BEFu;
        float3 bMin = float3(
            (sub == 2u) ? 0.8125f : (isInset ? 0.0625f : 0.0f),
            (sub == 4u) ? 0.8125f : ((sub == 8u) ? 0.5f : 0.0f),
            (sub == 6u) ? 0.8125f : (isInset ? 0.0625f : 0.0f)
        );
        float3 bMax = float3(
            (sub == 1u) ? 0.1875f : (isInset ? 0.9375f : 1.0f),
            float(((maxYPack >> ((sub & 7u) * 4u)) & 0xFu) + 1u) * 0.0625f,
            (sub == 5u) ? 0.1875f : (isInset ? 0.9375f : 1.0f)
        );
        return rayHitsLocalAabb(o, invDir, bMin, bMax, occStopDist, isStartCell);
    }
    if (tag == 15u) {
        bool topBase = ((sub & 4u) != 0u);
        float3 aMin = float3(0.0f, topBase ? 0.5f : 0.0f, 0.0f);
        float3 aMax = float3(1.0f, topBase ? 1.0f : 0.5f, 1.0f);
        if (rayHitsLocalAabb(o, invDir, aMin, aMax, occStopDist, isStartCell)) {
            return true;
        }
        uint dir = sub & 3u;
        bool isOuter = ((sub & 8u) != 0u);
        float3 sMin = float3(
            isOuter ? (((dir & 1u) != 0u) ? 0.5f : 0.0f) : ((dir == 1u) ? 0.5f : 0.0f),
            topBase ? 0.0f : 0.5f,
            isOuter ? (((dir & 2u) != 0u) ? 0.5f : 0.0f) : ((dir == 3u) ? 0.5f : 0.0f)
        );
        float3 sMax = float3(
            isOuter ? (((dir & 1u) != 0u) ? 1.0f : 0.5f) : ((dir == 0u) ? 0.5f : 1.0f),
            topBase ? 0.5f : 1.0f,
            isOuter ? (((dir & 2u) != 0u) ? 1.0f : 0.5f) : ((dir == 2u) ? 0.5f : 1.0f)
        );
        return rayHitsLocalAabb(o, invDir, sMin, sMax, occStopDist, isStartCell);
    }
    return false;
}

bool intersectSubBlockVoxel(uint tag, uint sub, int3 cell, float3 pos, float3 rayDir, float occStopDist) {
    return intersectSubBlockVoxel(tag, sub, cell, pos, rayDir, occStopDist, false);
}

float2 evalSkyZenithPairTap(StructuredBuffer<uint> vGrid, int gx, int gz, float sampleY, float surfNormalY, int selfBlockTop) {
    int2 top0 = sampleTopSolidAndOpaqueWorldY(vGrid, clamp(gx, 0, 511), clamp(gz, 0, 511));
    bool selfTopFace = (surfNormalY > 0.3f) && (top0.x == selfBlockTop);
    float solidDiff = (!selfTopFace && top0.x > -9000) ? max(float(top0.x) - sampleY, 0.0f) : 0.0f;
    float opaqueDiff = (top0.y > -9000) ? max(float(top0.y) - (selfTopFace ? float(selfBlockTop) : sampleY), 0.0f) : 0.0f;
    return float2(1.0f - smoothstep(0.05f, 0.72f, solidDiff),
                  1.0f - smoothstep(0.05f, 1.15f, opaqueDiff));
}

float2 evalSkySlopeSinTap(StructuredBuffer<uint> vGrid, int gx, int gz, float sampleY, float invR) {
    int2 ty = sampleTopSolidAndOpaqueWorldY(vGrid, clamp(gx, 0, 511), clamp(gz, 0, 511));
    float sSolid = (ty.x > -9000) ? max((float(ty.x) - sampleY) * invR, 0.0f) : 0.0f;
    float sCanopy = (ty.y > -9000) ? max((float(ty.y) - sampleY) * invR, 0.0f) : 0.0f;
    return float2(
        sSolid * rsqrt(1.0f + sSolid * sSolid),
        sCanopy * rsqrt(1.0f + sCanopy * sCanopy)
    );
}

// 3-tap C0-continuous barycentric heightmap horizon slope sample (zero step discontinuities, zero random scatter, L1/L2 coherent)
float2 sampleBarycentricSkySlopeSin(StructuredBuffer<uint> vGrid, float2 posXZ, float sampleY, float invR) {
    float2 gp = posXZ - float2(0.5f, 0.5f);
    int2 g0 = int2(floor(gp));
    float2 fr = gp - float2(g0);
    float2 v00 = evalSkySlopeSinTap(vGrid, g0.x,     g0.y,     sampleY, invR);
    float2 v11 = evalSkySlopeSinTap(vGrid, g0.x + 1, g0.y + 1, sampleY, invR);
    if (fr.x >= fr.y) {
        float2 v10 = evalSkySlopeSinTap(vGrid, g0.x + 1, g0.y, sampleY, invR);
        return v00 * (1.0f - fr.x) + v10 * (fr.x - fr.y) + v11 * fr.y;
    }
    float2 v01 = evalSkySlopeSinTap(vGrid, g0.x, g0.y + 1, sampleY, invR);
    return v00 * (1.0f - fr.y) + v01 * (fr.y - fr.x) + v11 * fr.x;
}

float3 evalSkyHorizonDirAccum(StructuredBuffer<uint> vGrid, float3 gridP, float sampleY, float3 surfNormal, float2 dir2, float r0, float invR0, float r1, float invR1) {
    float2 sin0 = sampleBarycentricSkySlopeSin(vGrid, gridP.xz + dir2 * r0, sampleY, invR0);
    float2 sin1 = sampleBarycentricSkySlopeSin(vGrid, gridP.xz + dir2 * r1, sampleY, invR1);
    float2 sinEff = max(sin0, sin1 * 0.85f);
    float align = dot(float3(dir2.x, 0.35f, dir2.y), surfNormal);
    float w = lerp(0.08f, 1.0f, smoothstep(-0.35f, 0.25f, align));
    return float3(saturate(1.0f - sinEff.x) * w, saturate(1.0f - sinEff.y) * w, w);
}

// Compact 2D-XZ 3D-voxel sky-escape DDA for under-roof surfaces (0 stack arrays, lightweight sub-block check):
// blocks on opaque walls/closed doors/stairs, attenuates through glass windows, exits upon reaching open sky.
float evalUnderRoofDirSkyVisDda(
    StructuredBuffer<uint> vGrid,
    float3 pos,
    int3 startCell,
    float sampleY,
    float fracY,
    float2 dir2,
    float maxDist
) {
    int3 cell = startCell;
    float2 invDir = float2(
        abs(dir2.x) > 1e-5f ? (1.0f / dir2.x) : 1e5f,
        abs(dir2.y) > 1e-5f ? (1.0f / dir2.y) : 1e5f
    );
    int2 stepV = int2(dir2.x >= 0.0f ? 1 : -1, dir2.y >= 0.0f ? 1 : -1);
    float2 tMax = float2(
        ((float(cell.x) + (stepV.x > 0 ? 1.0f : 0.0f)) - pos.x) * invDir.x,
        ((float(cell.z) + (stepV.y > 0 ? 1.0f : 0.0f)) - pos.z) * invDir.y
    );
    float2 tDelta = abs(invDir);
    float dist = 0.0f;
    float trans = 1.0f;

    for (int iter = 0; iter < 10; ++iter) {
        if (cell.x < 0 || cell.x >= 512 || cell.z < 0 || cell.z >= 512) {
            break;
        }
        uint idx = voxelLinearIndex512Common(cell);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        if (b != 0u) {
            uint tag = (b >> 4u) & 0xFu;
            uint sub = b & 0xFu;
            if (tag == 8u) {
                trans *= (sub >= 2u) ? 0.15f : 0.35f;
            } else if (tag == 14u) {
                if (sub >= 1u && sub <= 13u) {
                    bool otherCell = any(cell != startCell);
                    bool hitDoor = (sub == 1u && (otherCell || dir2.x < 0.0f)) ||
                                   (sub == 2u && (otherCell || dir2.x > 0.0f)) ||
                                   (sub == 5u && (otherCell || dir2.y < 0.0f)) ||
                                   (sub == 6u && (otherCell || dir2.y > 0.0f));
                    bool hitSlab = otherCell && ((sub == 7u && fracY < 0.5f) || (sub == 8u && fracY >= 0.5f));
                    if (hitDoor || hitSlab) {
                        return 0.0f;
                    }
                    if (otherCell) {
                        trans *= 0.45f;
                    }
                }
            } else if ((tag == 15u || isOpaqueShadowVoxelCommon(tag)) && any(cell != startCell)) {
                return 0.0f;
            }
        }
        if (iter > 0) {
            int topSolidY = sampleTopSolidWorldY(vGrid, cell.x, cell.z);
            if (topSolidY <= -9000 || float(topSolidY) <= sampleY + 0.35f) {
                return trans * (1.0f - smoothstep(0.35f, maxDist, dist));
            }
        }
        if (tMax.x < tMax.y) {
            dist = tMax.x;
            tMax.x += tDelta.x;
            cell.x += stepV.x;
        } else {
            dist = tMax.y;
            tMax.y += tDelta.y;
            cell.z += stepV.y;
        }
        if (dist >= maxDist) {
            break;
        }
    }
    return 0.0f;
}

float evalSmoothSkyOpennessFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 sampleWorld, float3 surfNormal, float2 ign, bool allowUnderRoofDda) {
    float3 gridP = sampleWorld - float3(vOrg);
    int gxBase = int(floor(gridP.x));
    int gzBase = int(floor(gridP.z));
    if (gxBase < 0 || gxBase >= 512 || gzBase < 0 || gzBase >= 512) {
        return saturate(0.65f + 0.35f * surfNormal.y);
    }

    // Bilinearly filter zenith solid & canopy visibility so overhanging tree leaves never snap on 1x1 integer block seams
    bool isVert = (abs(surfNormal.y) < 0.35f);
    float3 zenithP = isVert ? (gridP + surfNormal * 0.65f) : gridP;
    float3 horizonP = isVert ? (gridP + surfNormal * 0.55f) : gridP;
    float sampleY = sampleWorld.y + 0.10f;
    int selfBlockTop = int(floor(sampleWorld.y)) + 1;
    float2 zCoord = zenithP.xz - float2(0.5f, 0.5f);
    int2 z0 = int2(floor(zCoord));
    float2 zFr = zCoord - float2(z0);
    float2 z00 = evalSkyZenithPairTap(vGrid, z0.x,     z0.y,     sampleY, surfNormal.y, selfBlockTop);
    float2 z10 = evalSkyZenithPairTap(vGrid, z0.x + 1, z0.y,     sampleY, surfNormal.y, selfBlockTop);
    float2 z01 = evalSkyZenithPairTap(vGrid, z0.x,     z0.y + 1, sampleY, surfNormal.y, selfBlockTop);
    float2 z11 = evalSkyZenithPairTap(vGrid, z0.x + 1, z0.y + 1, sampleY, surfNormal.y, selfBlockTop);
    float2 zVis = lerp(lerp(z00, z10, zFr.x), lerp(z01, z11, zFr.x), zFr.y);
    float zenithSolidVis = zVis.x * smoothstep(-0.60f, -0.15f, surfNormal.y);
    float zenithCanopyVis = zVis.y;

    // Deterministic 4-direction (22.5-deg symmetric) x 2-ring C0-barycentric horizon scan:
    // Zero per-pixel IGN noise (eliminates dark-area blotches), zero 1-block cookie-cutter steps, zero trig/divides, 100% SIMD-coherent.
    float2 d0 = float2( 0.9238795f,  0.3826834f);
    float2 d1 = float2(-0.3826834f,  0.9238795f);
    float r0 = 2.2f;
    float invR0 = 1.0f / 2.2f;
    float r1 = 6.2f;
    float invR1 = 1.0f / 6.2f;
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    acc += evalSkyHorizonDirAccum(vGrid, horizonP, sampleY, surfNormal,  d0, r0, invR0, r1, invR1);
    acc += evalSkyHorizonDirAccum(vGrid, horizonP, sampleY, surfNormal,  d1, r0, invR0, r1, invR1);
    acc += evalSkyHorizonDirAccum(vGrid, horizonP, sampleY, surfNormal, -d0, r0, invR0, r1, invR1);
    acc += evalSkyHorizonDirAccum(vGrid, horizonP, sampleY, surfNormal, -d1, r0, invR0, r1, invR1);

    float invW = 1.0f / max(acc.z, 1e-4f);
    float solidHorizon = acc.x * invW;
    float canopyHorizon = acc.y * invW;
    float ceilAtten = lerp(0.50f, 1.0f, smoothstep(-0.75f, -0.25f, surfNormal.y));

    // Under a solid roof (zenithSolidVis < 0.98f): verify 3D horizontal voxel enclosure so
    // 2D heightmap probes never leak outdoor sky through walls or 1-block roofs into sealed rooms.
    // Open outdoor surfaces & spruce canopies (zenithSolidVis >= 0.98f), deep caves (solidHorizon < 0.06f),
    // and night callers (!allowUnderRoofDda) skip the 3D voxel DDA completely (zero extra reads!).
    float solidOpen;
    if (zenithSolidVis >= 0.98f) {
        solidOpen = clamp(solidHorizon * ceilAtten, 0.0f, 1.0f);
    } else {
        float underRoofSolidOpen = 0.0f;
        if (solidHorizon >= 0.06f) {
            int3 startCell = int3(floor(gridP));
            if (allowUnderRoofDda && inVoxelBounds512Common(startCell)) {
                // 4 symmetric 90-deg rotated directions (22.5 + k*90 deg, 0 stack arrays, 0 trig/muls, low L2 footprint)
                float2 dir2 = float2(0.9238795f, 0.3826834f);
                float2 uAcc = float2(0.0f, 0.0f);
                float fracY = gridP.y - floor(gridP.y);
                for (uint k = 0u; k < 4u; ++k) {
                    float w = isVert ? smoothstep(-0.15f, 0.35f, dot(dir2, surfNormal.xz)) : 1.0f;
                    if (w > 0.005f) {
                        float v = evalUnderRoofDirSkyVisDda(vGrid, gridP, startCell, sampleY, fracY, dir2, 6.5f);
                        uAcc += float2(v * w, w);
                    }
                    dir2 = float2(-dir2.y, dir2.x);
                }
                float underRoofHorizon = uAcc.x / max(uAcc.y, 1e-4f);
                underRoofSolidOpen = clamp(underRoofHorizon * 0.72f * ceilAtten, 0.0f, solidHorizon * ceilAtten);
            } else {
                float zFallback = lerp(0.10f, 0.35f, smoothstep(0.10f, 0.52f, solidHorizon));
                underRoofSolidOpen = clamp(solidHorizon * zFallback * ceilAtten, 0.0f, 1.0f);
            }
        }
        solidOpen = clamp(lerp(underRoofSolidOpen, solidHorizon * ceilAtten, zenithSolidVis), 0.0f, 1.0f);
    }

    float caveGate = smoothstep(0.08f, 0.24f, solidHorizon);
    float zFloorBase = lerp(0.10f, 0.72f, smoothstep(0.10f, 0.52f, solidHorizon));
    float zenithFloor = isVert ? max(zFloorBase, 0.68f * caveGate) : zFloorBase;
    float canopyOpen = clamp(canopyHorizon * lerp(zenithFloor, 1.0f, zenithCanopyVis) * ceilAtten, 0.0f, 1.0f);
    canopyOpen = min(canopyOpen, solidOpen);
    return clamp(lerp(solidOpen, canopyOpen, 0.52f), 0.0f, 1.0f);
}

float evalSmoothSkyOpennessFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 sampleWorld, float3 surfNormal, float2 ign) {
    return evalSmoothSkyOpennessFast(vGrid, vOrg, sampleWorld, surfNormal, ign, true);
}

float evalSmoothSkyOpennessFast(StructuredBuffer<uint> vGrid, int3 vOrg, float3 sampleWorld, float3 surfNormal) {
    return evalSmoothSkyOpennessFast(vGrid, vOrg, sampleWorld, surfNormal, float2(0.0f, 0.0f), true);
}

float3 safeShadowOriginOffset(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 worldPos,
    float3 surfNormal,
    float3 domAxis,
    float3 viewDir,
    float camDist
) {
    float nPush = clamp(0.024f + camDist * 0.0008f, 0.024f, 0.052f);
    float vPush = clamp(0.016f + camDist * 0.0004f, 0.016f, 0.030f);
    float3 c0 = domAxis * nPush + viewDir * vPush;
    if (!isPointBlockedInVoxel(vGrid, vOrg, worldPos + c0)) {
        return c0;
    }
    float3 c1 = domAxis * nPush;
    if (!isPointBlockedInVoxel(vGrid, vOrg, worldPos + c1)) {
        return c1;
    }
    if (dot(viewDir, viewDir) > 0.25f) {
        float3 c2 = viewDir * max(0.045f, nPush * 1.4f);
        if (!isPointBlockedInVoxel(vGrid, vOrg, worldPos + c2)) {
            return c2;
        }
    }
    return surfNormal * max(0.015f, nPush * 0.65f);
}

float3 safeShadowOriginOffset(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 worldPos,
    float3 surfNormal,
    float3 domAxis,
    float3 viewDir
) {
    return safeShadowOriginOffset(vGrid, vOrg, worldPos, surfNormal, domAxis, viewDir, 0.0f);
}

float3 safeShadowOriginOffset(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 worldPos,
    float3 surfNormal,
    float3 domAxis
) {
    return safeShadowOriginOffset(vGrid, vOrg, worldPos, surfNormal, domAxis, float3(0.0f, 0.0f, 0.0f), 0.0f);
}

float3 traceStainedGlassFilterDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 startWorld,
    float3 rayDir,
    float maxDist,
    int maxSteps,
    bool inheritFramePenumbra
) {
    float3 pos = startWorld - float3(vOrg);
    int3 cell = int3(floor(pos));
    int3 startCell = cell;
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
    float3 tMax = float3(
        ((float(cell.x) + (stepV.x > 0 ? 1.0f : 0.0f)) - pos.x) * invDir.x,
        ((float(cell.y) + (stepV.y > 0 ? 1.0f : 0.0f)) - pos.y) * invDir.y,
        ((float(cell.z) + (stepV.z > 0 ? 1.0f : 0.0f)) - pos.z) * invDir.z
    );
    float3 tDelta = abs(invDir);

    float3 optDepth = float3(0.0f, 0.0f, 0.0f);
    bool hitGlass = false;
    bool hitOpaque = false;
    float hitDist = 0.0f;
    int hitAxis = 2;
    float maxAxisDir = max(abs(rayDir.x), max(abs(rayDir.y), abs(rayDir.z)));
    float dist = 0.0f;
    for (int iter = 0; iter < 56; ++iter) {
        if (iter >= maxSteps || !inVoxelBounds512Common(cell)) {
            break;
        }
        float nextDist = min(tMax.x, min(tMax.y, tMax.z));
        uint idx = voxelLinearIndex512Common(cell);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        uint tag = (b >> 4u) & 0xFu;
        uint sub = b & 0xFu;
        if (tag == 8u && sub >= 2u) {
            float cellDt = max(min(nextDist, maxDist) - dist, 0.0f);
            float effThick = cellDt * maxAxisDir;
            optDepth += -log(max(decodeStainedGlassColor(sub), float3(0.05f, 0.05f, 0.05f))) * effThick;
            hitGlass = true;
        } else if (tag >= 14u) {
            if (intersectSubBlockVoxel(tag, sub, cell, pos, rayDir, maxDist, all(cell == startCell))) {
                hitOpaque = true;
                hitDist = dist;
                break;
            }
        } else if (isOpaqueShadowVoxelCommon(tag) && dist > 0.22f && dist < maxDist - 0.65f && any(cell != startCell)) {
            hitOpaque = true;
            hitDist = dist;
            break;
        }
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                dist = tMax.x;
                tMax.x += tDelta.x;
                cell.x += stepV.x;
                hitAxis = 0;
            } else {
                dist = tMax.z;
                tMax.z += tDelta.z;
                cell.z += stepV.z;
                hitAxis = 2;
            }
        } else {
            if (tMax.y < tMax.z) {
                dist = tMax.y;
                tMax.y += tDelta.y;
                cell.y += stepV.y;
                hitAxis = 1;
            } else {
                dist = tMax.z;
                tMax.z += tDelta.z;
                cell.z += stepV.z;
                hitAxis = 2;
            }
        }
        if (dist >= maxDist) {
            break;
        }
    }
    if (hitGlass) {
        return exp(-optDepth);
    }
    if (inheritFramePenumbra && hitOpaque) {
        // Probe 4 in-plane neighbors of the hit wall frame (zero stack arrays!)
        float3 hitWorld = startWorld + rayDir * (hitDist + 0.05f);
        float3 aU = (hitAxis == 0) ? float3(0.0f, 0.64f, 0.0f) : float3(0.64f, 0.0f, 0.0f);
        float3 aV = (hitAxis == 2) ? float3(0.0f, 0.64f, 0.0f) : float3(0.0f, 0.0f, 0.64f);
        float3 glassAccum = float3(0.0f, 0.0f, 0.0f);
        float glassCnt = 0.0f;
        float clearAirCnt = 0.0f;
        for (int pi = 0; pi < 4; ++pi) {
            float3 probeW = (pi == 0) ? (hitWorld + aU) : ((pi == 1) ? (hitWorld - aU) : ((pi == 2) ? (hitWorld + aV) : (hitWorld - aV)));
            int3 pc = int3(floor(probeW)) - vOrg;
            if (!inVoxelBounds512Common(pc)) continue;
            uint pidx = voxelLinearIndex512Common(pc);
            uint pb = (vGrid[pidx >> 2u] >> ((pidx & 3u) * 8u)) & 0xFFu;
            uint pTag = (pb >> 4u) & 0xFu;
            uint pSub = pb & 0xFu;
            if (pTag == 8u && pSub >= 2u) {
                glassAccum += decodeStainedGlassColor(pSub);
                glassCnt += 1.0f;
            } else if (pTag == 0u || pTag == 8u) {
                clearAirCnt += 1.0f;
            }
        }
        if (glassCnt < 0.5f && clearAirCnt < 0.5f) {
            for (int di = 0; di < 4; ++di) {
                float su = ((di & 1) == 0) ? 1.0f : -1.0f;
                float sv = ((di & 2) == 0) ? 1.0f : -1.0f;
                int3 pc = int3(floor(hitWorld + aU * su + aV * sv)) - vOrg;
                if (!inVoxelBounds512Common(pc)) continue;
                uint pidx = voxelLinearIndex512Common(pc);
                uint pb = (vGrid[pidx >> 2u] >> ((pidx & 3u) * 8u)) & 0xFFu;
                uint pTag = (pb >> 4u) & 0xFu;
                uint pSub = pb & 0xFu;
                if (pTag == 8u && pSub >= 2u) {
                    glassAccum += decodeStainedGlassColor(pSub);
                    glassCnt += 1.0f;
                } else if (pTag == 0u || pTag == 8u) {
                    clearAirCnt += 1.0f;
                }
            }
        }
        if (glassCnt > 0.5f) {
            return (glassAccum + float3(clearAirCnt, clearAirCnt, clearAirCnt)) / (glassCnt + clearAirCnt);
        }
    }
    return float3(1.0f, 1.0f, 1.0f);
}

float3 traceStainedGlassFilterDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 startWorld,
    float3 rayDir,
    float maxDist,
    int maxSteps
) {
    return traceStainedGlassFilterDda(vGrid, vOrg, startWorld, rayDir, maxDist, maxSteps, false);
}

float evalPointLightMaxRange(float power) {
    return clamp(sqrt(max(power, 0.5f)) * 3.6f + 2.5f, 6.0f, 34.0f);
}

float evalPointLightWindow(float dist, float maxR) {
    float r = saturate(dist / max(maxR, 1e-4f));
    float w = 1.0f - r * r;
    return w * w;
}

float evalPointLightGeomTerm(float3 surfNormal, float3 lDir, float dist, uint voxelTag, uint voxelSub, bool isWater) {
    if (voxelTag == 14u && voxelSub == 15u) {
        return max(abs(dot(surfNormal, lDir)) * 0.65f, 0.30f);
    }
    if (voxelTag == 8u) {
        return 0.45f + 0.55f * abs(dot(surfNormal, lDir));
    }
    if (isWater) {
        return max(dot(surfNormal, lDir), 0.12f);
    }
    float nd = dot(surfNormal, lDir);
    float front = max(nd, 0.0f);
    float nearBase = (1.0f - smoothstep(0.85f, 1.55f, dist)) * 0.12f * smoothstep(-0.02f, 0.20f, nd);
    return saturate(front + nearBase);
}

float evalPointLightGeomTerm(float3 surfNormal, float3 lDir, float dist, uint voxelTag, bool isWater) {
    return evalPointLightGeomTerm(surfNormal, lDir, dist, voxelTag, 0u, isWater);
}

uint getTileLightOffset(StructuredBuffer<uint> tileLightGrid, float2 uv) {
    uint tilesX = tileLightGrid[0];
    uint tilesY = tileLightGrid[1];
    if (tilesX == 0u || tilesY == 0u || tileLightGrid[3] == 0u) {
        return 0u;
    }
    uint packedDim = tileLightGrid[2];
    uint tx;
    uint ty;
    if (packedDim > 5u) {
        uint scrW = max(packedDim & 0xFFFFu, 1u);
        uint scrH = max(packedDim >> 16u, 1u);
        tx = min(uint(saturate(uv.x) * float(scrW)) >> 5u, tilesX - 1u);
        ty = min(uint(saturate(uv.y) * float(scrH)) >> 5u, tilesY - 1u);
    } else {
        tx = min(uint(saturate(uv.x) * float(tilesX)), tilesX - 1u);
        ty = min(uint(saturate(uv.y) * float(tilesY)), tilesY - 1u);
    }
    return 4u + (ty * tilesX + tx) * 64u;
}

float3 tracePointLightVoxelVisibilityAndGlassDda(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 startWorld,
    float3 rayDir,
    float totalDist,
    bool hasStainedGlass,
    int maxSteps
) {
    float3 pos = startWorld - float3(vOrg);
    int3 cell = int3(floor(pos));
    int3 startCell = cell;
    int3 lightCell = int3(floor(startWorld + rayDir * totalDist - float3(vOrg)));
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
    float3 tMax = float3(
        ((float(cell.x) + (stepV.x > 0 ? 1.0f : 0.0f)) - pos.x) * invDir.x,
        ((float(cell.y) + (stepV.y > 0 ? 1.0f : 0.0f)) - pos.y) * invDir.y,
        ((float(cell.z) + (stepV.z > 0 ? 1.0f : 0.0f)) - pos.z) * invDir.z
    );
    float3 tDelta = abs(invDir);

    float3 optDepth = float3(0.0f, 0.0f, 0.0f);
    bool hitGlass = false;
    float maxAxisDir = max(abs(rayDir.x), max(abs(rayDir.y), abs(rayDir.z)));
    float dist = 0.0f;
    float stopDist = max(totalDist + 0.08f, 0.10f);
    float occStopDist = max(totalDist - 0.10f, 0.0f);

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
                float effThick = cellDt * maxAxisDir;
                optDepth += -log(max(decodeStainedGlassColor(sub), float3(0.05f, 0.05f, 0.05f))) * effThick;
                hitGlass = true;
            } else if (tag >= 14u) {
                if (any(cell != lightCell) &&
                    intersectSubBlockVoxel(tag, sub, cell, pos, rayDir, occStopDist, all(cell == startCell))) {
                    return float3(0.0f, 0.0f, 0.0f);
                }
            } else if (isOpaqueShadowVoxelCommon(tag) &&
                       dist < occStopDist &&
                       any(cell != startCell) && any(cell != lightCell)) {
                int3 dLight = abs(cell - lightCell);
                bool sameLavaEmitter = (tag == 4u && sub >= 10u && dLight.x <= 4 && dLight.z <= 4 && dLight.y <= 10);
                if (!sameLavaEmitter) {
                    return float3(0.0f, 0.0f, 0.0f);
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
    return hitGlass ? exp(-optDepth) : float3(1.0f, 1.0f, 1.0f);
}

float evalEntityShadowSoftVis(
    StructuredBuffer<float4> casters,
    float3 surfRel,
    float3 rayOrigin,
    float3 rayDir,
    float3 lightRelPos,
    float maxDist,
    float lightRadius
) {
    uint cnt = min(uint(casters[0].x + 0.5f), 32u);
    if (cnt == 0u || maxDist <= 0.02f) {
        return 1.0f;
    }
    float vis = 1.0f;
    for (uint ci = 0u; ci < cnt; ++ci) {
        float4 c0 = casters[1u + 2u * ci];
        float4 c1 = casters[2u + 2u * ci];
        float3 ext = float3(c0.w, c1.x, c1.y);

        // Ultra-fast ray-sphere coarse reject padded by max penumbra before computing trig
        float3 toC = c0.xyz - rayOrigin;
        float proj = dot(toC, rayDir);
        float rBound = ext.x + ext.y + ext.z + 0.42f;
        if (proj < -rBound || proj > maxDist + rBound) {
            continue;
        }
        float3 perp = toC - rayDir * proj;
        if (dot(perp, perp) > rBound * rBound) {
            continue;
        }

        float sinY = sin(c1.z);
        float cosY = cos(c1.z);
        float sinP = sin(c1.w);
        float cosP = cos(c1.w);

        // Avoid self-shadowing on pixels belonging to this articulated box
        float3 ds = surfRel - c0.xyz;
        float3 s1 = float3(ds.x * cosY + ds.z * sinY, ds.y, -ds.x * sinY + ds.z * cosY);
        float3 sLoc = float3(s1.x, s1.y * cosP + s1.z * sinP, -s1.y * sinP + s1.z * cosP);
        if (abs(sLoc.x) < ext.x + 0.065f && abs(sLoc.y) < ext.y + 0.065f && abs(sLoc.z) < ext.z + 0.065f) {
            continue;
        }

        // Avoid self-occlusion when point/dynamic light emitter is inside this caster box
        float3 dl = lightRelPos - c0.xyz;
        float3 l1 = float3(dl.x * cosY + dl.z * sinY, dl.y, -dl.x * sinY + dl.z * cosY);
        float3 lLoc = float3(l1.x, l1.y * cosP + l1.z * sinP, -l1.y * sinP + l1.z * cosP);
        if (abs(lLoc.x) < ext.x + 0.12f && abs(lLoc.y) < ext.y + 0.12f && abs(lLoc.z) < ext.z + 0.12f) {
            continue;
        }

        float3 dr = rayOrigin - c0.xyz;
        float3 o1 = float3(dr.x * cosY + dr.z * sinY, dr.y, -dr.x * sinY + dr.z * cosY);
        float3 oLoc = float3(o1.x, o1.y * cosP + o1.z * sinP, -o1.y * sinP + o1.z * cosP);
        float3 d1 = float3(rayDir.x * cosY + rayDir.z * sinY, rayDir.y, -rayDir.x * sinY + rayDir.z * cosY);
        float3 dLoc = float3(d1.x, d1.y * cosP + d1.z * sinP, -d1.y * sinP + d1.z * cosP);
        float3 invD = float3(
            abs(dLoc.x) > 1e-5f ? 1.0f / dLoc.x : 1e5f,
            abs(dLoc.y) > 1e-5f ? 1.0f / dLoc.y : 1e5f,
            abs(dLoc.z) > 1e-5f ? 1.0f / dLoc.z : 1e5f
        );

        float tEst = clamp(proj, 0.04f, maxDist);
        float coneR = max(lightRadius * (tEst / max(maxDist, 0.25f)), 0.055f);
        float penumbraW = clamp(lightRadius * (tEst / max(maxDist - tEst, 0.25f)) * 0.65f, 0.055f, 0.42f);
        float3 extPad = ext + float3(penumbraW, penumbraW, penumbraW);
        float3 tp0 = (-extPad - oLoc) * invD;
        float3 tp1 = ( extPad - oLoc) * invD;
        float3 tpMin = min(tp0, tp1);
        float3 tpMax = max(tp0, tp1);
        float tNearPad = max(tpMin.x, max(tpMin.y, tpMin.z));
        float tFarPad = min(tpMax.x, min(tpMax.y, tpMax.z));
        if (tFarPad < max(tNearPad, 0.02f) || tNearPad >= maxDist) {
            continue;
        }

        float tMid = clamp(0.5f * (max(tNearPad, 0.02f) + min(tFarPad, maxDist)), 0.02f, maxDist);
        float3 pMid = oLoc + dLoc * tMid;
        float3 q = abs(pMid) - ext;
        float signedDist = length(max(q, float3(0.0f))) + min(max(q.x, max(q.y, q.z)), 0.0f);
        float rawBoxVis = smoothstep(-penumbraW * 0.35f, penumbraW, signedDist);
        float umbraStrength = saturate((min(ext.x, ext.z) * 1.45f) / coneR) * (1.0f - smoothstep(3.5f, 8.5f, tEst));
        float boxVis = lerp(1.0f, rawBoxVis, umbraStrength);
        vis = min(vis, boxVis);
        if (vis <= 0.005f) {
            return 0.0f;
        }
    }
    return vis;
}

// Full-resolution crisp articulated OBB entity sun/moon shadow (tight anti-aliased silhouette, Pythagoras sphere early-reject, zero stack arrays)
float evalEntitySunShadowCrisp(
    StructuredBuffer<float4> casters,
    float3 surfRel,
    float3 rayOrigin,
    float3 sunDir,
    float maxDist
) {
    uint cnt = min(uint(casters[0].x + 0.5f), 128u);
    if (cnt == 0u || maxDist <= 0.02f) {
        return 1.0f;
    }
    float vis = 1.0f;
    for (uint ci = 0u; ci < cnt; ++ci) {
        float4 c0 = casters[1u + 2u * ci];
        float4 c1 = casters[2u + 2u * ci];
        float3 ext = float3(c0.w, c1.x, c1.y);

        // Tight exact-radius ray-sphere early reject (~99% of pixels exit in 6 FMAs before any trig)
        float3 toC = c0.xyz - rayOrigin;
        float proj = dot(toC, sunDir);
        float rBound = length(ext) + 0.06f;
        if (proj < -rBound || proj > maxDist + rBound) {
            continue;
        }
        float dPerp2 = dot(toC, toC) - proj * proj;
        if (dPerp2 > rBound * rBound) {
            continue;
        }

        float sinY = sin(c1.z);
        float cosY = cos(c1.z);
        float sinP = sin(c1.w);
        float cosP = cos(c1.w);

        // Guard entity skin/armor surfaces against self-shadowing while keeping crisp foot contact on the ground below (-ext.y)
        float3 ds = surfRel - c0.xyz;
        float3 s1 = float3(ds.x * cosY + ds.z * sinY, ds.y, -ds.x * sinY + ds.z * cosY);
        float3 sLoc = float3(s1.x, s1.y * cosP + s1.z * sinP, -s1.y * sinP + s1.z * cosP);
        if (abs(sLoc.x) < ext.x + 0.055f && sLoc.y >= -ext.y + 0.015f && sLoc.y <= ext.y + 0.055f && abs(sLoc.z) < ext.z + 0.055f) {
            continue;
        }

        float3 dr = rayOrigin - c0.xyz;
        float3 o1 = float3(dr.x * cosY + dr.z * sinY, dr.y, -dr.x * sinY + dr.z * cosY);
        float3 oLoc = float3(o1.x, o1.y * cosP + o1.z * sinP, -o1.y * sinP + o1.z * cosP);
        float3 d1 = float3(sunDir.x * cosY + sunDir.z * sinY, sunDir.y, -sunDir.x * sinY + sunDir.z * cosY);
        float3 dLoc = float3(d1.x, d1.y * cosP + d1.z * sinP, -d1.y * sinP + d1.z * cosP);
        float3 invD = float3(
            abs(dLoc.x) > 1e-5f ? 1.0f / dLoc.x : 1e5f,
            abs(dLoc.y) > 1e-5f ? 1.0f / dLoc.y : 1e5f,
            abs(dLoc.z) > 1e-5f ? 1.0f / dLoc.z : 1e5f
        );

        float tEst = max(proj, 0.05f);
        float penumbraW = clamp(0.012f + tEst * 0.011f, 0.012f, 0.12f);
        float3 extPad = ext + float3(penumbraW, penumbraW, penumbraW);
        float3 tp0 = (-extPad - oLoc) * invD;
        float3 tp1 = ( extPad - oLoc) * invD;
        float3 tpMin = min(tp0, tp1);
        float3 tpMax = max(tp0, tp1);
        float tNearPad = max(tpMin.x, max(tpMin.y, tpMin.z));
        float tFarPad = min(tpMax.x, min(tpMax.y, tpMax.z));
        if (tFarPad < max(tNearPad, 0.015f) || tNearPad >= maxDist) {
            continue;
        }

        float tMid = clamp(0.5f * (max(tNearPad, 0.015f) + min(tFarPad, maxDist)), 0.015f, maxDist);
        float3 pMid = oLoc + dLoc * tMid;
        float3 q = abs(pMid) - ext;
        float sd = length(max(q, float3(0.0f))) + min(max(q.x, max(q.y, q.z)), 0.0f);
        float boxVis = smoothstep(-penumbraW * 0.55f, penumbraW * 0.75f, sd);
        float distFade = 1.0f - smoothstep(maxDist * 0.78f, maxDist, tEst);
        vis = min(vis, lerp(1.0f, boxVis, distFade));
        if (vis <= 0.005f) {
            return 0.0f;
        }
    }
    return vis;
}

MeteorSurfaceVoxel sampleMeteorSurfaceVoxel(
    StructuredBuffer<uint> vGrid,
    int3 vOrg,
    float3 worldPos,
    float3 rayDir,
    float3 surfNormal
) {
    MeteorSurfaceVoxel r;
    r.activeTag = 0u; r.activeSub = 0u; r.emitLvl = 0u;
    r.solidTag = 0u; r.spriteTag = 0u; r.frontTag = 0u; r.upTag = 0u;
    uint solidSub = 0u, spriteSub = 0u, frontSub = 0u;

    // Snap normal to its dominant cardinal axis so concave corners never step diagonally behind adjacent blocks
    float3 domAxis = snapCardinalAxis(surfNormal);

    int3 cSprite = int3(floor(worldPos + surfNormal * 0.02f)) - vOrg;
    int3 cSolid = int3(floor(worldPos - surfNormal * 0.04f)) - vOrg;
    int3 cFront = int3(floor(worldPos + domAxis * 0.28f)) - vOrg;
    int3 cUp = int3(floor(worldPos + float3(0.0f, 0.58f, 0.0f))) - vOrg;

    if (inVoxelBounds512Common(cSprite)) {
        uint idx = voxelLinearIndex512Common(cSprite);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        r.spriteTag = (b >> 4u) & 0xFu;
        spriteSub = b & 0xFu;
    }
    if (inVoxelBounds512Common(cSolid)) {
        uint idx = voxelLinearIndex512Common(cSolid);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        r.solidTag = (b >> 4u) & 0xFu;
        solidSub = b & 0xFu;
    }
    if (inVoxelBounds512Common(cFront)) {
        uint idx = voxelLinearIndex512Common(cFront);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        r.frontTag = (b >> 4u) & 0xFu;
        frontSub = b & 0xFu;
    }
    if (inVoxelBounds512Common(cUp)) {
        uint idx = voxelLinearIndex512Common(cUp);
        uint b = (vGrid[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
        r.upTag = (b >> 4u) & 0xFu;
    }

    // Distance from the solid block face plane along domAxis: on a solid wall seen through
    // transparent cutout holes (e.g. glow_lichen), wallOffset ~ 0.0m, whereas flat cutout decals sit at >= 0.00625m (0.1/16).
    float3 solidBase = floor(worldPos - domAxis * 0.045f) + max(domAxis, float3(0.0f));
    float wallOffset = abs(dot(worldPos - solidBase, domAxis));
    bool spriteOffWall = (r.solidTag == 0u) || (wallOffset > 0.003f);

    if (r.solidTag == 8u) {
        r.activeTag = 8u;
        r.activeSub = solidSub;
    } else if (r.spriteTag == 8u && (r.solidTag == 0u || spriteOffWall)) {
        r.activeTag = 8u;
        r.activeSub = spriteSub;
    } else if ((r.solidTag == 0u || r.solidTag == 1u || r.solidTag == 2u || r.solidTag == 12u || r.solidTag >= 14u) && spriteOffWall &&
               (r.spriteTag == 9u || r.spriteTag == 10u || r.spriteTag == 11u || r.spriteTag == 13u)) {
        r.activeTag = r.spriteTag;
        r.activeSub = spriteSub;
    } else if (r.spriteTag == 14u && spriteSub == 15u && (r.solidTag == 0u || spriteOffWall)) {
        r.activeTag = 14u;
        r.activeSub = 15u;
    } else if (r.solidTag != 0u) {
        r.activeTag = r.solidTag;
        r.activeSub = solidSub;
    } else if (r.spriteTag != 0u) {
        r.activeTag = r.spriteTag;
        r.activeSub = spriteSub;
    } else if (r.frontTag == 8u || r.frontTag == 9u || r.frontTag == 10u || r.frontTag == 11u || r.frontTag == 13u) {
        r.activeTag = r.frontTag;
        r.activeSub = frontSub;
    }
    r.emitLvl = (r.activeTag == 3u || r.activeTag == 8u || r.activeTag == 12u || r.activeTag >= 14u) ? 0u : r.activeSub;
    return r;
}

MeteorPbrMaterial decodePbrMaterial(uint vTag, uint vSub, float3 rawSrgb, float3 rawLin) {
    MeteorPbrMaterial m;
    m.metallic = 0.0f;
    m.roughness = 0.85f;
    m.f0Scalar = 0.04f;
    m.bumpScale = 0.0f;
    m.reflectivity = 0.0f;
    m.active = false;

    if (vTag == 8u) {
        bool isIce = (vSub == 1u);
        m.metallic = 0.0f;
        m.roughness = isIce ? 0.11f : 0.035f;
        m.f0Scalar = isIce ? 0.045f : 0.055f;
        m.bumpScale = isIce ? 0.24f : 0.28f;
        m.reflectivity = isIce ? 0.72f : 0.96f;
        m.active = true;
        return m;
    }

    if (vTag != 12u || vSub == 0u) {
        return m;
    }

    if (vSub == 1u) {
        m.metallic = 0.96f; m.roughness = 0.075f; m.f0Scalar = 0.92f; m.bumpScale = 0.34f; m.reflectivity = 1.00f; m.active = true;
    } else if (vSub == 2u) {
        m.metallic = 0.92f; m.roughness = 0.130f; m.f0Scalar = 0.84f; m.bumpScale = 0.30f; m.reflectivity = 0.94f; m.active = true;
    } else if (vSub == 3u) {
        m.metallic = 0.94f; m.roughness = 0.095f; m.f0Scalar = 0.88f; m.bumpScale = 0.36f; m.reflectivity = 0.96f; m.active = true;
    } else if (vSub == 4u) {
        m.metallic = 0.72f; m.roughness = 0.210f; m.f0Scalar = 0.58f; m.bumpScale = 0.34f; m.reflectivity = 0.74f; m.active = true;
    } else if (vSub == 5u) {
        float patina = saturate((rawSrgb.g + rawSrgb.b - rawSrgb.r * 1.60f) * 2.2f);
        m.metallic = lerp(0.84f, 0.32f, patina);
        m.roughness = lerp(0.14f, 0.34f, patina);
        m.f0Scalar = lerp(0.76f, 0.14f, patina);
        m.bumpScale = 0.38f;
        m.reflectivity = lerp(0.88f, 0.44f, patina);
        m.active = true;
    } else if (vSub == 6u) {
        m.metallic = 0.85f; m.roughness = 0.250f; m.f0Scalar = 0.62f; m.bumpScale = 0.34f; m.reflectivity = 0.70f; m.active = true;
    } else if (vSub == 7u) {
        m.metallic = 0.18f; m.roughness = 0.060f; m.f0Scalar = 0.14f; m.bumpScale = 0.28f; m.reflectivity = 0.94f; m.active = true;
    } else if (vSub == 8u) {
        m.metallic = 0.0f; m.roughness = 0.360f; m.f0Scalar = 0.04f; m.bumpScale = 0.24f; m.reflectivity = 0.24f; m.active = true;
    } else if (vSub == 9u) {
        m.metallic = 0.0f; m.roughness = 0.320f; m.f0Scalar = 0.04f; m.bumpScale = 0.28f; m.reflectivity = 0.30f; m.active = true;
    } else if (vSub == 10u) {
        float maxC = max(rawSrgb.r, max(rawSrgb.g, rawSrgb.b));
        float minC = min(rawSrgb.r, min(rawSrgb.g, rawSrgb.b));
        float vein = smoothstep(0.08f, 0.28f, maxC - minC);
        m.metallic = 0.84f * vein;
        m.roughness = lerp(0.55f, 0.11f, vein);
        m.f0Scalar = lerp(0.04f, 0.74f, vein);
        m.bumpScale = 0.38f * vein + 0.08f;
        m.reflectivity = 0.85f * vein;
        m.active = (vein > 0.04f);
    } else if (vSub == 11u) {
        float maxC = max(rawSrgb.r, max(rawSrgb.g, rawSrgb.b));
        float minC = min(rawSrgb.r, min(rawSrgb.g, rawSrgb.b));
        float vein = saturate(smoothstep(0.08f, 0.24f, maxC - minC) + smoothstep(0.68f, 0.88f, minC) * 0.8f);
        m.metallic = 0.35f * vein;
        m.roughness = lerp(0.65f, 0.10f, vein);
        m.f0Scalar = lerp(0.04f, 0.18f, vein);
        m.bumpScale = 0.28f * vein;
        m.reflectivity = 0.62f * vein;
        m.active = (vein > 0.05f);
    }
    return m;
}

float3 evaluateSemanticEmission(uint vTag, uint emitLvl, float3 rawSrgb, float3 rawLin, float dist, MetalityFrame f) {
    float maxC = max(rawSrgb.r, max(rawSrgb.g, rawSrgb.b));
    float minC = min(rawSrgb.r, min(rawSrgb.g, rawSrgb.b));
    float emitScale = clamp(f.lightingParams.z, 0.25f, 3.0f);

    if (vTag == 12u && emitLvl == 11u) {
        // Crystal & Precious Ores selective vein glow (host rock stays dark 0.0f)
        float oreScale = clamp(emitScale, 0.35f, 2.5f) / (1.0f + dist * 0.006f);
        float diamondMask = smoothstep(0.10f, 0.28f, rawSrgb.b - rawSrgb.r)
                          * smoothstep(0.08f, 0.24f, rawSrgb.g - rawSrgb.r)
                          * smoothstep(0.32f, 0.62f, maxC);
        float emeraldMask = smoothstep(0.12f, 0.28f, rawSrgb.g - rawSrgb.r)
                          * smoothstep(0.06f, 0.22f, rawSrgb.g - rawSrgb.b)
                          * smoothstep(0.30f, 0.58f, maxC);
        float redstoneMask = smoothstep(0.14f, 0.34f, rawSrgb.r - max(rawSrgb.g, rawSrgb.b))
                           * smoothstep(0.36f, 0.68f, rawSrgb.r)
                           * (1.0f - smoothstep(0.30f, 0.50f, rawSrgb.g));
        float lapisMask = smoothstep(0.14f, 0.30f, rawSrgb.b - rawSrgb.r)
                        * smoothstep(0.07f, 0.22f, rawSrgb.b - rawSrgb.g)
                        * smoothstep(0.28f, 0.56f, rawSrgb.b);
        float goldMask = smoothstep(0.20f, 0.42f, rawSrgb.r - rawSrgb.b)
                       * smoothstep(0.12f, 0.32f, rawSrgb.g - rawSrgb.b)
                       * smoothstep(0.42f, 0.72f, rawSrgb.g);
        float copperOrangeMask = smoothstep(0.14f, 0.30f, rawSrgb.r - rawSrgb.b)
                               * smoothstep(0.06f, 0.20f, rawSrgb.r - rawSrgb.g)
                               * smoothstep(0.40f, 0.68f, rawSrgb.r)
                               * (1.0f - goldMask);
        float copperGreenMask = smoothstep(0.06f, 0.18f, rawSrgb.g - rawSrgb.r)
                              * smoothstep(0.02f, 0.12f, rawSrgb.b - rawSrgb.r)
                              * (1.0f - diamondMask)
                              * (1.0f - emeraldMask);
        float quartzMask = (f.dimensionId == -1) ? smoothstep(0.68f, 0.86f, minC) : 0.0f;

        float3 oreGlow = diamondMask * float3(0.45f, 3.85f, 5.20f)
                       + emeraldMask * float3(0.32f, 4.40f, 1.35f)
                       + redstoneMask * float3(4.80f, 0.32f, 0.16f)
                       + lapisMask * float3(0.35f, 1.15f, 5.00f)
                       + goldMask * float3(4.20f, 2.55f, 0.28f)
                       + copperOrangeMask * float3(2.40f, 0.95f, 0.34f)
                       + copperGreenMask * float3(0.30f, 1.65f, 1.30f)
                       + quartzMask * float3(2.40f, 2.25f, 2.10f);
        return oreGlow * oreScale;
    }

    if (vTag == 4u) {
        // Molten Lava (emit >= 10) vs Magma Block (emit < 10) blackbody radiation
        float lavaScale = lerp(1.0f, emitScale, 0.65f);
        if (emitLvl > 0u && emitLvl < 10u) {
            float crackMask = smoothstep(0.32f, 0.78f, maxC) * smoothstep(0.08f, 0.36f, rawSrgb.r - rawSrgb.b);
            return float3(7.2f, 1.75f, 0.12f) * crackMask * lavaScale;
        }
        float texLuma = saturate((luminance(rawSrgb) - 0.30f) / 0.56f);
        float pulse = 0.93f + 0.07f * sin(f.timeSeconds * 1.8f + rawSrgb.r * 6.2f);
        float heat = pow(texLuma, 1.35f);
        float3 crustCol = float3(2.15f, 0.26f, 0.018f);
        float3 moltenCol = float3(8.40f, 1.90f, 0.095f);
        float3 hotCoreCol = float3(16.8f, 5.45f, 0.38f);
        float3 lavaHdr = lerp(crustCol, moltenCol, smoothstep(0.04f, 0.56f, heat));
        lavaHdr = lerp(lavaHdr, hotCoreCol, smoothstep(0.60f, 0.96f, heat));
        return lavaHdr * pulse * lavaScale;
    }

    if (vTag == 13u) {
        // Ruby-red Redstone Torch / Repeater / Comparator tip isolation (no pink washout)
        float rubyMask = smoothstep(0.38f, 0.82f, rawSrgb.r) * smoothstep(0.10f, 0.44f, rawSrgb.r - max(rawSrgb.g, rawSrgb.b));
        float baseScale = (emitLvl > 0u) ? (float(emitLvl) / 15.0f) : 0.7f;
        return float3(10.5f, 0.85f, 0.30f) * rubyMask * max(baseScale, 0.45f) * emitScale;
    }

    if (vTag == 5u || vTag == 9u) {
        // Lit Redstone Ore ruby glow vs Gold Glowstone/Ochre Froglight vs Warm Flame Amber Torches/Lanterns
        float baseScale = (emitLvl > 0u) ? (float(emitLvl) / 15.0f) : 0.9f;
        if (vTag == 5u && rawSrgb.r > rawSrgb.g * 1.45f && rawSrgb.r > rawSrgb.b * 1.55f) {
            float rubyOreMask = smoothstep(0.38f, 0.78f, rawSrgb.r) * smoothstep(0.12f, 0.40f, rawSrgb.r - max(rawSrgb.g, rawSrgb.b));
            return float3(8.5f, 0.70f, 0.22f) * rubyOreMask * max(baseScale, 0.6f) * emitScale;
        }
        if (vTag == 9u) {
            float flameMask = smoothstep(0.76f, 0.94f, maxC)
                            * smoothstep(0.58f, 0.82f, luminance(rawSrgb))
                            * smoothstep(0.06f, 0.28f, rawSrgb.r - rawSrgb.b);
            float3 warmFlame = lerp(float3(4.2f, 1.75f, 0.32f), float3(7.5f, 4.6f, 1.35f), smoothstep(0.80f, 0.97f, maxC));
            return warmFlame * flameMask * baseScale * emitScale;
        }
        float texBoost = smoothstep(0.24f, 0.88f, luminance(rawSrgb));
        float isGoldOchre = smoothstep(0.55f, 0.86f, rawSrgb.g);
        float3 tint = lerp(float3(1.1f, 0.50f, 0.14f), float3(1.25f, 0.82f, 0.28f), isGoldOchre);
        return (rawLin * 2.8f + tint * (1.8f + 4.4f * texBoost)) * baseScale * emitScale;
    }

    if (vTag == 6u || vTag == 10u) {
        // Verdant Froglight / Glow Lichen (Emerald Green) vs Sea Lantern/Beacon (Aqua White) vs Soul Cyan
        float baseScale = (emitLvl > 0u) ? (float(emitLvl) / 15.0f) : 0.9f;
        if (vTag == 10u && emitLvl == 7u) {
            // Glow Lichen (emitLvl == 7): subtle warm-sage glow strictly gated on lichen palette, never Soul Fire cyan
            float lichenMask = smoothstep(0.32f, 0.68f, maxC)
                             * smoothstep(-0.04f, 0.08f, rawSrgb.g - rawSrgb.r * 0.96f)
                             * smoothstep(-0.02f, 0.10f, rawSrgb.g - rawSrgb.b)
                             * (1.0f - smoothstep(0.18f, 0.35f, rawSrgb.b - rawSrgb.r));
            return (rawLin * 1.35f + float3(0.38f, 0.62f, 0.44f) * 0.85f) * lichenMask * baseScale * emitScale;
        }
        if (rawSrgb.g > rawSrgb.b * 1.10f && rawSrgb.g > rawSrgb.r * 1.05f) {
            float gBoost = smoothstep(0.28f, 0.85f, maxC);
            return (rawLin * 3.8f + float3(1.1f, 7.2f, 2.2f) * (0.45f + 0.55f * gBoost)) * baseScale * emitScale;
        }
        if (vTag == 6u && luminance(rawSrgb) > 0.62f && rawSrgb.b >= rawSrgb.g * 0.92f && rawSrgb.r > 0.42f) {
            float aBoost = smoothstep(0.45f, 0.92f, maxC);
            return (rawLin * 3.5f + float3(3.2f, 7.6f, 8.8f) * (0.45f + 0.55f * aBoost)) * baseScale * emitScale;
        }
        if (vTag == 10u && (emitLvl == 0u || emitLvl >= 10u)) {
            float soulMask = smoothstep(0.32f, 0.76f, maxC) * smoothstep(0.06f, 0.38f, rawSrgb.b + rawSrgb.g - rawSrgb.r * 1.2f);
            float3 soulFlame = lerp(float3(0.45f, 5.2f, 7.6f), float3(1.8f, 11.5f, 15.0f), smoothstep(0.65f, 0.96f, maxC));
            return soulFlame * soulMask * baseScale * emitScale;
        }
        float aquaMask = smoothstep(0.24f, 0.54f, min(rawSrgb.g, rawSrgb.b)) * (1.0f - smoothstep(0.10f, 0.28f, rawSrgb.r - max(rawSrgb.g, rawSrgb.b)));
        float texBoost = smoothstep(0.22f, 0.90f, luminance(rawSrgb));
        return (rawLin * 4.0f + float3(0.20f, 0.92f, 1.12f) * (3.6f + 6.8f * texBoost)) * aquaMask * baseScale * emitScale;
    }

    if (vTag == 7u || vTag == 11u) {
        // End Rod (Pearl-Lilac White) vs Pearlescent Froglight / Portal / Crying Obsidian (Arcane Violet)
        float baseScale = (emitLvl > 0u) ? (float(emitLvl) / 15.0f) : 0.85f;
        if (vTag == 11u && minC > 0.65f) {
            float rodMask = smoothstep(0.45f, 0.82f, minC);
            return float3(6.8f, 5.8f, 8.2f) * rodMask * baseScale * emitScale;
        }
        if (vTag == 11u) {
            float rodMask = smoothstep(0.30f, 0.72f, maxC);
            return (rawLin * 4.5f + float3(0.84f, 0.40f, 1.12f) * 7.0f) * rodMask * baseScale * emitScale;
        }
        float veinMask = smoothstep(0.26f, 0.76f, maxC) * smoothstep(0.04f, 0.25f, rawSrgb.b + rawSrgb.r - rawSrgb.g * 1.3f);
        return (rawLin * 4.0f + float3(0.82f, 0.32f, 1.05f) * 5.8f) * veinMask * baseScale * emitScale;
    }

    // Distance fallback ONLY beyond voxelGrid (240m+) in The Nether for far lava seas
    if (f.dimensionId == -1 && dist > 235.0f) {
        float farLava = smoothstep(0.72f, 0.92f, rawSrgb.r) *
                        smoothstep(0.30f, 0.65f, rawSrgb.g) *
                        (1.0f - smoothstep(0.20f, 0.55f, rawSrgb.b));
        return float3(9.0f, 2.2f, 0.14f) * farLava;
    }

    return float3(0.0f, 0.0f, 0.0f);
}

float evalHorizonAoSample(float3 centerRelPos, float3 nrm, float3 sampleRelPos) {
    float3 diff = sampleRelPos - centerRelPos;
    float dist2 = dot(diff, diff);
    if (dist2 > 1e-4f && dist2 < 9.0f) {
        float horizon = max(dot(normalize(diff), nrm) - 0.12f, 0.0f);
        return horizon * (1.0f / (1.0f + dist2 * 0.6f));
    }
    return 0.0f;
}

// Blender AgX Filmic + Punchy Look (true 0->0 dark response, shadow-protected contrast & chroma)
float3 agxDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return max(
        15.5f * x4 * x2
      - 40.14f * x4 * x
      + 31.96f * x4
      - 6.868f * x2 * x
      + 0.4298f * x2
      + 0.1191f * x,
        float3(0.0f, 0.0f, 0.0f)
    );
}

float3 meteorAgxTonemap(float3 hdr, MetalityFrame f) {
    float shadowDepth = clamp(f.lightingParams.y, 0.5f, 2.0f);
    float contrast = clamp(f.tonemapParams.y, 0.7f, 1.8f);
    float saturation = clamp(f.tonemapParams.z, 0.5f, 1.8f);

    hdr = max(hdr, float3(0.0f, 0.0f, 0.0f));

    // Clean AgX matrix with restrained inset so colors stay rich and never look milky/pastel
    float3 agxIn;
    agxIn.r = dot(hdr, float3(0.896f, 0.052f, 0.052f));
    agxIn.g = dot(hdr, float3(0.032f, 0.924f, 0.044f));
    agxIn.b = dot(hdr, float3(0.032f, 0.052f, 0.916f));
    agxIn = max(agxIn, float3(1e-5f, 1e-5f, 1e-5f));
    float3 logVal = clamp((log2(agxIn) + 12.47393f) / 16.5f, float3(0.0f), float3(1.0f));
    float3 curved = agxDefaultContrastApprox(logVal);

    float3 agxLin = pow(max(curved, float3(1e-6f, 1e-6f, 1e-6f)), float3(2.2f, 2.2f, 2.2f));
    float3 outRgb;
    outRgb.r = dot(agxLin, float3(1.121f, -0.060f, -0.061f));
    outRgb.g = dot(agxLin, float3(-0.038f, 1.092f, -0.054f));
    outRgb.b = dot(agxLin, float3(-0.038f, -0.060f, 1.098f));
    outRgb = max(outRgb, float3(0.0f, 0.0f, 0.0f));

    // Soft shadow depth toe (slope 0.09) + shadow-protected contrast curve (never crushes dark wood/interiors)
    float toePow = 1.0f + (shadowDepth - 0.95f) * 0.09f;
    float3 shadowToned = pow(max(outRgb, float3(1e-5f, 1e-5f, 1e-5f)), float3(toePow, toePow, toePow));
    outRgb = lerp(outRgb, shadowToned, 1.0f - smoothstep(0.08f, 0.58f, max(outRgb.r, max(outRgb.g, outRgb.b))));

    float luma = dot(outRgb, float3(0.2126f, 0.7152f, 0.0722f));
    float sc = smoothstep(0.0f, 1.0f, luma);
    float contrastW = (contrast - 1.0f) * 0.52f * smoothstep(0.045f, 0.26f, luma);
    float targetLuma = lerp(luma, sc, contrastW);
    outRgb *= targetLuma / max(luma, 1e-4f);

    float punchLuma = dot(outRgb, float3(0.2126f, 0.7152f, 0.0722f));
    float effSat = lerp(1.0f, saturation, smoothstep(0.010f, 0.065f, punchLuma));
    outRgb = max(float3(0.0f, 0.0f, 0.0f), float3(punchLuma) + effSat * (outRgb - float3(punchLuma)));
    return clamp(outRgb, float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
}

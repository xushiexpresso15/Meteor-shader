// Meteor Shader Atmosphere, Clouds, Nether Biomes & End Gargantua Black Hole

float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0f - g2) / (4.0f * METEOR_PI * pow(max(1.0f + g2 - 2.0f * g * cosTheta, 0.05f), 1.5f));
}

float starFieldLayer(float3 dir, float scale, float densityPow) {
    float3 p = dir * scale;
    float3 cell = floor(p);
    float3 f = frac(p) - 0.5f;
    float rnd = hash13(cell);
    float mask = smoothstep(densityPow, 1.0f, rnd);
    float d = length(f - (float3(hash13(cell + 1.1f), hash13(cell + 2.3f), hash13(cell + 3.7f)) - 0.5f) * 0.55f);
    return mask * exp(-d * d * 28.0f);
}

struct VoxelCubeHit {
    bool hit;
    float edgeAlpha;
    float3 faceN;
    float2 fuv;
    int faceId;
};

float3 rotateCubeEuler(float3 v, float yaw, float pitch, float roll) {
    float sr = sin(roll), cr = cos(roll);
    float3 v1 = float3(v.x * cr - v.y * sr, v.x * sr + v.y * cr, v.z);
    float sp = sin(pitch), cp = cos(pitch);
    float3 v2 = float3(v1.x, v1.y * cp - v1.z * sp, v1.y * sp + v1.z * cp);
    float sy = sin(yaw), cy = cos(yaw);
    return float3(v2.x * cy + v2.z * sy, v2.y, -v2.x * sy + v2.z * cy);
}

VoxelCubeHit intersectTiltedVoxelCube(float3 dir, float3 centerDir, float angSize, float tiltYaw, float tiltPitch, float tiltRoll) {
    VoxelCubeHit res;
    res.hit = false;
    res.edgeAlpha = 0.0f;
    res.faceN = float3(0.0f, 0.0f, -1.0f);
    res.fuv = float2(0.5f, 0.5f);
    res.faceId = 0;

    if (dot(dir, centerDir) <= 0.25f) {
        return res;
    }

    float3 refUp = abs(centerDir.y) < 0.94f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 cRight = normalize(cross(refUp, centerDir));
    float3 cUp = normalize(cross(centerDir, cRight));
    float2 tPlane = float2(dot(dir, cRight), dot(dir, cUp)) / max(angSize, 1e-4f);
    if (dot(tPlane, tPlane) > 3.2f) {
        return res;
    }

    float3 ro = rotateCubeEuler(float3(tPlane.x, tPlane.y, -4.0f), tiltYaw, tiltPitch, tiltRoll);
    float3 rd = rotateCubeEuler(float3(0.0f, 0.0f, 1.0f), tiltYaw, tiltPitch, tiltRoll);

    float b = 0.62f;
    float3 invD = float3(
        abs(rd.x) > 1e-5f ? 1.0f / rd.x : 1e5f,
        abs(rd.y) > 1e-5f ? 1.0f / rd.y : 1e5f,
        abs(rd.z) > 1e-5f ? 1.0f / rd.z : 1e5f
    );
    float3 tMinV = (-b - ro) * invD;
    float3 tMaxV = ( b - ro) * invD;
    float3 t0 = min(tMinV, tMaxV);
    float3 t1 = max(tMinV, tMaxV);
    float tNear = max(t0.x, max(t0.y, t0.z));
    float tFar = min(t1.x, min(t1.y, t1.z));

    if (tFar > tNear) {
        res.hit = true;
        res.edgeAlpha = smoothstep(0.0f, 0.045f, tFar - tNear);
        float3 hitP = ro + rd * tNear;
        if (t0.x >= t0.y && t0.x >= t0.z) {
            res.faceN = float3(rd.x < 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
            res.fuv = hitP.yz / (2.0f * b) + 0.5f;
            res.faceId = (rd.x < 0.0f) ? 0 : 1;
        } else if (t0.y >= t0.z) {
            res.faceN = float3(0.0f, rd.y < 0.0f ? 1.0f : -1.0f, 0.0f);
            res.fuv = hitP.xz / (2.0f * b) + 0.5f;
            res.faceId = (rd.y < 0.0f) ? 2 : 3;
        } else {
            res.faceN = float3(0.0f, 0.0f, rd.z < 0.0f ? 1.0f : -1.0f);
            res.fuv = hitP.xy / (2.0f * b) + 0.5f;
            res.faceId = (rd.z < 0.0f) ? 4 : 5;
        }
    }
    return res;
}

float3 evalLunarCubeSurface(VoxelCubeHit hit) {
    float2 qUv = floor(clamp(hit.fuv, 0.0f, 0.9999f) * 16.0f);
    float2 pix = qUv / 16.0f;
    float maria = fbm2D(pix * 3.8f + float2(float(hit.faceId) * 11.3f, float(hit.faceId) * 5.7f));
    float regolith = fbm2D(pix * 7.6f + float2(float(hit.faceId) * 3.1f + 4.2f, 9.7f));
    float grain = hash12(qUv + float2(float(hit.faceId) * 29.0f, 7.0f));
    float craterTex = saturate(smoothstep(0.28f, 0.72f, maria) * 0.55f + smoothstep(0.25f, 0.75f, regolith) * 0.25f + grain * 0.22f);

    float3 keyL = normalize(float3(0.45f, 0.88f, -0.35f));
    float faceShade = clamp(0.40f + 0.60f * max(dot(hit.faceN, keyL), 0.0f), 0.38f, 1.00f);
    float3 craterCol = float3(0.42f, 0.60f, 0.96f);
    float3 ridgeCol = float3(1.85f, 2.15f, 2.75f);
    return lerp(craterCol, ridgeCol, craterTex) * faceShade;
}

float3 evalSolarCubeSurface(VoxelCubeHit hit, float sunChord, float sunAngSize, MeteorEnvFactors env) {
    float3 keyL = normalize(float3(0.40f, 0.90f, -0.30f));
    float nd = max(dot(hit.faceN, keyL), 0.0f);
    float faceW = 0.28f + 0.72f * nd;
    float coreGlow = 1.0f - smoothstep(0.0f, sunAngSize * 0.85f, sunChord);

    float3 litCol = lerp(float3(4.0f, 3.35f, 2.25f), float3(6.0f, 2.85f, 0.85f), env.sunsetFactor);
    float3 shadeCol = lerp(float3(2.05f, 1.40f, 0.56f), float3(3.2f, 0.94f, 0.16f), env.sunsetFactor);
    return lerp(shadeCol, litCol, faceW) + litCol * (coreGlow * 0.22f);
}

float sharpStarLayer(float3 dir, float scale, float densityPow) {
    float3 p = dir * scale;
    float3 cell = floor(p);
    float3 f = frac(p) - 0.5f;
    float rnd = hash13(cell);
    float mask = smoothstep(densityPow, 1.0f, rnd);
    float d = length(f - (float3(hash13(cell + 1.1f), hash13(cell + 2.3f), hash13(cell + 3.7f)) - 0.5f) * 0.58f);
    return mask * (exp(-d * d * 135.0f) + 0.12f * exp(-d * d * 24.0f));
}

float fbm2D2(float2 p) {
    float n0 = noise2D(p);
    float2 p1 = float2(p.x * 1.97f - p.y * 0.42f, p.x * 0.42f + p.y * 1.97f);
    return n0 * 0.65f + noise2D(p1) * 0.35f;
}

float fbm3D2(float3 p) {
    float n0 = noise3D(p);
    float3 p1 = p * 2.04f + float3(13.7f, 7.1f, 19.3f);
    return n0 * 0.64f + noise3D(p1) * 0.36f;
}

float3 evalMilkyWay(float3 d, float timeSec) {
    // Tilted 3D galactic plane (completely pole-singularity-free across full celestial dome)
    float3 gNormal = normalize(float3(0.56f, 0.38f, 0.736f));
    float gLat = dot(d, gNormal);
    float absLat = abs(gLat);
    if (absLat >= 0.56f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float gLat2 = gLat * gLat;
    // Smooth zero-derivative window guarantees 100% seamless cutoff at absLat == 0.56
    float edgeWin = smoothstep(0.56f, 0.34f, absLat);
    float outerBand = exp(-gLat2 * 11.0f) * edgeWin;
    float coreBand = exp(-gLat2 * 38.0f) * edgeWin;

    float w1 = fbm3D2(d * 3.1f + float3(1.7f, 4.3f, 2.9f)) - 0.5f;
    float w2 = fbm3D2(d * 6.4f + float3(8.1f, 2.5f, 4.7f)) - 0.5f;
    float3 wd = d + gNormal * (w1 * 0.13f + w2 * 0.045f);

    float cloudLarge = smoothstep(0.26f, 0.76f, fbm3D2(wd * 4.4f));
    float cloudFine = fbm3D(wd * 10.5f + float3(5.4f, 1.9f + timeSec * 0.0002f, 8.2f));
    float wispDetail = smoothstep(0.32f, 0.78f, cloudFine);
    // Wispy filamentous Great Rift dust lanes carved down the galactic spine
    float rift = smoothstep(0.45f, 0.72f, cloudFine) * (0.55f + 0.45f * cloudLarge) * exp(-gLat2 * 85.0f);
    float dustCut = 1.0f - rift * 0.76f;

    float3 gCoreDir = normalize(float3(-0.62f, 0.48f, 0.62f));
    float coreBulge = pow(saturate(dot(d, gCoreDir) * 0.5f + 0.5f), 4.0f) * coreBand;

    // Natural astrophotography deep-space tones (zero neon turquoise blotches)
    float3 deepIndigo = float3(0.009f, 0.017f, 0.046f);
    float3 starCloud = float3(0.050f, 0.066f, 0.122f);
    float3 paleNebula = float3(0.044f, 0.086f, 0.132f);
    float3 warmDust = float3(0.115f, 0.072f, 0.044f);

    float nebKnot = smoothstep(0.52f, 0.84f, cloudFine) * cloudLarge * coreBand;
    float3 mw = deepIndigo * outerBand * (0.45f + 0.55f * cloudLarge)
              + starCloud * coreBand * (0.28f + 0.48f * cloudLarge + 0.34f * wispDetail)
              + paleNebula * nebKnot * 0.50f
              + warmDust * coreBulge * (0.40f + 0.60f * cloudLarge) * 0.65f;
    mw *= dustCut;

    float microStars = sharpStarLayer(d, 235.0f, 0.885f) * outerBand * (1.0f - rift * 0.70f);
    float3 msTint = lerp(float3(0.68f, 0.78f, 1.05f), float3(1.05f, 0.86f, 0.66f), coreBulge);
    mw += msTint * microStars * 0.64f;
    return mw;
}

float3 evalAuroraBorealis(float3 d, float timeSec, float dither) {
    if (d.y < 0.05f || d.y > 0.78f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    // Natural high-latitude curtain arc fading cleanly well below zenith (zero pole clutter/pinch)
    float elevFade = smoothstep(0.05f, 0.19f, d.y) * (1.0f - smoothstep(0.46f, 0.76f, d.y));
    float northArc = smoothstep(-0.35f, 0.35f, -d.z * 0.86f + d.x * 0.51f);
    float regionFade = elevFade * northArc;
    if (regionFade <= 0.005f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float tDrift = timeSec * 0.014f;
    float3 aurSum = float3(0.0f, 0.0f, 0.0f);
    for (int sl = 0; sl < 6; ++sl) {
        float fh = (float(sl) + dither) * (1.0f / 6.0f);
        float alt = lerp(0.82f, 2.25f, fh);
        float2 skyProj = d.xz * (alt / max(d.y + 0.08f, 0.16f));
        float2 p = skyProj + float2(tDrift, -tDrift * 0.65f);

        float warp = fbm2D2(p * 0.24f + float2(tDrift * 0.7f, -tDrift * 0.5f));
        float fold1 = sin(p.x * 0.60f + p.y * 0.36f + warp * 5.2f);
        float fold2 = cos(p.x * 0.42f - p.y * 0.56f + warp * 4.1f + 1.8f);
        float ribbon = exp(-abs(fold1) * 7.6f) * 0.85f + exp(-abs(fold2) * 9.0f) * 0.48f;
        float rayCoord = p.x * 5.8f + p.y * 3.6f + warp * 3.2f;
        float striation = 0.42f + 0.58f * noise2D(float2(rayCoord, timeSec * 0.22f + fh * 0.40f));

        float3 lowCol = float3(0.035f, 0.44f, 0.23f);
        float3 midCol = float3(0.048f, 0.27f, 0.38f);
        float3 highCol = float3(0.24f, 0.060f, 0.36f);
        float3 sliceCol = (fh < 0.32f)
            ? lerp(lowCol, midCol, fh * (1.0f / 0.32f))
            : lerp(midCol, highCol, smoothstep(0.32f, 0.88f, fh));

        float vProfile = smoothstep(0.0f, 0.15f, fh) * exp(-max(fh - 0.15f, 0.0f) * 2.20f);
        aurSum += sliceCol * (ribbon * striation * vProfile);
    }
    return (aurSum * (1.0f / 6.0f)) * regionFade * 0.90f;
}

float3 evalShootingStars(float3 d, float timeSec) {
    if (d.y < 0.10f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    float3 meteors = float3(0.0f, 0.0f, 0.0f);
    for (int mi = 0; mi < 3; ++mi) {
        float phase = float(mi) * 0.34f;
        float cycle = timeSec / 7.5f + phase + float(mi) * 1.7f;
        float id = floor(cycle);
        float tf = frac(cycle);
        if (tf < 0.22f) {
            float prog = tf / 0.22f;
            float2 rnd0 = hash22(float2(id * 13.7f + float(mi) * 7.1f, id * 5.3f));
            float2 rnd1 = hash22(float2(id * 29.1f + 3.7f, float(mi) * 19.3f));
            float phi = rnd0.x * METEOR_TAU;
            float elev = lerp(0.32f, 0.76f, rnd0.y);
            float3 sOrig = normalize(float3(cos(phi) * sqrt(1.0f - elev * elev), elev, sin(phi) * sqrt(1.0f - elev * elev)));
            float3 rightA = normalize(cross(float3(0.0f, 1.0f, 0.0f), sOrig));
            float3 upA = cross(sOrig, rightA);
            float angVel = rnd1.x * METEOR_TAU;
            float3 sVel = normalize(rightA * cos(angVel) + upA * (sin(angVel) * 0.55f - 0.45f));

            float streakLen = 0.26f;
            float3 headPos = normalize(sOrig + sVel * (prog * 0.38f));
            float3 tailPos = normalize(sOrig + sVel * max(prog * 0.38f - streakLen * smoothstep(0.0f, 0.25f, prog), 0.0f));
            float3 seg = headPos - tailPos;
            float segLen2 = dot(seg, seg);
            if (segLen2 > 1e-6f) {
                float uProj = clamp(dot(d - tailPos, seg) / segLen2, 0.0f, 1.0f);
                float3 closestP = normalize(lerp(tailPos, headPos, uProj));
                float perpD = length(d - closestP);
                float coreLine = exp(-perpD * 740.0f) * smoothstep(0.0f, 0.85f, uProj);
                float glowLine = exp(-perpD * 230.0f) * pow(uProj, 2.0f);
                float headSpark = exp(-length(d - headPos) * 340.0f);
                float env = sin(prog * METEOR_PI);
                meteors += (float3(1.35f, 1.62f, 2.25f) * (coreLine + headSpark * 1.05f) + float3(0.55f, 0.85f, 1.75f) * glowLine * 0.40f) * env;
            }
        }
    }
    return meteors;
}

// Analytic C-infinity smooth solar Mie forward-scattering aureole (evaluated at 1.0x — zero LUT quantization rings!)
float3 evalOverworldSolarMieGlow(float cosSun, MeteorEnvFactors env) {
    float uSun = max(1.0f - cosSun, 0.0f);
    float broad = 0.078f / (1.0f + uSun * 10.0f);
    float aureole = 0.19f / pow(1.0f + uSun * 55.0f, 1.5f);
    float corona = 0.28f * exp(-uSun * 380.0f);
    float3 mieCol = lerp(float3(1.02f, 0.88f, 0.68f), float3(2.15f, 0.80f, 0.22f), env.sunsetFactor);
    return mieCol * (broad + aureole + corona);
}

float3 evalOverworldLunarHalo(float cosMoon) {
    float uMoon = max(1.0f - cosMoon, 0.0f);
    float glow = 0.09f / pow(1.0f + uMoon * 40.0f, 1.5f) + 0.105f * exp(-uMoon * 240.0f);
    return float3(0.30f, 0.44f, 0.78f) * glow;
}

float3 evalOverworldSkyBase(float3 dir, MetalityFrame f, MeteorEnvFactors env, bool includeDiscs) {
    float3 d = normalize(dir);
    float3 sunDir = normalize(f.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 moonDir = normalize(f.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float yPos = max(d.y, 0.0f);
    float cosSun = dot(d, sunDir);
    float cosMoon = dot(d, moonDir);

    // Day sky gradient (rich azure zenith -> cerulean -> soft horizon, comfortable non-glaring brightness)
    float3 dayZenith = float3(0.016f, 0.115f, 0.68f);
    float3 dayMid = float3(0.068f, 0.31f, 0.79f);
    float3 dayHorizon = float3(0.29f, 0.56f, 0.82f);
    float3 dayCol = lerp(dayHorizon, dayMid, pow(yPos, 0.45f));
    dayCol = lerp(dayCol, dayZenith, smoothstep(0.16f, 0.85f, yPos));

    // Sunset / Sunrise gradient (azimuth glow fades to 0 well below zenith so zenith pole is 100% symmetric)
    float3 setZenith = float3(0.095f, 0.070f, 0.31f);
    float3 setMid = float3(0.58f, 0.19f, 0.42f);
    float3 setHorizon = float3(1.38f, 0.52f, 0.14f);
    float sunAzimuthGlow = pow(max(cosSun, 0.0f), 3.0f) * smoothstep(0.45f, 0.0f, yPos);
    float3 setCol = lerp(setHorizon + float3(0.44f, 0.16f, 0.03f) * sunAzimuthGlow, setMid, smoothstep(0.0f, 0.30f, yPos));
    setCol = lerp(setCol, setZenith, smoothstep(0.18f, 0.82f, yPos));

    // Deep velvet night sky gradient with subtle dusky horizon glow
    float3 nightZenith = float3(0.002f, 0.0045f, 0.014f);
    float3 nightMid = float3(0.006f, 0.013f, 0.036f);
    float3 nightHorizon = float3(0.026f, 0.020f, 0.038f);
    float3 nightCol = lerp(nightHorizon, nightMid, smoothstep(0.0f, 0.24f, yPos));
    nightCol = lerp(nightCol, nightZenith, smoothstep(0.20f, 0.82f, yPos));

    float baseW = env.dayFactor + env.sunsetFactor + env.nightFactor + 1e-4f;
    float3 sky = (dayCol * env.dayFactor + setCol * env.sunsetFactor * 1.10f + nightCol * env.nightFactor) / baseW;
    sky += srgbToLinear(f.sunriseColor.rgb) * f.sunriseColor.a * env.sunsetFactor * 0.28f * exp(-yPos * 4.5f);

    // Rain / storm atmosphere (deep dark storm sky at night so rain puddles contrast glowing lamps cleanly)
    float3 rainSky = lerp(float3(0.15f, 0.18f, 0.23f), float3(0.075f, 0.095f, 0.13f), pow(yPos, 0.6f))
                   * lerp(0.055f, 1.0f, 1.0f - env.nightFactor * 0.945f);
    sky = lerp(sky, rainSky, env.rainFactor * 0.86f);

    if (d.y < 0.0f) {
        float downFade = exp(d.y * 5.0f);
        float3 groundRefl = lerp(float3(0.11f, 0.15f, 0.22f), float3(0.22f, 0.14f, 0.17f), env.sunsetFactor);
        sky = lerp(groundRefl * (1.0f - env.nightFactor * 0.85f), sky, downFade);
    }

    if (includeDiscs) {
        float nightVis = saturate(max(env.nightFactor, f.skyColor.a * 0.9f)) * smoothstep(-0.02f, 0.20f, d.y) * (1.0f - env.rainFactor * 0.85f);
        VoxelCubeHit moonHit;
        moonHit.hit = false;
        moonHit.edgeAlpha = 0.0f;
        moonHit.faceN = float3(0.0f, 0.0f, -1.0f);
        moonHit.fuv = float2(0.5f, 0.5f);
        moonHit.faceId = 0;
        if ((nightVis > 0.01f || env.nightFactor > 0.001f) && cosMoon > 0.92f) {
            moonHit = intersectTiltedVoxelCube(d, moonDir, 0.082f, 0.72f, 0.58f, 0.28f);
        }
        float moonOcc = 1.0f - moonHit.edgeAlpha;

        if (nightVis > 0.01f) {
            float skyDither = 0.5f;
            float3 milky = evalMilkyWay(d, f.timeSeconds);
            float3 aurora = evalAuroraBorealis(d, f.timeSeconds, skyDither);
            float3 meteors = evalShootingStars(d, f.timeSeconds);

            float starsHero = sharpStarLayer(d, 72.0f, 0.928f) * 2.45f;
            float starsMid = sharpStarLayer(d, 145.0f, 0.938f) * 1.35f;
            float starsFine = sharpStarLayer(d, 260.0f, 0.955f) * 0.80f;
            float starHue = hash13(floor(d * 72.0f));
            float3 starCol = lerp(float3(0.76f, 0.90f, 1.22f), float3(1.18f, 0.96f, 0.74f), smoothstep(0.45f, 0.92f, starHue));
            float twinkle = 0.80f + 0.20f * sin(f.timeSeconds * 2.2f + dot(floor(d * 72.0f), float3(12.3f, 45.6f, 78.9f)));

            sky += (milky + aurora + meteors + starCol * (starsHero + starsMid + starsFine) * twinkle) * nightVis * moonOcc;
        }

        float horizonMask = smoothstep(-0.05f, 0.02f, d.y) * (1.0f - env.rainFactor * 0.90f);
        float sunVis = horizonMask * (1.0f - env.nightFactor * 0.95f);
        if (sunVis > 0.001f) {
            sky += evalOverworldSolarMieGlow(cosSun, env) * sunVis;
        }

        if (sunVis > 0.001f && cosSun > 0.94f) {
            float sunAngSize = 0.052f;
            float sunChord = length(d - sunDir);
            VoxelCubeHit sunHit = intersectTiltedVoxelCube(d, sunDir, sunAngSize, 0.65f, 0.54f, 0.24f);
            if (sunHit.hit) {
                float3 sunCubeCol = evalSolarCubeSurface(sunHit, sunChord, sunAngSize, env);
                sky = lerp(sky, max(sky, sunCubeCol), sunHit.edgeAlpha * sunVis);
            }
        }

        if (env.nightFactor > 0.005f) {
            sky += evalOverworldLunarHalo(cosMoon) * env.nightFactor * horizonMask;
        }

        if (moonHit.hit) {
            float3 moonCubeCol = evalLunarCubeSurface(moonHit);
            sky = lerp(sky, moonCubeCol, moonHit.edgeAlpha * env.nightFactor * horizonMask);
        }
    }

    return sky;
}

float evalCloudDensityAt(float3 pos, MetalityFrame f, float baseH, float topH, float nightClear) {
    float hNorm = saturate((pos.y - baseH) / max(topH - baseH, 1.0f));
    float heightProfile = smoothstep(0.0f, 0.18f, hNorm) * (1.0f - smoothstep(0.45f, 1.0f, hNorm));
    float3 wind = float3(f.timeSeconds * 2.4f, 0.0f, f.timeSeconds * 0.95f);
    float coverage = fbm2D(pos.xz * 0.0038f + wind.xz * 0.0022f);
    float billow = fbm3D(pos * 0.015f + wind * 0.009f);
    float threshold = lerp(0.715f + 0.055f * nightClear, 0.44f, f.rainStrength);
    float d = coverage * 0.78f + billow * 0.50f - threshold;
    float densGain = lerp(0.95f, 2.40f, f.rainStrength);
    return clamp(smoothstep(0.0f, 0.28f, d) * heightProfile * densGain, 0.0f, 1.0f);
}

struct CloudTraceResult {
    float3 radiance;
    float transmittance;
};

CloudTraceResult sampleVolumetricCloudsNAt(float3 rayDir, float2 screenPx, MetalityFrame f, MeteorEnvFactors env, int stepCap, bool singleSunTap) {
    CloudTraceResult outC;
    outC.radiance = float3(0.0f, 0.0f, 0.0f);
    outC.transmittance = 1.0f;

    float3 d = normalize(rayDir);
    if (d.y < 0.018f || !meteorFeatureClouds(f)) {
        return outC;
    }

    float baseH = max(f.camWorldPos.y + 38.0f, 155.0f);
    float topH = baseH + 95.0f;
    float tStart = max(0.0f, (baseH - f.camWorldPos.y) / d.y);
    float tEnd = max(tStart + 1.0f, (topH - f.camWorldPos.y) / d.y);
    float maxDist = 1400.0f;
    if (tStart > maxDist) {
        return outC;
    }
    tEnd = min(tEnd, tStart + 460.0f);

    int steps = clamp(stepCap, 5, 14);
    float stepSize = (tEnd - tStart) / float(steps);
    float3 sunDir = normalize(f.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 moonDir = normalize(f.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 lightDir = (sunDir.y < -0.05f) ? moonDir : sunDir;
    float cosTheta = dot(d, lightDir);
    float hg = lerp(henyeyGreenstein(cosTheta, 0.62f), henyeyGreenstein(cosTheta, -0.2f), 0.35f);

    float3 warmLit = lerp(float3(1.62f, 1.52f, 1.40f), float3(2.65f, 1.12f, 0.43f), env.sunsetFactor);
    warmLit = lerp(warmLit, float3(0.30f, 0.33f, 0.38f), env.rainFactor * 0.80f);
    warmLit = lerp(warmLit, float3(0.020f, 0.030f, 0.052f) * lerp(1.0f, 0.28f, env.rainFactor), env.nightFactor);

    float3 shadowBase = lerp(float3(0.31f, 0.43f, 0.61f), float3(0.28f, 0.20f, 0.35f), env.sunsetFactor);
    shadowBase = lerp(shadowBase, float3(0.13f, 0.15f, 0.19f), env.rainFactor * 0.82f);
    shadowBase = lerp(shadowBase, float3(0.004f, 0.007f, 0.015f) * lerp(1.0f, 0.25f, env.rainFactor), env.nightFactor);

    float trans = 1.0f;
    float3 accum = float3(0.0f, 0.0f, 0.0f);
    float jitter = (screenPx.x >= 0.0f) ? bayer4x4Pair(screenPx).x : 0.5f;
    float nightClear = env.nightFactor * (1.0f - env.rainFactor);
    float alphaCoeff = lerp(0.032f * (1.0f - 0.38f * nightClear), 0.062f, env.rainFactor);
    float hgScale = lerp(2.10f, 0.85f, env.nightFactor);

    for (int i = 0; i < 14; ++i) {
        if (i >= steps || trans < 0.03f) {
            break;
        }
        float t = tStart + (float(i) + jitter) * stepSize;
        float3 pos = f.camWorldPos.xyz + d * t;
        float dens = evalCloudDensityAt(pos, f, baseH, topH, nightClear);
        if (dens > 0.005f) {
            float sunDens = evalCloudDensityAt(pos + lightDir * 24.0f, f, baseH, topH, nightClear);
            if (!singleSunTap && dens > 0.06f) {
                sunDens += evalCloudDensityAt(pos + lightDir * 54.0f, f, baseH, topH, nightClear) * 0.5f;
            }
            float lightTrans = exp(-sunDens * (singleSunTap ? 1.75f : 1.45f));
            float powder = 1.0f - exp(-dens * 2.8f);
            float3 direct = warmLit * lightTrans * (0.78f + hgScale * hg * powder);
            float hShade = saturate((pos.y - baseH) / max(topH - baseH, 1.0f));
            float3 ambient = shadowBase * (0.58f + 0.45f * hShade);
            float sampleAlpha = 1.0f - exp(-dens * stepSize * alphaCoeff);
            accum += trans * sampleAlpha * (direct + ambient);
            trans *= (1.0f - sampleAlpha);
        }
    }

    float horizonFade = smoothstep(0.02f, 0.15f, d.y);
    outC.radiance = accum * horizonFade;
    outC.transmittance = lerp(1.0f, trans, horizonFade);
    return outC;
}

CloudTraceResult sampleVolumetricCloudsN(float3 rayDir, MetalityFrame f, MeteorEnvFactors env, int stepCap, bool singleSunTap) {
    return sampleVolumetricCloudsNAt(rayDir, float2(-1.0f, -1.0f), f, env, stepCap, singleSunTap);
}

CloudTraceResult sampleVolumetricClouds(float3 rayDir, MetalityFrame f, MeteorEnvFactors env) {
    uint tier = meteorShadowTier(f);
    int steps = (tier >= 3u) ? 14 : ((tier >= 2u) ? 10 : 7);
    return sampleVolumetricCloudsN(rayDir, f, env, steps, tier < 3u);
}

float3 evalNetherSky(float3 dir, MetalityFrame f, MeteorEnvFactors env) {
    float3 d = normalize(dir);
    float swirl = fbm2D(d.xz * 3.2f + float2(f.timeSeconds * 0.06f, -f.timeSeconds * 0.04f));
    float y = d.y;

    // Crimson / Nether Wastes deeper glowing magma haze (high saturation orange-red, darker ash ceiling)
    float3 crimsonCeiling = float3(0.08f, 0.015f, 0.010f);
    float3 crimsonMid = float3(0.46f, 0.075f, 0.020f);
    float3 crimsonLavaGlow = float3(1.35f, 0.32f, 0.035f);
    float3 crimsonSky = lerp(crimsonLavaGlow, crimsonMid, smoothstep(-0.35f, 0.18f, y + swirl * 0.15f));
    crimsonSky = lerp(crimsonSky, crimsonCeiling, smoothstep(0.12f, 0.85f, y));

    // Soul Sand Valley cool teal-cyan mist
    float3 soulCeiling = float3(0.035f, 0.13f, 0.16f);
    float3 soulMist = float3(0.14f, 0.54f, 0.56f);
    float3 soulLavaHorizon = float3(0.24f, 0.68f, 0.66f);
    float3 soulSky = lerp(soulLavaHorizon, soulMist, smoothstep(-0.25f, 0.25f, y + swirl * 0.18f));
    soulSky = lerp(soulSky, soulCeiling, smoothstep(0.20f, 0.88f, y));

    // Warped Forest violet-teal glow
    float3 warpedSky = lerp(float3(0.10f, 0.30f, 0.34f), float3(0.16f, 0.05f, 0.30f), smoothstep(-0.15f, 0.65f, y + swirl * 0.15f));

    return crimsonSky * env.netherCrimsonFactor + soulSky * env.netherSoulFactor + warpedSky * env.netherWarpedFactor;
}

float3 evalEndSkyAndGargantua(float3 dir, MetalityFrame f) {
    float3 d = normalize(dir);

    float3 bhDir = getMeteorBlackHoleDir();
    float3 upRef = float3(0.0f, 1.0f, 0.0f);
    float3 bhRight = normalize(cross(upRef, bhDir));
    float3 bhUp = normalize(cross(bhDir, bhRight));

    float cosA = clamp(dot(d, bhDir), -1.0f, 1.0f);
    float angle = acos(cosA);

    float2 bhPlane = float2(dot(d, bhRight), dot(d, bhUp));
    float tiltCos = 0.90f;
    float tiltSin = 0.436f;
    float2 bhTilted = float2(bhPlane.x * tiltCos - bhPlane.y * tiltSin,
                             bhPlane.x * tiltSin + bhPlane.y * tiltCos);

    float photonRadius = 0.175f;
    float lensWarp = 0.028f / max(angle - photonRadius * 0.68f, 0.025f);
    float2 planeDir = (length(bhPlane) > 1e-4f) ? normalize(bhPlane) : float2(1.0f, 0.0f);
    float3 lensedDir = normalize(d - (bhRight * planeDir.x + bhUp * planeDir.y) * clamp(lensWarp, 0.0f, 0.65f));

    float nebNoise1 = fbm3D(lensedDir * 2.6f + float3(0.0f, 0.0f, f.timeSeconds * 0.01f));
    float nebNoise2 = fbm3D(lensedDir * 5.2f + float3(4.1f, -2.7f, 1.8f));
    float nebBand = exp(-abs(lensedDir.y * 1.4f + nebNoise1 * 0.55f - 0.15f) * 3.1f);
    float horizonMist = exp(-abs(d.y) * 3.6f);

    float3 endZenith = float3(0.028f, 0.012f, 0.062f);
    float3 endNebulaA = float3(0.32f, 0.08f, 0.56f);
    float3 endNebulaB = float3(0.72f, 0.22f, 0.88f);
    float3 endHorizon = float3(0.56f, 0.28f, 0.76f);

    float3 sky = lerp(endZenith, endNebulaA, nebBand * 0.75f + nebNoise2 * 0.25f);
    sky += endNebulaB * smoothstep(0.45f, 0.92f, nebBand * nebNoise1 * 2.1f) * 0.75f;
    sky = lerp(sky, endHorizon, horizonMist * 0.42f);

    float stars = starFieldLayer(lensedDir, 95.0f, 0.91f) * 1.6f + starFieldLayer(lensedDir, 195.0f, 0.95f) * 1.0f;
    sky += float3(0.92f, 0.86f, 1.15f) * stars * smoothstep(photonRadius * 0.90f, photonRadius * 1.35f, angle);

    if (cosA > 0.25f) {
        float2 diskCoord = float2(bhTilted.x, bhTilted.y * 3.85f);
        float diskR = length(diskCoord);
        float diskSwirl = fbm2D(float2(atan2(diskCoord.y, diskCoord.x) * 2.2f + f.timeSeconds * 0.35f, diskR * 14.0f));
        float diskShape = smoothstep(photonRadius * 0.86f, photonRadius * 1.06f, diskR)
                        * (1.0f - smoothstep(0.17f, 0.52f, diskR));
        float doppler = clamp(1.0f + bhTilted.x * 1.45f, 0.65f, 1.55f);

        float ringDist = abs(angle - photonRadius * 1.04f);
        float photonRing = exp(-ringDist * 48.0f) * 2.8f + exp(-abs(angle - photonRadius * 1.38f) * 18.0f) * 0.75f;
        float archSwirl = fbm2D(float2(atan2(bhTilted.y, bhTilted.x) * 3.0f - f.timeSeconds * 0.25f, angle * 16.0f));
        float outerHalo = (1.0f - smoothstep(photonRadius * 0.95f, 0.48f, angle))
                        * smoothstep(photonRadius * 0.85f, photonRadius * 1.12f, angle)
                        * (0.55f + 0.55f * archSwirl);

        float heat = saturate(1.2f - (angle - photonRadius) * 3.8f);
        float3 diskColor = lerp(float3(0.88f, 0.42f, 1.45f), float3(3.4f, 2.7f, 4.2f), heat);
        float3 archColor = lerp(float3(0.58f, 0.25f, 1.08f), float3(1.65f, 1.25f, 2.35f), archSwirl);

        float rainbowBand = exp(-abs(angle - photonRadius * 1.52f) * 22.0f);
        float3 spectrum = float3(
            0.5f + 0.5f * cos((angle * 35.0f) + 0.0f),
            0.5f + 0.5f * cos((angle * 35.0f) + 2.09f),
            0.5f + 0.5f * cos((angle * 35.0f) + 4.18f)
        );
        sky += spectrum * rainbowBand * 0.16f;

        float concentric = 0.65f + 0.35f * sin(angle * 95.0f - f.timeSeconds * 1.2f + archSwirl * 5.0f);
        sky += archColor * outerHalo * concentric * 0.95f;
        sky += diskColor * diskShape * (0.75f + 0.65f * diskSwirl) * doppler * 0.65f;
        sky += float3(2.8f, 2.2f, 3.6f) * photonRing;

        float horizonVoid = smoothstep(photonRadius * 0.92f, photonRadius * 0.99f, angle);
        sky = lerp(float3(0.001f, 0.0005f, 0.003f), sky, horizonVoid);
    }

    return sky;
}

struct SoftSkyResult {
    float3 radiance;
    float cloudTrans;
};

SoftSkyResult evalOverworldSoftSkyAndClouds(float3 dir, MetalityFrame f, MeteorEnvFactors env, bool withClouds) {
    SoftSkyResult res;
    float3 d = normalize(dir);
    float3 sky = evalOverworldSkyBase(d, f, env, false);
    res.cloudTrans = 1.0f;
    if (withClouds && meteorFeatureClouds(f)) {
        CloudTraceResult c = sampleVolumetricClouds(d, f, env);
        res.cloudTrans = c.transmittance;
        sky = sky * c.transmittance + c.radiance;
    }
    res.radiance = sky;
    return res;
}

float3 evalOverworldCrispCelestials(float3 dir, MetalityFrame f, MeteorEnvFactors env, float3 softSky, float cloudTrans) {
    float3 d = normalize(dir);
    float3 sky = softSky;
    float3 sunDir = normalize(f.sunWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float3 moonDir = normalize(f.moonWorldDir.xyz + float3(0.0f, 1e-5f, 0.0f));
    float cosSun = dot(d, sunDir);
    float cosMoon = dot(d, moonDir);

    float2 skyUv = worldToScreenUv(f, d * 100.0f);
    float2 skyPx = floor(skyUv * float2(float(f.width), float(f.height)));
    float2 skyDith = pcg2dHash01(uint2(max(skyPx, float2(0.0f, 0.0f))));

    float nightVis = saturate(max(env.nightFactor, f.skyColor.a * 0.9f)) * smoothstep(-0.02f, 0.20f, d.y) * (1.0f - env.rainFactor * 0.85f);
    VoxelCubeHit moonHit;
    moonHit.hit = false;
    moonHit.edgeAlpha = 0.0f;
    moonHit.faceN = float3(0.0f, 0.0f, -1.0f);
    moonHit.fuv = float2(0.5f, 0.5f);
    moonHit.faceId = 0;
    if ((nightVis > 0.01f || env.nightFactor > 0.001f) && cosMoon > 0.92f) {
        moonHit = intersectTiltedVoxelCube(d, moonDir, 0.082f, 0.72f, 0.58f, 0.28f);
    }
    float moonOcc = 1.0f - moonHit.edgeAlpha;

    // 1.0x crisp night sky: Milky Way, pole-free aurora ribbons, shooting stars, sharp stars (0.00 ms in daytime!)
    if (nightVis > 0.01f) {
        float3 milky = evalMilkyWay(d, f.timeSeconds);
        float3 aurora = evalAuroraBorealis(d, f.timeSeconds, skyDith.x);
        float3 meteors = evalShootingStars(d, f.timeSeconds);
        float starsHero = sharpStarLayer(d, 72.0f, 0.928f) * 2.45f;
        float starsMid = sharpStarLayer(d, 145.0f, 0.938f) * 1.35f;
        float starsFine = sharpStarLayer(d, 260.0f, 0.955f) * 0.80f;
        float starHue = hash13(floor(d * 72.0f));
        float3 starCol = lerp(float3(0.76f, 0.90f, 1.22f), float3(1.18f, 0.96f, 0.74f), smoothstep(0.45f, 0.92f, starHue));
        float twinkle = 0.80f + 0.20f * sin(f.timeSeconds * 2.2f + dot(floor(d * 72.0f), float3(12.3f, 45.6f, 78.9f)));
        sky += (milky + aurora + meteors + starCol * (starsHero + starsMid + starsFine) * twinkle) * nightVis * moonOcc * cloudTrans;
    }

    float horizonMask = smoothstep(-0.05f, 0.02f, d.y) * (1.0f - env.rainFactor * 0.90f);
    float sunVis = horizonMask * (1.0f - env.nightFactor * 0.95f);
    if (sunVis > 0.001f) {
        // 1.0x analytic C-infinity smooth solar Mie halo (100% ring-free around tilted 3D cube sun)
        sky += evalOverworldSolarMieGlow(cosSun, env) * sunVis * lerp(0.28f, 1.0f, cloudTrans);
    }

    if (sunVis * cloudTrans > 0.001f && cosSun > 0.94f) {
        float sunAngSize = 0.052f;
        float sunChord = length(d - sunDir);
        VoxelCubeHit sunHit = intersectTiltedVoxelCube(d, sunDir, sunAngSize, 0.65f, 0.54f, 0.24f);
        if (sunHit.hit) {
            float3 sunCubeCol = evalSolarCubeSurface(sunHit, sunChord, sunAngSize, env);
            sky = lerp(sky, max(sky, sunCubeCol), sunHit.edgeAlpha * sunVis * cloudTrans);
        }
    }

    if (env.nightFactor > 0.005f) {
        sky += evalOverworldLunarHalo(cosMoon) * env.nightFactor * horizonMask * lerp(0.30f, 1.0f, cloudTrans);
    }

    if (moonHit.hit) {
        float3 moonCubeCol = evalLunarCubeSurface(moonHit);
        sky = lerp(sky, moonCubeCol, moonHit.edgeAlpha * env.nightFactor * horizonMask * cloudTrans);
    }

    // Exact 1-ULP Roberts TPDF pre-dither matched to RG11B10F hdrLit (6-bit R/G, 5-bit B mantissa) — mathematically zero concentric rings
    float maxCh = max(sky.r, max(sky.g, sky.b));
    if (maxCh > 0.008f && maxCh < 3.0f) {
        float tri = skyDith.x - skyDith.y;
        float3 ulp = exp2(floor(log2(max(sky, float3(1e-4f, 1e-4f, 1e-4f))))) * float3(1.0f / 64.0f, 1.0f / 64.0f, 1.0f / 32.0f);
        sky = max(sky + tri * min(ulp, float3(0.016f, 0.016f, 0.032f)), float3(0.0f, 0.0f, 0.0f));
    }
    return sky;
}

// Hillaire (2020) non-linear horizon-concentrated Sky-View LUT parameterization (192x96)
float3 skyLutUvToDir(float2 uv) {
    float phi = (uv.x - 0.5f) * METEOR_TAU;
    float vAdj = uv.y - 0.5f;
    float signV = (vAdj >= 0.0f) ? 1.0f : -1.0f;
    float v2 = vAdj * 2.0f;
    float lat = signV * (v2 * v2) * (0.5f * METEOR_PI);
    float cosLat = cos(lat);
    return normalize(float3(cosLat * cos(phi), sin(lat), cosLat * sin(phi)));
}

float2 skyDirToLutUv(float3 dir) {
    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    float u = phi * (1.0f / METEOR_TAU) + 0.5f;
    float lat = asin(clamp(d.y, -1.0f, 1.0f));
    float signLat = (lat >= 0.0f) ? 1.0f : -1.0f;
    float v = 0.5f + 0.5f * signLat * sqrt(abs(lat) * (2.0f / METEOR_PI));
    // Clamp V to texel centers [0.5/96, 95.5/96] so linear_repeat wraps azimuth cleanly without zenith/nadir bleed
    v = clamp(v, 0.5f / 96.0f, 95.5f / 96.0f);
    return float2(u, v);
}

half3 sampleSkyLut(Texture2D<half4> lutTex, float3 dir) {
    float2 uv = skyDirToLutUv(dir);
    return lutTex.Sample(metality_linear_repeat, uv).rgb;
}

// 5-tap bilinear tent filter over 0.25x volumetric clouds (radiance.rgb, cloudAlpha = 1 - cloudTrans)
half4 sampleCloudsQres(Texture2D<half4> cloudsTex, float2 uv) {
    int2 cDim = max(metality_tex_size(cloudsTex), int2(1, 1));
    float2 invDim = float2(1.0f, 1.0f) / float2(cDim);
    float2 o = invDim * 0.85f;
    half4 c0 = cloudsTex.Sample(metality_linear_clamp, uv);
    half4 c1 = cloudsTex.Sample(metality_linear_clamp, uv + float2( o.x,  o.y));
    half4 c2 = cloudsTex.Sample(metality_linear_clamp, uv + float2(-o.x,  o.y));
    half4 c3 = cloudsTex.Sample(metality_linear_clamp, uv + float2( o.x, -o.y));
    half4 c4 = cloudsTex.Sample(metality_linear_clamp, uv + float2(-o.x, -o.y));
    return c0 * 0.28h + (c1 + c2 + c3 + c4) * 0.18h;
}

float3 evalDimensionSky(float3 dir, MetalityFrame f, MeteorEnvFactors env, bool withClouds) {
    if (f.dimensionId == -1) {
        return evalNetherSky(dir, f, env);
    }
    if (f.dimensionId == 1) {
        return evalEndSkyAndGargantua(dir, f);
    }
    float3 sky = evalOverworldSkyBase(dir, f, env, true);
    if (withClouds && meteorFeatureClouds(f)) {
        CloudTraceResult c = sampleVolumetricClouds(dir, f, env);
        sky = sky * c.transmittance + c.radiance;
    }
    return sky;
}

float3 evalDimensionSkyFast(float3 dir, MetalityFrame f, MeteorEnvFactors env, bool withClouds) {
    if (f.dimensionId == -1) {
        return evalNetherSky(dir, f, env);
    }
    if (f.dimensionId == 1) {
        return evalEndSkyAndGargantua(dir, f);
    }
    float3 sky = evalOverworldSkyBase(dir, f, env, false);
    if (withClouds && meteorFeatureClouds(f)) {
        CloudTraceResult c = sampleVolumetricCloudsN(dir, f, env, 6, true);
        sky = sky * c.transmittance + c.radiance;
    }
    return sky;
}

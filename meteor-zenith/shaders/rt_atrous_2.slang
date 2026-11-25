// Meteor Shader Pass 2c2 — Pure FP16 Cached 0.5x G-Buf Edge-Stopping À-Trous Step 2 (step=3, 0.5x MRT: GI + Tinted SunVis)
module meteor_rt_atrous_2;
import metality;

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> rtGiStep1;
Texture2D<half4> rtGbufCurr;
Texture2D<half4> rtSunStep1;

struct RtAtrous2Out {
    float4 gi : SV_Target0;
    float4 sun : SV_Target1;
};

[shader("fragment")]
RtAtrous2Out fsRtAtrous2(float4 pixelPos : SV_Position) {
    RtAtrous2Out o;
    int2 rawDim = max(metality_tex_size(rtGiStep1), int2(1, 1));
    int2 maxPx = rawDim - int2(1, 1);
    int2 rawPx = clamp(int2(pixelPos.xy), int2(0, 0), maxPx);

    half4 centerVal = rtGiStep1.Load(int3(rawPx, 0));
    half3 centerSun = max(rtSunStep1.Load(int3(rawPx, 0)).rgb, half3(0.0h, 0.0h, 0.0h));
    half4 centerGb = rtGbufCurr.Load(int3(rawPx, 0));
    half cZ = centerGb.x;
    if (cZ <= 0.005h || centerVal.a < -0.5h) {
        o.gi = float4(centerVal);
        o.sun = float4(float3(centerSun), 1.0f);
        return o;
    }

    half3 cN = centerGb.yzw;
    half zTol = (cN.y > 0.85h) ? max(0.16h + cZ * 0.040h, 0.16h) : max(0.090h + cZ * 0.022h, 0.090h);
    half invZTol = 1.0h / zTol;
    half maxDz = zTol * 2.2h;

    half4 accum = centerVal * 0.55h;
    half3 sunAccum = centerSun * 0.55h;
    half wSum = 0.55h;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 nPx = clamp(rawPx + int2(dx * 3, dy * 3), int2(0, 0), maxPx);
            half4 sGb = rtGbufCurr.Load(int3(nPx, 0));
            half sZ = sGb.x;
            if (sZ <= 0.005h) continue;
            half dz = abs(sZ - cZ);
            if (dz > maxDz) continue;
            half nDot = saturate(dot(cN, sGb.yzw));
            if (nDot <= 0.84h) continue;
            half4 sVal = rtGiStep1.Load(int3(nPx, 0));
            if (sVal.a < -0.5h) continue;

            half zNorm = min(dz * invZTol, 4.0h);
            half n2 = nDot * nDot;
            half spatialW = (dx == 0 || dy == 0) ? 0.85h : 0.60h;
            half w = spatialW * exp(-(zNorm * zNorm)) * (n2 * n2);
            accum += sVal * w;
            sunAccum += max(rtSunStep1.Load(int3(nPx, 0)).rgb, half3(0.0h, 0.0h, 0.0h)) * w;
            wSum += w;
        }
    }

    half invW = 1.0h / max(wSum, 1e-4h);
    o.gi = float4(accum * invW);
    o.sun = float4(max(float3(sunAccum * invW), float3(0.0f)), 1.0f);
    return o;
}

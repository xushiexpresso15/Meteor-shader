// Meteor Shader Pass 2c — Pure FP16 Cached 0.5x G-Buf Edge-Stopping À-Trous Step 1 (step=1, 0.5x MRT: GI + Tinted SunVis)
module meteor_rt_atrous_1;
import metality;

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> rtGiTemporal;
Texture2D<half4> rtGbufCurr;
Texture2D<half4> rtSunTemporal;

struct RtAtrous1Out {
    float4 gi : SV_Target0;
    float4 sun : SV_Target1;
};

[shader("fragment")]
RtAtrous1Out fsRtAtrous1(float4 pixelPos : SV_Position) {
    RtAtrous1Out o;
    int2 rawDim = max(metality_tex_size(rtGiTemporal), int2(1, 1));
    int2 maxPx = rawDim - int2(1, 1);
    int2 rawPx = clamp(int2(pixelPos.xy), int2(0, 0), maxPx);

    half4 centerVal = rtGiTemporal.Load(int3(rawPx, 0));
    half3 centerSun = max(rtSunTemporal.Load(int3(rawPx, 0)).rgb, half3(0.0h, 0.0h, 0.0h));
    half4 centerGb = rtGbufCurr.Load(int3(rawPx, 0));
    half cZ = centerGb.x;
    if (cZ <= 0.005h || centerVal.a < -0.5h) {
        o.gi = float4(centerVal);
        o.sun = float4(float3(centerSun), 1.0f);
        return o;
    }

    half3 cN = centerGb.yzw;
    half zTol = (cN.y > 0.85h) ? max(0.11h + cZ * 0.026h, 0.11h) : max(0.060h + cZ * 0.015h, 0.060h);
    half invZTol = 1.0h / zTol;
    half maxDz = zTol * 2.2h;

    half4 accum = centerVal * 0.55h;
    half3 sunAccum = centerSun * 0.55h;
    half wSum = 0.55h;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 nPx = clamp(rawPx + int2(dx, dy), int2(0, 0), maxPx);
            half4 sGb = rtGbufCurr.Load(int3(nPx, 0));
            half sZ = sGb.x;
            if (sZ <= 0.005h) continue;
            half dz = abs(sZ - cZ);
            if (dz > maxDz) continue;
            half nDot = saturate(dot(cN, sGb.yzw));
            if (nDot <= 0.84h) continue;
            half4 sVal = rtGiTemporal.Load(int3(nPx, 0));
            if (sVal.a < -0.5h) continue;

            half zNorm = min(dz * invZTol, 4.0h);
            half n2 = nDot * nDot;
            half spatialW = (dx == 0 || dy == 0) ? 0.85h : 0.60h;
            half w = spatialW * exp(-(zNorm * zNorm)) * (n2 * n2);
            accum += sVal * w;
            sunAccum += max(rtSunTemporal.Load(int3(nPx, 0)).rgb, half3(0.0h, 0.0h, 0.0h)) * w;
            wSum += w;
        }
    }

    half invW = 1.0h / max(wSum, 1e-4h);
    o.gi = float4(accum * invW);
    o.sun = float4(max(float3(sunAccum * invW), float3(0.0f)), 1.0f);
    return o;
}

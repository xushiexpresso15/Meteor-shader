// Meteor Dusk Pass 2b — Pure FP16 Cached 0.5x G-Buf 3x3 Joint-Bilateral À-Trous Step 2 (step=2, 1-2-1 B-Spline, 0.5x -> 0.5x)
module meteor_dusk_lighting_blur_2;
import metality;

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> duskLightingStep1;
Texture2D<half4> duskSunStep1;
Texture2D<half4> duskHalfGbuf;

struct DuskLightingBlur2Out {
    float4 col0 : SV_Target0;
    float4 col1 : SV_Target1;
};

[shader("fragment")]
DuskLightingBlur2Out fsDuskLightingBlur2(float4 pixelPos : SV_Position) {
    DuskLightingBlur2Out o;
    int2 rawDim = max(metality_tex_size(duskLightingStep1), int2(1, 1));
    int2 maxPx = rawDim - int2(1, 1);
    int2 rawPx = clamp(int2(pixelPos.xy), int2(0, 0), maxPx);

    half4 centerVal = duskLightingStep1.Load(int3(rawPx, 0));
    half4 centerSun = duskSunStep1.Load(int3(rawPx, 0));
    half4 centerGb = duskHalfGbuf.Load(int3(rawPx, 0));
    half cZ = centerGb.x;
    if (cZ <= 0.005h || centerVal.a < -0.5h) {
        o.col0 = float4(centerVal);
        o.col1 = float4(max(float3(centerSun.rgb), float3(0.0f)), 1.0f);
        return o;
    }

    half3 cN = centerGb.yzw;
    half zTol = (cN.y > 0.85h) ? max(0.16h + cZ * 0.040h, 0.16h) : max(0.095h + cZ * 0.024h, 0.095h);
    half invZTol = 1.0h / zTol;
    half maxDz = zTol * 2.2h;

    half4 accum = centerVal * 1.00h;
    half3 sunAccum = centerSun.rgb * 1.00h;
    half wSum = 1.00h;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int2 nPx = clamp(rawPx + int2(dx * 2, dy * 2), int2(0, 0), maxPx);
            half4 sGb = duskHalfGbuf.Load(int3(nPx, 0));
            half sZ = sGb.x;
            if (sZ <= 0.005h) continue;
            half dz = abs(sZ - cZ);
            if (dz > maxDz) continue;
            half nDot = saturate(dot(cN, sGb.yzw));
            if (nDot <= 0.84h) continue;
            half4 sVal = duskLightingStep1.Load(int3(nPx, 0));
            if (sVal.a < -0.5h) continue;

            half zNorm = min(dz * invZTol, 4.0h);
            half n2 = nDot * nDot;
            half spatialW = (dx == 0 || dy == 0) ? 0.50h : 0.25h;
            half w = spatialW * exp(-(zNorm * zNorm)) * (n2 * n2);
            accum += sVal * w;
            sunAccum += duskSunStep1.Load(int3(nPx, 0)).rgb * w;
            wSum += w;
        }
    }

    half invW = 1.0h / max(wSum, 1e-4h);
    o.col0 = float4(accum * invW);
    o.col1 = float4(max(float3(sunAccum * invW), float3(0.0f)), 1.0f);
    return o;
}

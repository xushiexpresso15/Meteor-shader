// Meteor Shader Pass 4a — FP16 Karis-Weighted Half-Res HDR Bloom Threshold Downsample (1.0x -> 0.5x)
module meteor_bloom_downsample;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> hdrVolumetricTex;

half3 karisSoftBrightHalf(half3 raw, half threshold) {
    half3 c = clamp(raw, half3(0.0h, 0.0h, 0.0h), half3(512.0h, 512.0h, 512.0h));
    half lum = max(c.r, max(c.g, c.b));
    half soft = clamp(lum - threshold + 0.48h, 0.0h, 0.96h);
    soft = (soft * soft) * (1.0h / (4.0h * 0.48h + 1e-4h));
    half contrib = max(soft, lum - threshold) / max(lum, 1e-4h);
    half3 res = c * max(contrib, 0.0h);
    half rLum = dot(res, half3(0.2126h, 0.7152h, 0.0722h));
    return res / (1.0h + rLum * 0.18h);
}

[shader("fragment")]
float4 fsBloomDownsample(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    int2 srcDim = max(metality_tex_size(hdrVolumetricTex), int2(1, 1));
    float2 srcTexel = float2(1.0f, 1.0f) / float2(srcDim);
    float bloomThresh = clamp(uFrame.tonemapParams.w, 0.40f, 3.00f);
    half threshold = half((uFrame.dimensionId == -1) ? (bloomThresh * 0.52f) : bloomThresh);

    half3 c0 = karisSoftBrightHalf(hdrVolumetricTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb, threshold);
    half3 c1 = karisSoftBrightHalf(hdrVolumetricTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-1.5f, -1.5f) * srcTexel, 0.001f, 0.999f), 0.0f).rgb, threshold);
    half3 c2 = karisSoftBrightHalf(hdrVolumetricTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( 1.5f, -1.5f) * srcTexel, 0.001f, 0.999f), 0.0f).rgb, threshold);
    half3 c3 = karisSoftBrightHalf(hdrVolumetricTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-1.5f,  1.5f) * srcTexel, 0.001f, 0.999f), 0.0f).rgb, threshold);
    half3 c4 = karisSoftBrightHalf(hdrVolumetricTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( 1.5f,  1.5f) * srcTexel, 0.001f, 0.999f), 0.0f).rgb, threshold);

    half3 bloom = c0 * 0.36h + (c1 + c2 + c3 + c4) * 0.16h;
    return float4(max(float3(bloom), float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

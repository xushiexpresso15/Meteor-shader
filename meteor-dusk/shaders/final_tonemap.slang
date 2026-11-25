// Meteor Shader Pass 5 — FP32 AgX Punchy Filmic Tonemap & sRGB Output (Ring-Free Full-Precision Curve)
module meteor_final_tonemap;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> hdrLitTex;
Texture2D<half4> bloomBlurTex;
Texture2D<half4> godraysShaftTex;
Texture2D<float4> sceneDepthTex;

[shader("fragment")]
float4 fsFinalTonemap(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    float3 hdr = float3(hdrLitTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb);
    float bloomStrengthF = clamp(uFrame.userParams.x, 0.0f, 2.0f);
    bool useBloom = meteorFeatureBloom(uFrame) && (bloomStrengthF > 0.005f);
    bool useGodrays = meteorFeatureGodrays(uFrame);
    float3 bloom = useBloom ? float3(bloomBlurTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb) : float3(0.0f, 0.0f, 0.0f);
    float3 shafts = float3(0.0f, 0.0f, 0.0f);
    if (useGodrays) {
        int2 gDim = max(metality_tex_size(godraysShaftTex), int2(1, 1));
        float2 gOff = (float2(1.0f, 1.0f) / float2(gDim)) * 0.85f;
        half3 s0 = godraysShaftTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb;
        half3 s1 = godraysShaftTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( gOff.x,  gOff.y), 0.001f, 0.999f), 0.0f).rgb;
        half3 s2 = godraysShaftTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-gOff.x,  gOff.y), 0.001f, 0.999f), 0.0f).rgb;
        half3 s3 = godraysShaftTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( gOff.x, -gOff.y), 0.001f, 0.999f), 0.0f).rgb;
        half3 s4 = godraysShaftTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-gOff.x, -gOff.y), 0.001f, 0.999f), 0.0f).rgb;
        shafts = float3(s0 * 0.28h + (s1 + s2 + s3 + s4) * 0.18h);
    }
    if (!isfinite(hdr.r) || !isfinite(hdr.g) || !isfinite(hdr.b)) {
        hdr = float3(0.0f, 0.0f, 0.0f);
    }
    if (!isfinite(bloom.r) || !isfinite(bloom.g) || !isfinite(bloom.b)) {
        bloom = float3(0.0f, 0.0f, 0.0f);
    }
    if (!isfinite(shafts.r) || !isfinite(shafts.g) || !isfinite(shafts.b)) {
        shafts = float3(0.0f, 0.0f, 0.0f);
    }

    if (uFrame.dimensionId == 1 && luminance(hdr) < 0.007f) {
        bloom *= 0.05f;
    }
    hdr += shafts + bloom * (0.30f * bloomStrengthF);

    float2 vigDelta = uv - 0.5f;
    float vignette = clamp(1.0f - dot(vigDelta, vigDelta) * 0.28f, 0.78f, 1.0f);
    float expUser = clamp(uFrame.tonemapParams.x, 0.40f, 2.00f);
    float exposure = expUser * ((uFrame.dimensionId == -1) ? 1.12f : ((uFrame.dimensionId == 1) ? 1.10f : 1.00f));
    hdr = clamp(hdr * (exposure * vignette), float3(0.0f, 0.0f, 0.0f), float3(2048.0f, 2048.0f, 2048.0f));

    // Full-FP32 AgX curve avoids FP16 catastrophic cancellation in the 6th-order polynomial (-40.14*x^5 + 31.96*x^4)
    float3 graded = meteorAgxTonemap(hdr, uFrame);
    float3 srgbOut = pow(max(graded, float3(1e-6f, 1e-6f, 1e-6f)), float3(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f));
    float d1 = interleavedGradientNoise(pixelPos.xy, uFrame.camWorldPos.w);
    float d2 = interleavedGradientNoise(pixelPos.xy + float2(13.0f, 29.0f), uFrame.camWorldPos.w);
    float ditherMask = step(0.002f, max(srgbOut.r, max(srgbOut.g, srgbOut.b)));
    srgbOut = saturate(srgbOut + ((d1 - d2) * (0.5f / 255.0f)) * ditherMask);
    return float4(srgbOut, 1.0f);
}

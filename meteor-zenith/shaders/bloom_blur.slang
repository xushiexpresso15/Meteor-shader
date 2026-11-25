// Meteor Shader Pass 4d — FP16 Multi-Octave Wide Bloom Tent Combine (0.5x + 0.125x + 0.03125x -> 0.25x)
module meteor_bloom_blur;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> bloomHalfTex;
Texture2D<half4> bloomEighthTex;
Texture2D<half4> bloom32Tex;

[shader("fragment")]
float4 fsBloomBlur(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    float2 qPx = float2(1.0f / max(float(uFrame.width), 1.0f), 1.0f / max(float(uFrame.height), 1.0f));

    // Octave 1 (tight 0.25x isotropic diagonal tent — zero horizontal/vertical torch streaks)
    float2 o1 = qPx * 0.5f;
    half3 m1 = bloomHalfTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb * 0.28h;
    m1 += bloomHalfTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o1.x, -o1.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m1 += bloomHalfTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o1.x, -o1.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m1 += bloomHalfTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o1.x,  o1.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m1 += bloomHalfTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o1.x,  o1.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;

    // Octave 2 (medium 0.125x isotropic diagonal tent)
    float2 o2 = qPx * 1.5f;
    half3 m2 = bloomEighthTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb * 0.28h;
    m2 += bloomEighthTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o2.x, -o2.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m2 += bloomEighthTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o2.x, -o2.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m2 += bloomEighthTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o2.x,  o2.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m2 += bloomEighthTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o2.x,  o2.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;

    // Octave 3 (restrained 0.03125x isotropic diagonal atmospheric bloom veil)
    float2 o3 = qPx * 4.5f;
    half3 m3 = bloom32Tex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb * 0.28h;
    m3 += bloom32Tex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o3.x, -o3.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m3 += bloom32Tex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o3.x, -o3.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m3 += bloom32Tex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-o3.x,  o3.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;
    m3 += bloom32Tex.SampleLevel(metality_linear_clamp, clamp(uv + float2( o3.x,  o3.y), 0.001f, 0.999f), 0.0f).rgb * 0.18h;

    half3 combined = m1 * 0.46h + m2 * 0.40h + m3 * 0.14h;
    return float4(max(float3(combined), float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

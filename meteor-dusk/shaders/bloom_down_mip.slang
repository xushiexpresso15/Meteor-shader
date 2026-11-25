// Meteor Shader Pass 4b/4c — FP16 Progressive 4x Downsample Pyramid Mip (0.5x -> 0.125x -> 0.03125x)
module meteor_bloom_down_mip;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<half4> srcMipTex;

[shader("fragment")]
float4 fsBloomDownMip(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    int2 srcDim = max(metality_tex_size(srcMipTex), int2(1, 1));
    float2 st = float2(1.0f, 1.0f) / float2(srcDim);

    half3 sum = srcMipTex.SampleLevel(metality_linear_clamp, uv, 0.0f).rgb * 0.20h;
    sum += srcMipTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-st.x, -st.y), 0.001f, 0.999f), 0.0f).rgb * 0.20h;
    sum += srcMipTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( st.x, -st.y), 0.001f, 0.999f), 0.0f).rgb * 0.20h;
    sum += srcMipTex.SampleLevel(metality_linear_clamp, clamp(uv + float2(-st.x,  st.y), 0.001f, 0.999f), 0.0f).rgb * 0.20h;
    sum += srcMipTex.SampleLevel(metality_linear_clamp, clamp(uv + float2( st.x,  st.y), 0.001f, 0.999f), 0.0f).rgb * 0.20h;

    return float4(max(float3(sum), float3(0.0f, 0.0f, 0.0f)), 1.0f);
}

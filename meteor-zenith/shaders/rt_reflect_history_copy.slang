// Meteor Shader Pass 2d — Persistent Reflection History Copy (0.5x)
module meteor_rt_reflect_history_copy;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> rtReflectTemporal;

[shader("fragment")]
float4 fsRtReflectHistoryCopy(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    return rtReflectTemporal.SampleLevel(metality_nearest_clamp, uv, 0.0f);
}

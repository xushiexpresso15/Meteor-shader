// Meteor Shader Pass 2b — Exact-Texel Persistent Temporal History Copy (0.5x MRT: GI + Tinted SunVis)
module meteor_rt_history_copy;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> rtGiTemporal;
Texture2D<float4> rtSunTemporal;

struct RtHistoryCopyOut {
    float4 gi : SV_Target0;
    float4 sun : SV_Target1;
};

[shader("fragment")]
RtHistoryCopyOut fsRtHistoryCopy(float4 pixelPos : SV_Position) {
    RtHistoryCopyOut o;
    int2 dim = max(metality_tex_size(rtGiTemporal), int2(1, 1));
    int2 px = clamp(int2(pixelPos.xy), int2(0, 0), dim - int2(1, 1));
    o.gi = rtGiTemporal.Load(int3(px, 0));
    o.sun = rtSunTemporal.Load(int3(px, 0));
    return o;
}

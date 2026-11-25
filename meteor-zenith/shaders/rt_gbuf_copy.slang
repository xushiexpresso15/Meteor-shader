// Meteor Shader Pass 2b2 — Persistent Half-Res G-Buffer History (ViewZ, WorldNormal.xyz)
module meteor_rt_gbuf_copy;
import metality;

#include "lib/meteor_common.slang"

ConstantBuffer<MetalityFrame> uFrame;
Texture2D<float4> sceneDepthTex;

[shader("fragment")]
float4 fsRtGbufCopy(float4 pixelPos : SV_Position) : SV_Target0 {
    float2 uv = safeScreenUv(pixelPos.xy, uFrame, 1.0f);
    MeteorDepthSample ds = loadDepthAtUv(sceneDepthTex, uv);
    if (isSkyDepth(ds.depth)) {
        return float4(0.0f, 0.0f, 1.0f, 0.0f);
    }
    float3 viewPos = reconstructViewPos(uFrame, ds.centerUv, ds.depth);
    float3 relPos = mul(uFrame.invViewRot, float4(viewPos, 0.0f)).xyz;
    float3 nrm = reconstructFastNormal(uFrame, ds.centerUv, relPos, sceneDepthTex);
    float viewZ = max(-viewPos.z, 0.01f);
    return float4(viewZ, nrm.x, nrm.y, nrm.z);
}

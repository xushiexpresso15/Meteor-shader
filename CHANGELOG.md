# Changelog

All notable changes to **Meteor Shader** (`Meteor Dusk` & `Meteor Zenith`) are documented here.

## [0.1.0-beta.1] - 2026-11-25

First public beta release synchronized with **Metality Engine `v0.1.0-beta.1`**.

### Meteor Dusk (`meteor-dusk-0.1.0-beta.1.zip`)
- 2-cascade texel-snapped PCSS shadows, screen-space contact shadows, multi-direction GTAO + SSGI, and hierarchical SSR.
- Full-resolution (`1.0x`) crisp entity sun shadows while keeping terrain shadows soft.
- Analytical soft-penumbra point-light shadows and smooth falloff for torches, lava, and glow lichen (with cave lava-fall flicker eliminated).
- Under-roof voxel sky-openness & cave sun-leak guards for dark sealed rooms and deep caves.

### Meteor Zenith (`meteor-zenith-0.1.0-beta.1.zip`)
- Metal Hardware BVH ray tracing on Apple M3–M5 (`512x128x512` voxel DDA fallback on M1–M2) with 1-bounce colored GI and temporal/À-Trous denoising.
- 3D ray-traced water refraction, caustics, and reflections, plus deep-water attenuation on all profiles (`LOW`–`ULTRA`).

### Shared Core Aesthetics (`Meteor Dusk` & `Meteor Zenith`)
- `RGBA16F` sky & volumetric godray pipeline with FP32 AgX tonemapping (zero sun banding or thin-glass godray glitches).
- Balanced daytime sun & nighttime moon lighting with soft cloud occlusion.
- Full-resolution (`1.0x`) Milky Way, northern aurora, and moonlit clouds.

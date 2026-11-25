import {
  Format,
  PipelineConfig,
  RendererConfig,
  StorageHint,
  defineOption,
  defineProfile,
} from "../../sdk/src/index";

/**
 * Meteor Dusk v0.1.0-beta.1 — High-Performance Non-Ray-Tracing Apple Silicon Edition
 * Full Meteor aesthetic powered purely by 2-Cascade Texel-Snapped PCSS Shadow Maps,
 * Screen-Space Contact Shadows (SSCS), Multi-Horizon GTAO + SSGI Color Bounce,
 * Hierarchical Binary-Refined SSR, Deferred Clustered Point Lights, 3D Volumetric Sky & Water.
 */
defineOption("BLOOM_STRENGTH", 0.78, [0.25, 0.50, 0.78, 1.05, 1.40], true, "PostProcessing");
defineOption("BLOOM_THRESHOLD", 1.35, [0.85, 1.10, 1.35, 1.65, 2.0], true, "PostProcessing");
defineOption("EXPOSURE", 0.86, [0.70, 0.80, 0.86, 1.00, 1.20], true, "ColorGrading");
defineOption("CONTRAST", 1.12, [0.95, 1.05, 1.12, 1.20, 1.32], true, "ColorGrading");
defineOption("SATURATION", 1.16, [0.90, 1.02, 1.16, 1.28, 1.40], true, "ColorGrading");
defineOption("HIGHLIGHT_BOOST", 1.12, [0.90, 1.02, 1.12, 1.22, 1.40], true, "Lighting");
defineOption("SHADOW_DEPTH", 1.02, [0.80, 0.92, 1.02, 1.15, 1.35], true, "Lighting");
defineOption("EMISSIVE_INTENSITY", 1.12, [0.65, 0.85, 1.12, 1.40, 1.80], true, "Lighting");
defineOption("ATMOS_HAZE", 0.65, [0.20, 0.45, 0.65, 0.90, 1.25], true, "Atmosphere");
defineOption("GODRAY_STRENGTH", 0.95, [0.0, 0.5, 0.95, 1.25, 1.6], true, "Atmosphere");
defineOption("WATER_CLARITY", 1.0, [0.5, 0.8, 1.0, 1.3, 1.6], true, "Water");
defineOption("WATER_WAVE_HEIGHT", 1.15, [0.4, 0.8, 1.15, 1.5, 2.0], true, "Water");
defineOption("WATER_RIPPLE_STRENGTH", 1.15, [0.0, 0.6, 1.15, 1.5, 2.0], true, "Water");
defineOption("SPECULAR_STRENGTH", 1.15, [0.4, 0.8, 1.15, 1.5, 2.0], true, "PBR");
defineOption("REFLECTION_STRENGTH", 1.10, [0.0, 0.4, 0.8, 1.10, 1.4, 1.8], true, "PBR");
defineOption("SHADOW_QUALITY", 1.0, [0.5, 0.85, 1.0, 1.5, 2.0], true, "Shadows");
defineOption("ENABLE_SHADOWS", true, [false, true], false, "Performance");
defineOption("ENABLE_REFLECTIONS", true, [false, true], false, "Performance");
defineOption("ENABLE_WATER_SHADER", true, [false, true], false, "Performance");
defineOption("ENABLE_VOLUMETRIC_CLOUDS", true, [false, true], false, "Performance");
defineOption("ENABLE_GODRAYS", true, [false, true], false, "Performance");
defineOption("ENABLE_BLOOM", true, [false, true], false, "Performance");
defineOption("ENABLE_GI_AO", true, [false, true], false, "Performance");
defineOption("ENABLE_PBR_RAIN", true, [false, true], false, "Performance");

defineProfile("LOW", {
  BLOOM_STRENGTH: "0.0",
  BLOOM_THRESHOLD: "1.45",
  EXPOSURE: "0.90",
  CONTRAST: "1.08",
  SATURATION: "1.10",
  HIGHLIGHT_BOOST: "1.05",
  SHADOW_DEPTH: "0.95",
  EMISSIVE_INTENSITY: "1.05",
  ATMOS_HAZE: "0.45",
  GODRAY_STRENGTH: "0.0",
  WATER_CLARITY: "0.90",
  WATER_WAVE_HEIGHT: "0.80",
  WATER_RIPPLE_STRENGTH: "0.0",
  SPECULAR_STRENGTH: "0.40",
  REFLECTION_STRENGTH: "0.0",
  SHADOW_QUALITY: "0.5",
  ENABLE_SHADOWS: "false",
  ENABLE_REFLECTIONS: "false",
  ENABLE_WATER_SHADER: "false",
  ENABLE_VOLUMETRIC_CLOUDS: "false",
  ENABLE_GODRAYS: "false",
  ENABLE_BLOOM: "false",
  ENABLE_GI_AO: "false",
  ENABLE_PBR_RAIN: "false",
});

defineProfile("MEDIUM", {
  BLOOM_STRENGTH: "0.68",
  BLOOM_THRESHOLD: "1.35",
  EXPOSURE: "0.88",
  CONTRAST: "1.10",
  SATURATION: "1.14",
  HIGHLIGHT_BOOST: "1.10",
  SHADOW_DEPTH: "1.00",
  EMISSIVE_INTENSITY: "1.12",
  ATMOS_HAZE: "0.60",
  GODRAY_STRENGTH: "0.80",
  WATER_CLARITY: "1.00",
  WATER_WAVE_HEIGHT: "1.05",
  WATER_RIPPLE_STRENGTH: "1.00",
  SPECULAR_STRENGTH: "0.90",
  REFLECTION_STRENGTH: "0.80",
  SHADOW_QUALITY: "0.85",
  ENABLE_SHADOWS: "true",
  ENABLE_REFLECTIONS: "false",
  ENABLE_WATER_SHADER: "true",
  ENABLE_VOLUMETRIC_CLOUDS: "true",
  ENABLE_GODRAYS: "true",
  ENABLE_BLOOM: "true",
  ENABLE_GI_AO: "false",
  ENABLE_PBR_RAIN: "false",
});

defineProfile("HIGH", {
  BLOOM_STRENGTH: "0.82",
  BLOOM_THRESHOLD: "1.32",
  EXPOSURE: "0.90",
  CONTRAST: "1.12",
  SATURATION: "1.18",
  HIGHLIGHT_BOOST: "1.12",
  SHADOW_DEPTH: "1.02",
  EMISSIVE_INTENSITY: "1.16",
  ATMOS_HAZE: "0.65",
  GODRAY_STRENGTH: "1.00",
  WATER_CLARITY: "1.15",
  WATER_WAVE_HEIGHT: "1.25",
  WATER_RIPPLE_STRENGTH: "1.20",
  SPECULAR_STRENGTH: "1.25",
  REFLECTION_STRENGTH: "1.20",
  SHADOW_QUALITY: "1.0",
  ENABLE_SHADOWS: "true",
  ENABLE_REFLECTIONS: "true",
  ENABLE_WATER_SHADER: "true",
  ENABLE_VOLUMETRIC_CLOUDS: "true",
  ENABLE_GODRAYS: "true",
  ENABLE_BLOOM: "true",
  ENABLE_GI_AO: "true",
  ENABLE_PBR_RAIN: "true",
});

defineProfile("ULTRA", {
  BLOOM_STRENGTH: "0.92",
  BLOOM_THRESHOLD: "1.28",
  EXPOSURE: "0.92",
  CONTRAST: "1.14",
  SATURATION: "1.20",
  HIGHLIGHT_BOOST: "1.16",
  SHADOW_DEPTH: "1.04",
  EMISSIVE_INTENSITY: "1.22",
  ATMOS_HAZE: "0.70",
  GODRAY_STRENGTH: "1.15",
  WATER_CLARITY: "1.30",
  WATER_WAVE_HEIGHT: "1.35",
  WATER_RIPPLE_STRENGTH: "1.30",
  SPECULAR_STRENGTH: "1.35",
  REFLECTION_STRENGTH: "1.30",
  SHADOW_QUALITY: "1.5",
  ENABLE_SHADOWS: "true",
  ENABLE_REFLECTIONS: "true",
  ENABLE_WATER_SHADER: "true",
  ENABLE_VOLUMETRIC_CLOUDS: "true",
  ENABLE_GODRAYS: "true",
  ENABLE_BLOOM: "true",
  ENABLE_GI_AO: "true",
  ENABLE_PBR_RAIN: "true",
});

export function configureRenderer(config: RendererConfig): void {
  config.disableShade = true;
  config.ambientOcclusionLevel = 1.0;

  config.shadow.enabled = true;
  config.shadow.resolution = 1536;
  config.shadow.cascades = 2;
  config.shadow.gpuDrivenCulling = true;

  config.appleSilicon.enableEDR = true;
  config.appleSilicon.enableHardwareRayTracing = false;
  config.appleSilicon.thermalAdaptiveScaling = true;
}

export function configurePipeline(pipeline: PipelineConfig): void {
  const skyViewLut = pipeline
    .createTexture("sky_view_lut")
    .format(Format.RGBA16F)
    .size(192, 96)
    .storage(StorageHint.Persistent)
    .build();

  const skyReflLut = pipeline
    .createTexture("sky_refl_lut")
    .format(Format.RGBA16F)
    .size(192, 96)
    .storage(StorageHint.Persistent)
    .build();

  const cloudsQres = pipeline
    .createTexture("clouds_qres")
    .format(Format.RGBA16F)
    .scale(0.25)
    .storage(StorageHint.Persistent)
    .build();

  const duskLightingRaw = pipeline
    .createTexture("dusk_lighting_raw")
    .format(Format.RGBA16F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskSunRaw = pipeline
    .createTexture("dusk_sun_raw")
    .format(Format.RG11B10F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskHalfGbuf = pipeline
    .createTexture("dusk_half_gbuf")
    .format(Format.RGBA16F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskLightingStep1 = pipeline
    .createTexture("dusk_lighting_step1")
    .format(Format.RGBA16F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskSunStep1 = pipeline
    .createTexture("dusk_sun_step1")
    .format(Format.RG11B10F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskLightingFiltered = pipeline
    .createTexture("dusk_lighting_filtered")
    .format(Format.RGBA16F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskSunFiltered = pipeline
    .createTexture("dusk_sun_filtered")
    .format(Format.RG11B10F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const duskReflRaw = pipeline
    .createTexture("dusk_refl_raw")
    .format(Format.RGBA16F)
    .scale(0.5)
    .storage(StorageHint.Persistent)
    .build();

  const hdrLit = pipeline
    .createTexture("dusk_hdr_lit")
    .format(Format.RG11B10F)
    .scale(1.0)
    .storage(StorageHint.AutoTileOptimal)
    .build();

  const godraysShaft = pipeline
    .createTexture("dusk_godrays_shaft")
    .format(Format.RGBA16F)
    .scale(0.25)
    .storage(StorageHint.Persistent)
    .build();

  const bloomHalf = pipeline
    .createTexture("dusk_bloom_half")
    .format(Format.RG11B10F)
    .scale(0.25)
    .storage(StorageHint.Persistent)
    .build();

  const bloomEighth = pipeline
    .createTexture("dusk_bloom_eighth")
    .format(Format.RG11B10F)
    .scale(0.125)
    .storage(StorageHint.Persistent)
    .build();

  const bloom32 = pipeline
    .createTexture("dusk_bloom_32")
    .format(Format.RG11B10F)
    .scale(0.03125)
    .storage(StorageHint.Persistent)
    .build();

  const bloomCombined = pipeline
    .createTexture("dusk_bloom_combined")
    .format(Format.RG11B10F)
    .scale(0.25)
    .storage(StorageHint.Persistent)
    .build();

  pipeline
    .createCommandList("DEFERRED")
    .fullscreenPass({
      name: "meteor_sky_lut",
      shader: "shaders/sky_lut.slang",
      entry: "fsSkyLut",
      reads: [],
      writes: [skyViewLut, skyReflLut],
    })
    .fullscreenPass({
      name: "meteor_clouds_qres",
      shader: "shaders/clouds_qres.slang",
      entry: "fsCloudsQres",
      reads: [pipeline.mainDepthTex],
      writes: [cloudsQres],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_dusk_shadow_ssgi",
      shader: "shaders/dusk_shadow_ssgi.slang",
      entry: "fsDuskShadowSsgi",
      reads: [pipeline.builtinSceneColor, pipeline.mainDepthTex, pipeline.builtinShadowMap],
      writes: [duskLightingRaw, duskSunRaw, duskHalfGbuf],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_dusk_lighting_blur",
      shader: "shaders/dusk_lighting_blur.slang",
      entry: "fsDuskLightingBlur",
      reads: [duskLightingRaw, duskSunRaw, duskHalfGbuf],
      writes: [duskLightingStep1, duskSunStep1],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_dusk_lighting_blur_2",
      shader: "shaders/dusk_lighting_blur_2.slang",
      entry: "fsDuskLightingBlur2",
      reads: [duskLightingStep1, duskSunStep1, duskHalfGbuf],
      writes: [duskLightingFiltered, duskSunFiltered],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_dusk_ssr",
      shader: "shaders/dusk_ssr.slang",
      entry: "fsDuskSsr",
      reads: [pipeline.builtinSceneColor, pipeline.mainDepthTex, duskLightingFiltered, duskSunFiltered, skyReflLut],
      writes: [duskReflRaw],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_dusk_world_composite",
      shader: "shaders/dusk_world_composite.slang",
      entry: "fsDuskWorldComposite",
      reads: [pipeline.builtinSceneColor, pipeline.mainDepthTex, duskLightingFiltered, duskSunFiltered, duskReflRaw, skyViewLut, skyReflLut, cloudsQres],
      writes: [hdrLit],
    });

  pipeline
    .createCommandList("POST_TRANSLUCENT")
    .fullscreenPass({
      name: "meteor_volumetric_godrays",
      shader: "shaders/volumetric_godrays.slang",
      entry: "fsVolumetricGodrays",
      reads: [pipeline.mainDepthTex, cloudsQres],
      writes: [godraysShaft],
      samplesNeighbors: true,
    });

  pipeline
    .createCommandList("FINAL")
    .fullscreenPass({
      name: "meteor_bloom_down_half",
      shader: "shaders/bloom_downsample.slang",
      entry: "fsBloomDownsample",
      reads: [hdrLit],
      writes: [bloomHalf],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_bloom_down_eighth",
      shader: "shaders/bloom_down_mip.slang",
      entry: "fsBloomDownMip",
      reads: [bloomHalf],
      writes: [bloomEighth],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_bloom_down_32",
      shader: "shaders/bloom_down_mip.slang",
      entry: "fsBloomDownMip",
      reads: [bloomEighth],
      writes: [bloom32],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_bloom_upsample",
      shader: "shaders/bloom_blur.slang",
      entry: "fsBloomBlur",
      reads: [bloomHalf, bloomEighth, bloom32],
      writes: [bloomCombined],
      samplesNeighbors: true,
    })
    .fullscreenPass({
      name: "meteor_final_tonemap",
      shader: "shaders/final_tonemap.slang",
      entry: "fsFinalTonemap",
      reads: [hdrLit, bloomCombined, godraysShaft, pipeline.mainDepthTex],
      writes: [pipeline.builtinSceneColor],
    })
    .present(pipeline.builtinSceneColor);
}

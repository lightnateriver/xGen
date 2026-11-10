# Experimental R2VA reference-image resolution

H3 normally normalizes each reference image to a short edge of 2048 before the VAE and Qwen3-VL encoders read it. Reference-image resolution is independent of output-video resolution. Reducing it removes image detail and changes conditioning; it is not an equivalent numerical optimization.

The default stays **2048** with the released H3 checkpoint, including the `cuda-5090` preset. The 768 setting used in the experiment is an explicit override, not a new default. Omitting the option leaves the pipeline configuration untouched, so custom checkpoints also retain their own setting.

Append this option to an existing Ref2VA Base inference command:

```bash
--h3_reference_image_short_edge 768
```

Use `--h3_reference_image_short_edge 2048`, or remove the override, to return to the released model's original reference resolution. This is a pixel short-edge value, not a video resolution preset.

With the existing CUDA 5090 preset:

```bash
bash example/diffusers/scripts/run_h3.sh cuda-5090 r2va 5 768p inputs/ref01.png -- \
  --h3_reference_image_short_edge 768
```

The HTTP server accepts the same startup option, or `"h3_reference_image_short_edge": 768` in its H3 startup configuration. It is not a MaaS request field; restart the service to change it. Existing SP8, precision and VAE-decoder options remain independently selectable. Omit the option to preserve the checkpoint's original setting; the released checkpoint uses 2048.

The adapter forwards the value through Diffusers' existing `update_components(reference_image_short_edge=...)` API. The existing normalizer resizes the image once, and both encoders consume that normalized image. No custom resize, tokenizer, attention or caching path is introduced. Output resolution, reference video/audio processing and media-count limits are unchanged. T2VA, I2VA/FL2VA and Base+2K Refiner reject an explicit override in this first experiment. The value must be an integer multiple of 32 from 32 through 2048.

## CPU preprocessing verification

For a 2560×1440 image, using the existing Diffusers setup block and Qwen3-VL processor:

| Short edge | Actual image after 32-alignment | VAE packed tokens | Qwen visual tokens | Content tokens combined |
|---:|---:|---:|---:|---:|
|2048|3648×2048|7,296|7,296|14,592|
|1536|2720×1536|4,080|4,080|8,160|
|1024|1824×1024|1,824|1,824|3,648|
|768|1376×768|1,032|1,032|2,064|

Counts exclude prompt text, image labels/delimiters, audio and target-video tokens. Here 768 retains about 14.1% of the original image-content tokens. The actual reference becomes 1376×768, rather than the 1344×768 output canvas, because reference images have their own resize rule without the output-area cap.

The CPU preprocessing check verified configuration propagation, normalized image dimensions, actual Qwen processor grid, VAE packing geometry and unchanged 1344×768/124-frame output geometry. VAE token counts in the table are derived from encoder/patch geometry.

Subsequent CUDA runs with reference short edge 768, native-stream and BF16 AdaLN tables technically passed single-/five-image 5-second, 20-step generation on SP4 and SP8. These are combined-configuration results, not an isolated measurement of reference resizing. Sampled frames were inspected; complete human motion/audio acceptance and NPU generation with the reduced reference size remain unverified. Inspect identity, multiple subjects, small objects and text details before adopting it.

## Supported deployment

This release keeps the validated eight-rank CUDA precision contract; it does not publish the separate four-rank experiment. The new default also requires prepared [AdaLN assets](H3-CUDA-ADALN.md). Native SP8 stays the default; TP/CP remains an independent experimental option.

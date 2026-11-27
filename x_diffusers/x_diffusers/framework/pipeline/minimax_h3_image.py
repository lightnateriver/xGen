"""Model-specific H3 Image output processing and PDD invariants."""
import functools
import torch


def validate_pdd_config(pdd):
    if pdd.get("num_inference_steps") != 6 or len(pdd.get("sigmas", [])) != 7:
        raise ValueError("Expected the six-step H3 Image PDD checkpoint")


def install_image_rgb8_output(pipe):
    """Transport the released quantized pixels as RGB8, avoiding a float32 D2H round trip."""
    from PIL import Image
    processor = pipe.image_processor
    original = processor.postprocess
    @functools.wraps(original)
    def postprocess(image, output_type="pil", do_denormalize=None):
        if image.device.type != "npu" or output_type != "pil" or processor.config.do_normalize or (do_denormalize is not None and any(do_denormalize)):
            return original(image, output_type=output_type, do_denormalize=do_denormalize)
        # DecodeStep already applies the checkpoint's truncating quantizer before this call.
        pixels = (image * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [Image.fromarray(item) for item in pixels]
    processor.postprocess = postprocess

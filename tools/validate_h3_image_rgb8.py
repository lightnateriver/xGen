from types import SimpleNamespace
import json
from pathlib import Path
import torch
from diffusers.image_processor import VaeImageProcessor
from x_diffusers.framework.transformer.h3_image_optimizations import install_image_rgb8_output
torch.npu.set_device(0)
processor=VaeImageProcessor(do_normalize=False);torch.manual_seed(42)
pixels=torch.randint(0,256,(2,3,37,113),device='npu').float()/255
expected=processor.postprocess(pixels,output_type='pil')
install_image_rgb8_output(SimpleNamespace(image_processor=processor))
actual=processor.postprocess(pixels,output_type='pil')
equal=[a.tobytes()==b.tobytes() for a,b in zip(expected,actual)]
assert all(equal)
print(json.dumps(dict(matched=True,batch=2,width=113,height=37,scope='same quantized tensor; original FP32 D2H processor versus RGB8 D2H',exact_pixels_equal=equal),indent=2))

from torch import nn

from x_base.config import quant_config_manager


def test_h3_video_int8_keeps_adaln_projections_unquantized():
    class MiniMaxH3Transformer3DModel(nn.Module):
        pass

    config = quant_config_manager.get_config_for_model(MiniMaxH3Transformer3DModel())
    linear = nn.Linear(4, 4)
    assert not config.should_quantize("transformer_blocks.0.adaln_proj.linear", linear)
    assert not config.should_quantize("transformer_blocks.0.adaln_proj.linear.base_layer", linear)
    assert not config.should_quantize("norm_out.linear", linear)
    assert config.should_quantize("transformer_blocks.0.attn.to_q", linear)
    assert config.should_quantize("transformer_blocks.0.ff.net.0.proj", linear)

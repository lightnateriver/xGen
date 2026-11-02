from ascendx_video.base.vae_parallelism.vae_mgr import VAEManager
from ascendx_video.base.utils.infer_info import infer_info

def spatial_tiled_decode(
    hidden_state,
    device, 
    decoder_wrapper,
    pbar=True
):
    b,c,t,h,w = hidden_state.shape
    infer_info.update_shape(w*8, h*8)

    vae_manager = VAEManager(hidden_state.shape)
    sigle_clock = vae_manager.get_tile_from_x(hidden_state)
    out_ = decoder_wrapper(sigle_clock, device, pbar)
    out_ = vae_manager.align_out(out_)
    outputs = vae_manager.collect_out(out_)

    return outputs

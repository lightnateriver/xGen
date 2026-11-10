"""Translate the public H3 decoder choice into the existing VAE switches."""


def resolve_vae_backend(args):
    selected = getattr(args, 'h3_vae_backend', None)
    if selected is None:
        selected = ('native-stream' if (getattr(args, 'h3_cuda_preset', None) == 'sage-fp8-full-parallel'
                    or getattr(args, 'h3_cuda_precision', None) in ('fp8-bf16','fp8-sage'))
                    else 'legacy')
    if getattr(args, 'model', None) != 'MiniMax-H3':
        if selected != 'legacy':
            raise ValueError('--h3_vae_backend requires MiniMax-H3')
        args.h3_vae_backend = selected
        return selected
    if selected not in ('legacy', 'lightning', 'native-stream'):
        raise ValueError('H3 VAE backend must be legacy, lightning or native-stream')
    # Keep the internal switch used by VAEManager/InferInfo. The public H3
    # interface has one selector; other model families retain their old flag.
    previous = getattr(args, '_h3_vae_selection', None)
    supplied = getattr(args, 'vae_lightning', 'none')
    expected = previous[1] if previous is not None else 'none'
    if supplied != expected:
        raise ValueError('For MiniMax-H3 use --h3_vae_backend lightning instead of --vae_lightning')
    if selected == 'lightning' and getattr(args, 'sp', 1) <= 1:
        raise ValueError('H3 Lightning requires --sp greater than 1')
    args.vae_lightning = 'decoder' if selected == 'lightning' else 'none'
    args.h3_vae_backend = selected
    args._h3_vae_selection = (selected, args.vae_lightning)
    return selected

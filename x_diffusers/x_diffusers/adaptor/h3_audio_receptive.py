"""Native H3 audio convolution/anti-alias receptive-field accounting."""
import math

import torch

def _conv_span(lo,hi,kernel,stride=1,padding=0,dilation=1,transpose=False):
    if transpose:
        return (-((-(lo+padding-dilation*(kernel-1)))//stride), (hi-1+padding)//stride+1)
    return lo*stride-padding,(hi-1)*stride-padding+dilation*(kernel-1)+1


def audio_span(module,lo,hi):
    """Back-propagate exact convolution/AA receptive intervals, no audio cuts."""
    name=type(module).__name__
    if isinstance(module,(torch.nn.Conv1d,torch.nn.ConvTranspose1d)):
        return _conv_span(lo,hi,module.kernel_size[0],module.stride[0],module.padding[0],module.dilation[0],
                          isinstance(module,torch.nn.ConvTranspose1d))
    if isinstance(module,torch.nn.Sequential):
        for child in reversed(list(module)):lo,hi=audio_span(child,lo,hi)
        return lo,hi
    if name in ('MiniMaxH3AudioEncoder','MiniMaxH3AudioEncoderBlock'):
        return audio_span(module.block,lo,hi)
    if name=='MiniMaxH3AudioResidualUnit':
        a,b=audio_span(module.block,lo,hi);return min(a,lo),max(b,hi)
    if name=='MiniMaxH3AudioActivation1d':
        lo,hi=audio_span(module.downsample,lo,hi)
        return audio_span(module.upsample,lo,hi)
    if name=='MiniMaxH3AudioDownSample1d':return audio_span(module.lowpass,lo,hi)
    if name=='MiniMaxH3AudioLowPassFilter1d':
        return _conv_span(lo,hi,module.filter.shape[-1],module.stride,module.pad_left)
    if name=='MiniMaxH3AudioUpSample1d':
        lo,hi=_conv_span(lo+module.pad_left,hi+module.pad_left,module.filter.shape[-1],module.stride,transpose=True)
        return lo-module.pad,hi-module.pad
    if name=='MiniMaxH3AudioAMPBlock':
        for i in reversed(range(len(module.convs1))):
            a,b=audio_span(module.convs2[i],lo,hi)
            a,b=audio_span(module.activations[2*i+1],a,b)
            a,b=audio_span(module.convs1[i],a,b)
            a,b=audio_span(module.activations[2*i],a,b)
            lo,hi=min(lo,a),max(hi,b)
        return lo,hi
    if name=='MiniMaxH3AudioBigVGANDecoder':
        lo,hi=audio_span(module.conv_post,lo,hi)
        lo,hi=audio_span(module.activation_post,lo,hi)
        for i in reversed(range(module.num_upsamples)):
            intervals=[audio_span(module.resblocks[i*module.num_kernels+j],lo,hi) for j in range(module.num_kernels)]
            lo,hi=min(a for a,b in intervals),max(b for a,b in intervals)
            lo,hi=audio_span(module.ups[i][0],lo,hi)
        return audio_span(module.conv_pre,lo,hi)
    if name in ('MiniMaxH3AudioSnake1d','MiniMaxH3AudioSnakeBeta'):return lo,hi
    raise TypeError('Unhandled receptive field module: '+name)


def audio_halos(model):
    hop=model.hop_length;frame=1000
    a,b=audio_span(model.encoder,frame,frame+1)
    encoder=max(math.ceil((frame*hop-a)/hop),math.ceil((b-(frame+1)*hop)/hop))
    a,b=audio_span(model.decoder,frame*hop,(frame+1)*hop)
    a,b=audio_span(model.dec_in_proj,a,b)
    decoder=max(frame-a,b-frame-1)
    return dict(encoder_required=encoder,decoder_required=decoder,
                encoder_used=math.ceil((encoder+1)/4)*4,decoder_used=math.ceil((decoder+1)/4)*4)



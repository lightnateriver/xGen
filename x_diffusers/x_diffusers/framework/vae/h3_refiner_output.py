"""Refiner pixel/crop adapter over the existing SaveAVStream codec, not a new encoder."""
from pathlib import Path
import tempfile

import torch
from x_base.vae_parallelism.audio_utils import save_audio_to_wav
from x_base.vae_parallelism.save_av_stream import SaveAVStream


class RefinerOutput:
    def __init__(self, output, frames, metrics, audio):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.frames, self.sent, self.metrics = frames, 0, metrics
        self.closed = False
        self.work = tempfile.TemporaryDirectory(prefix='.h3-refiner-', dir=self.output.parent)
        self.staged = Path(self.work.name)/'output.mp4'
        wav = str(Path(self.work.name)/'audio.wav') if audio is not None else None
        if wav:
            save_audio_to_wav(audio[..., :frames//24*32000],32000,audio_path=wav)
        self.stream = SaveAVStream(str(self.staged),wav,24,2560,1440,strict=True)

    def enqueue(self, chunk):
        chunk = chunk[:,:,:max(0,self.frames-self.sent)]
        if chunk.shape[2] == 0:
            return
        if not torch.isfinite(chunk).all():
            raise ValueError('Nonfinite pixels before video export')
        with self.metrics.span('video.rgb8',sync=False):
            mean = chunk.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
            std = chunk.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
            pixels = ((chunk.float()*std+mean).clamp(0,1)*255).round().byte()[0].permute(1,2,3,0)
        with self.metrics.host_span('video.d2h_and_enqueue'):
            for frame in pixels.cpu().numpy():
                self.stream.put_frame(frame)
                self.sent += 1

    def finish(self):
        if self.sent != self.frames:
            raise ValueError(f'Expected {self.frames} exported frames, got {self.sent}')
        with self.metrics.host_span('video.encode_and_audio_mux_wait'):
            self.stream.close()
            self.closed = True
        if not self.staged.is_file() or self.staged.stat().st_size == 0:
            raise RuntimeError('Encoder produced no video')
        self.staged.replace(self.output)

    def close(self):
        if not self.closed:
            self.stream.abort()
        self.work.cleanup()

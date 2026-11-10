"""Explicit test doubles. Replace the backend object to integrate a real provider.

These implementations NEVER call a model or vendor API. They require an opt-in
startup switch and log MOCK for every request. Do not use for quality or timing.
"""
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class MockBackends:
    def context_ir(self, request, normalized):
        logger.warning('MOCK H3-Context-IR: returning the input prompt unchanged; no model inference')
        return dict(prompt=normalized['prompt'])

    def video_2k(self, path, width, height, seconds, *, request, normalized):
        logger.warning('MOCK H3 2K: generating a labelled test card; no model inference')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.mock-', dir=path.parent) as directory:
            card = Image.new('RGB', (640, 360), '#fff3f2')
            draw = ImageDraw.Draw(card)
            draw.rectangle((0, 0, 640, 18), fill='#c7000b')
            draw.text((80, 140), 'MOCK 2K - NOT MODEL OUTPUT', fill='#c7000b')
            draw.text((80, 175), 'AIGC TEST PLACEHOLDER - DO NOT USE FOR EVALUATION', fill='#3d3333')
            card.resize((width, height)).save(Path(directory)/'card.png')
            subprocess.run([os.getenv('FFMPEG_PATH', 'ffmpeg'), '-v', 'error', '-y',
                '-loop', '1', '-framerate', '24', '-i', str(Path(directory)/'card.png'),
                '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', str(seconds),
                '-c:v', 'libx264', '-threads', '1', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', str(path)], check=True,
                timeout=float(os.getenv('XGEN_MAAS_MOCK_TIMEOUT_S', '180')))
        return dict(path=str(path.resolve()))

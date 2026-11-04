"""CPU tests with real FFmpeg fixtures and a simulated OBS client."""

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from maas_artifacts import StorageError, publish_obs, publish_video


class MaaSArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = TemporaryDirectory()
        cls.addClassCleanup(root.cleanup)
        cls.fixture = Path(root.name) / 'good.mp4'
        cls.silent = Path(root.name) / 'silent.mp4'
        ffmpeg = os.getenv('FFMPEG_PATH', 'ffmpeg')
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'color=s=1344x768:r=24',
                        '-f', 'lavfi', '-i', 'anullsrc', '-frames:v', '124', '-c:v', 'libx264',
                        '-preset', 'ultrafast', '-c:a', 'aac', '-movflags', '+faststart', '-shortest', str(cls.fixture)],
                       check=True, timeout=30)
        subprocess.run([ffmpeg, '-v', 'error', '-i', str(cls.fixture), '-map', '0:v:0',
                        '-c', 'copy', str(cls.silent)], check=True, timeout=15)

    def test_output_is_only_visible_after_validation(self):
        with TemporaryDirectory() as root:
            output = Path(root) / 'result.mp4'

            def encode(path):
                shutil.copyfile(self.fixture, path)
                self.assertFalse(output.exists())
                self.assertEqual(Path(path).parent, output.parent)

            publish_video(output, encode, require_audio=True)
            self.assertEqual(list(Path(root).iterdir()), [output])
            self.assertEqual(output.read_bytes(), self.fixture.read_bytes())

    def test_bad_or_incomplete_output_is_never_published(self):
        with TemporaryDirectory() as root:
            output = Path(root) / 'result.mp4'
            good = self.fixture.read_bytes()
            for contents in (b'', good[:100], good[:len(good) * 7 // 10], self.silent.read_bytes()):
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    publish_video(output, lambda path: Path(path).write_bytes(contents), require_audio=True)
                self.assertEqual(list(Path(root).iterdir()), [])

            def broken_encoder(path):
                Path(path).write_bytes(b'incomplete')
                raise OSError('disk full')

            with self.assertRaises(OSError):
                publish_video(output, broken_encoder, require_audio=True)
            self.assertEqual(list(Path(root).iterdir()), [])
            publish_video(output, lambda path: shutil.copyfile(self.silent, path), require_audio=False)
            self.assertTrue(output.is_file())

    def test_object_confirmation_precedes_signed_url(self):
        client = MagicMock()
        client.__enter__.return_value = client
        factory = SimpleNamespace(ObsStorageClient=lambda: client)
        with TemporaryDirectory() as root, patch.dict('sys.modules', {'obs_client': factory}), \
             patch.dict(os.environ, {'OBS_STORAGE_PATH': 'outputs'}), patch('maas_artifacts.urlopen') as read:
            client.wait_object_ready.return_value = False
            with self.assertRaises(StorageError):
                publish_obs(self.fixture, root)
            client.generate_url.assert_not_called()
            client.wait_object_ready.return_value = True
            client.generate_url.return_value = 'https://example.invalid/result'
            read.return_value.__enter__.return_value.read.return_value = b''
            with self.assertRaises(StorageError):
                publish_obs(self.fixture, root)
            read.return_value.__enter__.return_value.read.return_value = b'v'
            self.assertEqual(publish_obs(self.fixture, root), client.generate_url.return_value)
            self.assertEqual((Path(root) / self.fixture.name).read_bytes(), self.fixture.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)

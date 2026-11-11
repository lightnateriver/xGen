"""Native files use the existing atomic publisher, without re-encoding."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import video_server
from maas_artifacts import publish_video
from test_maas_artifacts import MaaSArtifactTests as _ArtifactFixtures


class NativeStreamOnlineTests(unittest.TestCase):
    setUpClass=classmethod(_ArtifactFixtures.setUpClass.__func__)
    def test_native_file_is_moved_through_atomic_publication(self):
        with TemporaryDirectory() as directory:
            staged=Path(directory)/'.native.mp4'
            staged.write_bytes(self.fixture.read_bytes())
            final=Path(directory)/'final.mp4'
            service=object.__new__(video_server.VideoGenerationService)
            args=SimpleNamespace(h3_vae_backend='native-stream')
            output={'_h3_native_saved_path':str(staged.resolve())}
            with patch.object(video_server.infer_info,'save_path',str(staged),create=True):
                publish_video(final,lambda path:service._save_minimax_h3_output(output,path,args),require_audio=True)
            self.assertEqual(final.read_bytes(),self.fixture.read_bytes())
            self.assertFalse(staged.exists())
            self.assertEqual(list(Path(directory).iterdir()),[final])

    def test_native_file_cannot_come_from_another_request(self):
        service=object.__new__(video_server.VideoGenerationService)
        with patch.object(video_server.infer_info,'save_path','expected.mp4',create=True):
            with self.assertRaisesRegex(ValueError,'does not match'):
                service._save_minimax_h3_output({'_h3_native_saved_path':'another.mp4'},'target.mp4',
                    SimpleNamespace(h3_vae_backend='native-stream'))

    def test_table_condition_uses_shared_selector_even_without_condition_change(self):
        manager=video_server.VideoInferenceManager()
        transformer=SimpleNamespace(_h3_cuda_adaln_table=True,_adaln_cache=SimpleNamespace(_conditioning_type='none'))
        manager.pipe=SimpleNamespace(transformer=transformer)
        args=SimpleNamespace(adaln_cache_file='table.safetensors')
        with patch('x_diffusers.adaptor.h3_references.select_adaln') as selected:
            manager._mmh3_switch_adaln_cache(args,transformer,[],0)
        selected.assert_called_once_with(manager.pipe,args,'none')


if __name__=='__main__':unittest.main()

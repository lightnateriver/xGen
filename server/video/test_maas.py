"""CPU tests: real Flask admission and real image/audio/video probes; no weights."""
import base64
from fractions import Fraction
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import urlsplit
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from maas import normalize, output_size, parameter_spec, validate_startup
from maas_fl2va import InvalidInput
from maas_media import inspect_asset, public_target, cleanup, download, enable_image_codecs
from maas_artifacts import publish_video
from video_server import VideoGenerationService


def image(width=256, height=256, fmt='PNG'):
    data=io.BytesIO(); Image.new('RGB',(width,height)).save(data,format=fmt)
    return 'data:image/'+fmt.lower()+';base64,'+base64.b64encode(data.getvalue()).decode()


class MaaSTests(unittest.TestCase):
    def setUp(self):
        self.service=VideoGenerationService(maas=True)
        self.service.default_config.update(model='MiniMax-H3',task_type='t2va',num_inference_steps=20)
        self.service.is_initialized=True
        self.client=self.service.app.test_client()

    def submit(self, data, identifier='test'):
        return self.client.post('/v1/tasks',json=dict(id=identifier,input=data))

    def tearDown(self):
        for job in self.service.maas_pending.values(): cleanup(job['data'])

    def test_output_uses_configured_temp_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('video_server.tempfile.gettempdir',return_value=directory):
                response=self.submit(dict(prompt='test',task_type='t2va'))
            self.assertEqual(response.status_code,200,response.json)
            job,_=self.service._claim_maas_task()
            self.assertEqual(Path(job['save_path']).parent,Path(directory)/'generated_videos')
            cleanup(job['data'])

    def test_modes_and_both_routes(self):
        for task,fields in [('t2va',{}),('fl2va',dict(first_frame=image())),
                            ('fl2va',dict(last_frame=image())),
                            ('fl2va',dict(first_frame=image(),last_frame=image()))]:
            for route in ('/generate','/v1/tasks'):
                data=dict(prompt='test',task_type=task,duration=15,**fields)
                body=dict(id='test',input=data) if route=='/v1/tasks' else data
                response=self.client.post(route,json=body)
                self.assertEqual(response.status_code,200,response.json)
                job,_=self.service._claim_maas_task()
                self.assertEqual(job['data']['duration'],15)
                args=self.service._build_args(job['data'])
                if 'last_frame' in fields:self.assertTrue(args.keyframe_last_image)
                self.assertEqual((args.width,args.height), (992,992) if fields else (1344,768))
                cleanup(job['data']);self.service.tasks.clear()

    def test_ref2va_counts_and_cleanup(self):
        self.service.default_config.update(task_type='ref2va',service_mode='ref2va')
        for count,status in ((1,200),(5,200),(9,200),(10,400),(13,400)):
            response=self.submit(dict(prompt='test',ref_path=[image()]*count))
            self.assertEqual(response.status_code,status,response.json)
            if status==200:
                job,_=self.service._claim_maas_task()
                self.assertEqual(len(job['data']['ref_path']),count)
                cleanup(job['data']);self.service.tasks.clear()
            else:self.assertFalse(self.service.maas_pending)

    def test_duplicates_cancel_and_conflict_clean_assets(self):
        data=dict(prompt='test',first_frame=image(),task_type='fl2va')
        first=self.submit(data);self.assertEqual(first.status_code,200)
        job=self.service.maas_pending['test'];directory=Path(job['data']['_maas_dir'])
        self.assertTrue(directory.exists())
        self.assertEqual(self.submit(data).status_code,200)
        self.assertEqual(self.submit(dict(data,seed=42,model='MiniMax-H3',resolution='768P')).status_code,200)
        self.assertEqual(self.submit(dict(data,prompt='different')).status_code,409)
        self.assertEqual(self.client.delete('/v1/tasks/test').status_code,200)
        self.assertFalse(directory.exists())

    def test_expiry_cleans_assets(self):
        self.submit(dict(prompt='test',first_frame=image(),task_type='fl2va'))
        directory=Path(self.service.maas_pending['test']['data']['_maas_dir'])
        with self.service.tasks_lock:
            self.service._expire_maas_pending(self.service.tasks['test'].queued_at+601)
        self.assertFalse(directory.exists())

    def test_bad_combinations_and_sizes(self):
        cases=[dict(task_type='t2va',first_frame=image()),dict(task_type='fl2va'),
               dict(task_type='fl2va',first_frame=image(),ref_path=[image()]),
               dict(task_type='fl2va',first_frame=image(255,256)),
               dict(task_type='fl2va',first_frame=image(768,256)),
               dict(task_type='fl2va',first_frame='/private.png'),
               dict(duration=3),dict(duration=16),dict(duration=True),dict(resolution='2k'),
               dict(width=1024),dict(aspect_ratio='adaptive'),dict(prompt='🙂'*3501)]
        for fields in cases:
            with self.subTest(fields=list(fields)):
                self.assertEqual(self.submit(dict(dict(prompt='test',task_type='t2va'),**fields)).status_code,400)
                self.assertFalse(self.service.maas_pending)

    def test_formats_and_dimensions(self):
        for fmt in ('JPEG','PNG','WEBP'):
            result=normalize(dict(prompt='test',task_type='fl2va',first_frame=image(640,256,fmt)),self.service.default_config)
            self.assertEqual(result['_maas_size'],(1600,640));cleanup(result)
        for ratio in ('21:9','16:9','4:3','1:1','3:4','9:16'):
            a,b=map(int,ratio.split(':'));w,h=output_size(Fraction(a,b))
            self.assertEqual(w%32+h%32,0)
            self.assertLessEqual(w*h,1344*768)
            self.assertLess(abs(w/h-a/b)/(a/b),.05)
        self.assertEqual(self.client.get('/params').json['max_body_bytes'],64*1024*1024)

    def test_cli_aliases_are_parseable(self):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        for flag in ('--maas','--maas-fl2va-input'):
            result=subprocess.run([__import__('sys').executable,str(Path(__file__).with_name('video_server.py')),
                                   flag,'--maas-queue-capacity','0'],env=env,capture_output=True,text=True,timeout=40)
            self.assertEqual(result.returncode,2,result.stderr[-1000:])
            self.assertIn('MaaS queue capacity and timeout must be positive',result.stderr)

    def test_heif_and_heic_decode(self):
        enable_image_codecs()
        uri=image(256,256,'HEIF')
        for source in (uri,uri.replace('image/heif','image/heic')):
            result=normalize(dict(prompt='test',task_type='fl2va',first_frame=source),self.service.default_config)
            self.assertEqual(Image.open(result['first_frame']).format,'PNG')
            cleanup(result)

    def test_public_url_resolution(self):
        for address in ('127.0.0.1','10.0.0.1','169.254.169.254','::1','192.168.1.1'):
            with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,80))]):
                with self.assertRaises(InvalidInput):public_target('http://example.invalid/a')
        with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443))]):
            self.assertEqual(public_target('https://example.invalid/a')[2],'8.8.8.8')

    def test_streamed_url_download_and_limits(self):
        payload=b'0123456789'
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path=='/redirect':
                    self.send_response(302);self.send_header('Location','/data');self.end_headers();return
                self.send_response(200);self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://test.invalid:{server.server_port}/redirect'
            # Replace only address selection for this controlled loopback transport test.
            with tempfile.TemporaryDirectory() as tmp, patch('maas_media.public_target',
                    side_effect=lambda u:(urlsplit(u),server.server_port,'127.0.0.1')):
                target=Path(tmp)/'asset'
                download(url,target,maximum=10);self.assertEqual(target.read_bytes(),payload)
                with self.assertRaises(InvalidInput) as error:download(url,target,maximum=9)
                self.assertEqual(error.exception.status,413)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_media_probe_and_combined_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            wav=root/'audio.wav'
            subprocess.run([os.environ.get('FFMPEG_PATH','ffmpeg'),'-v','error','-f','lavfi','-i','anullsrc=r=32000:cl=stereo',
                            '-t','2','-y',str(wav)],check=True)
            item=inspect_asset(wav)
            self.assertEqual(item['kind'],'audio');self.assertEqual(item['seconds'],2)
            video=root/'clip.mp4'
            subprocess.run([os.environ.get('FFMPEG_PATH','ffmpeg'),'-v','error','-f','lavfi','-i','color=s=256x256:r=24',
                            '-t','2','-c:v','libx264','-threads','1','-y',str(video)],check=True)
            item=inspect_asset(video);self.assertEqual(item['kind'],'video')
            destination=root/'published.mp4'
            publish_video(destination,lambda p:Path(p).write_bytes(video.read_bytes()),
                          require_audio=False,expected=(256,256,48))
            self.assertTrue(destination.exists())
            with self.assertRaises(ValueError):
                publish_video(destination,lambda p:Path(p).write_bytes(video.read_bytes()),
                              require_audio=False,expected=(256,256,49))
            uri='data:audio/wav;base64,'+base64.b64encode(wav.read_bytes()).decode()
            cfg=dict(self.service.default_config,task_type='ref2va',service_mode='ref2va')
            with self.assertRaises(InvalidInput):normalize(dict(prompt='test',ref_path=[uri]),cfg)
            result=normalize(dict(prompt='test',ref_path=[image(),uri]),cfg);cleanup(result)
            fake=lambda *args,**kwargs: dict(kind='video',path='/unused.mp4',width=256,height=256,
                                           seconds=Fraction(8),sha256='0'*64)
            with patch('maas.materialize',side_effect=fake):
                with self.assertRaises(InvalidInput):normalize(dict(prompt='test',ref_path=['a','b']),cfg)


if __name__=='__main__':unittest.main()

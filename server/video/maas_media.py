"""CPU-only MaaS asset admission. Public downloads are pinned to a validated IP."""
import base64
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
from fractions import Fraction
from urllib.parse import urljoin, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError
from maas_fl2va import InvalidInput

MIB = 1024 * 1024
BYTE_LIMITS = {'image': 30*MIB, 'video': 50*MIB, 'audio': 15*MIB}
PIXEL_LIMITS = {'frame': 4096*2048, 'reference': 4096*4096}
IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP', 'HEIF', 'HEIC'}


def enable_image_codecs():
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        raise ValueError('--maas requires pillow-heif; install server/requirements-maas.txt') from None
    register_heif_opener()


def public_target(url):
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise InvalidInput('media', 'Malformed media URL') from None
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise InvalidInput('media', 'Expected a public HTTP(S) URL without user credentials')
    try:
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        ips = [item[4][0] for item in addresses]
        if not ips or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast
                          or ipaddress.ip_address(ip).is_reserved for ip in ips):
            raise ValueError('non-public address')
    except (OSError, ValueError):
        raise InvalidInput('media', 'URL must resolve exclusively to public addresses') from None
    return parsed, port, ips[0]


def download(url, target, *, timeout=60, maximum=50*MIB):
    if not 0 < timeout <= 3600:
        raise InvalidInput('media', 'Download timeout must be in (0, 3600] seconds')
    deadline = time.monotonic() + timeout
    for redirect in range(4):
        if time.monotonic() >= deadline:
            raise InvalidInput('media', 'Media download deadline exceeded')
        parsed, port, ip = public_target(url)
        connection = http.client.HTTPConnection(ip, port, timeout=min(10, max(.1, deadline-time.monotonic())))
        try:
            # Never resolve the hostname again between validation and connection.
            connection.connect()
            if parsed.scheme == 'https':
                connection.sock = ssl.create_default_context().wrap_socket(connection.sock, server_hostname=parsed.hostname)
            authority = parsed.netloc
            resource = (parsed.path or '/') + ('?'+parsed.query if parsed.query else '')
            connection.request('GET', resource, headers={'Host': authority, 'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location or redirect == 3:
                    raise InvalidInput('media', 'Too many or invalid URL redirects')
                url = urljoin(url, location)
                continue
            if response.status != 200 or response.getheader('Content-Encoding', 'identity') != 'identity':
                raise InvalidInput('media', 'Media URL must return an uncompressed HTTP 200 response')
            total = 0
            with target.open('wb') as output:
                while True:
                    if time.monotonic() >= deadline:
                        raise InvalidInput('media', 'Media download deadline exceeded')
                    if connection.sock:
                        connection.sock.settimeout(min(10, max(.1, deadline-time.monotonic())))
                    chunk = response.read(min(64*1024, maximum-total+1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise InvalidInput('media', 'Media file exceeds the byte limit', 413)
                    output.write(chunk)
            return
        except InvalidInput:
            raise
        except (OSError, http.client.HTTPException, ValueError):
            raise InvalidInput('media', 'Media download failed') from None
        finally:
            connection.close()


def dimensions(width, height, field='media'):
    if not (256 <= width <= 5760 and 256 <= height <= 5760 and .4 <= width/height <= 2.5):
        raise InvalidInput(field, 'Each edge must be 256..5760 pixels and width/height must be 0.4..2.5')


def inspect_asset(path, *, frame=False):
    try:
        with Image.open(path) as image:
            if image.format not in IMAGE_FORMATS or getattr(image, 'n_frames', 1) != 1:
                raise InvalidInput('media', 'Expected a still JPEG, PNG, WebP, HEIC or HEIF image')
            dimensions(image.width, image.height)
            if image.width*image.height > PIXEL_LIMITS['frame' if frame else 'reference']:
                raise InvalidInput('media', 'Image exceeds the pixel limit', 413)
            if path.stat().st_size > BYTE_LIMITS['image']:
                raise InvalidInput('media', 'Image exceeds 30 MiB', 413)
            image.verify()
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert('RGB')
            width, height = normalized.size
            result = path.with_suffix('.png')
            normalized.save(result)
        return dict(kind='image', path=str(result), width=width, height=height)
    except InvalidInput:
        raise
    except UnidentifiedImageError:
        if frame:
            raise InvalidInput('media', 'Frame input must decode as a supported image') from None
    except (OSError, ValueError, Image.DecompressionBombError):
        raise InvalidInput('media', 'Invalid image') from None
    try:
        probe = subprocess.run([os.environ.get('FFPROBE_PATH', 'ffprobe'), '-v', 'error', '-show_streams',
                                '-show_format', '-of', 'json', str(path)], capture_output=True, text=True,
                               check=True, timeout=30)
        info = json.loads(probe.stdout)
        streams = [s for s in info.get('streams', []) if not s.get('disposition', {}).get('attached_pic')]
        videos = [s for s in streams if s.get('codec_type') == 'video']
        audios = [s for s in streams if s.get('codec_type') == 'audio']
        kind = 'video' if videos else 'audio'
        stream = (videos or audios)[0]
        fmt = info['format']['format_name'].split(',')
        if videos:
            if not {'mp4', 'mov'}.intersection(fmt) or any(s['codec_name'] not in ('h264', 'hevc') for s in videos) or any(s['codec_name'] not in ('aac', 'mp3') for s in audios):
                raise InvalidInput('media', 'Video requires MP4/MOV with H.264/H.265 and AAC/MP3 audio')
            dimensions(stream['width'], stream['height'])
            fps = Fraction(stream.get('avg_frame_rate', '0'))
            if not Fraction('23.976') <= fps <= 60:
                raise InvalidInput('media', 'Video frame rate must be 23.976..60')
            suffix = '.mov' if info['format'].get('tags', {}).get('major_brand', '').strip() == 'qt' else '.mp4'
        else:
            if not {'wav', 'mp3'}.intersection(fmt):
                raise InvalidInput('media', 'Audio must be WAV or MP3')
            suffix = '.wav' if 'wav' in fmt else '.mp3'
        seconds = Fraction(str(stream.get('duration', info['format'].get('duration'))))
        if not 2 <= seconds <= 15:
            raise InvalidInput('media', 'Each video/audio clip must be 2..15 seconds')
        if path.stat().st_size > BYTE_LIMITS[kind]:
            raise InvalidInput('media', f'{kind} exceeds its byte limit', 413)
        result = path.with_suffix(suffix)
        path.rename(result)
        width, height = stream.get('width'), stream.get('height')
        rotation = stream.get('tags', {}).get('rotate', 0)
        for side in stream.get('side_data_list', []):
            rotation = side.get('rotation', rotation)
        if videos and int(float(rotation)) % 180:
            width, height = height, width
        return dict(kind=kind, path=str(result), seconds=seconds, width=width, height=height)
    except InvalidInput:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, ZeroDivisionError, TypeError):
        raise InvalidInput('media', 'Media could not be probed as a supported video or audio') from None


def materialize(value, directory, index, *, frame=False):
    if not isinstance(value, str):
        raise InvalidInput('media', 'Expected a media URL or base64 data URI')
    path = directory / f'{index}.bin'
    maximum = BYTE_LIMITS['image'] if frame else BYTE_LIMITS['video']
    if value.startswith('data:'):
        header, sep, encoded = value.partition(',')
        if not sep or not header.endswith(';base64') or len(encoded) > 4*((maximum+2)//3):
            raise InvalidInput('media', 'Invalid or oversized base64 media', 413)
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise InvalidInput('media', 'Invalid base64 media') from None
        if len(payload) > maximum:
            raise InvalidInput('media', 'Media exceeds the byte limit', 413)
        path.write_bytes(payload)
    else:
        download(value, path, timeout=float(os.environ.get('XGEN_MAAS_DOWNLOAD_TIMEOUT_S', '60')), maximum=maximum)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(inspect_asset(path, frame=frame), sha256=digest)


def new_directory():
    return Path(tempfile.mkdtemp(prefix='xgen-maas-', dir=os.environ.get('XGEN_MAAS_MEDIA_ROOT')))


def cleanup(data):
    directory = data.get('_maas_dir')
    if directory:
        path = Path(directory)
        root = Path(os.environ.get('XGEN_MAAS_MEDIA_ROOT', tempfile.gettempdir())).resolve()
        if path.resolve().parent == root and path.name.startswith('xgen-maas-'):
            shutil.rmtree(path, ignore_errors=True)

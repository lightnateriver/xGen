"""Validate and publish the optional MaaS FL2VA profile's output."""

from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


class StorageError(RuntimeError):
    pass


def validate_media_tools():
    for variable, default in (("FFMPEG_PATH", "ffmpeg"), ("FFPROBE_PATH", "ffprobe")):
        if shutil.which(os.getenv(variable, default)) is None:
            raise ValueError(f"MaaS FL2VA requires {default} on PATH or configured via {variable}")


def _atomic_file(destination, write):
    destination = Path(destination)
    with NamedTemporaryFile(dir=destination.parent, prefix=".pending-", suffix=".mp4", delete=False) as file:
        temporary = Path(file.name)
    try:
        write(str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_video(path, require_audio, expected=(1344, 768, 124)):
    result = subprocess.run(
        [os.getenv("FFPROBE_PATH", "ffprobe"), "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,nb_frames,nb_read_packets", "-of", "json", path],
        capture_output=True, text=True, check=True, timeout=15,
    )
    if result.stderr.strip():
        raise ValueError("ffprobe reported damaged media packets")
    streams = json.loads(result.stdout)["streams"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    width, height, frames = expected
    if len(video) != 1 or (video[0].get("width"), video[0].get("height")) != (width, height):
        raise ValueError(f"Expected one {width}x{height} video stream")
    if (Fraction(video[0]["r_frame_rate"]) != 24 or int(video[0]["nb_frames"]) != frames
            or int(video[0]["nb_read_packets"]) != frames):
        raise ValueError(f"Expected {frames} video frames at 24 fps")
    if require_audio and not any(stream.get("codec_type") == "audio" and int(stream.get("nb_read_packets", 0)) > 0
                                 for stream in streams):
        raise ValueError("Expected an audio stream with data packets")


def publish_video(path, encode, *, require_audio, expected=(1344, 768, 124)):
    def write(temporary):
        encode(temporary)
        _validate_video(temporary, require_audio, expected)

    _atomic_file(path, write)


def publish_obs(path, directory):
    """Publish to the existing OBS-mounted directory and confirm object metadata."""
    try:
        from obs_client import ObsStorageClient

        filename = Path(path).name
        key = os.path.join(os.environ["OBS_STORAGE_PATH"], filename)
        _atomic_file(Path(directory) / filename, lambda target: shutil.copyfile(path, target))
        with ObsStorageClient() as client:
            if not client.wait_object_ready(key):
                raise StorageError("Output object could not be confirmed")
            url = client.generate_url(filename)
        with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=5) as response:
            if not response.read(1):
                raise StorageError("Output object is empty")
        return url
    except Exception as error:
        if isinstance(error, StorageError):
            raise
        raise StorageError("Could not publish output to object storage") from None

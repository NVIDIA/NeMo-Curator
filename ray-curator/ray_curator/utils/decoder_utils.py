import json
import subprocess
from pathlib import Path
from dataclasses import dataclass

from ray_curator.utils.operation_utils import make_pipeline_named_temporary_file

@dataclass
class VideoMetadata:
    """Metadata for video content including dimensions, timing, and codec information.

    This class stores essential video properties such as resolution, frame rate,
    duration, and encoding details.
    """

    height: int = None
    width: int = None
    fps: float = None
    num_frames: int = None
    video_codec: str = None
    pixel_format: str = None
    video_duration: float = None
    audio_codec: str = None
    bit_rate_k: int = None

def extract_video_metadata(video: str | bytes) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Args:
        video: Path to video file or video data as bytes.

    Returns:
        VideoMetadata object containing video properties.

    """
    inp = None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
    ]
    with make_pipeline_named_temporary_file(sub_dir="extract_video_metadata") as video_path:
        if isinstance(video, bytes):
            video_path.write_bytes(video)
            real_video_path = video_path
        else:
            real_video_path = Path(str(video))
        if not real_video_path.exists():
            error_msg = f"{real_video_path} not found!"
            raise FileNotFoundError(error_msg)
        cmd.append(real_video_path.as_posix())
        result = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)  # noqa: UP022, S603
        video_info = json.loads(result.stdout)

    video_stream, audio_codec = None, None
    for stream in video_info["streams"]:
        if stream["codec_type"] == "video":
            video_stream = stream
        elif stream["codec_type"] == "audio":
            audio_codec = stream["codec_name"]
    if not video_stream:
        error_msg = "No video stream found!"
        raise ValueError(error_msg)

    # Convert avg_frame_rate to float
    num, denom = map(int, video_stream["avg_frame_rate"].split("/"))
    fps = num / denom

    # not all formats store duration at stream level, so fallback to format container
    if "duration" in video_stream:
        video_duration = float(video_stream["duration"])
    elif "format" in video_info and "duration" in video_info["format"]:
        video_duration = float(video_info["format"]["duration"])
    else:
        error_msg = "Could not find `duration` in video metadata."
        raise KeyError(error_msg)
    num_frames = int(video_duration * fps)

    # store bit_rate if available
    bit_rate_k = 2000  # default to 2000K (2M) bit rate
    if "bit_rate" in video_stream:
        bit_rate_k = int(int(video_stream["bit_rate"]) / 1024)

    return VideoMetadata(
        height=video_stream["height"],
        width=video_stream["width"],
        fps=fps,
        num_frames=num_frames,
        video_codec=video_stream["codec_name"],
        pixel_format=video_stream["pix_fmt"],
        audio_codec=audio_codec,
        video_duration=video_duration,
        bit_rate_k=bit_rate_k,
    )
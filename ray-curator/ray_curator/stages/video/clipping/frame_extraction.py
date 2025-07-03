from dataclasses import dataclass
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import VideoTask
from ray_curator.backends.base import WorkerMetadata
from loguru import logger
from ray_curator.utils.nvcodec_utils import PyNvcFrameExtractor
from ray_curator.utils.operation_utils import make_pipeline_named_temporary_file
import numpy as np
import numpy.typing as npt
import subprocess
from pathlib import Path
from ray_curator.stages.resources import Resources

def get_frames_from_ffmpeg(
    video_file: Path,
    width: int,
    height: int,
    *,
    use_gpu: bool = False,
) -> npt.NDArray[np.uint8] | None:
    """Fetch resized frames for video."""
    if use_gpu:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "1",
            "-hwaccel",
            "auto",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            video_file.as_posix(),
            "-vf",
            f"scale_npp={width}:{height},hwdownload,format=nv12",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-threads",
            "4",
            "-i",
            video_file.as_posix(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-",
        ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    video_stream, err = process.communicate()
    if process.returncode != 0:
        if use_gpu:
            logger.warning("Caught ffmpeg runtime error with `use_gpu=True` option, falling back to CPU.")
            return get_frames_from_ffmpeg(video_file, width, height, use_gpu=False)
        logger.exception(f"FFmpeg error: {err.decode('utf-8')}")
        return None
    return np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])


@dataclass
class VideoFrameExtractionStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that extracts frames from videos into numpy arrays.

    This stage handles video frame extraction using either FFmpeg (CPU/GPU) or PyNvCodec,
    converting video content into standardized frame arrays for downstream processing.
    """
    output_hw: tuple[int, int] = (27, 48)
    batch_size: int = 64
    decoder_mode: str = "pynvc"
    pynvc_frame_extractor: PyNvcFrameExtractor | None = None
    verbose: bool = False

    @property
    def name(self) -> str:
        return "frame_extraction"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """
        if self.decoder_mode == "pynvc":
            self.pynvc_frame_extractor = PyNvcFrameExtractor(
                width=self.output_hw[1],
                height=self.output_hw[0],
                batch_size=self.batch_size,
            )
    

    def process(self, task: VideoTask) -> VideoTask:
        assert self.pynvc_frame_extractor is not None, "PyNvCodec frame extractor is not initialized"
        width, height = self.output_hw
        video = task.data

        if video.source_bytes is None:
            raise ValueError("Video source bytes are not available")
        
        if not video.has_metadata():
            logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
            video.errors["metadata"] = "incomplete"
            return task
        
        with make_pipeline_named_temporary_file(sub_dir="frame_extraction") as video_path:
            with video_path.open("wb") as fp:
                fp.write(video.source_bytes)
            if self.decoder_mode == "pynvc":
                try:
                    video.frame_array = self.pynvc_frame_extractor(video_path).cpu().numpy().astype(np.uint8)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Got exception {e} with PyNvVideoCodec decode, trying ffmpeg CPU fallback")
                    video.frame_array = get_frames_from_ffmpeg(
                        video_path,
                        width=width,
                        height=height,
                        use_gpu=False,
                    )
            else:
                video.frame_array = get_frames_from_ffmpeg(
                    video_path,
                    width=width,
                    height=height,
                    use_gpu=self.decoder_mode == "ffmpeg_gpu",
                )
            if video.frame_array is None:
                logger.error("Frame extraction failed, exiting...")
                return None
            if self.verbose:
                logger.info(f"Loaded video as numpy uint8 array with shape {video.frame_array.shape}")
        return task

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        if self.decoder_mode == "pynvc":
            return Resources(gpu_memory_gb=10)
        else:
            return Resources(cpus=1.0)
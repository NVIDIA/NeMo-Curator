from dataclasses import dataclass
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import VideoTask, Clip, SplitPipeTask
from loguru import logger
import uuid
from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.resources import Resources
from ray_curator.utils.operation_utils import make_pipeline_temporary_dir
import pathlib
import subprocess
from ray_curator.utils import grouping
from ray_curator.tasks import Video
import copy

@dataclass
class ClipTranscodingStage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that transcodes video clips into a standardized format.

    This stage handles the conversion of video clips using FFmpeg, supporting both
    software (libopenh264) and hardware (NVENC) encoding with configurable parameters.
    """
    num_cpus_per_worker: float = 6.0
    encoder: str = "libopenh264"
    encoder_threads: int = 1
    encode_batch_size: int = 16
    nb_streams_per_gpu: int = 3
    use_hwaccel: bool = False
    use_input_bit_rate: bool = False
    num_clips_per_chunk: int = 32
    ffmpeg_verbose: bool = False
    verbose: bool = False
    
    @property
    def name(self) -> str:
        return "clip_transcoding"
    
    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """
        if self.encoder not in {"libopenh264", "h264_nvenc"}:
            error_msg = f"Expected encoder of `libopenh264` or `h264_nvenc`. Got {self.encoder}"
            raise ValueError(error_msg)
    
    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        if self.encoder == "h264_nvenc" or self.use_hwaccel:
            if self.nb_streams_per_gpu > 0:
                return Resources(gpus=1.0 / self.nb_streams_per_gpu)
            else:
                return Resources(gpus=1.0)

        return Resources(cpus=self.num_cpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        if video.source_bytes is None:
            raise ValueError("Video source bytes are not available")
        
        if not video.clips:
            logger.warning(f"No clips to transcode for {video.input_video}. Skipping...")
            video.source_bytes = None
            return task
        
        with make_pipeline_temporary_dir(sub_dir='transcode') as tmp_dir:
            # write video to file
            video_file = tmp_dir / "input.mp4"
            video_file.write_bytes(video.source_bytes)
            force_pix_fmt = video.is_10_bit_color() or False

            # use input video bit-rate
            use_bit_rate = None
            if self.use_input_bit_rate:
                use_bit_rate = str(video.metadata.bit_rate_k) + "K"

            # extract clips in batches
            for i in range(0, len(video.clips), self.encode_batch_size):
                batch = video.clips[i : i + self.encode_batch_size]
                self._extract_clips(
                    tmp_dir,
                    video_file.name,
                    force_pix_fmt=force_pix_fmt,
                    use_bit_rate=use_bit_rate,
                    clips=batch,
                    input_video=str(video.input_video),
                )
        
        # we are done with source_bytes
        video.source_bytes = None

        # TODO log_stats

        # Consider craking into smaller chunks of clips
        output_tasks = []
        clip_durations = [clip.duration for clip in video.clips]
        if len(clip_durations) > 0:
            logger.info(
                f"video {video.input_video} has {len(video.clips)} "
                f"clips and weight={video.weight:.2f}; "
                f"min-clip={min(clip_durations):.2f}s, "
                f"max-clip={max(clip_durations):.1f}s.",
            )
        clip_chunks = list(
            grouping.split_by_chunk_size(
                video.clips,
                self.num_clips_per_chunk * 8,
                lambda x: int(x.span[1] - x.span[0]),
            ),
        )
        for idx in range(len(clip_chunks)):
            # create subtask for each video task
            subtask = VideoTask(
                task_id=f"{task.task_id}_chunk_{idx}",
                dataset_name=task.dataset_name,
                data=Video(
                    input_video=video.input_video,
                    metadata=video.metadata,
                    clips=clip_chunks[idx],
                    num_total_clips=len(video.clips),
                    num_clip_chunks=len(clip_chunks),
                    clip_chunk_index=idx,
                    errors=copy.deepcopy(video.errors),
                ),
                # stage_perf=copy.deepcopy(task.stage_perf),
            )
            # if idx > 0:
            #     for stats in subtask.stage_perf.values():
            #         stats.reset()
            if self.verbose:
                logger.info(
                    f"Spawning subtask {idx} with {len(subtask.video.clips)} clips and weight={subtask.weight:.2f}",
                )
            output_tasks.append(subtask)
        logger.info(f"Creating {len(clip_chunks)} tasks for downstream from {video.input_video}.")


        return output_tasks
    
    def _extract_clips(
            self,
            working_dir: pathlib.Path,
            video_filename: str,
            *,
            force_pix_fmt: bool,
            use_bit_rate: str,
            clips: list[Clip],
            input_video: str,
    ) -> None:
# construct ffmpeg command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning" if self.ffmpeg_verbose else "error",
        ]

        for i, clip in enumerate(clips):
            # set decoder threads
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self.encoder_threads)])
            # hwaccel needs to specified before each input
            if self.use_hwaccel:
                if self.encoder == "h264_nvenc":
                    command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
                else:
                    command.extend(["-hwaccel", "auto"])
            start_s, end_s = clip.span
            command.extend(
                [
                    "-ss",
                    str(start_s),
                    "-to",
                    str(end_s),
                    "-i",
                    video_filename,
                    "-map",
                    f"{i}:v:0",
                    "-c:v",
                    self.encoder,
                ],
            )
            if use_bit_rate is not None:
                command.extend(
                    [
                        "-b:v",
                        use_bit_rate,
                    ],
                )
            if self.encoder == "h264_nvenc":
                # IMPORTANT! these settings are necessary for high quality!
                command.extend(
                    [
                        "-rc:v",
                        "vbr",
                        "-cq:v",
                        "21",
                        "-tune",
                        "hq",
                        "-b_ref_mode",
                        "middle",
                        "-temporal-aq",
                        "1",
                        "-rc-lookahead",
                        "20",
                        "-spatial-aq",
                        "1",
                    ],
                )
                # To fix `10 bit encode not supported` error
                if force_pix_fmt:
                    command.extend(["-pix_fmt", "yuv420p"])
            if self.resources.gpus > 0:
                command.extend(["-threads", str(1)])
            else:
                command.extend(["-threads", str(self.encoder_threads)])
            command.extend(
                [
                    "-map",
                    f"{i}:a:0?",
                    "-c:a",
                    "copy",
                    f"{clip.uuid}.mp4",
                ],
            )

        # run ffmpeg command
        try:
            output = subprocess.check_output(  # noqa: S603
                command, cwd=working_dir, stderr=subprocess.STDOUT
            )
            if output and self.ffmpeg_verbose:
                logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed with return code {e.returncode} on {input_video}")
            logger.warning(f"Command: {' '.join(command)}")
            if e.output:
                logger.warning(f"Error output: {e.output.decode('utf-8')}")
            for clip in clips:
                clip.errors["transcode"] = e.output.decode("utf-8") if e.output else str(e)
            return

        # read clips back into memory
        for clip in clips:
            clip.buffer = (working_dir / f"{clip.uuid}.mp4").read_bytes()

        
@dataclass
class FixedStrideExtractorSrage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that extracts video clips using fixed-length intervals.

    This stage splits videos into clips of specified length and stride, ensuring
    each clip meets minimum length requirements and optionally limiting total clips.
    """
    clip_len_s: float
    clip_stride_s: float
    min_clip_length_s: float
    limit_clips: int

    @property
    def name(self) -> str:
        return "fixed_stride_extractor"
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
    
    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        if video.source_bytes is None:
            raise ValueError("Video source bytes are not available")
        
        if not video.has_metadata():
            logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
            video.errors["metadata"] = "incomplete"
            return task

        if self.limit_clips > 0 and len(video.clips) >= self.limit_clips:
            logger.warning(f"Skipping {video.input_video} because it has already been clipped")
            return task
        
        file = video.input_video
        assert video.metadata.num_frames, "num_frames is not set"
        assert video.metadata.framerate, "framerate is not set"
        duration = video.metadata.num_frames / video.metadata.framerate if video.metadata.framerate > 0 else -1

        # create clip bounds based on clip_len_s and clip_stride_s
        clip_start = 0.0
        clip_bounds: list[tuple[float, float]] = []
        while clip_start < duration:
            clip_end = min(clip_start + self.clip_len_s, duration)
            if (clip_end - clip_start) >= self.min_clip_length_s:
                clip_bounds.append((clip_start, clip_end))
            clip_start += self.clip_stride_s

        for span in clip_bounds:
            start_event = int(span[0] * video.metadata.framerate)
            end_event = int(span[1] * video.metadata.framerate)
            clip = Clip(
                uuid=uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{file}_{start_event}_{end_event}",
                ),
                source_video=str(file),
                span=span,
            )
            video.clips.append(clip)

        logger.info(f"Extracted {len(task.data.clips)} clips from {task.data.input_video}")
        return task
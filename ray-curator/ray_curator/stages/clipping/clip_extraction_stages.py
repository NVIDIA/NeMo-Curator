from dataclasses import dataclass
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import VideoTask, Clip
from loguru import logger
import uuid

@dataclass
class FixedStrideExtractorSrage(ProcessingStage[VideoTask, VideoTask]):
    """Stage that extracts clips from a video."""
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
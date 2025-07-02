import argparse

from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader.video_download import VideoDownloadStage
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.clipping.clip_extraction_stages import FixedStrideExtractorSrage, ClipTranscodingStage
from ray_curator.stages.clipping.frame_extraction import VideoFrameExtractionStage

def create_video_splitting_pipeline(args: argparse.Namespace) -> Pipeline:
    
    # Define pipeline
    pipeline = Pipeline(name="video_splitting", description="Split videos into clips")

    # Add stages
    pipeline.add_stage(VideoDownloadStage(folder_path=args.video_folder))

    if args.splitting_algorithm == "fixed_stride":
        pipeline.add_stage(
            FixedStrideExtractorSrage(
                clip_len_s=args.fixed_stride_split_duration,
                clip_stride_s=args.fixed_stride_split_duration,
                min_clip_length_s=args.fixed_stride_min_clip_length_s,
                limit_clips=args.limit_clips,
            )
        )
    elif args.splitting_algorithm == "transnetv2":
        pipeline.add_stage(
            VideoFrameExtractionStage(
                decoder_mode=args.transnetv2_frame_decoder_mode,
            )
        )
        # TODO: add transnetv2 stage
    else:
        raise ValueError(f"Splitting algorithm {args.splitting_algorithm} not supported")

    pipeline.add_stage(ClipTranscodingStage(
        num_cpus_per_worker=args.transcode_cpus_per_worker,
        encoder=args.transcode_encoder,
        encoder_threads=args.transcode_encoder_threads,
        encode_batch_size=args.transcode_ffmpeg_batch_size,
        use_hwaccel=args.transcode_use_hwaccel,
        use_input_bit_rate=args.transcode_use_input_video_bit_rate,
        num_clips_per_chunk=args.clip_re_chunk_size,
        verbose=args.verbose,
        # log_stats=args.perf_profile,
    ))
    return pipeline


def main(args: argparse.Namespace) -> None:

    pipeline = create_video_splitting_pipeline(args)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--video_folder", type=str, default='~/Videos')
    parser.add_argument("--verbose", action="store_true", default=False)

    # Splitting parameters
    parser.add_argument(
        "--splitting-algorithm", 
        type=str, 
        default="fixed_stride", 
        choices=["fixed_stride", "transnetv2"],
        help="Splitting algorithm to use",
    )
    parser.add_argument(
        "--fixed-stride-split-duration", 
        type=float, 
        default=10.0,
        help="Duration of clips (in seconds) generated from the fixed stride splitting stage.",
    )
    parser.add_argument(
        "--fixed-stride-min-clip-length-s", 
        type=float, 
        default=2.0,
        help="Minimum length of clips (in seconds) for fixed stride splitting stage.",
    )
    parser.add_argument(
        "--limit-clips", 
        type=int, 
        default=0, 
        help="limit number of clips from each input video to process. 0 means no limit.",
    )
    parser.add_argument(
        "--transnetv2-frame-decoder-mode",
        type=str,
        default="pynvc",
        choices=["pynvc", "ffmpeg_gpu", "ffmpeg_cpu"],
        help="Choose between ffmpeg on CPU or GPU or PyNvVideoCodec for video decode.",
    )

    # Transcoding arguments
    parser.add_argument(
        "--transcode-cpus-per-worker",
        type=float,
        default=6.0,
        help="Number of CPU threads per worker. The stage uses a batched ffmpeg "
        "commandline with batch_size (-transcode-ffmpeg-batch-size) of ~64 and per-batch thread count of 1.",
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "h264_nvenc"],
        help="Codec for transcoding clips; None to skip transocding.",
    )
    parser.add_argument(
        "--transcode-encoder-threads",
        type=int,
        default=1,
        help="Number of threads per ffmpeg encoding sub-command for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-ffmpeg-batch-size",
        type=int,
        default=16,
        help="FFMPEG batchsize for transcoding clips. Each clip/sub-command in "
        "the batch uses --transcode-encoder-threads number of CPU threads",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Whether to use cuda acceleration for decoding in transcoding stage.",
    )
    parser.add_argument(
        "--transcode-use-input-video-bit-rate",
        action="store_true",
        default=False,
        help="Whether to use input video's bit rate for encoding clips.",
    )
    parser.add_argument(
        "--clip-re-chunk-size",
        type=int,
        default=32,
        help="Number of clips per chunk after transcoding stage.",
    )

    # parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--output_format", type=str, required=True)
    # parser.add_argument("--executor", type=str, required=True)
    args = parser.parse_args()
    main(args)
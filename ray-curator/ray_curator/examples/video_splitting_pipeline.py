import argparse

from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader.video_download import VideoDownloadStage
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.clipping.clip_extraction_stages import FixedStrideExtractorSrage
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
    else:
        raise ValueError(f"Splitting algorithm {args.splitting_algorithm} not supported")

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
    # Video folder
    parser.add_argument("--video_folder", type=str, default='~/Videos')
    # Splitting parameters
    parser.add_argument("--splitting_algorithm", type=str, default="fixed_stride")
    parser.add_argument("--fixed_stride_split_duration", type=float, default=10.0)
    parser.add_argument("--fixed_stride_min_clip_length_s", type=float, default=2.0)
    parser.add_argument("--limit_clips", type=int, default=0)

    # parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--output_format", type=str, required=True)
    # parser.add_argument("--executor", type=str, required=True)
    args = parser.parse_args()
    main(args)
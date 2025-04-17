import os


def get_files(input_dir: str) -> list[str]:
    # TODO: update with a more robust fsspec method
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

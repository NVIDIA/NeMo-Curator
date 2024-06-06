import os

import pandas as pd
import pytest
from dask import dataframe as dd
from dask.dataframe.utils import assert_eq

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import write_to_disk
from nemo_curator.utils.file_utils import separate_by_metadata


@pytest.fixture
def tmp_path_w_data(tmp_path):
    def _write_data(num_files):
        out_path = tmp_path / "jsonl"
        df = pd.DataFrame(
            {
                "id": [1, 2, 300, 4, -1],
                "text": ["abc", "aba", "abb", "aba", "abc"],
                "metadata": ["doc", "code", "test", "code", "doc"],
            }
        )
        dfs = []
        for i in range(num_files):
            partition = df.copy()
            partition["filename"] = f"f{i}.jsonl"
            dfs.append(partition)

        df = dd.concat(dfs)
        write_to_disk(
            df=df,
            output_file_dir=str(out_path),
            write_to_filename=True,
            output_type="jsonl",
        )
        return out_path

    return _write_data


@pytest.mark.parametrize(
    "backend", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
)
class TestMetadataSep:

    @pytest.mark.parametrize("files_per_partition", [1, 3])
    def test_metadatasep(self, tmp_path_w_data, files_per_partition, backend):
        data_dir = tmp_path_w_data(5)
        output_dir = data_dir / "metadata_sep"
        df = DocumentDataset.read_json(
            str(data_dir),
            backend=backend,
            files_per_partition=files_per_partition,
            add_filename=True,
        ).df
        separate_by_metadata(
            df=df,
            output_dir=str(output_dir),
            metadata_field="metadata",
            output_type="jsonl",
        ).compute()

        found_metadata = set(os.listdir(output_dir))
        expected_metadata = {"code", "doc", "test"}
        assert found_metadata == expected_metadata

        dfs = []
        for metadata in expected_metadata:
            meta_df = DocumentDataset.read_json(
                str(output_dir / metadata),
                backend=backend,
                files_per_partition=1,
                add_filename=True,
            ).df
            dfs.append(meta_df)
        got_df = dd.concat(dfs)
        assert_eq(got_df, df, check_index=False)

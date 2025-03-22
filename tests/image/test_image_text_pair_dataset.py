# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

# These imports should only work on GPU systems
cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")

MaybeToTensor = gpu_only_import("timm.data.transforms", "MaybeToTensor")
ImageTextPairDataset = gpu_only_import_from(
    "nemo_curator.datasets.image_text_pair_dataset", "ImageTextPairDataset"
)
TimmImageEmbedder = gpu_only_import_from(
    "nemo_curator.image.embedders.timm", "TimmImageEmbedder"
)
NsfwClassifier = gpu_only_import_from(
    "nemo_curator.image.classifiers.nsfw", "NsfwClassifier"
)
AestheticClassifier = gpu_only_import_from(
    "nemo_curator.image.classifiers.aesthetic", "AestheticClassifier"
)


# Common fixtures for all test classes
@pytest.fixture
def sample_data_path():
    """Returns the path to the sample data file."""
    return Path(__file__).parent.parent / "image_data" / "00000.tar"


# Add a helper function to create proper mock transforms
def create_mock_transforms():
    """Create a mock Compose object with the required transforms structure."""
    return Compose(
        [
            Resize(
                224,
                interpolation=InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            CenterCrop(224),
            MaybeToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture
def temp_dataset_dir():
    """Creates and returns a temporary directory for dataset operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_mock_metadata(sample_id="0", caption="Sample caption"):
    """Create a mock metadata DataFrame for testing."""
    return cudf.DataFrame({"id": [sample_id], "caption": [caption]})


# Helper classes for mocking models and classifiers
class MockTimmModel(torch.nn.Module):
    """A mock image embedding model for testing."""

    def __init__(self):
        super().__init__()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.ones((batch_size, 512), device=x.device)


class MockClassifier:
    """A mock image classifier for testing."""

    def __init__(self, name="mock_classifier"):
        self.model_name = name
        self.pred_column = f"{name}_score"
        self.pred_type = "float32"

    def load_model(self, device):
        return lambda x: torch.ones((x.shape[0], 1), device=device) * 0.5

    def postprocess(self, series):
        return series


class MockTarFileInfo:
    """Mock TarInfo for testing."""

    def __init__(self, name):
        self.name = name
        self.size = 100


def create_mock_tar_content(member_name, content_type="image"):
    """Create mock content for a tar file member based on its type."""
    if content_type == "image":
        return b"fake_image_data"
    elif content_type == "text":
        return b"This is a fake caption"
    elif content_type == "json":
        return json.dumps({"id": member_name.split(".")[0]}).encode()
    return b""


# Helper function to use with mock.patch wrapping
def fsspec_open_wrapper(path, mode="rb", **kwargs):
    """Wrapper for fsspec.open that creates parent directories if needed for writing."""
    if "w" in mode:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode)


class TestImageTextPairDatasetBase:
    """Tests for the basic functionality of the ImageTextPairDataset class."""

    @pytest.mark.gpu
    def test_init_basic(self):
        """Test basic initialization of ImageTextPairDataset."""
        # Create mock data
        metadata = mock.MagicMock()
        tar_files = ["file1.tar", "file2.tar"]

        # Initialize dataset
        dataset = ImageTextPairDataset("test/path", metadata, tar_files, "id")

        # Verify attributes
        assert dataset.path == "test/path"
        assert dataset.metadata is metadata
        assert dataset.tar_files == tar_files
        assert dataset.id_col == "id"

    @pytest.mark.gpu
    def test_get_tar_files(self):
        """Test the _get_tar_files static method."""
        with mock.patch(
            "nemo_curator.datasets.image_text_pair_dataset.open_files"
        ) as mock_open_files:
            # Setup mock return values
            mock_file1 = mock.MagicMock()
            mock_file1.path = "path/to/00000.tar"
            mock_file2 = mock.MagicMock()
            mock_file2.path = "path/to/00001.tar"
            mock_open_files.return_value = [mock_file1, mock_file2]

            # Call the method
            tar_files = ImageTextPairDataset._get_tar_files("path/to")

            # Verify the result
            assert tar_files == ["path/to/00000.tar", "path/to/00001.tar"]
            mock_open_files.assert_called_once_with("path/to/*.tar")

    @pytest.mark.gpu
    def test_name_partition(self):
        """Test the _name_partition static method."""
        # Test with default parameters
        assert ImageTextPairDataset._name_partition(42) == "00042.parquet"

        # Test with temp flag
        assert (
            ImageTextPairDataset._name_partition(42, temp=True) == "temp_00042.parquet"
        )

        # Test with custom max_shards
        assert ImageTextPairDataset._name_partition(42, max_shards=3) == "042.parquet"

        # Test with custom extension
        assert ImageTextPairDataset._name_partition(42, ext="tar") == "00042.tar"

    @pytest.mark.gpu
    def test_sort_partition(self):
        """Test the _sort_partition static method."""
        # Create test data
        df = cudf.DataFrame({"id": ["3", "1", "2"], "value": ["c", "a", "b"]})

        # Call the method
        sorted_df = ImageTextPairDataset._sort_partition(df, id_col="id")

        # Verify the result
        expected_ids = ["1", "2", "3"]
        expected_values = ["a", "b", "c"]
        assert sorted_df["id"].to_pandas().tolist() == expected_ids
        assert sorted_df["value"].to_pandas().tolist() == expected_values
        assert sorted_df.index.to_pandas().tolist() == [0, 1, 2]

    @pytest.mark.gpu
    def test_combine_id(self):
        """Test the _combine_id static method."""
        # Test with default parameters
        combined_id = ImageTextPairDataset._combine_id(42, 7)
        assert combined_id == "000420007"

        # Test with custom max_shards and max_samples_per_shard
        combined_id = ImageTextPairDataset._combine_id(
            42, 7, max_shards=3, max_samples_per_shard=2
        )
        assert combined_id == "04207"

    @pytest.mark.gpu
    def test_filter_valid_members(self):
        """Test the _filter_valid_members static method."""
        # Create mock members
        member1 = mock.MagicMock()
        member1.name = "000001.jpg"

        member2 = mock.MagicMock()
        member2.name = "000002.txt"

        member3 = mock.MagicMock()
        member3.name = "000003.json"

        members = [member1, member2, member3]
        valid_ids = {1, 3}

        # Call the method
        filtered_members = ImageTextPairDataset._filter_valid_members(
            members, valid_ids
        )

        # Verify the result
        assert len(filtered_members) == 2
        assert filtered_members[0].name == "000001.jpg"
        assert filtered_members[1].name == "000003.json"

    @pytest.mark.gpu
    def test_from_webdataset(self, sample_data_path, temp_dataset_dir):
        """Test the from_webdataset class method with actual sample data."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a mock metadata Parquet file
        metadata_df = pd.DataFrame(
            {
                "id": ["0"],
                "caption": [
                    "A wine bottle outfitted with two forks in its cork and a duck head on top."
                ],
            }
        )
        metadata_path = Path(temp_dataset_dir) / "00000.parquet"
        metadata_df.to_parquet(metadata_path)

        # Instead of mocking the _sort_partition method, modify the test approach
        with mock.patch("dask_cudf.read_parquet") as mock_read_parquet:
            # Create a mock DataFrame that doesn't require map_partitions
            mock_df = mock.MagicMock()
            mock_df.columns = ["id", "caption"]
            mock_read_parquet.return_value = mock_df

            # Call the method
            dataset = ImageTextPairDataset.from_webdataset(temp_dataset_dir, "id")

            # Verify the result
            assert dataset.path == temp_dataset_dir
            assert dataset.id_col == "id"
            assert len(dataset.tar_files) == 1
            assert dataset.tar_files[0].endswith("00000.tar")

    @pytest.mark.gpu
    def test_save_metadata(self, temp_dataset_dir):
        """Test the save_metadata method."""
        # Create test data using a mock dask_cudf DataFrame

        # Create a regular DataFrame
        df = cudf.DataFrame(
            {
                "id": ["0", "1"],
                "caption": ["caption 0", "caption 1"],
                "extra": ["e0", "e1"],
            }
        )

        # Mock a dask DataFrame
        metadata = mock.MagicMock(spec=dd.DataFrame)
        metadata.columns = df.columns

        # Create a dataset with the test data
        tar_files = ["path/to/00000.tar"]
        dataset = ImageTextPairDataset("test/path", metadata, tar_files, "id")

        # Mock to_parquet to avoid actually writing files
        metadata.to_parquet = mock.MagicMock()

        # Call the method to save all columns
        dataset.save_metadata(temp_dataset_dir)

        # Verify that to_parquet was called with the right parameters
        metadata.to_parquet.assert_called_once_with(
            temp_dataset_dir, name_function=ImageTextPairDataset._name_partition
        )

        # Test saving specific columns
        metadata.reset_mock()
        # Mock the selection operation
        metadata.__getitem__.return_value = metadata

        dataset.save_metadata(temp_dataset_dir, columns=["id", "caption"])

        # Verify that __getitem__ was called with the right columns
        metadata.__getitem__.assert_called_once_with(["id", "caption"])

    @pytest.mark.gpu
    def test_integration_with_sample_data(self, sample_data_path, temp_dataset_dir):
        """Integration test with the actual sample data file."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a metadata Parquet file based on the contents of the tar file
        # Extract metadata from the tar file
        with tarfile.open(temp_tar_path, "r") as tar:
            json_files = [f for f in tar.getmembers() if f.name.endswith(".json")]
            if json_files:
                metadata_json = json.loads(
                    tar.extractfile(json_files[0]).read().decode("utf-8")
                )
                sample_id = metadata_json.get("id", "0")
            else:
                sample_id = "0"

        metadata_df = pd.DataFrame(
            {"id": [sample_id], "caption": ["Sample caption"], "filter_col": [True]}
        )

        metadata_path = Path(temp_dataset_dir) / "00000.parquet"
        metadata_df.to_parquet(metadata_path)

        # Create dataset
        with mock.patch("dask_cudf.read_parquet") as mock_read_parquet:
            # Create a mock DataFrame that's already sorted and doesn't need map_partitions
            mock_df = mock.MagicMock()
            mock_df.columns = ["id", "caption", "filter_col"]
            mock_read_parquet.return_value = mock_df

            dataset = ImageTextPairDataset.from_webdataset(temp_dataset_dir, "id")

        # Test save_metadata
        output_dir = Path(temp_dataset_dir) / "output"
        os.makedirs(output_dir, exist_ok=True)

        # Mock to_parquet to avoid actually writing files
        dataset.metadata.to_parquet = mock.MagicMock()
        dataset.save_metadata(output_dir)

        # Verify to_parquet was called with the right parameters
        dataset.metadata.to_parquet.assert_called_once_with(
            output_dir, name_function=ImageTextPairDataset._name_partition
        )


class TestImageTextPairDatasetEmbedderIntegration:
    """Tests for integration between ImageTextPairDataset and image embedders."""

    @pytest.mark.gpu
    def test_dataset_embedder_integration(self, sample_data_path, temp_dataset_dir):
        """Test the integration between ImageTextPairDataset and TimmImageEmbedder."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a mock metadata Parquet file
        metadata_df = pd.DataFrame(
            {
                "id": ["0"],
                "caption": [
                    "A wine bottle outfitted with two forks in its cork and a duck head on top."
                ],
            }
        )
        metadata_path = Path(temp_dataset_dir) / "00000.parquet"
        metadata_df.to_parquet(metadata_path)

        # Setup mocks for dataset and embedder
        with (
            mock.patch("dask_cudf.read_parquet") as mock_read_parquet,
            mock.patch("timm.create_model") as mock_create_model,
            mock.patch("timm.data.create_transform") as mock_create_transform,
            mock.patch("timm.data.resolve_data_config") as mock_resolve_config,
            mock.patch.object(
                TimmImageEmbedder, "load_dataset_shard"
            ) as mock_load_dataset_shard,
        ):
            # Configure mocks
            mock_model = MockTimmModel()
            mock_create_model.return_value = mock_model
            mock_model.pretrained_cfg = {}
            mock_resolve_config.return_value = {}
            mock_create_transform.return_value = create_mock_transforms()

            # Create a mock DataFrame that doesn't require map_partitions
            mock_df = mock.MagicMock()
            mock_df.columns = ["id", "caption"]
            mock_read_parquet.return_value = mock_df

            # Mock the dataset generator to yield a single batch
            batch = torch.ones((1, 3, 224, 224), device="cuda")
            metadata = [{"id": "0", "caption": "Test caption"}]
            mock_load_dataset_shard.return_value = [(batch, metadata)]

            # Load the dataset
            dataset = ImageTextPairDataset.from_webdataset(temp_dataset_dir, "id")

            # Create the embedder with a test classifier
            mock_classifier = MockClassifier()
            embedder = TimmImageEmbedder(
                model_name="resnet18",
                batch_size=1,
                image_embedding_column="embeddings",
                classifiers=[mock_classifier],
            )

            # Mock the result DataFrame to include expected columns after embedding
            result_mock_df = mock.MagicMock()
            result_mock_df.columns = [
                "id",
                "caption",
                "embeddings",
                f"{mock_classifier.model_name}_score",
            ]
            mock_df.map_partitions.return_value.map_partitions.return_value = (
                result_mock_df
            )

            # Call the embedder on the dataset
            with mock.patch(
                "nemo_curator.image.embedders.base.load_object_on_worker",
                side_effect=lambda name, fn, args: (
                    MockTimmModel()
                    if name == "resnet18"
                    else (lambda x: torch.ones((x.shape[0], 1), device="cuda") * 0.5)
                ),
            ):
                result_dataset = embedder(dataset)

            # Verify the result
            assert result_dataset is not None
            assert result_dataset.path == dataset.path
            assert result_dataset.id_col == dataset.id_col
            assert result_dataset.tar_files == dataset.tar_files

            # The embeddings column should be added to the metadata
            assert "embeddings" in result_dataset.metadata.columns

            # The classifier score column should be added to the metadata
            assert (
                f"{mock_classifier.model_name}_score" in result_dataset.metadata.columns
            )

    @pytest.mark.gpu
    def test_dataset_with_multiple_classifiers(
        self, sample_data_path, temp_dataset_dir
    ):
        """Test using the dataset with multiple image classifiers."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a mock metadata Parquet file
        metadata_df = pd.DataFrame(
            {
                "id": ["0"],
                "caption": [
                    "A wine bottle outfitted with two forks in its cork and a duck head on top."
                ],
            }
        )
        metadata_path = Path(temp_dataset_dir) / "00000.parquet"
        metadata_df.to_parquet(metadata_path)

        # Setup mocks for dataset, embedder and classifiers
        with (
            mock.patch("dask_cudf.read_parquet") as mock_read_parquet,
            mock.patch("timm.create_model") as mock_create_model,
            mock.patch("timm.data.create_transform") as mock_create_transform,
            mock.patch("timm.data.resolve_data_config") as mock_resolve_config,
            mock.patch.object(
                TimmImageEmbedder, "load_dataset_shard"
            ) as mock_load_dataset_shard,
            mock.patch.object(
                NsfwClassifier,
                "load_model",
                return_value=lambda x: torch.ones((x.shape[0], 1), device="cuda") * 0.1,
            ),
            mock.patch.object(
                AestheticClassifier,
                "load_model",
                return_value=lambda x: torch.ones((x.shape[0], 1), device="cuda") * 0.9,
            ),
            mock.patch.object(
                NsfwClassifier, "_get_default_model", return_value="mock_nsfw_model"
            ),
            mock.patch.object(
                AestheticClassifier,
                "_get_default_model",
                return_value="mock_aesthetic_model",
            ),
        ):
            # Configure mocks
            mock_model = MockTimmModel()
            mock_create_model.return_value = mock_model
            mock_model.pretrained_cfg = {}
            mock_resolve_config.return_value = {}
            mock_create_transform.return_value = create_mock_transforms()

            # Create a mock DataFrame that doesn't require map_partitions
            mock_df = mock.MagicMock()
            mock_df.columns = ["id", "caption"]
            mock_read_parquet.return_value = mock_df

            # Mock the dataset generator to yield a single batch
            batch = torch.ones((1, 3, 224, 224), device="cuda")
            metadata = [{"id": "0", "caption": "Test caption"}]
            mock_load_dataset_shard.return_value = [(batch, metadata)]

            # Load the dataset
            dataset = ImageTextPairDataset.from_webdataset(temp_dataset_dir, "id")

            # Create the embedder with multiple classifiers
            nsfw_classifier = NsfwClassifier()
            aesthetic_classifier = AestheticClassifier()

            embedder = TimmImageEmbedder(
                model_name="resnet18",
                batch_size=1,
                image_embedding_column="embeddings",
                classifiers=[nsfw_classifier, aesthetic_classifier],
            )

            # Mock the result DataFrame to include expected columns after embedding
            result_mock_df = mock.MagicMock()
            result_mock_df.columns = [
                "id",
                "caption",
                "embeddings",
                "nsfw_score",
                "aesthetic_score",
            ]
            mock_df.map_partitions.return_value.map_partitions.return_value = (
                result_mock_df
            )

            # Call the embedder on the dataset
            with mock.patch(
                "nemo_curator.image.embedders.base.load_object_on_worker",
                side_effect=lambda name, fn, args: (
                    MockTimmModel()
                    if name == "resnet18"
                    else (
                        lambda x: torch.ones((x.shape[0], 1), device="cuda")
                        * (0.1 if name == "nsfw_classifier" else 0.9)
                    )
                ),
            ):
                result_dataset = embedder(dataset)

            # Verify the result
            assert result_dataset is not None

            # The embeddings column should be added to the metadata
            assert "embeddings" in result_dataset.metadata.columns

            # Both classifier score columns should be added to the metadata
            assert "nsfw_score" in result_dataset.metadata.columns
            assert "aesthetic_score" in result_dataset.metadata.columns

    @pytest.mark.gpu
    def test_save_and_load_with_embeddings(self, sample_data_path, temp_dataset_dir):
        """Test saving and loading a dataset with embeddings."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a mock metadata DataFrame with embeddings
        embeddings = [[0.1] * 512]  # Mock embeddings
        metadata_df = cudf.DataFrame(
            {"id": ["0"], "caption": ["A sample caption"], "embeddings": embeddings}
        )

        # Convert to Dask-cuDF DataFrame to support name_function parameter
        dask_metadata_df = dask_cudf.from_cudf(metadata_df, npartitions=1)

        # Create the dataset
        dataset = ImageTextPairDataset(
            temp_dataset_dir, dask_metadata_df, [str(temp_tar_path)], "id"
        )

        # Save the metadata
        output_dir = os.path.join(temp_dataset_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_metadata(output_dir)

        # Verify the saved file
        output_parquet = os.path.join(output_dir, "00000.parquet")
        assert os.path.exists(output_parquet)

        # Load the saved metadata and verify
        loaded_df = pd.read_parquet(output_parquet)
        assert "id" in loaded_df.columns
        assert "caption" in loaded_df.columns
        assert "embeddings" in loaded_df.columns

        # Verify that the embeddings were saved correctly
        assert len(loaded_df["embeddings"][0]) == 512


class TestImageTextPairDatasetConversion:
    """Tests for the WebDataset conversion functionality."""

    @pytest.mark.gpu
    def test_to_webdataset_mock(self, temp_dataset_dir):
        """Test to_webdataset method using mocks to control all IO operations."""
        # Create a sample dataset
        metadata = cudf.DataFrame(
            {
                "id": ["0", "1", "2"],
                "caption": ["caption 0", "caption 1", "caption 2"],
                "filter_col": [True, True, False],  # One sample should be filtered out
            }
        )

        # Convert to Dask-cuDF DataFrame to support name_function parameter
        dask_metadata = dask_cudf.from_cudf(metadata, npartitions=1)

        tar_files = [os.path.join(temp_dataset_dir, "00000.tar")]
        dataset = ImageTextPairDataset(temp_dataset_dir, dask_metadata, tar_files, "id")

        # Create mock tar files and Parquet files
        output_dir = os.path.join(temp_dataset_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create and write a temp parquet file
        metadata.to_pandas().to_parquet(
            os.path.join(temp_dataset_dir, "temp_00000.parquet")
        )

        # Mock the file contents
        mock_members = [
            MockTarFileInfo("000000.jpg"),
            MockTarFileInfo("000000.txt"),
            MockTarFileInfo("000000.json"),
            MockTarFileInfo("000001.jpg"),
            MockTarFileInfo("000001.txt"),
            MockTarFileInfo("000001.json"),
            MockTarFileInfo("000002.jpg"),
            MockTarFileInfo("000002.txt"),
            MockTarFileInfo("000002.json"),
        ]

        # Mock internal methods
        with (
            mock.patch.object(
                ImageTextPairDataset, "_get_eligible_samples"
            ) as mock_get_eligible_samples,
            mock.patch("tarfile.open", autospec=True) as mock_tarfile_open,
            mock.patch("fsspec.open", autospec=True) as mock_fsspec_open,
        ):
            # Mock the tarfile
            mock_tar = mock.MagicMock()
            mock_tar.getmembers.return_value = mock_members

            # Set up extractfile to return different content based on extension
            def mock_extractfile(member):
                ext = member.name.split(".")[-1]
                content_type = {"jpg": "image", "txt": "text", "json": "json"}[ext]
                mock_file = mock.MagicMock()
                mock_file.read.return_value = create_mock_tar_content(
                    member.name, content_type
                )
                return mock_file

            mock_tar.extractfile = mock_extractfile
            mock_tarfile_open.return_value = mock_tar

            # Mock the filesystem opening with a proper file-like object
            class MockFileContext:
                def __init__(self, path, mode):
                    self.path = path
                    self.mode = mode
                    self.buffer = io.BytesIO()

                def __enter__(self):
                    return self.buffer

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            mock_fsspec_open.side_effect = lambda path, mode, **kwargs: MockFileContext(
                path, mode
            )

            # Mock the eligible samples generator
            filtered_df = (
                metadata[metadata["filter_col"]].reset_index(drop=True).to_pandas()
            )
            mock_get_eligible_samples.return_value = [
                (
                    filtered_df,
                    # Return tar entries for the two samples that pass the filter
                    [
                        (
                            mock_members[0],
                            create_mock_tar_content("000000.jpg", "image"),
                        ),
                        (
                            mock_members[1],
                            create_mock_tar_content("000000.txt", "text"),
                        ),
                        (
                            mock_members[2],
                            create_mock_tar_content("000000.json", "json"),
                        ),
                        (
                            mock_members[3],
                            create_mock_tar_content("000001.jpg", "image"),
                        ),
                        (
                            mock_members[4],
                            create_mock_tar_content("000001.txt", "text"),
                        ),
                        (
                            mock_members[5],
                            create_mock_tar_content("000001.json", "json"),
                        ),
                    ],
                )
            ]

            # Call to_webdataset
            dataset.to_webdataset(
                output_dir,
                filter_column="filter_col",
                samples_per_shard=2,
                old_id_col="original_id",
            )

            # Verify the results
            # Check that fsspec.open was called for the output files
            assert (
                mock_fsspec_open.call_count >= 2
            )  # At least once for parquet and once for tar

            # Verify tarfile operations
            mock_tarfile_open.assert_called()
            mock_tar.addfile.assert_called()

            # Check number of calls to addfile (should be 6 for 2 samples * 3 files per sample)
            assert mock_tar.addfile.call_count >= 6

    @pytest.mark.gpu
    def test_to_webdataset_integration(self, sample_data_path, temp_dataset_dir):
        """Test to_webdataset method with actual file operations but mocked dataset content."""
        # Copy the sample tar file to the temp directory
        temp_tar_path = Path(temp_dataset_dir) / "00000.tar"
        shutil.copy(sample_data_path, temp_tar_path)

        # Create a metadata DataFrame
        metadata_df = cudf.DataFrame(
            {"id": ["0"], "caption": ["A sample caption"], "filter_col": [True]}
        )

        # Convert to Dask-cuDF DataFrame to support name_function parameter
        dask_metadata_df = dask_cudf.from_cudf(metadata_df, npartitions=1)

        # Create the dataset
        dataset = ImageTextPairDataset(
            temp_dataset_dir, dask_metadata_df, [str(temp_tar_path)], "id"
        )

        # Set up the output directory
        output_dir = os.path.join(temp_dataset_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create temp parquet file that would normally be created by to_webdataset
        temp_parquet_path = os.path.join(output_dir, "temp_00000.parquet")
        metadata_df.to_pandas().to_parquet(temp_parquet_path)

        # Mock _get_eligible_samples to use our real data
        with (
            mock.patch.object(
                ImageTextPairDataset, "_get_eligible_samples"
            ) as mock_get_eligible_samples,
            mock.patch("fsspec.open", wraps=fsspec_open_wrapper) as mock_fsspec_open,
        ):
            # Extract real tar content
            with tarfile.open(temp_tar_path, "r") as tar:
                members = tar.getmembers()
                content = []
                for member in members:
                    member_content = (
                        tar.extractfile(member).read()
                        if tar.extractfile(member)
                        else b""
                    )
                    content.append((member, member_content))

            # Mock the eligible samples generator to return our real data
            mock_get_eligible_samples.return_value = [
                (metadata_df.to_pandas(), content)
            ]

            # Call to_webdataset
            dataset.to_webdataset(
                output_dir,
                filter_column="filter_col",
                samples_per_shard=1,
                old_id_col="original_id",
            )

            # Verify the results
            output_tar_path = os.path.join(output_dir, "00000.tar")
            output_parquet_path = os.path.join(output_dir, "00000.parquet")

            # Check that files were created
            assert os.path.exists(output_tar_path)
            assert os.path.exists(output_parquet_path)

            # Check the parquet file content
            output_df = pd.read_parquet(output_parquet_path)
            assert "id" in output_df.columns
            assert (
                "original_id" in output_df.columns
            )  # should have preserved the old ID
            assert output_df.shape[0] == 1  # One sample

            # Check the tar file structure
            with tarfile.open(output_tar_path, "r") as tar:
                tar_members = tar.getmembers()
                # The number of members should match the original (3 files per sample)
                assert len(tar_members) == len(members)

                # Check that all expected extensions are present
                extensions = [m.name.split(".")[-1] for m in tar_members]
                assert "jpg" in extensions
                assert "txt" in extensions
                assert "json" in extensions

    @pytest.mark.gpu
    def test_combine_id_consistency(self):
        """Test that the _combine_id method creates consistent and parseable IDs."""
        # Test several combinations to ensure the method is robust
        for shard_id in range(5):
            for sample_id in range(10):
                combined_id = ImageTextPairDataset._combine_id(shard_id, sample_id)

                # The ID should be a string with 9 characters (5 for shard, 4 for sample)
                assert isinstance(combined_id, str)
                assert len(combined_id) == 9

                # The first 5 digits should represent the shard ID with leading zeros
                assert combined_id[:5] == f"{shard_id:05d}"

                # The last 4 digits should represent the sample ID with leading zeros
                assert combined_id[5:] == f"{sample_id:04d}"

    @pytest.mark.gpu
    def test_filter_valid_members_real_data(self, sample_data_path):
        """Test the _filter_valid_members method with real tar data."""
        # Extract member info from the real tar file
        with tarfile.open(sample_data_path, "r") as tar:
            all_members = tar.getmembers()

            # Get the real IDs from the tar file
            member_ids = set()
            for member in all_members:
                file_id = int(member.name.split(".")[0])
                member_ids.add(file_id)

            # Test filtering with all IDs
            filtered_members = ImageTextPairDataset._filter_valid_members(
                all_members, member_ids
            )
            assert len(filtered_members) == len(all_members)

            # Test filtering with a subset of IDs
            if len(member_ids) > 0:
                subset_ids = {next(iter(member_ids))}  # Get the first ID
                filtered_subset = ImageTextPairDataset._filter_valid_members(
                    all_members, subset_ids
                )

                # The filtered list should contain only members with the specified ID
                assert all(
                    int(member.name.split(".")[0]) in subset_ids
                    for member in filtered_subset
                )

                # There should be multiple files (jpg, txt, json) for the same ID
                if len(filtered_subset) > 1:
                    extensions = [
                        member.name.split(".")[-1] for member in filtered_subset
                    ]
                    assert len(set(extensions)) > 1

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

import os
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nemo_curator.utils.import_utils import gpu_only_import_from

# These imports should only work on GPU systems
NsfwClassifier = gpu_only_import_from(
    "nemo_curator.image.classifiers.nsfw", "NsfwClassifier"
)
NSFWModel = gpu_only_import_from("nemo_curator.image.classifiers.nsfw", "NSFWModel")
ImageTextPairDataset = gpu_only_import_from(
    "nemo_curator.datasets.image_text_pair_dataset", "ImageTextPairDataset"
)
TimmImageEmbedder = gpu_only_import_from(
    "nemo_curator.image.embedders.timm", "TimmImageEmbedder"
)


# Test initialization parameters
@pytest.mark.gpu
def test_init_defaults():
    """Test that NsfwClassifier initializes with default parameters correctly."""
    classifier = NsfwClassifier()
    assert classifier.model_name == "nsfw_classifier"
    assert classifier.embedding_column == "image_embedding"
    assert classifier.pred_column == "nsfw_score"
    assert classifier.pred_type == float
    assert classifier.batch_size == -1
    assert classifier.embedding_size == 768


@pytest.mark.gpu
def test_init_custom_params():
    """Test that NsfwClassifier initializes with custom parameters correctly."""
    classifier = NsfwClassifier(
        embedding_column="custom_embedding",
        pred_column="custom_score",
        batch_size=64,
        model_path="/custom/path/model.pth",
    )
    assert classifier.model_name == "nsfw_classifier"
    assert classifier.embedding_column == "custom_embedding"
    assert classifier.pred_column == "custom_score"
    assert classifier.pred_type == float
    assert classifier.batch_size == 64
    assert classifier.embedding_size == 768
    assert classifier.model_path == "/custom/path/model.pth"


# Test _get_default_model method
@pytest.mark.gpu
def test_get_default_model():
    """Test that _get_default_model returns the correct path and downloads if needed."""
    with (
        mock.patch("os.path.exists", return_value=False),
        mock.patch("requests.get") as mock_get,
        mock.patch("builtins.open", mock.mock_open()),
        mock.patch("zipfile.ZipFile") as mock_zipfile,
        mock.patch("os.makedirs"),
    ):

        # Mock the response
        mock_response = mock.MagicMock()
        mock_response.content = b"mock_content"
        mock_get.return_value = mock_response

        model_path = NsfwClassifier._get_default_model()

        # Check that the model path contains the expected filename
        assert "clip_autokeras_binary_nsfw.pth" in model_path
        # Check that request.get was called with the correct URL
        mock_get.assert_called_once()
        assert "CLIP-based-NSFW-Detector" in mock_get.call_args[0][0]
        # Check that the zip file was extracted
        mock_zipfile.assert_called_once()
        mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once()


# Test load_model method
@pytest.mark.gpu
def test_load_model(gpu_client):
    """Test that the load_model method correctly loads the NSFW model."""
    classifier = NsfwClassifier()

    # Create a simple mock to test the evaluation mode and device assignment
    class MockNSFWModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eval_called = False
            self.to_device = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def forward(self, x):
            return torch.ones((x.shape[0], 1), device=x.device)

    # Mock torch.load and NSFWModel
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("nemo_curator.image.classifiers.nsfw.NSFWModel") as mock_model_class,
    ):

        # Configure the mock to return our custom module
        mock_model_instance = MockNSFWModel()
        mock_model_class.return_value = mock_model_instance

        # Call the method
        model = classifier.load_model(device="cuda")

        # Check that NSFWModel was initialized and sent to the right device
        mock_model_class.assert_called_once()

        # Check that model.eval() was called
        assert mock_model_instance.eval_called

        # Check that model.to() was called with the right device
        assert mock_model_instance.to_device == "cuda"

        # Check that load_state_dict was called
        assert hasattr(mock_model_instance, "state_dict")

        # Verify it's the same instance we mocked
        assert model is mock_model_instance


# Test _configure_forward method
@pytest.mark.gpu
def test_configure_forward(gpu_client):
    """Test that the _configure_forward method correctly modifies the forward method."""
    classifier = NsfwClassifier()

    # Create a mock model with a forward method
    class MockModel:
        def __init__(self):
            self.forward_called = False
            self.forward_args = None
            self.forward_kwargs = None

        def forward(self, *args, **kwargs):
            self.forward_called = True
            self.forward_args = args
            self.forward_kwargs = kwargs
            # Return a tensor with a single dimension (shape [batch_size, 1])
            return torch.ones((2, 1), device="cuda")

    mock_model = MockModel()

    # Configure the forward method
    configured_model = classifier._configure_forward(mock_model)

    # Call the configured forward method
    test_input = torch.ones((2, 768), device="cuda")
    output = configured_model.forward(test_input)

    # Check that the original forward was called
    assert mock_model.forward_called

    # Check that the output has the expected shape (should be squeezed to [batch_size])
    assert output.shape == (2,)  # Squeezed from [2, 1] to [2]


# Test postprocess method
@pytest.mark.gpu
def test_postprocess(gpu_client):
    """Test that the postprocess method correctly extracts values from list series."""
    classifier = NsfwClassifier()

    # Create a mock series with list data
    mock_series = mock.MagicMock()
    mock_leaves = mock.MagicMock()
    mock_series.list.leaves = mock_leaves
    mock_series.index = [0, 1, 2]

    # Call the postprocess method
    result = classifier.postprocess(mock_series)

    # Check that leaves was accessed and index was preserved
    assert result == mock_leaves
    assert result.index == mock_series.index


# Test complete workflow with mocks
@pytest.mark.gpu
def test_classifier_with_embedder_workflow(gpu_client):
    """Test integration of NsfwClassifier with TimmImageEmbedder."""
    # Get the test sample dataset path
    sample_tar_path = Path(__file__).parent.parent.parent / "image_data" / "00000.tar"

    # Ensure the file exists
    assert sample_tar_path.exists(), f"Sample dataset not found at {sample_tar_path}"

    # Setup a minimal embedder with our classifier
    data_dir = str(sample_tar_path.parent)

    # Create the classifier
    classifier = NsfwClassifier(pred_column="nsfw_score")

    # Mock the classifier's load_model to return a simple function
    def mock_load_model(device):
        return lambda x: torch.ones((x.shape[0]), device=x.device) * 0.25

    classifier.load_model = mock_load_model

    # Mock function to replace load_object_on_worker
    def mock_load_object(name, fn, args):
        # Simply call the function with the args
        return fn(**args)

    # Create the image embedder with our classifier
    with (
        mock.patch(
            "nemo_curator.image.embedders.timm.TimmImageEmbedder.load_embedding_model"
        ) as mock_load_embedding,
        mock.patch(
            "nemo_curator.image.embedders.timm.TimmImageEmbedder.load_dataset_shard"
        ) as mock_load_dataset,
        mock.patch(
            "nemo_curator.image.embedders.base.load_object_on_worker",
            side_effect=mock_load_object,
        ),
    ):

        # Create a mock embedding model
        def mock_embedding_fn(x):
            # Return embeddings with the correct shape for the classifier
            return torch.ones((x.shape[0], 768), device=x.device) * 0.5

        mock_load_embedding.return_value = mock_embedding_fn

        # Mock the dataset loading function
        def mock_load_dataset_shard(tar_path):
            # Create a small batch of fake images and metadata
            batch = torch.ones((1, 3, 224, 224), device="cuda")
            metadata = [{"id": "0"}]
            yield batch, metadata

        mock_load_dataset.side_effect = mock_load_dataset_shard

        embedder = TimmImageEmbedder(
            model_name="resnet18", batch_size=1, classifiers=[classifier]
        )

        # Create a mock dataset with metadata that includes our ID
        mock_metadata = mock.MagicMock()
        mock_metadata.dtypes.to_dict.return_value = {
            "id": "object",
            "caption": "object",
        }

        # Create a mock for map_partitions that processes our test data
        def mock_map_partitions(func, *args, **kwargs):
            import cudf

            # Create a DataFrame with test data
            df = cudf.DataFrame({"id": ["0"], "caption": ["test caption"]})
            # Call the function directly with our test data
            result = func(df, [str(sample_tar_path)], "id", {"number": 0})
            # Return a mock that just gives back the processed DataFrame
            mock_result = mock.MagicMock()
            mock_result.__iter__ = lambda self: iter([result])
            return mock_result

        mock_metadata.map_partitions.side_effect = mock_map_partitions

        # Create a test dataset
        test_dataset = mock.MagicMock(spec=ImageTextPairDataset)
        test_dataset.path = data_dir
        test_dataset.id_col = "id"
        test_dataset.tar_files = [str(sample_tar_path)]
        test_dataset.metadata = mock_metadata

        # Call the embedder
        with mock.patch(
            "nemo_curator.image.embedders.base.ImageTextPairDataset"
        ) as mock_dataset_class:
            # Configure the mock to return our dataset
            mock_dataset_class.return_value = test_dataset

            # Call the embedder (which will call our classifier)
            result = embedder(test_dataset)

            # Verify the classifier column was added to the metadata
            assert mock_metadata.map_partitions.called
            meta_arg = mock_metadata.map_partitions.call_args[1]["meta"]
            assert "nsfw_score" in meta_arg


# Test NSFWModel architecture
@pytest.mark.gpu
def test_nsfw_model_architecture(gpu_client):
    """Test that the NSFWModel has the expected architecture."""
    # Initialize the model
    model = NSFWModel().to("cuda")

    # Test with mock input
    test_input = torch.ones((2, 768), device="cuda")
    with torch.no_grad():
        output = model(test_input)

    # Check output shape
    assert output.shape == (2, 1)

    # Check output range (should be between 0 and 1 because of sigmoid)
    assert torch.all(output >= 0) and torch.all(output <= 1)


# Test NSFWModel with real parameters
@pytest.mark.gpu
def test_nsfw_model_with_random_weights(gpu_client):
    """Test that the NSFWModel correctly processes inputs with random weights."""
    # Initialize the model
    model = NSFWModel().to("cuda")

    # Test with random input
    test_input = torch.randn((2, 768), device="cuda")
    with torch.no_grad():
        output = model(test_input)

    # Check output shape
    assert output.shape == (2, 1)

    # Check output range (should be between 0 and 1 because of sigmoid)
    assert torch.all(output >= 0) and torch.all(output <= 1)

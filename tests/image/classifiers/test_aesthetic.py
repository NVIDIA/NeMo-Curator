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
from pathlib import Path
from unittest import mock

import cudf
import numpy as np
import pytest
import torch

from nemo_curator.utils.import_utils import gpu_only_import_from

# These imports should only work on GPU systems
AestheticClassifier = gpu_only_import_from(
    "nemo_curator.image.classifiers.aesthetic", "AestheticClassifier"
)
MLP = gpu_only_import_from("nemo_curator.image.classifiers.aesthetic", "MLP")
ImageTextPairDataset = gpu_only_import_from(
    "nemo_curator.datasets.image_text_pair_dataset", "ImageTextPairDataset"
)
TimmImageEmbedder = gpu_only_import_from(
    "nemo_curator.image.embedders.timm", "TimmImageEmbedder"
)


# Test initialization parameters
@pytest.mark.gpu
def test_init_defaults():
    """Test that AestheticClassifier initializes with default parameters correctly."""
    classifier = AestheticClassifier()
    assert classifier.model_name == "aesthetic_classifier"
    assert classifier.embedding_column == "image_embedding"
    assert classifier.pred_column == "aesthetic_score"
    assert classifier.pred_type == float
    assert classifier.batch_size == -1
    assert classifier.embedding_size == 768


@pytest.mark.gpu
def test_init_custom_params():
    """Test that AestheticClassifier initializes with custom parameters correctly."""
    classifier = AestheticClassifier(
        embedding_column="custom_embedding",
        pred_column="custom_score",
        batch_size=64,
        model_path="/custom/path/model.pth",
    )
    assert classifier.model_name == "aesthetic_classifier"
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
        mock.patch("os.makedirs"),
    ):

        # Mock the response
        mock_response = mock.MagicMock()
        mock_response.content = b"mock_content"
        mock_get.return_value = mock_response

        model_path = AestheticClassifier._get_default_model()

        # Check that the model path contains the expected filename
        assert "sac+logos+ava1-l14-linearMSE.pth" in model_path
        # Check that request.get was called with the correct URL
        mock_get.assert_called_once()
        assert "improved-aesthetic-predictor" in mock_get.call_args[0][0]


# Test load_model method
@pytest.mark.gpu
def test_load_model(gpu_client):
    """Test that the load_model method correctly loads the aesthetic model."""
    classifier = AestheticClassifier()

    # Create a simple mock to test the evaluation mode and device assignment
    class MockModule(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.input_size = input_size
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

    # Mock torch.load and MLP
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("nemo_curator.image.classifiers.aesthetic.MLP") as mock_mlp_class,
    ):

        # Configure the mock to return our custom module
        mock_mlp_instance = MockModule(classifier.embedding_size)
        mock_mlp_class.return_value = mock_mlp_instance

        # Call the method
        model = classifier.load_model(device="cuda")

        # Check that MLP was initialized with the correct embedding size
        mock_mlp_class.assert_called_once_with(classifier.embedding_size)

        # Check that model.eval() was called
        assert mock_mlp_instance.eval_called

        # Check that model.to() was called with the right device
        assert mock_mlp_instance.to_device == "cuda"

        # Verify it's the same instance we mocked
        assert model is mock_mlp_instance


# Test _configure_forward method
@pytest.mark.gpu
def test_configure_forward(gpu_client):
    """Test that the _configure_forward method correctly modifies the forward method."""
    classifier = AestheticClassifier()

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
    classifier = AestheticClassifier()

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
    """Test integration of AestheticClassifier with TimmImageEmbedder."""
    # Get the test sample dataset path
    sample_tar_path = Path(__file__).parent.parent.parent / "image_data" / "00000.tar"

    # Ensure the file exists
    assert sample_tar_path.exists(), f"Sample dataset not found at {sample_tar_path}"

    # Setup a minimal embedder with our classifier
    data_dir = str(sample_tar_path.parent)

    # Create the classifier
    classifier = AestheticClassifier(pred_column="aesthetic_score")

    # Mock the classifier's load_model to return a simple function
    def mock_load_model(device):
        return lambda x: torch.ones((x.shape[0]), device=x.device) * 0.75

    classifier.load_model = mock_load_model

    # Create the image embedder with our classifier
    with mock.patch(
        "nemo_curator.image.embedders.timm.TimmImageEmbedder.load_embedding_model"
    ) as mock_load_embedding:
        # Create a mock embedding model
        def mock_embedding_fn(x):
            # Return embeddings with the correct shape for the classifier
            return torch.ones((x.shape[0], 768), device=x.device) * 0.5

        mock_load_embedding.return_value = mock_embedding_fn

        embedder = TimmImageEmbedder(
            model_name="resnet18", batch_size=1, classifiers=[classifier]
        )

        # Create a mock ImageTextPairDataset
        mock_dataset = mock.MagicMock(spec=ImageTextPairDataset)
        mock_dataset.path = data_dir
        mock_dataset.id_col = "id"
        mock_dataset.tar_files = [str(sample_tar_path)]

        # Create a mock metadata
        mock_metadata = mock.MagicMock()
        mock_metadata.dtypes.to_dict.return_value = {
            "id": "object",
            "caption": "object",
        }
        mock_metadata.map_partitions.return_value = mock.MagicMock()
        mock_dataset.metadata = mock_metadata

        # Mock the load_dataset_shard to yield known data
        def mock_load_dataset_shard(tar_path):
            # Create a small batch of fake images and metadata
            batch = torch.ones((1, 3, 224, 224), device="cuda")
            metadata = [{"id": "test_id"}]
            yield batch, metadata

        embedder.load_dataset_shard = mock_load_dataset_shard

        # Call the embedder, which should process the classifier
        with mock.patch("nemo_curator.image.embedders.base.ImageTextPairDataset"):
            result = embedder(mock_dataset)

            # Check that map_partitions was called
            mock_metadata.map_partitions.assert_called_once()

            # Verify the classifier's column was added to the metadata
            meta_arg = mock_metadata.map_partitions.call_args[1]["meta"]
            assert "aesthetic_score" in meta_arg
            assert meta_arg["aesthetic_score"] == float


# Test with mock cuDF Series for postprocessing
@pytest.mark.gpu
def test_postprocess_with_real_data(gpu_client):
    """Test postprocess with a simulated cuDF Series containing list data."""
    try:
        import cupy as cp
        from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar

        classifier = AestheticClassifier()

        # Create a sample array of scores
        scores = cp.array([[0.75], [0.85], [0.65]])

        # Create a DataFrame with index
        df = cudf.DataFrame({"id": ["a", "b", "c"]})

        # Create a list series from the array
        list_series = create_list_series_from_1d_or_2d_ar(scores, index=df.index)

        # Apply postprocessing
        result = classifier.postprocess(list_series)

        # Verify the result is correct
        assert len(result) == 3
        assert result.index.to_pandas().tolist() == [0, 1, 2]

        # Convert to host memory for comparison
        result_values = result.to_pandas().values
        assert np.isclose(result_values[0], 0.75)
        assert np.isclose(result_values[1], 0.85)
        assert np.isclose(result_values[2], 0.65)

    except (ImportError, AttributeError):
        # Skip this test if running without GPU or required libraries
        pytest.skip("This test requires cuDF and related GPU libraries")

.. _data-curator-image-embedding:

=========================
Image Embedders
=========================

--------------------
Timm Image Embedder
--------------------

PyTorch Image Models (timm) is a library containing SOTA computer vision models.
Many of these models are useful in generating image embeddings for modules in NeMo Curator.
NeMo Curator provides easy support for all these models through ``TimmImageEmbedder``.
This module can also automatically convert the image transformations from PyTorch transformations to DALI transformations in the supported models.

.. code-block:: python

    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
        autocast=False,
    )

    dataset_with_embeddings = embedding_model(dataset)

    dataset_with_embeddings.save_metadata()

Here, we load a dataset in and compute the image embeddings using ``vit_large_patch14_clip_quickgelu_224.openai``.
This model is the base for NeMo Curator's aesthetic classifier, so we use it for this example.

A more thorough list of parameters can be found in the `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/embedders.html>`_.

Under the hood, the image embedding model performs the following operations:

1. Load a shard of metadata (a `.parquet` file) onto each GPU you have available using Dask-cuDF.
1. Load a copy of `vit_large_patch14_clip_quickgelu_224.openai` onto each GPU.
1. Repeatedly load images into batches of size `batch_size` onto each GPU with a given threads per worker (`num_threads_per_worker`) using DALI.
1. The model is run on the batch (without `torch.autocast()` since `autocast=False`).
1. The output embeddings of the model are normalized since `normalize_embeddings=True`.


------------------------
Custom Image Embedder
------------------------

To write your own custom embedder, you inherit from ``nemo_curator.image.embedders.ImageEmbedder`` and override two methods as shown below:

.. code-block:: python

    from nemo_curator.image.embedders import ImageEmbedder

    class MyCustomEmbedder(ImageEmbedder):

        def load_dataset_shard(self, tar_path: str) -> Iterable:
            # Implement me!
            pass

        def load_embedding_model(self, device: str) -> Callable:
            # Implement me!
            pass


* ``load_dataset_shard()`` will take in a path to a tar file and return an iterable over the shard. The iterable should return a tuple of (a batch of data, metadata).
  The batch of data can be of any form. It will be directly passed to the model returned by ``load_embedding_model()``.
  The metadata should be a dictionary of metadata, with a field corresponding to the ``id_col`` of the dataset.
  In our example, the metadata should include a value for ``"key"``.
* ``load_embedding_model()`` will take a device and return a callable object.
  This callable will take as input a batch of data produced by ``load_dataset_shard()``.
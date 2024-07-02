# SemDeDup Pipeline

This pipeline is used to cluster and deduplicate data points based on their embeddings.
Please edit `config.yaml` to configure the pipeline and run it using the following commands.

```sh
bash scripts/end_to_end_script.sh
```

## Pipeline Explanation

1) Modify `config.yaml`

2) Compute embeddings:
    ```sh
    python compute_embeddings.py
    ```
    **Input:** `config.embeddings.input_data_dir/*.jsonl` and output from step (2)

    **Output:** Embedding  parquet files in the embedding directory, including `text` and `adlr`

3) Clustering
    ```sh
    python clustering.py
    ```
    **Input:** Output from step (3)

    **Output:** Under `{config.root}/{config.clustering.save_loc}` directory, including:

        - `kmeans_centroids.npy`
        - `embs_by_nearest_center` directory, containing `nearest_cent={x}` where x ranges from 0 to `num_clusters - 1`
        - Parquet files within `embs_by_nearest_center/nearest_cent={x}` containing the data points in each cluster

4) Sort the clusters
    ```sh
    python sort_clusters.py
    ```
    **Input:** Output from step (4)

    **Output:** Under `config.root/centroids/sorted`,
                `cluster_x.npy` (where x ranges from 0 to `num_clusters - 1`)

5) Run SemDeDup
    This helps in deduplicating the data points within each cluster using semantic similarity
    and generates a deduplicated dataset
    ```sh
    python semdedup.py
    ```
    **Input:** Output from step (5)

    **Output:** Under `config.root/centroids/dataframes`,
                `cluster_x.parquet`(where x ranges from 0 to `num_clusters - 1`)

6) Extract deduplicated data
    ```sh
    python extract_dedup_data.py
    ```
    **Input:** Output from step (6)

    **Output:** `config.root/centroids/results.csv`

<!-- 8) Analysis in `pynb/eda_dups.ipynb` -->

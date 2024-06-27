# SemDeDup Pipeline

## Pipeline

0) Modify `configs.yaml`

1) Add IDs to the dataset
    ```sh
    python create_idmap.py
    ```
    **Input:**  `config.embeddings.input_data_dir`
    **Output:** `id_mapping.csv` and `adlr_ids` in `config.embeddings.input_data_dir`

3) Compute embeddings:
    ```sh
    python compute_embeddings.py
    ```
    **Input:** `config.embeddings.input_data_dir/*.jsonl` and output from step (2)
    **Output:** Embedding  parquet files in the embedding directory, including `text` and `adlr`

4) Clustering
    ```sh
    python clustering.py
    ```
    **Input:** Output from step (3)
    **Output:** Under `{config.root}/{config.clustering.save_loc}` directory, including:
        - `kmeans_centroids.npy`
        - `embs_by_nearest_center` directory, containing `nearest_cent={x}` where x ranges from 0 to `num_clusters - 1`
        - Parquet files within `embs_by_nearest_center/nearest_cent={x}` containing the data points in each cluster

5) Sort the clusters
    ```sh
    python sort_clusters.py
    ```
    **Input:** Output from step (4)
    **Output:** Under `config.root/centroids/sorted`, `cluster_x.npy` where x ranges from 0 to `num_clusters - 1`

6) Run SemDeDup
    ```sh
    python semdedup.py
    ```
    **Input:** Output from step (5)
    **Output:** Under `config.root/centroids/dataframes`, `cluster_x.parquet` where x ranges from 0 to `num_clusters - 1`

7) Extract deduplicated data
    ```sh
    python extract_dedup_data.py
    ```
    **Input:** Output from step (6)
    **Output:** `config.root/centroids/results.csv`

<!-- 8) Analysis in `pynb/eda_dups.ipynb` -->

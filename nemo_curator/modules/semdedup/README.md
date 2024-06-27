# SemDeDup Pipeline

## Pipeline

0) Modify `configs.yaml`

1) Prepare the dataset
    ```sh
    cd {root}/datasets/prospector-lm
    ln -s /datasets/prospector-lm/cleaned_exact_dedup_all_cc/ cleaned_exact_dedup_all_cc
    ```

2) Add IDs to the dataset
    ```sh
    python create_idmap.py
    ```
    **Input:** `/datasets/prospector-lm/cleaned_exact_dedup_all_cc/*.jsonl`
    **Output:** `id_mapping.csv` and `adlr_ids` in `{root}/datasets/prospector-lm`

3) Compute embeddings:
    ```sh
    python compute_embeddings_multigpu.py
    ```
    **Input:** `/datasets/prospector-lm/cleaned_exact_dedup_all_cc/*.jsonl` and output from step (2)
    **Output:** Embedding memmap files in the embedding directory, including `text` and `adlr`

4) Clustering
    ```sh
    python clustering.py
    ```
    **Input:** Output from step (3)
    **Output:** Under `results/centroids` directory, including:
        - `kmeans_centroids.npy`
        - `embs_by_nearest_center` directory, containing `nearest_cent={x}` where x ranges from 0 to `num_clusters - 1`
        - Parquet files within `embs_by_nearest_center/nearest_cent={x}` containing the data points in each cluster

5) Sort the clusters
    ```sh
    python sort_clusters.py
    ```
    **Input:** Output from step (4)
    **Output:** Under `results/centroids/sorted`, `cluster_x.npy` where x ranges from 0 to `num_clusters - 1`

6) Run SemDeDup
    ```sh
    python semdedup.py
    ```
    **Input:** Output from step (5)
    **Output:** Under `results/centroids/dataframes`, `cluster_x.pkl` where x ranges from 0 to `num_clusters - 1`

7) Extract deduplicated data
    ```sh
    python extract_dedup_data.py
    ```
    **Input:** Output from step (6)
    **Output:** `results/centroids/results.csv`

8) Analysis in `pynb/eda_dups.ipynb`

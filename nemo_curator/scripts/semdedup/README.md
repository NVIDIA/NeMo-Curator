# SemDeDup Pipeline

This pipeline is used to cluster and deduplicate data points based on their embeddings.
Please edit "semdedup_config.yaml" to configure the pipeline and run it using the following commands.


## Pipeline Steps

1) Modify  "semdedup_config.yaml"

2) Compute embeddings:
    ```sh
    python compute_embeddings.py --input-data-dir "$INPUT_DATA_DIR" --input-file-type "json" --config-file "semdedup_config.yaml"
    ```
    **Input:** `config.embeddings.input_data_dir/*.jsonl` and output from step (2)

    **Output:** Embedding  parquet files in the embedding directory, including `text` and `adlr`

3) Clustering
    ```sh
    python clustering.py --config-file "semdedup_config.yaml"
    ```
    **Input:** Output from step (3)

    **Output:** Under `{config.root}/{config.clustering.save_loc}` directory, including:

        - `kmeans_centroids.npy`
        - `embs_by_nearest_center` directory, containing `nearest_cent={x}` where x ranges from 0 to `num_clusters - 1`
        - Parquet files within `embs_by_nearest_center/nearest_cent={x}` containing the data points in each cluster


3) Extract deduplicated data
    ```sh
    python extract_dedup_data.py --config-file "semdedup_config.yaml"
    ```
    **Input:** Output from step (3)

    **Output:** `config.root/centroids/results.csv`

## End to End Script

python3 end_to_end_example.py --input-data-dir "/datasets/semdedup/c4/realnewslike/modified" --input-file-type "jsonl" --config-file "semdedup_config.yaml"

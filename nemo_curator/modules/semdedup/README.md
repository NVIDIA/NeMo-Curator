# SemDeDup Pipeline

## Setup
Package installations:
```
conda create -n rapids-24.04 -c rapidsai -c conda-forge -c nvidia  rapids=24.04 python=3.10 cuda-version=12.0 pytorch
pip install transformers
pip install torch torchvision
pip install submitit
pip install jupyter
```

## Pipeline

0) Modify configs.yaml

1) Prepare the dataset
    ```
    cd {root}/datasets/prospector-lm
    ln -s /datasets/prospector-lm/cleaned_exact_dedup_all_cc/ cleaned_exact_dedup_all_cc
    ```

2) Add IDs to the dataset
    ```
    python create_idmap.py
    ```

    in: ```/datasets/prospector-lm/cleaned_exact_dedup_all_cc/*.jsonl```

    out: ```id_mapping.csv``` and ```adlr_ids``` in {root}/datasets/prospector-lm


3) Compute embeddings:
    ```
    python compute_embeddings_multigpu.py
    ```
    in: ```/datasets/prospector-lm/cleaned_exact_dedup_all_cc/*.jsonl``` and out from (2)

    out: embedding memmap files in embedding dir including ```text``` and ```adlr```

4) Clustering
    ```
    python clustering_dask_crossfit.py
    ```
    in: out from (3)

    out: under ```results/centroids dir```, including
        ```kmeans_centroids.npy```,
        ```embs_by_nearest_center dir```
        Where ```embs_by_nearest_center``` contains ```nearest_cent={x}``` where x from 0 to num_clusters - 1
        and ```embs_by_nearest_center/nearest_cent={x}``` contains parquet files which contain the data points in the cluster

5) Sort the clusters
    ```
    python sort_clusters.py
    ```
    in: out from (4)

    out:
        under ```results/centroids/sorted```, ```cluster_x.npy``` where x from 0 to num_clusters - 1

6) Run semdedup
    ```
    python semdedup.py
    ```
    in: out from (5)

    out:
    under ```results/centroids/dataframes```,
    ```cluster_x.pkl``` where x from 0 to num_clusters - 1

7) Extract dedup data
    ```
    python extract_dedup_data.py
    ```
    in: out from (6)

    out: ```results/centroids/results.csv```

8) in ```pynb/eda_dups.ipynb```

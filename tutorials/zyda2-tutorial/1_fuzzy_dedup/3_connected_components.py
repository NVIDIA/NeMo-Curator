import logging
import os
import time

from nemo_curator.cache import initialize_cache_directory
from nemo_curator.modules.fuzzy_dedup import ConnectedComponents
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")
    # Input
    buckets_to_edges_out = os.path.join(DATA_BASE, "fuzzy/buckets_to_edges/data")
    initialize_cache_directory(buckets_to_edges_out)

    # Output
    connected_component_base_output_path = os.path.join(DATA_BASE, "fuzzy/cc")
    connected_component_output_path = os.path.join(
        connected_component_base_output_path, "connected_components.parquet"
    )

    # Relevant parameters
    input_id_field = "id"

    t0 = time.time()

    components_stage = ConnectedComponents(
        id_column=input_id_field,
        false_positive_check=False,
    )

    # Load and run connected components
    components_stage.cc_workflow(output_path=connected_component_output_path)
    logging.info(f"Time taken for Connected Components: {time.time() - t0:.2f} s")

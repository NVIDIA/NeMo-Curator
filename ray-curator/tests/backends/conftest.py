"""Shared test configuration for Ray Curator tests.

This module provides shared Ray cluster setup and teardown for all test modules.
Using a single Ray instance across tests improves performance while maintaining
proper isolation through Ray's actor/task lifecycle management.
"""

import os

import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster():
    """Set up a shared Ray cluster for all tests in the session.

    This fixture automatically sets up Ray at the beginning of the test session
    and tears it down at the end. It configures Ray with fixed resources for
    consistent testing behavior.
    """
    # Set Ray environment variables for testing
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = "40000"
    os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = "40000"

    # Initialize Ray with fixed resources for consistent testing
    # Use 8 CPUs and 0 GPUs for reproducible behavior
    ray.init(
        num_cpus=8,
        num_gpus=0,
        ignore_reinit_error=True,
        log_to_driver=True,
        local_mode=False,  # Use cluster mode for better testing of distributed features
    )

    # Try to get the actual Ray address more reliably
    try:
        # Get the GCS address which is what we need for RAY_ADDRESS
        ray_address = ray.get_runtime_context().gcs_address
    except Exception:  # noqa: BLE001
        # Fallback to localhost:10001 if we can't get the actual address
        ray_address = "127.0.0.1:10001"

    # Set RAY_ADDRESS so Xenna will connect to our cluster
    os.environ["RAY_ADDRESS"] = ray_address
    print(f"Set RAY_ADDRESS to: {ray_address}")

    yield ray_address

    # Shutdown Ray after all tests complete
    ray.shutdown()

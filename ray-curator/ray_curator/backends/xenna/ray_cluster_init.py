from typing import TYPE_CHECKING

import ray
from loguru import logger

if TYPE_CHECKING:
    import loguru


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    # Initialize a default logger
    return logger


def init_or_connect_to_cluster() -> None:
    """Initialize a new local Ray cluster or connects to an existing one."""
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
    )

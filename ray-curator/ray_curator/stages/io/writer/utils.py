import hashlib


def get_deterministic_hash(inputs: list[str], seed: str = "") -> str:
    """Create a deterministic hash from inputs."""
    combined = "|".join(sorted(inputs)) + "|" + seed
    return hashlib.sha256(combined.encode()).hexdigest()[:12]

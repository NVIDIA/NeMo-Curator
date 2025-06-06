"""Resource requirements for pipeline stages."""

from dataclasses import dataclass


def _get_gpu_memory_gb() -> float:
    """Get GPU memory in GB for the current device."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.total) / (1024**3)  # Convert bytes to GB
    except Exception:
        # Fallback to 24GB if detection fails
        return 24.0


@dataclass
class Resources:
    """Define resource requirements for a processing stage.
    
    Attributes:
        cpus: Number of CPU cores required
        gpu_memory_gb: GPU memory required in GB
        nvdecs: Number of NVDEC units required 
        nvencs: Number of NVENC units required
        entire_gpu: Whether to allocate entire GPU regardless of memory
        gpus: Number of GPUs required (calculated from gpu_memory_gb)
    """

    cpus: float = 1.0
    gpu_memory_gb: float = 0.0
    nvdecs: int = 0
    nvencs: int = 0
    entire_gpu: bool = False
    gpus: float = 0.0

    def __post_init__(self):
        """Calculate GPU count based on memory requirements."""
        if self.entire_gpu:
            self.gpus = 1.0
        elif self.gpu_memory_gb > 0:
            # Get actual GPU memory for current device
            gpu_memory_per_device = _get_gpu_memory_gb()
            # Calculate required GPUs and round to 1 decimal place
            required_gpus = self.gpu_memory_gb / gpu_memory_per_device
            self.gpus = round(required_gpus, 1)
        else:
            self.gpus = 0.0

    @property
    def requires_gpu(self) -> bool:
        """Check if this stage requires GPU resources."""
        return self.gpus > 0 or self.gpu_memory_gb > 0 or self.entire_gpu 

import pynvml
import psutil
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total_mb: int
    vram_free_mb: int
    ram_total_mb: int
    ram_available_mb: int
    cpu_cores: int

def get_hardware_profile() -> HardwareProfile:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()

    vmem = psutil.virtual_memory()

    return HardwareProfile(
        gpu_name=gpu_name if isinstance(gpu_name, str) else gpu_name.decode(),
        vram_total_mb=mem.total // (1024 * 1024),
        vram_free_mb=mem.free // (1024 * 1024),
        ram_total_mb=vmem.total // (1024 * 1024),
        ram_available_mb=vmem.available // (1024 * 1024),
        cpu_cores=psutil.cpu_count(logical=False),
    )

if __name__ == "__main__":
    print(get_hardware_profile())
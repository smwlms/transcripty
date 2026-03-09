"""Hardware detection and optimal settings recommendation."""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_cached_profile: HardwareProfile | None = None


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    cpu: str
    cores: int
    ram_gb: float
    gpu: str | None
    mps: bool
    device: str  # "cuda" / "mps" / "cpu"

    def suggest_settings(self) -> dict:
        """Suggest optimal transcripty settings for this hardware.

        Returns:
            Dict usable as TranscriptyConfig overrides.
        """
        settings: dict = {}

        # Model size based on RAM
        if self.ram_gb >= 16:
            settings["model_size"] = "large-v3"
        elif self.ram_gb >= 8:
            settings["model_size"] = "medium"
        else:
            settings["model_size"] = "small"

        # Compute type based on device
        if self.device == "cuda":
            settings["compute_type"] = "float16"
        elif self.device == "mps":
            # CTranslate2 doesn't support MPS, but pyannote does
            settings["compute_type"] = "int8"
        else:
            settings["compute_type"] = "int8"

        # Workers based on cores
        if self.cores >= 8:
            settings["num_workers"] = 4
        elif self.cores >= 4:
            settings["num_workers"] = 2
        else:
            settings["num_workers"] = 1

        return settings


def _get_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()
    if system == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown"


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        if platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        else:
            # Try /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)
            except Exception:
                pass
    except Exception:
        pass
    return 0.0


def _detect_gpu() -> tuple[str | None, bool, str]:
    """Detect GPU, MPS availability, and best device.

    Returns:
        (gpu_name, mps_available, device)
    """
    gpu = None
    mps = False
    device = "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            device = "cuda"
        elif (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            mps = True
            device = "mps"
    except ImportError:
        pass

    return gpu, mps, device


def detect_hardware() -> HardwareProfile:
    """Detect hardware and return a profile. Result is cached."""
    global _cached_profile
    if _cached_profile is not None:
        return _cached_profile

    cpu = _get_cpu_name()
    cores = os.cpu_count() or 1
    ram_gb = round(_get_ram_gb(), 1)
    gpu, mps, device = _detect_gpu()

    _cached_profile = HardwareProfile(
        cpu=cpu,
        cores=cores,
        ram_gb=ram_gb,
        gpu=gpu,
        mps=mps,
        device=device,
    )

    logger.info(
        "Hardware: %s, %d cores, %.1f GB RAM, device=%s",
        cpu,
        cores,
        ram_gb,
        device,
    )

    return _cached_profile


def reset_cache() -> None:
    """Reset cached hardware profile (for testing)."""
    global _cached_profile
    _cached_profile = None

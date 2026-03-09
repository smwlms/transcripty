"""Tests for hardware detection."""

from transcripty.hardware import HardwareProfile, detect_hardware, reset_cache


def test_hardware_profile_suggest_large_ram():
    profile = HardwareProfile(
        cpu="Test CPU", cores=10, ram_gb=32.0, gpu=None, mps=True, device="mps"
    )
    settings = profile.suggest_settings()
    assert settings["model_size"] == "large-v3"
    assert settings["compute_type"] == "int8"  # MPS → int8 for faster-whisper
    assert settings["num_workers"] == 4


def test_hardware_profile_suggest_medium_ram():
    profile = HardwareProfile(
        cpu="Test CPU", cores=4, ram_gb=12.0, gpu=None, mps=False, device="cpu"
    )
    settings = profile.suggest_settings()
    assert settings["model_size"] == "medium"
    assert settings["num_workers"] == 2


def test_hardware_profile_suggest_small_ram():
    profile = HardwareProfile(
        cpu="Test CPU", cores=2, ram_gb=4.0, gpu=None, mps=False, device="cpu"
    )
    settings = profile.suggest_settings()
    assert settings["model_size"] == "small"
    assert settings["num_workers"] == 1


def test_hardware_profile_cuda():
    profile = HardwareProfile(
        cpu="Test CPU", cores=8, ram_gb=32.0, gpu="NVIDIA RTX 4090", mps=False, device="cuda"
    )
    settings = profile.suggest_settings()
    assert settings["compute_type"] == "float16"


def test_detect_hardware_returns_profile():
    reset_cache()
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.cores > 0
    assert profile.ram_gb > 0
    assert profile.device in ("cpu", "mps", "cuda")


def test_detect_hardware_caching():
    reset_cache()
    p1 = detect_hardware()
    p2 = detect_hardware()
    assert p1 is p2

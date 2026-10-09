"""Device and module loading shared by experiments, notebooks, and tests."""

from functools import cache
from pathlib import Path

import slangpy as spy

REPO_ROOT = Path(__file__).resolve().parents[2]
SHADER_DIR = REPO_ROOT / "shaders"


@cache
def create_device(device_type: spy.DeviceType = spy.DeviceType.automatic) -> spy.Device:
    """Create (once) a GPU device with the repo's shader directory on the include path."""
    return spy.create_device(type=device_type, include_paths=[SHADER_DIR])


def load_module(name: str, device: spy.Device | None = None) -> spy.Module:
    """Load a Slang module from `shaders/`, e.g. ``load_module("hello.slang")``."""
    return spy.Module.load_from_file(device or create_device(), name)

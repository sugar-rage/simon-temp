"""SIMON — Smart Intelligent Mobility & Outdoor Navigator.

Minimal bootstrap entry point.  All orchestration lives in AppController.
See TDR-009 for rationale.
"""

import logging
import signal
import sys
import os
import glob


def _setup_cuda_paths():
    """Dynamically add PyTorch/NVIDIA CUDA DLL paths to the environment."""
    if os.name != "nt":
        return
        
    import site
    import sys
    
    site_packages = []
    if hasattr(site, "getsitepackages"):
        site_packages.extend(site.getsitepackages())
    site_packages.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    
    paths_to_add = []
    for sp in site_packages:
        paths_to_add.append(os.path.join(sp, "torch", "lib"))
        paths_to_add.extend(glob.glob(os.path.join(sp, "nvidia", "*", "bin")))
        paths_to_add.extend(glob.glob(os.path.join(sp, "nvidia", "*", "bin", "x86_64")))
        
    for p in paths_to_add:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

_setup_cuda_paths()

from core.config.loader import load_config
from core.app_controller import AppController


def main() -> None:
    """Bootstrap SIMON: load config → create AppController → run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    app = AppController(config)

    # Graceful shutdown on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: app.stop())

    try:
        app.start()
        app.run()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
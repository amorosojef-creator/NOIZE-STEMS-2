"""Download and verify a separation model from the command line.

  python scripts/install_model.py htdemucs --yes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noize.model_manager import ModelManager  # noqa: E402
from noize.models import MODELS  # noqa: E402
from noize.paths import models_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--yes", action="store_true", help="consent to downloading weights from Meta's Demucs release")
    a = ap.parse_args()
    m = MODELS[a.model]
    print(f"{m.name}: ~{m.approx_mb} MB from dl.fbaipublicfiles.com -> {models_dir()}")
    mgr = ModelManager(models_dir())
    mgr.install(a.model, a.yes, lambda f, s: print(f"\r{f * 100:5.1f}%  {s}", end="", flush=True))
    print("\nverified:", mgr.verify(a.model))
    return 0 if mgr.verify(a.model) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Model registry. Weights are NOT bundled; they are downloaded after consent."""
from __future__ import annotations
from dataclasses import dataclass

ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    stems: tuple[str, ...]
    description: str
    files: tuple[str, ...]          # '<sig>-<sha256 prefix>.th'
    approx_mb: int
    license: str = ("Demucs code: MIT (Meta). Pretrained weights are published by "
                    "Meta alongside the MIT-licensed repository; confirm terms at "
                    "github.com/facebookresearch/demucs.")

    @property
    def urls(self) -> dict[str, str]:
        return {f: ROOT_URL + f for f in self.files}


MODELS: dict[str, ModelInfo] = {
    "htdemucs": ModelInfo(
        "htdemucs", "HT Demucs (4 stems)", ("drums", "bass", "other", "vocals"),
        "Default hybrid-transformer model. Best speed/quality balance.",
        ("955717e8-8726e21a.th",), 80),
    "htdemucs_ft": ModelInfo(
        "htdemucs_ft", "HT Demucs Fine-Tuned (4 stems)", ("drums", "bass", "other", "vocals"),
        "Higher quality; four specialised models, ~4x slower.",
        ("f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th",
         "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th"), 320),
    "htdemucs_6s": ModelInfo(
        "htdemucs_6s", "HT Demucs 6-stem", ("drums", "bass", "other", "vocals", "guitar", "piano"),
        "Adds guitar and piano. The piano stem is known to be weaker.",
        ("5c90dfd2-34c22ccb.th",), 55),
}
DEFAULT_MODEL = "htdemucs"


def get_model_info(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}") from None


def select_model(model_id: str | None, installed: set[str]) -> str:
    """Pick a usable model: requested if installed, else fallback to any installed."""
    if model_id and model_id in MODELS and model_id in installed:
        return model_id
    for mid in MODELS:
        if mid in installed:
            return mid
    raise LookupError("No separation model is installed. Open MODELS and install one.")

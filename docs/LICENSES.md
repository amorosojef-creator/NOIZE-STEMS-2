# Licenses

**NOIZE source code:** MIT (see `LICENSE`).

| Component | License | Notes |
|---|---|---|
| Demucs (code) | MIT, Meta Platforms | github.com/facebookresearch/demucs |
| Demucs pretrained weights | Published by Meta alongside the MIT-licensed repository | **Not bundled.** Downloaded on user request. Confirm current terms upstream before redistributing weights. |
| PyTorch / torchaudio | BSD-3-Clause | |
| PySide6 / Qt | LGPLv3 (Qt for Python) | Dynamically linked; the onedir build keeps Qt as separate DLLs so users can replace them. Provide this notice and the LGPL text with redistributions. |
| NumPy, SciPy | BSD-3-Clause | |
| SoundFile / libsndfile | BSD-3 / LGPL-2.1 | |
| psutil | BSD-3-Clause | |
| imageio-ffmpeg | BSD-2-Clause wrapper | The bundled FFmpeg binary is typically a **GPL** build; if you redistribute NOIZE, review FFmpeg's license obligations or swap in an LGPL build placed at `ffmpeg/ffmpeg.exe` |
| PyInstaller | GPL-2.0 with bootloader exception | Permits distributing the packaged app under its own license |
| julius, einops, openunmix, dora-search, PyYAML, tqdm, lameenc | MIT / BSD / Apache-style | Pulled in by Demucs. Verify with `pip-licenses` before redistributing |

**Brand assets:** the NOIZE emblem and icon are the owner's brand assets, included for this project.

This file is a good-faith summary, not legal advice. Re-verify before distributing binaries.

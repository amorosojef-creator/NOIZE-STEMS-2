# NOIZE v0.1.0 - Feature completion report

Legend: **V-sandbox** = verified by automated tests run in the Linux build sandbox (Python 3.12, newer
library versions than the pins). **CI** = will be verified by the GitHub Actions run on pinned versions.
**UNVERIFIED** = written against the documented APIs but never executed end to end.

## What was actually run
- `pytest -m "not realmodel"`: **55 passed, 2 skipped** (skips: the optional real-model test, and `test_missing_model_gives_clear_error` which needs torch+demucs installed - both run in CI). Real result in sandbox, not a claim.
- Offscreen Qt render of the main window and Models page (screenshot inspected).
- Demucs 4.0.1 source was read to match its real API (`LocalRepo`, `BagOnlyRepo`, `apply_model`, file-name checksums).

## Not run anywhere yet (needs you / Windows / weights)
- **Real neural separation.** PyTorch and the Meta weight host were unavailable in the sandbox, so `DemucsSeparator` has **never executed with real torch+demucs+weights**. Chunking/stitching logic is tested with a mathematical fake model only. Use CI option *real_model_test* or `pytest -m realmodel`.
- **Windows executable / installer.** Not built (sandbox is Linux). The workflow, `noize.spec` and `.iss` are unexecuted; expect to iterate once on first CI run (most likely: PyInstaller hidden imports for torch/dora).
- **Audio playback on real hardware.** `playback.py` could not run (no audio server). Pure-render logic (mix, loop wrap, alignment) is tested; the Qt `QAudioSink` path is UNVERIFIED.
- **CUDA path**, GPU detection with a real GPU, and OOM->CPU retry: UNVERIFIED.
- Real MP3/M4A decode/encode in the pinned environment: tested in sandbox with system FFmpeg only.

## Feature status
| Area | Status |
|---|---|
| Window, nav, dark flat design system, emblem/icon, splash, About | Built; rendering V-sandbox |
| Import: WAV/MP3/FLAC/OGG/AIFF (libsndfile), M4A + video via FFmpeg, validation, drag-drop | V-sandbox (decode); drag-drop UNVERIFIED |
| Demucs separation (htdemucs / ft / 6s), chunked, progress, cancel, OOM fallback | Implemented; **UNVERIFIED with real model** |
| Model manager: consent, download, SHA-256-prefix verify, cancel, remove | V-sandbox with fake server |
| Synchronized playback, M/S/volume/pan, loop, zoom, seek | Logic V-sandbox; audio output UNVERIFIED |
| Waveforms (overview + per-stem lanes, shared timeline zoom/scroll) | V-sandbox render |
| Export WAV/FLAC (16/24-bit), resample, clip protection, MP3 via FFmpeg, custom mix | V-sandbox |
| Vocals/Instrumental (sum of non-vocal stems) | V-sandbox |
| CLEAN (Wiener-style spectral mask + soft gate) | V-sandbox on synthetic tones; perceptual quality on real music UNVERIFIED |
| COMPARE (second model, A/B at same position) | Implemented; UNVERIFIED with real models |
| REMIX | Via lane controls + custom-mix export; V-sandbox |
| BATCH (sequential, error isolation, cancel, reorder) | Queue logic V-sandbox; GUI run UNVERIFIED |
| HISTORY (SQLite) | V-sandbox |
| Settings persistence | V-sandbox |
| GitHub Actions: tests, PyInstaller onedir, exe check, packaged self-test, portable ZIP, Inno installer, artifacts | Written; CI |

## Known limitations / next steps
1. First CI run will likely need small packaging fixes.
2. Windows SmartScreen warning (unsigned).
3. No MDX-Net models yet (Demucs only). No stem-level undo for CLEAN beyond RESTORE.
4. CPU speed figures are unmeasured.
5. Settings page reset requires restart to refresh all controls.

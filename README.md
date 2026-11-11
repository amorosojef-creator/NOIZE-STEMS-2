# NOIZE - AI Stem Extractor for Windows

Free, offline-capable stem separation for DJs, producers and remixers.
Splits a song into **drums / bass / other / vocals** (or 6 stems with guitar and piano), lets you
audition and remix the stems in sync, and exports WAV / FLAC / MP3.
Powered by Meta's open-source **Demucs** (hybrid transformer) models, running locally on your PC.

> **Status:** v0.1.0. The source tree and its automated tests are complete; the Windows
> executable is built by GitHub Actions (see below). Read `docs/FEATURE_REPORT.md` for exactly what
> has and has not been verified.

---

## A. Get a Windows build (no Python needed on your PC)

### 1. Upload the source ZIP to a new GitHub repository
1. Create a free account at github.com and click **New repository** (name it e.g. `noize-stems`; Public or Private).
2. Unzip `NOIZE-STEMS-SOURCE.zip` on your computer.
3. On the empty repository page click **uploading an existing file**, drag the **contents** of the
   unzipped folder in (the folder with `noize.spec`, `README.md`, `.github`, ... must be the
   repository root; make sure the hidden `.github` folder is included), then **Commit changes**.
   - If drag-and-drop skips `.github`, use **Add file -> Create new file**, type
     `.github/workflows/build-windows.yml` as the name and paste the file's contents.

### 2. Enable GitHub Actions
Open the **Actions** tab. If GitHub shows a button "I understand my workflows, go ahead and enable
them", click it. (Private repos: Settings -> Actions -> General -> allow actions.)

### 3. Start the Windows build
Actions -> **Build NOIZE for Windows** -> **Run workflow** -> Run workflow.
Tick *real_model_test* if you also want CI to download htdemucs and run real inference on a short
synthetic signal. The build takes roughly 15-40 minutes (PyTorch is large).

### 4. Download the build artifact
When the run turns green, open it and scroll to **Artifacts**:
- `NOIZE-portable-win64` - ZIP with the portable app (recommended)
- `NOIZE-installer-win64` - Inno Setup installer (best effort; may be absent if that step failed)

If a step fails, open it to read the log - nothing is hidden. The build also runs the unit tests and a
packaged self-test of the built `NOIZE.exe`.

### 5. Extract and launch NOIZE
Unzip `NOIZE-portable-win64.zip` anywhere (e.g. `C:\NOIZE`; avoid very long paths) and double-click
`NOIZE\NOIZE.exe`. Windows SmartScreen may warn because the app is not code-signed: choose
*More info -> Run anyway*. Python is **not** required.

### 6. Install separation models (first run)
Model weights are **not bundled**. Open the **MODELS** tab, pick a model and click **INSTALL**; NOIZE asks
for your permission, downloads from Meta's Demucs release (`dl.fbaipublicfiles.com`), and verifies the
SHA-256 checksum. If you skip this, the first EXTRACT will ask the same question.

| Model | Stems | Download |
|---|---|---|
| HT Demucs (`htdemucs`) | drums, bass, other, vocals | ~80 MB |
| HT Demucs Fine-Tuned (`htdemucs_ft`) | same, higher quality, ~4x slower | ~320 MB |
| HT Demucs 6-stem (`htdemucs_6s`) | + guitar, piano (piano is weak) | ~55 MB |

Models are stored in `%APPDATA%\NOIZE\models`. Offline afterwards.

FFmpeg: a copy is bundled through `imageio-ffmpeg` and enables M4A input, video audio extraction and
MP3 export. WAV/FLAC/MP3/OGG input works without it. See `docs/LICENSES.md` (FFmpeg licensing).

### 7. Extract your first song
1. **EXTRACT** tab -> drop an audio/video file on the big drop area (or **+ ADD FILES**).
2. Choose a model, processor (CPU/Auto) and quality, then press **EXTRACT STEMS**. Progress is shown; **CANCEL** any time.
3. Audition: **PLAY** (or Space), M/S = mute/solo per stem, sliders = volume/pan. **Shift+drag** on a
   waveform sets a loop region (turn on **LOOP**); double-click clears it; **Ctrl+wheel** zooms; click to seek.
4. Choose stems (tick *Instrumental* for vocals-free), format, sample rate, bit depth and folder, then
   **EXPORT STEMS** (or **EXPORT CUSTOM MIX** to render your M/S/volume/pan mix).

Extras: **CLEAN** (spectral bleed reduction), **COMPARE** (run a second model and flip A/B at the same
playback position), **BATCH** (queue many files), **HISTORY** (local log of runs).

CPU speed: expect very roughly 1-3x the song length for `htdemucs` on a modern 6+ core CPU
(unmeasured here - hardware dependent). A CUDA GPU is much faster (see below).

---

## B. Run from source (developers)
```
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt        # CPU-only PyTorch (Windows PyPI wheels)
python -m noize                            # or: python run_noize.py
python scripts/install_model.py htdemucs --yes
python -m pytest -q -m "not realmodel"     # unit tests
set NOIZE_REAL_MODEL=1 && python -m pytest -q -m realmodel -s   # real inference smoke test
pyinstaller noize.spec --noconfirm         # -> dist\NOIZE\NOIZE.exe
iscc installer\noize.iss                   # -> dist\NOIZE-Setup-0.1.0.exe (needs Inno Setup 6)
```
**CUDA (optional, source installs):** after the step above run
`pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`.
NOIZE then offers CUDA under PROCESSOR (Auto picks it). The default portable build is CPU-only to stay
reliable and small; torch is pinned `<2.6` because Demucs 4.0.1 loads weights with `torch.load` defaults.
If CUDA runs out of memory NOIZE retries on CPU automatically.

## Project layout
```
noize/            app code        ui/ (theme, widgets, main window)
  audio_io.py     validation, decoding (libsndfile + FFmpeg), resampling
  separation.py   chunked Demucs inference, progress, cancel, OOM->CPU fallback
  models.py / model_manager.py   registry, consent-based verified downloads
  playback.py     single-buffer synchronized multi-stem player (Qt audio)
  mixing.py export.py clean.py waveform.py batch.py history.py settings.py workers.py
tests/            pytest suite (+ optional real-model smoke test)
noize.spec  installer/noize.iss  .github/workflows/build-windows.yml  scripts/
docs/LICENSES.md  docs/FEATURE_REPORT.md
```

## Limits (be aware)
- Separation is never perfect: bleed and "watery" artifacts remain, especially on dense mixes. CLEAN helps a little; it is classical DSP, not magic.
- Mono input is duplicated to stereo; input is resampled to the model's 44.1 kHz and exported at your chosen rate.
- Not code-signed; antivirus/SmartScreen may warn on first run.

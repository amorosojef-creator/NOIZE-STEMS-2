# PyInstaller spec - directory (onedir) build; most reliable for PyTorch + Qt.
# Build:  pyinstaller noize.spec --noconfirm
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [
    ("noize/assets/NOIZE_app_icon.ico", "assets"),
    ("noize/assets/NOIZE_emblem_transparent.png", "assets"),
    ("LICENSE", "."),
    ("docs/LICENSES.md", "docs"),
]
datas += collect_data_files("demucs")            # remote/*.yaml model bag definitions (required)
datas += collect_data_files("imageio_ffmpeg")    # bundled ffmpeg binary for M4A/video/MP3
datas += collect_data_files("dora")

hidden = (collect_submodules("demucs") + collect_submodules("dora") + collect_submodules("openunmix")
          + ["julius", "einops", "yaml", "lameenc", "soundfile", "scipy.signal", "scipy.ndimage",
             "PySide6.QtMultimedia"])

a = Analysis(["run_noize.py"], pathex=["."], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib", "IPython", "pytest"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NOIZE", console=False,
          icon="noize/assets/NOIZE_app_icon.ico", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="NOIZE", upx=False)

"""Application entry point."""
from __future__ import annotations
import sys


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        return selftest(argv)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    if sys.platform == "win32":
        try:  # taskbar groups under NOIZE and shows our icon
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NOIZE.StemExtractor")
        except Exception:
            pass
    from . import APP_NAME
    from .paths import asset
    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(asset("NOIZE_app_icon.ico"))))
    app.setStyleSheet(theme.STYLE)

    pm = QPixmap(420, 260)
    pm.fill(QColor(theme.BG))
    p = QPainter(pm)
    emb = QPixmap(str(asset("NOIZE_emblem_transparent.png"))).scaled(
        120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    p.drawPixmap((420 - emb.width()) // 2, 40, emb)
    f = QFont("Segoe UI", 18, QFont.Weight.Bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8)
    p.setFont(f)
    p.setPen(QColor(theme.TEXT))
    p.drawText(pm.rect().adjusted(0, 70, 0, 0), Qt.AlignmentFlag.AlignCenter, "NOIZE")
    p.setPen(QColor(theme.TEXT2))
    p.setFont(QFont("Segoe UI", 9))
    p.drawText(pm.rect().adjusted(0, 130, 0, 0), Qt.AlignmentFlag.AlignCenter, "AI STEM EXTRACTOR  -  loading")
    p.setPen(QColor(theme.LINE))
    p.drawRect(0, 0, 419, 259)
    p.end()
    splash = QSplashScreen(pm)
    splash.show()
    app.processEvents()

    win = MainWindow()
    win.show()
    splash.finish(win)
    return app.exec()


def selftest(argv) -> int:
    """Headless packaging check: verifies bundled runtime imports. Used by CI on the built exe."""
    import importlib
    lines, ok = [], True
    for mod in ("numpy", "scipy", "soundfile", "PySide6.QtCore", "PySide6.QtMultimedia",
                "torch", "demucs.pretrained", "demucs.apply", "julius", "einops", "yaml"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"OK   {mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            lines.append(f"FAIL {mod}: {e}")
    try:
        from demucs.pretrained import REMOTE_ROOT
        found = sorted(p.stem for p in REMOTE_ROOT.glob("*.yaml"))
        need = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}
        good = need <= set(found)
        ok &= good
        lines.append(("OK   " if good else "FAIL ") + f"demucs model bags: {', '.join(found)}")
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"FAIL demucs remote bags: {e}")
    from .audio_io import find_ffmpeg
    from .paths import asset
    for a in ("NOIZE_app_icon.ico", "NOIZE_emblem_transparent.png"):
        good = asset(a).exists()
        ok &= good
        lines.append(("OK   " if good else "FAIL ") + f"asset {a}")
    lines.append(f"INFO ffmpeg: {find_ffmpeg()}")
    lines.append("SELFTEST " + ("PASSED" if ok else "FAILED"))
    out = argv[argv.index("--selftest-out") + 1] if "--selftest-out" in argv else None
    text = "\n".join(lines)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

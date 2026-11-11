import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from noize import audio_io
from noize.audio_io import AudioError
from conftest import tone


def test_validate_missing(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        audio_io.validate_input(tmp_path / "nope.wav")


def test_validate_unsupported(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(AudioError, match="Unsupported"):
        audio_io.validate_input(p)


def test_validate_empty(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        audio_io.validate_input(p)


def test_validate_directory(tmp_path):
    with pytest.raises(AudioError):
        audio_io.validate_input(tmp_path)


def test_decode_wav(wav_file):
    a, sr = audio_io.load_audio(wav_file)
    assert sr == 44100 and a.shape == (2, 44100) and a.dtype == np.float32


def test_decode_flac(tmp_path):
    p = tmp_path / "a.flac"
    sf.write(str(p), tone().T, 44100, format="FLAC")
    a, sr = audio_io.load_audio(p)
    assert a.shape[0] == 2 and abs(a.shape[1] - 44100) <= 1


def test_corrupt_file_gives_clear_error(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not audio at all" * 100)
    with pytest.raises(AudioError):
        audio_io.load_audio(p)


def test_to_stereo():
    assert audio_io.to_stereo(np.zeros((1, 10), np.float32)).shape == (2, 10)
    assert audio_io.to_stereo(np.zeros((6, 10), np.float32)).shape == (2, 10)


def test_resample_length_and_identity():
    x = tone(secs=1.0)
    assert audio_io.resample(x, 44100, 44100) is x
    y = audio_io.resample(x, 44100, 48000)
    assert y.shape == (2, 48000)


needs_ffmpeg = pytest.mark.skipif(audio_io.find_ffmpeg() is None, reason="no ffmpeg")


@needs_ffmpeg
def test_decode_mp3_and_m4a(wav_file, tmp_path):
    ff = audio_io.find_ffmpeg()
    for ext, codec in (("mp3", "libmp3lame"), ("m4a", "aac")):
        out = tmp_path / f"s.{ext}"
        r = subprocess.run([ff, "-v", "error", "-y", "-i", str(wav_file), "-c:a", codec, str(out)])
        if r.returncode != 0:
            pytest.skip(f"ffmpeg lacks {codec}")
        a, sr = audio_io.load_audio(out)
        assert a.shape[0] == 2 and a.shape[1] > 40000


@needs_ffmpeg
def test_extract_audio_from_video(wav_file, tmp_path):
    ff = audio_io.find_ffmpeg()
    out = tmp_path / "v.mkv"
    r = subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                        "-i", str(wav_file), "-shortest", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(out)])
    if r.returncode != 0:
        pytest.skip("ffmpeg cannot build test video")
    a, sr = audio_io.load_audio(out)
    assert a.shape[0] == 2 and a.shape[1] > 40000

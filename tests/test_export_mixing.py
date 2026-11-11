import numpy as np
import pytest
import soundfile as sf

from noize import export
from noize.audio_io import AudioError, find_ffmpeg
from noize.mixing import MixState, pan_gains, prevent_clipping, render_mix
from noize.waveform import compute_peaks
from conftest import tone


def stems():
    return {"drums": tone(freq=100), "vocals": tone(freq=800)}


@pytest.mark.parametrize("depth,sub", [(16, "PCM_16"), (24, "PCM_24")])
def test_wav_export_depth(tmp_path, depth, sub):
    p = export.write_audio(tmp_path / "a", tone(), 44100, "wav", depth)
    info = sf.info(str(p))
    assert info.subtype == sub and info.frames == 44100 and info.channels == 2


def test_flac_export(tmp_path):
    p = export.write_audio(tmp_path / "a", tone(), 44100, "flac", 24)
    assert p.suffix == ".flac" and sf.info(str(p)).format == "FLAC"


def test_sample_rate_conversion(tmp_path):
    p = export.write_audio(tmp_path / "a", tone(), 44100, "wav", 16, target_sr=48000)
    i = sf.info(str(p))
    assert i.samplerate == 48000 and i.frames == 48000


def test_no_clipping(tmp_path):
    p = export.write_audio(tmp_path / "a", tone(amp=3.0), 44100, "wav", 16)
    d, _ = sf.read(str(p))
    assert np.max(np.abs(d)) <= 1.0 and np.max(np.abs(d)) > 0.9


def test_bad_options(tmp_path):
    with pytest.raises(AudioError):
        export.write_audio(tmp_path / "a", tone(), 44100, "ogg")
    with pytest.raises(AudioError):
        export.write_audio(tmp_path / "a", tone(), 44100, "wav", 32)


def test_export_stems_aligned_and_named(tmp_path):
    out = export.export_stems(stems(), 44100, tmp_path, "My: Song?.mp3")
    assert sorted(p.name for p in out) == ["My_ Song_ - drums.wav", "My_ Song_ - vocals.wav"]
    assert len({sf.info(str(p)).frames for p in out}) == 1


def test_export_selected_and_unknown(tmp_path):
    out = export.export_stems(stems(), 44100, tmp_path, "s.wav", selected=["vocals"])
    assert len(out) == 1
    with pytest.raises(AudioError):
        export.export_stems(stems(), 44100, tmp_path, "s.wav", selected=["kazoo"])


def test_no_overwrite(tmp_path):
    a = export.write_audio(tmp_path / "x", tone(), 44100)
    b = export.write_audio(tmp_path / "x", tone(), 44100)
    assert a != b and a.exists() and b.exists()


def test_output_dir_generation(tmp_path):
    assert export.output_dir_for("/m/Track One.mp3", tmp_path, True) == tmp_path / "Track One"
    assert export.output_dir_for("/m/Track One.mp3", tmp_path, False) == tmp_path


@pytest.mark.skipif(find_ffmpeg() is None, reason="no ffmpeg")
def test_mp3_export(tmp_path):
    try:
        p = export.write_audio(tmp_path / "a", tone(), 44100, "mp3")
    except AudioError:
        pytest.skip("ffmpeg has no libmp3lame")
    assert p.exists() and p.stat().st_size > 1000


def test_mix_solo_mute_pan_volume(tmp_path):
    st = stems()
    ms = MixState.for_names(st)
    full = render_mix(st, ms)
    ms.stems["drums"].mute = True
    only_v = render_mix(st, ms)
    np.testing.assert_allclose(only_v, st["vocals"] * pan_gains(0)[0], atol=1e-5)
    ms.stems["drums"].mute = False
    ms.stems["drums"].solo = True
    np.testing.assert_allclose(render_mix(st, ms), st["drums"], atol=1e-5)
    ms.stems["drums"].solo = False
    ms.stems["vocals"].volume = 0.0
    np.testing.assert_allclose(render_mix(st, ms), st["drums"], atol=1e-5)
    ms.stems["vocals"].volume, ms.stems["vocals"].pan = 1.0, 1.0
    r = render_mix({"vocals": st["vocals"]}, MixState.for_names(["vocals"]))
    assert np.abs(r).max() > 0
    assert full.shape == st["drums"].shape


def test_hard_pan_is_one_sided():
    l, r = pan_gains(-1.0)
    assert l > 1.3 and r < 1e-6


def test_custom_mix_export(tmp_path):
    st = stems()
    ms = MixState.for_names(st)
    ms.stems["drums"].mute = True
    p = export.export_mix(st, ms, 44100, tmp_path, "t.wav", "wav", 16)
    assert sf.info(str(p)).frames == 44100


def test_prevent_clipping_only_reduces():
    assert prevent_clipping(np.array([0.5], np.float32))[0] == 0.5
    assert abs(prevent_clipping(np.array([2.0], np.float32))[0]) <= 0.98 + 1e-6


def test_waveform_peaks():
    p = compute_peaks(tone(secs=1), 100)
    assert p.shape == (100, 2) and p[:, 1].max() <= 0.5 + 1e-6 and p[:, 0].min() >= -0.5 - 1e-6
    assert compute_peaks(np.zeros((2, 0), np.float32), 10).shape == (10, 2)

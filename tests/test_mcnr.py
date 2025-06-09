import numpy as np
import soundfile as sf
import os
import subprocess
import sys

import mcnr


def generate_audio(path, length=16000, channels=2, sr=16000):
    data = np.random.randn(channels, length).astype(np.float32)
    sf.write(path, data.T, sr)
    return data

def test_noise_reduction_shape(tmp_path):
    inp = tmp_path / "in.wav"
    generate_audio(inp)
    x, sr = sf.read(str(inp))
    x = x.T
    y = mcnr.do_multi_channel_noise_reduction(x)
    assert y.shape == x.shape


def test_chunk_size(tmp_path):
    inp = tmp_path / "in.wav"
    generate_audio(inp, length=32000)
    x, sr = sf.read(str(inp))
    x = x.T
    y = mcnr.do_multi_channel_noise_reduction(x, chunk_size=16000)
    assert y.shape == x.shape


def test_cli(tmp_path):
    inp = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    generate_audio(inp)
    cmd = [sys.executable, "-m", "mcnr.main", "-i", str(inp), str(out)]
    subprocess.check_call(cmd)
    assert out.exists()

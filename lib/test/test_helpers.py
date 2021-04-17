import torch
from torchaudio import load
from torchfsdd import TrimSilence

original, sr = load('lib/test/data/sample.wav')
original = original.flatten()

def test_max_threshold():
    trimmed = TrimSilence(threshold=1.)(original)
    assert len(trimmed) == 0

def test_min_threshold():
    trimmed = TrimSilence(threshold=0.)(original)
    assert torch.eq(trimmed, original).all()

def test_normal_threshold():
    trimmed = TrimSilence(threshold=0.1)(original)
    assert len(trimmed) != len(original)
    assert torch.eq(trimmed, original[9:2301]).all()
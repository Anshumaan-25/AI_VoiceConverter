"""Exercise upstream functions without loading pretrained weights."""
import ast
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]

def function_from_file(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]

class UpstreamTests(unittest.TestCase):
    def test_pcm16_pcm24_float_training_have_equal_amplitude(self):
        read = function_from_file(ROOT / "so-vits-svc/utils.py", "load_wav_to_torch", {"np": np, "torch": torch})
        with tempfile.TemporaryDirectory() as directory:
            for subtype in ("PCM_16", "PCM_24", "FLOAT"):
                path = Path(directory) / (subtype + ".wav")
                sf.write(path, np.full(1000, .25), 48000, subtype=subtype)
                audio, sr = read(path)
                self.assertEqual(sr, 48000)
                self.assertAlmostEqual(float(audio.max()), .25, places=4)

    def test_resampling_cache_tracks_input_rate(self):
        import time
        import torchaudio
        infer = function_from_file(ROOT / "so-vits-svc/inference/infer_tool.py", "infer",
                                   {"np": np, "torch": torch, "torchaudio": torchaudio, "time": time})
        seen = []
        def features(wav, *args, **kwargs):
            seen.append(len(wav))
            return torch.ones(1, 2, 100), torch.ones(1, 100), torch.ones(1, 100)
        net = SimpleNamespace(infer=lambda *args, **kwargs: (torch.zeros(1,1,100), None))
        model = SimpleNamespace(target_sample=44100, dev=torch.device("cpu"), spk2id={"speaker":0},
                                dtype=torch.float32, only_diffusion=False, shallow_diffusion=False,
                                vol_embedding=False, net_g_ms=net, nsf_hifigan_enhance=False,
                                get_unit_f0=features)
        for sr in (48000, 44100, 48000):
            with patch.object(torchaudio, "load", return_value=(torch.zeros(2,sr),sr)):
                output, count, frames = infer(model, "speaker", 0, None)
        self.assertEqual(seen, [44100, 44100, 44100])

    def test_padding_enforces_requested_length(self):
        pad = function_from_file(ROOT / "so-vits-svc/inference/infer_tool.py", "pad_array", {"np": np})
        self.assertEqual(len(pad(np.ones(8), 5)), 5)
        self.assertEqual(len(pad(np.ones(3), 5)), 5)

    def test_chunk_stitch_zero_padding_and_cross_rate_overlap(self):
        source = ROOT / "so-vits-svc/inference/infer_tool.py"
        namespace = {"np": np, "Path": Path, "io": io, "soundfile": sf}
        namespace["pad_array"] = function_from_file(source, "pad_array", namespace)
        namespace["split_list_by_n"] = function_from_file(source, "split_list_by_n", namespace)
        namespace["slicer"] = SimpleNamespace(cut=lambda *a, **k: {},
            chunks2audio=lambda *a: ([(False, np.ones(6*48000)*.1)], 48000))
        slice_inference = function_from_file(source, "slice_inference", namespace)
        def infer(speaker, transpose, buffer, **kwargs):
            audio, sr = sf.read(buffer)
            count = round(len(audio) * 44100 / sr)
            return torch.ones(count)*.1, count, count//512
        model = SimpleNamespace(target_sample=44100, hop_size=512, infer=infer, clear_empty=lambda: None)
        for overlap in (0, .1):
            output = slice_inference(model, "dummy.wav", "speaker", 0, -40, 0, False, .4,
                                     pad_seconds=0, clip_seconds=5, lg_num=overlap)
            self.assertEqual(len(output), 6*44100)
            self.assertTrue(np.isfinite(output).all())

if __name__ == "__main__":
    unittest.main()

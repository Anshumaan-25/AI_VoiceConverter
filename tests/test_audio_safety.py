import ast
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import audio_io
from dataset_tools import prepare_clips, validate_training_data

ROOT = Path(__file__).resolve().parents[1]

class AudioSafetyTests(unittest.TestCase):
    def test_trim_keeps_padding_and_rejects_silence(self):
        from speech.validation import finish_chunk
        from speech.contracts import SpeechError
        signal = np.r_[np.full(24000, .2), np.zeros(72000)].astype(np.float32)
        trimmed = finish_chunk(signal, 24000)
        self.assertTrue(24000 <= len(trimmed) <= 27000)
        with self.assertRaises(SpeechError):
            finish_chunk(np.zeros(24000), 24000)

    def test_path_escape_and_reserved_names(self):
        for name in ("..", "../other", "C:\\other", "CON", "a/b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audio_io.safe_child(ROOT / "dataset", name)

    def test_pcm24_storage_and_resampling(self):
        with tempfile.TemporaryDirectory(prefix="voice test ") as directory:
            path = Path(directory) / "音声 test.wav"
            t = np.arange(24000) / 24000
            signal = .4 * np.sin(2 * np.pi * 1000 * t)
            audio_io.write_wav(path, signal, 24000)
            info = sf.info(str(path))
            self.assertEqual((info.samplerate, info.subtype, info.frames), (48000, "PCM_24", 48000))
            decoded, sr = audio_io.read_audio(path, 44100)
            self.assertEqual(len(decoded), 44100)
            self.assertLess(abs(np.max(decoded) - .4), .003)

    def test_invalid_audio(self):
        for signal in (np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((10, 3))):
            with self.assertRaises(ValueError):
                audio_io.validate_audio(signal, 48000)

    def test_dataset_bounded_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            audio_io.write_wav(source, .1*np.sin(np.arange(48000*20)*.05), 48000)
            records = prepare_clips([source], root / "prepared")
            self.assertGreaterEqual(len(records), 2)
            for record in records:
                info = sf.info(str(root / "prepared" / record["file"]))
                self.assertTrue(2 <= info.duration <= 10)
                self.assertEqual(info.subtype, "PCM_24")

    def test_quiet_recording_cannot_replace_previous_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_io.write_wav(root / "first.wav", .1*np.sin(np.arange(48000*5)*.05), 48000)
            quiet = np.zeros(48000*6 + 24000, dtype=np.float32)
            quiet[-1] = .001
            audio_io.write_wav(root / "quiet.wav", quiet, 48000)
            records = prepare_clips([root/"first.wav", root/"quiet.wav"], root/"prepared")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source"], "first.wav")
            kept, sr = audio_io.read_audio(root/"prepared"/records[0]["file"])
            self.assertEqual(len(kept), 5*sr)
            self.assertGreater(float(np.sqrt(np.mean(kept**2))), .06)

    def test_real_ffmpeg_export_bitrate(self):
        from export_mix import export_mix
        import shutil
        if not shutil.which("ffprobe"):
            self.skipTest("ffprobe unavailable")
        with tempfile.TemporaryDirectory(prefix="mix test ") as directory:
            root = Path(directory)
            signal = .5 * np.sin(np.arange(48000*2)*.7)
            for name in ("v.wav", "i.wav"):
                audio_io.write_wav(root/name, signal, 48000)
            export_mix(root/"v.wav", root/"i.wav", root/"out.mp3")
            data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(root/"out.mp3")]))
            stream = data["streams"][0]
            self.assertEqual(int(stream["bit_rate"]), 320000)
            self.assertEqual(int(stream["sample_rate"]), 48000)

    def test_training_rejects_existing_unsliced_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "filelists").mkdir()
            audio_io.write_wav(root / "long.wav", np.ones(48000 * 11) * .1, 48000)
            for split in ("train", "val"):
                (root / "filelists" / (split + ".txt")).write_text("long.wav")
            with self.assertRaisesRegex(ValueError, "2–10"):
                validate_training_data(root)

    def test_no_python_syntax_regressions(self):
        paths = list(ROOT.glob("*.py")) + list((ROOT / "speech").rglob("*.py")) + list((ROOT / "so-vits-svc").rglob("*.py"))
        for path in paths:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8-sig"))

if __name__ == "__main__":
    unittest.main()

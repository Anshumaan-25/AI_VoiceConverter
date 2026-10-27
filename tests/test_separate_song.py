import unittest
import tempfile
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from separate_song import select_interval,separate


class StemSelectionTests(unittest.TestCase):
    def test_selects_active_window_after_intro(self):
        vocal=np.zeros((6400,2),dtype=np.float32)
        vocal[1000:4800]=.1
        chosen=select_interval(vocal,100)
        self.assertGreaterEqual(chosen['start_seconds'],8)
        self.assertLess(chosen['start_seconds'],10)
        self.assertEqual(chosen['seconds'],30)
        self.assertGreaterEqual(chosen['active_second_fraction'],.95)

    def test_rejects_silence_short_and_nonfinite_audio(self):
        for audio in (np.zeros(4000),np.ones(2900)*.1,np.full(4000,np.nan)):
            with self.subTest(),self.assertRaises(ValueError): select_interval(audio,100)

    def test_time_selection_clamps_leading_context_to_zero(self):
        chosen=select_interval(np.ones(4000)*.1,100)
        self.assertEqual(chosen['start_seconds'],0)
        with self.assertRaises(ValueError): select_interval(np.ones(4000),0)

    def test_short_input_is_rejected_before_worker_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'too-short.wav'; output=root/'output'
            sf.write(source,np.zeros((4800,2)),48000,subtype='PCM_24')
            with self.assertRaisesRegex(ValueError,'30 seconds'): separate(source,Path(sys.executable),output)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()

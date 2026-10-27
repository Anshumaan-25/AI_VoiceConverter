from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from audio_io import write_wav
from download_rvc_assets import sha256
from prepare_rvc_dataset import prepare,Settings


class DatasetTests(unittest.TestCase):
    def recording(self,path,seconds=180):
        rng = np.random.default_rng(123)
        write_wav(path,rng.normal(0,.03,round(48000*seconds)).astype(np.float32),48000)

    def test_provisional_split_idempotence_and_source_preservation(self):
        with tempfile.TemporaryDirectory(prefix='RVC dataset 音声 ') as temporary:
            root = Path(temporary); source = root/'source.wav'; self.recording(source)
            before = sha256(source)
            with patch('webrtcvad.Vad') as vad:
                vad.return_value.is_speech.return_value = True
                target,report = prepare([source,source],root/'data',Settings(min_usable_seconds=10))
            self.assertEqual(sha256(source),before)
            self.assertEqual(len(report['sources']),1)
            train = [c for c in report['clips'] if c['split']=='train']
            val = [c for c in report['clips'] if c['split']=='val']
            # A long enough single source is required for a guarded holdout.
            self.assertTrue(train and val)
            self.assertLessEqual(max(c['end_sample'] for c in train)+20*48000,min(c['start_sample'] for c in val))
            for clip in report['clips']:
                info = sf.info(target/clip['file'])
                self.assertTrue(2 <= info.duration <= 10)
                self.assertEqual((info.samplerate,info.subtype,info.channels),(48000,'PCM_24',1))
            same,again = prepare([source],root/'data',Settings(min_usable_seconds=10))
            self.assertEqual(same,target)
            self.assertEqual(report['summary'],again['summary'])
            path = target/report['clips'][0]['file']; path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare([source],root/'data',Settings(min_usable_seconds=10))

    def test_two_sources_use_recording_groups_and_collapse_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.recording(root/'a.wav',12)
            write_wav(root/'b.wav',np.random.default_rng(1).normal(0,.02,48000*12),48000)
            with patch('webrtcvad.Vad') as vad:
                vad.return_value.is_speech.return_value=True
                _,report=prepare([root/'a.wav',root/'b.wav'],root/'data',Settings(min_usable_seconds=1))
            groups={s:{c['source'] for c in report['clips'] if c['split']==s} for s in ('train','val')}
            self.assertFalse(groups['train'] & groups['val'])
            self.assertEqual(report['split_quality'],'recording_groups')

    def test_silence_short_input_and_nonfinite_fail_without_partial_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name,data in [('silence',np.zeros(48000*40)),('short',np.ones(48000)*.1),('nan',np.full(48000*40,np.nan))]:
                source=root/(name+'.wav')
                sf.write(source,data,48000,subtype='FLOAT')
                before=sha256(source)
                with self.subTest(name=name),self.assertRaises(ValueError):
                    prepare([source],root/'data',Settings(min_usable_seconds=0))
                self.assertEqual(sha256(source),before)
                self.assertEqual(list((root/'data').iterdir()),[])

    def test_interrupted_preparation_leaves_no_completed_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.recording(root/'source.wav',180)
            with patch('webrtcvad.Vad') as vad,patch('prepare_rvc_dataset.sf.write',side_effect=OSError('disk failure')):
                vad.return_value.is_speech.return_value=True
                with self.assertRaises(OSError):
                    prepare([root/'source.wav'],root/'data',Settings(min_usable_seconds=0))
            self.assertEqual(list((root/'data').iterdir()),[])


if __name__=='__main__':
    unittest.main()

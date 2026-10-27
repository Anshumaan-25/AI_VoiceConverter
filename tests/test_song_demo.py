import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from song_demo import extract_stem,demonstrate


class SongDemoTests(unittest.TestCase):
    def test_interval_duration_scaling_and_alias_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source.wav'
            samples=np.sin(np.arange(31*44100)*.03).astype(np.float32)*.2
            sf.write(source,samples,44100,subtype='PCM_24')
            result=extract_stem(source,root/'clip.wav',1,mono=True)
            info=sf.info(str(root/'clip.wav'))
            self.assertEqual((info.frames,info.samplerate,info.subtype),(1440000,48000,'PCM_24'))
            self.assertEqual(result['source_start_sample'],44100)
            with self.assertRaises(ValueError): extract_stem(source,source,0)
            with self.assertRaises(ValueError): extract_stem(source,root/'bad.wav',2)

    def test_fixture_conversion_mixes_stereo_pcm24_and_true_320k_mp3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); sr=48000; t=np.arange(31*sr)/sr
            vocal=root/'vocal.wav'; instrumental=root/'instrumental.wav'; model=root/'fixture.pth'
            sf.write(vocal,.3*np.sin(2*np.pi*180*t),sr,subtype='PCM_24')
            stereo=np.column_stack((.3*np.sin(2*np.pi*440*t),.3*np.sin(2*np.pi*660*t)))
            sf.write(instrumental,stereo,sr,subtype='PCM_24'); model.write_bytes(b'fixture')
            originals=(vocal.read_bytes(),instrumental.read_bytes())
            def convert(speech,output): shutil.copyfile(speech.path,output); return output
            # This fixture verifies transport/mixing only; it does not synthesize a voice.
            with patch('rvc_adapter.RvcConverter.convert',side_effect=convert):
                destination=demonstrate(vocal,instrumental,model,root/'demo')
            report=json.loads((destination/'report.json').read_text())
            self.assertEqual(report['status'],'complete')
            audio,sr=sf.read(str(destination/'your-voice-mix.wav'))
            self.assertEqual(audio.shape,(1440000,2)); self.assertEqual(sr,48000)
            self.assertGreater(float(np.std(audio[:,0]-audio[:,1])),.1)
            self.assertLessEqual(float(np.max(np.abs(audio))),.891)
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,sample_rate,bit_rate,channels',
                                  '-of','json',str(destination/'your-voice-mix.mp3')],capture_output=True,text=True,check=True,timeout=15)
            stream=json.loads(probe.stdout)['streams'][0]
            self.assertEqual((stream['codec_name'],stream['sample_rate'],stream['bit_rate'],stream['channels']),('mp3','48000','320000',2))
            self.assertEqual((vocal.read_bytes(),instrumental.read_bytes()),originals)


if __name__ == '__main__': unittest.main()

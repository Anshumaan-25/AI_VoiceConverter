from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
import soundfile as sf
from audio_io import write_wav
from rvc_adapter import RvcConverter,RvcOptions
from rvc_audio import segment_ranges,stitch_segments,fit_segment,validate_conversion
from rvc_runtime import checkpoint_metadata,load_index
from speech.contracts import SpeechResult,SpeechError,WorkerTimeout

ROOT = Path(__file__).resolve().parents[1]


class RvcTests(unittest.TestCase):
    def test_controller_has_no_model_imports(self):
        code = "import sys,rvc_adapter,rvc_runtime; assert 'torch' not in sys.modules; assert 'fairseq' not in sys.modules"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,timeout=30)

    def test_invalid_options_and_requested_missing_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)/'model.pth'; model.write_bytes(b'fixture')
            base = RvcOptions(str(model))
            base.validate()
            for options in (replace(base,index_rate=float('nan')),replace(base,index_rate=1.1),
                            replace(base,transpose=1.5),replace(base,speaker_id=-1),
                            replace(base,device='auto'),replace(base,load_timeout=0)):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    options.validate()
            with self.assertRaises(FileNotFoundError):
                replace(base,index_rate=.5).validate()
            replace(base,index='missing.index',index_rate=0).validate()

    def test_metadata_rejects_other_model_types_and_wrong_feature_width(self):
        fixture = dict(config=[0]*17+[40000],weight={'emb_g.weight':np.zeros((1,4)),
                       'enc_p.emb_phone.weight':np.zeros((8,768))},version='v2',f0=1)
        self.assertEqual(checkpoint_metadata(fixture)['dimension'],768)
        for checkpoint in ({},dict(fixture,version='v3'),dict(fixture,version='v1'),
                           dict(fixture,config=[0]*17+[44100])):
            with self.assertRaises(ValueError):
                checkpoint_metadata(checkpoint)

    def test_index_readability_dimension_and_empty_index(self):
        import faiss
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'selected.index'
            index = faiss.IndexFlatL2(256)
            faiss.write_index(index,str(path))
            with self.assertRaises(ValueError):
                load_index(path,256)
            index.add(np.ones((2,256),dtype=np.float32))
            faiss.write_index(index,str(path))
            with self.assertRaises(ValueError):
                load_index(path,768)
            self.assertEqual(load_index(path,256)[1].shape,(2,256))
            path.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):
                load_index(path,256)

    def test_segments_reconstruct_exact_lengths_including_short_tail(self):
        for sr in (32000,40000,48000):
            for seconds in (.15,5,5.001,9.9,10.1,20.123):
                n = round(seconds*sr)
                reference = np.sin(np.arange(n)*.01).astype(np.float32)*.2
                parts = [(a,reference[a:b]) for a,b,_,_ in segment_ranges(n,sr)]
                assembled = stitch_segments(parts,sr,n)
                self.assertEqual(len(assembled),n)
                np.testing.assert_allclose(assembled,reference,atol=1e-7)
            with self.assertRaises(ValueError):
                fit_segment(np.zeros(sr),2*sr,sr)
            with self.assertRaises(ValueError):
                validate_conversion(np.zeros(sr),sr,1)

    @patch('rvc_runtime.python_path',return_value=Path(sys.executable))
    def test_adapter_alias_and_failure_preserve_destination(self,_interpreter):
        with tempfile.TemporaryDirectory(prefix='RVC path 音声 ') as temporary:
            root = Path(temporary); source = root/'source.wav'; model = root/'model.pth'
            write_wav(source,np.ones(48000)*.1,48000); model.write_bytes(b'fixture')
            speech = SpeechResult(source,'test',[])
            converter = RvcConverter(RvcOptions(str(model)))
            with self.assertRaisesRegex(ValueError,'alias'):
                converter.convert(speech,source)
            link = root/'linked.wav'; os.link(source,link)
            with self.assertRaisesRegex(ValueError,'alias'):
                converter.convert(speech,link)
            output = root/'out.wav'; output.write_bytes(b'previous')
            with patch('rvc_runtime.verify_source'),patch('rvc_runtime.verify_assets'), \
                 patch('rvc_adapter.gpu_job',side_effect=nullcontext), \
                 patch('rvc_adapter.run_worker',side_effect=WorkerTimeout('fixture timeout')):
                with self.assertRaises(WorkerTimeout):
                    converter.convert(speech,output)
            self.assertEqual(output.read_bytes(),b'previous')

    def test_application_routes_rvc_without_so_vits_settings(self):
        from app_service import VoiceApplication
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source.wav'; model=root/'model.pth'
            write_wav(source,np.ones(48000)*.1,48000); model.write_bytes(b'fixture')
            def fake_conversion(adapter,speech,output):
                self.assertEqual(adapter.options.index_rate,0)
                self.assertEqual(adapter.options.speaker_id,0)
                self.assertEqual(sf.info(speech.path).subtype,'PCM_24')
                write_wav(output,np.ones(48000)*.1,48000)
                return output
            with patch.object(RvcConverter,'convert',autospec=True,side_effect=fake_conversion) as convert:
                output=VoiceApplication(root/'app').convert('session',source,model,'','',backend='rvc')
            self.assertTrue(output.is_file()); convert.assert_called_once()

    @unittest.skipUnless((ROOT/'vendor'/'rvc'/'infer'/'modules'/'vc'/'pipeline.py').is_file(),
                         'Optional restored RVC source fixture')
    def test_patched_retrieval_accepts_exact_matches_and_preserves_float_audio(self):
        import torch
        import faiss
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'vendor'/'rvc'))
        from infer.modules.vc.pipeline import Pipeline
        pipeline=Pipeline(40000,SimpleNamespace(x_pad=1,x_query=5,x_center=30,x_max=32,is_half=False,device='cpu'))
        class HubertFixture:
            def extract_features(self,**kwargs):
                return (torch.zeros(1,max(1,kwargs['source'].shape[-1]//320),768),)
        class NetworkFixture:
            def infer(self,features,length,speaker):
                if not torch.isfinite(features).all():
                    raise AssertionError('Nonfinite exact-match retrieval weights')
                return (torch.full((1,1,int(length[0])*400),.1234567),)
        index=faiss.IndexFlatL2(768); index.add(np.zeros((1,768),dtype=np.float32))
        raw=pipeline.vc(HubertFixture(),NetworkFixture(),torch.tensor([0]),np.ones(16000,dtype=np.float32),
                        None,None,[0,0,0],index,np.zeros((1,768),dtype=np.float32),.5,'v2',.5)
        self.assertEqual(raw.dtype,np.float32)
        self.assertAlmostEqual(float(raw[0]),.1234567,places=7)
        final=pipeline.pipeline(HubertFixture(),NetworkFixture(),0,np.ones(16000,dtype=np.float32),
                                'fixture',[0,0,0],0,'rmvpe','',0,0,3,40000,0,1,'v2',.5)
        self.assertEqual(final.dtype,np.float32)
        self.assertAlmostEqual(float(final[0]),.1234567,places=7)

    @unittest.skipUnless(os.environ.get('VOICE_TEST_GPU') == '1' and
                         (ROOT/'venv-rvc'/'Scripts'/'python.exe').is_file(),
                         'Explicit isolated RVC CUDA lifecycle test with installed runtime')
    def test_isolated_cuda_allocation_reclaimed_on_timeout(self):
        from speech.supervisor import run_worker
        from speech.contracts import SpeechOptions
        from benchmark_speech import gpu_usage
        import time
        before=gpu_usage()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            with self.assertRaises(WorkerTimeout):
                run_worker([str(ROOT/'venv-rvc'/'Scripts'/'python.exe'),'-B',
                    str(ROOT/'tests'/'worker_fixture.py'),str(directory),'gpu'],
                    directory,SpeechOptions(chunk_timeout=.5),time.monotonic()+60,backend='bark')
            self.assertGreater(json.loads((directory/'allocated.json').read_text())['bytes'],100*1024**2)
            self.assertTrue(json.loads((directory/'process.json').read_text())['exited'])
        self.assertLessEqual(gpu_usage()[0],before[0]+64)

    @unittest.skipUnless((ROOT/'venv-rvc'/'Scripts'/'python.exe').is_file(),'Optional installed isolated runtime')
    def test_real_isolated_worker_rejects_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'source.wav'; model = root/'bad.pth'
            write_wav(source,np.ones(48000)*.1,48000); model.write_bytes(b'not a checkpoint')
            converter = RvcConverter(RvcOptions(str(model),device='cpu',load_timeout=60),diagnostics=root/'diag')
            with self.assertRaises(SpeechError):
                converter.convert(SpeechResult(source,'test',[]),root/'out.wav')
            self.assertFalse((root/'out.wav').exists())
            process = json.loads(next((root/'diag').rglob('process.json')).read_text())
            self.assertTrue(process['exited'])


if __name__ == '__main__':
    unittest.main()

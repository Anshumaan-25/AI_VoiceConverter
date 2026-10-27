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

from rvc_training import (aligned_frames,child,checkpoint_set,publish_checkpoint,load_config,
                          reserve_budget,reconcile_budget,digest_json,verify_cache,extraction_identity)

ROOT=Path(__file__).resolve().parents[1]


class TrainingSafetyTests(unittest.TestCase):
    def test_controller_imports_do_not_load_models(self):
        subprocess.run([sys.executable,'-B','-c',"import sys,train_rvc,prepare_rvc_training; assert 'torch' not in sys.modules; assert 'fairseq' not in sys.modules"],check=True,timeout=30)

    def test_alignment_bounds(self):
        self.assertEqual(aligned_frames(48000,49,101),98)
        for sizes in ((48000,40,100),(48000,50,120),(0,0,0)):
            with self.assertRaises(ValueError): aligned_frames(*sizes)

    def test_path_escape_and_budget_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError): child(root,'../outside')
            config=load_config()
            with self.assertRaises(ValueError): reserve_budget(root,config,501,1)
            with self.assertRaises(ValueError): reserve_budget(root,config,1,1801)
            ticket=reserve_budget(root,config,10,300)
            with self.assertRaises(RuntimeError): reserve_budget(root,config,1,1)
            reconcile_budget(root,ticket,3,12,{'stage':'fixture'})
            data=json.loads((root/'budget.json').read_text())
            self.assertEqual(data['attempted_steps'],3)
            self.assertEqual(data['seconds'],12)
            reserve_budget(root,config,497,1788)

    def test_checkpoint_commit_corruption_and_interrupted_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); identity={'fixture':True}
            result=publish_checkpoint(root,1,identity,lambda p:p.write_bytes(b'matched G and D fixture'))
            self.assertEqual(result['step'],1)
            latest=checkpoint_set(root)
            def interrupted(p):
                p.write_bytes(b'partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError): publish_checkpoint(root,2,identity,interrupted)
            self.assertEqual(checkpoint_set(root),latest)
            latest[0].write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'): checkpoint_set(root)

    def test_checkpoint_retention_and_partial_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for step in (1,2,3): publish_checkpoint(root,step,{'fixture':1},lambda p:p.write_bytes(b'weights'))
            self.assertEqual(len(list(root.glob('checkpoint-*'))),2)
            pointer=json.loads((root/'latest.json').read_text())
            (root/pointer['directory']/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError): checkpoint_set(root)

    def test_config_rejects_unbounded_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'config.json'; config=load_config()
            for change in ({'max_steps':501},{'max_seconds':1801},{'batch_size':6},{'segment_size':999},{'max_steps':True}):
                path.write_text(json.dumps(dict(config,**change)))
                with self.assertRaises(ValueError): load_config(path)

    def test_float_pcm24_loading_preserves_amplitude(self):
        from audio_io import read_audio
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'reference.wav'
            audio=np.sin(np.arange(48000)*.04).astype(np.float32)*.125
            sf.write(str(path),audio,48000,subtype='PCM_24')
            loaded,sr=read_audio(path)
            self.assertEqual(sr,48000)
            np.testing.assert_allclose(loaded,audio,atol=1.3e-7)
            self.assertLessEqual(float(np.max(np.abs(loaded))),.126)

    def test_stale_cache_and_split_provenance_rejected(self):
        from download_rvc_assets import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); dataset=root/'dataset'; dataset.mkdir()
            (dataset/'manifest.json').write_text('{}')
            original={'clips':[{'file':'train/a.wav','sha256':'a','split':'train'}]}
            tensor=root/'features.npz'; tensor.write_bytes(b'fixture')
            clip=dict(source_file='train/a.wav',source_sha256='a',split='train',file=tensor.name,sha256=sha256(tensor))
            identity=extraction_identity(dataset,original)
            cache=dict(dataset=str(dataset),identity=identity,clips=[clip])
            with patch('rvc_training.dataset_manifest',return_value=(original,{'train':1,'val':1})):
                (root/'features.json').write_text(json.dumps(cache))
                verify_cache(root)
                cache['clips'][0]['split']='val'
                (root/'features.json').write_text(json.dumps(cache))
                with self.assertRaisesRegex(ValueError,'provenance'): verify_cache(root)
                cache['clips'][0]['split']='train'
                cache['identity']=dict(identity,layer=9)
                (root/'features.json').write_text(json.dumps(cache))
                with self.assertRaisesRegex(ValueError,'Stale'): verify_cache(root)

    def test_training_controller_records_worker_failure_and_preserves_checkpoint(self):
        from train_rvc import execute
        from speech.processes import ProcessTree
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'features.json').write_text('{}')
            class FailedWorker:
                pid=0
                returncode=7
                def poll(self): return 7
            with patch('rvc_runtime.verify_source'),patch('rvc_runtime.verify_assets'), \
                 patch('train_rvc.verify_training_assets'),patch('train_rvc.verify_cache'), \
                 patch('gpu_lock.gpu_job',side_effect=nullcontext), \
                 patch('train_rvc.gpu_memory',return_value=0), \
                 patch('train_rvc.subprocess.Popen',return_value=FailedWorker()), \
                 patch('train_rvc.ProcessTree') as tree:
                with self.assertRaisesRegex(RuntimeError,'exited 7'): execute(root)
                tree.return_value.close.assert_called_once()
            budget=json.loads((root/'budget.json').read_text())
            self.assertIsNone(budget['active'])
            self.assertEqual(budget['attempted_steps'],0)
            self.assertIn('exited 7',budget['attempts'][0]['error'])

    def test_qualification_refuses_failed_or_active_training(self):
        from qualify_rvc_model import check_training_gate
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for budget in ({'active':{'id':'fixture'},'attempts':[]},
                           {'active':None,'attempts':[{'stage':'pilot','error':'Nonfinite discriminator loss'}]},
                           {'active':None,'attempts':[{'stage':'smoke','error':None}]}):
                (root/'budget.json').write_text(json.dumps(budget))
                with self.assertRaises(RuntimeError): check_training_gate(root)
            (root/'budget.json').write_text(json.dumps({'active':None,'attempts':[{'stage':'pilot','error':None}]}))
            check_training_gate(root)

    def test_index_uses_training_features_only(self):
        from qualify_rvc_model import build_index
        import faiss
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); clips=[]
            for split,value in (('train',1.),('val',100.)):
                path=root/(split+'.npz')
                np.savez(path,features=np.full((8,768),value,dtype=np.float32))
                clips.append(dict(split=split,file=path.name,source_file=split+'.wav',sha256='fixture'))
            (root/'features.json').write_text('{}')
            with patch('qualify_rvc_model.verify_cache',return_value={'clips':clips}): build_index(root)
            index=faiss.read_index(str(root/'pilot-v2-48k.index'))
            self.assertEqual(index.ntotal,4)
            np.testing.assert_array_equal(index.reconstruct_n(0,4),np.ones((4,768),dtype=np.float32))

    def test_only_one_successful_oom_retry_resolves_failure_gate(self):
        from train_rvc import unresolved_failure
        oom=dict(stage='smoke',error='CUDA out of memory')
        retry=dict(stage='retry',error=None)
        self.assertTrue(unresolved_failure({'attempts':[oom]}))
        self.assertFalse(unresolved_failure({'attempts':[oom,retry]}))
        self.assertTrue(unresolved_failure({'attempts':[dict(stage='pilot',error='Nonfinite discriminator loss'),retry]}))
        self.assertTrue(unresolved_failure({'attempts':[oom,dict(stage='retry',error='out of memory')]}))

    @unittest.skipUnless((ROOT/'vendor/rvc/.git').exists(),'pinned source not restored')
    def test_patch_restores_clean_disposable_source(self):
        from rvc_runtime import manifest
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'source'
            subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT/'vendor/rvc'),str(target)],check=True,timeout=30)
            subprocess.run(['git','-C',str(target),'checkout','--quiet','--detach',manifest()['commit']],check=True,timeout=30)
            patch=(ROOT/manifest()['patch']).read_bytes()
            subprocess.run(['git','-C',str(target),'apply','-'],input=patch,check=True,timeout=30)
            self.assertEqual(subprocess.check_output(['git','-C',str(target),'diff','--no-ext-diff','--binary','HEAD'],timeout=30),patch)


if __name__ == '__main__': unittest.main()

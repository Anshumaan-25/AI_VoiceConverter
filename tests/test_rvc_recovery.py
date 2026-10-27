import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from rvc_numerics import NumericalProbe,NumericalFailure
from speech.diagnostics import Trace
from rvc_recovery import parent_checkpoint


class RecoveryTests(unittest.TestCase):
    def test_first_nonfinite_location_and_finite_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); probe=NumericalProbe(root,Trace(root))
            probe.inspect('fixture',torch.ones(3),record=True)
            with self.assertRaisesRegex(NumericalFailure,'g.dec'):
                probe.inspect('g.dec',(torch.tensor([1.,float('nan')]),))
            stats=json.loads((root/'first-nonfinite.json').read_text())
            self.assertEqual(stats['module'],'g.dec'); self.assertEqual(stats['nan'],1)

    def test_bf16_range_and_guard_rejects_unscaled_bad_gradients(self):
        large=torch.tensor([100000.],dtype=torch.float32)
        self.assertFalse(bool(torch.isfinite(large.half()).all()))
        self.assertTrue(bool(torch.isfinite(large.bfloat16()).all()))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); probe=NumericalProbe(root,Trace(root))
            parameter=torch.nn.Parameter(torch.ones(2)); parameter.grad=torch.tensor([1.,float('inf')])
            self.assertFalse(probe.gradients('g',[parameter]))

    def test_parent_migration_preserves_immutable_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            identity=dict(features_sha256='features',training_assets={'g':'pinned'},source_commit='source',
                          config=dict(schema=1,version='v2',sample_rate=48000,f0=1,batch_size=1,segment_size=17280,seed=1234))
            marker=dict(parent_workspace=str(root/'parent'),parent_sha256='checkpoint',parent_identity=identity,
                        authorization='User requested numerical recovery and Suno stem demonstration')
            (root/'recovery.json').write_text(json.dumps(marker))
            parent=(root/'state.pth',dict(sha256='checkpoint',identity=identity))
            with patch('rvc_recovery.checkpoint_set',return_value=parent):
                self.assertEqual(parent_checkpoint(root,identity),parent)
                with self.assertRaisesRegex(ValueError,'immutable'):
                    parent_checkpoint(root,dict(identity,features_sha256='wrong'))
            with patch('rvc_recovery.checkpoint_set',return_value=(root/'state',dict(sha256='wrong',identity=identity))):
                with self.assertRaisesRegex(ValueError,'changed'): parent_checkpoint(root,identity)

    def test_bark_is_parked_and_automatic_speech_is_cpu(self):
        from speech.contracts import SpeechOptions
        from speech.supervisor import automatic_backends
        with self.assertRaisesRegex(ValueError,'parked'): SpeechOptions(backend='bark').validate()
        self.assertEqual(automatic_backends(),['sapi'])


if __name__ == '__main__': unittest.main()

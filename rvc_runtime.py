"""Runtime verification and worker-only model loading for a pinned RVC revision."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from download_rvc_assets import sha256

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'vendor'/'rvc'


def manifest():
    return json.loads((ROOT/'rvc_manifest.json').read_text(encoding='utf-8'))


def python_path():
    return ROOT/'venv-rvc'/'Scripts'/'python.exe'


def verify_source():
    config = manifest()
    if not (SOURCE/'.git').exists():
        raise FileNotFoundError('RVC source missing; run setup_rvc.ps1')
    head = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True,timeout=30).strip()
    diff = subprocess.check_output(['git','-C',str(SOURCE),'diff','--no-ext-diff','--binary','HEAD'],
                                   stderr=subprocess.DEVNULL,timeout=30)
    if head != config['commit'] or diff != (ROOT/config['patch']).read_bytes():
        raise RuntimeError('RVC source differs from the pinned revision/patch')


def verify_assets():
    for asset in manifest()['assets'].values():
        path = SOURCE/asset['file']
        if not path.is_file() or path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise RuntimeError(f'RVC asset missing or hash mismatch: {path.name}; run download_rvc_assets.py')


def checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('config'),(list,tuple)) or len(checkpoint['config']) != 18:
        raise ValueError('Expected an exported RVC inference checkpoint with config/weight')
    version, f0 = checkpoint.get('version','v1'),checkpoint.get('f0',1)
    if version not in ('v1','v2') or type(f0) is not int or f0 not in (0,1):
        raise ValueError('Unsupported RVC checkpoint version/f0 configuration')
    config = list(checkpoint['config'])
    sr = {'32k':32000,'40k':40000,'48k':48000}.get(config[-1],config[-1])
    if sr not in (32000,40000,48000):
        raise ValueError('Unsupported RVC native sample rate')
    weight = checkpoint.get('weight')
    if not isinstance(weight,dict) or 'emb_g.weight' not in weight or 'enc_p.emb_phone.weight' not in weight:
        raise ValueError('Missing speaker or content-encoder weights')
    speakers = weight['emb_g.weight'].shape[0]
    dimension = 256 if version == 'v1' else 768
    if weight['enc_p.emb_phone.weight'].shape[1] != dimension or not 1 <= speakers <= 1024:
        raise ValueError('Checkpoint feature width or speaker table does not match version')
    config[-3],config[-1] = speakers,sr
    return dict(version=version,f0=f0,sample_rate=sr,speakers=speakers,dimension=dimension,config=config)


def load_index(path,dimension):
    import faiss
    import numpy as np
    index = faiss.read_index(str(path))
    if index.d != dimension or not index.is_trained or not 1 <= index.ntotal <= 200000:
        raise ValueError('Retrieval index width/training state/size is incompatible')
    if index.metric_type != faiss.METRIC_L2:
        raise ValueError('RVC retrieval requires an L2 index')
    vectors = index.reconstruct_n(0,index.ntotal)
    if vectors.shape != (index.ntotal,dimension) or not np.isfinite(vectors).all():
        raise ValueError('Retrieval vectors are invalid')
    return index,vectors


def load_models(options,trace):
    # This function is called only inside the contained RVC worker.
    import os
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    sys.path.insert(0,str(SOURCE))
    from fairseq import checkpoint_utils
    from infer.lib.infer_pack import models as networks
    from infer.modules.vc.pipeline import Pipeline
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    cuda = options['device'] == 'cuda'
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for RVC but unavailable')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.85,3.4*1024**3/total))
    config = SimpleNamespace(device='cuda:0' if cuda else 'cpu',is_half=cuda,
                             x_pad=1,x_query=5,x_center=30,x_max=32)
    checkpoint = torch.load(options['model'],map_location='cpu')
    metadata = checkpoint_metadata(checkpoint)
    if options['speaker_id'] >= metadata['speakers']:
        raise ValueError('Speaker ID is outside the checkpoint speaker table')
    name = f"SynthesizerTrnMs{metadata['dimension']}NSFsid"+('' if metadata['f0'] else '_nono')
    net = getattr(networks,name)(*metadata['config'],is_half=config.is_half)
    del net.enc_q
    net.load_state_dict(checkpoint['weight'],strict=True)
    net = net.eval().to(config.device)
    net = net.half() if cuda else net.float()
    del checkpoint
    hubert_path = SOURCE/manifest()['assets']['hubert']['file']
    ensemble,_,_ = checkpoint_utils.load_model_ensemble_and_task([str(hubert_path)],suffix='')
    hubert = ensemble[0].eval().to(config.device)
    hubert = hubert.half() if cuda else hubert.float()
    pipeline = Pipeline(metadata['sample_rate'],config)
    if options['index_rate']:
        pipeline.validated_index,pipeline.validated_vectors = load_index(options['index'],metadata['dimension'])
    os.environ['rmvpe_root'] = str((SOURCE/manifest()['assets']['rmvpe']['file']).parent)
    trace.event('rvc_models_loaded',metadata={k:v for k,v in metadata.items() if k != 'config'},
                profile=vars(config),index_used=bool(options['index_rate']),source_commit=manifest()['commit'])
    return hubert,net,pipeline,metadata


def health():
    import importlib.metadata as packages
    verify_source(); verify_assets()
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != (ROOT/'venv-rvc').resolve():
        raise RuntimeError('Run RVC health with the isolated venv-rvc Python')
    import torch,torchaudio,faiss,librosa,fairseq,pyworld,parselmouth,torchcrepe
    if torch.__version__ != '2.1.0+cu121' or torchaudio.__version__ != torch.__version__:
        raise RuntimeError('Unexpected RVC CUDA stack')
    sys.path.insert(0,str(SOURCE))
    from infer.modules.vc.pipeline import Pipeline
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
    from infer.lib.rmvpe import RMVPE
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.cuda.is_available(),
                          source=manifest()['commit'],imports='passed',trained_voice='not qualified'),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--health',action='store_true',required=True)
    parser.parse_args()
    health()

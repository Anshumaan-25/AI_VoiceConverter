"""Single-GPU pilot loop using the pinned RVC networks and training losses."""
import json
import os
from pathlib import Path
import sys
import time

from speech.diagnostics import Trace,atomic_json


def train(directory,trace):
    import random
    import numpy as np
    import soundfile as sf
    import torch
    from torch.cuda.amp import autocast,GradScaler
    from rvc_runtime import SOURCE,ROOT,manifest,verify_source,verify_assets
    from rvc_training import (verify_training_assets,verify_cache,child,digest_json,checkpoint_set,
                              publish_checkpoint,load_config)
    from download_rvc_assets import sha256
    request = json.loads((directory/'request.json').read_text())
    workspace = Path(request['workspace'])
    config = load_config(request['config'])
    verify_source(); verify_assets(); verify_training_assets()
    cache = verify_cache(workspace)
    identity = dict(features_sha256=sha256(workspace/'features.json'),config=config,
                    training_assets=manifest()['training_assets'],
                    worker_sha256=sha256(Path(__file__)),source_commit=manifest()['commit'],
                    numerics_sha256=sha256(ROOT/'rvc_numerics.py'))
    if identity != request['identity']:
        raise ValueError('Training identity changed before worker start')
    sys.path.insert(0,str(SOURCE))
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid,MultiPeriodDiscriminatorV2
    from infer.lib.infer_pack import commons
    from infer.lib.train.losses import discriminator_loss,generator_loss,feature_loss,kl_loss
    from infer.lib.train.mel_processing import spectrogram_torch,spec_to_mel_torch,mel_spectrogram_torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pilot does not silently fall back to CPU')
    precision=config.get('precision','fp16')
    if precision not in ('fp16','bf16'):
        raise ValueError('Supported training precision is fp16 or bf16')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU/runtime does not support BF16; no silent precision fallback')
    amp_dtype=torch.bfloat16 if precision == 'bf16' else torch.float16
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(.85,3.4*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cudnn.benchmark = False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    hps = json.loads((SOURCE/'configs/v2/48k.json').read_text())
    data,model = hps['data'],hps['model']
    segment = config['segment_size']
    net_g = SynthesizerTrnMs768NSFsid(data['filter_length']//2+1,segment//480,**model,is_half=True,sr=48000)
    net_d = MultiPeriodDiscriminatorV2(model['use_spectral_norm'])
    checkpoint = checkpoint_set(workspace)
    migrating=False
    if checkpoint is None and (workspace/'recovery.json').exists():
        from rvc_recovery import parent_checkpoint
        checkpoint=parent_checkpoint(workspace,identity)
        migrating=True
    if not checkpoint:
        for net,key in ((net_g,'generator'),(net_d,'discriminator')):
            state = torch.load(SOURCE/manifest()['training_assets'][key]['file'],map_location='cpu')
            net.load_state_dict(state['model'],strict=True)
            del state
    net_g.cuda().train(); net_d.cuda().train()
    optim_g = torch.optim.AdamW(net_g.parameters(),lr=1e-4,betas=(.8,.99),eps=1e-9)
    optim_d = torch.optim.AdamW(net_d.parameters(),lr=1e-4,betas=(.8,.99),eps=1e-9)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g,gamma=.999875)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d,gamma=.999875)
    scaler = GradScaler(enabled=precision == 'fp16',init_scale=1024)
    step,epoch,offset = 0,0,0
    if checkpoint:
        state = torch.load(checkpoint[0],map_location='cpu')
        if not migrating and (state['identity'] != identity or checkpoint[1]['identity'] != identity):
            raise ValueError('Resume configuration/data/code identity mismatch')
        if migrating and state['identity'] != checkpoint[1]['identity']:
            raise ValueError('Parent state and complete-set manifest disagree')
        net_g.load_state_dict(state['g'],strict=True); net_d.load_state_dict(state['d'],strict=True)
        optim_g.load_state_dict(state['optim_g']); optim_d.load_state_dict(state['optim_d'])
        scheduler_g.load_state_dict(state['scheduler_g']); scheduler_d.load_state_dict(state['scheduler_d'])
        scaler.load_state_dict(state['scaler'])
        step,epoch,offset = state['step'],state['epoch'],state['offset']
        random.setstate(state['random']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
        del state
    clips = [c for c in cache['clips'] if c['split'] == 'train']
    if not clips:
        raise ValueError('Training split is empty')
    trace.event('training_loaded',step=step,epoch=epoch,offset=offset,clips=len(clips),
                profile=dict(batch_size=1,mixed_precision=precision,workers=0,gpu_cache=False,segment_size=segment),resumed=bool(checkpoint),
                migrated_parent=checkpoint[1]['sha256'] if migrating else None)
    from rvc_numerics import NumericalProbe
    probe=NumericalProbe(directory,trace)
    if config.get('diagnostic'): probe.attach(net_g,net_d)
    initial_step = step
    updates = {'g':0,'d':0}
    optim_g.register_step_post_hook(lambda *args: updates.__setitem__('g',updates['g']+1))
    optim_d.register_step_post_hook(lambda *args: updates.__setitem__('d',updates['d']+1))
    attempted,consecutive_skips,skipped = 0,0,0
    last_saved = step if checkpoint else -1

    def save():
        nonlocal last_saved
        for net in (net_g,net_d):
            if any(not torch.isfinite(v).all() for v in net.state_dict().values()):
                raise RuntimeError('Nonfinite model state; refusing checkpoint publication')
        state = dict(identity=identity,step=step,epoch=epoch,offset=offset,
                     g=net_g.state_dict(),d=net_d.state_dict(),optim_g=optim_g.state_dict(),optim_d=optim_d.state_dict(),
                     scheduler_g=scheduler_g.state_dict(),scheduler_d=scheduler_d.state_dict(),scaler=scaler.state_dict(),
                     random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        trace.event('checkpoint_start',step=step)
        metadata = publish_checkpoint(workspace,step,identity,lambda p: torch.save(state,p))
        last_saved = step
        trace.event('checkpoint_complete',step=step,bytes=metadata['bytes'])

    for attempt in range(request['steps']):
        if (directory/'stop').exists():
            break
        trace.begin_chunk(attempt)
        attempted += 1
        probe.step=step
        atomic_json(directory/'progress.json',dict(attempted_steps=attempted,step=step,initial_step=initial_step))
        order = list(range(len(clips))); random.Random(config['seed']+epoch).shuffle(order)
        clip = clips[order[offset]]
        source = child(Path(cache['dataset']),clip['source_file'])
        audio,sr = sf.read(str(source),dtype='float32')
        with np.load(child(workspace,clip['file']),allow_pickle=False) as arrays:
            features = arrays['features']; pitch = arrays['pitch']; coarse = arrays['coarse']
        frames = clip['frames']
        if sr != 48000 or audio.ndim != 1 or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
            raise ValueError('Invalid training PCM scaling')
        if features.shape != (frames,768) or pitch.shape != (frames,) or coarse.shape != (frames,):
            raise ValueError('Cached frame alignment/width mismatch')
        wave = torch.from_numpy(audio[:frames*480].copy())[None,None].cuda()
        phone = torch.from_numpy(features.copy())[None].cuda()
        pitch_t = torch.from_numpy(coarse.copy())[None].cuda()
        pitchf = torch.from_numpy(pitch.copy())[None].cuda()
        length = torch.tensor([frames],device='cuda',dtype=torch.long)
        sid = torch.zeros(1,device='cuda',dtype=torch.long)
        spec = spectrogram_torch(wave[:,0],2048,48000,480,2048,center=False)
        if spec.shape[-1] != frames:
            raise ValueError('Spectrogram frame alignment differs from cached features')
        before_updates = dict(updates)
        trace.event('training_input',step=step,epoch=epoch,offset=offset,clip=clip['source_file'])
        with autocast(dtype=amp_dtype):
            y_hat,ids_slice,_,z_mask,latent = net_g(phone,length,pitch_t,pitchf,spec,length,sid)
            z,z_p,m_p,logs_p,m_q,logs_q = latent
            mel = spec_to_mel_torch(spec,2048,128,48000,0,None)
            y_mel = commons.slice_segments(mel,ids_slice,segment//480)
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1),2048,128,48000,480,2048,0,None)
            real = commons.slice_segments(wave,ids_slice*480,segment)
            dr,dg,_,_ = net_d(real,y_hat.detach())
            loss_d,_,_ = discriminator_loss(dr,dg)
        if not torch.isfinite(loss_d):
            raise RuntimeError('Nonfinite discriminator loss')
        optim_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward(); scaler.unscale_(optim_d)
        if config.get('diagnostic') or precision == 'bf16':
            finite=probe.gradients('d',net_d.parameters())
            if precision == 'bf16' and not finite:
                raise RuntimeError('Nonfinite unscaled discriminator gradients; update rejected')
        scaler.step(optim_d)
        # D is fixed for G's update; gradients still propagate through its input.
        optim_d.zero_grad(set_to_none=True)
        for p in net_d.parameters(): p.requires_grad_(False)
        with autocast(dtype=amp_dtype):
            dr,dg,fr,fg = net_d(real,y_hat)
            with autocast(enabled=False):
                loss_mel = torch.nn.functional.l1_loss(y_mel.float(),y_hat_mel.float())*45
                loss_kl = kl_loss(z_p,logs_q,m_p,logs_p,z_mask)
                loss_fm = feature_loss(fr,fg)
                loss_gen,_ = generator_loss(dg)
                loss_g = loss_gen+loss_fm+loss_mel+loss_kl
        if not torch.isfinite(loss_g):
            raise RuntimeError('Nonfinite generator loss')
        optim_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward(); scaler.unscale_(optim_g)
        if config.get('diagnostic') or precision == 'bf16':
            finite=probe.gradients('g',net_g.parameters())
            if precision == 'bf16' and not finite:
                raise RuntimeError('Nonfinite unscaled generator gradients; update rejected')
        scaler.step(optim_g); scaler.update()
        for p in net_d.parameters(): p.requires_grad_(True)
        optim_g.zero_grad(set_to_none=True)
        success = all(updates[k] > before_updates[k] for k in updates)
        skipped += not success
        consecutive_skips = 0 if success else consecutive_skips+1
        if consecutive_skips >= config['max_consecutive_skips']:
            raise RuntimeError('Repeated AMP skipped updates; last complete checkpoint preserved')
        step += int(success)
        offset += 1
        if offset == len(clips):
            offset = 0; epoch += 1
            scheduler_g.step(); scheduler_d.step()
        trace.event('optimizer_step',attempted_steps=attempted,step=step,success=success,
                    skipped_updates=skipped,g_updates=updates['g'],d_updates=updates['d'],
                    loss_g=float(loss_g.detach()),loss_d=float(loss_d.detach()),
                    loss_mel=float(loss_mel.detach()),loss_kl=float(loss_kl.detach()),scale=scaler.get_scale(),
                    clip=clip['source_file'])
        atomic_json(directory/'progress.json',dict(attempted_steps=attempted,step=step,initial_step=initial_step))
        # Release graphs before checkpoint serialization and the next example.
        del wave,phone,pitch_t,pitchf,length,sid,spec,y_hat,ids_slice,z_mask,latent,z,z_p,m_p,logs_p,m_q,logs_q
        del mel,y_mel,y_hat_mel,real,dr,dg,fr,fg,loss_g,loss_d,loss_mel,loss_kl,loss_fm,loss_gen
        if success and step % config['save_every'] == 0:
            save()
    if step > 0 and step != last_saved:
        save()
    if step > 0:
        from rvc_training import checkpoint_set
        parent = checkpoint_set(workspace)
        cfg = [1025,32,model['inter_channels'],model['hidden_channels'],model['filter_channels'],model['n_heads'],
               model['n_layers'],model['kernel_size'],model['p_dropout'],model['resblock'],model['resblock_kernel_sizes'],
               model['resblock_dilation_sizes'],model['upsample_rates'],model['upsample_initial_channel'],
               model['upsample_kernel_sizes'],model['spk_embed_dim'],model['gin_channels'],48000]
        weights = {k:v.detach().cpu().half() for k,v in net_g.state_dict().items() if not k.startswith('enc_q.')}
        export = workspace/'pilot-v2-48k.pth'; temporary=export.with_suffix('.tmp')
        torch.save(dict(weight=weights,config=cfg,version='v2',f0=1,sr='48k',
                        info=f'Engineering pilot; {step} successful paired updates; not voice-quality qualified',
                        training_step=step,parent_sha256=parent[1]['sha256']),temporary)
        os.replace(temporary,export)
        atomic_json(workspace/'export.json',dict(file=export.name,sha256=sha256(export),step=step,parent_sha256=parent[1]['sha256'],quality='unqualified pilot'))
    result = dict(initial_step=initial_step,step=step,successful_steps=step-initial_step,attempted_steps=attempted,
                  skipped_updates=skipped,epoch=epoch,offset=offset,stopped=(directory/'stop').exists(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(directory/'training.json',result)
    trace.event('complete',**result)


def main(directory):
    while not (directory/'go').exists(): time.sleep(.02)
    trace=Trace(directory); trace.event('worker_start')
    try:
        train(directory,trace)
        return 0
    except Exception as exc:
        import traceback
        import torch
        error=dict(message=str(exc),type=type(exc).__name__,traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            error.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(directory/'error.json',error)
        trace.event('failed',error_type=type(exc).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1])))

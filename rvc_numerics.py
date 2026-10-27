"""Worker-only numerical probes; preserve the first failing tensor's location."""
import json


class NumericalFailure(RuntimeError):
    pass


def tensors(value, name='output'):
    import torch
    if isinstance(value,torch.Tensor):
        yield name,value
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):
            yield from tensors(item,f'{name}.{index}')
    elif isinstance(value,dict):
        for key,item in value.items():
            yield from tensors(item,f'{name}.{key}')


class NumericalProbe:
    def __init__(self,directory,trace):
        self.directory,self.trace=directory,trace
        self.step=-1
        self.hooks=[]

    def inspect(self,name,value,record=False):
        import torch
        for suffix,tensor in tensors(value):
            if not tensor.is_floating_point() or not tensor.numel(): continue
            detached=tensor.detach()
            finite=torch.isfinite(detached)
            valid=bool(finite.all())
            if record or not valid:
                values=detached[finite].float()
                stats=dict(step=self.step,module=name,tensor=suffix,dtype=str(tensor.dtype),shape=list(tensor.shape),
                           finite=valid,nan=int(torch.isnan(detached).sum()),inf=int(torch.isinf(detached).sum()),
                           minimum=float(values.min()) if values.numel() else None,
                           maximum=float(values.max()) if values.numel() else None)
                with (self.directory/'numerics.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(stats)+'\n')
                if not valid:
                    from speech.diagnostics import atomic_json
                    atomic_json(self.directory/'first-nonfinite.json',stats)
                    self.trace.event('first_nonfinite',**stats)
                    raise NumericalFailure(f'First nonfinite tensor at {name}/{suffix} ({tensor.dtype})')

    def attach(self,generator,discriminator):
        selected={'enc_p','enc_q','enc_q.pre','enc_q.proj','flow','dec','dec.conv_pre','dec.m_source'}
        for name,module in generator.named_modules():
            if name in selected or name.startswith('dec.ups.') or name.startswith('enc_p.encoder.attn_layers.') and name.count('.') == 3:
                self.hooks.append(module.register_forward_hook(
                    lambda mod,args,result,label='g.'+name:self.inspect(label,result,record=self.step%25 == 0)))
        for name,module in discriminator.named_modules():
            if name.startswith('discriminators.') and name.count('.') == 1:
                self.hooks.append(module.register_forward_hook(
                    lambda mod,args,result,label='d.'+name:self.inspect(label,result,record=self.step%25 == 0)))

    def gradients(self,name,parameters):
        import torch
        grads=[p.grad.detach() for p in parameters if p.grad is not None]
        if not grads: return
        finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
        self.trace.event('gradient_check',network=name,step=self.step,finite=bool(finite))
        return bool(finite)

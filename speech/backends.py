"""Model imports live exclusively in the worker. All methods return mono float PCM."""
import base64
import io
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from .contracts import SpeechError


def seed_torch(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(16, int(os.environ.get('VOICE_CPU_THREADS', '4')))))


class BarkBackend:
    def __init__(self, options, trace):
        os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
        os.environ['SUNO_OFFLOAD_CPU'] = 'True'
        import torch
        from bark import generation as bark
        self.bark, self.options, self.trace = bark, options, trace
        available = torch.cuda.is_available()
        if options.device == 'cuda' and not available:
            raise SpeechError('CUDA was requested but is unavailable')
        if options.device == 'auto' and not available:
            raise SpeechError('CUDA unavailable; automatic mode will use the CPU fallback. Select --device cpu for explicit Bark CPU diagnostics')
        use_gpu = available and options.device != 'cpu'
        if use_gpu:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.85, 3.4*1024**3/total))
            torch.cuda.reset_peak_memory_stats()
        seed_torch(options.seed)
        trace.event('bark_load', torch=torch.__version__, uses_cuda=use_gpu,
                    small_models=True, cpu_offload=True)
        bark.preload_models(text_use_gpu=use_gpu, coarse_use_gpu=use_gpu,
                            fine_use_gpu=use_gpu, codec_use_gpu=use_gpu)
        self.history = None if options.bark_voice == 'none' else options.bark_voice
        self.sample_rate = int(bark.SAMPLE_RATE)
        if options.diagnostics:
            from audio_io import write_wav
            from .diagnostics import fingerprint
            trace.event('bark_checkpoint_fingerprints', checkpoints={name: fingerprint(bark._get_ckpt_path(name, use_small=True))
                        for name in ('text','coarse','fine')})
            if self.history:
                trace.event('bark_reference_codec_start')
                prompt = bark._load_history_prompt(self.history)
                decoded = bark.codec_decode(prompt['fine_prompt'][:, :300])
                write_wav(trace.directory/'reference-codec.wav', decoded, self.sample_rate)
                trace.event('bark_reference_codec_done', seconds=len(decoded)/self.sample_rate)

    def generate(self, text, max_seconds, index):
        b, o, t = self.bark, self.options, self.trace
        seed_torch((o.seed+index) % 2**32)
        tokenizer = b.models['text']['tokenizer']
        if len(tokenizer.encode(text, add_special_tokens=False)) > 256:
            raise SpeechError('Bark tokenizer limit exceeded; refusing text truncation')
        t.event('bark_semantic_start', duration_limit=max_seconds)
        semantic = b.generate_text_semantic(text, history_prompt=self.history,
                    temp=o.bark_text_temp, min_eos_p=o.bark_min_eos_p,
                    max_gen_duration_s=max_seconds, use_kv_caching=True, silent=True)
        t.event('bark_semantic_done', tokens=len(semantic), unique=int(len(np.unique(semantic))))
        if o.diagnostics:
            np.save(t.directory/f'{index:03d}-semantic.npy', semantic, allow_pickle=False)
        if not len(semantic) or len(semantic)/b.SEMANTIC_RATE_HZ >= max_seconds:
            raise SpeechError('Bark semantic decoder reached its duration limit or returned no tokens')
        t.event('bark_coarse_start')
        coarse = b.generate_coarse(semantic, history_prompt=self.history,
                    temp=o.bark_waveform_temp, use_kv_caching=True, silent=True)
        t.event('bark_coarse_done', shape=list(coarse.shape), unique=int(len(np.unique(coarse))))
        t.event('bark_fine_start')
        fine = b.generate_fine(coarse, history_prompt=self.history, temp=.5, silent=True)
        t.event('bark_fine_done', shape=list(fine.shape), unique=int(len(np.unique(fine))))
        if o.diagnostics:
            np.savez(t.directory/f'{index:03d}-acoustic.npz', coarse=coarse, fine=fine)
        t.event('bark_decode_start')
        audio = b.codec_decode(fine)
        t.event('bark_decode_done', frames=len(audio), sample_rate=self.sample_rate)
        return np.asarray(audio, dtype=np.float32), self.sample_rate


class CoquiBackend:
    def __init__(self, options, trace):
        seed_torch(options.seed)
        from TTS.api import TTS
        self.options, self.trace = options, trace
        trace.event('coqui_load', model=options.coqui_model, device='cpu')
        self.tts = TTS(model_name=options.coqui_model, progress_bar=False, gpu=False)
        synth = self.tts.synthesizer
        if options.diagnostics:
            from .diagnostics import fingerprint
            trace.event('coqui_checkpoint_fingerprints', checkpoints={name: fingerprint(value)
                        for name in ('tts_checkpoint','vocoder_checkpoint')
                        if (value := getattr(synth, name, None)) and Path(value).is_file()})
        self.sample_rate = int(synth.output_sample_rate)
        self.model_rate = int(synth.tts_config.audio.sample_rate)
        self.hop = int(synth.tts_config.audio.hop_length)
        self.decoder = getattr(synth.tts_model, 'decoder', None)
        if self.decoder is None or not hasattr(self.decoder, 'max_decoder_steps'):
            raise SpeechError('This Coqui model has no supported hard decoder bound; select a supported Tacotron model')
        self.original_inference = self.decoder.inference
        # Observe and reject a decoder-cap hit before postnet/vocoder synthesis.
        def bounded_inference(*args, **kwargs):
            result = self.original_inference(*args, **kwargs)
            outputs, alignment, stops = result
            steps = int(stops.numel())
            trace.event('coqui_decoder_done', steps=steps,
                        limit=self.decoder.max_decoder_steps,
                        final_stop_probability=float(stops.reshape(-1)[-1]),
                        alignment_shape=list(alignment.shape))
            if options.diagnostics:
                np.save(trace.directory/f'{trace.chunk:03d}-alignment.npy',
                        alignment.detach().cpu().numpy(), allow_pickle=False)
            if steps >= self.decoder.max_decoder_steps:
                raise SpeechError('Coqui decoder reached its hard step limit before completing speech')
            return result
        self.decoder.inference = bounded_inference

    def generate(self, text, max_seconds, index):
        seed_torch((self.options.seed+index) % 2**32)
        reduction = int(self.decoder.r)
        self.decoder.max_decoder_steps = max(1, math.floor(max_seconds*self.model_rate/(self.hop*reduction)))
        self.trace.event('coqui_synthesis_start', decoder_limit=self.decoder.max_decoder_steps,
                         model_sample_rate=self.model_rate, output_sample_rate=self.sample_rate)
        kwargs = {}
        if self.options.coqui_speaker:
            kwargs['speaker'] = self.options.coqui_speaker
        if self.options.coqui_language:
            kwargs['language'] = self.options.coqui_language
        audio = self.tts.tts(text=text, split_sentences=False, **kwargs)
        return np.asarray(audio, dtype=np.float32), self.sample_rate


class SapiBackend:
    """Offline Windows voices. Native PCM stays in memory until PCM24 export."""
    def __init__(self, options, trace):
        if os.name != 'nt':
            raise SpeechError('The SAPI fallback requires Windows and an installed speech voice')
        self.options, self.trace = options, trace
        self.script = Path(__file__).with_name('sapi.ps1')

    def generate(self, text, max_seconds, index):
        import json
        import soundfile as sf
        self.trace.event('sapi_synthesis_start', voice=self.options.sapi_voice or 'Windows default')
        request = json.dumps(dict(text=text, voice=self.options.sapi_voice), ensure_ascii=False)
        result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive',
                                 '-ExecutionPolicy', 'Bypass', '-File', str(self.script)],
                                input=request.encode('utf-8'), capture_output=True,
                                timeout=min(self.options.chunk_timeout, 60),
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise SpeechError('Windows speech failed: '+result.stderr.decode('utf-8', errors='replace')[-1500:])
        response = json.loads(result.stdout.decode('utf-8-sig'))
        self.trace.event('sapi_synthesis_done', voice=response['voice'])
        audio, sr = sf.read(io.BytesIO(base64.b64decode(response['wav'])), dtype='float32')
        return audio, sr


def load_backend(name, options, trace):
    return {'bark': BarkBackend, 'coqui': CoquiBackend, 'sapi': SapiBackend}[name](options, trace)

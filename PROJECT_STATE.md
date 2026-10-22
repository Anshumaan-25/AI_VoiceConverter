# Project state — 2026-10-23

Branch: `bark-chunk-and-stitch`.

The active architecture is Bark/Coqui TTS plus so-vits-svc conversion and a Gradio frontend. A standalone CPU mix/export utility now exists. The Demucs/RVC implementation from the supplied handoff is absent.

The runtime was repaired in `venv310`: Python 3.10.0, torch and torchaudio 2.1.0+cu121, working Bark, Gradio, fairseq, RMVPE dependencies, and a passing package consistency check. See requirements-runtime.txt and requirements-lock.txt.

Safety fixes cover trimming, chunk failures, audio validation/PCM24 storage, path containment, temporary inputs, dataset split guards, GPU concurrency, memory profiles, tensor devices, cached resampling, index weights, checkpoint validation, and subprocess failure propagation. Upstream fixes are stored in a reproducible patch rather than pushing to the third-party repository.

The original recordings, existing checkpoints and previously generated features were preserved. Existing unsliced recordings are rejected for training. `G_0.pth` is retained but rejected by the supported inference wrapper. A separate synthetic low-level test used it only to test execution and memory, not quality.

Superseded decisions: five-word punctuation-free Bark chunks and a 16-bit/44.1 kHz file contract. Current persisted audio is 24-bit/48 kHz; model rates remain checkpoint-defined. Bark uses speaker 6 with text temperature 0.3, waveform temperature 0.35 and EOS probability 0.2.

Read AUDIT_REPORT.md for verified results, corrections to prior diagnoses, and remaining work. The real Bark smoke test still produced a sustained ~625 Hz tone. A new speech-only quality guard rejects it, and auto mode falls back to Coqui. The CPU Coqui smoke also ran away to 85.68 seconds for four words; a new duration guard rejects that output. TTS quality remains a blocker. The generative defects and voice similarity are not claimed solved. Optional upstream features remain unqualified.

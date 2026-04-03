# PROJECT STATE
Date: 2026-04-01
Purpose: Handoff context for migration to antigravity google agent (Opus 4.6).

## Current Snapshot
This project is operational for so-vits-svc workflow orchestration and standalone TTS generation, with one unresolved audio-quality issue: periodic beeping artifacts around ~4-second behavior windows in some Bark outputs.

## Canonical TTS Run Command
python tts_engine.py --text "testing the system no more mistakes just clean lines thats all it takes" --output test.wav --model bark

## Sterile-Room Protocol
- Extreme Text Sanitization: Use Regex to remove ALL punctuation (re.sub(r'[^a-zA-Z0-9\s]', '', text)). No commas, periods, or slashes.
- Inference Control: Set min_eos_p = 0.2, text_temp = 0.3, and waveform_temp = 0.35 to stabilize the output.
- Stop-Loss Trimmer: Implement a Hard Noise Gate. If audio drops below -40dB for >100ms, crop the file immediately to kill trailing hallucinations.
- Speaker Lock: Hard-code the speaker to v2/en_speaker_6 for maximum stability.

## Open Issue: 4-Second Beeping
Status: Unresolved, priority blocker.

Impact:
- Audible periodic beeping/tonal artifact appears around roughly 4-second behavior windows in generated output.
- Artifact reduces production usability for clean TTS handoff into downstream conversion.

Current hypothesis:
- Bark stability and trailing-generation hallucination/noise-floor behavior are interacting with chunk boundaries or tail regions.

Current mitigation status:
- Sterile-Room Protocol is the active control strategy.
- Issue remains open pending further validation and targeted trimming/noise-gate hardening.

## Working Modules
- tts_engine.py: Standalone TTS engine (Bark primary, Coqui fallback), WAV output pipeline.
- voice_autopilot.py: End-to-end workflow wrapper (check, prepare-data, preprocess, train, infer, performance).
- gradio_app.py: Inference-first local UI for so-vits-svc conversion.
- setup_autopilot.ps1: Environment bootstrap and compatibility pinning flow.
- download_pretrain.ps1: Pretrained asset bootstrap.
- so-vits-svc/: Core conversion/training stack integrated with wrappers and UI.

## Locked Environment Specs
- Environment: venv310
- Python: 3.10
- Torch: 2.1.x

## Environment Drift Note
- setup_autopilot.ps1 currently creates .venv310 and installs torch from the CUDA 12.1 index without an explicit torch==2.1.x pin.
- For strict reproducibility during migration, treat venv310 + Python 3.10 + Torch 2.1.x as the contract and align bootstrap automation accordingly.

## Operational Notes
- Workflow remains inference-first, then custom training.
- Compatibility constraints currently used in this repo include pip < 24.1 and setuptools < 81 for fairseq/librosa behavior.
- Untitled0.ipynb is non-canonical to this production pipeline.

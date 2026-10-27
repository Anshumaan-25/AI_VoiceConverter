# Project state — 2026-10-27

Branch: `bark-chunk-and-stitch`. TTS/UI integration and the Milestone 2 RVC foundation are implemented. See `MILESTONE_1_REPORT.md`, `INTEGRATION_REPORT.md` and `MILESTONE_2_REPORT.md`. RVC real-model qualification remains blocked by the absent trained checkpoint.

The TTS controller delegates to bounded, isolated process trees. Speech generation, per-chunk audio validation and conversion are separate interfaces. Bark uses small/offloaded models at temperatures 0.7/0.7. Automatic mode tries Bark then offline Windows SAPI. Coqui remains diagnostic-only because the existing Tacotron model still has seed/text-sensitive decoder stopping failures.

English qualification: Bark completed six benchmark requests with 5.56% aggregate ASR word error; SAPI completed twelve with 1.85%. Coqui completed five of six, with the failure rejected before vocoder synthesis. VAD does not establish transcript fidelity. Expressive quality, multilingual speech and singing require further qualification.

The Windows job contains workers and descendants before they can load models. Startup, per-chunk and whole-request timeouts, cancellation and crash cleanup are covered by tests. A real CUDA allocation was reclaimed after forced worker termination. Main-process imports do not load PyTorch. All persisted speech remains 48 kHz PCM24 WAV; MP3 encoding remains restricted to final export.

The UI connects speech generation to the existing so-vits converter through a separate adapter. It defaults to offline SAPI, preserves uploaded PCM24 bytes, isolates cancellation per session and bounds conversion worker lifetime. Automatic TTS reserves time for fallback. The Gradio/Starlette template incompatibility is fixed by pinning Starlette 0.52.1; health checks now render the UI over HTTP. All 37 tests passed with the real CUDA lifecycle test enabled, and a live queued UI request produced 48 kHz PCM24 mono speech.

RVC now uses isolated `venv-rvc`, official source commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`, a reviewed inference patch and hash-verified HuBERT/RMVPE assets. The controller remains free of model imports. RVC and so-vits have separate UI controls. All 51 tests passed with both CUDA lifecycle tests enabled, and both environment health checks passed. A real encoder/pitch GPU probe peaked at 545112576 allocated bytes; this is not trained-voice conversion evidence.

`Anshumaan_RVC.wav` is unmodified, SHA256 `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`. Derived dataset version `0dddb82392f1860af8a3` contains 179 training clips (947.26 seconds) and 17 validation clips (97.003 seconds), all 48 kHz PCM24 mono. Its same-session temporal holdout is provisional, with a 20-second source gap. The original file and older so-vits dataset remain preserved. No full training or voice checkpoint was created.

Next milestone: plan a bounded 4 GB training pilot, export a compatible trained inference checkpoint, then complete real RVC inference/index/memory/voice-quality qualification. Current model-dependent acceptance criteria are explicitly pending. Human review concerns speaker/noise consistency and audio quality, not implementation/debugging work.

The historical repository audit remains in `AUDIT_REPORT.md`; its original TTS blocker section is superseded by the milestone report. Original recordings and model checkpoints are preserved.

## Working agreement for subsequent tasks

After closing this interrupted implementation, this planning assistant should produce Markdown task plans only. The user intends to use GPT-6.1 sol for execution. Future plans should specify scope, affected files, dependencies, expected benefits and risks, verification steps, stop conditions and a clear execution handoff. Do not start implementing the next milestone from this planning conversation without a new explicit change to that agreement.

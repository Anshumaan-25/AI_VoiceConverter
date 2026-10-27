# Project state — 2026-10-27

Branch: `bark-chunk-and-stitch`. The interrupted TTS milestone and UI integration are implemented and verified; see `MILESTONE_1_REPORT.md` and `INTEGRATION_REPORT.md`.

The TTS controller now delegates to bounded, isolated process trees. Speech generation, per-chunk audio validation and the future conversion boundary are separate interfaces. Bark uses small/offloaded models at temperatures 0.7/0.7. Automatic mode tries Bark then offline Windows SAPI. Coqui remains diagnostic-only because the existing Tacotron model still has seed/text-sensitive decoder stopping failures.

English qualification: Bark completed six benchmark requests with 5.56% aggregate ASR word error; SAPI completed twelve with 1.85%. Coqui completed five of six, with the failure rejected before vocoder synthesis. VAD does not establish transcript fidelity. Expressive quality, multilingual speech and singing require further qualification.

The Windows job contains workers and descendants before they can load models. Startup, per-chunk and whole-request timeouts, cancellation and crash cleanup are covered by tests. A real CUDA allocation was reclaimed after forced worker termination. Main-process imports do not load PyTorch. All persisted speech remains 48 kHz PCM24 WAV; MP3 encoding remains restricted to final export.

The UI connects speech generation to the existing so-vits converter through a separate adapter. It defaults to offline SAPI, preserves uploaded PCM24 bytes, isolates cancellation per session and bounds conversion worker lifetime. Automatic TTS reserves time for fallback. The Gradio/Starlette template incompatibility is fixed by pinning Starlette 0.52.1; health checks now render the UI over HTTP. All 37 tests passed with the real CUDA lifecycle test enabled, and a live queued UI request produced 48 kHz PCM24 mono speech.

`Anshumaan_RVC.wav` is unmodified. A read-only scan measured 1351.143 seconds (22:31), 48 kHz PCM24 mono, zero near-fullscale samples and approximately 1020.88 seconds detected as speech. The SHA256 is unchanged. No slices, training runs or voice model were created from it. The intended RVC adapter, dataset preparation and training remain future work, not manual prerequisites for the completed TTS milestone.

The historical repository audit remains in `AUDIT_REPORT.md`; its original TTS blocker section is superseded by the milestone report. Original recordings and model checkpoints are preserved.

## Working agreement for subsequent tasks

After closing this interrupted implementation, this planning assistant should produce Markdown task plans only. The user intends to use GPT-6.1 sol for execution. Future plans should specify scope, affected files, dependencies, expected benefits and risks, verification steps, stop conditions and a clear execution handoff. Do not start implementing the next milestone from this planning conversation without a new explicit change to that agreement.

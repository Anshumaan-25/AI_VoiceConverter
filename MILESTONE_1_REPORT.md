# Milestone 1 — isolated TTS generation

Date: 2026-10-26. Windows 11, Python 3.10.0, PyTorch 2.1.0+cu121, RTX 3050 Laptop (4096 MiB).

Follow-up completed 2026-10-27: the UI integration, additional dependency/lifecycle fixes and expanded 37-test verification are documented in [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md). This document retains the original synthesis qualification evidence below.

The custom recording `Anshumaan_RVC.wav` was not processed, trained on, uploaded or edited. Its SHA256 remains `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`.

## Result

The application generates intelligible English speech through isolated Bark workers or an offline Windows fallback. All 28 regression/lifecycle tests passed with the optional real CUDA test enabled. This qualifies the tested workflow; it does not establish perfect transcript fidelity, musical suitability, universal model reliability or the future RVC integration.

Automatic mode tries Bark then SAPI. Coqui is excluded from automatic mode because a reproducible decoder stopping failure remains. Explicit `--model coqui` remains available for diagnosis. For dependable script delivery and zero GPU use, select `--model sapi`; Bark remains the more expressive but less faithful option.

## Implementation

| Component | Change |
|---|---|
| `tts_engine.py` | Thin CLI preserving voice/model controls, with diagnostics, CPU/CUDA selection, stage/request deadlines and chunk-duration limits. Removes the old in-process model path and obsolete processing helpers. |
| `speech/contracts.py`, `text.py` | Generation, validation and future conversion interfaces. Sentence-preserving chunks: at most 20 words/180 characters each, 128 chunks and 10000 characters per request. |
| `speech/backends.py` | Bark semantic-duration bound, KV caching and separately traced semantic/coarse/fine/codec stages. Coqui step bound derived from sample rate, hop length and reduction factor; cap hits fail before vocoder synthesis. SAPI uses native PCM in memory. |
| `speech/validation.py` | Per-chunk duration, finite-value, silence, saturation, carrier-tone and WebRTC VAD checks before stitching and after resampling. Retains 120 ms edge padding and only attenuates gain. Floating PCM above unity is attenuated rather than incorrectly classified as clipped. |
| `speech/processes.py`, `supervisor.py` | Workers start suspended, join a Windows kill-on-close job, then resume. This contains the venv launcher's interpreter and SAPI descendants. Confirmed termination precedes fallback. Controller imports do not load PyTorch. |
| `speech/diagnostics.py`, `worker.py` | Stage times, PID, seed/settings, native rates, package/model provenance, optional checkpoint hashes, CUDA allocation peaks, sampled process-tree RAM, tokens and alignment traces. Unique directories prevent stale/cross-request files. Atomic replacement retries tolerate brief Windows reader locks. |
| `gpu_lock.py` | Distinguishes GPU contention so automatic mode can select its CPU fallback. |
| `speech/sapi.ps1` | Offline installed English voice, preferring Microsoft David Desktop. JSON stdin carries plain speech text; it is never executable PowerShell or SSML. |
| `benchmark_speech.py` | Fixed corpus, repeat runs, independent CPU transcription, word-error reporting and interrupted-run resume support. |

Fallback restarts the entire utterance with one engine. Existing destinations survive generation failures. Publication is atomic; publication errors do not trigger regeneration with a different voice. Default loading/chunk/request deadlines are 240/180/900 seconds. Decoder limits stop excess model work; a hung native call is terminated at its wall-clock deadline, not instantaneously. If every backend is unavailable, failure is bounded and explicit.

## Diagnosed failures

**Bark:** At the previous temperatures 0.3/0.35, speaker 6 and seed 42, the four-word phrase reached the semantic-duration cap: 300 semantic tokens with only 14 unique values. The reference prompt's acoustic tokens decoded to recognizable speech, showing that the codec could operate correctly. At 0.7/0.7, the same phrase/voice/seed produced the requested words. The new defaults match upstream. This supports low-temperature semantic repetition as a contributor to this reproduced failure; it does not prove every tone artifact has that cause.

**Coqui:** At seed 42, the four-word phrase reached the 516-step cap with final stop probability approximately 0.0083. The worker rejected it before vocoder synthesis. Seed 43 succeeded, and five of six corpus requests succeeded at seed 42. This is a text/seed-sensitive decoder/attention failure. We did not weaken the stop threshold or label truncated speech successful. The exact learned-model cause remains unresolved; automatic mode bypasses this configuration.

**Fallback:** Edge TTS uses compressed MP3 transport. The offline Windows alternative preserves the PCM/WAV intermediate policy and avoids a network-dependent fallback. Its tested native output is 22.05 kHz PCM16 in memory; PCM24 storage and upsampling do not create additional source detail. No intermediate MP3 is generated or downloaded.

## Real speech qualification

Independent recognizer: Vosk small English US 0.15 on CPU, using unrestricted recognition with no supplied transcript grammar. Corpus/transcripts are in [validation/milestone1-results.json](validation/milestone1-results.json). Persisted audio is 48 kHz PCM24 mono.

| Engine | Completed requests | Aggregate ASR word error | Measured request time |
|---|---:|---:|---:|
| Bark, revised defaults | 6 / 6 | 5.56% | 35.7–57.8 seconds |
| Coqui, seed 42 | 5 / 6 | 0% on completed cases only | 19.7–39.6 seconds |
| SAPI, two repetitions | 12 / 12 | 1.85% | 3.9–5.8 seconds |

Benchmark screening required no rejected requests, aggregate WER at most 15%, and per-request WER at most 35%. These are milestone thresholds, not human-approved production quality. Coqui fails because one request did not complete. The recognizer itself can make errors: both SAPI substitutions were “ready” recognized as “reading.”

A final three-sentence Bark request passed chunk checks but ASR detected three inserted words/fillers among 31 reference words (9.68% WER). Bark remains imperfect for exact scripts. A real automatic-mode test reproduced the low-temperature Bark failure and fell back to SAPI; its four-word result transcribed exactly.

Local review samples: `logs/milestone1/ready-for-review.wav` (Bark) and `logs/milestone1/verified-fallback.wav` (SAPI). Full traces/audio stay in ignored `logs/`; compact evidence is in the validation JSON. Listening approval has not been claimed.

## Memory and recovery

The measured Bark benchmark peak PyTorch allocation was 989.5 MiB. Sampled peak process-tree system RAM was approximately 3965 MiB; system RAM is distinct from VRAM. The allocator ceiling is at most 85% of this 4 GB GPU, with CPU offloading. It does not constrain other applications or every native GPU allocation.

Device usage returned to the observed idle range of 0–9 MiB after Bark workers exited, without an accumulating worker allocation. SAPI used no GPU allocation. Some Coqui device-wide readings overlapped separate Bark tests and cannot be attributed to Coqui; its worker hides CUDA and runs on CPU.

The CUDA lifecycle regression allocated over 100 MiB in a real worker, forced a timeout, verified process death and checked device memory returned within 32 MiB of baseline. Other tests verified loading/chunk timeout, cancellation, worker crash, abrupt controller death, descendant cleanup and GPU contention. No retained worker GPU allocation was observed in these bounded tests; this is not a proof for every driver/workload combination.

## Verification and limits

- All 28 tests passed, including the real CUDA test. PCM24, resampling, dataset/path and FFmpeg 320 kbps regressions remain covered.
- Runtime health and package-consistency checks passed. The CUDA stack was preserved. Added WebRTC VAD; Vosk is an optional benchmark dependency.
- Generation, validation and conversion boundaries are separate. No RVC adapter, recording ingestion or training changes are included.
- VAD/tone/duration checks cannot prove transcript fidelity. English speech was tested; singing and multilingual output remain unqualified.
- Downloads count against loading timeouts. Missing voices/models fail explicitly. Diagnostic text/audio is retained only on opt-in; ordinary scratch files are removed.
- The initial qualification was performed locally. See the integration report and branch history for the subsequent delivery. No system power action is part of this milestone.

References: [Bark implementation](https://github.com/suno-ai/bark/blob/main/bark/generation.py), [Coqui Tacotron](https://docs.coqui.ai/en/latest/models/tacotron1-2.html), [WebRTC VAD](https://github.com/wiseman/py-webrtcvad), [Edge TTS transport](https://github.com/rany2/edge-tts/blob/master/src/edge_tts/communicate.py), [Vosk models](https://alphacephei.com/vosk/models).

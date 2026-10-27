# TTS application integration — 2026-10-27

The interrupted implementation is complete for the tested local English speech workflow. RVC implementation and custom-model training remain a separate milestone. This is not a claim of universal speech quality or complete production qualification.

## Implemented and verified

- `gradio_app.py` now provides speech generation, playback, optional diagnostics, cancellation and a handoff to the existing so-vits conversion tab. Offline SAPI is the UI default; Bark and automatic fallback remain selectable. Coqui is explicitly diagnostic-only.
- `app_service.py` owns unique output directories and one active job per browser session. Cancellation cannot target another session. Gradio 3 session state supplies session identity; its Request object does not expose the newer session-hash API.
- `conversion.py` implements the modular `VoiceConverter` boundary for so-vits. It requires 48 kHz PCM24 mono input and a separate WAV destination. Uploads are normalized into a temporary copy; originals remain unchanged. A missing/unqualified model fails explicitly.
- File uploads and playback preserve PCM24 bytes. Gradio's audio-input preprocessing is bypassed to avoid silently reducing uploaded audio to PCM16.
- `speech/supervisor.py` reserves time for fallback when the primary times out. Reservation is bounded by 180 seconds and 30% of the total deadline; it provides an opportunity rather than a completion guarantee.
- `voice_autopilot.py` now launches commands inside the suspended-process containment mechanism. Conversion inference defaults to a 900-second process deadline and supports cancellation. Training retains its existing one-day command limit. The conversion wrapper rejects outputs that violate the stored audio contract.
- `speech/processes.py` closes the Windows job even if descendant inspection is denied, checks handle closure, and cleans process groups on the non-Windows fallback path. Windows is the tested platform.
- The installed Starlette 1.7.0 caused the Gradio page to fail with `TypeError: unhashable type: 'dict'`. Starlette 0.52.1 restores the template signature expected by Gradio 3.50.2. Runtime and lock files pin the tested combination; PyTorch/torchaudio remain 2.1.0+cu121. Health checks exercise actual HTTP page rendering, not just imports.

The compatibility explanation is supported by [Starlette 0.52.1's template implementation](https://github.com/Kludex/starlette/blob/0.52.1/starlette/templating.py) and [Gradio 3.50.2's dependency declarations](https://github.com/gradio-app/gradio/blob/gradio%403.50.2/requirements.txt). The failure and successful repair were reproduced locally.

## Verification evidence

- **37 tests passed**, including real CUDA allocation/reclamation after forced timeout; load/chunk/request timeout; cancellation; controller-death cleanup; conversion descendant cleanup; fallback reservation; failure-safe publication; isolated session state; PCM24 upload/playback; HTTP page/config rendering; existing audio, dataset, FFmpeg 320 kbps and so-vits regression checks.
- Full runtime health and `pip check` passed after the dependency repair. Existing upstream deprecation warnings and FAISS's optional AVX2 fallback do not fail these checks.
- A real Gradio client submitted a queued SAPI request through the local server and downloaded its result: **48 kHz, PCM24, mono, 1.803646 seconds** for “Testing the audio pipeline.” The temporary test server was closed afterwards.
- Real Bark/SAPI/Coqui speech qualification remains in `MILESTONE_1_REPORT.md` and `validation/milestone1-results.json`. This follow-up did not repeat that corpus or claim new recognition scores.
- Local logs: `logs/readiness-20261027/tests.txt`, `health.txt`, `ui-smoke.json`, and `recording.json`. No recordings or generated audio are tracked in Git.

## Read-only custom-recording assessment

`Anshumaan_RVC.wav` remains byte-for-byte unchanged, SHA256 `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`.

| Measurement | Result |
|---|---:|
| Duration | 1351.143 seconds (22:31) |
| Stored format | 48 kHz, PCM24, mono |
| Peak amplitude | 0.64354 |
| RMS amplitude | 0.03719 |
| Samples at or above 0.9999 magnitude | 0 |
| WebRTC VAD detected speech | 1020.88 seconds, approximately 75.6% |

The scan read bounded blocks and did not slice, denoise, normalize, upload or train on the recording. VAD does not establish transcript accuracy, speaker count, noise quality or dataset diversity. One continuous recording does not establish independent validation coverage.

## Remaining human review and future engineering

For this completed TTS milestone, listen to `logs/milestone1/ready-for-review.wav` (Bark) and `logs/milestone1/verified-fallback.wav` (Windows voice). Judge pronunciation, unwanted fillers, pacing and preferred voice. Bark can add words; SAPI is less expressive. No listening approval is assumed.

For a future RVC milestone, the next planning document must cover the RVC adapter, isolated dependency strategy, dataset segmentation/provenance, validation split, bounded 4 GB training experiments and voice-quality evaluation. A trained so-vits or RVC model cannot be inferred from an unprocessed WAV. The existing initial so-vits checkpoint remains unqualified. These are future engineering tasks, not work being handed back to the user to implement manually.

Following the user's updated workflow, subsequent work in the planning conversation is limited to Markdown plans for a separate execution model. No new RVC implementation or training was started while closing this task. No shutdown is scheduled or performed.

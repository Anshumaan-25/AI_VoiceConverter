# Repository audit and repair report

Historical audit from October 23. For the subsequent isolated TTS implementation, revised generation settings and current qualification results, see [MILESTONE_1_REPORT.md](MILESTONE_1_REPORT.md). Line references below describe the audit revision.

Date: 2026-10-23. Scope: root scripts and the local so-vits-svc checkout, dependency metadata, configurations, dataset headers, training logs, CPU regressions and selected GPU inference. The initial scan parsed 131 Python files. No Demucs/RVC pipeline.py or test_sweep.py exists here. Generated models/recordings were not replaced.

## 1. Critical bugs / execution risks

| Location (current file/line) | Root cause and impact | Repair / evidence |
|---|---|---|
| tts_engine.py:144,195,376 | Silence trimming could leave long tails; all-silent output could reach max() on an empty array; empty chunks silently omitted text. | Remove the fallback 300 ms trimming cap; gate each chunk; reject empty/non-finite audio and missing generated words. Regression: 1 s signal + 3 s silence now trims to about 1.01 s. This does not establish the cause of audible beeping. |
| voice_autopilot.py:603; audio_io.py:75 | Unvalidated speaker names could escape dataset_raw and feed recursive deletion. | Validate one safe Windows path component, resolve containment, reject reserved names, archive previous targets and prevent source/target overlap. |
| so-vits-svc/train.py:166 | Optional volume tensor stayed on CPU while model/tensors moved to GPU. | Move volume to the training device; inference casts it to model dtype. The default config does not use volume embedding, so this was conditional. |
| so-vits-svc/inference/infer_tool.py:280 | Input-resampler cache checked the encoder resampler's source rate, allowing a stale transform for a different input rate. | Check audio_resample_transform.orig_freq. Regression alternates 48k/44.1k/48k and maintains expected length. |
| so-vits-svc/inference/infer_tool.py:394,518; flask_api.py; flask_api_full_song.py | dict_keys indexed as a list; callers unpacked two values from a three-value inference return. | Use next(iter(...)); update legacy return unpacking. Optional HTTP/realtime modes still need end-to-end qualification. |
| so-vits-svc/resample.py:89 | Worker exceptions were ignored because completed futures were never read. | Call future.result(); malformed/silent recordings fail preprocessing rather than silently disappearing. |
| so-vits-svc/utils.py:159; inference/infer_tool.py:195 | Mismatched checkpoint tensors could retain randomly initialized parameters under a permissive loader. | Inference requests strict compatibility; training keeps explicit partial pretrain loading. Failed checkpoint resume with existing weights now raises instead of silently restarting. |
| dataset_tools.py:50; preprocess_flist_config.py | First two files always assigned to validation; one-file rebalance disguised an unusable dataset. | Recording-group split, nonempty split validation, 2–10 s bounds and minimum 20 training clips. No synthetic claim that this minimum ensures training quality. |
| gradio_app.py:11; voice_autopilot.py:695 | UI selected an initial checkpoint automatically. | Require a chosen checkpoint and reject G_0 through the supported wrapper. Renaming a checkpoint does not qualify it; actual training and listening validation remain required. |
| setup_autopilot.ps1; health_check.py:7 | Missing runtime packages and unpinned bootstrap made the advertised entry points fail. | Install pinned runtime, standardize venv310, resolve Python through launcher/argument, check native exit codes, imports, CUDA, FFmpeg and pip check. |

## 2. Audio quality / pipeline flaws

| Location | Root cause and impact | Repair / qualification |
|---|---|---|
| tts_engine.py:120,125 | Five-word splitting flattened phrasing and removed Unicode/punctuation. | Preserve sentence boundaries/Unicode; bound chunks by words and characters. This is a generation-control change, not proof of intelligibility or hallucination elimination. |
| audio_io.py:58; tts_engine.py:294; so-vits-svc/resample.py:32; inference_main.py | Writers used PCM16 and/or 44.1 kHz defaults. | Supported persisted audio uses explicit PCM24 at 48 kHz with atomic writes and headroom. Existing source files remain untouched. |
| so-vits-svc/utils.py:315; data_utils.py:47 | Loader divided all integer PCM by 32768; 24-bit data read as int32 would be amplified drastically, and float WAV would be attenuated. | Read normalized float32 with SoundFile; resample in memory to model rate. Tests confirm equal amplitude for PCM16, PCM24 and float WAV. See SciPy's documented left-justified 24-bit representation. |
| so-vits-svc/inference/infer_tool.py:389,485 | Zero padding produced [0:-0], deleting the entire signal; overlap counts mixed input/output sample rates; output padding never truncated overshoot. | Guard zero padding, express crossfade output lengths at target rate, enforce requested output length; PCM24 in chunk buffers. |
| export_mix.py:8 | No stem EQ/mix/export implementation existed in this checkout. | Add CPU exporter with +8 dB peak at ~5.657 kHz, one-octave bandwidth, amix normalize=0, limiter, 48 kHz libmp3lame 320k. FFprobe confirms actual stream bitrate. This is separate from the absent RVC pipeline. |
| analyze_baseline.py; compare_analysis.py | Hardcoded missing files, PCM16 assumptions and a verdict treating shorter duration as improvement. | Explicit paths, normalized format-aware decoding, finite-input checks, amplitude/spectral metrics; no automatic quality verdict. |

The supplied 15x band-energy decline does not mathematically imply an exact +8 dB correction: 10*log10(15) is approximately 11.76 dB. Relative band fractions also depend on other bands. The requested +8 dB is a tunable listening choice, not a proven inverse of the vocoder. No ruled-out Demucs/index-rate/chunk-seam hypothesis was re-tested.

## 3. VRAM / performance

| Location | Risk | Repair / evidence |
|---|---|---|
| tts_engine.py:312,469 | Default Bark preloads full models; failure/fallback could retain models; no 4 GB policy. | Set small-model and CPU-offload flags before import; bounded allocator budget; serialized job; finally clean Bark model cache. First-run pretrained downloads are required. |
| gpu_lock.py; voice_autopilot.py:630,654,667 | Multiple conversion/preprocess/train processes could concurrently exhaust the same GPU. | Windows cross-process lock around supported wrapper GPU jobs. |
| inference/infer_tool.py:137,389; configs/config.json | Unbounded voiced chunks and filename-selected precision. | Cap inference slices at five seconds; CUDA model FP16 with dtype derived after conversion; 85% PyTorch allocator fraction. This does not constrain unrelated GPU processes or all native allocations. |
| preprocess_hubert_f0.py:106,168 | CPU device request could be overridden; multiple workers duplicated models on one GPU. | Respect explicit CPU choice and require one CUDA worker. |
| data_utils.py:112; train.py:285 | max_speclen was ignored; evaluation retained generated GPU audio for every item. | Honor configured 256-frame crop; transfer logged waveforms to CPU; batch size 1 and mixed precision. Full training has not been qualified. |
| inference/infer_tool.py unload_model | Unloading removed generator but retained encoder/pitch/diffusion references. | Release owned model/resampler references then collect and release unused cache. |
| tts_engine.py:78 | Fixed two-thread CPU setting lacked measurement or override. | Bounded configurable thread count; no unsupported speedup claim. |

Synthetic GPU evidence: three two-second inference calls at input rates 48k, 44.1k, 48k returned finite outputs. Peak allocated memory was 1174.9, 1031.8 and 1032.2 MiB; reserved memory was 1424, 1432 and 1432 MiB. Unload returned allocated memory to 8.75 MiB. This tests a short workload and does not certify arbitrary song lengths or training.

## 4. Maintenance / secondary audit

- voice_autopilot.py:197: list-based subprocess arguments already handled spaces safely. Added cancellation/timeout child-tree cleanup. Generated inputs now use UUIDs and finally cleanup; outputs are matched to the exact request.
- so-vits-svc/inference/infer_tool.py:250: clamp zero index distances, cap neighbors to index size, reject empty/missing requested indices and out-of-range speaker IDs.
- dataset_tools.py:8: bounded clips and provenance; exact duplicate rejection. Fixed a secondary-review bug where a quiet recording's tail could overwrite the previous recording's accepted clip; regression verifies both provenance and waveform preservation. Distinct encodings/processed variants of the same take still require human provenance review; use independent takes for validation.
- download_pretrain.ps1: script-relative paths, independently preserved existing assets, staged downloads, size validation and scoped cleanup. Mocked downloads verified both one-weight-missing cases without replacing the other weight. Weight source immutability/signature verification remains a future reproducibility improvement.
- patches/so-vits-svc.patch and restore_svc.py: upstream directory is ignored by the parent Git repo. Save the complete local upstream delta against its pinned revision as a tracked patch, including pre-existing changes. No model/data/environment upload is required.
- requirements-runtime.txt and requirements-lock.txt: declared pins and a tested environment snapshot. Windows compilation dependencies can still matter on a fresh machine; current fairseq installation used an available compatible cached wheel.
- Historical PROJECT_STATE.md values conflicted with actual code. README and project state now document the supported entry points and known limitations.

## Corrections to the supplied baseline

1. warmup_epochs=0 was not currently fatal: the training epoch is clamped to at least 1. An explicit positive-warmup guard was added defensively.
2. The 300 ms fallback trim bound was confirmed, but librosa's preceding trim could already remove more. The claim that the entire function could never trim more than 300 ms was too broad.
3. Logs label 385.19 as a sum of losses, not discriminator loss; subsequent sums decreased to 235.16 and 195.58. This is insufficient evidence of GAN collapse.
4. DataLoader uses drop_last=False, so batch_size=6 with one item produces a smaller batch rather than six distinct samples. Dataset insufficiency remains real.
5. G_0 is saved after the first optimization step in this code. Its filename alone does not prove random initialization or white-noise output.
6. Missing empty_cache/gc calls alone do not prove a leak: live references must be released first. The selected content encoder already executes feature extraction with no_grad.

## Verification and remaining work

- Python syntax, 17 focused regression tests, real PCM files, real FFmpeg export, environment import/CUDA checks and package consistency verified. The expanded tests caught and fixed an extra one-sample drift in cross-rate overlap stitching.
- Gradio Blocks construction and HTTP /config response (200) verified. The upstream patch was applied and reverse-checked against a fresh local clone of the pinned revision.
- Initial checkpoint used only for low-level synthetic execution; no production-quality inference or converged training claim. A separate single optimization step with volume embedding enabled passed with finite parameters; peak allocated memory was 1402.2 MiB and reserved memory 1474 MiB. No checkpoint files were written by that test.
- The existing unsliced dataset is rejected, intentionally, until suitable recordings are prepared. Source recordings and checkpoints are preserved.
- **TTS remains a readiness blocker.** Bark real smoke test generated a PCM24/48 kHz file but failed quality qualification: the short phrase became 10.61 seconds, mostly a stable ~625 Hz tone. Added a speech-only guard rejecting at least three seconds of sustained narrow-band concentration. It rejects this actual sample; `auto` mode falls back to Coqui after releasing Bark. Tests cover carrier rejection, non-tonal/brief-tone acceptance and automatic fallback. The real CPU Coqui smoke also failed qualification: four words produced 85.68 seconds. A duration guard now rejects this decoder runaway, and both engines are subject to speech validation. Both recorded failure samples were run through the final guards and rejected. Root cause of the neural generation failures remains unproven; next qualification must isolate Bark semantic generation versus EnCodec decoding and inspect Coqui decoder/stop-token behavior against a known-good reference. These model-quality defects are not claimed cured; automatic fallback is not a guarantee of usable output. Listening, model qualification and representative recordings remain required.
- Demucs/RVC code, target RVC .pth/.index models and their 4 GB config remain absent. Their integration cannot be verified or repaired in this checkout.
- Diffusion, ONNX export, older bundled webUI/realtime APIs and multi-GPU training have not been qualified. Supported entry points are those in README.
- A static audit and bounded tests cannot establish 100% bug freedom. The repository is hardened and testable; dataset/model/audio quality qualification is still required.

References: [Suno Bark](https://github.com/suno-ai/bark) and [SciPy WAV data representation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html). Bark must be installed from the Suno source; the unrelated PyPI package named bark is not the engine used here.

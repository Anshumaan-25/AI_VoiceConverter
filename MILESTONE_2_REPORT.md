# Milestone 2 — RVC foundation and dataset preparation

Date: 2026-10-27. Branch: `bark-chunk-and-stitch`.

**Implementation and dataset preparation complete; real trained-model conversion qualification blocked.** No compatible trained RVC checkpoint/index was present. The existing `so-vits-svc/logs/44k/G_0.pth` and discriminator are not exported RVC models. No unrelated voice model was downloaded and no training was started to bypass this gate.

## Gate status

| Gate | Result and practical limit |
|---|---|
| A — isolated runtime | Passed. Separate Python 3.10 environment, pinned official source/patch/assets, imports and package consistency verified |
| B — adapter/worker | Implemented and structural/lifecycle tests passed. Real target-voice conversion, actual model precision and real conversion seams remain pending trained weights |
| C — application | Existing TTS and HTTP checks pass. RVC routing, validation and queued missing-checkpoint behavior pass; real RVC UI conversion remains pending |
| D — custom dataset | Passed for preparation/provenance/storage/split checks. Speaker consistency and acoustic suitability still need listening |
| E — qualification/delivery | 51 tests and both environment health checks passed. Model-dependent quality, three real conversion requests, conversion memory and source/output ASR comparison cannot run without weights |

## Delivered implementation

- `rvc_adapter.py`: typed model/index/pitch/speaker/deadline options, local file validation, input/output alias protection, checkpoint/index size limits, optional index at zero strength, isolated process execution, shared GPU locking, cancellation and atomic publication. No RVC/PyTorch model imports in the controller.
- `rvc_worker.py`, `rvc_runtime.py`, `rvc_audio.py`: strict exported-checkpoint metadata and state loading; v1/v2 feature checks; 32/40/48 kHz native rates; explicit fp16 CUDA profile (`x_pad=1`, `x_query=5`, `x_center=30`, `x_max=32`); allocator ceiling; streamed five-second core segments with context and sample-counted 100 ms joins. Native features/output stay in memory; only final 48 kHz PCM24 intermediate audio is persisted.
- Index validation rejects unreadable, empty, untrained, non-L2 and incompatible-dimensional indices. Selected index vectors are validated/cached once per worker. The local safety limits are 512 MiB for index files and 200000 reconstructed vectors; larger indices require an explicit engineering change. A requested index never silently becomes no-index inference.
- `patches/rvc-inference.patch`: removes upstream int16 output quantization, rejects invalid retrieval neighbors, handles exact-match zero distances, and uses the worker's validated index cache. No upstream automatic model/index search is used by the supported adapter.
- `app_service.py`, `gradio_app.py`: RVC/so-vits selector and separate controls, no trained model default, session cancellation and retained existing speech workflow. RVC input defaults to at most ten minutes per request; the original 22-minute recording is intended for dataset preparation.
- `restore_rvc.py`, `setup_rvc.ps1`, dependency locks and asset downloader: reproducible source, isolated packages, conservative disk estimate/reserve, hash-verified staged downloads and refusal to overwrite unexpected existing assets/source changes.
- `prepare_rvc_dataset.py`: bounded-memory VAD preparation, 2–10 second clips, conservative padding/quiet cuts, exact PCM duplicate checks, complete source intervals/settings/hashes, exclusion/review reports, staged dataset publication and verified rerun reuse. Original recordings are opened read-only.

## Pinned source, packages and assets

The current official main branch targets a newer runtime. The selected historical official commit is compatible with the existing Python/PyTorch generation:

- Repository: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), commit [`7ef19867780cf703841ebafb565a4e47d1ea86ff`](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/7ef19867780cf703841ebafb565a4e47d1ea86ff), MIT license.
- Isolated environment: Python 3.10.0, torch/torchaudio 2.1.0+cu121, NumPy 1.23.5, fairseq 0.12.2, FAISS CPU 1.7.4, librosa 0.10.0. Complete tested versions are in `requirements-rvc-lock.txt`; packaging tools are pinned by setup. This installation used an existing compatible cached fairseq Windows wheel. Fresh installations may require native build tools.
- The original TTS environment retains NumPy 1.22.0, torch/torchaudio 2.1.0+cu121 and its prior package stack. Its health check and `pip check` still pass.
- Asset repository: [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d), pinned revision `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`. The model card declares MIT. SHA256 values below were obtained from its Git LFS metadata and matched against the local bytes, rather than merely recording hashes after download.

| Asset | Bytes | Verified SHA256 |
|---|---:|---|
| HuBERT base | 189507909 | `f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96` |
| RMVPE | 181184272 | `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` |

Assets, environment, vendor checkout and custom audio stay ignored. Runtime restoration refuses a checkout that does not match the recorded source plus patch.

## Verification

**51 tests passed** with `VOICE_TEST_GPU=1`; full log: `logs/milestone2/all-tests.txt`.

Compact delivery evidence is tracked in `validation/milestone2-results.json`. All 196 real dataset clip headers and hashes were verified at delivery, along with the original source hash, pinned source patch and pretrained assets. No owned workers remained, the GPU read 0 MiB used, and approximately 23.0 GiB disk remained free at the final check.

The 37 existing regressions remain passing. Added tests cover controller import isolation, invalid options, checkpoint metadata/feature mismatch, missing/corrupt/empty/incompatible indices, exact retrieval matches, float precision, no-index routing, sample-accurate segmentation across native rates, hard-link input aliases, failure-safe destinations, real isolated corrupt-checkpoint rejection, isolated CUDA timeout cleanup, dataset split leakage, repeatability, altered artifacts, silence/short/nonfinite sources and interrupted builds. Retrieval/network fixtures are structural tests, not a synthesized target voice.

Both `health_check.py` and `venv-rvc/Scripts/python.exe rvc_runtime.py --health` passed, including `pip check`. Upstream optional tensorboardX/AVX2 and packaging deprecation messages are informational for this inference-only path.

A live queued Gradio request through localhost confirmed the RVC missing-checkpoint error, followed by successful SAPI generation and PCM24/48 kHz mono playback/download. The temporary test server was closed. Evidence: `logs/milestone2/ui-smoke.json`.

### Real asset GPU probe

`qualify_rvc_runtime.py` loaded the actual pinned HuBERT and RMVPE weights in the isolated contained process and processed the existing SAPI speech sample. It produced finite 768-wide content features and 181 pitch frames (136 voiced). Peak PyTorch allocation was **545112576 bytes (519.86 MiB)**; reserved memory was **572522496 bytes (546 MiB)**. Peak sampled process-tree system RAM was approximately 2.05 GiB, distinct from VRAM. Probe duration was approximately 9.58 seconds including startup/loading. Evidence: `logs/milestone2/asset-probe/`.

The separate isolated-runtime lifecycle regression allocated over 100 MiB on CUDA, forced termination and checked memory returned within 64 MiB of baseline. The original lifecycle test also passed. These demonstrate asset compatibility and process recovery; they do not measure the absent voice synthesizer's memory requirements.

## Prepared custom recording

Original: `Anshumaan_RVC.wav`. SHA256 remains `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`.

Dataset: `datasets/rvc/anshumaan/0dddb82392f1860af8a3/`.

| Split | Clips | Stored duration |
|---|---:|---:|
| Train | 179 | 947.260 seconds |
| Provisional validation | 17 | 97.0033125 seconds |
| Total | 196 | 1044.2633125 seconds |

All clips are mono PCM24/48 kHz; lengths range from 2.02 to 9.98 seconds. Retained duration includes boundary context and is not the same as pure VAD speech duration. No denoising, EQ, dynamic compression or per-clip normalization was applied.

Exact source split ranges at 48 kHz: training `[0, 57888960)`, validation `[58848960, 64854879)`. The 960000-sample gap is 20 seconds, ten on each side of the chosen boundary. Clips and context remain inside their assigned ranges. No exact PCM duplicate crosses splits. This is a **same-session temporal holdout**, not independent recording-session validation.

The local `manifest.json` contains source/clip hashes, intervals, transforms, VAD settings, counts and 395 excluded regions. `review.json` lists ten suggested/flagged clips. The default 60-second usable-duration minimum is a local guard, not a scientific sufficiency threshold. A rerun verified hashes, ranges and unchanged source content and reused the completed dataset. Evidence summary: `logs/milestone2/dataset-summary.json`.

### Human listening, left until the end

Review speaker consistency, background noise, clipped syllables and natural boundaries. A quick representative selection from the original recording timeline:

| Source time | Split | Derived file under the dataset directory |
|---|---|---|
| 00:00.78 | train | `train/c3cae9cd4b08709a_0000037440_0000191040.wav` |
| 05:01.74 | train | `train/c3cae9cd4b08709a_0014483520_0014606400.wav` |
| 10:18.52 | train | `train/c3cae9cd4b08709a_0029688960_0029823360.wav` |
| 16:32.86 | train | `train/c3cae9cd4b08709a_0047657280_0048057600.wav` |
| 22:27.70 | val | `val/c3cae9cd4b08709a_0064689600_0064854879.wav` |

Listening results have not been assumed. Quiet/noisy flags are review prompts, not proof of a defect. An independent recording session will be preferable for final voice-quality qualification if available.

## Remaining engineering and delivery boundary

Milestone 3 must define and run a bounded training pilot, export a compatible RVC inference checkpoint, optionally build a matching retrieval index, and finish short/multi-segment/repeated real conversion, actual synthesis-memory, ASR-retention and listening checks. Those are engineering tasks, not manual implementation being handed to the user. No full training, main-branch merge or shutdown is part of this delivery.

The implementation plan and this report are included with the code on `bark-chunk-and-stitch`. Commit/push outcome is verified in the task's final delivery; no claim of real custom-voice readiness should be inferred from a successful push.

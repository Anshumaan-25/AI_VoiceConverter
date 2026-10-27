# Milestone 3 — Bounded training implementation and numerical safety stop

Date: 2026-10-27. Branch: `bark-chunk-and-stitch`.

**Safely bounded partial delivery.** The new training path passed smoke and resume checks, then stopped on a nonfinite discriminator loss during the pilot. It reached 239 successful paired updates. The last complete resume checkpoint is step 200. No recovery training, real conversion qualification or custom-voice quality claim followed the failed gate.

## Gate results

| Gate | Result |
|---|---|
| A — preflight/assets | Passed: unchanged data, pinned source and assets, compatible G/D state dictionaries, both environment health checks, disk reserve |
| B — training inputs | Passed: all 196 clips represented, split-isolated finite features/pitch, correct float PCM24 scaling, explicit frame alignment, no clip exclusions |
| C — bounded pilot | Smoke and real resume passed. Pilot **failed safely** on a nonfinite discriminator loss; checkpoint preservation and process/VRAM recovery passed |
| D — real conversion | **Blocked by Gate C.** No real target-voice conversion, full inference-memory measurement, index delivery, ASR comparison or real-model UI qualification was attempted |
| Delivery | Code, tests, compact evidence and this report delivered as a partial result. Exact commit/push outcome is verified in the final delivery message |

## Implementation

- `prepare_rvc_training.py`: sequential contained HuBERT and RMVPE extraction; reads immutable 48 kHz PCM24 audio as scaled float; performs 16 kHz resampling only in memory. Final content/pitch/coarse-pitch arrays retain clip/source hashes and split identity. Reuse verifies dataset, code, assets and every final array hash.
- `rvc_training.py`: source/clip/split verification, explicit frame tolerances, path containment, free-space guards, durable budget reservations and atomic complete checkpoint publication. A single serialized resume state contains the matched G/D state, both optimizers/schedulers, AMP scaler, CPU/CUDA/NumPy/Python RNG state and exact sample-order offset. The complete-set manifest and latest pointer are published after the bytes are complete; two complete sets are retained.
- `train_rvc.py`: controller without model imports, shared GPU lock, Windows process-tree containment, progress and whole-worker deadlines, graceful-stop signal before the hard deadline, sampled device memory and persistent aggregate budgets. Parent termination leaves a pessimistic reservation requiring inspection. A recorded unresolved safety failure blocks subsequent training. One reduced-crop retry is supported only for an initial smoke OOM; none was needed here.
- `rvc_training_worker.py`: single-GPU, batch-one AMP loop reusing the exact pinned RVC generator, discriminator, spectral functions and GAN losses. It tracks actual G/D optimizer execution and paired successful steps, skipped AMP updates, unclamped losses and sample position. Nonfinite loss/model state aborts publication. The learning rate is 0.0001, original 17280-sample decoder crop, no GPU dataset cache and no data-loader child workers.
- `qualify_rvc_model.py`: later qualification entry point with a bounded training-only L2 index, known-text conversion pairs, ASR retention, holdout diagnostic, cancellation/timeout/recovery and queued UI checks. Full model execution is **unverified and intentionally blocked** by the recorded training failure. Its index split logic and failure-gate behavior are covered by fixtures; these do not establish real conversion quality.
- `rvc_manifest.json`, downloader and setup: optional immutable v2 48 kHz pretrained G/D assets. `setup_rvc.ps1 -TrainingAssets` opts into these downloads. The inference source patch was unchanged; clean disposable restoration was tested. No source reset or additional upstream patch was needed.

The supported loop does not invoke `infer/modules/train/train.py` or its data loader/exporter. Inspection found raw integer WAV scaling in `infer/lib/train/utils.py::load_wav_to_torch`, adjacent cache writes, fixed truncation, automatic resume exception fallback, DDP child-result issues and nonstandard exit behavior. The new path avoids those entry points and uses explicit float loading, frame counts, controlled serialization and propagated worker status. TensorBoard/matplotlib are absent and unnecessary for this JSON-logging loop; no packages or original TTS dependency pins were changed.

## Provenance and resources

Official source remains `7ef19867780cf703841ebafb565a4e47d1ea86ff`, with `patches/rvc-inference.patch` unchanged. Pretrained assets use [the pinned asset revision](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/pretrained_v2), covered by the asset repository's declared MIT license. Sizes and SHA256 values came from its Git LFS metadata and matched the local bytes.

| Asset | Bytes | SHA256 |
|---|---:|---|
| `pretrained_v2/f0G48k.pth` | 75465569 | `b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4` |
| `pretrained_v2/f0D48k.pth` | 142875703 | `2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748` |

Both pretrained networks loaded with all keys matching. Architecture: RVC v2, 768 content dimensions, pitch-conditioned, 48 kHz native synthesis. The compatible pretrained speaker table has 109 rows; only target speaker ID **0** was used in this pilot. Other rows are not trained target voices.

Original `Anshumaan_RVC.wav` SHA256 remains `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`. Dataset version `0dddb82392f1860af8a3` remains unchanged, with 179 training and 17 provisional validation clips and the original 20-second temporal gap.

Workspace: `logs/milestone3/training/`. All 196 clips have finite aligned arrays. HuBERT's 50 Hz features are repeated to the 100 Hz training grid. Up to four frames of endpoint disagreement are allowed explicitly; observed exclusions were 960 or 1119 tail samples, approximately 20–23.3 ms. No sample was omitted by a bucket sampler; every clip remains eligible. No resampled WAV, spectrogram cache beside immutable audio, denoising or gain normalization was written.

Content extraction took 15.593 seconds and pitch extraction 19.032 seconds. Their peak allocated VRAM was 387210752 and 359113728 bytes, respectively, with separate processes exiting before training.

| Training stage | Attempts | Successful paired updates | Worker wall time | Sampled peak device use | GPU after exit |
|---|---:|---:|---:|---:|---:|
| Smoke | 10 | 8 | 19.125 s | 2987 MiB | 0 MiB |
| Resume from step 8 | 1 | 1 | 16.453 s | 3093 MiB | 0 MiB |
| Pilot from step 9 | 234 | 230 | 135.203 s | 3081 MiB | 0 MiB |
| Total | 245 | 239 | 170.781 s | 3093 MiB maximum | 0 MiB |

Five completed attempts did not produce paired updates because AMP skipped an optimizer update; the final attempt failed before its discriminator update. The 500-attempt budget is conservative relative to a 500-successful-step ceiling. The 1800-second allowance was not exhausted and is **not permission to continue after the failed gate**.

Peak PyTorch allocation across training was 2848023552 bytes (about 2.65 GiB); peak reservation was 3107979264 bytes (2964 MiB). Sampled device use includes allocations outside the PyTorch allocator and is sampled, not a continuous hardware maximum. The allocator ceiling remained the existing 3.4 GiB policy. There was no OOM and no crop reduction. Startup free disk was about 25.34 GiB; the 10 GiB reserve remained satisfied. Each complete resume state is 1309418432 bytes, and exactly two complete sets remain. Final resource measurements are recorded in the compact evidence.

## Failure analysis and retained artifacts

At pilot-local attempt 234, global successful step 239, the discriminator loss became nonfinite before its update. The prior five discriminator losses were approximately 3.79, 3.86, 3.33, 3.73 and 3.70; these do not establish a steadily diverging GAN. The exact first nonfinite intermediate was not instrumented in this run. A generator output, latent computation, discriminator autocast activation or evolving state could be involved; assigning a particular cause would be speculation.

The triggering source interval is 15:47.58–15:54.66, derived clip `train/c3cae9cd4b08709a_0045483840_0045823680.wav`. Its audio is finite, peak 0.4975134, RMS 0.0602546. Its aligned features are finite, shape `[706, 768]`, range approximately -4.414 to 3.563; pitch is finite, 0–302.217 Hz. It was not silently removed or blamed as defective audio.

Retained complete resume sets are steps **100 and 200**. Step 200:

- `logs/milestone3/training/checkpoint-000200-7459275e/state.pth`
- SHA256 `964b2622f5f9207fb1bbcf7d7e6cddac8ed8f5ed1187cefd6d77517103320e2d`
- Both model states and both optimizer states reloaded on CPU and checked finite. This is not a claim that the unsaved step-239 state was finite or that step 200 has passed post-failure GPU resume.

The older inference export remains `logs/milestone3/training/pilot-v2-48k.pth`, **step 9**, SHA256 `7bde330ed867316688caa05459b3aba979c86c48cb9d6fd1cd653b6387bfe6f2`. It was created by the passing resume smoke and is labeled an unqualified engineering pilot. Step 200 was not promoted to an inference export after the failure.

Local detailed evidence: `failure-analysis.json`, `guard-results.json`, each worker's request/events/error/process files and `budget.json` under `logs/milestone3/`. A normal resume command and qualification command were tested after the stop: both rejected before launching a model worker and left the budget unchanged. No counter reset, alternative workspace, CPU training fallback or hidden recovery run occurred.

## Verification and practical limits

**64 tests passed** in the final suite with `VOICE_TEST_GPU=1`; result and hashes are in `validation/milestone3-results.json`, with verbose output in `logs/milestone3/all-tests.txt`. Coverage includes both existing real CUDA cleanup regressions and new checks for frame mismatch, exact float PCM24 amplitude, stale features, split/index isolation, path escapes, budget exhaustion/reservations, partial/corrupt checkpoints, checkpoint retention, worker failure, clean source patch restoration and persistent failure gates. Both environment health checks and package consistency passed. Python AST, PowerShell syntax and Git whitespace checks passed.

The smoke checkpoint resumed from step 8 to step 9 with saved optimizer/scaler/RNG and sample offset. Sample order is deterministically reconstructed from the epoch seed and offset; bitwise CUDA reproducibility is not claimed.

No real target-voice conversion, three-request synthesis stability check, matching delivered index, model-loaded conversion cancellation test, converted ASR comparison or real-model queued UI check passed in this delivery. Existing UI/TTS regressions continue to pass, but they cannot close those model-dependent gates. Source and converted listening pairs therefore do not exist yet.

Commands executed, in order:

```powershell
./venv310/Scripts/python.exe -X utf8 -B health_check.py
./venv-rvc/Scripts/python.exe -X utf8 -B rvc_runtime.py --health
./venv310/Scripts/python.exe -X utf8 -B download_rvc_assets.py --training
./venv310/Scripts/python.exe -X utf8 -B prepare_rvc_training.py
./venv310/Scripts/python.exe -X utf8 -B train_rvc.py --stage smoke
./venv310/Scripts/python.exe -X utf8 -B train_rvc.py --stage resume
./venv310/Scripts/python.exe -X utf8 -B train_rvc.py --stage pilot
# Pilot stopped on a nonfinite loss. No further training was run.
$env:VOICE_TEST_GPU='1'
./venv310/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -v
```

Both health checks were also repeated after the safety stop. Training assets were separately checked against the actual network classes on CPU. The existing cached environment was used; this does not establish a fresh machine's native fairseq build compatibility.

## Next work and human handoff

The next engineering task is a **new, narrowly bounded numerical investigation**: instrument generator latent/output and per-discriminator activations, losses and unscaled gradients; locate the first nonfinite value; compare precision at that specific operation; then propose and test the smallest stability change. Any recovery experiment needs an explicit new budget and must preserve this failed run, source/config identities and finite checkpoint. Do not blindly lower precision, change architecture, remove the clip or restart a 500-step run. Full voice conversion qualification follows a passing stability gate.

Human dataset listening from `MILESTONE_2_REPORT.md` remains useful: speaker consistency, background noise and boundary cuts. No listening approval has been assumed, and lack of listening approval is not the cause of this numerical stop. A separate recording session will improve eventual final evaluation. Debugging and implementation remain engineering tasks.

Models, recordings, features, logs and environments remain ignored. No main merge, extended training or shutdown forms part of this delivery.

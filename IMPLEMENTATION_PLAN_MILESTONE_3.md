# Milestone 3 — Bounded RVC training pilot and real conversion qualification

Status: executed 2026-10-27 with a safely bounded partial result. Gate C stopped on a nonfinite discriminator loss; Gate D remains blocked. See `MILESTONE_3_REPORT.md`. No further training is authorized by this recorded result.
Date: 2026-10-27. Baseline branch: `bark-chunk-and-stitch`, reported delivery `a3f2bd0`.

## Objective and honest completion boundary

Build a reproducible, resumable training path for the prepared custom recording, run a strictly bounded pilot on the 4 GB GPU, export a compatible inference checkpoint, and complete the model-dependent checks left open by Milestone 2. A successful pilot proves training/inference integration and resource safety; 500 steps do not establish a finished voice model or production-quality speech.

Read `MILESTONE_2_REPORT.md`, `PROJECT_STATE.md`, the existing RVC modules and relevant tests first. Recheck actual working-tree state instead of assuming the documented baseline is unchanged. Preserve unrelated work.

## Scope and invariants

- Preserve the original `Anshumaan_RVC.wav`, prepared dataset version `0dddb82392f1860af8a3`, existing TTS environment, so-vits implementation and verified application behavior.
- Use isolated `venv-rvc`, official source commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`, recorded patches and verified assets. Do not upgrade the source or PyTorch stack as a shortcut.
- All persisted audio remains mono/stereo 48 kHz PCM24 WAV. Compute 16 kHz encoder inputs in memory. Feature/pitch tensors may be cached as arrays. No MP3 in preparation, training or intermediate conversion.
- Start with RVC v2, pitch conditioning, 768-wide content features and native 48 kHz synthesis. Verify compatible pretrained generator/discriminator availability before committing to the run. No random-initialization substitution if pretrained assets are unavailable.
- No long training, automatic repeated experiments, cloud uploads, full dataset redesign, new UI training dashboard, main-branch merge or shutdown. Do not change the default speech engine.
- Human dataset listening remains unconfirmed. Permit a clearly labeled engineering pilot on the existing provisional dataset; never imply that speaker consistency or final voice quality has passed human review.

## Gate A — Preflight and reproducible training assets

1. Record Git status, Python/package health, GPU name/free memory and free disk. Confirm original SHA256 `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`, dataset manifest/clip hashes, 179 train and 17 validation clips, and split boundaries. Never regenerate the dataset silently.
2. Inspect the pinned trainer, data loader, losses, checkpoint exporter and extraction code. Resolve the exact CLI/API from these local files, not a tutorial. Existing inference-only health is not proof of training dependency completeness.
3. Add only required training packages to isolated requirements/lock files, preserving the original environment. Check TensorBoard and Windows native dependencies explicitly.
4. Resolve a compatible official v2 48 kHz pitch-conditioned pretrained G/D pair from authoritative source/asset metadata. Record immutable revision, exact names, licenses, bytes and authoritative SHA256s; verify download bytes and state/config compatibility. Do not invent filenames/hashes or download unrelated target voices. Extend the bounded asset downloader or a training-specific equivalent.
5. Reserve at least 10 GiB free after estimated downloads, features, optimizer checkpoints and temporary writes. Measure actual checkpoint size after the first save and update the estimate. Refuse insufficient disk rather than deleting user artifacts.

Acceptance: assets reproducibly identified and validated, both environment checks pass, original data unchanged, resource preflight passes. Missing assets or incompatible dependencies are a reported blocker, not grounds to bypass checks.

## Gate B — Correct training inputs and split isolation

Create a manifest-driven training workspace separate from immutable dataset clips. Cache spectrograms/features there; upstream loaders write `.spec.pt` beside audio, so never point them directly at immutable dataset files without redirecting their cache behavior.

- Extract HuBERT layer/features and RMVPE pitch in sequential contained GPU phases. Release each extraction process before training; do not retain encoder/pitch models alongside G/D and optimizer states.
- Use only training clips for optimizer batches and retrieval-index construction. Validation features may be generated separately for evaluation, never mixed into training lists or index vectors.
- Store provenance for each derived tensor: source clip SHA, split, extractor asset hashes, code/config hash, feature width, sample/frame counts and finite-value checks. Reuse caches only when these match.
- Verify actual audio normalization in `infer/lib/train/utils.py::load_wav_to_torch` and loader behavior with a known PCM24 fixture. Avoid integer-scale mistakes, hidden normalization, frame-count estimates based on WAV byte sizes, or silent truncation of materially misaligned pitch/features/audio.
- Check all clips survive intended sampler/bucket rules; report exact exclusions instead of silently shrinking the dataset. Validate feature/pitch frame alignment against the selected hop sizes with an explicit rounding tolerance.
- Preserve the 20-second temporal gap and label validation as same-session provisional holdout. Do not claim independent-session generalization.

Acceptance: reproducible cached inputs; no train/validation leakage; finite correctly scaled audio/features; nonempty loader; all exclusions accounted for. Add focused tests for cache invalidation, scaling, frame alignment and split isolation.

## Gate C — Bounded trainer, checkpoint safety and resource control

Prefer a thin supported entry point plus a minimal tracked upstream patch over copying the entire trainer. Update source restoration/verification so it recognizes the exact complete approved patch set. Test restoration against a clean disposable checkout; never reset the active checkout to test it.

Observed planning-time hazards requiring inspection and repair:

- `infer/modules/train/train.py`: four workers, persistent workers and prefetch factor eight; broad exception around automatic resume; global step inferred from epoch; spawned child exit codes not checked; nonstandard `os._exit(2333333)` completion.
- `infer/lib/train/data_utils.py`: file-size-derived bucket lengths, fixed feature truncation, adjacent spectrogram caches.
- `infer/lib/train/process_ckpt.py`: export returns error text rather than necessarily raising, and writes to a fixed relative directory.

Required behavior:

1. Single GPU, batch size 1, mixed precision with gradient scaling, no GPU dataset caching, initial data-loader workers 0. Omit persistent-worker/prefetch arguments when workers are zero. Preserve compatible architecture/pretrained shapes. Begin with the pinned 48 kHz segment size 17280 samples; reduce training crop once only if measured OOM requires it and compatibility remains valid.
2. Use shared GPU lock and existing Windows process-tree containment for extraction, training and conversion. Apply an allocator cap no greater than the existing 3.4 GiB policy; record device-wide memory separately because the allocator cap does not cover every CUDA allocation.
3. Parent enforces deadlines independently of training callbacks. Track progress heartbeat, finite losses, successful optimizer updates and AMP skipped updates. Abort on nonfinite model/loss state or repeated inability to update; do not mistake finite GAN losses for audible quality.
4. Smoke stage: at most 10 optimizer steps or 5 minutes of training-worker wall time. Pilot stage: aggregate at most 500 steps or 30 minutes across smoke, pilot and resume verification; persist consumed budget. No automatic budget reset on restart. Allow at most one reduced-crop smoke retry for OOM within those same limits; then report blocked.
5. Save matched G/D, optimizers, schedulers, scaler, exact global step/epoch, RNG states, sampler progress where supported, asset hashes and dataset/config identity. Publish a checkpoint-set manifest atomically only after all files are complete. Corrupt/incompatible resume must fail explicitly, not fall back to pretrained initialization.
6. Save every 100 steps and at normal bounded completion. Reserve a short graceful-stop period before parent hard termination; never extend the 30-minute aggregate worker cap. Preserve the last complete checkpoint after cancellation or forced termination. Retain at most two complete resume sets plus selected export; only prune owned artifacts after replacement verification.
7. Test resume for at least one further optimizer step, counted within the pilot budget. State whether sample order is exactly resumed or only epoch-level; do not assert bitwise CUDA reproducibility without evidence.

Acceptance: successful finite updates, measured peak allocation/reservation and device-wide VRAM, bounded time, clean process exit, checkpoint reload/resume verified, no owned descendant left behind. If 48 kHz training cannot fit, stop with measured evidence and propose a separate alternative plan; do not silently violate the persisted audio contract or switch hardware/model architecture.

## Gate D — Export, index and real RVC qualification

Export a nonzero-step pilot checkpoint through an atomic path, recording the parent training checkpoint/hash and actual step. Validate strict loading through the existing adapter. Keep the label **pilot / not voice-quality qualified**.

Build an optional L2 retrieval index from training features only, preserving the adapter's dimensional and size limits (512 MiB, 200000 vectors). A bounded flat L2 index is acceptable for this pilot; if deterministic subsampling is necessary, record its seed and selected provenance. Do not weaken limits or include validation clips. Test index strength zero separately from a documented nonzero setting.

Use real model requests to close pending Milestone 2 checks:

- Short existing verified SAPI speech, a multi-segment sentence with known transcript, and a longer sample crossing multiple five-second joins; repeat at least three sequential requests.
- For every request record model/index hashes, settings, input/output duration, 48 kHz PCM24 format, finite samples, peak/RMS, elapsed time, CUDA allocated/reserved peaks, sampled device memory and post-exit recovery. Require no growing residual GPU use; compare with a quiet baseline and allow the existing 64 MiB recovery tolerance.
- Test real cancellation/timeout in a model-loaded worker, preserving any existing output and releasing the GPU lock. Confirm a subsequent request succeeds.
- Compare source and converted ASR using the same available recognizer and known text. Report source WER, converted WER and change; investigate new omissions/repetitions. ASR alone does not prove voice similarity. Do not keep retraining automatically to improve a metric.
- Inspect join windows for discontinuities and provide listening samples. Run a live queued UI conversion with the pilot model and verify playback/download plus an unchanged SAPI request. Close the temporary server.

Acceptance is split: engineering integration may pass while acoustic quality remains provisional or fails. Document each separately. Poor pilot audio is evidence for the next training decision, not permission for an unlimited run.

## Files and verification strategy

Suggested new modules: `prepare_rvc_training.py`, `train_rvc.py` (controller), `rvc_training_worker.py`, `qualify_rvc_model.py`, focused training tests and a training configuration template. Reuse existing containment, GPU lock, audio I/O, trace and adapter abstractions. Exact names may change if the existing architecture provides a cleaner fit.

Likely existing changes: training requirements/lock, `setup_rvc.ps1`, asset manifest/downloader, source restoration/patch verification, tracked upstream patch, README and project state. Avoid unrelated refactors and TTS retuning.

Run focused tests after each gate; once integration is complete run the full existing suite with CUDA lifecycle tests enabled and both environment health checks. Include negative tests for corrupt resume, partial checkpoint sets, exhausted budgets, missing assets, stale features and worker failure. Structural fixtures must be distinguished from real training/conversion evidence. Do not rerun expensive speech benchmarks without a relevant regression.

## Benefits, costs and stop conditions

The pilot replaces assumptions about trainability with measured resource and model evidence. Batch size 1 and isolated phases reduce memory pressure but slow throughput. Native 48 kHz honors storage requirements but may not fit the GPU; this is a gate, not a guarantee. The short budget may produce weak voice similarity, and the temporal holdout cannot establish robust generalization.

Stop dependent work on unresolved data/hash changes, corrupt caches/checkpoints, insufficient disk, unsupported pretrained shapes, persistent OOM, nonfinite updates, containment failure or budget exhaustion. Finish independent tests/documentation and report the exact blocker. Do not hide a failure with CPU fallback, unrelated weights or an untrained export.

## Delivery and human handoff

Create `MILESTONE_3_REPORT.md` and compact `validation/milestone3-results.json`: exact commands, source/assets/data identity, test results, steps/time consumed, memory/disk measurements, resume evidence, checkpoint/index paths and hashes, conversion metrics, failures and gates remaining. Keep recordings, features, checkpoints, indices, environments and verbose logs ignored. Update `PROJECT_STATE.md` and README without calling the voice production-ready.

Provide source/converted listening pairs and the existing representative dataset clips. Human tasks are listening for intended speaker, noise, missing syllables, naturalness, similarity and seams; an independent recording session is desirable for final qualification. Do not fabricate listening approval or make the user perform implementation/debugging.

When this plan is explicitly executed, commit only scoped code/docs and push `bark-chunk-and-stitch` after gates pass or a safely bounded, clearly documented partial result is ready. Use the user's existing Git identity, no assistant attribution, no forced push, no main merge. Verify remote commit and report remaining local changes. Never schedule or execute shutdown.

## Execution prompt

> Execute IMPLEMENTATION_PLAN_MILESTONE_3.md. Implement the bounded RVC training path, run its gated pilot within the aggregate step/time limits, and qualify real conversion where feasible. Preserve original recordings, immutable datasets and the TTS baseline. Make routine implementation decisions autonomously; stop dependent work on explicit gate failures and report evidence instead of bypassing limits. Leave human listening judgments until the final handoff. Deliver the report, validation evidence and verified Git outcome. Do not run extended training or shut down the computer.

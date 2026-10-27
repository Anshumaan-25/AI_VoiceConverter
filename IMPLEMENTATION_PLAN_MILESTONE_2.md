# Milestone 2 execution plan: RVC foundation and recording preparation

Prepared: 2026-10-27. Intended executor: GPT-6.1 sol.
Status: **executed 2026-10-27; implementation/dataset complete, real trained-model qualification blocked**. See `MILESTONE_2_REPORT.md` for gate results and remaining work.
Workspace: `C:\Users\anshu\Documents\PROJECTS\AI_VoiceConverter`.
Verified starting commit: `d8d65b7fed74ede640597e820c60e1992d0a8468`, branch `bark-chunk-and-stitch`.

## 1. Objective and boundaries

Add RVC as a separate voice-conversion backend, preserve the working speech application, and prepare reproducible derived clips from the custom recording. Deliver a tested implementation and an honest readiness report. Full acoustic training and final voice-quality approval belong to Milestone 3.

Implement this plan autonomously when the user hands it to the execution model. Make routine implementation choices without repeatedly asking permission. If an external asset is missing, finish independent work and report that specific blocker; do not call a mock conversion a successful real-model test.

Do not replace the existing so-vits backend, redesign the UI, retune Bark/Coqui, add Demucs/song processing, upgrade the existing ML environment, run extended training, or shut down the computer. Do not load an unrelated or random checkpoint to imply the custom voice is ready.

## 2. Existing evidence and non-negotiable constraints

Read `PROJECT_STATE.md`, `INTEGRATION_REPORT.md`, and the relevant parts of `MILESTONE_1_REPORT.md` first. Consult the historical audit only for the subsystem being changed.

- Windows, Python 3.10, RTX 3050 Laptop, physical 4096 MiB VRAM. The existing `venv310` uses torch/torchaudio 2.1.0+cu121. Preserve it.
- Existing baseline: 37 tests passed, real worker CUDA memory reclaimed, local queued SAPI output confirmed, HTTP rendering repaired. Run checks appropriate to new changes; do not repeat the entire old speech benchmark without a regression reason.
- Persist all intermediate audio as **48 kHz PCM24 WAV**. Model-native sample rates and feature buffers may differ in memory. Final MP3 alone uses explicit 320 kbps. A tensor/feature cache is not an audio export.
- Keep the controller free of RVC/PyTorch model imports. Reuse `gpu_job()` and Windows process-tree containment; TTS and RVC GPU inference must not overlap.
- Preserve `Anshumaan_RVC.wav` byte-for-byte. Expected SHA256: `c3cae9cd4b08709aa50ea1641519a4b1c9788264d1db7654110377fabd20035e`.
- Recording baseline: 1351.143 seconds (22:31), 48 kHz PCM24 mono, no near-fullscale samples, about 1020.88 seconds detected as speech. These measurements do not establish speaker consistency or training quality.
- No audio, checkpoints, indices, environments or downloaded repositories in Git. No recording upload or external speech service.
- Preserve existing user edits and Git identity. Never add an assistant author or co-author. No force push or merge into `main`.

## 3. Execution order and gates

### A. Confirm the baseline and isolate dependencies

1. Inspect branch/status, applicable repository instructions, disk space, CUDA availability, existing model assets and the recording hash. Do not infer model availability from filenames alone. Record what actually exists.
2. Inspect the official RVC source and select a concrete compatible commit. Pin its full SHA, license and required asset sources in a manifest. Inspect that revision's inference entry points, checkpoint metadata, feature dimensionality, pitch extraction and dependency requirements before choosing imports or commands.
3. Default to a **separate `venv-rvc` environment and ignored `vendor/rvc` checkout**. Add a reproducible setup/restore command and a separate dependency lock. Do not install RVC requirements into `venv310`, use `--system-site-packages`, or let an upstream installer alter the tested TTS stack.
4. Prefer Python 3.10 and torch 2.1.0+cu121 if the selected revision supports them. If it does not, choose a documented compatible revision or isolate the required runtime entirely. Do not invent a version combination or silently change the host driver.
5. Estimate environment/model/cache disk use before downloading. Maintain at least 10 GiB free as a local operating reserve, and report if the estimated footprint cannot fit. Reuse verified downloads where practical; do not delete user files to free space.
6. Store required model assets with URL, license/source information, size and SHA256. Compare against an authoritative digest when available; otherwise distinguish a locally recorded digest from verified authenticity. Use staged downloads and atomic publication. Never auto-load arbitrary third-party `.pth` files: pickle-based checkpoints execute code.

**Gate A:** the isolated runtime imports successfully, its package consistency check passes, source/assets are reproducible, and the original environment remains unchanged. Missing trained voice weights do not prevent implementing the adapter and dataset tools.

### B. Implement the conversion adapter and bounded worker

Suggested files: `rvc_adapter.py`, `rvc_worker.py`, `rvc_runtime.py` or a small `rvc_backend/` package; `requirements-rvc.txt`, a tested RVC lock, a setup/restore script and a runtime manifest. Choose one consistent structure, not both.

1. Implement `RvcConverter.convert(SpeechResult, output: Path) -> Path`, matching the existing protocol in `speech/contracts.py`. Keep `SvcConverter` working. Introduce a shared process helper only if needed; avoid a large rewrite of the proven speech supervisor.
2. Add typed options for model path, optional index path, pitch shift, pitch extractor, index strength, device and deadlines. Reject nonfinite values and out-of-range controls before spawning a worker. Explicitly select a speaker ID only when the checkpoint supports multiple speakers.
3. Resolve paths without shell interpolation. Support spaces, Unicode and backslashes. Reject input/output aliasing, including same-file aliases. Validate source format, decoded finite samples, required assets, checkpoint version, native sample rate and feature width. Checkpoint inspection belongs in the isolated worker, not the UI process.
4. Support no-index inference with index strength zero. If the user requests retrieval with nonzero strength, require a readable compatible index; reject missing, corrupt or dimension-mismatched indices clearly. Do not silently choose another index. Log effective settings.
5. Launch the exact isolated Python executable in a Windows kill-on-close job before model initialization. Hold the GPU lock until worker and descendants terminate. Bound startup and the whole request; expose cancellation through the existing session mechanism. Only read a worker result after confirmed successful completion and cleanup.
6. Start with RMVPE if supported by the pinned revision. Explicitly enforce the requested 4 GB profile: CUDA half precision where supported, `x_pad=1`, `x_center=30`, `x_max=32`, or verified equivalent settings if the API differs. Log actual settings after initialization; upstream GPU-name heuristics must not silently override them. Derive any additional limits from the inspected source.
7. Apply an allocator ceiling no higher than the existing 3.4 GiB/85% policy, retain headroom and measure device-wide usage as well as PyTorch allocation. Start real tests with short inputs. Treat this as a measured operating target, not a guarantee that other apps or native allocations cannot exhaust VRAM.
8. Preserve HuBERT/content features at the rate expected by the pinned implementation, typically 16 kHz; honor the checkpoint's native output rate in memory; resample once to 48 kHz at publication. Do not write an upstream 16/40/44.1 kHz intermediate WAV and then claim compliance by converting it afterwards.
9. Use bounded audio segmentation compatible with the model's receptive field. Start qualification with at most five seconds of core audio per segment, plus required context. Account for padding, overlap and crossfades by sample count. Test exact-length boundaries, a short final tail and long inputs; do not introduce repeated/dropped words through naive stitching.
10. Validate output for finite samples, silence, implausible gain, excessive duration drift and malformed format before atomic publication. Measure amplitude before attenuation. Do not pass through unchanged source audio or silently switch to so-vits on RVC failure. Preserve existing output files on timeout, cancellation, validation or write failure.

**Gate B:** adapter and process lifecycle tests pass without importing RVC in the controller. If trusted compatible trained weights exist, a short real conversion also passes. If none exist, report runtime readiness and real-model qualification separately.

### C. Connect RVC to the existing application

Affected files: `conversion.py` as needed, `app_service.py`, `gradio_app.py`, and focused integration tests.

1. Add a converter selector with distinct RVC and so-vits settings. RVC needs checkpoint/index controls; so-vits retains its matching config/speaker controls. Do not conflate the two checkpoint formats.
2. Preserve the existing speech tab, SAPI default, session state, generated-speech handoff, file-based lossless upload and local-only server binding.
3. Do not activate a trained-model default automatically. When the RVC runtime or checkpoint is unavailable, display the missing requirement precisely while retaining usable TTS/so-vits functionality.
4. Keep cancellation effective while a worker runs, prevent a second operation in the same session, and keep GPU ownership across cleanup. Test actual HTTP rendering and queued callback execution, not only construction of Gradio components.

**Gate C:** existing workflows still pass and RVC routing validates its own requirements. A real-model UI test is conditional on a qualified asset; mocked routing tests must be labelled accordingly.

### D. Prepare the custom recording reproducibly

Suggested files: `prepare_rvc_dataset.py`, `rvc_dataset.py` if separation helps, and dataset tests. Use a new ignored destination such as `datasets/rvc/anshumaan/<manifest-id>/`. Do not change the existing so-vits dataset or weaken its recording-group safeguards.

1. Verify the source hash before and after preparation. Read bounded blocks or use memory-mapped access rather than duplicating the entire 22-minute waveform in several arrays. Work only on derived copies.
2. Detect speech regions with VAD and short context padding. Merge nearby regions, choose quiet boundaries where possible, and produce 2–10 second clips. Log threshold settings and all excluded regions. Do not manufacture speech from silent padding to satisfy a duration threshold.
3. Exclude silence/nonfinite/corrupt clips and flag possible clipping or substantial noise for review. Do not apply automatic denoising, compression, presence EQ, aggressive gating or per-clip loudness normalization. Preserve quiet speech and identify uncertain clips rather than silently treating every VAD decision as truth.
4. Store derived clips as 48 kHz PCM24 WAV. Record source hash, exact source sample interval, clip hash, speech fraction, amplitude metrics and any transform in a manifest. Give every output a deterministic identity; reruns should verify/reuse a matching completed dataset or create a new version, never partially mix versions.
5. Detect exact duplicate clips. Track rejected clips and reasons, retained speech duration, duration distribution and representative review timestamps. A configurable minimum usable-duration check must be explicit, not described as a universal quality threshold.
6. Prefer recording/session-level train/validation separation. With only this one recording, create a **provisional temporal holdout**: select a contiguous approximately 10% interval before slicing, exclude at least 10 seconds around its boundary from both sets, and ensure no speech context/overlap crosses the split. Label it as same-session validation, not an independent generalization test. Check source intervals and duplicates across splits. Leave room to replace it with a separate recording later.
7. Produce a local dataset report and a small review playlist or timestamp list. No custom audio needs to be uploaded or committed. Dataset preparation does not authorize full training or certify speaker identity.

**Gate D:** the original is unchanged; every retained clip satisfies the storage/duration contract; source ranges and split isolation are auditable; reports show actual counts/durations and exclusions. Any inability to construct a valid split is reported, not bypassed.

### E. Qualify, document and deliver

Use a focused test ladder to conserve time and model usage:

1. Run targeted tests during development. Cover absent/corrupt/incompatible model and index, no-index mode, paths with spaces/Unicode, aliasing, invalid options, native-rate conversion, segment joins, failed-publication preservation, GPU lock contention, cancellation and forced worker/controller termination.
2. Test dataset silence, short tails, continuous speech, bad samples, duplicate sources/clips, interrupted preparation, idempotent reruns and split leakage. Test temporary inputs, not alterations to the user's recording.
3. With a trusted trained RVC checkpoint, run short conversion, multi-segment conversion and three consecutive requests. Capture time, native/output rates, model/index hashes, output validation, sampled device memory and peak allocator metrics. Confirm no surviving workers or accumulating device memory after each request. Investigate more than 64 MiB retained above a stable idle baseline; device-wide readings may include other apps.
4. Force one timeout after a real GPU allocation and verify cleanup before any next job. Stop the GPU test on OOM; record diagnostics and make at most one evidence-based reduction to the memory/chunk profile before reporting a persistent hardware limit. Do not loop through broad parameter sweeps.
5. Use the existing SAPI sample/corpus as the stable conversion input. If CPU ASR is available, compare source/output recognition and report its limitations; do not equate ASR accuracy with target-voice similarity. Do not run Bark benchmarks or retrain Coqui to qualify RVC.
6. Run the full existing regression suite once after integration, plus added tests; run the original environment health check and the new isolated runtime health check. A new failure justifies targeted reruns. Do not repeat passing heavyweight checks without a relevant change.
7. Write `MILESTONE_2_REPORT.md` with implemented features, exact versions/SHAs, test counts, real-versus-mocked evidence, memory results, recording hash, dataset summary, blockers and manual review items. Update README and PROJECT_STATE with the actual result. Preserve earlier qualification reports.
8. Commit logical completed changes under the user's existing identity on the existing working branch, review the staged files for audio/models/secrets, and push without force. Confirm the remote commit matches. If credentials fail, retain the local commits and state the exact remaining action. Do not merge into main or perform a shutdown.

## 4. Acceptance criteria and honest completion states

| Area | Required evidence |
|---|---|
| Original functionality | Existing regression suite and UI HTTP health pass; original environment's critical versions unchanged |
| Isolation | RVC uses its own pinned interpreter/source; controller does not load its models |
| Worker lifetime | Timeout, cancellation and controller death reclaim owned workers; next GPU operation waits for cleanup |
| Audio | Supported paths persist only 48 kHz PCM24 WAV; atomic output publication; no silent backend substitution |
| Model/index | Explicit compatibility checks; optional index semantics tested; missing real weights reported |
| Recording | Original SHA256 unchanged; only derived, ignored files created |
| Dataset | Bounded clips, provenance, repeatability and nonoverlapping provisional split demonstrated |
| Real GPU qualification | Required before claiming RVC inference works on this laptop; mock tests alone are insufficient |
| Delivery | Report and docs match evidence; commit/push status verified |

If real trained weights are absent, the deliverable may be **implementation and dataset preparation complete; real-model qualification blocked**. Do not call the entire milestone qualified. Explain that the next training milestone must supply/export compatible weights and finish those tests. Do not ask the user to write code or debug dependencies.

## 5. Benefits, risks and mitigations

| Decision | Benefit | Cost/risk and mitigation |
|---|---|---|
| Separate RVC environment | Protects the tested TTS stack | Extra disk/download cost; estimate first and pin only required inference dependencies |
| Reuse contained workers | Predictable cancellation and VRAM recovery | Cold-start latency; measure it before considering resident model pools |
| Short bounded segments | Smaller inference memory peaks | Join/context artifacts; retain context and verify sample-accurate stitching |
| Explicit no-index support | Allows model-only inference | May change voice fidelity; record settings and do not conceal a missing requested index |
| Conservative VAD slicing | Removes long gaps with reproducible provenance | May reject quiet speech or cut phonemes; retain padding and review flagged boundaries |
| Temporal holdout for one file | Enables a useful provisional split | Correlated recording conditions; label limitation and seek independent material for final qualification |
| Separate training milestone | Avoids wasting compute before plumbing works | Custom voice is not produced yet; finish training after readiness/quality gates |

## 6. Manual review, left until the end

Ask only for information the executor cannot establish: preferred voice/pacing after listening; whether the source contains one intended speaker and unwanted background content; and representative independent recording material if final voice qualification needs it. Supply exact clips/timestamps to review. Listening does not block dependency isolation, adapter development, tests or provisional dataset preparation.

Milestone 3 will plan a bounded 4 GB training pilot, then checkpoint selection, feature-index creation, held-out inference and listening evaluation. It must specify a step/time budget, batch/precision settings measured on this laptop, resumability and stop criteria before any longer run. This plan does **not** start that training.

## 7. Handoff prompt

> Execute IMPLEMENTATION_PLAN_MILESTONE_2.md in the current repository. Follow its gates and preserve the verified TTS baseline and original recording. Implement the RVC foundation and dataset preparation, run the relevant tests, and document real results and blockers. Keep full acoustic training out of scope. Make routine decisions autonomously, leave only human listening/data questions until the end, and provide the final report plus verified Git status. Do not schedule or execute shutdown.

## 8. Upstream references and verification caveat

- [Official RVC repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Upstream configuration inspected during planning](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/master/configs/config.py)

Planning-time inspection found that the older `tools/infer_cli.py` URL returned 404 and current configuration code differs from the older API assumptions. These moving references are discovery aids, not a dependency lock. Resolve the selected commit and inspect its actual code during Gate A; do not assume an old tutorial's paths, CLI flags or precision heuristics still apply.

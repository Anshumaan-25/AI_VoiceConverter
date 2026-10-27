# RVC numerical recovery and Suno song workflow

Date: 2026-10-27. Branch: `bark-chunk-and-stitch`.

**Follow-up, 2026-10-28:** The user identified VocalMixer. Fresh full-track separation and the real 30-second custom-voice mix are now complete; see `SUNO_DEMO_REPORT.md`. The missing-input descriptions below record the original recovery delivery state.

**Bark is parked indefinitely. A bounded BF16 training recovery and real RVC speech conversion now pass. The requested 30-second Suno song has not been rendered because no confirmed Suno vocal/instrumental files were supplied.** The final mix workflow is implemented and tested; provide the local stem paths to run it. No unidentified old song was substituted.

## Numerical investigation and recovery

The original failed Milestone 3 workspace, step-200 checkpoint, data, feature arrays and budget were preserved. The user explicitly authorized recovery; `RECOVERY_AND_SUNO_PLAN.md` records the new budgets and boundaries. New workspaces have explicit checkpoint lineage and immutable data/assets/source/architecture checks, rather than resetting the failed run.

An instrumented FP16 replay restored the original step-200 networks, optimizers, schedulers, scaler, RNG and sample offset. It completed **80 updates to step 280** in **93.625 seconds** without reproducing the original nonfinite loss. Checks covered generator content/posterior/flow/decoder outputs, attention layers, discriminator branches and unscaled gradients. The original first invalid operation and exact cause therefore remain **unproven**. This result does not prove a deterministic bug was fixed.

The recovery experiment changed autocast from FP16 to hardware-supported BF16, retaining FP32 parameters and the original model architecture, 48 kHz native rate, batch size one and 17280-sample crop. BF16 provides a wider exponent range with the same two-byte tensor storage. Loss scaling is disabled for BF16; mandatory finite unscaled-gradient checks reject an invalid update. No sample was dropped to hide a failure. This is tested numerical hardening, not a confirmed explanation of the old error.

| Stage | Initial → final paired step | Attempts / successful pairs | Worker seconds | Skipped updates |
|---|---|---:|---:|---:|
| BF16 smoke | 200 → 210 | 10 / 10 | 32.407 | 0 |
| BF16 resume | 210 → 211 | 1 / 1 | 64.953 | 0 |
| BF16 main run | 211 → 700 | 489 / 489 | 526.906 | 0 |
| BF16 total | 200 → 700 | **500 / 500** | **624.266** | **0** |

The 500-new-attempt limit was reached and training stopped. No extra training was launched because time remained. Peak allocated training VRAM was **2848413696 bytes**; peak reservation **3120562176 bytes**; sampled peak device use **3109 MiB**, below the 4096 MiB device ceiling. Every training worker exited and GPU use returned to **0 MiB**. Diagnostic FP16 peak sampled device use was 3071 MiB.

A redundant generated step-211 recovery checkpoint was removed only after a newer complete set was verified, maintaining disk headroom. Original failed-run checkpoints were preserved. No packages or pretrained source pins changed. Source audio and feature hashes were reverified.

## Current model and retrieval index

- Model: `logs/rvc-recovery/bf16/pilot-v2-48k.pth`, step **700**.
- Model SHA256: `5ec373301bae3a2a942dc0cc834490d48f69a98f8a7761393a3c701071fbaa5d`.
- Parent complete resume-state SHA256: `b12a66e8098e5241e373468c45a5c3d96c1678b750f180fceb4334f8b62798f1`; `latest.json` identifies the matching resume directory.
- Index: `logs/rvc-recovery/bf16/pilot-v2-48k.index`, flat L2, **47184 vectors**, 768 dimensions, training split only.
- Index SHA256: `11c086614a5f116697f3f89a6503460af7d2d469453dea947598c627d5325a12`.
- Target speaker ID: **0**. Other pretrained speaker-table entries are not qualified target voices.

This remains a pilot model. Finite training and correct transcripts do not establish voice resemblance, natural singing or production quality.

## Actual conversion results

The exported model was loaded through the real isolated RVC adapter, including its strict state loader and PCM24 output validation. Four consecutive measured conversions passed:

| Source | Retrieval strength | Duration | Source WER | Converted WER | Memory recovered |
|---|---:|---:|---:|---:|---|
| Short known-text SAPI speech | 0 | 1.804 s | 0% | 0% | Yes |
| Same short speech | 0.5 | 1.804 s | 0% | 0% | Yes |
| Multi-segment known-text speech | 0 | 9.775 s | 0% | 0% | Yes |
| Longer known-text speech | 0 | 20.277 s | 1.695% | 1.695% | Yes |

The same existing CPU Vosk recognizer and known text were used for both source and output. Longest measured conversion used 696661504 peak allocated bytes, 811597824 reserved bytes and 763 MiB sampled device memory. Outputs retained exact input frame counts and 48 kHz PCM24 mono storage. Segment-join window measurements are in the local report; audible seam quality remains a listening judgment.

Additional real checks passed:

- Cancellation after synthesizer/encoder loading: owned process tree terminated, previous destination bytes preserved, GPU returned to 0 MiB.
- Forced per-chunk timeout after loading: same cleanup and destination-preservation checks passed.
- A subsequent conversion completed successfully.
- The live queued Gradio UI converted speech with step-700 weights, then generated a separate SAPI request. Playback/download files retained 48 kHz PCM24 mono. The temporary server was closed.

A target-recording validation clip was also converted. Its English ASR source transcript was empty, while converted ASR returned one word. This comparison is **unscorable**, not a valid 100% word-error finding or evidence of transcript retention. The raw empty-reference calculation was retained in the local report history and explicitly reclassified. Human reference/listening or a suitable recognizer is required. The same-session holdout also remains correlated with the training recording session.

Listening artifacts under `logs/rvc-recovery/qualification/`:

- `01-source.wav` and `01-converted-0.wav`: approximately ten-second source/converted speech pair.
- `02-source.wav` and `02-converted-0.wav`: approximately twenty-second pair crossing several joins.
- `00-converted-0.5.wav`: retrieval comparison.
- `heldout-converted.wav`: the unscorable target-recording diagnostic, with source path in `report.json`.

These are real speech conversions. No Bark generation or singing result is inferred from them.

## Song workflow and remaining input

`song_demo.py` accepts matching vocal and instrumental stems, a model, output directory and start time. It prepares exactly the same 30-second interval from each, performs in-memory rate conversion, stores 48 kHz PCM24 intermediates, converts the mono vocal through RVC and preserves the accompaniment's stereo information. Original files remain unchanged.

It publishes the original comparison mix, raw converted vocal, stereo PCM24/48 kHz master and a final MP3 explicitly encoded at **320 kbps**. Presence EQ defaults to zero. A recorded whole-clip RMS gain, bounded to +/-12 dB, matches converted-vocal mix level to the guide; the raw converted stem remains available. The mix limiter ceiling is 0.89, with latency compensation. Existing output directories and input/output aliases are rejected. Failed staging retains diagnostics.

The synthetic fixture verified actual FFmpeg mixing/export, exact 1440000-sample WAV duration, stereo-channel preservation, source preservation, limiter headroom, MP3 codec, 48000 Hz and 320000-bit/s stream metadata. Its converter was mocked explicitly: **this fixture is not a Suno conversion or a voice-quality demonstration**.

No confirmed Suno stems were found in the project or the immediate Downloads audio inventory. A question requesting their local paths was left pending. An unidentified old song was not used as a substitute. The only input needed to render the requested demonstration is the intended Suno vocal/instrumental pair, with an optional preferred start time. If only the full song exists, identify that file so separation can be handled explicitly.

Reproducible command once the stems exist:

```powershell
./venv310/Scripts/python.exe -X utf8 -B song_demo.py --vocals "C:/path/song-vocals.wav" --instrumental "C:/path/song-instrumental.wav" --model logs/rvc-recovery/bf16/pilot-v2-48k.pth --output-dir logs/song-demo --start 0
```

This command is a reproducibility record; the user is not being handed implementation/debugging work. The application can run it once the files are identified. Voice resemblance, preferred register and singing artifacts are judged on the resulting listening pair.

## Verification and delivery

**70 tests passed** with both real CUDA lifecycle regressions enabled; both environment health checks and package consistency passed. Additional direct checks covered empty-reference ASR classification. Python syntax and Git whitespace checks passed. Compact evidence is in `validation/rvc-recovery-results.json`; verbose tests, budgets, traces and qualification evidence remain under ignored `logs/rvc-recovery/`.

Bark is removed from UI/CLI selection, explicit public generation is rejected, and the automatic backend configuration is SAPI only with a guard against reintroducing Bark. Legacy code/assets remain preserved. No further Bark model work or downloads occurred.

Current code/docs are delivered on `bark-chunk-and-stitch` using the existing user Git identity. Recordings, stems, models, indices, environments and logs remain ignored. The final response records the verified commit/push and remaining missing input. No main merge or shutdown is part of this work.

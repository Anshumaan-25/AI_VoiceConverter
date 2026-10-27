# Tragic — fresh separation and custom-voice demonstration

Date: 2026-10-28. The provided `VocalMixer` folder contains the full track at `input_songs/tragic.mp3`; the shorter `tragic(cut).mp3` was not used.

**Completed:** fresh full-song Demucs separation, a matching 30-second vocal/instrumental excerpt, and the previously requested custom-voice conversion/mix. No additional training or Bark generation was run. Listening quality and voice resemblance remain for the user to judge.

## Source and separation

- Full source: `C:/Users/anshu/Documents/PROJECTS/VocalMixer/input_songs/tragic.mp3`, 166.68 seconds decoded.
- Source SHA256: `334d57f13cbafa257f36705b306e2319393de73ece302d76c045f614f70e338f`, verified unchanged after processing.
- Runtime: existing `VocalMixer/venv310`, Demucs **4.0.1**, torch **2.1.0+cu121**. No dependencies were installed or changed in either environment.
- Model: **htdemucs_ft**, the four pretrained models applied sequentially, overlap 0.5, shifts 0, no parallel data-loader workers. Cached asset SHA256 prefixes were checked against the installed official model metadata; full measured hashes and package-source hashes are recorded in `separation.json`.
- Native 44.1 kHz processing remained in memory. All persisted stems are **48 kHz PCM24 WAV**. The same peak-preserving gain was applied to both stems; observed gain was **1.0**.
- Full separation worker time: **71.156 seconds**. Peak PyTorch allocation: **577705472 bytes**; peak reservation: **805306368 bytes**. Observed GPU use during separation was about 0.9 GB; all owned workers exited and usage returned to 0 MiB before RVC started.

New full-song stems are under `logs/suno-tragic-20261028/separation/`. Existing inputs, old Demucs stems, previous finals, models and code in VocalMixer were preserved.

## Selected excerpt and prepared inputs

Selected source interval: **00:11.8–00:41.8**, exactly **30 seconds**. Selection used the earliest sustained-vocal window with leading context; 29 of 30 scored seconds exceeded the vocal-energy threshold. Energy does not establish absence of bleed or metallic separation artifacts, so the vocal is supplied for listening review.

| File under `logs/suno-tragic-20261028/separation/` | Format |
|---|---|
| `vocals.wav` | Full 166.68-second vocal, stereo PCM24/48 kHz |
| `instrumental.wav` | Full 166.68-second instrumental, stereo PCM24/48 kHz |
| `vocals-30s.wav` | Exactly 1440000 samples, mono PCM24/48 kHz, ready for RVC |
| `instrumental-30s.wav` | Same interval and frame count, stereo PCM24/48 kHz |

The source is MP3; subsequent audio stages are lossless WAV. Changing storage format does not recover information lost in the original source encoding.

## Actual custom-voice test

The matching full stems were passed to `song_demo.py` at start time 11.8 seconds. Conversion used the step-700 custom model and matching training-only retrieval index from the bounded recovery, speaker ID 0, retrieval strength **0.5**, RMVPE and **zero pitch transposition**. The new model was used instead of VocalMixer's legacy placeholder voice model.

- Conversion took **20.047 seconds**, using the isolated RVC worker and shared GPU lock.
- The raw converted vocal is retained. A single **+3.69832 dB** whole-clip RMS gain was applied only during mixing to balance it against the guide vocal.
- Presence EQ: **0 dB**. Limiter ceiling: **0.89**, with latency compensation.
- Both source comparison and converted master have exactly **1440000 frames**, 48 kHz PCM24 stereo. Observed WAV peak was approximately 0.8900001, within PCM quantization tolerance.
- Final MP3 stream verified by FFprobe: **48000 Hz, stereo, 320000 bit/s, 30.000000 seconds**.
- Final MP3 SHA256: `019798d444725fe020dd62b98f778eab101cdfb05e0169e7441bbbf90d6bec03`.

Listening files under `logs/suno-tragic-20261028/custom-voice-demo/`:

- `source-vocals.wav`: prepared original vocalist.
- `converted-vocals.wav`: raw custom-model vocal, before mix gain/EQ.
- `original-mix.wav`: original vocal plus the same instrumental.
- `your-voice-mix.wav`: lossless custom-model mix.
- `your-voice-mix.mp3`: final 320 kbps preview/download.

The model remains a bounded pilot. Successful conversion and format checks do not establish subjective voice identity, natural singing or a perfect stem. Compare the original and converted mixes, then inspect the isolated vocals if something sounds smeared or synthetic. No source lyrics/transcript accuracy claim was made for this musical excerpt.

## Verification and reproducibility

The fresh separation and real 30-second RVC/mix completed successfully. Header/frame/finite-sample checks passed; FFprobe confirmed the encoded bitrate/channel/rate/duration; input hashes were unchanged. Four new targeted tests passed, covering an intro followed by vocals, insufficient/silent/nonfinite material, leading-context bounds and input-duration rejection before worker/output creation. The earlier 70-test regression baseline remains intact; no converter or mixer implementation changed in this follow-up. Compact results: `validation/suno-tragic-demo-results.json`.

`separate_song.py` is a reusable contained entry point accepting an explicit Python interpreter with the inspected Demucs runtime. It refuses existing output directories, enforces deadlines and disk reserve, preserves failed staging, and writes audio only after finite-value checks. It can use CPU explicitly when needed; this run used CUDA successfully.

All new media and verbose evidence remain under ignored `logs/suno-tragic-20261028/`. Code/docs/results are committed on `bark-chunk-and-stitch`; exact commit/push outcome is verified in the final delivery. No external project files were edited, no main merge occurred and no shutdown was scheduled.

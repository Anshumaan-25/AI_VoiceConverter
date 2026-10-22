from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from voice_autopilot import ROOT_DIR, infer_once, list_speakers_from_config


DEFAULT_MODEL = ""
DEFAULT_CONFIG = ROOT_DIR / "so-vits-svc" / "configs" / "config.json"


def _safe_speakers(config_path: Path) -> list[str]:
    try:
        speakers = list_speakers_from_config(config_path)
        return speakers if speakers else []
    except Exception:
        return []


def reload_speakers(config_path: str):
    speakers = _safe_speakers(Path(config_path))
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)


def run_conversion(audio_path: str, model_path: str, config_path: str, speaker: str, transpose: int, f0_predictor: str):
    if not audio_path:
        return None, "Please provide input audio."

    if not model_path or not config_path or not speaker:
        return None, "Select a trained model, matching configuration and target speaker."

    try:
        out_path = infer_once(
            input_audio=Path(audio_path),
            model_path=Path(model_path),
            config_path=Path(config_path),
            speaker=speaker,
            transpose=transpose,
            f0_predictor=f0_predictor,
            wav_format="wav",
        )
        return str(out_path), f"Conversion finished: {out_path}"
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"Conversion failed: {exc}\n\n{traceback.format_exc()}"


def build_ui() -> gr.Blocks:
    config_speakers = _safe_speakers(DEFAULT_CONFIG)

    with gr.Blocks(title="AI Voice Converter (so-vits-svc)") as app:
        gr.Markdown(
            "# AI Voice Converter\n"
            "Inference-first Gradio UI for so-vits-svc. Provide a trained model and config, then convert source speech to your target speaker."
        )

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(label="Input Speech", type="filepath")
                model_path = gr.Textbox(label="Model Path", value=str(DEFAULT_MODEL))
                config_path = gr.Textbox(label="Config Path", value=str(DEFAULT_CONFIG))

                with gr.Row():
                    speaker = gr.Dropdown(label="Target Speaker", choices=config_speakers, value=config_speakers[0] if config_speakers else None)
                    reload_btn = gr.Button("Reload Speakers")

                transpose = gr.Slider(label="Pitch Shift (semitones)", minimum=-12, maximum=12, step=1, value=0)
                f0_predictor = gr.Dropdown(
                    label="F0 Predictor",
                    choices=["rmvpe", "dio", "pm", "harvest", "crepe", "fcpe"],
                    value="rmvpe",
                )
                convert_btn = gr.Button("Convert", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(label="Converted Output", type="filepath")
                logs = gr.Textbox(label="Logs", lines=12)

        reload_btn.click(fn=reload_speakers, inputs=[config_path], outputs=[speaker])
        convert_btn.click(
            fn=run_conversion,
            inputs=[input_audio, model_path, config_path, speaker, transpose, f0_predictor],
            outputs=[output_audio, logs],
        )

    return app


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)

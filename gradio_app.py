from __future__ import annotations
import logging
import uuid
from pathlib import Path
import gradio as gr
from app_service import VoiceApplication
from voice_autopilot import ROOT_DIR, list_speakers_from_config

DEFAULT_CONFIG = ROOT_DIR/'so-vits-svc'/'configs'/'config.json'
SERVICE = VoiceApplication()
LOGGER = logging.getLogger(__name__)


def _safe_speakers(config_path):
    try:
        return list_speakers_from_config(Path(config_path))
    except (OSError, ValueError, TypeError):
        return []


def reload_speakers(config_path):
    speakers = _safe_speakers(config_path)
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)


def _session_id(state):
    # Gradio 3 stores a separate mutable State copy for each browser/API session.
    return state.setdefault('id', uuid.uuid4().hex)


def generate_speech(text, backend, diagnostics, session):
    try:
        result = SERVICE.generate(_session_id(session), text, backend, diagnostics)
        failed = [a['backend'] for a in result.attempts if a['status'] == 'failed']
        status = f'Speech ready. Engine: {result.backend}; chunks: {len(result.chunks)}.'
        if failed:
            status += ' Fallback used after '+', '.join(failed)+' failed.'
        if result.diagnostics:
            status += f' Diagnostics: {result.diagnostics}'
        return str(result.path), str(result.path), status
    except Exception as exc:
        LOGGER.exception('Speech generation failed')
        return None, None, f'Speech generation failed: {exc}'


def cancel_job(session):
    active = SERVICE.jobs.cancel(_session_id(session))
    return 'Cancellation requested; waiting for worker cleanup.' if active else 'No running job. Pending jobs cancelled.'


def use_speech(path):
    if not path or not Path(path).is_file():
        raise gr.Error('Generate speech first.')
    return path


def run_conversion(audio_file, model_path, config_path, speaker, transpose, f0_predictor, session):
    if not audio_file:
        return None, 'Provide an audio file or use the generated speech.'
    if not model_path or not config_path or not speaker:
        return None, 'Select a trained model, matching configuration and target speaker.'
    try:
        source = getattr(audio_file, 'name', audio_file)
        output = SERVICE.convert(_session_id(session), Path(source), model_path, config_path,
                                 speaker, transpose, f0_predictor)
        return str(output), 'Conversion finished.'
    except Exception as exc:
        LOGGER.exception('Voice conversion failed')
        return None, f'Conversion failed: {exc}'


def build_ui():
    speakers = _safe_speakers(DEFAULT_CONFIG)
    with gr.Blocks(title='AI Voice Converter', analytics_enabled=False) as app:
        gr.Markdown('# AI Voice Converter\nGenerate speech, listen, then send it to a trained voice model.')
        generated_path = gr.State(None)
        session = gr.State({})
        with gr.Tab('Generate speech'):
            text = gr.Textbox(label='Script', lines=5, max_lines=20)
            backend = gr.Dropdown(label='Speech engine', value='sapi', choices=[
                ('Windows voice — reliable, offline', 'sapi'),
                ('Bark — expressive, may add words', 'bark'),
                ('Automatic — Bark with Windows fallback', 'auto'),
                ('Coqui — diagnostic only', 'coqui')])
            diagnostics = gr.Checkbox(label='Save diagnostic traces and audio locally', value=False)
            with gr.Row():
                generate = gr.Button('Generate speech', variant='primary')
                stop_speech = gr.Button('Cancel')
            speech_audio = gr.Audio(label='Generated speech', type='filepath')
            speech_status = gr.Textbox(label='Status', interactive=False)
            use_generated = gr.Button('Use this speech for conversion')
        with gr.Tab('Convert voice'):
            gr.Markdown('The current converter is **so-vits-svc**. It needs a trained so-vits checkpoint and its matching configuration. RVC models are not supported yet.')
            # File upload preserves PCM24 bytes; Audio preprocessing can re-encode to PCM16.
            input_audio = gr.File(label='Source audio (WAV or FLAC)', type='file', file_types=['.wav', '.flac'])
            model = gr.Textbox(label='Trained model path', value='')
            config = gr.Textbox(label='Configuration path', value=str(DEFAULT_CONFIG))
            with gr.Row():
                speaker = gr.Dropdown(label='Target speaker', choices=speakers, value=speakers[0] if speakers else None)
                reload = gr.Button('Reload speakers')
            transpose = gr.Slider(label='Pitch shift (semitones)', minimum=-12, maximum=12, step=1, value=0)
            predictor = gr.Dropdown(label='Pitch detector', choices=['rmvpe','dio','pm','harvest','crepe','fcpe'], value='rmvpe')
            with gr.Row():
                convert = gr.Button('Convert voice', variant='primary')
                stop_conversion = gr.Button('Cancel')
            converted = gr.Audio(label='Converted speech', type='filepath')
            conversion_status = gr.Textbox(label='Status', interactive=False)
        reload.click(reload_speakers, config, speaker, queue=False, api_name=False)
        generation = generate.click(generate_speech, [text, backend, diagnostics, session],
                                    [speech_audio, generated_path, speech_status], api_name='generate_speech')
        conversion = convert.click(run_conversion, [input_audio, model, config, speaker, transpose, predictor, session],
                                   [converted, conversion_status], api_name='convert_voice')
        use_generated.click(use_speech, generated_path, input_audio, queue=False, api_name=False)
        stop_speech.click(cancel_job, session, speech_status, queue=False, cancels=[generation], api_name=False)
        stop_conversion.click(cancel_job, session, conversion_status, queue=False, cancels=[conversion], api_name=False)
    return app.queue(concurrency_count=2, max_size=8, api_open=False)


if __name__ == '__main__':
    build_ui().launch(server_name='127.0.0.1', server_port=7860, share=False)

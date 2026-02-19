from pathlib import Path

import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils.subprocess import run_script_with_log
from gradio_tabs.common import create_no_model_alert


def call_convert_onnx(
    model: str,
):
    if model == "":
        return "Error: Por favor ingrese el nombre del modelo."
    logger.info("Start converting model to onnx...")
    cmd = [
        "convert_onnx.py",
        "--model",
        model,
    ]
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "Conversión a ONNX completada."


initial_md = """
Convierte el modelo en formato safetensors a formato ONNX.
Este modelo ONNX se puede utilizar en bibliotecas externas compatibles. Por ejemplo, al convertirlo a formato AIVM/AIVMX con [AIVM Generator](https://aivm-generator.aivis-project.com/), puede utilizarlo en [AivisSpeech](https://aivis-project.com/).

**La conversión tarda más de 5 minutos**. Consulte el registro del terminal para ver el progreso.

Después de la conversión, se generará un archivo con el mismo nombre que el modelo seleccionado y extensión `.onnx`.
"""


def create_onnx_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def get_model_files(model_name: str):
        return [str(f) for f in model_holder.model_files_dict[model_name]]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"No se encontraron modelos. Por favor coloque los modelos en {model_holder.root_dir}."
        )
        return create_no_model_alert(model_holder.root_dir)
    initial_id = 0
    initial_pth_files = get_model_files(model_names[initial_id])

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="Lista de modelos",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path = gr.Dropdown(
                    label="Archivo del modelo",
                    choices=initial_pth_files,
                    value=initial_pth_files[0],
                )
            refresh_button = gr.Button("Actualizar")
        convert_button = gr.Button("Convertir a formato ONNX", variant="primary")
        info = gr.Textbox(label="Información")

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        def refresh_fn() -> tuple[gr.Dropdown, gr.Dropdown]:
            names, files, _ = model_holder.update_model_names_for_gradio()
            return names, files

        refresh_button.click(
            refresh_fn,
            outputs=[model_name, model_path],
        )
        convert_button.click(
            call_convert_onnx,
            inputs=[model_path],
            outputs=[info],
        )

    return app


if __name__ == "__main__":
    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    model_holder = TTSModelHolder(assets_root, "cpu", "", ignore_onnx=True)
    app = create_onnx_app(model_holder)
    app.launch(inbrowser=True)

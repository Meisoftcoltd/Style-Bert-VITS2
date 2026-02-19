import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Error: Por favor ingrese el nombre del modelo."
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # Ignore ONNX warnings
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "División de audio completada."


def do_transcribe(
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Error: Por favor ingrese el nombre del modelo."
    if hf_repo_id == "litagin/anime-whisper":
        logger.info(
            "Since litagin/anime-whisper does not support initial prompt, it will be ignored."
        )
        initial_prompt = ""

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}. Si el mensaje de error está vacío, puede que no haya problemas, verifique los archivos de transcripción."
    return "Transcripción de audio completada."


how_to_md = """
Herramienta para crear conjuntos de datos de entrenamiento para Style-Bert-VITS2. Consta de dos partes:

- Cortar y dividir (slice) segmentos de habla de una longitud adecuada del audio dado.
- Transcripción del audio.

Puede usar ambos, o solo el último si no necesita dividir. **Si ya tiene archivos de audio de una longitud adecuada, como un corpus, no es necesario dividir.**

## Requisitos

Varios archivos de audio que contengan la voz que desea entrenar (el formato puede ser wav, mp3, etc.).
Es mejor tener una duración total razonable, se ha informado que incluso 10 minutos pueden funcionar. Puede ser un solo archivo o varios.

## Uso de División (Slice)
1. Coloque todos los archivos de audio en la carpeta `inputs` (si desea separar estilos, colóquelos en subcarpetas por estilo).
2. Ingrese el `Nombre del modelo`, ajuste la configuración si es necesario y presione el botón `Ejecutar división`.
3. Los archivos de audio resultantes se guardarán en `Data/{nombre_modelo}/raw`.

## Uso de Transcripción

1. Asegúrese de que haya archivos de audio en `Data/{nombre_modelo}/raw` (no es necesario que estén directamente en la raíz).
2. Ajuste la configuración si es necesario y presione el botón.
3. El archivo de transcripción se guardará en `Data/{nombre_modelo}/esd.list`.

## Notas

- ~~Los archivos wav demasiado largos (¿más de 12-15 segundos?) no parecían usarse para el entrenamiento. También los demasiado cortos pueden no ser buenos.~~ Esta restricción desapareció en la Ver 2.5 si selecciona "No usar muestreador de lotes personalizado" durante el entrenamiento. Sin embargo, los audios demasiado largos pueden aumentar el consumo de VRAM o causar inestabilidad, por lo que se recomienda dividirlos en una longitud adecuada.
- Cuánto se debe corregir el resultado de la transcripción depende del conjunto de datos.
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(
            "**Si ya tiene una colección de archivos de audio de 2-12 segundos y sus datos de transcripción, puede entrenar sin usar esta pestaña.**"
        )
        with gr.Accordion("Cómo usar", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="Ingrese el nombre del modelo (también se usa como nombre del hablante)."
        )
        with gr.Accordion("División de audio"):
            gr.Markdown(
                "**Si ya tiene datos que consisten en archivos de audio de longitud adecuada, coloque los audios en Data/{nombre_modelo}/raw y omita este paso.**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="Ruta de la carpeta con audios originales",
                        value="inputs",
                        info="Coloque archivos wav, mp3, etc. en esta carpeta",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="Descartar si es menor a estos segundos",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="Descartar si es mayor a estos segundos",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="Duración mínima de silencio para dividir (ms)",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="Añadir rango de tiempo al final del nombre del archivo WAV",
                    )
                    slice_button = gr.Button("Ejecutar división")
                result1 = gr.Textbox(label="Resultado")
        with gr.Row():
            with gr.Column():
                use_hf_whisper = gr.Checkbox(
                    label="Usar Whisper de HuggingFace (más rápido pero usa más VRAM)",
                    value=False,
                )
                whisper_model = gr.Dropdown(
                    [
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Modelo Whisper",
                    value="large-v3",
                    visible=True,
                )
                hf_repo_id = gr.Dropdown(
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-large-v3",
                        "openai/whisper-large-v2",
                        "kotoba-tech/kotoba-whisper-v2.1",
                        "litagin/anime-whisper",
                    ],
                    label="HuggingFace Whisper repo_id",
                    value="openai/whisper-large-v3-turbo",
                    visible=False,
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="Precisión de cálculo",
                    value="bfloat16",
                    visible=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="Tamaño del lote (Batch size)",
                    info="Aumentarlo acelera pero usa más VRAM",
                    visible=False,
                )
                language = gr.Dropdown(["ja", "en", "zh", "es"], value="es", label="Idioma")
                initial_prompt = gr.Textbox(
                    label="Prompt inicial",
                    value="Hola. ¿Cómo estás? Jaja, yo... ¡estoy bien!",
                    info="Ejemplo de cómo quieres que se transcriba (puntuación, risas, nombres propios, etc.)",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Número de haces (beam search)",
                    info="Menor es más rápido",
                )
            transcribe_button = gr.Button("Transcripción de audio")
            result2 = gr.Textbox(label="Resultado")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )
        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=not x),
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[whisper_model, hf_repo_id, batch_size, compute_type],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)

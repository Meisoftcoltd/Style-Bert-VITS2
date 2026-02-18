import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import gradio as gr
import yaml

from config import get_path_config
from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_bert_vits2.utils.subprocess import run_script_with_log, second_elem_of


logger_handler = None
tensorboard_executed = False

path_config = get_path_config()
dataset_root = path_config.dataset_root


@dataclass
class PathsForPreprocess:
    dataset_path: Path
    esd_path: Path
    train_path: Path
    val_path: Path
    config_path: Path


def get_path(model_name: str) -> PathsForPreprocess:
    assert model_name != "", "El nombre del modelo no puede estar vacío"
    dataset_path = dataset_root / model_name
    esd_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"
    return PathsForPreprocess(dataset_path, esd_path, train_path, val_path, config_path)


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_ES_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # Save preprocess log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_ES_bert: {freeze_ES_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}"
    )

    default_config_path = (
        "configs/config.json" if not use_jp_extra else "configs/config_jp_extra.json"
    )

    with open(default_config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(paths.train_path)
    config["data"]["validation_files"] = str(paths.val_path)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_EN_bert"] = freeze_EN_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_ES_bert"] = freeze_ES_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False

    config["data"]["use_jp_extra"] = use_jp_extra

    model_path = paths.dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
        )
        shutil.copytree(
            src=model_path,
            dst=paths.dataset_path / "models_backup",
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    pretrained_dir = Path("pretrained" if not use_jp_extra else "pretrained_jp_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Step 1: {pretrained_dir} folder not found.")
        return False, f"Step 1, Error: No se encontró la carpeta {pretrained_dir}."

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not Path("config.yml").exists():
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: initialization finished.")
    return True, "Step 1, Success: Configuración inicial completada"


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Step 2: start resampling...")
    dataset_path = get_path(model_name).dataset_path
    input_dir = dataset_path / "raw"
    output_dir = dataset_path / "wavs"
    cmd = [
        "resample.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 2: resampling failed.")
        return False, f"Step 2, Error: Falló el preprocesamiento de audio:\n{message}"
    elif message:
        logger.warning("Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: Preprocesamiento de audio completado con advertencias:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Step 2, Success: Preprocesamiento de audio completado"


def preprocess_text(
    model_name: str, use_jp_extra: bool, val_per_lang: int, yomi_error: str
):
    logger.info("Step 3: start preprocessing text...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Step 3: {paths.esd_path} not found.")
        return (
            False,
            f"Step 3, Error: No se encontró el archivo de transcripción {paths.esd_path}.",
        )

    cmd = [
        "preprocess_text.py",
        "--config-path",
        str(paths.config_path),
        "--transcription-path",
        str(paths.esd_path),
        "--train-path",
        str(paths.train_path),
        "--val-path",
        str(paths.val_path),
        "--val-per-lang",
        str(val_per_lang),
        "--yomi_error",
        yomi_error,
        "--correct_path",
    ]
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 3: preprocessing text failed.")
        return (
            False,
            f"Step 3, Error: Falló el preprocesamiento de transcripción:\n{message}",
        )
    elif message:
        logger.warning("Step 3: preprocessing text finished with stderr.")
        return (
            True,
            f"Step 3, Success: Preprocesamiento de transcripción completado con advertencias:\n{message}",
        )
    logger.success("Step 3: preprocessing text finished.")
    return True, "Step 3, Success: Preprocesamiento de transcripción completado"


def bert_gen(model_name: str):
    logger.info("Step 4: start bert_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path)]
    )
    if not success:
        logger.error("Step 4: bert_gen failed.")
        return False, f"Step 4, Error: Falló la generación de características BERT:\n{message}"
    elif message:
        logger.warning("Step 4: bert_gen finished with stderr.")
        return (
            True,
            f"Step 4, Success: Generación de características BERT completada:\n{message}",
        )
    logger.success("Step 4: bert_gen finished.")
    return True, "Step 4, Success: Generación de características BERT completada"


def style_gen(model_name: str, num_processes: int):
    logger.info("Step 5: start style_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            str(config_path),
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error("Step 5: style_gen failed.")
        return (
            False,
            f"Step 5, Error: Falló la generación de características de estilo:\n{message}",
        )
    elif message:
        logger.warning("Step 5: style_gen finished with stderr.")
        return (
            True,
            f"Step 5, Success: Generación de características de estilo completada:\n{message}",
        )
    logger.success("Step 5: style_gen finished.")
    return True, "Step 5, Success: Generación de características de estilo completada"


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_ES_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    val_per_lang: int,
    log_interval: int,
    yomi_error: str,
):
    if model_name == "":
        return False, "Error: Ingrese el nombre del modelo"
    success, message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_ES_bert=freeze_ES_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        log_interval=log_interval,
    )
    if not success:
        return False, message
    success, message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if not success:
        return False, message

    success, message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
    )
    if not success:
        return False, message
    success, message = bert_gen(
        model_name=model_name
    )
    if not success:
        return False, message
    success, message = style_gen(model_name=model_name, num_processes=num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return (
        True,
        "Success: Todo el preprocesamiento completado. Se recomienda revisar la terminal para verificar si hay errores.",
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    # Use Spanish training script if use_jp_extra is True (which we adapted)
    # Actually, we should check if language is ES or something, but here use_jp_extra flag switches the script.
    # The user asked to clone train_ms_jp_extra.py to train_ms_es_extra.py.
    # So if use_jp_extra is True, we should use train_ms_es_extra.py?
    # Or should we add another checkbox?
    # But `use_jp_extra` in data_utils triggers the single-bert mode.
    # If we use `train_ms_es_extra.py`, it expects single bert.
    # So `use_jp_extra` must be True.
    # I will stick with `train_ms_es_extra.py` if `use_jp_extra` is True,
    # assuming the user is training Spanish model with this "Extra" mode.
    # However, if user wants JP extra, they might be confused.
    # But the prompt is to adapt to COMPLETE SPANISH.
    # So `train_ms_jp_extra.py` logic should be replaced by `train_ms_es_extra.py`.

    train_py = "train_ms.py" if not use_jp_extra else "train_ms_es_extra.py"

    cmd = [
        train_py,
        "--config",
        str(paths.config_path),
        "--model",
        str(paths.dataset_path),
    ]
    if skip_style:
        cmd.append("--skip_default_style")
    if speedup:
        cmd.append("--speedup")
    if not_use_custom_batch_sampler:
        cmd.append("--not_use_custom_batch_sampler")
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        logger.error("Train failed.")
        return False, f"Error: Falló el entrenamiento:\n{message}"
    elif message:
        logger.warning("Train finished with stderr.")
        return True, f"Success: Entrenamiento completado:\n{message}"
    logger.success("Train finished.")
    return True, "Success: Entrenamiento completado"


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            pass

        if time.time() - start_time > timeout:
            return False

        time.sleep(0.1)


def run_tensorboard(model_name: str):
    global tensorboard_executed
    if not tensorboard_executed:
        python = sys.executable
        tensorboard_cmd = [
            python,
            "-m",
            "tensorboard.main",
            "--logdir",
            f"Data/{model_name}/models",
        ]
        subprocess.Popen(
            tensorboard_cmd,
            stdout=SAFE_STDOUT,  # type: ignore
            stderr=SAFE_STDOUT,  # type: ignore
        )
        yield gr.Button("Iniciando...")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard did not start in the expected time.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Abrir Tensorboard")


change_log_md = """
**Cambios desde Ver 2.5**

- Al colocar audios en subdirectorios dentro de la carpeta `raw/`, se crearán automáticamente estilos correspondientes. Consulte "Cómo usar / Preparación de datos" para más detalles.
- Anteriormente, los archivos de audio de más de 14 segundos no se usaban para el entrenamiento, pero desde la Ver 2.5, al marcar "Desactivar muestreador de lotes personalizado", se puede entrenar sin esa restricción (por defecto desactivado). Sin embargo:
    - La eficiencia del entrenamiento con archivos largos puede ser mala y el comportamiento no está verificado.
    - Marcar esto aumenta considerablemente el uso de VRAM. Si falla el entrenamiento o falta VRAM, reduzca el tamaño del lote o desmarque esta opción.
"""

how_to_md = """
## Cómo usar

- Prepare los datos, ingrese el nombre del modelo, ajuste la configuración si es necesario y presione el botón "Ejecutar preprocesamiento automático". El progreso se mostrará en la terminal.

- Use "Preprocesamiento manual" si desea ejecutar paso a paso (normalmente el automático es suficiente).

- Una vez finalizado el preprocesamiento, presione "Iniciar entrenamiento".

- Para reanudar el entrenamiento, simplemente ingrese el nombre del modelo y presione "Iniciar entrenamiento".

## Sobre la versión Extra (JP-Extra / ES-Extra)

Permite usar la estructura del modelo [Bert-VITS2 Japanese-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta).
Mejora la naturalidad, acento y entonación en Japonés y Español, pero pierde la capacidad de hablar en Inglés y Chino (multilenguaje deshabilitado).
"""

prepare_md = """
Primero prepare los datos de audio y el texto de transcripción.

Organícelos de la siguiente manera:
```
├── Data/
│   ├── {nombre_del_modelo}
│   │   ├── esd.list
│   │   ├── raw/
│   │   │   ├── foo.wav
│   │   │   ├── bar.mp3
│   │   │   ├── style1/
│   │   │   │   ├── baz.wav
│   │   │   │   ├── qux.wav
│   │   │   ├── style2/
│   │   │   │   ├── corge.wav
│   │   │   │   ├── grault.wav
...
```

### Cómo organizar
- Si coloca archivos de audio dentro de carpetas `style1/`, `style2/`, etc., se crearán automáticamente esos estilos además del predeterminado.
- Si no necesita estilos, coloque todo directamente en `raw/`.
- Soporta formatos como wav, mp3, etc.

### Archivo de transcripción `esd.list`

El archivo `Data/{nombre_del_modelo}/esd.list` debe tener el siguiente formato:

```
path/to/audio.wav|{nombre_hablante}|{idioma: ZH, JP, EN, ES}|{texto_transcripción}
```

- La ruta es relativa a `raw/`.
- Escriba `wav` como extensión en el archivo aunque el archivo real sea mp3.

Ejemplo:
```
foo.wav|maria|ES|Hola, ¿cómo estás?
style1/baz.wav|maria|ES|Hoy hace buen tiempo.
...
```
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("Cómo usar", open=False):
            gr.Markdown(how_to_md)
            with gr.Accordion(label="Preparación de datos", open=False):
                gr.Markdown(prepare_md)
        model_name = gr.Textbox(label="Nombre del modelo")
        gr.Markdown("### Preprocesamiento automático")
        with gr.Row(variant="panel"):
            with gr.Column():
                use_jp_extra = gr.Checkbox(
                    label="Usar versión Extra (Recomendado para Español, pierde EN/ZH)",
                    value=True,
                )
                batch_size = gr.Slider(
                    label="Tamaño del lote (Batch size)",
                    info="Si es lento, redúzcalo. Si hay VRAM de sobra, auméntelo. Uso VRAM aprox (Extra): 1: 6GB, 2: 8GB, 4: 12GB",
                    value=2,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs = gr.Slider(
                    label="Épocas",
                    info="100 suele ser suficiente, más puede mejorar la calidad",
                    value=100,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                save_every_steps = gr.Slider(
                    label="Guardar cada X pasos",
                    info="Nota: pasos, no épocas",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                normalize = gr.Checkbox(
                    label="Normalizar volumen del audio",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="Recortar silencio inicial y final",
                    value=False,
                )
                yomi_error = gr.Radio(
                    label="Manejo de archivos con transcripción ilegible",
                    choices=[
                        ("Detener si hay error", "raise"),
                        ("Saltar archivos ilegibles", "skip"),
                        ("Forzar uso (intentar leer)", "use"),
                    ],
                    value="skip",
                )
                with gr.Accordion("Configuración avanzada", open=False):
                    num_processes = gr.Slider(
                        label="Número de procesos",
                        info="Procesos paralelos en preprocesamiento. Reducir si se congela.",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    val_per_lang = gr.Slider(
                        label="Datos de validación",
                        info="Número de archivos para validación (no entrenamiento)",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    log_interval = gr.Slider(
                        label="Intervalo de log de Tensorboard",
                        info="Reducir para ver más detalles",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    gr.Markdown("Congelar partes del modelo durante entrenamiento")
                    freeze_EN_bert = gr.Checkbox(
                        label="Congelar BERT Inglés",
                        value=False,
                    )
                    freeze_JP_bert = gr.Checkbox(
                        label="Congelar BERT Japonés",
                        value=False,
                    )
                    freeze_ZH_bert = gr.Checkbox(
                        label="Congelar BERT Chino",
                        value=False,
                    )
                    freeze_ES_bert = gr.Checkbox(
                        label="Congelar BERT Español",
                        value=False,
                    )
                    freeze_style = gr.Checkbox(
                        label="Congelar Estilo",
                        value=False,
                    )
                    freeze_decoder = gr.Checkbox(
                        label="Congelar Decodificador",
                        value=False,
                    )

            with gr.Column():
                preprocess_button = gr.Button(
                    value="Ejecutar preprocesamiento automático", variant="primary"
                )
                info_all = gr.Textbox(label="Estado")
        with gr.Accordion(open=False, label="Preprocesamiento manual"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Paso 1: Generar configuración")
                    use_jp_extra_manual = gr.Checkbox(
                        label="Usar versión Extra",
                        value=True,
                    )
                    batch_size_manual = gr.Slider(
                        label="Tamaño del lote",
                        value=2,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    epochs_manual = gr.Slider(
                        label="Épocas",
                        value=100,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    save_every_steps_manual = gr.Slider(
                        label="Guardar cada X pasos",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                    )
                    log_interval_manual = gr.Slider(
                        label="Intervalo de log",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    freeze_EN_bert_manual = gr.Checkbox(
                        label="Congelar BERT Inglés",
                        value=False,
                    )
                    freeze_JP_bert_manual = gr.Checkbox(
                        label="Congelar BERT Japonés",
                        value=False,
                    )
                    freeze_ZH_bert_manual = gr.Checkbox(
                        label="Congelar BERT Chino",
                        value=False,
                    )
                    freeze_ES_bert_manual = gr.Checkbox(
                        label="Congelar BERT Español",
                        value=False,
                    )
                    freeze_style_manual = gr.Checkbox(
                        label="Congelar Estilo",
                        value=False,
                    )
                    freeze_decoder_manual = gr.Checkbox(
                        label="Congelar Decodificador",
                        value=False,
                    )
                with gr.Column():
                    generate_config_btn = gr.Button(value="Ejecutar", variant="primary")
                    info_init = gr.Textbox(label="Estado")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Paso 2: Preprocesamiento de audio")
                    num_processes_resample = gr.Slider(
                        label="Procesos",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    normalize_resample = gr.Checkbox(
                        label="Normalizar volumen",
                        value=False,
                    )
                    trim_resample = gr.Checkbox(
                        label="Recortar silencios",
                        value=False,
                    )
                with gr.Column():
                    resample_btn = gr.Button(value="Ejecutar", variant="primary")
                    info_resample = gr.Textbox(label="Estado")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Paso 3: Preprocesamiento de transcripción")
                    val_per_lang_manual = gr.Slider(
                        label="Datos de validación",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    yomi_error_manual = gr.Radio(
                        label="Manejo de errores de lectura",
                        choices=[
                            ("Detener", "raise"),
                            ("Saltar", "skip"),
                            ("Forzar", "use"),
                        ],
                        value="raise",
                    )
                with gr.Column():
                    preprocess_text_btn = gr.Button(value="Ejecutar", variant="primary")
                    info_preprocess_text = gr.Textbox(label="Estado")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Paso 4: Generar características BERT")
                with gr.Column():
                    bert_gen_btn = gr.Button(value="Ejecutar", variant="primary")
                    info_bert = gr.Textbox(label="Estado")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Paso 5: Generar características de estilo")
                    num_processes_style = gr.Slider(
                        label="Procesos",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                with gr.Column():
                    style_gen_btn = gr.Button(value="Ejecutar", variant="primary")
                    info_style = gr.Textbox(label="Estado")
        gr.Markdown("## Entrenamiento")
        with gr.Row():
            skip_style = gr.Checkbox(
                label="Saltar generación de archivos de estilo",
                info="Marcar si se reanuda el entrenamiento",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="Usar versión Extra",
                value=True,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="Desactivar muestreador personalizado",
                info="Permite audios largos (usa más VRAM)",
                value=False,
            )
            speedup = gr.Checkbox(
                label="Acelerar (experimental)",
                value=False,
                visible=False,
            )
            train_btn = gr.Button(value="Iniciar entrenamiento", variant="primary")
            tensorboard_btn = gr.Button(value="Abrir Tensorboard")
        gr.Markdown(
            "Verifique el progreso en la terminal. Los resultados se guardan periódicamente. Para detener, cierre la terminal o detenga el proceso."
        )
        info_train = gr.Textbox(label="Estado")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_ES_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                val_per_lang,
                log_interval,
                yomi_error,
            ],
            outputs=[info_all],
        )

        # Manual preprocess
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                freeze_EN_bert_manual,
                freeze_JP_bert_manual,
                freeze_ZH_bert_manual,
                freeze_ES_bert_manual,
                freeze_style_manual,
                freeze_decoder_manual,
                use_jp_extra_manual,
                log_interval_manual,
            ],
            outputs=[info_init],
        )
        resample_btn.click(
            second_elem_of(resample),
            inputs=[
                model_name,
                normalize_resample,
                trim_resample,
                num_processes_resample,
            ],
            outputs=[info_resample],
        )
        preprocess_text_btn.click(
            second_elem_of(preprocess_text),
            inputs=[
                model_name,
                use_jp_extra_manual,
                val_per_lang_manual,
                yomi_error_manual,
            ],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name],
            outputs=[info_bert],
        )
        style_gen_btn.click(
            second_elem_of(style_gen),
            inputs=[model_name, num_processes_style],
            outputs=[info_style],
        )

        # Train
        train_btn.click(
            second_elem_of(train),
            inputs=[
                model_name,
                skip_style,
                use_jp_extra_train,
                speedup,
                not_use_custom_batch_sampler,
            ],
            outputs=[info_train],
        )
        tensorboard_btn.click(
            run_tensorboard, inputs=[model_name], outputs=[tensorboard_btn]
        )

        use_jp_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra],
            outputs=[use_jp_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)

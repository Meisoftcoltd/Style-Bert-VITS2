import datetime
import json
from pathlib import Path
from typing import Optional

import gradio as gr

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers
from gradio_tabs.common import create_no_model_alert


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# Web UI での学習時の無駄な GPU VRAM 消費を避けるため、あえてここでは BERT モデルの事前ロードを行わない
# データセットの BERT 特徴量は事前に bert_gen.py により抽出されているため、学習時に BERT モデルをロードしておく必要はない
# BERT モデルの事前ロードは「ロード」ボタン押下時に実行される TTSModelHolder.get_model_for_gradio() 内で行われる
# Web UI での学習時、音声合成タブの「ロード」ボタンを押さなければ、BERT モデルが VRAM にロードされていない状態で学習を開始できる

languages = [lang.value for lang in Languages]

initial_text = "Hola, encantado de conocerte. ¿Cómo te llamas?"

examples = [
    [initial_text, "ES"],
    [
        """Me alegra mucho que digas eso.
Me enfada mucho que digas eso.
Me sorprende mucho que digas eso.
Me duele mucho que digas eso.""",
        "ES",
    ],
    [  # Ejemplo de confesión
        """Te he estado observando desde hace mucho tiempo. Tu sonrisa, tu amabilidad, tu fuerza... me han cautivado.
Al pasar tiempo contigo como amigos, me di cuenta de que te estabas convirtiendo en alguien especial.
Esto... ¡Me gustas! Si te parece bien, ¿te gustaría salir conmigo?""",
        "ES",
    ],
    [  # Don Quijote
        """En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.
Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.""",
        "ES",
    ],
    [  # Cien años de soledad
        """Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano Buendía había de recordar aquella tarde remota en que su padre lo llevó a conocer el hielo.
Macondo era entonces una aldea de veinte casas de barro y cañabrava construidas a la orilla de un río de aguas diáfanas que se precipitaban por un lecho de piedras pulidas, blancas y enormes como huevos prehistóricos.""",
        "ES",
    ],
    [  # Emociones
        """¡Lo logré! ¡Saqué la máxima nota en el examen! ¡Estoy tan feliz!
¿Por qué ignoras mi opinión? ¡Es imperdonable! ¡Me molesta mucho! Ojalá te mueras.
¡Jajaja! Este manga es muy gracioso, mira esto, jiji, jaja.
Ahora que te has ido, me he quedado sola, estoy tan triste que podría llorar.""",
        "ES",
    ],
    [  # Explicación técnica
        """La síntesis de voz es una tecnología que utiliza el aprendizaje automático para reproducir la voz humana a partir de texto. Esta tecnología analiza la estructura del lenguaje y genera voz en base a ella.
Utilizando los últimos resultados de investigación en este campo, es posible generar una voz más natural y expresiva. Mediante la aplicación del aprendizaje profundo, también es posible reproducir cambios sutiles en la calidad de la voz, incluyendo emociones y acentos.""",
        "ES",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
        "ZH",
    ],
    [
        "こんにちは、初めまして。あなたの名前はなんていうの？",
        "JP",
    ],
]

initial_md = """
- Los modelos predeterminados [`koharune-ami` (Koharu Ami)](https://huggingface.co/litagin/sbv2_koharune_ami) y [`amitaro` (Amitaro)](https://huggingface.co/litagin/sbv2_amitaro) añadidos en la Ver 2.5 fueron entrenados con permiso previo utilizando corpus de audio y transmisiones en vivo de [Amitaro's Voice Material Workshop](https://amitaro.net/). Por favor, asegúrese de **leer los términos de uso** a continuación antes de usarlos.

- Para descargar los modelos anteriores después de la actualización de la Ver 2.5, haga doble clic en `Initialize.bat` o descárguelos manualmente y colóquelos en el directorio `model_assets`.

- La **versión editor** añadida en la Ver 2.3 puede ser más fácil de usar para la lectura real. Puede iniciarla con `Editor.bat` o `python server_editor.py --inbrowser`.
"""

terms_of_use_md = """
## Licencia de los modelos predeterminados y peticiones

Consulte [aquí](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md) para ver las últimas peticiones y términos de uso. Siempre se aplica la versión más reciente.

Al usar Style-Bert-VITS2, le agradeceríamos que respetara las siguientes peticiones. Sin embargo, las partes anteriores a los términos de uso del modelo son solo "peticiones" y no tienen fuerza obligatoria, y no son los términos de uso de Style-Bert-VITS2. Por lo tanto, no contradicen la [licencia del repositorio](https://github.com/litagin02/Style-Bert-VITS2#license), y solo la licencia del repositorio tiene fuerza vinculante para el uso del repositorio.

### Lo que no queremos que haga

No queremos que use Style-Bert-VITS2 para los siguientes propósitos:

- Propósitos que violen la ley
- Propósitos políticos (prohibido en el Bert-VITS2 original)
- Propósitos para dañar a otros
- Propósitos de suplantación o creación de deepfakes

### Lo que nos gustaría que hiciera

- Al usar Style-Bert-VITS2, asegúrese de verificar los términos de uso y la licencia del modelo que está utilizando y sígalos si existen.
- Además, al utilizar el código fuente, siga la [licencia del repositorio](https://github.com/litagin02/Style-Bert-VITS2#license).

A continuación se muestran las licencias de los modelos incluidos por defecto.

### Corpus JVNV (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- La licencia del [Corpus JVNV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) es [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja), por lo que se hereda.

### Koharu Ami (koharune-ami) / Amitaro (amitaro)

Debe cumplir con todos los [términos de Amitaro's Voice Material Workshop](https://amitaro.net/voice/voice_rule/) y [Términos de uso de voz de transmisión en vivo de Amitaro](https://amitaro.net/voice/livevoice/#index_id6). En particular, cumpla con lo siguiente (puede usarlo comercial o no comercialmente si sigue las reglas):

#### Prohibiciones

- Uso en obras o propósitos con restricción de edad
- Obras o propósitos profundamente relacionados con nuevas religiones, política, esquemas multinivel, etc.
- Obras o propósitos que calumnien a grupos, individuos o naciones específicos
- Tratar la voz generada como la voz de Amitaro
- Tratar la voz generada como la voz de alguien que no sea Amitaro

#### Créditos

Al publicar el audio generado (independientemente del medio), asegúrese de incluir una notación de crédito en un lugar visible que indique que está utilizando un modelo de voz basado en la voz de `Amitaro's Voice Material Workshop (https://amitaro.net/)`.

Ejemplo de crédito:
- `Modelo Style-BertVITS2: Koharu Ami, Amitaro's Voice Material Workshop (https://amitaro.net/)`
- `Modelo Style-BertVITS2: Amitaro, Amitaro's Voice Material Workshop (https://amitaro.net/)`

#### Fusión de modelos

Para la fusión de modelos, cumpla con [la respuesta a las preguntas frecuentes de Amitaro's Voice Material Workshop](https://amitaro.net/voice/faq/#index_id17):
- Este modelo solo se puede fusionar con otro modelo si el titular de los derechos de la voz utilizada para entrenar ese otro modelo ha dado permiso.
- Si las características de la voz de Amitaro permanecen (si la proporción de fusión es del 25% o más), el uso se limita al alcance de los [términos de Amitaro's Voice Material Workshop](https://amitaro.net/voice/voice_rule/), y estos términos también se aplican a ese modelo.
"""

how_to_md = """
Coloque los archivos del modelo en el directorio `model_assets` como se muestra a continuación.
```
model_assets
├── su_modelo
│   ├── config.json
│   ├── archivo_de_modelo1.safetensors
│   ├── archivo_de_modelo2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── otro_modelo
    ├── ...
```
Cada modelo necesita los siguientes archivos:
- `config.json`: Archivo de configuración durante el entrenamiento
- `*.safetensors`: Archivo del modelo entrenado (se requiere al menos uno, pueden ser varios)
- `style_vectors.npy`: Archivo de vectores de estilo

Los dos primeros se guardan automáticamente en la posición correcta mediante el entrenamiento con `Train.bat`. Genere `style_vectors.npy` ejecutando `Style.bat` y siguiendo las instrucciones.
"""

style_md = f"""
- Puede controlar el tono, la emoción y el estilo de la lectura desde preajustes o archivos de audio.
- Incluso con el {DEFAULT_STYLE} predeterminado, la lectura es lo suficientemente expresiva según el texto. Este control de estilo es como sobrescribirlo con un peso.
- Si la intensidad es demasiado alta, la pronunciación puede volverse extraña o la voz puede romperse.
- La intensidad adecuada parece variar según el modelo y el estilo.
- Si ingresa un archivo de audio, es posible que no obtenga un buen efecto a menos que el hablante tenga un tono similar a los datos de entrenamiento (especialmente el mismo género).
"""
voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]


def make_interactive():
    return gr.update(interactive=True, value="Síntesis de voz")


def make_non_interactive():
    return gr.update(interactive=False, value="Síntesis de voz (Cargue un modelo)")


def gr_util(item):
    if item == "Seleccionar de preajuste":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


null_models_frame = 0


def change_null_model_row(
    null_model_index: int,
    null_model_name: str,
    null_model_path: str,
    null_voice_weights: float,
    null_voice_pitch_weights: float,
    null_speech_style_weights: float,
    null_tempo_weights: float,
    null_models: dict[int, NullModelParam],
):
    null_models[null_model_index] = NullModelParam(
        name=null_model_name,
        path=Path(null_model_path),
        weight=null_voice_weights,
        pitch=null_voice_pitch_weights,
        style=null_speech_style_weights,
        tempo=null_tempo_weights,
    )
    if len(null_models) > null_models_frame:
        keys_to_keep = list(range(null_models_frame))
        result = {k: null_models[k] for k in keys_to_keep}
    else:
        result = null_models
    return result, True


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
        text,
        language,
        reference_audio_path,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        line_split,
        split_interval,
        assist_text,
        assist_text_weight,
        use_assist_text,
        style,
        style_weight,
        kata_tone_json_str,
        use_tone,
        speaker,
        pitch_scale,
        intonation_scale,
        null_models: dict[int, NullModelParam],
        force_reload_model: bool,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None
        logger.debug(f"Null models setting: {null_models}")

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "La generación de acentos actualmente solo es compatible con japonés."
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "La generación de acentos solo es compatible cuando no se divide por saltos de línea."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"La especificación de acento es inválida: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = model_holder.current_model.spk2id[speaker]

        start_time = datetime.datetime.now()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise_scale,
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
                null_model_params=null_models,
                force_reload_model=force_reload_model,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Error: La especificación de acento es inválida:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Éxito, tiempo: {duration} segundos."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str, False

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
        gr.Markdown(terms_of_use_md)
        null_models = gr.State({})
        force_reload_model = gr.State(False)
        with gr.Accordion(label="Cómo usar", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
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
                    refresh_button = gr.Button("Actualizar", scale=1, visible=True)
                    load_button = gr.Button("Cargar", scale=1, variant="primary")
                text_input = gr.TextArea(label="Texto", value=initial_text)
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="Tono (1 es normal)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="Entonación (1 es normal)",
                )

                line_split = gr.Checkbox(
                    label="Dividir por saltos de línea (mejora la emoción)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="Silencio entre líneas (s)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="Ajuste de acento (solo JP)",
                    info="Solo disponible si no se divide por saltos de línea. No es infalible.",
                )
                use_tone = gr.Checkbox(label="Usar ajuste de acento", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="ES", label="Idioma")
                speaker = gr.Dropdown(label="Hablante")
                with gr.Accordion(label="Configuración avanzada", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Ruido",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Ruido_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Velocidad (Length)",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Usar texto de asistencia", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Texto de asistencia",
                        placeholder="Texto para imitar estilo...",
                        info="El tono y la emoción se parecerán a este texto. Puede afectar la entonación y el tempo.",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Intensidad del texto de asistencia",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
                    )
                with gr.Accordion(label="Modelo Nulo", open=False):
                    with gr.Row():
                        null_models_count = gr.Number(
                            label="Número de modelos nulos", value=0, step=1
                        )
                    with gr.Column(variant="panel"):

                        @gr.render(inputs=[null_models_count])
                        def render_null_models(
                            null_models_count: int,
                        ):
                            global null_models_frame
                            null_models_frame = null_models_count
                            for i in range(null_models_count):
                                with gr.Row():
                                    null_model_index = gr.Number(
                                        value=i,
                                        key=f"null_model_index_{i}",
                                        visible=False,
                                    )
                                    null_model_name = gr.Dropdown(
                                        label="Lista de modelos",
                                        choices=model_names,
                                        key=f"null_model_name_{i}",
                                        value=model_names[initial_id],
                                    )
                                    null_model_path = gr.Dropdown(
                                        label="Archivo del modelo",
                                        key=f"null_model_path_{i}",
                                        # FIXME: 再レンダー時に選択肢が消えるのでどうにかしたい
                                        # 現在は再レンダーでvalueは保存されるが選択肢は保存されないので選択肢が空になる
                                        # そのときに選択肢にない値となるので、それを許す
                                        allow_custom_value=True,
                                    )
                                    null_voice_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_weights_{i}",
                                        label="Voz",
                                    )
                                    null_voice_pitch_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_pitch_weights_{i}",
                                        label="Tono",
                                    )
                                    null_speech_style_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_speech_style_weights_{i}",
                                        label="Estilo",
                                    )
                                    null_tempo_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_tempo_weights_{i}",
                                        label="Tempo",
                                    )

                                    null_model_name.change(
                                        model_holder.update_model_files_for_gradio,
                                        inputs=[null_model_name],
                                        outputs=[null_model_path],
                                    )
                                    null_model_path.change(
                                        make_non_interactive, outputs=[tts_button]
                                    )
                                    # 愚直すぎるのでもう少しなんとかしたい
                                    null_model_path.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_pitch_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_speech_style_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_tempo_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )

                    add_btn = gr.Button("Añadir modelo nulo")
                    del_btn = gr.Button("Quitar modelo nulo")
                    add_btn.click(
                        lambda x: x + 1,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )
                    del_btn.click(
                        lambda x: x - 1 if x > 0 else 0,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )

            with gr.Column():
                with gr.Accordion("Detalles sobre estilos", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["Seleccionar de preajuste", "Ingresar archivo de audio"],
                    label="Método de especificación de estilo",
                    value="Seleccionar de preajuste",
                )
                style = gr.Dropdown(
                    label=f"Estilo ({DEFAULT_STYLE} es el estilo promedio)",
                    choices=["Por favor cargue un modelo"],
                    value="Por favor cargue un modelo",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="Intensidad del estilo (reducir si la voz se distorsiona)",
                )
                ref_audio_path = gr.Audio(
                    label="Audio de referencia", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "Síntesis de voz (Cargue un modelo)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="Información")
                audio_output = gr.Audio(label="Resultado")
                with gr.Accordion("Ejemplos de texto", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
                pitch_scale,
                intonation_scale,
                null_models,
                force_reload_model,
            ],
            outputs=[text_output, audio_output, tone, force_reload_model],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    import torch

    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(
        assets_root, device, torch_device_to_onnx_providers(device)
    )
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)

# Style-Bert-VITS2

**Por favor, asegúrese de leer los [Términos de uso y las Peticiones](/docs/TERMS_OF_USE.md) antes de usar.**

Bert-VITS2 con estilos de voz más controlables.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/e853f9a2-db4a-4202-a1dd-56ded3c562a0

Puede instalarlo vía `pip install style-bert-vits2` (solo inferencia), vea [library.ipynb](/library.ipynb) para ejemplos de uso.

- **Video Tutorial** [YouTube](https://youtu.be/aTUSzgDl1iY)　[NicoNico](https://www.nicovideo.jp/watch/sm43391524)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- [**Preguntas frecuentes** (FAQ)](/docs/FAQ.md)
- [🤗 Demo en línea aquí](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)
- [Artículo explicativo en Zenn](https://zenn.dev/litagin/articles/034819a5256ff4)

- [**Página de lanzamientos**](https://github.com/litagin02/Style-Bert-VITS2/releases/)、[Historial de actualizaciones](/docs/CHANGELOG.md)
  - 2025-08-24: Ver 2.7.0: Se añadió GUI para conversión a ONNX para integración con bibliotecas externas como [Aivis Project](https://aivis-project.com/), y se añadió `litagin/anime-whisper` como modelo de reconocimiento de voz.
  - 2024-09-09: Ver 2.6.1: Corrección de errores en Google Colab, etc.
  - 2024-06-16: Ver 2.6.0 (Añadida fusión de diferencias de modelos, fusión ponderada, fusión de modelos nulos. Vea [este artículo](https://zenn.dev/litagin/articles/1297b1dc7bdc79) para usos).
  - 2024-06-14: Ver 2.5.1 (Cambio de términos de uso a peticiones).
  - 2024-06-02: Ver 2.5.0 (**[Añadidos Términos de Uso](/docs/TERMS_OF_USE.md)**, generación de estilos desde carpetas, adición de modelos Koharu Ami y Amitaro, instalación más rápida, etc.).
  - 2024-03-16: ver 2.4.1 (**Cambio en el método de instalación mediante archivos bat**).
  - 2024-03-15: ver 2.4.0 (Refactorización a gran escala y varias mejoras, conversión a librería).
  - 2024-02-26: ver 2.3 (Funciones de diccionario y editor).
  - 2024-02-09: ver 2.2
  - 2024-02-07: ver 2.1
  - 2024-02-03: ver 2.0 (JP-Extra)
  - 2024-01-09: ver 1.3
  - 2023-12-31: ver 1.2
  - 2023-12-29: ver 1.1
  - 2023-12-27: ver 1.0

Este repositorio se basa en [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 y Japanese-Extra, ¡muchas gracias al autor original!

**Resumen**

- Basado en Bert-VITS2 v2.1 y Japanese-Extra, que genera voz expresiva basada en el contenido del texto de entrada, permitiendo controlar libremente la emoción y el estilo de habla con intensidad.
- Incluso si no tiene Git o Python (para usuarios de Windows), puede instalar y entrenar fácilmente (tomado en gran parte de [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2/)). También soporta entrenamiento en Google Colab.
- Si solo lo usa para síntesis de voz, funciona en CPU sin tarjeta gráfica.
- Para síntesis de voz, se puede instalar como librería Python con `pip install style-bert-vits2`. Vea [library.ipynb](/library.ipynb) para ejemplos.
- Incluye un servidor API que se puede usar para integración con otras herramientas (PR por [@darai0512](https://github.com/darai0512), gracias).
- La fortaleza de Bert-VITS2 es "leer textos alegres con alegría y textos tristes con tristeza", por lo que puede generar voz expresiva incluso con el estilo predeterminado.


## Cómo usar

- Para uso en CLI, consulte [aquí](/docs/CLI.md).
- Consulte también las [Preguntas frecuentes](/docs/FAQ.md).

### Entorno de ejecución

Se ha confirmado el funcionamiento de cada UI y API Server en Símbolo del sistema de Windows, WSL2 y Linux (Ubuntu Desktop). Si no tiene una GPU NVidia, no puede entrenar, pero puede realizar síntesis de voz y fusión.

### Instalación

Consulte [library.ipynb](/library.ipynb) para la instalación y uso como librería Python con pip.

#### Para quienes no están familiarizados con Git o Python

Se asume Windows.

1. Descargue [este archivo zip](https://github.com/litagin02/Style-Bert-VITS2/releases/latest/download/sbv2.zip) y extráigalo en una ubicación **sin espacios ni caracteres japoneses (o especiales) en la ruta**.
  - Si tiene tarjeta gráfica, haga doble clic en `Install-Style-Bert-VITS2.bat`.
  - Si no tiene tarjeta gráfica, haga doble clic en `Install-Style-Bert-VITS2-CPU.bat`. La versión CPU no permite entrenamiento, solo síntesis y fusión.
2. Espere a que se instale el entorno necesario automáticamente.
3. Si el editor de síntesis de voz se inicia automáticamente, la instalación fue exitosa. Los modelos predeterminados se descargan, así que puede jugar con ellos de inmediato.

Si desea actualizar, haga doble clic en `Update-Style-Bert-VITS2.bat`.

Sin embargo, si actualiza desde una versión anterior a **2.4.1** (2024-03-16), debe eliminar todo e instalar de nuevo. Disculpe las molestias. Consulte [CHANGELOG.md](/docs/CHANGELOG.md) para la migración.

#### Para quienes saben usar Git y Python

Se recomienda usar [uv](https://github.com/astral-sh/uv), una herramienta de gestión de paquetes y entornos virtuales de Python más rápida que pip.
(Si no desea usarlo, pip normal está bien).

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv venv venv
venv\Scripts\activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py  # Descarga modelos necesarios y el modelo TTS predeterminado
```
No olvide el último paso.

### Síntesis de voz

El editor de síntesis de voz se inicia haciendo doble clic en `Editor.bat` o ejecutando `python server_editor.py --inbrowser` (use `--device cpu` para modo CPU). En la pantalla puede crear guiones cambiando la configuración para cada línea, guardar, cargar y editar diccionarios.
Los modelos predeterminados se descargan al instalar, por lo que puede usarlos sin entrenar.

La parte del editor está separada en [otro repositorio](https://github.com/litagin02/Style-Bert-VITS2-Editor).

Para la WebUI de síntesis de voz de versiones anteriores a 2.2, haga doble clic en `App.bat` o ejecute `python app.py`. También puede abrir solo la pestaña de síntesis con `Inference.bat`.

La estructura de archivos del modelo necesaria para la síntesis es la siguiente (no necesita colocarla manualmente):
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
Para la inferencia se necesitan `config.json`, `*.safetensors` y `style_vectors.npy`. Si comparte modelos, comparta estos 3 archivos.

`style_vectors.npy` es necesario para controlar el estilo. Al entrenar, se genera un estilo promedio "Neutral" por defecto.
Si desea controlar el estilo con más detalle usando múltiples estilos, consulte "Generación de estilos" a continuación.

### Entrenamiento

- Para detalles de entrenamiento en CLI, consulte [aquí](docs/CLI.md).
- Para entrenamiento en Paperspace consulte [aquí](docs/paperspace.md), y en Colab [aquí](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb).

Para el entrenamiento se necesitan varios archivos de audio de 2-14 segundos y sus datos de transcripción.

- Si ya tiene archivos de audio divididos y datos de transcripción (como un corpus existente), puede usarlos tal cual (corrigiendo el archivo de transcripción si es necesario). Consulte "WebUI de Entrenamiento" abajo.
- Si no, y solo tiene archivos de audio (de cualquier longitud), se incluye una herramienta para crear un conjunto de datos listo para entrenar.

#### Creación de conjunto de datos

- Desde la pestaña "Crear Dataset" en la WebUI (`App.bat` o `python app.py`), puede dividir archivos de audio en longitudes adecuadas y transcribirlos automáticamente. O use `Dataset.bat` para abrir esa pestaña sola.
- Después de seguir las instrucciones, puede entrenar directamente en la pestaña "Entrenamiento".

#### WebUI de Entrenamiento

- Siga las instrucciones en la pestaña "Entrenamiento" de la WebUI (`App.bat` o `python app.py`). O use `Train.bat`.

### Generación de estilos

- Por defecto, se genera el estilo "Neutral" y estilos basados en las subcarpetas de la carpeta de entrenamiento.
- Esto es para quienes quieren crear estilos manualmente de otras formas.
- Desde la pestaña "Crear Estilos" de la WebUI (`App.bat` o `python app.py`), puede generar estilos usando archivos de audio. O use `StyleVectors.bat`.
- Es independiente del entrenamiento, por lo que puede hacerlo durante o después del entrenamiento tantas veces como quiera (el preprocesamiento debe haber terminado).

### API Server

Ejecute `python server_fastapi.py` en el entorno construido para iniciar el servidor API.
Verifique la especificación de la API en `/docs` después de iniciar.

- El límite de caracteres de entrada es 100 por defecto. Esto se puede cambiar en `server.limit` de `config.yml`.
- Por defecto, CORS está permitido para todos los dominios. Cambie `server.origins` en `config.yml` para restringirlo a dominios confiables si es posible.

El servidor API del editor de síntesis de voz se inicia con `python server_editor.py`. Aún no está muy desarrollado y solo implementa lo mínimo necesario para el [repositorio del editor](https://github.com/litagin02/Style-Bert-VITS2-Editor).

Para el despliegue web del editor, consulte [este Dockerfile](Dockerfile.deploy).

### Fusión (Merge)

Puede mezclar dos modelos en términos de "calidad de voz", "tono", "expresión emocional" y "tempo" para crear un nuevo modelo, o "sumar la diferencia de otros dos modelos a un modelo", etc.
Desde la pestaña "Fusión" de la WebUI (`App.bat` o `python app.py`), puede seleccionar y fusionar modelos. O use `Merge.bat`.

### Conversión ONNX

Desde la pestaña "Conversión ONNX" o `ConvertONNX.bat`, puede convertir archivos safetensors entrenados a formato ONNX. Esto es útil si necesita archivos ONNX para librerías externas. Por ejemplo, en [Aivis Project](https://aivis-project.com/) puede usar [AIVM Generator](https://aivm-generator.aivis-project.com/) para crear modelos para Aivis Speech.

### Evaluación de naturalidad

Se proporciona un script usando [SpeechMOS](https://github.com/tarepan/SpeechMOS) como un indicador para elegir el mejor paso de entrenamiento:
```bash
python speech_mos.py -m <nombre_del_modelo>
```
Se mostrará la evaluación de naturalidad por paso y se guardarán los resultados en `mos_results/mos_{nombre_modelo}.csv` y `.png`. Es solo una referencia que no considera acento o emoción, así que lo mejor es escuchar y seleccionar.

## Relación con Bert-VITS2

Básicamente es una ligera modificación de la estructura del modelo Bert-VITS2. Tanto el [modelo pre-entrenado antiguo](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) como el [modelo pre-entrenado JP-Extra](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) son prácticamente iguales a Bert-VITS2 v2.1 o JP-Extra (con pesos innecesarios eliminados y convertidos a safetensors).

Las diferencias específicas son:

- Fácil de usar para quienes no saben Python o Git, como [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2).
- Cambio del modelo de incrustación de emociones (a [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) de 256 dimensiones, más una incrustación de identificación de hablante que de emoción).
- Se eliminó la cuantización vectorial de la incrustación de emociones, dejándola como una capa totalmente conectada.
- Al crear el archivo de vectores de estilo `style_vectors.npy`, se puede generar voz especificando continuamente la intensidad del estilo.
- Creación de varias WebUI.
- Soporte para entrenamiento en bf16.
- Soporte para formato safetensors, uso predeterminado.
- Otras correcciones de errores menores y refactorización.


## Referencias
Además de la referencia original (abajo), utilicé los siguientes repositorios:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[El modelo pre-entrenado](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) y la [versión JP-Extra](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) son esencialmente tomados del [modelo base original de Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) y [modelo pre-entrenado JP-Extra de Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra), así que todos los créditos van al autor original ([Fish Audio](https://github.com/fishaudio)):


Además, el módulo [text/user_dict/](text/user_dict) se basa en:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
y la licencia de este módulo es LGPL v3.

## LICENCIA

Este repositorio está licenciado bajo la GNU Affero General Public License v3.0, igual que el repositorio original de Bert-VITS2. Para más detalles, vea [LICENSE](LICENSE).

Además, el módulo [text/user_dict/](text/user_dict) está licenciado bajo la GNU Lesser General Public License v3.0, heredado del repositorio original de VOICEVOX engine. Para más detalles, vea [LGPL_LICENSE](LGPL_LICENSE).



Abajo está el README.md original.
---

<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Bert-VITS2

VITS2 Backbone with multilingual bert

For quick guide, please refer to `webui_preprocess.py`.

简易教程请参见 `webui_preprocess.py`。

## 请注意，本项目核心思路来源于[anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS) 一个非常好的tts项目
## MassTTS的演示demo为[ai版峰哥锐评峰哥本人,并找回了在金三角失落的腰子](https://www.bilibili.com/video/BV1w24y1c7z9)

[//]: # (## 本项目与[PlayVoice/vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41; 没有任何关系)

[//]: # ()
[//]: # (本仓库来源于之前朋友分享了ai峰哥的视频，本人被其中的效果惊艳，在自己尝试MassTTS以后发现fs在音质方面与vits有一定差距，并且training的pipeline比vits更复杂，因此按照其思路将bert)

## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应当参阅代码自己学习如何训练。

### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
#### QQ Group：815818430
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [emotional-vits](https://github.com/innnky/emotional-vits)
+ [fish-speech](https://github.com/fishaudio/fish-speech)
+ [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)
## 感谢所有贡献者作出的努力
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>

[//]: # (# 本项目所有代码引用均已写明，bert部分代码思路来源于[AI峰哥]&#40;https://www.bilibili.com/video/BV1w24y1c7z9&#41;，与[vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41;无任何关系。欢迎各位查阅代码。同时，我们也对该开发者的[碰瓷，乃至开盒开发者的行为]&#40;https://www.bilibili.com/read/cv27101514/&#41;表示强烈谴责。)

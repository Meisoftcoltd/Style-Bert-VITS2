"""
TODO:
importが重いので、WebUI全般が重くなっている。どうにかしたい。
"""

import json
import shutil
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from umap import UMAP

from config import get_path_config
from default_style import save_styles_by_dirs
from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger


path_config = get_path_config()
dataset_root = path_config.dataset_root
assets_root = path_config.assets_root

MAX_CLUSTER_NUM = 10
MAX_AUDIO_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
umap = UMAP(n_components=2, random_state=42, metric="cosine", n_jobs=1, min_dist=0.0)

wav_files: list[Path] = []
x = np.array([])
x_reduced = None
y_pred = np.array([])
mean = np.array([])
centroids = []


def load(model_name: str, reduction_method: str):
    global wav_files, x, x_reduced, mean
    wavs_dir = dataset_root / model_name / "wavs"
    style_vector_files = [f for f in wavs_dir.rglob("*.npy") if f.is_file()]
    # foo.wav.npy -> foo.wav
    wav_files = [f.with_suffix("") for f in style_vector_files]
    logger.info(f"Found {len(style_vector_files)} style vectors in {wavs_dir}")
    style_vectors = [np.load(f) for f in style_vector_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)
    if reduction_method == "t-SNE":
        x_reduced = tsne.fit_transform(x)
    elif reduction_method == "UMAP":
        x_reduced = umap.fit_transform(x)
    else:
        raise ValueError("Invalid reduction method")
    x_reduced = np.asarray(x_reduced)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids, x_reduced, y_pred
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after reduction":
        assert x_reduced is not None
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_reduced)
    elif method == "Agglomerative after reduction":
        assert x_reduced is not None
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_reduced)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def do_dbscan(eps=2.5, min_samples=15):
    global centroids, x_reduced, y_pred
    model = DBSCAN(eps=eps, min_samples=min_samples)
    assert x_reduced is not None
    y_pred = model.fit_predict(x_reduced)
    n_clusters = max(y_pred) + 1
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))
    return y_pred, centroids


def representative_wav_files(cluster_id, num_files=1):
    # Find medoid for cluster_index in y_pred
    cluster_indices = np.where(y_pred == cluster_id)[0]
    cluster_vectors = x[cluster_indices]

    distances = pdist(cluster_vectors)
    distance_matrix = squareform(distances)

    mean_distances = distance_matrix.mean(axis=1)

    closest_indices = np.argsort(mean_distances)[:num_files]

    return cluster_indices[closest_indices]


def do_dbscan_gradio(eps=2.5, min_samples=15):
    global x_reduced, centroids

    y_pred, centroids = do_dbscan(eps, min_samples)

    assert x_reduced is not None

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(max(y_pred) + 1):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    # Noise cluster (-1) is black
    plt.scatter(
        x_reduced[y_pred == -1, 0],
        x_reduced[y_pred == -1, 1],
        color="black",
        label="Noise",
    )
    plt.legend()

    n_clusters = int(max(y_pred) + 1)

    if n_clusters > MAX_CLUSTER_NUM:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"Demasiados clústeres, intente cambiar los parámetros: {n_clusters}",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    elif n_clusters == 0:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            "El número de clústeres es 0. Intente cambiar los parámetros.",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    return [plt, gr.Slider(maximum=n_clusters, value=1), n_clusters] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def representative_wav_files_gradio(cluster_id, num_files=1):
    cluster_id = cluster_id - 1
    closest_indices = representative_wav_files(cluster_id, num_files)
    actual_num_files = len(closest_indices)
    return [
        gr.Audio(wav_files[i], visible=True, label=str(wav_files[i]))
        for i in closest_indices
    ] + [gr.update(visible=False)] * (MAX_AUDIO_NUM - actual_num_files)


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_reduced, centroids
    y_pred, centroids = do_clustering(n_clusters, method)

    assert x_reduced is not None
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    plt.legend()

    return [plt, gr.Slider(maximum=n_clusters, value=1)] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def save_style_vectors_from_clustering(model_name: str, style_names_str: str):
    """centerとcentroidsを保存する"""
    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)
    logger.success(f"Saved style vectors to {style_vector_path}")

    # config.jsonの更新
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} no existe."
    style_names = [name.strip() for name in style_names_str.split(",")]
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(style_name_list) != len(centroids) + 1:
        return f"El número de estilos no coincide. Verifique que estén separados por `,` y sean {len(centroids)}: {style_names_str}"
    if len(set(style_names)) != len(style_names):
        return "Los nombres de estilo están duplicados."

    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.success(f"Updated {config_path}")
    return f"¡Éxito!\nGuardado en {style_vector_path} y actualizado {config_path}."


def save_style_vectors_from_files(
    model_name: str, audio_files_str: str, style_names_str: str
):
    """音声ファイルからスタイルベクトルを作成して保存する"""
    global mean
    if len(x) == 0:
        return "Error: Por favor cargue los vectores de estilo."
    mean = np.mean(x, axis=0)

    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [name.strip() for name in audio_files_str.split(",")]
    style_names = [name.strip() for name in style_names_str.split(",")]
    if len(audio_files) != len(style_names):
        return f"El número de archivos de audio y nombres de estilo no coincide. Verifique que estén separados por `,`: {audio_files_str} y {style_names_str}"
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(set(style_names)) != len(style_names):
        return "Los nombres de estilo están duplicados."
    style_vectors = [mean]

    wavs_dir = dataset_root / model_name / "wavs"
    for audio_file in audio_files:
        path = wavs_dir / audio_file
        if not path.exists():
            return f"{path} no existe."
        style_vectors.append(np.load(f"{path}.npy"))
    style_vectors = np.stack(style_vectors)
    assert len(style_name_list) == len(style_vectors)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)

    # config.jsonの更新
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} no existe."
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"¡Éxito!\nGuardado en {style_vector_path} y actualizado {config_path}."


def save_style_vectors_by_dirs(model_name: str, audio_dir_str: str):
    if model_name == "":
        return "Ingrese el nombre del modelo."
    if audio_dir_str == "":
        return "Ingrese el directorio que contiene los archivos de audio."

    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    from tqdm import tqdm

    from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
    from style_gen import save_style_vector

    # First generate style vectors for each audio file

    audio_dir = Path(audio_dir_str)
    audio_suffixes = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    audio_files = [f for f in audio_dir.rglob("*") if f.suffix in audio_suffixes]

    def process(file: Path):
        # f: `test.wav` -> search `test.wav.npy`
        if (file.with_name(file.name + ".npy")).exists():
            return file, None
        try:
            save_style_vector(str(file))
        except Exception as e:
            return file, e
        return file, None

    with ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
        _ = list(
            tqdm(
                executor.map(
                    process,
                    audio_files,
                ),
                total=len(audio_files),
                file=SAFE_STDOUT,
                desc="Generating style vectors",
                dynamic_ncols=True,
            )
        )

    result_dir = assets_root / model_name
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path} no existe."
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    save_styles_by_dirs(
        wav_dir=audio_dir,
        output_dir=result_dir,
        config_path=config_path,
        config_output_path=config_path,
    )
    return f"¡Éxito!\nSe guardaron los vectores de estilo en {result_dir}."


how_to_md = f"""
Para especificar estilos detallados en Style-Bert-VITS2, es necesario crear un archivo de vectores de estilo `style_vectors.npy` para cada modelo.

Sin embargo, durante el proceso de entrenamiento, automáticamente se guarda el estilo promedio "{DEFAULT_STYLE}" y (**desde la Ver 2.5.0**) los estilos para cada subcarpeta si los audios estaban divididos en subcarpetas.

## Métodos

- Método 0: Dividir los audios en subcarpetas por estilo y crear vectores de estilo para cada carpeta.
- Método 1: Dividir automáticamente los archivos de audio por estilo, tomar el promedio de cada estilo y guardarlo.
- Método 2: Seleccionar manualmente archivos de audio representativos de un estilo y guardar sus vectores de estilo.
- Método 3: Crear manualmente con más cuidado (si ya existen etiquetas de estilo como en el corpus JVNV, esto puede ser mejor).
"""

method0 = """
Cree subcarpetas para cada estilo y coloque los archivos de audio dentro.

**Nota**

- Desde la Ver 2.5.0, si coloca los archivos de audio en subdirectorios dentro de `inputs/` o `raw/`, los vectores de estilo se crean automáticamente, por lo que este paso no es necesario.
- Úselo si desea agregar nuevos vectores de estilo a un modelo entrenado en una versión anterior, o si desea crear vectores de estilo con audios diferentes a los usados en el entrenamiento.
- Para mantener la consistencia con el entrenamiento, si **está entrenando o planea entrenar**, guarde los archivos de audio en **un directorio nuevo y separado**, no en la carpeta `Data/{nombre_del_modelo}/wavs`.

Ejemplo:

```bash
audio_dir
├── style1
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── style2
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```
"""

method1 = f"""
Cargue los vectores de estilo extraídos durante el entrenamiento y divida los estilos observando la visualización.

Pasos:
1. Observar el gráfico.
2. Decidir el número de estilos (excluyendo el estilo promedio).
3. Ejecutar la clasificación de estilos y verificar el resultado.
4. Nombrar los estilos y guardar.


Detalle: Agrupa los vectores de estilo (256 dimensiones) con un algoritmo adecuado y guarda el vector central de cada grupo (y el vector promedio general).

El estilo promedio ({DEFAULT_STYLE}) se guarda automáticamente.
"""

dbscan_md = """
Realiza la clasificación de estilos utilizando el método DBSCAN.
Esto puede extraer solo aquellos con características claras mejor que el Método 1, y podría crear mejores vectores de estilo.
Sin embargo, no se puede especificar el número de estilos de antemano.

Parámetros:
- eps: Los puntos más cercanos a este valor se conectan para formar la misma clasificación de estilo. Cuanto menor es, más estilos tiende a haber; cuanto mayor, menos estilos.
- min_samples: Número de puntos vecinos necesarios para considerar un punto como núcleo de un estilo. Cuanto menor es, más estilos tiende a haber; cuanto mayor, menos estilos.

Para UMAP, eps alrededor de 0.3, y para t-SNE alrededor de 2.5 podría ser bueno. min_samples depende del número de datos, así que pruebe varios valores.

Detalles:
https://es.wikipedia.org/wiki/DBSCAN
"""


def create_style_vectors_app():
    with gr.Blocks() as app:
        with gr.Accordion("Cómo usar", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(placeholder="nombre_de_su_modelo", label="Nombre del modelo")
        with gr.Tab("Método 0: Crear vectores por subcarpeta"):
            gr.Markdown(method0)
            audio_dir = gr.Textbox(
                placeholder="ruta/a/carpeta_de_audio",
                label="Carpeta de audio",
                info="Guarde los archivos de audio en subcarpetas separadas por estilo.",
            )
            method0_btn = gr.Button("Crear vectores de estilo", variant="primary")
            method0_info = gr.Textbox(label="Resultado")
            method0_btn.click(
                save_style_vectors_by_dirs,
                inputs=[model_name, audio_dir],
                outputs=[method0_info],
            )
        with gr.Tab("Otros métodos"):
            with gr.Row():
                reduction_method = gr.Radio(
                    choices=["UMAP", "t-SNE"],
                    label="Método de reducción de dimensión",
                    info="Antes de v 1.3 era t-SNE, pero UMAP podría ser mejor.",
                    value="UMAP",
                )
                load_button = gr.Button("Cargar vectores de estilo", variant="primary")
            output = gr.Plot(label="Visualización de estilos de voz")
            load_button.click(
                load, inputs=[model_name, reduction_method], outputs=[output]
            )
            with gr.Tab("Método 1: Clasificación automática"):
                with gr.Tab("Clasificación 1"):
                    n_clusters = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=4,
                        label="Número de estilos a crear (excluyendo promedio)",
                        info="Pruebe diferentes números mientras observa el gráfico.",
                    )
                    c_method = gr.Radio(
                        choices=[
                            "Agglomerative after reduction",
                            "KMeans after reduction",
                            "Agglomerative",
                            "KMeans",
                        ],
                        label="Algoritmo",
                        info="Seleccione el algoritmo de clasificación (clustering). Pruebe varios.",
                        value="Agglomerative after reduction",
                    )
                    c_button = gr.Button("Ejecutar clasificación")
                with gr.Tab("Clasificación 2: DBSCAN"):
                    gr.Markdown(dbscan_md)
                    eps = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        step=0.01,
                        value=0.3,
                        label="eps",
                    )
                    min_samples = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=15,
                        label="min_samples",
                    )
                    with gr.Row():
                        dbscan_button = gr.Button("Ejecutar clasificación")
                        num_styles_result = gr.Textbox(label="Número de estilos")
                gr.Markdown("Resultado de la clasificación")
                gr.Markdown(
                    "Nota: Se ha reducido de 256 dimensiones a 2, por lo que la relación de posición no es exacta."
                )
                with gr.Row():
                    gr_plot = gr.Plot()
                    with gr.Column():
                        with gr.Row():
                            cluster_index = gr.Slider(
                                minimum=1,
                                maximum=MAX_CLUSTER_NUM,
                                step=1,
                                value=1,
                                label="Número de estilo",
                                info="Muestra el audio representativo del estilo seleccionado.",
                            )
                            num_files = gr.Slider(
                                minimum=1,
                                maximum=MAX_AUDIO_NUM,
                                step=1,
                                value=5,
                                label="Cuántos audios representativos mostrar",
                            )
                            get_audios_button = gr.Button("Obtener audios representativos")
                        with gr.Row():
                            audio_list = []
                            for i in range(MAX_AUDIO_NUM):
                                audio_list.append(
                                    gr.Audio(visible=False, show_label=True)
                                )
                    c_button.click(
                        do_clustering_gradio,
                        inputs=[n_clusters, c_method],
                        outputs=[gr_plot, cluster_index] + audio_list,
                    )
                    dbscan_button.click(
                        do_dbscan_gradio,
                        inputs=[eps, min_samples],
                        outputs=[gr_plot, cluster_index, num_styles_result]
                        + audio_list,
                    )
                    get_audios_button.click(
                        representative_wav_files_gradio,
                        inputs=[cluster_index, num_files],
                        outputs=audio_list,
                    )
                gr.Markdown("Si el resultado parece bueno, guárdelo.")
                style_names = gr.Textbox(
                    "Enojo, Tristeza, Felicidad",
                    label="Nombres de los estilos",
                    info=f"Ingrese los nombres de los estilos separados por `,`. Ejemplo: `Enojo, Tristeza, Felicidad`. El audio promedio se guarda automáticamente como {DEFAULT_STYLE}.",
                )
                with gr.Row():
                    save_button1 = gr.Button(
                        "Guardar vectores de estilo", variant="primary"
                    )
                    info2 = gr.Textbox(label="Resultado de guardado")

                save_button1.click(
                    save_style_vectors_from_clustering,
                    inputs=[model_name, style_names],
                    outputs=[info2],
                )
            with gr.Tab("Método 2: Selección manual"):
                gr.Markdown(
                    "Ingrese los nombres de archivo de los audios representativos de cada estilo separados por `,`, y los nombres de estilo correspondientes separados por `,`."
                )
                gr.Markdown("Ejemplo: `angry.wav, sad.wav, happy.wav` y `Enojo, Tristeza, Felicidad`")
                gr.Markdown(
                    f"Nota: El estilo {DEFAULT_STYLE} se guarda automáticamente, no especifique un estilo con ese nombre manualmente."
                )
                with gr.Row():
                    audio_files_text = gr.Textbox(
                        label="Nombres de archivos de audio",
                        placeholder="angry.wav, sad.wav, happy.wav",
                    )
                    style_names_text = gr.Textbox(
                        label="Nombres de estilos", placeholder="Enojo, Tristeza, Felicidad"
                    )
                with gr.Row():
                    save_button2 = gr.Button(
                        "Guardar vectores de estilo", variant="primary"
                    )
                    info2 = gr.Textbox(label="Resultado de guardado")
                    save_button2.click(
                        save_style_vectors_from_files,
                        inputs=[model_name, audio_files_text, style_names_text],
                        outputs=[info2],
                    )

    return app


if __name__ == "__main__":
    app = create_style_vectors_app()
    app.launch(inbrowser=True)

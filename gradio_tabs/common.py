import gradio as gr
import sys
import os
import time
from initialize import download_default_models
from style_bert_vits2.logging import logger

def restart():
    logger.info("Restarting application...")
    # Give some time for the response to be sent (though connection will break)
    time.sleep(1)
    # Restart the current process
    # os.execl replaces the current process
    python = sys.executable
    os.execl(python, python, *sys.argv)

def download_and_restart():
    logger.info("Downloading default models...")
    try:
        # Provide some feedback via logger (which user sees in terminal)
        download_default_models()
        logger.info("Download finished. Restarting...")
        restart()
        return "Modelos descargados. Reiniciando..."
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return f"Error downloading models: {e}"

def create_no_model_alert(root_dir):
    with gr.Blocks() as app:
        gr.Markdown(
            f"### Error: No se encontraron modelos.\n\nPor favor coloque los modelos en `{root_dir}` o descárguelos automáticamente."
        )
        download_btn = gr.Button("Descargar modelos predeterminados y reiniciar", variant="primary")
        info = gr.Textbox(label="Estado", interactive=False)

        download_btn.click(
            fn=download_and_restart,
            inputs=[],
            outputs=[info]
        )
    return app

# -*- coding: utf-8 -*-
# output-directory: /tmp/stable-diffusion

# # Edit images with Flux Kontext (Final Production-Ready Version)

# ## Core Imports and Setup
import logging
import random
import threading
import time
import os
from io import BytesIO
from pathlib import Path

import modal
from rich.logging import RichHandler

# ======================================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ======================================================================================

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("rich")

class AppConfig:
    MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
    MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"
    DIFFUSERS_COMMIT_SHA = "00f95b9755718aabb65456e791b8408526ae6e76"
    CACHE_DIR = Path("/cache")
    LORAS_DIR = Path("/loras")
    HF_HUB_CACHE_VOLUME = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
    VOLUMES = {str(CACHE_DIR): HF_HUB_CACHE_VOLUME}
    SECRETS = [modal.Secret.from_name("huggingface-secret")]
    GPU_CONFIG = "B200"
    MODEL_MIN_CONTAINERS = 1
    MODEL_MAX_CONTAINERS = 5
    CONTAINER_SCALEDOWN_WINDOW = 30
    CONTAINER_TIMEOUT = 1800
    WEB_UI_MIN_CONTAINERS = 1
    WEB_UI_MAX_CONTAINERS = 1
    WEB_UI_MAX_INPUTS = 100
    WEB_UI_SCALEDOWN_WINDOW = 60 * 20
    WEB_UI_TIMEOUT = 3600

# ======================================================================================
# 2. MODAL IMAGE DEFINITION
# ======================================================================================

model_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "rich", "accelerate~=1.8.1", f"git+https://github.com/huggingface/diffusers.git@{AppConfig.DIFFUSERS_COMMIT_SHA}",
        "huggingface-hub[hf-transfer]~=0.33.1", "Pillow~=11.2.1", "safetensors==0.4.3",
        "transformers~=4.53.0", "sentencepiece~=0.2.0", "torch==2.7.1", "optimum-quanto==0.2.7", "peft~=0.15.0",
        extra_options="--index-strategy unsafe-best-match", extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir("./loras", str(AppConfig.LORAS_DIR), copy=True)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(AppConfig.CACHE_DIR)})
)

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "rich", "fastapi", "pillow", "psutil",
    "gradio==4.44.1", "pydantic==2.10.6"
)

app = modal.App("example-image-to-image-final-pretty")

with model_image.imports():
    import gc
    import torch
    import diffusers
    import transformers
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from diffusers.utils.logging import disable_progress_bar
    from PIL import Image

# ======================================================================================
# 4. MODEL HANDLER CLASS
# ======================================================================================
@app.cls(
    image=model_image, gpu=AppConfig.GPU_CONFIG, volumes=AppConfig.VOLUMES, secrets=AppConfig.SECRETS,
    min_containers=AppConfig.MODEL_MIN_CONTAINERS, max_containers=AppConfig.MODEL_MAX_CONTAINERS,
    scaledown_window=AppConfig.CONTAINER_SCALEDOWN_WINDOW, timeout=AppConfig.CONTAINER_TIMEOUT,
)
class Model:
    @modal.enter()
    def load_model(self):
        disable_progress_bar()
        transformers.logging.set_verbosity_error()
        diffusers.logging.set_verbosity_error()
        
        log.info("Container starting: loading model...")
        self.device = "cuda"
        self.pipe = FluxKontextPipeline.from_pretrained(
            AppConfig.MODEL_NAME, revision=AppConfig.MODEL_REVISION,
            torch_dtype=torch.bfloat16, cache_dir=AppConfig.CACHE_DIR,
        ).to(self.device)
        self.inference_lock = threading.Lock()
        self.current_lora = None
        log.info("✅ Model and lock initialized successfully.")

    def _manage_lora_safely(self, lora_name, lora_strength):
        if not lora_name or lora_name == "None":
            if self.current_lora: self.pipe.unload_lora_weights(); self.current_lora = None
            return
        if self.current_lora != lora_name:
            if self.current_lora: self.pipe.unload_lora_weights()
            lora_path = AppConfig.LORAS_DIR / f"{lora_name}.safetensors"
            self.pipe.load_lora_weights(str(lora_path), adapter_name="lora")
            self.current_lora = lora_name
        self.pipe.set_adapters(["lora"], adapter_weights=[lora_strength])

    @modal.method()
    def process_image(self, item, prompt, guidance_scale, num_inference_steps, lora_name, lora_strength, seed):
        start_time = time.time()
        filename = item["filename"]
        with self.inference_lock:
            try:
                self._manage_lora_safely(lora_name, lora_strength)
                init_image = load_image(Image.open(BytesIO(item["image_bytes"])))
                # Use provided seed if not -1, otherwise generate random seed
                if seed == -1:
                    seed_value = random.randint(0, 2**32 - 1)
                else:
                    seed_value = seed
                generator = torch.Generator(device=self.device).manual_seed(seed_value)
                result = self.pipe(
                    image=init_image, prompt=prompt, guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, generator=generator,
                    max_sequence_length=512, output_type="pil",
                )
                output_image = result.images[0]
                byte_stream = BytesIO()
                output_image.save(byte_stream, format="PNG", optimize=False, compress_level=0)
                item["processed_bytes"] = byte_stream.getvalue()
                item["success"] = True
            except Exception as e:
                log.error(f"❌ Failed to process [bold]{filename}[/bold]: {e}")
                item["success"] = False
                item["error"] = str(e)
        item["processing_time"] = time.time() - start_time
        if item["success"]: log.info(f"✅ Processed [bold green]{filename}[/bold green] in {item['processing_time']:.2f}s")
        return item

    @modal.method()
    def get_available_loras(self) -> list[str]:
        if not AppConfig.LORAS_DIR.exists(): return []
        return sorted([f.stem for f in AppConfig.LORAS_DIR.glob("*.safetensors")])

# ======================================================================================
# 5. GRADIO WEB UI
# ======================================================================================
@app.function(
    image=web_image, min_containers=AppConfig.WEB_UI_MIN_CONTAINERS, max_containers=AppConfig.WEB_UI_MAX_CONTAINERS,
    scaledown_window=AppConfig.WEB_UI_SCALEDOWN_WINDOW, timeout=AppConfig.WEB_UI_TIMEOUT,
)
@modal.concurrent(max_inputs=AppConfig.WEB_UI_MAX_INPUTS)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    import zipfile
    import tempfile
    import os
    
    model = Model()
    
    def get_lora_choices():
        try: return ["None"] + model.get_available_loras.remote()
        except Exception: return ["None"]

    def extract_images_from_zip(zip_path):
        images, exts = [], {".png", ".jpg", ".jpeg", ".webp"}
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir() and Path(info.filename).suffix.lower() in exts:
                        images.append({"image_bytes": zf.read(info.filename), "filename": Path(info.filename).name, "relative_path": info.filename})
            log.info(f"Extracted {len(images)} images from ZIP.")
            return images
        except Exception as e:
            log.error(f"Failed to extract from ZIP {zip_path}: {e}")
            return []
            
    def create_output_zip(results, input_filename=None):
        # Создание осмысленного имени файла
        timestamp = int(time.time())

        # Извлечение базового имени из входного файла
        if input_filename:
            base_name = os.path.splitext(os.path.basename(input_filename))[0].lower() + "_processed"
        else:
            base_name = "batch_processed_images"

        # Формирование финального имени
        output_filename = f"{base_name}_{timestamp}.zip"
        output_path = os.path.join("/tmp", output_filename)

        successful = [r for r in results if r.get("success")]
        if not successful:
            raise ValueError("No successful images to create a ZIP file from.")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for result in successful:
                zf.writestr(result["relative_path"], result["processed_bytes"])

        return output_path

    def format_summary_message(results, duration):
        successful = sum(1 for r in results if r.get("success"))
        total = len(results)
        avg_time = duration / total if total > 0 else 0
        return f"""## 🎉 Batch Processing Complete!
- **Successfully Processed**: ✅ {successful}/{total}
- **Total Wall-Clock Time**: ⏱️ {duration:.1f}s
- **Average Time/Image**: ⚡ {avg_time:.1f}s (True parallel execution)"""

    def handle_batch_process(zip_file, prompt, g_scale, steps, lora, lora_strength, seed, progress=gr.Progress()):
        if not zip_file: return None, "❌ Error: Please upload a ZIP file."
        if not prompt.strip(): return None, "❌ Error: Please enter a prompt."
        progress(0, desc="🔍 Extracting images...")
        images = extract_images_from_zip(zip_file.name)
        if not images: return None, "❌ Error: No valid images found."

        num_images = len(images)
        log.info(f"Starting parallel processing for [cyan]{num_images}[/cyan] images on [cyan]{AppConfig.MODEL_MAX_CONTAINERS}[/cyan] containers.")
        progress(0.1, desc=f"🚀 Processing {num_images} images in parallel...")

        args = (prompt, g_scale, steps, lora, lora_strength, seed)
        starmap_args = [(item, *args) for item in images]
        all_results = []
        start_time = time.time()
        
        for i, result in enumerate(model.process_image.starmap(starmap_args)):
            all_results.append(result)
            progress((i + 1) / num_images, desc=f"Processed {i + 1}/{num_images} images...")
            
        duration = time.time() - start_time
        log.info(f"Parallel processing finished in [bold yellow]{duration:.2f}s[/bold yellow].")
        
        progress(0.98, desc="📦 Creating output archive...")
        
        try:
            # Передаем имя входного файла в create_output_zip
            input_filename = zip_file.name if zip_file else None
            output_zip_path = create_output_zip(all_results, input_filename)
            summary = format_summary_message(all_results, duration)
            progress(1.0, desc="✅ Complete!")
            return output_zip_path, summary
        except Exception as e:
            log.error(f"Error in final step: {e}", exc_info=True)
            return None, f"❌ Error creating output: {e}"

    web_app = FastAPI()
    with gr.Blocks(title="🎨 Flux Kontext Image Editor") as demo:
        gr.Markdown("# 🎨 Flux Kontext Image Editor (Production Ready)")
        lora_choices = get_lora_choices()
        with gr.Tabs():
            with gr.TabItem("📦 Batch Processing"):
                gr.Markdown(f"**Parallel Processing Enabled:** App will scale up to **{AppConfig.MODEL_MAX_CONTAINERS}** containers.")
                with gr.Row():
                    with gr.Column():
                        # --- ИЗМЕНЕНИЕ: Используем правильный компонент gr.File для загрузки файлов ---
                        zip_upload = gr.File(label="📦 Upload ZIP file", type="filepath")
                        batch_output = gr.File(label="📥 Download Processed Images")
                        
                        batch_prompt = gr.Textbox(label="Batch Edit Prompt", lines=3, value="make this in pokraslampas style")
                        with gr.Row():
                            batch_g_scale = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="Guidance")
                            batch_steps = gr.Slider(15, 50, 28, step=1, label="Inference Steps")
                        with gr.Row():
                            batch_lora = gr.Dropdown(lora_choices, value="None", label="LoRA")
                            batch_lora_strength = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="LoRA Strength")
                        with gr.Row():
                            batch_seed = gr.Slider(-1, 2147483647, -1, step=1, label="Seed (-1 for random)")
                        process_btn = gr.Button("🚀 Process Batch", variant="primary")
                    with gr.Column():
                        batch_summary = gr.Markdown("📊 Results will appear here.")
                        
        process_btn.click(fn=handle_batch_process, inputs=[zip_upload, batch_prompt, batch_g_scale, batch_steps, batch_lora, batch_lora_strength, batch_seed], outputs=[batch_output, batch_summary])
    
    demo.queue(max_size=20)
    web_app = gr.mount_gradio_app(web_app, demo, path="/")
    return web_app

@app.local_entrypoint()
def main():
    log.info("🚀 Starting UI server...")
    try: ui.remote()
    except KeyboardInterrupt: log.info("\n🛑 User interrupted.")
    except Exception as e: log.error(f"\n❌ An error occurred: {e}", exc_info=True)
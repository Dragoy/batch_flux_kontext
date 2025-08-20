# -*- coding: utf-8 -*-
# output-directory: /tmp/stable-diffusion

# # Edit images with Flux Kontext

# In this example, we run the Flux Kontext model in _image-to-image_ mode:
# the model takes in a prompt and an image and edits the image to better match the prompt.

# For example, the model edited the first image into the second based on the prompt
# "A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style".

#  <img src="https://modal-cdn.com/dog-wizard-ghibli-flux-kontext.jpg" alt="A photo of a dog transformed into a cartoon of a cute dog wizard" />

# The model is Black Forest Labs' [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
# Learn more about the model [here](https://bfl.ai/announcements/flux-1-kontext-dev).

# ## Credits
# Original script by Modal Labs
# Gradio UI implementation added by skfrost19 (https://github.com/skfrost19)

# ## Define a container image

# First, we define the environment the model inference will run in,
# the [container image](https://modal.com/docs/guide/custom-container).

from io import BytesIO
from pathlib import Path

import modal

diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install(
        "accelerate~=1.8.1",
        "git+https://github.com/huggingface/diffusers.git@" + diffusers_commit_sha,
        "huggingface-hub[hf-transfer]~=0.33.1",
        "Pillow~=11.2.1",
        "safetensors~=0.5.3",
        "transformers~=4.53.0",
        "sentencepiece~=0.2.0",
        "torch==2.7.1",
        "optimum-quanto==0.2.7",
        "peft~=0.15.0",  # Required for LoRA support
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir("./loras", "/loras", copy=True)  # Copy LoRA files into image
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

CACHE_DIR = Path("/cache")
LORAS_DIR = Path("/loras")  # LoRA files are now in the image, not a volume
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {str(CACHE_DIR): cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

# Web UI image with Gradio dependencies
web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi",
    "gradio",  # Using a more stable version
    "pillow",
    "psutil",  # For memory monitoring
)

app = modal.App("example-image-to-image")

# LoRA files are now included in the image via add_local_dir
# No need for complex sync functions

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    import gc
    import time


# ## Setting up and running Flux Kontext

# The Modal `Cls` defined below contains all the logic to set up and run Flux Kontext.

# The [container lifecycle](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta) decorator
# (`@modal.enter()`) ensures that the model is loaded into memory when a container starts, before it picks up any inputs.

# The `inference` method runs the actual model inference. It takes in an image as a collection of `bytes` and a string `prompt` and returns
# a new image (also as a collection of `bytes`).

# To avoid excessive cold-starts, we set the `scaledown_window` to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down.


@app.cls(
    image=image,
    gpu="B200",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=240,
    timeout=1800,  # 30 minutes timeout for long-running inference tasks
    max_inputs=1   # Modal only supports max_inputs=1 for classes
)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16

        self.seed = 42
        self.device = "cuda"
        self.current_lora = None  # Track currently loaded LoRA

        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,  # This is crucial for the correct model version
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
        ).to(self.device)

    @modal.exit()
    def exit(self):
        """Clean up resources on container shutdown."""
        print("🧹 Cleaning up resources...")
        try:
            if hasattr(self, 'current_lora') and self.current_lora:
                print(f"Unloading LoRA: {self.current_lora}")
                self.pipe.unload_lora_weights()

            if hasattr(self, 'pipe'):
                print("Clearing pipeline cache...")
                del self.pipe
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning during cleanup: {e}")

        print("✅ Cleanup completed")

    @modal.method()
    def get_available_loras(self) -> list[str]:
        """Get list of available LoRA files in the loras directory."""
        try:
            lora_files = []
            print(f"Checking for LoRA files in: {LORAS_DIR}")

            if LORAS_DIR.exists():
                all_files = list(LORAS_DIR.iterdir())
                print(f"All files in LoRA directory: {[f.name for f in all_files]}")

                safetensors_files = list(LORAS_DIR.glob("*.safetensors"))
                print(f"Found .safetensors files: {[f.name for f in safetensors_files]}")

                for file in safetensors_files:
                    lora_files.append(file.stem)  # Get filename without extension
            else:
                print(f"LoRA directory does not exist: {LORAS_DIR}")

            print(f"Returning LoRA files: {lora_files}")
            return sorted(lora_files)

        except Exception as e:
            print(f"Error getting available LoRAs: {e}")
            return []

    @modal.method()
    def check_lora_compatibility(self, lora_name: str) -> dict:
        """Check LoRA compatibility by attempting to load it."""
        try:
            lora_path = LORAS_DIR / f"{lora_name}.safetensors"
            if not lora_path.exists():
                return {"compatible": False, "error": "File not found"}

            # Try to load the LoRA to check compatibility
            try:
                # Create a temporary pipeline instance for testing
                test_pipe = self.pipe
                test_pipe.load_lora_weights(str(lora_path), adapter_name="test_lora")
                test_pipe.unload_lora_weights()

                return {
                    "compatible": True,
                    "message": "LoRA loaded successfully"
                }

            except Exception as load_error:
                error_str = str(load_error).lower()

                if "lora_unet" in error_str or "adaLN_modulation" in error_str:
                    return {
                        "compatible": False,
                        "error": "Incompatible architecture",
                        "suggestion": "This LoRA appears to be for SDXL/SD1.5, not Flux"
                    }
                elif "peft backend" in error_str:
                    return {
                        "compatible": False,
                        "error": "PEFT backend required",
                        "suggestion": "Make sure PEFT library is properly installed"
                    }
                else:
                    return {
                        "compatible": False,
                        "error": str(load_error),
                        "suggestion": "Unknown compatibility issue"
                    }

        except Exception as e:
            return {"compatible": False, "error": str(e)}

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
    ) -> bytes:
        # Load the original image without resizing to preserve dimensions
        init_image = load_image(Image.open(BytesIO(image_bytes)))

        # Handle LoRA loading/unloading
        try:
            if lora_name and lora_name != "None":
                # Check if we need to switch LoRAs
                if self.current_lora != lora_name:
                    # Unload current LoRA if any
                    if self.current_lora:
                        try:
                            self.pipe.unload_lora_weights()
                            print(f"Unloaded previous LoRA: {self.current_lora}")
                        except Exception as e:
                            print(f"Warning: Error unloading previous LoRA: {e}")

                    # Load new LoRA
                    lora_path = LORAS_DIR / f"{lora_name}.safetensors"
                    if lora_path.exists():
                        try:
                            print(f"Loading LoRA from: {lora_path}")
                            self.pipe.load_lora_weights(str(lora_path), adapter_name="lora")
                            self.current_lora = lora_name
                            print(f"✅ Successfully loaded LoRA: {lora_name}")
                        except Exception as e:
                            print(f"❌ Error loading LoRA {lora_name}: {e}")

                            # Try alternative loading methods
                            try:
                                print(f"🔄 Trying alternative loading method...")
                                # Try loading without specifying adapter name
                                self.pipe.load_lora_weights(str(lora_path))
                                self.current_lora = lora_name
                                print(f"✅ Successfully loaded LoRA with alternative method: {lora_name}")
                            except Exception as alt_error:
                                print(f"❌ Alternative loading also failed: {alt_error}")

                                # Provide specific error guidance
                                error_str = str(e).lower()
                                if "lora_unet" in error_str or "lora_down" in error_str or "adaLN_modulation" in error_str:
                                    print("💡 This LoRA appears to be incompatible with FluxKontextPipeline")
                                    print("💡 It may be designed for a different model architecture (e.g., SDXL, SD1.5)")
                                    print("💡 Try using LoRAs specifically trained for Flux models")
                                elif "peft backend is required" in error_str:
                                    print("💡 PEFT library issue - make sure peft~=0.15.0 is installed")
                                elif "cliptext" in error_str:
                                    print("💡 Text encoder compatibility issue - this may be normal for some LoRAs")

                                self.current_lora = None
                    else:
                        print(f"❌ LoRA file not found: {lora_path}")
                        self.current_lora = None

                # Set LoRA strength if LoRA is loaded
                if self.current_lora:
                    try:
                        self.pipe.set_adapters(["lora"], adapter_weights=[lora_strength])
                        print(f"✅ Set LoRA strength to {lora_strength}")
                    except Exception as e:
                        print(f"❌ Error setting LoRA strength: {e}")
            else:
                # Unload LoRA if "None" is selected or no LoRA specified
                if self.current_lora:
                    try:
                        self.pipe.unload_lora_weights()
                        print(f"✅ Unloaded LoRA: {self.current_lora}")
                        self.current_lora = None
                    except Exception as e:
                        print(f"Warning: Error unloading LoRA: {e}")
                        self.current_lora = None
        except Exception as e:
            print(f"❌ Unexpected error handling LoRA: {e}")
            # Continue without LoRA if there's an error

        # Use a different seed for each inference to avoid repetitive results
        import random

        seed = random.randint(0, 2**32 - 1)

        result = self.pipe(
            image=init_image,
            guidance_scale=guidance_scale,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(seed),
            max_sequence_length=512,  # Maximum for best quality
        )

        # Handle different return types from the pipeline
        try:
            # Try to access images attribute first (standard diffusers format)
            image = result.images[0]  # type: ignore
        except (AttributeError, IndexError, TypeError):
            # Fallback for different pipeline return formats
            if isinstance(result, (list, tuple)) and len(result) > 0:
                image = result[0]
            else:
                image = result

        # Ensure we have a PIL Image
        if not hasattr(image, 'save'):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        byte_stream = BytesIO()
        # Save with maximum quality to preserve details
        # Type ignore because we've already checked that image has save method
        image.save(byte_stream, format="PNG", optimize=False, compress_level=0)  # type: ignore
        image_bytes = byte_stream.getvalue()

        return image_bytes

    def _process_single_image(
        self,
        image_bytes: bytes,
        prompt: str,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
    ) -> bytes:
        """Internal method to process a single image (uses same logic as inference)."""
        # Load the original image without resizing to preserve dimensions
        init_image = load_image(Image.open(BytesIO(image_bytes)))

        # Handle LoRA loading/unloading
        try:
            if lora_name and lora_name != "None":
                # Check if we need to switch LoRAs
                if self.current_lora != lora_name:
                    # Unload current LoRA if any
                    if self.current_lora:
                        try:
                            self.pipe.unload_lora_weights()
                            print(f"Unloaded previous LoRA: {self.current_lora}")
                        except Exception as e:
                            print(f"Warning: Error unloading previous LoRA: {e}")

                    # Load new LoRA
                    lora_path = LORAS_DIR / f"{lora_name}.safetensors"
                    if lora_path.exists():
                        try:
                            print(f"Loading LoRA from: {lora_path}")
                            self.pipe.load_lora_weights(str(lora_path), adapter_name="lora")
                            self.current_lora = lora_name
                            print(f"✅ Successfully loaded LoRA: {lora_name}")
                        except Exception as e:
                            print(f"❌ Error loading LoRA {lora_name}: {e}")

                            # Try alternative loading methods
                            try:
                                print(f"🔄 Trying alternative loading method...")
                                # Try loading without specifying adapter name
                                self.pipe.load_lora_weights(str(lora_path))
                                self.current_lora = lora_name
                                print(f"✅ Successfully loaded LoRA with alternative method: {lora_name}")
                            except Exception as alt_error:
                                print(f"❌ Alternative loading also failed: {alt_error}")
                                self.current_lora = None
                    else:
                        print(f"❌ LoRA file not found: {lora_path}")
                        self.current_lora = None

                # Set LoRA strength if LoRA is loaded
                if self.current_lora:
                    try:
                        self.pipe.set_adapters(["lora"], adapter_weights=[lora_strength])
                        print(f"✅ Set LoRA strength to {lora_strength}")
                    except Exception as e:
                        print(f"❌ Error setting LoRA strength: {e}")
            else:
                # Unload LoRA if "None" is selected or no LoRA specified
                if self.current_lora:
                    try:
                        self.pipe.unload_lora_weights()
                        print(f"✅ Unloaded LoRA: {self.current_lora}")
                        self.current_lora = None
                    except Exception as e:
                        print(f"Warning: Error unloading LoRA: {e}")
                        self.current_lora = None
        except Exception as e:
            print(f"❌ Unexpected error handling LoRA: {e}")
            # Continue without LoRA if there's an error

        # Use a different seed for each inference to avoid repetitive results
        import random
        seed = random.randint(0, 2**32 - 1)

        result = self.pipe(
            image=init_image,
            guidance_scale=guidance_scale,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(seed),
            max_sequence_length=512,  # Maximum for best quality
        )

        # Handle different return types from the pipeline
        try:
            # Try to access images attribute first (standard diffusers format)
            image = result.images[0]  # type: ignore
        except (AttributeError, IndexError, TypeError):
            # Fallback for different pipeline return formats
            if isinstance(result, (list, tuple)) and len(result) > 0:
                image = result[0]
            else:
                image = result

        # Ensure we have a PIL Image
        if not hasattr(image, 'save'):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        byte_stream = BytesIO()
        # Save with maximum quality to preserve details
        image.save(byte_stream, format="PNG", optimize=False, compress_level=0)  # type: ignore
        image_bytes = byte_stream.getvalue()

        return image_bytes

    @modal.method()
    def batch_inference(
        self,
        batch_data: list[dict],  # [{"image_bytes": bytes, "filename": str, "relative_path": str}]
        prompt: str,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
    ) -> list[dict]:
        """Process multiple images with the same prompt."""
        results = []

        print(f"🚀 Starting batch processing of {len(batch_data)} images")
        print(f"📝 Prompt: {prompt}")
        print(f"⚙️ Settings: guidance_scale={guidance_scale}, steps={num_inference_steps}")
        if lora_name:
            print(f"🎨 LoRA: {lora_name} (strength={lora_strength})")

        for i, item in enumerate(batch_data):
            start_time = time.time()
            print(f"\n📸 Processing image {i+1}/{len(batch_data)}: {item['filename']}")

            try:
                # Process single image using internal method
                processed_bytes = self._process_single_image(
                    item["image_bytes"],
                    prompt,
                    guidance_scale,
                    num_inference_steps,
                    lora_name,
                    lora_strength
                )

                processing_time = time.time() - start_time
                print(f"✅ Successfully processed {item['filename']} in {processing_time:.2f}s")

                results.append({
                    "success": True,
                    "filename": item["filename"],
                    "relative_path": item["relative_path"],
                    "processed_bytes": processed_bytes,
                    "index": i,
                    "processing_time": processing_time
                })

                # Memory cleanup every 5 images
                if (i + 1) % 5 == 0:
                    print(f"🧹 Cleaning up memory after {i+1} images...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ Failed to process {item['filename']} after {processing_time:.2f}s: {str(e)}")

                results.append({
                    "success": False,
                    "filename": item["filename"],
                    "relative_path": item["relative_path"],
                    "error": str(e),
                    "index": i,
                    "processing_time": processing_time
                })

        # Final cleanup
        print("🧹 Final memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        successful = sum(1 for r in results if r["success"])
        total_time = sum(r["processing_time"] for r in results)
        print(f"\n🎉 Batch processing complete!")
        print(f"📊 Results: {successful}/{len(batch_data)} successful")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        print(f"⚡ Average time per image: {total_time/len(batch_data):.2f}s")

        return results


# ## Gradio Web UI

# You can deploy the Gradio web interface with:
# ```bash
# modal deploy flux_kontext.py
# ```
# This will create a web interface accessible to anyone with the URL.


@app.function(
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,  # 20 minutes
    timeout=3600,  # 1 hour timeout for web interface
    # Gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """A Gradio interface for Flux Kontext image editing."""
    import io
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from PIL import Image
    import zipfile
    import tempfile
    import shutil
    from pathlib import Path
    import psutil
    import gc
    import os

    web_app = FastAPI()

    def get_lora_choices():
        """Get available LoRA choices for the dropdown."""
        try:
            print("Getting LoRA choices from Modal...")
            lora_files = Model().get_available_loras.remote()
            print(f"Retrieved LoRA files: {lora_files}")

            choices = ["None"] + lora_files if lora_files else ["None"]
            print(f"Final choices: {choices}")
            return choices

        except Exception as e:
            print(f"Error getting LoRA choices: {e}")
            return ["None"]

    def extract_images_from_folder(folder_path: str) -> list[dict]:
        """Extract image files from a folder, preserving structure."""
        try:
            folder = Path(folder_path)

            # If the path is a file, get its parent directory
            if folder.is_file():
                folder = folder.parent

            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
            images = []

            print(f"📁 Scanning folder: {folder}")

            if not folder.exists():
                print(f"❌ Folder does not exist: {folder}")
                return []

            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    try:
                        with open(file_path, 'rb') as f:
                            image_bytes = f.read()

                        # Validate it's a valid image
                        Image.open(io.BytesIO(image_bytes)).verify()

                        relative_path = file_path.relative_to(folder)
                        images.append({
                            "image_bytes": image_bytes,
                            "filename": file_path.name,
                            "relative_path": str(relative_path)
                        })
                        print(f"✅ Found image: {relative_path}")

                    except Exception as e:
                        print(f"⚠️ Skipping invalid image {file_path}: {e}")
                        continue

            print(f"📊 Total images found: {len(images)}")
            return images

        except Exception as e:
            print(f"❌ Error scanning folder {folder_path}: {e}")
            return []

    def extract_images_from_zip(zip_path: str) -> list[dict]:
        """Extract image files from a ZIP archive, preserving structure."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
        images = []

        print(f"📦 DEBUG: Extracting from ZIP: {zip_path}")
        print(f"📦 DEBUG: ZIP path type: {type(zip_path)}")

        try:
            # Ensure the ZIP file exists
            if not os.path.exists(zip_path):
                print(f"❌ ZIP file does not exist: {zip_path}")
                # Try to list directory contents for debugging
                dir_path = os.path.dirname(zip_path)
                if os.path.exists(dir_path):
                    print(f"📁 Directory contents: {os.listdir(dir_path)}")
                return []

            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if not file_info.is_dir():
                        file_path = Path(file_info.filename)
                        if file_path.suffix.lower() in image_extensions:
                            try:
                                image_bytes = zip_file.read(file_info.filename)

                                # Validate it's a valid image
                                Image.open(io.BytesIO(image_bytes)).verify()

                                images.append({
                                    "image_bytes": image_bytes,
                                    "filename": file_path.name,
                                    "relative_path": file_info.filename
                                })
                                print(f"✅ Found image: {file_info.filename}")

                            except Exception as e:
                                print(f"⚠️ Skipping invalid image {file_info.filename}: {e}")
                                continue

        except zipfile.BadZipFile:
            print(f"❌ Invalid ZIP file: {zip_path}")
            return []
        except Exception as e:
            print(f"❌ Error reading ZIP file: {e}")
            return []

        print(f"📊 Total images found: {len(images)}")
        return images

    def extract_images_from_file_list(files: list) -> list[dict]:
        """Extract image files from a list of uploaded files (file paths as strings)."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
        images = []

        print(f"📁 Processing {len(files)} uploaded files")

        for file_item in files:
            try:
                # Handle both string paths and file objects
                if isinstance(file_item, str):
                    file_path = file_item
                elif hasattr(file_item, 'name'):
                    file_path = file_item.name
                else:
                    file_path = str(file_item)

                file_name = os.path.basename(file_path)

                print(f"🔍 Processing file: {file_path}")

                # Check if it's an image file
                if Path(file_name).suffix.lower() in image_extensions:
                    # Read the file content
                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()

                    # Validate it's a valid image
                    Image.open(io.BytesIO(image_bytes)).verify()

                    # Create relative path (remove common prefix if any)
                    relative_path = file_name

                    images.append({
                        "image_bytes": image_bytes,
                        "filename": file_name,
                        "relative_path": relative_path
                    })
                    print(f"✅ Found image: {file_name}")
                else:
                    print(f"⚠️ Skipping non-image file: {file_name}")

            except Exception as e:
                print(f"⚠️ Skipping invalid file {file_item}: {e}")
                continue

        print(f"📊 Total images found: {len(images)}")
        return images

    def create_output_zip(results: list[dict], output_path: str) -> str:
        """Create a ZIP file with processed images, preserving directory structure."""
        print(f"📦 Creating output ZIP: {output_path}")

        successful_results = [r for r in results if r["success"]]
        print(f"📊 Creating ZIP with {len(successful_results)} successful images out of {len(results)} total")

        if not successful_results:
            raise ValueError("No successful images to create ZIP file")

        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
                for result in results:
                    if result["success"]:
                        # Write processed image to ZIP with original path structure
                        zip_file.writestr(
                            result["relative_path"],
                            result["processed_bytes"]
                        )
                        print(f"✅ Added to ZIP: {result['relative_path']}")
                    else:
                        print(f"⚠️ Skipped failed image: {result['filename']}")

            # Verify the ZIP file was created and has content
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"ZIP file was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise ValueError(f"ZIP file is empty: {output_path}")

            print(f"📦 ZIP file created successfully: {output_path} ({file_size} bytes)")
            return output_path

        except Exception as e:
            print(f"❌ Error creating ZIP file: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise

    def monitor_memory_usage():
        """Monitor current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        except:
            return 0

    def cleanup_temp_files(temp_paths: list):
        """Clean up temporary files."""
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    print(f"🗑️ Cleaned up temp file: {path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up {path}: {e}")

    def validate_image_file(file_path: str, max_size_mb: int = 50) -> tuple[bool, str]:
        """Validate image file size and format."""
        try:
            # Check file size
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            if file_size > max_size_mb:
                return False, f"File too large: {file_size:.1f}MB (max: {max_size_mb}MB)"

            # Check if it's a valid image
            with Image.open(file_path) as img:
                img.verify()

            return True, "Valid image"

        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    def estimate_processing_time(num_images: int, avg_time_per_image: float = 15.0) -> str:
        """Estimate total processing time."""
        total_seconds = num_images * avg_time_per_image

        if total_seconds < 60:
            return f"~{total_seconds:.0f} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"~{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            return f"~{hours:.1f} hours"

    def edit_image(input_image, prompt, guidance_scale, num_inference_steps, lora_selection, lora_strength):
        """Process the image editing request."""
        if input_image is None:
            return None

        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format="PNG")
            input_image_bytes = img_byte_arr.getvalue()

            # Call the inference function with LoRA parameters
            output_image_bytes = Model().inference.remote(
                input_image_bytes,
                prompt,
                float(guidance_scale),
                int(num_inference_steps),
                lora_selection if lora_selection != "None" else None,
                float(lora_strength),
            )

            # Convert bytes back to PIL image
            output_image = Image.open(io.BytesIO(output_image_bytes))
            return output_image

        except Exception:
            return None

    def process_batch(
        input_source,
        batch_prompt,
        guidance_scale,
        num_inference_steps,
        lora_selection,
        lora_strength,
        progress=gr.Progress()
    ):
        """Process multiple images from folder or ZIP file with enhanced error handling."""
        temp_files = []  # Track temporary files for cleanup

        try:
            # Input validation
            print(f"🔍 DEBUG: input_source type: {type(input_source)}")
            print(f"🔍 DEBUG: input_source value: {input_source}")

            if input_source is None:
                return None, "❌ **Error:** Please upload a folder or ZIP file."

            if not batch_prompt or batch_prompt.strip() == "":
                return None, "❌ **Error:** Please enter a prompt for batch processing."

            progress(0, desc="🔍 Analyzing input source...")

            # Determine input type and extract images
            if isinstance(input_source, dict) and input_source.get("type") == "folder":
                # Handle folder upload (list of files)
                print(f"📁 Processing folder upload with {len(input_source['files'])} files")
                images = extract_images_from_file_list(input_source['files'])
                source_type = "Folder"
            else:
                # Handle file upload (ZIP or single file)
                try:
                    # Check if it's a file object with a name attribute
                    if hasattr(input_source, 'name') and not isinstance(input_source, dict):
                        input_path = input_source.name
                        print(f"📁 Processing file: {input_path}")

                        if input_path.lower().endswith('.zip'):
                            images = extract_images_from_zip(input_path)
                            source_type = "ZIP Archive"
                        else:
                            # Single file or folder path
                            images = extract_images_from_folder(input_path)
                            source_type = "Folder"
                    elif isinstance(input_source, str):
                        # String path
                        input_path = input_source
                        print(f"📁 Processing path: {input_path}")

                        if input_path.lower().endswith('.zip'):
                            images = extract_images_from_zip(input_path)
                            source_type = "ZIP Archive"
                        else:
                            images = extract_images_from_folder(input_path)
                            source_type = "Folder"
                    else:
                        print(f"❌ Invalid input source type: {type(input_source)}")
                        return None, f"❌ **Error:** Invalid input source type. Please try uploading again."

                except Exception as e:
                    print(f"❌ Error processing input source: {e}")
                    return None, f"❌ **Error:** Unable to process input source: {str(e)}"

            if not images:
                return None, f"❌ **Error:** No valid images found in the {source_type.lower()}.\n\nSupported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF"

            # Check for reasonable batch size
            if len(images) > 100:
                return None, f"❌ **Error:** Too many images ({len(images)}). Maximum supported: 100 images per batch.\n\nPlease split your images into smaller batches."

            # Estimate processing time
            estimated_time = estimate_processing_time(len(images))
            progress(0.05, desc=f"📊 Found {len(images)} images. Estimated time: {estimated_time}")

            # Memory check
            initial_memory = monitor_memory_usage()
            print(f"💾 Initial memory usage: {initial_memory:.1f}MB")

            progress(0.1, desc="🚀 Starting batch processing...")

            # Process images in smaller batches to manage memory
            batch_size = min(3, len(images))  # Adaptive batch size
            all_results = []
            processing_errors = []

            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_progress = 0.1 + (i / len(images)) * 0.8

                progress(
                    batch_progress,
                    desc=f"🔄 Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1} ({len(batch)} images)"
                )

                try:
                    # Call Modal batch inference with timeout handling
                    batch_results = Model().batch_inference.remote(
                        batch,
                        batch_prompt,
                        float(guidance_scale),
                        int(num_inference_steps),
                        lora_selection if lora_selection != "None" else None,
                        float(lora_strength),
                    )

                    all_results.extend(batch_results)

                    # Memory monitoring and cleanup
                    current_memory = monitor_memory_usage()
                    if current_memory > initial_memory * 2:  # Memory usage doubled
                        print(f"⚠️ High memory usage detected: {current_memory:.1f}MB")
                        gc.collect()  # Force garbage collection

                except Exception as batch_error:
                    error_msg = f"Batch {i//batch_size + 1} failed: {str(batch_error)}"
                    processing_errors.append(error_msg)
                    print(f"❌ {error_msg}")

                    # Add failed results for this batch
                    for item in batch:
                        all_results.append({
                            "success": False,
                            "filename": item["filename"],
                            "relative_path": item["relative_path"],
                            "error": str(batch_error),
                            "index": len(all_results),
                            "processing_time": 0
                        })

            if not all_results:
                return None, "❌ **Error:** All batches failed to process. Please check your input files and try again."

            progress(0.9, desc="📦 Creating output archive...")

            # Create output ZIP file
            try:
                # Create a temporary file but copy it to a more permanent location
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                    temp_output_path = tmp_file.name
                    temp_files.append(temp_output_path)

                create_output_zip(all_results, temp_output_path)

                # Verify the file was created successfully
                if not os.path.exists(temp_output_path):
                    raise FileNotFoundError(f"Output ZIP file was not created: {temp_output_path}")

                # Create a more permanent file that Gradio can handle
                import shutil
                import time

                # Create a unique filename based on input file name
                timestamp = int(time.time())

                # Extract base name from input source
                base_name = "batch_processed_images"
                try:
                    if isinstance(input_source, dict) and input_source.get("type") == "folder":
                        # For folder uploads, use "folder" as base name
                        base_name = "folder_processed"
                    elif hasattr(input_source, 'name') and not isinstance(input_source, dict):
                        # Extract filename without extension from file object
                        input_filename = os.path.basename(input_source.name)
                        base_name = os.path.splitext(input_filename)[0].lower() + "_processed"
                    elif isinstance(input_source, str):
                        # Extract from string path
                        input_filename = os.path.basename(input_source)
                        base_name = os.path.splitext(input_filename)[0].lower() + "_processed"
                except Exception as e:
                    print(f"⚠️ Could not extract input filename, using default: {e}")
                    base_name = "batch_processed_images"

                permanent_filename = f"{base_name}_{timestamp}.zip"
                permanent_output_path = os.path.join("/tmp", permanent_filename)

                # Copy the file
                shutil.copy2(temp_output_path, permanent_output_path)

                print(f"✅ Output ZIP created successfully: {permanent_output_path}")
                output_path = permanent_output_path

            except Exception as zip_error:
                cleanup_temp_files(temp_files)
                return None, f"❌ **Error creating output archive:** {str(zip_error)}"

            # Generate comprehensive summary
            successful = sum(1 for r in all_results if r["success"])
            failed = len(all_results) - successful
            total_time = sum(r.get("processing_time", 0) for r in all_results)
            avg_time = total_time / successful if successful > 0 else 0

            # Success rate calculation
            success_rate = (successful / len(all_results)) * 100 if all_results else 0

            summary = f"""
## 🎉 Batch Processing Complete!

**📊 Summary:**
- **Source**: {source_type}
- **Total Images**: {len(images)}
- **Successfully Processed**: ✅ {successful}
- **Failed**: ❌ {failed}
- **Success Rate**: {success_rate:.1f}%
- **Total Processing Time**: ⏱️ {total_time:.1f}s
- **Average Time per Image**: ⚡ {avg_time:.1f}s

**📝 Settings Used:**
- **Prompt**: "{batch_prompt}"
- **Guidance Scale**: {guidance_scale}
- **Inference Steps**: {num_inference_steps}
- **LoRA**: {lora_selection if lora_selection != "None" else "None"}
- **LoRA Strength**: {lora_strength if lora_selection != "None" else "N/A"}

📥 **Download the ZIP file below to get your processed images with preserved folder structure.**
            """

            # Add error details if any
            if failed > 0:
                failed_files = [r["filename"] for r in all_results if not r["success"]]
                summary += f"\n\n**⚠️ Failed Files ({failed}):**\n"
                for i, filename in enumerate(failed_files[:10]):
                    summary += f"- {filename}\n"
                if len(failed_files) > 10:
                    summary += f"- ... and {len(failed_files) - 10} more\n"

            if processing_errors:
                summary += f"\n\n**🔧 Processing Errors:**\n"
                for error in processing_errors[:5]:
                    summary += f"- {error}\n"
                if len(processing_errors) > 5:
                    summary += f"- ... and {len(processing_errors) - 5} more\n"

            progress(1.0, desc="✅ Complete!")

            # Final verification that the output file exists
            if not os.path.exists(output_path):
                return None, f"❌ **Error:** Output file was not created properly."

            print(f"✅ Returning output file: {output_path} (size: {os.path.getsize(output_path)} bytes)")

            # Try to return the file in a way that Gradio can handle
            try:
                # Create a simple file object that Gradio can process
                return output_path, summary
            except Exception as e:
                print(f"⚠️ Error preparing file for return: {e}")
                return None, f"{summary}\n\n⚠️ **Note:** File was created successfully but there was an issue with the download. File location: {output_path}"

        except Exception as e:
            # Cleanup on error
            cleanup_temp_files(temp_files)

            error_msg = f"""
❌ **Critical Error During Batch Processing**

**Error Details:** {str(e)}

**Troubleshooting Tips:**
- Ensure your images are in supported formats (PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF)
- Check that your ZIP file or folder is not corrupted
- Try with a smaller batch size (< 50 images)
- Verify you have a stable internet connection

**If the problem persists, please try processing images individually first.**
            """

            print(f"Critical batch processing error: {e}")
            import traceback
            traceback.print_exc()

            return None, error_msg

        finally:
            # Final cleanup
            gc.collect()

    # Create the enhanced Gradio interface with tabs
    with gr.Blocks(title="🎨 Flux Kontext Image Editor - Enhanced") as demo:
        gr.Markdown("# 🎨 Flux Kontext Image Editor")
        gr.Markdown("Edit single images or process entire folders/ZIP archives with the same prompt using the powerful Flux Kontext model.")

        with gr.Tabs():
            # Single Image Tab (existing functionality)
            with gr.TabItem("🖼️ Single Image", elem_id="single-tab"):
                gr.Markdown("### Transform a single image with your prompt")

                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="pil", label="Input Image", height=400)
                        prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Describe how you want to transform the image...",
                            value="A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style",
                            lines=3,
                        )

                        with gr.Row():
                            guidance_scale = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                            num_inference_steps = gr.Slider(15, 50, value=28, step=1, label="Inference Steps")

                        with gr.Row():
                            lora_choices = get_lora_choices()
                            lora_selection = gr.Dropdown(
                                choices=lora_choices, value="None", label="LoRA Selection",
                                info="Select a LoRA to apply style transfer"
                            )
                            lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Strength")

                        edit_btn = gr.Button("🎨 Edit Image", variant="primary", size="lg")

                    with gr.Column(scale=2, min_width=400):
                        output_image = gr.Image(
                            label="Edited Image", show_download_button=True, interactive=False
                        )

                # Example prompts for single image
                gr.Markdown("### 💡 Example Prompts")
                example_prompts = [
                    "make this in pokraslampas style",
                    "A cyberpunk cityscape with neon lights and flying cars",
                    "An oil painting of a serene forest with magical glowing mushrooms",
                    "A professional headshot in corporate style",
                ]

                for i, example_prompt in enumerate(example_prompts):
                    with gr.Row():
                        btn = gr.Button(f"Example {i+1}: {example_prompt[:50]}...", variant="secondary")
                        btn.click(lambda p=example_prompt: p, outputs=prompt)

            # Batch Processing Tab (new functionality)
            with gr.TabItem("📦 Batch Processing", elem_id="batch-tab"):
                gr.Markdown("### Process multiple images from folders or ZIP archives")
                gr.Markdown("Upload a ZIP file containing images or select a folder. All images will be processed with the same prompt while preserving the original folder structure.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📁 Input Source")

                        # Universal file/folder upload area
                        file_upload = gr.File(
                            label="📦 Upload ZIP file or 📁 Select multiple files/folders",
                            file_count="multiple",
                            height=150,
                            interactive=True
                        )
                        gr.Markdown("*Drag and drop a ZIP file or select multiple files/folders*")

                        # Hidden component to store the selected input
                        input_source = gr.File(visible=False)

                        # Status display
                        input_status = gr.Markdown("*No input selected*", elem_id="input-status")

                        batch_prompt = gr.Textbox(
                            label="Batch Edit Prompt",
                            placeholder="This prompt will be applied to all images in the batch...",
                            lines=3,
                            value="Transform this image into a beautiful oil painting"
                        )

                        with gr.Row():
                            batch_guidance_scale = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                            batch_num_inference_steps = gr.Slider(15, 50, value=28, step=1, label="Inference Steps")

                        with gr.Row():
                            batch_lora_selection = gr.Dropdown(
                                choices=lora_choices, value="None", label="LoRA Selection"
                            )
                            batch_lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Strength")

                        process_batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("#### 📊 Results")
                        batch_summary = gr.Markdown("*Upload images and click 'Process Batch' to start.*")
                        batch_output = gr.File(label="📥 Download Processed Images", visible=True)

        # Event handlers for single image processing
        edit_btn.click(
            fn=edit_image,
            inputs=[input_image, prompt, guidance_scale, num_inference_steps, lora_selection, lora_strength],
            outputs=output_image,
            show_progress=True,
        )

        # Universal event handler for file uploads
        def update_input_source(files):
            if files is None or len(files) == 0:
                return None, "*No input selected*"

            # Check if it's a single ZIP file
            if len(files) == 1:
                file = files[0]
                file_path = file if isinstance(file, str) else (file.name if hasattr(file, 'name') else str(file))
                file_name = os.path.basename(file_path)

                if file_name.lower().endswith('.zip'):
                    return file_path, f"✅ **ZIP file selected:** {file_name}"
                else:
                    # Single non-ZIP file, treat as folder upload
                    folder_info = {
                        "type": "folder",
                        "files": files
                    }
                    return folder_info, f"✅ **File selected:** {file_name}"
            else:
                # Multiple files, treat as folder upload
                folder_info = {
                    "type": "folder",
                    "files": files
                }

                # Get folder name from first file
                first_file = files[0]
                if isinstance(first_file, str):
                    folder_name = os.path.dirname(first_file)
                    folder_name = os.path.basename(folder_name) if folder_name else "Multiple files"
                elif hasattr(first_file, 'name'):
                    folder_name = os.path.dirname(first_file.name)
                    folder_name = os.path.basename(folder_name) if folder_name else "Multiple files"
                else:
                    folder_name = "Multiple files"

                return folder_info, f"✅ **Files selected:** {folder_name} ({len(files)} files)"

        file_upload.change(fn=update_input_source, inputs=[file_upload], outputs=[input_source, input_status])

        def process_batch_wrapper(*args):
            """Wrapper function to handle batch processing and file return."""
            try:
                file_path, summary = process_batch(*args)
                if file_path is not None:
                    # Return file update and summary
                    return gr.update(value=file_path, visible=True), summary
                else:
                    # Return no file and summary with error
                    return gr.update(value=None, visible=False), summary
            except Exception as e:
                error_msg = f"❌ **Critical Error:** {str(e)}"
                return gr.update(value=None, visible=False), error_msg

        process_batch_btn.click(
            fn=process_batch_wrapper,
            inputs=[
                input_source, batch_prompt, batch_guidance_scale,
                batch_num_inference_steps, batch_lora_selection, batch_lora_strength
            ],
            outputs=[batch_output, batch_summary],
            show_progress=True,
        )

        # Note: File visibility is now handled in the process_batch_wrapper function

        # Add footer
        footer_text = """---
**🚀 Powered by [Flux Kontext](https://bfl.ai/announcements/flux-1-kontext-dev) and [Modal](https://modal.com)**

✨ **Features:**
- Single image editing with custom prompts
- Batch processing of folders and ZIP archives
- LoRA support for style transfer
- Automatic memory management
- Progress tracking for batch operations

This interface automatically scales based on usage and only charges for compute time used."""
        gr.Markdown(footer_text)

    # Enable queueing for handling multiple requests (increased for batch processing)
    demo.queue(max_size=20)

    return mount_gradio_app(app=web_app, blocks=demo, path="/")


@app.local_entrypoint()
def main():
    """Main entrypoint to run the UI."""
    import signal
    import sys

    def signal_handler(signum, _):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🔄 Checking LoRA files...")

    # Check local LoRA files (they will be included in the image)
    local_loras_path = Path("./loras")
    if local_loras_path.exists():
        safetensors_files = list(local_loras_path.glob("*.safetensors"))
        print(f"📁 Found {len(safetensors_files)} LoRA files locally:")
        for f in safetensors_files:
            print(f"   - {f.name}")

        if safetensors_files:
            print("✅ LoRA files will be included in the Modal image")
        else:
            print("❌ No .safetensors files found in ./loras directory")
    else:
        print("❌ ./loras directory not found")

    print("🚀 Starting UI...")
    try:
        ui.remote()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"\n❌ Error running UI: {e}")
    finally:
        print("👋 Goodbye!")

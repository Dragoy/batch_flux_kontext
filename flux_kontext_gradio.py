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
)

app = modal.App("example-image-to-image")

# LoRA files are now included in the image via add_local_dir
# No need for complex sync functions

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image


# ## Setting up and running Flux Kontext

# The Modal `Cls` defined below contains all the logic to set up and run Flux Kontext.

# The [container lifecycle](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta) decorator
# (`@modal.enter()`) ensures that the model is loaded into memory when a container starts, before it picks up any inputs.

# The `inference` method runs the actual model inference. It takes in an image as a collection of `bytes` and a string `prompt` and returns
# a new image (also as a collection of `bytes`).

# To avoid excessive cold-starts, we set the `scaledown_window` to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down.


@app.cls(
    image=image, gpu="B200", volumes=volumes, secrets=secrets, scaledown_window=240
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


# ## Gradio Web UI

# You can deploy the Gradio web interface with:
# ```bash
# modal deploy flux_kontext.py
# ```
# This will create a web interface accessible to anyone with the URL.


@app.function(
    image=web_image,
    min_containers=1,
    scaledown_window=60 * 20,
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

    # Create the Gradio interface with more explicit typing
    with gr.Blocks(
        title="Flux Kontext Image Editor",
    ) as demo:
        gr.Markdown("# Flux Kontext Image Editor")
        gr.Markdown(
            "Upload an image and provide a prompt to edit it using the Flux Kontext model. "
            "The model will transform your image to better match your prompt while preserving its structure and quality."
        )

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
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=2.5,
                        step=0.1,
                        label="Guidance Scale",
                    )
                    num_inference_steps = gr.Slider(
                        minimum=15,
                        maximum=50,
                        value=28,
                        step=1,
                        label="Inference Steps",
                    )

                # LoRA Controls
                with gr.Row():
                    # Get available LoRAs
                    lora_choices = get_lora_choices()
                    lora_selection = gr.Dropdown(
                        choices=lora_choices,
                        value="None",
                        label="LoRA Selection",
                        info="Select a LoRA to apply style transfer"
                    )
                    lora_strength = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="LoRA Strength",
                        info="Control the strength of the LoRA effect"
                    )

                edit_btn = gr.Button("Edit Image", variant="primary", size="lg")

            with gr.Column(scale=2, min_width=400):
                output_image = gr.Image(
                    label="Edited Image",
                    height=None,  # Auto height based on content
                    width=None,   # Auto width based on content
                    show_download_button=True,
                    container=True,  # Enable container scaling
                    interactive=False,  # Read-only output
                )

        # Example images and prompts
        gr.Markdown("## Example Prompts")

        # Simplified examples without potential problematic schema
        example_prompts = [
            "make this in pokraslampas style",
            "A cyberpunk cityscape with neon lights and flying cars",
            "An oil painting of a serene forest with magical glowing mushrooms",
            "A professional headshot in corporate style",
        ]

        for i, example_prompt in enumerate(example_prompts):
            with gr.Row():
                btn = gr.Button(
                    f"Example {i+1}: {example_prompt[:50]}...", variant="secondary"
                )
                # Use default parameter to capture the value
                btn.click(lambda p=example_prompt: p, outputs=prompt)

        # Set up the event handler
        edit_btn.click(
            fn=edit_image,
            inputs=[input_image, prompt, guidance_scale, num_inference_steps, lora_selection, lora_strength],
            outputs=output_image,
            show_progress=True,
        )

        # Add footer
        footer_text = """---
Powered by [Flux Kontext](https://bfl.ai/announcements/flux-1-kontext-dev) and [Modal](https://modal.com)

This interface automatically scales based on usage and only charges for compute time used."""
        gr.Markdown(footer_text)

    # Enable queueing for handling multiple requests
    demo.queue(max_size=10)

    return mount_gradio_app(app=web_app, blocks=demo, path="/")


@app.local_entrypoint()
def main():
    """Main entrypoint to run the UI."""
    import signal
    import sys

    def signal_handler(signum, _):
        """Handle shutdown signals gracefully."""
        print(f"\n� Received signal {signum}. Shutting down gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("�🔄 Checking LoRA files...")

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

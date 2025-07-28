# Batch Image Processor with FLUX.1-Kontext and LoRA

This project provides a script for batch processing of local images using the `FLUX.1-Kontext-dev` model, accelerated by Modal's serverless GPU platform. It applies transformations based on a text prompt and a LoRA (Low-Rank Adaptation) adapter.

## Features

- **Batch Processing**: Processes all images from a specified input directory (`imgs/`), including subdirectories.
- **AI-powered Transformation**: Uses the `black-forest-labs/FLUX.1-Kontext-dev` model via the `diffusers` library.
- **LoRA Customization**: Applies a `.safetensors` LoRA file from the `lora/` directory to customize the image generation.
- **Cloud-based GPU**: Leverages `modal.com` for running the intensive processing on high-performance GPUs (e.g., B200).
- **Persistent Storage**: Saves processed images to a `modal.Volume` named `batch-processed-images`.
- **Resumability**: Tracks progress automatically, allowing the script to be re-run to process only the remaining or failed images.
- **Preserves Structure**: Maintains the original directory structure of the input images in the output location.
- **Smart Resizing**: Automatically adjusts image resolution for optimal quality with FLUX, preserving aspect ratio by default.
- **Metadata Storage**: Saves processing parameters (prompt, seed, etc.) directly into the output PNG files for reproducibility.
- **Memory Optimization**: Uses CPU offloading to handle large models on GPUs with less VRAM.

## Getting Started

### Prerequisites

- Python 3.12+
- A [Modal](https://modal.com/) account and authenticated CLI (`pip install modal && modal token new`).
- A [Hugging Face](https://huggingface.co/) account with an access token.

### Setup

1.  **Clone the repository (or download the script).**

2.  **Configure Hugging Face Secret**:
    Create a Modal secret to store your Hugging Face token. This allows the script to download the model securely.
    ```bash
    modal secret create huggingface-secret HF_TOKEN=<your-hugging-face-token>
    ```

3.  **Prepare Directories**:
    - Create an `imgs/` directory in the same folder as the script. Place the images you want to process inside it. You can create any number of subdirectories to organize your images.
    - Create a `lora/` directory and place at least one `.safetensors` LoRA file inside it. The script will use the first one it finds.

## Usage

Execute the script using the `modal` CLI:

```bash
modal run batch_image_processor.py
```

The script will find all images, load the LoRA file, and begin processing. Processed images will be saved to a Modal Volume called `batch-processed-images`.

### Customizing the Run

You can pass arguments to customize the image generation:

```bash
modal run batch_image_processor.py --prompt "A vibrant, abstract painting" --lora-strength 0.8 --target-resolution 768 --preserve-aspect-ratio False
```

- `--prompt`: The text prompt to guide the image transformation.
- `--lora-strength`: The weight of the LoRA adapter (from 0.0 to 1.0).
- `--guidance-scale`: How much the generation should adhere to the prompt.
- `--num-inference-steps`: The number of steps in the diffusion process.
- `--target-resolution`: The target resolution for the shorter side of the image (default: 1024).
- `--preserve-aspect-ratio`: Set to `False` to force square images (default: `True`).
- `--random-seed-per-image`: Use a random seed for each image instead of a fixed one.
- `--save-metadata`: Set to `False` to disable saving metadata to PNG files.

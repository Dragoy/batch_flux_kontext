# Gemini Code Assistant Workspace Context

## Project: Batch Image Processor with FLUX.1-Kontext

This project contains Python scripts for batch processing and interactive editing of local images using the `FLUX.1-Kontext-dev` model, accelerated by Modal's GPU infrastructure. It applies transformations based on a text prompt and a LoRA (Low-Rank Adaptation) adapter.

### Key Features:

- **Batch Processing**: Processes all images from a specified input directory (`imgs/`).
- **Interactive Web UI**: Provides a Gradio interface for single image editing and batch processing from a web browser.
- **AI-powered Transformation**: Uses the `black-forest-labs/FLUX.1-Kontext-dev` model via the `diffusers` library.
- **LoRA Customization**: Applies a `.safetensors` LoRA file from the `lora/` directory to customize the image generation.
- **Cloud-based GPU**: Leverages `modal.com` for running the intensive processing on high-performance GPUs (e.g., B200).
- **Persistent Storage**: Saves processed images to a `modal.Volume` named `batch-processed-images`.
- **Resumability**: Tracks progress in a `progress.json` file within the volume, allowing the script to be re-run to process only the remaining or failed images.
- **Preserves Structure**: Maintains the original directory structure of the input images in the output location.

## Core Components

- **`batch_image_processor.py`**: The main script for command-line batch processing. It contains both the remote Modal class (`BatchProcessor`) for the actual image processing and the local entrypoint (`main`) to orchestrate the task.
- **`flux_kontext_gradio.py`**: A Gradio-based web UI for interactive image editing and batch processing. It allows users to upload images, provide prompts, and see the results in a web interface.
- **`flux_kontext_gradio_optimized.py`**: A refactored and optimized version of the Gradio UI, with better performance, error handling, and a more modular structure.

## How to Run

### Command-line Batch Processing

1.  **Prerequisites**:
    - Python 3.12 and required packages installed (`pip install modal diffusers ...`).
    - A Modal account and authentication (`modal token new`).
    - A Hugging Face account with an access token set up as a Modal secret (`modal secret create huggingface-secret HF_TOKEN=...`).

2.  **Setup Directories**:
    - Create an `imgs/` directory and place the images you want to process inside it. You can create subdirectories.
    - Create a `lora/` directory and place at least one `.safetensors` LoRA file inside it.

3.  **Execute the script**:
    ```bash
    modal run batch_image_processor.py
    ```

    You can also pass arguments to customize the run:
    ```bash
    modal run batch_image_processor.py --prompt "A vibrant, abstract painting" --lora-strength 0.8
    ```

### Interactive Web UI

1. **Prerequisites**: Same as for command-line processing.
2. **Setup Directories**: Same as for command-line processing.
3. **Execute the script**:
    ```bash
    modal run flux_kontext_gradio_optimized.py
    ```
   This will start a web server with the Gradio interface. You can then access it in your browser to interactively edit images.
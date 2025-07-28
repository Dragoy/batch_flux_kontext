# ---
# output-directory: "/tmp/stable-diffusion-batch"
# ---

# # Batch Image Processing with Flux Kontext + LoRA

# This program processes all images from a local imgs folder using Flux Kontext
# with LoRA adapters, saving results to a Modal Volume with progress tracking
# and resume capability.

import json
import os
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional
import time

import modal

diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

# Enhanced image with additional dependencies for batch processing
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "uv",
        "accelerate~=1.8.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "peft~=0.15.0",
        "huggingface-hub[hf-transfer]~=0.33.1",
        "Pillow~=11.0.0",
        "safetensors~=0.5.3",
        "transformers~=4.53.0",
        "sentencepiece~=0.2.0",
        "optimum-quanto~=0.2.7",
        "tqdm~=4.66.0",
    )
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

# Cache and storage volumes
CACHE_DIR = Path("/cache")
STORAGE_DIR = Path("/storage")

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
storage_volume = modal.Volume.from_name("batch-processed-images", create_if_missing=True)

volumes = {
    "/cache": cache_volume,
    "/storage": storage_volume
}

secrets = [modal.Secret.from_name("huggingface-secret")]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(CACHE_DIR),
    }
)

app = modal.App("batch-image-processor")

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    import os


@app.cls(
    image=image, gpu="B200", volumes=volumes, secrets=secrets, scaledown_window=240
)
class BatchProcessor:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16
        self.seed = 42
        self.device = "cuda"

        hf_token = os.environ.get("HF_TOKEN")

        # Load the base pipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            token=hf_token,
        ).to(self.device)

        print("Pipeline loaded successfully")

    @modal.method()
    def load_lora(self, lora_bytes: bytes, lora_filename: str, adapter_strength: float = 0.9):
        """Load LoRA weights with specified strength"""
        try:
            # LoRA weights are saved to a temporary path in the container
            # so that diffusers can load them.
            lora_path = CACHE_DIR / lora_filename
            lora_path.write_bytes(lora_bytes)

            print(f"Loading LoRA from temporary path: {lora_path}")
            self.pipe.load_lora_weights(
                str(lora_path.parent), weight_name=lora_path.name
            )

            # Set adapter strength
            if hasattr(self.pipe, "set_adapters"):
                adapter_names = self.pipe.get_active_adapters()
                self.pipe.set_adapters(adapter_names, adapter_weights=[adapter_strength] * len(adapter_names))

            print(f"LoRA loaded with strength: {adapter_strength}")
            
            # Clean up the temporary file
            lora_path.unlink()
            return True
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
            return False

    @modal.method()
    def process_image(
        self,
        image_bytes: bytes,
        prompt: str,
        filename: str,
        relative_path: str,  # Add relative path parameter
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
    ) -> Dict[str, any]:  # type: ignore
        """Process a single image and return result info"""
        start_time = time.time()
        try:
            
            # Load and resize input image
            init_image = load_image(Image.open(BytesIO(image_bytes))).resize((512, 512))

            # Generate processed image
            result = self.pipe(
                image=init_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
            )
            # Handle different return types from the pipeline
            if hasattr(result, 'images'):
                image = result.images[0]
            elif isinstance(result, tuple):
                # Get the first PIL Image from the tuple
                image = next((img for img in result if img is not None and hasattr(img, 'save')), result[0])
            else:
                image = result

            # Convert to bytes
            # Ensure we have a valid PIL Image
            if image is None or not isinstance(image, Image.Image):
                raise ValueError("Pipeline did not return a valid PIL Image")
            
            byte_stream = BytesIO()
            image.save(byte_stream, format="PNG")
            output_bytes = byte_stream.getvalue()
            
            processing_time = time.time() - start_time

            # Save to storage volume with preserved folder structure
            relative_dir = Path(relative_path).parent
            output_filename = f"{Path(filename).stem}_processed.png"
            output_path = STORAGE_DIR / "processed" / relative_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(output_bytes)

            return {
                "success": True,
                "filename": filename,
                "relative_path": relative_path,
                "output_path": str(output_path),
                "processing_time": processing_time,
                "output_size": len(output_bytes)
            }

        except Exception as e:
            return {
                "success": False,
                "filename": filename,
                "relative_path": relative_path,
                "error": str(e),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    @modal.method()
    def get_progress(self) -> Dict:
        """Get current progress from storage volume"""
        progress_path = STORAGE_DIR / "progress.json"
        if progress_path.exists():
            try:
                return json.loads(progress_path.read_text())
            except:
                pass
        return {
            "processed_files": [],
            "failed_files": [],
            "total_files": 0,
            "start_time": None,
            "last_update": None
        }

    @modal.method()
    def update_progress(self, progress_data: Dict):
        """Update progress file in storage volume"""
        progress_path = STORAGE_DIR / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_data["last_update"] = time.time()
        progress_path.write_text(json.dumps(progress_data, indent=2))


@app.local_entrypoint()
def main(
    imgs_dir: str = "imgs",
    lora_dir: str = "lora", 
    prompt: str = "make this in pokraslampas style",
    lora_strength: float = 0.9,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    batch_size: int = 1
):
    """
    Process all images in the imgs directory using Flux Kontext with LoRA
    
    Args:
        imgs_dir: Directory containing input images
        lora_dir: Directory containing LoRA files
        prompt: Text prompt for image transformation
        lora_strength: LoRA adapter strength (0.0-1.0)
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        batch_size: Number of images to process in parallel
    """
    
    # Convert relative paths to absolute
    current_dir = Path(__file__).parent
    imgs_path = current_dir / imgs_dir
    lora_path = current_dir / lora_dir
    
    if not imgs_path.exists():
        print(f"Images directory not found: {imgs_path}")
        return
    
    if not lora_path.exists():
        print(f"LoRA directory not found: {lora_path}")
        return

    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    def find_images_recursive(directory):
        """Recursively find all image files in directory"""
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                image_files.append(item)
            elif item.is_dir():
                find_images_recursive(item)
    
    find_images_recursive(imgs_path)
    
    if not image_files:
        print(f"No image files found in {imgs_path}")
        return

    # Find LoRA files
    lora_files = list(lora_path.glob("*.safetensors"))
    if not lora_files:
        print(f"No LoRA files found in {lora_path}")
        return
    
    lora_file = lora_files[0]  # Use first LoRA file found
    print(f"Using LoRA: {lora_file.name}")

    # Read LoRA file bytes
    lora_bytes = lora_file.read_bytes()

    # Initialize processor
    processor = BatchProcessor()
    
    # Load LoRA
    print("Loading LoRA...")
    success = processor.load_lora.remote(lora_bytes, lora_file.name, lora_strength)
    if not success:
        print("Failed to load LoRA, aborting")
        return

    # Get current progress
    progress = processor.get_progress.remote()
    processed_files = set(progress.get("processed_files", []))
    failed_files = set(progress.get("failed_files", []))
    
    # Filter out already processed files (using relative paths)
    remaining_files = []
    for f in image_files:
        relative_path = str(f.relative_to(imgs_path))
        if relative_path not in processed_files:
            remaining_files.append(f)
    
    if not remaining_files:
        print("All images already processed!")
        return
    
    print(f"Found {len(image_files)} total images")
    print(f"Already processed: {len(processed_files)}")
    print(f"Previously failed: {len(failed_files)}")
    print(f"Remaining to process: {len(remaining_files)}")
    
    # Update progress with total count
    if progress.get("start_time") is None:
        progress["start_time"] = time.time()
    progress["total_files"] = len(image_files)
    processor.update_progress.remote(progress)

    # Process images with progress tracking
    print(f"Starting batch processing with prompt: '{prompt}'")
    
    for i, image_file in enumerate(remaining_files, 1):
        try:
            # Read image
            image_bytes = image_file.read_bytes()
            relative_path = str(image_file.relative_to(imgs_path))
            
            # Process image
            print(f"Processing {i}/{len(remaining_files)}: {relative_path}")
            result = processor.process_image.remote(
                image_bytes=image_bytes,
                prompt=prompt,
                filename=image_file.name,
                relative_path=relative_path,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            # Update progress
            if result["success"]:
                processed_files.add(relative_path)
                print(f"✓ Completed {relative_path} in {result['processing_time']:.1f}s")
            else:
                failed_files.add(relative_path)
                print(f"✗ Failed to process {relative_path}: {result.get('error', 'Unknown error')}")
            
            # Update progress file
            progress.update({
                "processed_files": list(processed_files),
                "failed_files": list(failed_files),
                "total_files": len(image_files)
            })
            processor.update_progress.remote(progress)
            
        except Exception as e:
            relative_path = str(image_file.relative_to(imgs_path))
            print(f"✗ Unexpected error processing {relative_path}: {e}")
            failed_files.add(relative_path)

    # Final summary
    total_time = time.time() - progress["start_time"]
    print("Batch processing completed!")
    print(f"Successfully processed: {len(processed_files)} images")
    print(f"Failed: {len(failed_files)} images")
    print(f"Total time: {total_time:.1f} seconds")
    print("Results saved to Modal volume: batch-processed-images")
    
    if failed_files:
        print("To retry failed images, simply run the script again")
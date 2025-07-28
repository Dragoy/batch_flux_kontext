# ---
# output-directory: "/tmp/stable-diffusion-batch"
# ---

# # Batch Image Processing with Flux Kontext + LoRA

# This program processes all images from a local imgs folder using Flux Kontext
# with LoRA adapters, saving results to a Modal Volume with progress tracking
# and resume capability.

import json
import os
import random
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple
import time

import modal

diffusers_commit_sha = "2841504"

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

# Enhanced image processing utilities
def get_optimal_resolution(image, max_resolution: int = 1024) -> Tuple[int, int]:
    """
    Определяет оптимальное разрешение для Flux без потери качества
    
    Args:
        image: Входное изображение
        max_resolution: Максимальное разрешение (1024 для лучшего качества)
    
    Returns:
        (width, height): Оптимальные размеры кратные 64 (требование Flux)
    """
    width, height = image.size
    
    # Если изображение уже подходящего размера - не трогаем
    if width == max_resolution and height == max_resolution:
        return width, height
    
    # Для шрифтов и графики предпочитаем сохранять пропорции
    if width == height:
        # Квадратное изображение - используем max_resolution
        return max_resolution, max_resolution
    
    # Прямоугольное - сохраняем пропорции с границей max_resolution
    scale = min(max_resolution / width, max_resolution / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Округляем до кратных 64 (требование Flux architecture)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    # Минимальный размер 512x512
    new_width = max(new_width, 512)
    new_height = max(new_height, 512)
    
    return new_width, new_height

def smart_resize(image, target_resolution: int = 1024):
    """
    Умное изменение размера без потери качества
    """
    width, height = image.size
    target_width, target_height = get_optimal_resolution(image, target_resolution)
    
    # Если размер уже оптимальный - возвращаем как есть
    if width == target_width and height == target_height:
        return image
    
    # Используем высококачественный LANCZOS для шрифтов
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def create_png_metadata(
    original_filename: str,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    processing_time: Optional[float] = None
):
    """
    Создает PNG метаданные для сохранения параметров обработки
    """
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Processing Software", "Flux Kontext Batch Processor v2.0")
    metadata.add_text("Original Filename", original_filename)
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Guidance Scale", str(guidance_scale))
    metadata.add_text("Inference Steps", str(num_inference_steps))
    
    if seed is not None:
        metadata.add_text("Seed", str(seed))
    
    if target_resolution:
        metadata.add_text("Target Resolution", f"{target_resolution[0]}x{target_resolution[1]}")
    
    if processing_time:
        metadata.add_text("Processing Time (seconds)", f"{processing_time:.2f}")
    
    return metadata

with image.imports():
    import torch
    from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
    from diffusers.utils.loading_utils import load_image
    from PIL import Image, PngImagePlugin

@app.cls(
    image=image, gpu="B200", volumes=volumes, secrets=secrets, scaledown_window=240
)
class BatchProcessor:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16
        self.base_seed = None  # Будет установлен через параметры
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
        
        # Оптимизация памяти
        self.pipe.enable_model_cpu_offload()

    @modal.method()
    def load_lora(self, lora_bytes: bytes, lora_filename: str, adapter_strength: float = 0.9):
        """Load LoRA weights with specified strength"""
        try:
            # LoRA weights are saved to a temporary path in the container
            # so that diffusers can load them.
            lora_path = CACHE_DIR / lora_filename
            lora_path.write_bytes(lora_bytes)

            print(f"Loading LoRA from temporary path: {lora_path}")
            # Load LoRA weights with adapter name
            self.pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name="default"
            )

            # Set the adapter strength
            self.pipe.set_adapters(["default"], adapter_weights=[adapter_strength])
            print(f"LoRA loaded successfully with strength {adapter_strength}")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            raise

    @modal.method()
    def process_image(
        self,
        image_bytes: bytes,
        prompt: str,
        filename: str,
        relative_path: str,  # Add relative path parameter
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        target_resolution: int = 1024,  # Новый параметр разрешения
        preserve_aspect_ratio: bool = True,  # Новый параметр сохранения пропорций
        seed: Optional[int] = None,  # Новый параметр сида
        save_metadata: bool = True  # Новый параметр сохранения метаданных
    ) -> Dict[str, any]:  # type: ignore
        """Process a single image and return result info"""
        start_time = time.time()
        try:
            # Load input image without forced resize
            original_image = Image.open(BytesIO(image_bytes))
            original_size = original_image.size
            
            # Smart resize based on target resolution and aspect ratio preference
            if preserve_aspect_ratio:
                processed_image = smart_resize(original_image, target_resolution)
            else:
                # Force square output
                processed_image = original_image.resize(
                    (target_resolution, target_resolution), 
                    Image.Resampling.LANCZOS
                )
            
            final_size = processed_image.size
            print(f"Processing {filename}: {original_size} -> {final_size}")

            # Prepare image for pipeline
            init_image = load_image(processed_image)

            # Determine seed for generation
            if seed is not None:
                current_seed = seed
            elif self.base_seed is not None:
                current_seed = self.base_seed
            else:
                current_seed = random.randint(0, 2**32 - 1)
            
            # Create generator with seed
            generator = torch.Generator(device=self.device).manual_seed(current_seed)

            # Generate processed image with dynamic resolution
            result = self.pipe(
                image=init_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                generator=generator,
                height=final_size[1],  # Explicitly set dimensions
                width=final_size[0]
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
            
            # Create metadata if requested
            metadata = None
            if save_metadata:
                metadata = create_png_metadata(
                    original_filename=filename,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=current_seed,
                    target_resolution=final_size,
                    processing_time=time.time() - start_time
                )
            
            # Save with metadata or without
            byte_stream = BytesIO()
            if metadata:
                image.save(byte_stream, format="PNG", pnginfo=metadata)
            else:
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
                "output_size": len(output_bytes),
                "original_size": original_size,
                "final_size": final_size,
                "seed_used": current_seed
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
    def get_progress(self) -> Dict[str, any]:
        """Get current processing progress"""
        progress_file = STORAGE_DIR / "progress.json"
        if progress_file.exists():
            try:
                return json.loads(progress_file.read_text())
            except Exception as e:
                print(f"Error reading progress file: {e}")
                return {"processed_files": [], "failed_files": []}
        else:
            return {"processed_files": [], "failed_files": []}

    @modal.method()
    def update_progress(self, processed_files: List[str], failed_files: List[str]):
        """Update processing progress"""
        progress_file = STORAGE_DIR / "progress.json"
        progress_data = {
            "processed_files": list(set(processed_files)),
            "failed_files": list(set(failed_files)),
            "last_updated": time.time()
        }
        try:
            progress_file.write_text(json.dumps(progress_data, indent=2))
            storage_volume.commit()
        except Exception as e:
            print(f"Error updating progress: {e}")

@app.local_entrypoint()
def main(
    imgs_dir: str = "imgs",
    lora_dir: str = "lora", 
    prompt: str = "make this in pokraslampas style",
    lora_strength: float = 0.9,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    batch_size: int = 1,
    target_resolution: int = 1024,  # Новый параметр
    preserve_aspect_ratio: bool = True,  # Новый параметр
    random_seed_per_image: bool = False,  # Новый параметр
    save_metadata: bool = True  # Новый параметр
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
        target_resolution: Target resolution for processing (512, 768, 1024, etc.)
        preserve_aspect_ratio: Whether to preserve original aspect ratio
        random_seed_per_image: Whether to use random seed for each image
        save_metadata: Whether to save processing metadata to PNG
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
        for item in directory.iterdir():
            if item.is_dir():
                find_images_recursive(item)
            elif item.suffix.lower() in image_extensions:
                image_files.append(item)
    
    find_images_recursive(imgs_path)
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No images found to process")
        return
    
    # Find LoRA files
    lora_files = list(lora_path.glob("*.safetensors"))
    if not lora_files:
        print("No LoRA files found")
        return
    
    lora_file = lora_files[0]  # Use first LoRA file
    print(f"Using LoRA file: {lora_file.name}")
    
    # Read LoRA file
    lora_bytes = lora_file.read_bytes()
    
    # Initialize processor
    processor = BatchProcessor()
    
    # Load LoRA weights
    processor.load_lora.remote(lora_bytes, lora_file.name, lora_strength)
    
    # Get current progress
    progress = processor.get_progress.remote()
    processed_files = set(progress["processed_files"])
    failed_files = set(progress["failed_files"])
    
    # Filter out already processed files
    remaining_files = [
        f for f in image_files 
        if str(f.relative_to(imgs_path)) not in processed_files
        and str(f.relative_to(imgs_path)) not in failed_files
    ]
    
    print(f"Processing {len(remaining_files)} remaining images...")
    
    if not remaining_files:
        print("No remaining files to process")
        return
    
    # Process images
    for i, image_file in enumerate(remaining_files, 1):
        print(f"Processing {i}/{len(remaining_files)}: {image_file.name}")
        
        # Read image file
        image_bytes = image_file.read_bytes()
        relative_path = str(image_file.relative_to(imgs_path))
        
        # Determine seed
        seed = None
        if random_seed_per_image:
            seed = random.randint(0, 2**32 - 1)
        
        # Process image
        result = processor.process_image.remote(
            image_bytes=image_bytes,
            prompt=prompt,
            filename=image_file.name,
            relative_path=relative_path,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            target_resolution=target_resolution,
            preserve_aspect_ratio=preserve_aspect_ratio,
            seed=seed,
            save_metadata=save_metadata
        )
        
        # Update progress
        if result["success"]:
            processed_files.add(result["relative_path"])
            print(f"✓ Processed {result['filename']} in {result['processing_time']:.2f}s")
            print(f"  Original size: {result['original_size']} -> Final size: {result['final_size']}")
            print(f"  Seed used: {result['seed_used']}")
        else:
            failed_files.add(result["relative_path"])
            print(f"✗ Failed to process {result['filename']}: {result['error']}")
        
        # Update progress file every 5 images or at the end
        if i % 5 == 0 or i == len(remaining_files):
            processor.update_progress.remote(list(processed_files), list(failed_files))
            print(f"Progress updated: {len(processed_files)} processed, {len(failed_files)} failed")
    
    print(f"Processing complete: {len(processed_files)} processed, {len(failed_files)} failed")

if __name__ == "__main__":
    main()
import torch
from pathlib import Path
from functools import partial
from typing import Iterator, List, Tuple
from PIL import Image
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_imports
import time
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

# Configuration options
OVERWRITE = True  # Boolean option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
BATCH_SIZE = 7 # How many images to process at one time. A 24gb VRAM 3090 can handle 7. A 6gb VRAM GPU can handle a batch size of 1.
PRINT_PROCESSING_STATUS = False  # Option to print processing status of images
PRINT_CAPTIONS = False  # Option to print captions to the console
DETAIL_MODE = 3 # The level of verbosity for the output caption.

print(f"Captioning with batch size: {BATCH_SIZE}")

def fixed_get_imports(filename: str | Path) -> List[str]:
    imports = get_imports(filename)
    return [imp for imp in imports if imp != "flash_attn"] if str(filename).endswith("modeling_florence2.py") else imports

def download_and_load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device available: {device}')

    model_path = Path("models") / model_name.replace('/', '_')
    if not model_path.exists():
        print(f"Downloading {model_name} model to: {model_path}")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)

    print(f"Loading model {model_name}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")
    model = torch.compile(model, mode="reduce-overhead")
    return model, processor

def load_image_paths_recursive(folder_path: str) -> Iterator[Path]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return (
        path for path in Path(folder_path).rglob("*")
        if path.suffix.lower() in valid_extensions and (OVERWRITE or not path.with_suffix('.txt').exists())
    )

def run_model_batch(image_paths: List[Path], model: AutoModelForCausalLM, processor: AutoProcessor,
                    task: str = 'caption', num_beams: int = 3, max_new_tokens: int = 1024, detail_mode: int = DETAIL_MODE) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompt = {1: '<CAPTION>', 2: '<DETAILED_CAPTION>', 3: '<MORE_DETAILED_CAPTION>'}.get(detail_mode, '<MORE_DETAILED_CAPTION>')

    inputs = {
        "input_ids": [],
        "pixel_values": []
    }

    for image_path in image_paths:
        if PRINT_PROCESSING_STATUS:
            print(f"Processing image: {image_path}")
        with Image.open(image_path).convert("RGB") as img:
            input_data = processor(text=prompt, images=img, return_tensors="pt", do_rescale=False)
            inputs["input_ids"].append(input_data["input_ids"])
            inputs["pixel_values"].append(input_data["pixel_values"])

    # Keep input_ids as Long type and only convert pixel_values to bfloat16
    inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(device)
    inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(device).to(torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )

    results = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return [result.replace('</s>', '').replace('<s>', '').replace('<pad>', '') for result in results]


def process_images_recursive(paths: Iterator[Path], model: AutoModelForCausalLM, processor: AutoProcessor, batch_size: int = 8) -> Tuple[int, float]:
    start_time = time.time()
    total_images = 0

    # Convert paths to a list
    path_list = list(paths)
    num_batches = len(path_list) // batch_size + (1 if len(path_list) % batch_size > 0 else 0)

    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch = path_list[i*batch_size:(i+1)*batch_size]
        # Use DETAIL_MODE variable here
        captions = run_model_batch(batch, model, processor, task='caption', detail_mode=DETAIL_MODE)
        for path, caption in zip(batch, captions):
            caption = f"{PREPEND_STRING}{caption}{APPEND_STRING}"
            if PRINT_CAPTIONS:
                print(f"Caption for {path}: {caption}")
            path.with_suffix('.txt').write_text(caption)
            total_images += 1

    total_time = time.time() - start_time
    return total_images, total_time

# Main execution
model_name = 'microsoft/Florence-2-large'
model, processor = download_and_load_model(model_name)

# Process images in the /input/ folder
folder_path = Path(__file__).parent / "input"
total_images, total_time = process_images_recursive(load_image_paths_recursive(folder_path), model, processor, batch_size=BATCH_SIZE)

print(f"Total images captioned: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")

# Fix for divide-by-zero when calculating average time per image
if total_images > 0:
    print(f"Average time per image: {total_time / total_images:.2f} seconds")
else:
    print("No images were processed, so no average time to display.")

# Count the number of files in the directory
file_count = len(list(folder_path.iterdir()))
print(f"Total files in folder: {file_count}")

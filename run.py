from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os
from accelerate import Accelerator

accelerator = Accelerator()
def load_model():
    # Tự động chọn dtype và device phù hợp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )

    base, refiner = accelerator.prepare(base, refiner)

    if device == "cuda":
        try:
            base.to(device)
            refiner.to(device)
        except RuntimeError:
            accelerator.free_memory()
    else:
        accelerator.free_memory()

    base.enable_attention_slicing()
    refiner.enable_attention_slicing()

    return base, refiner

base, refiner = load_model()

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_next_image_number():
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("output") and f.endswith(".jpg")]
    if not existing_files:
        return 1
    numbers = [int(f.replace("output", "").replace(".jpg", "")) for f in existing_files if f.replace("output", "").replace(".jpg", "").isdigit()]
    return max(numbers) + 1 if numbers else 1

prompt = "2D black and white architectural floor plan, top-down CAD style, two bedrooms, one bathroom, kitchen, living and dining room, dimension annotations, total area 80m², scale 1:100, clean technical drawing, no color."
n_steps = 100
high_noise_frac = 0.8
width = 1024
height =1024
guidance_scale = 12.5 

image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    width=width,
    height=height,
    guidance_scale=guidance_scale  
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
    guidance_scale=guidance_scale  
).images[0]

image_number = get_next_image_number()
output_path = os.path.join(OUTPUT_DIR, f"output{image_number}.jpg")
image.save(output_path)
print(f"Image saved at: {output_path}")
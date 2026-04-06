import torch
from diffusers import StableDiffusionXLPipeline
import os
import pandas as pd
import random

# 1. Setup Directory and Model
OUTPUT_DIR = "cfcs_pilot_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SDXL Pipeline... (This might take a minute)")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
pipe.to("cuda") # Move to GPU

# 2. Define the Counter-Factual Categories and Prompts
# We define the Prompt, the "Visual" token (reality), and the "Prior" token (expectation)
prompt_templates = [
    {
        "category": "physics",
        "prompt": "A realistic photo of an apple floating upwards into the sky away from a tree branch.",
        "query": "In this image, the apple is moving...",
        "visual_token": "up",
        "prior_token": "down"
    },
    {
        "category": "physics",
        "prompt": "A high-quality photograph of a campfire burning brightly entirely underwater in the ocean.",
        "query": "The fire in this image is currently...",
        "visual_token": "burning",
        "prior_token": "extinguished"
    },
    {
        "category": "scale",
        "prompt": "A macro photo of a tiny adult elephant sitting comfortably on the back of a giant ladybug.",
        "query": "Comparing the two animals, the elephant is...",
        "visual_token": "smaller",
        "prior_token": "larger"
    },
    {
        "category": "function",
        "prompt": "A photo of a person holding a soft slice of bread and using it to hammer a metal nail into wood.",
        "query": "The object being used to hit the nail is made of...",
        "visual_token": "bread",
        "prior_token": "metal"
    },
    {
        "category": "environment",
        "prompt": "A landscape photo of a dense forest where all the leaves and grass are a bright, vibrant red.",
        "query": "The predominant color of the foliage here is...",
        "visual_token": "red",
        "prior_token": "green"
    }
]

# 3. Generate the Dataset (Scaling to 100 images)
# We will generate 20 variations of each of the 5 templates by changing the random seed
IMAGES_PER_TEMPLATE = 20
metadata = []

print(f"Generating {len(prompt_templates) * IMAGES_PER_TEMPLATE} images...")

image_counter = 0
for template in prompt_templates:
    for i in range(IMAGES_PER_TEMPLATE):
        image_counter += 1
        filename = f"image_{image_counter:03d}_{template['category']}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Generate image with a unique seed for variation
        seed = random.randint(0, 1000000)
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Run inference
        image = pipe(
            prompt=template["prompt"], 
            generator=generator,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save image
        image.save(filepath)
        
        # Log metadata
        metadata.append({
            "image_id": filename,
            "filepath": filepath,
            "category": template["category"],
            "generation_prompt": template["prompt"],
            "vqa_query": template["query"],
            "target_visual": template["visual_token"],
            "target_prior": template["prior_token"],
            "seed": seed
        })
        print(f"Saved {filename}")

# 4. Save Metadata to CSV
df = pd.DataFrame(metadata)
csv_path = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")
df.to_csv(csv_path, index=False)

print(f"\nSuccess! Generated {image_counter} images.")
print(f"Metadata saved to {csv_path}")
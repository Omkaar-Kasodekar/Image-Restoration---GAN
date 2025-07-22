from datasets import load_dataset
from PIL import Image
import io
import os

dataset = load_dataset("huggan/night2day", split="train").select(range(20120))

# Save images locally
os.makedirs("./night2day/night", exist_ok=True)
os.makedirs("./night2day/day", exist_ok=True)

for idx, item in enumerate(dataset):
    # Save night image
    if isinstance(item["imageA"], dict):
        Image.open(io.BytesIO(item["imageA"]["bytes"])).convert("RGB").save(f"./night2day/night/{idx}.png")
    else:
        item["imageA"].convert("RGB").save(f"./night2day/night/{idx}.png")
    
    # Save day image
    if isinstance(item["imageB"], dict):
        Image.open(io.BytesIO(item["imageB"]["bytes"])).convert("RGB").save(f"./night2day/day/{idx}.png")
    else:
        item["imageB"].convert("RGB").save(f"./night2day/day/{idx}.png")

print(f"Done! Saved {len(dataset)} image pairs to ./night2day/")
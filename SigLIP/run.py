#import requests # later
import torch
from PIL import Image

from model import load_model_SigLIP2

device = "mps"
model_name = "google/siglip2-base-patch16-256"

model, processor = load_model_SigLIP2(model_name, device=device)

image = Image.open("/Users/simonholm/GitHub/multi-modal-Mimir/test_images/shuggibuggidubbi.png").convert("RGB")


inputs = processor(images = image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

patches = outputs.last_hidden_state

print("patch size:" + str(patches.shape))
print("patch:" + str(patches))
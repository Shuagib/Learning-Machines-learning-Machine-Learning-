import sys
from pathlib import Path

# Allow running as a plain script: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#import requests # later
import torch
from PIL import Image

from model import load_model

device = "mps"

model, processor = load_model(device=device)

image = Image.open("/Users/simonholm/GitHub/multi-modal-Mimir/test_images/shuggibuggidubbi.png").convert("RGB")


inputs = processor(images = image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

emb = outputs.pooler_output.cpu()
patches = outputs.last_hidden_state

print(emb.shape)
print(emb[0, :10])
print(emb.norm(dim=-1))
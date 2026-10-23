import requests # later
import torch
from PIL import Image

from SigLIP.model import load_model

device = "mps"

model, processor = load_model(device)

image = Image.open("/Users/simonholm/GitHub/Learning-Machines-learning-Machine-Learning-/test_images/shuggibuggidubbi.png").convert("RGB")


inputs = processor(images = image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

emb = outputs.pooler_output
patches = outputs.last_hidden_state

print(emb.shape)
print(emb[0, :10])
print(emb.norm(dim=-1))
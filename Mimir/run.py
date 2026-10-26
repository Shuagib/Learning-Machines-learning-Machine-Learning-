#import requests # later
import torch

from model import load_model_Mimir

device = "mps"
model, tokenizer = load_model_Mimir(device=device) # Mimir model load

prompts = [
    "Hvem er ham fyren her",
    "Har denne mand tøj på",
    "Hvor gammel er manden på billedet"
]
tokenizer.padding_side = "left"

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_token=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output_ids[0,inputs["input_ids"].shape[1]:]
output = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(output)
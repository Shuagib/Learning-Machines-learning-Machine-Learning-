#import requests # later
import torch

from model import load_model_Mimir

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, tokenizer = load_model_Mimir(device=device) # Mimir model load

# prompts = [
#     "Hvem er ham fyren her",
#     "Har denne mand tøj på",
#     "Hvor gammel er manden på billedet"
# ]

prompts = [
    {"role": "user", "content": "Hvad er Danmarks hovedstad?"},
    # {"role": "user", "content": "Hvilket land er verdens bedste land"},
    # {"role": "user", "content": "Hvad er Danmarks national ret"},
]

tokenizer.padding_side = "left"

inputs = tokenizer.apply_chat_template(
    prompts, 
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt", 
    ).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)

text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(text)
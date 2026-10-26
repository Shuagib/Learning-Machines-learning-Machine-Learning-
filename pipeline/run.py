import torch

from vision_model.model import SigLIP2
from projector.model import MLP
from language_model.model import Mimir

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
vm_model_name = "google/siglip2-base-patch16-256"
llm_model_name = "danish-foundation-models/DFM-Mimir-v1.5"

MLP_IN_DIM = 768
MLP_HID_DIM = 1024
MLP_OUT_DIM = 24

vm = SigLIP2(model_name=vm_model_name, device=device)
mlp = MLP(MLP_IN_DIM, MLP_HID_DIM, MLP_OUT_DIM, device=device)
llm = Mimir(model_name=vm_model_name, device=device)

vision_prompts = [
    {"role": "user", "content": "<image> Beskriv manden på billedet"}
]



for elem in vision_prompts:
    elem_embed = llm.embed(elem)
    print(elem_embed)
    print(elem_embed.shape)


images = None
import torch

from SigLIP.model import load_model_SigLIP2
from MLP.model import load_model_MLP
from Mimir.model import load_model_Mimir

MLP_IN_DIM = 768
MLP_HID_DIM = 1024
MLP_OUT_DIM = 24

VM = load_model_SigLIP2()
MLP = load_model_MLP(MLP_IN_DIM, MLP_HID_DIM, MLP_OUT_DIM)

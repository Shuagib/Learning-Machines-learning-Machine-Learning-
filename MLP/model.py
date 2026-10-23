import torch.nn as nn
from transformers import SiglipVisionModel, AutoImageProcessor

class MLP(nn.Module):
        def __init__(self, hidden_dim):
              super(MLP, self).__init__()
              self.fc = nn.Linear()

            

    def  load_model(hidden_dim: int, device: str = "mps"):
        model = MLP(hidden_dim)
        return model
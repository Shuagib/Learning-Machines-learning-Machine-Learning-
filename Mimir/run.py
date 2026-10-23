import sys
from pathlib import Path

# Allow running as a plain script: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#import requests # later
import torch

from model import load_model

device = "mps"
model, tokenizer = load_model(device=device) # Mimir model load


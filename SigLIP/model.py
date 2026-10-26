from transformers import SiglipVisionModel, AutoImageProcessor

def  load_model_SigLIP2(name: str = "google/siglip2-large-patch16-256", device: str = "mps"):
    model = SiglipVisionModel.from_pretrained(name, device_map="auto", attn_implementation="sdpa").to(device).eval()
    processor = AutoImageProcessor.from_pretrained(name)
    return model, processor
from transformers import SiglipVisionModel, AutoImageProcessor

def  load_model(name: str = "google/siglip2-giant-opt-patch16-384", device: str = "mps"):
    model = SiglipVisionModel.from_pretrained(name, device_map="auto", attn_implementation="sdpa").to(device).eval()
    processor = AutoImageProcessor.from_pretrained(name)
    return model, processor
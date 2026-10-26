from transformers import AutoTokenizer, AutoModelForCausalLM

def  load_model_Mimir(name: str = "danish-foundation-models/DFM-Mimir-v1.5", device: str = "mps"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto").to(device)
    return model, tokenizer
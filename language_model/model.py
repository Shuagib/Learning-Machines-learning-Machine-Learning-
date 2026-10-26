import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mimir():
    def __init__(self, model_name: str, device: str):
        self.model, self.tokenizer = self.load_model_Mimir(name=model_name, device=device)
        self.embedding = self.model.get_input_embeddings() 

    def load_model_Mimir(name: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto").to(device)
        return model, tokenizer

    def embed(self, prompt: str):
        tokens = self.tokenizer.apply_chat_template(
                [prompt], 
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt", 
                ).to(self.model.device)
        return self.embedding(tokens)

    def generate_and_decode(self, input_embeddings: torch.Tensor, max_new_tokens: int = 256) -> str:
        attention_mask = torch.ones(input_embeddings.shape[0:2], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
                    outputs = self.model.generate(
                        input_embeds=input_embeddings,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
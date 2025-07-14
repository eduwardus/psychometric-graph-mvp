# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:46:32 2025

@author: eggra
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Silencia advertencia

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict

class OptimizedPsychometricGenerator:
    def __init__(self):
        """Configuración optimizada para Windows/CPU"""
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="./model_cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.construct = "depresión"

    def generate_items(self, existing_items: List[str], n_items: int = 3) -> List[Dict]:
        """Generación con barra de progreso"""
        from tqdm import tqdm
        
        prompt = f"""Genera {n_items} ítems sobre {self.construct} como estos:\n"""
        prompt += "\n".join(existing_items) + "\nFormato: 'Texto | relevancia: 0.X'"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=n_items,
            do_sample=True
        )
        
        return [
            {"text": self.tokenizer.decode(seq, skip_special_tokens=True).split("|")[0].strip(),
             "relevance": 0.8}  # Valor dummy para prueba
            for seq in tqdm(outputs, desc="Generando ítems")
        ]

# Uso
if __name__ == "__main__":
    generator = OptimizedPsychometricGenerator()
    items = generator.generate_items([
        "Me siento sin energía durante el día",
        "Lloro con frecuencia sin razón aparente"
    ])
    
    for i, item in enumerate(items, 1):
        print(f"{i}. {item['text']} (Relevancia: {item['relevance']:.1f})")
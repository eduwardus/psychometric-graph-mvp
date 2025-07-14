# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:29:28 2025

@author: eggra
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc  # Para liberar memoria manualmente
import os

# =============================================
# CONFIGURACIÓN INICIAL (AJUSTA SEGÚN TU CPU/RAM)
# =============================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Modelo pequeño
MAX_TOKENS = 30  # Texto corto para ahorrar RAM
DEVICE = "cpu"   # Fuerza uso de CPU (aunque no tengas GPU)

# =============================================
# FUNCIÓN PRINCIPAL (OPTIMIZADA PARA 1GB RAM)
# =============================================
def generar_item_psicometrico(constructo: str, ejemplo_item: str) -> str:
    """
    Genera un ítem psicométrico nuevo basado en un ejemplo.
    - constructo: "depresión", "ansiedad", etc.
    - ejemplo_item: Texto de ejemplo para guiar a la IA.
    """
    
    # 1. Cargar SOLO el tokenizer (usa poca RAM)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Cargar el modelo con ajustes de bajo consumo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Mitad de precisión (ahorra RAM)
        low_cpu_mem_usage=True,     # Evita picos de memoria
        device_map=DEVICE           # Fuerza CPU
    )
    
    # 3. Prompt minimalista
    prompt = f"""
    Genera UNA oración para un test de {constructo}.
    Formato: "Me siento [frase]". Ejemplo: "{ejemplo_item}".
    Respuesta:"""
    
    # 4. Codificar input (truncar para ahorrar RAM)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128  # ¡No aumentar en 1GB RAM!
    ).to(DEVICE)
    
    # 5. Liberar RAM antes de generar
    gc.collect()
    
    # 6. Generar texto (configuración ultra-ligera)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=0.7,  # Controla la creatividad
        num_beams=1,       # Desactiva búsqueda por haz (ahorra RAM)
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 7. Decodificar y limpiar el output
    item_generado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    item_limpio = item_generado.split("Respuesta:")[-1].strip()
    
    # 8. Liberar toda la RAM posible
    del model
    del tokenizer
    gc.collect()
    
    return item_limpio

# =============================================
# EJEMPLO DE USO (GENERA 1 ÍTEM POR EJECUCIÓN)
# =============================================
if __name__ == "__main__":
    # Configuración básica
    constructo = "depresión"
    ejemplo = "Me siento cansado sin razón"
    
    # Generar ítem
    try:
        nuevo_item = generar_item_psicometrico(constructo, ejemplo)
        print(f"\nÍTEM GENERADO PARA '{constructo.upper()}':")
        print("----------------------------------")
        print(nuevo_item)
        print("----------------------------------")
    except Exception as e:
        print(f"Error (RAM insuficiente?): {str(e)}")
        print("Prueba reduciendo MAX_TOKENS o usando un modelo más pequeño.")
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:54:15 2025

@author: eggra
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import os
import time
import psutil  # Para monitoreo de recursos
from threading import Thread  # Para mostrar progreso

# =============================================
# CONFIGURACIÓN PARA MONITOREO
# =============================================
def monitor_resources():
    """Muestra el uso de recursos cada 2 segundos"""
    while getattr(monitor_resources, "running", True):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"\n[Monitor] RAM libre: {mem.available/1024/1024:.1f} MB | "
              f"Swap usado: {swap.used/1024/1024:.1f} MB | "
              f"CPU: {psutil.cpu_percent()}%")
        time.sleep(2)

# =============================================
# CONFIGURACIÓN DEL MODELO (TinyLlama FP16)
# =============================================
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"
TORCH_DTYPE = torch.float32
MAX_TOKENS = 15  # Reducido para mayor velocidad

# =============================================
# FUNCIÓN PRINCIPAL CON DIAGNÓSTICO
# =============================================
def generar_item_con_monitoreo(constructo: str, ejemplo: str) -> str:
    try:
        # Iniciar monitor
        monitor_resources.running = True
        Thread(target=monitor_resources, daemon=True).start()

        print("\n[1/4] Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        gc.collect()

        print("[2/4] Cargando modelo (esto puede tardar)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            device_map=DEVICE,
            offload_folder="./offload"
        )
        gc.collect()

        print("[3/4] Preparando prompt...")
        prompt = f"Genera un ítem sobre {constructo} como este ejemplo: '{ejemplo}'. Respuesta:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100).to(DEVICE)
        gc.collect()

        print("[4/4] Generando texto...")
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            num_beams=1
        )
        generation_time = time.time() - start_time

        item = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return item.split("Respuesta:")[-1].strip(), generation_time

    except Exception as e:
        return f"Error: {str(e)}", 0

    finally:
        monitor_resources.running = False
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        gc.collect()

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("=== Generador de Ítems Psicométricos con TinyLlama ===")
    print("Monitorizando recursos en tiempo real...\n")

    constructo = "depresión"
    ejemplo = "Me siento cansado incluso después de dormir"

    item, tiempo = generar_item_con_monitoreo(constructo, ejemplo)

    print("\n" + "="*50)
    if not item.startswith("Error"):
        print(f"ÍTEM GENERADO EN {tiempo:.1f} SEGUNDOS:")
        print("-"*30)
        print(item)
    else:
        print(f"FALLO: {item}")
    print("="*50)

    # Mostrar uso final de recursos
    mem = psutil.virtual_memory()
    print(f"\nUso final de RAM: {mem.used/1024/1024:.1f} MB / {mem.total/1024/1024:.1f} MB")
    print(f"Swap usado: {psutil.swap_memory().used/1024/1024:.1f} MB")
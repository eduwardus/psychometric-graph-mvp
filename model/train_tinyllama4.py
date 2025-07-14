from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import psutil
import time

# =============================================
# CONFIGURACIÓN PARA 1GB RAM
# =============================================
MODEL_NAME = "gpt2"  # Modelo ligero (300MB)
DEVICE = "cpu"       # Fuerza uso de CPU
MAX_TOKENS = 20      # Texto corto para ahorrar RAM

# =============================================
# MONITOREO DE RECURSOS EN TIEMPO REAL
# =============================================
def print_memory_usage():
    """Muestra uso actual de RAM y swap"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"\n[Memoria] Libre: {mem.available/1024/1024:.1f} MB | "
          f"Swap usado: {swap.used/1024/1024:.1f} MB | "
          f"CPU: {psutil.cpu_percent()}%")

# =============================================
# FUNCIÓN PRINCIPAL
# =============================================
def generar_item_psicometrico(constructo: str, ejemplo: str) -> str:
    try:
        # 1. Cargar tokenizer (poca RAM)
        print("\n[1/3] Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print_memory_usage()

        # 2. Cargar modelo
        print("[2/3] Cargando modelo GPT-2...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE,
            torch_dtype=torch.float32
        )
        print_memory_usage()

        # 3. Generar ítem
        print("[3/3] Generando texto...")
        prompt = f"Genera un ítem sobre {constructo} como: '{ejemplo}'. Respuesta:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=50)
        
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7
        )
        
        item = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return item.split("Respuesta:")[-1].strip(), time.time() - start_time

    except Exception as e:
        return f"Error: {str(e)}", 0

# =============================================
# EJECUCIÓN
# =============================================
if __name__ == "__main__":
    print("=== Generador de Ítems Psicométricos (GPT-2) ===")
    print(f"Modelo: {MODEL_NAME} | Dispositivo: {DEVICE}")
    
    constructo = "depresión"
    ejemplo = "Me siento cansado incluso después de dormir"
    
    item, tiempo = generar_item_psicometrico(constructo, ejemplo)
    
    print("\n" + "="*50)
    if not item.startswith("Error"):
        print(f"ÍTEM GENERADO EN {tiempo:.1f} SEGUNDOS:")
        print("-"*30)
        print(item)
    else:
        print(f"FALLO: {item}")
    print("="*50)
    print_memory_usage()
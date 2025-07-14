from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

# Configuración
model_name = os.getenv("MODEL_NAME")
constructo = os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Plantilla de prompt
prompt_template = """
[CONTEXTO PSICOMÉTRICO]
Constructo: {construct}
Ítems existentes:
{existing_items}

[GENERAR NUEVO ÍTEM VÁLIDO]
"""

# Generar nuevo ítem
def generate_item(construct, existing_items):
    inputs = tokenizer(
        prompt_template.format(construct=construct, existing_items="\n".join(existing_items)),
        return_tensors="pt"
    )
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ejemplo de uso
if __name__ == "__main__":
    existing = ["Me siento desanimado la mayor parte del día"]
    new_item = generate_item(constructo, existing)
    print("Nuevo ítem generado:", new_item.split("]")[-1].strip())

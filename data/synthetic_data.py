import pandas as pd
from irtpy.datagen import SyntheticData
import os

constructo = os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")

# Ítems base para el constructo
items = {
    "depresión": [
        "Me siento triste sin razón",
        "Tengo dificultad para relajarme",
        "Siento que he perdido interés en todo"
    ],
    "ansiedad": [
        "Me siento nervioso sin motivo aparente",
        "Experimento palpitaciones cardíacas frecuentes"
    ]
}

# Generar datos sintéticos
datagen = SyntheticData(n_items=len(items[constructo]), n_subjects=300)
data, _ = datagen.generate()

# Guardar como CSV
df = pd.DataFrame(data, columns=items[constructo])
df.to_csv(f"data/{constructo}_synthetic.csv", index=False)
print(f"Datos sintéticos guardados: data/{constructo}_synthetic.csv")

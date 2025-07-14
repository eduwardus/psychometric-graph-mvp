import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# --- Configuración ---
CONSTRUCTO = os.getenv("CONSTRUCTO_PRINCIPAL", "depresion")  # Sin tildes en variables
N_PARTICIPANTES = 500
ESCALA_LIKERT = [1, 2, 3, 4, 5]
MISSING_RATE = 0.05

# --- Ítems (sin tildes en claves) ---
ITEMS = {
    "depresion": [  # Clave sin tilde
        "Me siento triste sin razon",
        "Tengo dificultad para disfrutar actividades",
        "Me siento cansado/a casi todo el dia",
        "Tengo pensamientos de inutilidad",
        "Mi apetito ha cambiado significativamente"
    ],
    "ansiedad": [
        "Me siento nervioso/a sin motivo",
        "Experimento palpitaciones frecuentes",
        "Tengo dificultad para concentrarme",
        "Me preocupan situaciones cotidianas",
        "Siento tension muscular constante"  # Sin tilde
    ]
}

# --- 1. Generar parámetros (nombres en inglés por convención) ---
def generar_parametros_items(n_items):
    np.random.seed(42)
    return {
        "discrimination": np.random.uniform(1.0, 3.0, n_items),  # Clave en inglés
        "difficulty": np.random.uniform(-2.0, 2.0, n_items)       # Clave en inglés
    }

# --- 2. Simular respuestas (versión corregida) ---
def simular_respuestas(n_participantes, items, params):
    n_items = len(items)
    corr_matrix = np.eye(n_items) * 0.7
    for i in range(n_items - 1):
        corr_matrix[i, i+1] = 0.4
        corr_matrix[i+1, i] = 0.4
    
    latentes = multivariate_normal.rvs(
        mean=np.zeros(n_items),
        cov=corr_matrix,
        size=n_participantes
    )
    
    # Línea CORREGIDA (usa claves en inglés):
    probas = 1 / (1 + np.exp(-(params["discrimination"] * (latentes - params["difficulty"]))))
    
    respuestas = np.floor(probas * (len(ESCALA_LIKERT) - 1)) + 1
    mask = np.random.random(respuestas.shape) < MISSING_RATE
    respuestas[mask] = np.nan
    return respuestas

# --- 3. Ejecución ---
items_seleccionados = ITEMS[CONSTRUCTO]
params = generar_parametros_items(len(items_seleccionados))
respuestas = simular_respuestas(N_PARTICIPANTES, items_seleccionados, params)

# --- 4. Guardar datos ---
df = pd.DataFrame(respuestas, columns=items_seleccionados)
os.makedirs("data", exist_ok=True)
output_path = f"data/{CONSTRUCTO}_likert_psychometric.csv"
df.to_csv(output_path, index_label="participant_id")

print(f"✅ Datos guardados en {output_path}")
print("📊 Estadísticas descriptivas:\n", df.describe())

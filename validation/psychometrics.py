import pandas as pd
from psychometrics import CronbachAlpha
import os

def validate_items():
    constructo = os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")
    df = pd.read_csv(f"data/{constructo}_synthetic.csv")
    
    # Calcular Alpha de Cronbach
    alpha = CronbachAlpha(df.values).calculate()
    print(f"Alpha de Cronbach: {alpha:.3f}")
    
    # Guardar reporte
    with open(f"validation/{constructo}_report.txt", "w") as f:
        f.write(f"Constructo: {constructo}\n")
        f.write(f"Alpha de Cronbach: {alpha:.3f}\n")
        f.write(f"Interpretación: {'Aceptable' if alpha > 0.7 else 'Mejorable'}")
    
    return alpha

if __name__ == "__main__":
    validate_items()

"""
Módulo de Validación Psicométrica - Versión Corregida

Realiza análisis de consistencia interna y validez de contenido para instrumentos psicométricos generados.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from typing import Dict
from pathlib import Path

class PsychometricValidator:
    """Realiza validación psicométrica completa de instrumentos generados"""
    
    def __init__(self, construct: str = None):
        self.construct = construct or os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")
        self.data_dir = Path("data")
        self.report_dir = Path("validation")
        self.report_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        filepath = self.data_dir / f"{self.construct}_synthetic.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos: {filepath}")
        
        df = pd.read_csv(filepath)
        if df.empty or len(df.columns) < 3:
            raise ValueError("Datos insuficientes para análisis psicométrico")
            
        return df

    def cronbach_alpha(self, df: pd.DataFrame) -> Dict:
        items = df.values
        n_items = items.shape[1]
        
        var_total = np.var(items.sum(axis=1))
        var_items = np.var(items, axis=0).sum()
        alpha = (n_items / (n_items - 1)) * (1 - var_items / var_total)
        
        alphas = []
        for _ in range(1000):
            sample = df.sample(frac=1, replace=True)
            items_sample = sample.values
            var_total_sample = np.var(items_sample.sum(axis=1))
            var_items_sample = np.var(items_sample, axis=0).sum()
            alphas.append((n_items / (n_items - 1)) * (1 - var_items_sample / var_total_sample))
        
        ci_lower, ci_upper = np.percentile(alphas, [2.5, 97.5])
        
        return {
            'alpha': alpha,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': self._interpret_alpha(alpha)
        }
    
    def item_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        total_scores = df.sum(axis=1)
        results = []
        
        for item in df.columns:
            item_score = df[item]
            corr, _ = stats.pearsonr(item_score, total_scores - item_score)
            results.append({
                'item': item,
                'correlacion_item_total': corr,
                'deberia_invertirse': corr < 0
            })
            
        return pd.DataFrame(results)
    
    def _interpret_alpha(self, alpha: float) -> str:
        if alpha >= 0.9:
            return "Excelente consistencia interna"
        elif alpha >= 0.8:
            return "Buena consistencia interna"
        elif alpha >= 0.7:
            return "Aceptable para investigación"
        elif alpha >= 0.6:
            return "Marginal, requiere revisión"
        else:
            return "Inaceptable, necesita rediseño"
    
    def generate_report(self, results: Dict):
        alpha_value = results['alpha']['alpha']
        ci_lower = results['alpha']['ci_lower']
        ci_upper = results['alpha']['ci_upper']
        interpretation = results['alpha']['interpretation']
        
        report_path = self.report_dir / f"{self.construct}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"=== REPORTE PSICOMÉTRICO ===\n")
            f.write(f"Constructo: {self.construct}\n")
            f.write(f"Alpha de Cronbach: {alpha_value:.3f} ")
            f.write(f"(IC 95%: {ci_lower:.3f}-{ci_upper:.3f})\n")
            f.write(f"Interpretación: {interpretation}\n\n")
            f.write("Análisis Ítem-Total:\n")
            f.write(results['item_analysis'].to_string(index=False))
        
        json_path = self.report_dir / f"{self.construct}_report.json"
        with open(json_path, 'w') as f:
            json.dump({
                'construct': self.construct,
                'metrics': {
                    'cronbach_alpha': alpha_value,
                    'confidence_interval': [ci_lower, ci_upper],
                    'interpretation': interpretation
                },
                'item_analysis': results['item_analysis'].to_dict('records')
            }, f, indent=2)
        
        print(f"Reportes generados en: {self.report_dir}")

def validate_items(construct: str = None) -> Dict:
    validator = PsychometricValidator(construct)
    try:
        df = validator.load_data()
        results = {
            'alpha': validator.cronbach_alpha(df),
            'item_analysis': validator.item_analysis(df)
        }
        validator.generate_report(results)
        return results
    except Exception as e:
        print(f"Error en validación: {str(e)}")
        raise

if __name__ == "__main__":
    validate_items()
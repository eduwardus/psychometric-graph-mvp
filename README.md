# Psychometric Graph MVP

Generación de ítems psicométricos mediante grafos experimentales y LLMs

## Primeros pasos

1. Clona el repositorio:
```bash
git clone https://github.com/eduwardus/psychometric-graph-mvp.git
cd psychometric-graph-mvp
```

2. Instala dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el flujo:
```bash
# Generar datos sintéticos
python data/synthetic_data.py

# Generar nuevo ítem
python model/train_tinyllama.py

# Validar métricas
python validation/psychometrics.py

# Generar grafo
python graph_builder/experiment_graph.py
```

## Estructura
```
├── data/                   # Datos sintéticos y reales
├── model/                  # Modelos generativos
├── graph_builder/          # Grafos experimentales
├── validation/             # Validación psicométrica
├── requirements.txt        # Dependencias
└── .env                    # Configuración
```

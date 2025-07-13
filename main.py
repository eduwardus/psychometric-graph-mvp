from psychometric_graph.graph_generator import GraphGenerator
from psychometric_graph.item_generator import ItemGenerator
from psychometric_graph.utils import visualize_graph
import json

# 1. Cargar datos sintéticos
with open("synthetic_data/sample.json", "r") as f:
    data = json.load(f)

# 2. Generar grafo
graph_generator = GraphGenerator()
graph = graph_generator.create_graph_from_data(data)
print(f"Grafo generado con {len(graph.nodes())} nodos y {len(graph.edges())} aristas")

# 3. Visualizar grafo (opcional)
visualize_graph(graph, "psychometric_graph.png")

# 4. Generar ítem psicométrico
concepts = list(graph.nodes())[:3]  # Usar primeros 3 conceptos
prompt = f"Genera un ítem sobre razonamiento verbal usando: {', '.join(concepts)}"

item_generator = ItemGenerator()
generated_item = item_generator.generate_item(prompt)
print("\nÍtem generado:")
print(generated_item)

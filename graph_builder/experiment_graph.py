import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

def create_weighted_graph(items=None, construct=None, save_path="graphs"):
    """
    Crea y guarda un grafo psicométrico con escala de colores bivariante para los pesos.
    
    Args:
        items (list): Lista de ítems psicométricos
        construct (str): Nombre del constructo principal
        save_path (str): Directorio para guardar los gráficos
    """
    # Configuración inicial
    if construct is None:
        construct = os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")
    
    if items is None:
        items = [
            "Me siento vacío",
            "Pierdo interés en actividades",
            "Dificultad para concentrarme",
            "Cambios en el apetito",
            "Problemas de sueño",
            "Fatiga constante",
            "Sentimientos de culpa",
            "Irritabilidad",
            "Pensamientos de inutilidad",
            "Pensamientos suicidas"
        ]
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Crear grafo dirigido con pesos
    G = nx.DiGraph()
    
    # Añadir nodo central (constructo)
    G.add_node(construct, type="construct", size=5000)
    
    # Añadir ítems con pesos aleatorios (simulados)
    for item in items:
        G.add_node(item, type="item", size=3000)
        weight = round(random.uniform(0.3, 0.9), 2)  # Peso aleatorio entre 0.3 y 0.9
        G.add_edge(construct, item, weight=weight)
    
    # Configuración de estilo mejorada
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.labelcolor': 'black'
    })
    
    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    
    # Posicionamiento mejorado
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    # Dibujar nodos con bordes
    node_colors = ["#1f78b4" if G.nodes[n]["type"] == "construct" else "#33a02c" for n in G.nodes()]
    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Crear mapa de colores personalizado (azul a rojo)
    colors = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]  # Azul -> Verde -> Amarillo -> Naranja -> Rojo
    custom_cmap = LinearSegmentedColormap.from_list("custom_rdbu", colors)
    
    # Dibujar edges con flechas y pesos
    edge_widths = [G[u][v]['weight']*4 for u, v in G.edges()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=custom_cmap,
        edge_vmin=0.3,
        edge_vmax=0.9,
        width=edge_widths,
        alpha=0.8,
        arrowstyle='-|>',
        arrowsize=25,
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Dibujar labels con fondo blanco para mejor legibilidad
    for node, (x, y) in pos.items():
        ax.text(x, y, node, 
                fontsize=9, 
                fontweight="bold", 
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Añadir pesos en las conexiones
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='#6a3d9a',
        font_size=8,
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        ax=ax
    )
    
    # Añadir barra de color para los pesos
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0.3, vmax=0.9))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Peso de la relación', fontsize=10)
    
    # Configuración del gráfico
    ax.set_title(f"Modelo Psicométrico: {construct}\n(Relaciones con pesos)", fontsize=14, pad=20)
    plt.tight_layout()
    
    # Guardar y mostrar
    filename = os.path.join(save_path, f"weighted_graph_{construct.lower().replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Grafo con pesos guardado en {filename}")

if __name__ == "__main__":
    create_weighted_graph()
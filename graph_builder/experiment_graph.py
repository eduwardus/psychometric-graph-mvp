import networkx as nx
import matplotlib.pyplot as plt
import os

def create_graph():
    constructo = os.getenv("CONSTRUCTO_PRINCIPAL", "depresión")
    G = nx.DiGraph()
    G.add_node(constructo)
    G.add_node("Ítem 1")
    G.add_edge(constructo, "Ítem 1")
    
    nx.draw(G, with_labels=True)
    plt.savefig(f"graph_{constructo}.png")
    print(f"Grafo guardado en graph_{constructo}.png")

if __name__ == "__main__":
    create_graph()

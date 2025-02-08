import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from operators import UNARY_OPERATORS, BINARY_OPERATORS
from scipy.stats import norm
import pdb

def build_graph(graph, node, pos=None, parent=None, edge_label=None, x=0, y=0, layer=1, labels=None):
    """
    Costruisce il grafo ricorsivamente per la visualizzazione.
    """
    if pos is None:
        pos = {}
    if labels is None:
        labels = {}
    
    node_label = str(node.value)
    if isinstance(node.value, str) and node.value not in UNARY_OPERATORS and node.value not in BINARY_OPERATORS:
        node_label = f"{np.round(node.coefficient,4)} * {node.value}"  # Mostra il coefficiente per le variabili
    
    pos[node] = (x, y)
    labels[node] = node_label  # Etichetta con il valore e coefficiente del nodo
    
    if parent is not None:
        graph.add_edge(parent, node, label=edge_label)
    
    if node.left_child:
        build_graph(graph, node.left_child, pos, node, "L", x - 1 / 2 ** layer, y - 1, layer + 1, labels)
    if node.right_child:
        build_graph(graph, node.right_child, pos, node, "R", x + 1 / 2 ** layer, y - 1, layer + 1, labels)
    
    return graph, pos, labels

def plot_tree(root):
    """
    Visualizza l'albero utilizzando NetworkX e Matplotlib, mostrando il contenuto di ogni nodo e il coefficiente delle variabili.
    """
    graph = nx.DiGraph()
    graph, pos, labels = build_graph(graph, root)
    
    plt.figure(figsize=(10, 6))
    
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=2000, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()
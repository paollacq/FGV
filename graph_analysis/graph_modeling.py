# graph_analysis/graph_modeling.py

import pandas as pd
import networkx as nx

def build_graph_from_anomalous_transactions(anomaly_transactions_file):
    """
    Constrói um grafo a partir das transações anômalas contidas em anomaly_scores.csv.

    Parameters:
    - anomaly_transactions_file: Caminho para o arquivo CSV contendo as transações anômalas.

    Returns:
    - G: Um grafo direcionado representando as transações anômalas.
    """
    # Carregar as transações anômalas
    df_anomalous = pd.read_csv(anomaly_transactions_file)

    # Verificar se as colunas necessárias estão presentes
    required_columns = ['sender', 'receiver', 'amount']
    if not all(column in df_anomalous.columns for column in required_columns):
        raise ValueError(f"O arquivo {anomaly_transactions_file} deve conter as colunas: {required_columns}")

    # Construir o grafo
    G = nx.DiGraph()
    edges = list(zip(df_anomalous['sender'], df_anomalous['receiver']))
    weights = df_anomalous['amount'].tolist()
    G.add_weighted_edges_from([(u, v, w) for (u, v), w in zip(edges, weights)])

    print(f"Grafo construído com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas a partir de transações anômalas.")
    return G

def extract_graph_features(G):
    """
    Extrai características locais e globais do grafo.

    Parameters:
    - G: Grafo de transações.

    Returns:
    - features_df: DataFrame com as características extraídas.
    """
    total_sent = dict(G.out_degree(weight='weight'))
    total_received = dict(G.in_degree(weight='weight'))
    num_transactions_sent = dict(G.out_degree())
    num_transactions_received = dict(G.in_degree())
    pagerank = nx.pagerank(G, alpha=0.85)

    undirected_G = G.to_undirected()
    components = list(nx.connected_components(undirected_G))
    component_id = {node: idx for idx, comp in enumerate(components) for node in comp}

    nodes = list(G.nodes)
    features_df = pd.DataFrame({
        'address': nodes,
        'total_sent': [total_sent.get(node, 0) for node in nodes],
        'total_received': [total_received.get(node, 0) for node in nodes],
        'num_transactions_sent': [num_transactions_sent.get(node, 0) for node in nodes],
        'num_transactions_received': [num_transactions_received.get(node, 0) for node in nodes],
        'pagerank': [pagerank.get(node, 0) for node in nodes],
        'component_id': [component_id.get(node, -1) for node in nodes],
    })

    return features_df

def visualize_graph(G, output_path, num_nodes=100):
    """
    Visualiza e salva uma imagem do grafo.

    Parameters:
    - G: Grafo NetworkX.
    - output_path: Caminho para salvar a imagem do grafo.
    - num_nodes: Número de nós a serem incluídos no subgrafo para visualização.
    """
    import matplotlib.pyplot as plt

    # Selecionar um subgrafo com os nós de maior grau
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:num_nodes]
    subgraph = G.subgraph(top_nodes)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, k=0.15, seed=42)
    nx.draw(subgraph, pos, node_size=50, node_color='blue', edge_color='gray', with_labels=False)
    plt.title(f'Subgrafo das Transações Anômalas (Top {num_nodes} Nós)')
    plt.savefig(output_path, format='PNG')
    plt.close()
    print(f"Imagem do grafo salva em {output_path}")

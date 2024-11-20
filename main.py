# main.py

import os
import sys
import pandas as pd
import networkx as nx

# Adicionar o diretório raiz ao sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importações existentes
from src.preprocess import preprocess_pipeline_chunked
from src.models import train_isolation_forest
from src.visualization import plot_feature_distributions
from src.models import explain_model_with_shap

# Importações para análise de grafos
from graph_analysis.graph_modeling import build_graph_from_anomalous_transactions, extract_graph_features, visualize_graph

# Caminhos
RAW_FILE = "data/raw/btc_tx_2011_2013.csv"  # Arquivo bruto
PROCESSED_FILE = "data/processed/btc_transactions_processed.csv"  # Arquivo processado
ANOMALY_SCORES_FILE = "results/anomaly_scores.csv"  # Arquivo com scores de anomalia
ANOMALOUS_TRANSACTIONS_FILE = "results/anomalous_transactions.csv"  # Arquivo com transações anômalas
VISUALIZATIONS_DIR = "results/visualizations"  # Diretório para visualizações

# Criar pastas necessárias
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

# 1. Pré-processamento em chunks
print("Executando pré-processamento em chunks...")
preprocess_pipeline_chunked(
    input_file=RAW_FILE,
    output_file=PROCESSED_FILE,
    chunk_size=100000
)

# 2. Carregar dados processados
print("Carregando dados processados...")
transactions_df = pd.read_csv(PROCESSED_FILE)

# Verificar as estatísticas básicas
numeric_columns = ['total_sent', 'total_received', 'net_flow']
print("Estatísticas das features numéricas:")
print(transactions_df[numeric_columns].describe())

# 3. Detecção de Anomalias com Isolation Forest
print("\nExecutando detecção de anomalias com Isolation Forest...")
features = ['total_sent', 'total_received', 'net_flow']
model, processed_data = train_isolation_forest(transactions_df, features, contamination=0.01)

# Filtrar apenas as transações anômalas
anomalous_data = processed_data[processed_data['is_anomaly'] == 1]

# Salvar os scores de anomalia apenas das transações anômalas
anomalous_data.to_csv(ANOMALY_SCORES_FILE, index=False)
print(f"Scores de anomalia salvos em {ANOMALY_SCORES_FILE}")

# 4. Filtrar transações anômalas
print("\nFiltrando transações anômalas...")
anomalous_addresses = set(anomalous_data['address'])

def filter_transactions_for_anomalous_addresses(input_file, output_file, anomalous_addresses, chunksize=100000):
    """
    Filtra as transações que envolvem endereços anômalos e salva em um novo arquivo CSV.
    """
    column_names = ['sender', 'receiver', 'timestamp', 'amount']
    with open(output_file, 'w') as f_out:
        for chunk_number, chunk in enumerate(pd.read_csv(input_file, header=None, names=column_names, chunksize=chunksize)):
            # Filtrar transações que envolvem endereços anômalos
            filtered_chunk = chunk[
                (chunk['sender'].isin(anomalous_addresses)) |
                (chunk['receiver'].isin(anomalous_addresses))
            ]
            if not filtered_chunk.empty:
                filtered_chunk.to_csv(f_out, header=(chunk_number == 0), index=False)
    print(f"Transações anômalas salvas em {output_file}")

# Executar a filtragem das transações anômalas
filter_transactions_for_anomalous_addresses(
    input_file=RAW_FILE,
    output_file=ANOMALOUS_TRANSACTIONS_FILE,
    anomalous_addresses=anomalous_addresses,
    chunksize=100000
)

# 5. Construir o grafo a partir das transações anômalas
print("\nConstruindo o grafo a partir das transações anômalas...")
G_anomalous = build_graph_from_anomalous_transactions(ANOMALOUS_TRANSACTIONS_FILE)

# 6. Extrair características do grafo anômalo
print("Extraindo características do grafo anômalo...")
anomalous_features_df = extract_graph_features(G_anomalous)

# Salvar características do grafo anômalo
anomalous_features_df.to_csv("results/anomalous_graph_features.csv", index=False)
print("Características do grafo anômalo salvas em results/anomalous_graph_features.csv")

# 7. Visualização do Grafo Anômalo
print("\nVisualizando o grafo anômalo...")
graph_image_path = os.path.join(VISUALIZATIONS_DIR, 'anomalous_graph.png')
visualize_graph(G_anomalous, graph_image_path, num_nodes=100)

# 8. Visualização das Distribuições das Features
print("\nGerando visualizações das distribuições das features...")
plot_feature_distributions(anomalous_data, features, label_column='is_anomaly', save_path=VISUALIZATIONS_DIR)

# 9. Explicação do modelo com SHAP
print("Explicando o modelo com SHAP...")
explain_model_with_shap(model, anomalous_data, features)

print("\nPipeline concluído!")

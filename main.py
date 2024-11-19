import os
import pandas as pd
from src.preprocess import preprocess_pipeline_chunked
from src.models import train_isolation_forest, detect_anomalies
from src.visualization import plot_anomaly_score_distribution

# Caminhos
RAW_FILE = "data/raw/btc_tx_2011_2013.csv"  # Arquivo bruto
PROCESSED_FILE = "data/processed/btc_transactions_processed.csv"  # Arquivo processado
ANOMALY_SCORES_FILE = "results/anomaly_scores.csv"  # Scores de anomalias
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

# 2. Treinamento do modelo de detecção de anomalias
print("Carregando dados processados...")
processed_data = pd.read_csv(PROCESSED_FILE)

# Definir as colunas numéricas para treinamento
numeric_columns = ['total_sent', 'total_received', 'net_flow']

print("Treinando modelo de detecção de anomalias...")
model = train_isolation_forest(processed_data, numeric_columns=numeric_columns)

# 3. Detecção de anomalias
print("Detectando anomalias...")
scores, predictions = detect_anomalies(model, processed_data[numeric_columns])

# Adicionar os resultados ao DataFrame
results = processed_data.copy()
results["anomaly_score"] = scores
results["is_anomaly"] = predictions

# Salvar os resultados
results.to_csv(ANOMALY_SCORES_FILE, index=False)
print(f"Resultados salvos em {ANOMALY_SCORES_FILE}")

# 4. Gerar visualização dos scores de anomalias
print("Gerando visualização dos scores de anomalia...")
plot_anomaly_score_distribution(results, save_path=f"{VISUALIZATIONS_DIR}/anomaly_score_distribution.png")
print(f"Visualização salva em {VISUALIZATIONS_DIR}/btc_anomaly_score_distribution.png")

print("Pipeline concluído!")

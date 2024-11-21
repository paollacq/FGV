# main.py

import os
import sys
import pandas as pd
import numpy as np

# Adicionar o diretório raiz ao sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importações dos módulos personalizados
from src.preprocess import preprocess_pipeline_chunked
from src.models import train_isolation_forest, train_random_forest, evaluate_model
from src.visualization import (
    plot_anomalous_transactions_per_month,
    plot_anomalous_volume_per_month,
    plot_transactions_per_address,
    plot_heatmap_top_addresses,
    plot_boxplot_transaction_values,
    plot_total_sent_anomaly_scores,
    plot_kde_anomaly_scores
)

# Caminhos dos arquivos
RAW_FILE = "data/raw/btc_tx_2011_2013.csv"  # Arquivo bruto
PROCESSED_FILE = "data/processed/btc_transactions_processed.csv"  # Arquivo processado
VISUALIZATIONS_DIR = "results/visualizations"  # Diretório para visualizações

# Criar pastas necessárias
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

# 1. Carregar os hashes de transações anômalas
def load_anomaly_hashes():
    # Carregar todos os arquivos de anomalias e retornar os hashes únicos de transações
    anomalies_files = [
        'data/database/anomalies_theft_tx.csv',
        'data/database/anomalies_seizure1_tx.csv',
        'data/database/anomalies_seizure2_tx.csv',
        'data/database/anomalies_misc_tx.csv',
        'data/database/anomalies_loss_tx.csv'
    ]
    anomaly_dataframes = []
    for f in anomalies_files:
        df = pd.read_csv(f, header=None, names=['tx_hash'])
        anomaly_dataframes.append(df)
    anomaly_hashes = pd.concat(anomaly_dataframes)['tx_hash'].unique()
    return set(anomaly_hashes)

print("Carregando hashes de transações anômalas...")
anomaly_hashes = load_anomaly_hashes()
print(f"Total de transações anômalas conhecidas: {len(anomaly_hashes)}")

# 2. Pré-processamento em chunks com labels de anomalias
print("Executando pré-processamento em chunks com labels de anomalias...")
preprocess_pipeline_chunked(
    input_file=RAW_FILE,
    output_file=PROCESSED_FILE,
    anomaly_hashes=anomaly_hashes,
    chunk_size=100000
)

# 3. Carregar dados processados
print("Carregando dados processados...")
transactions_df = pd.read_csv(PROCESSED_FILE)

# Verificar se transactions_df está vazio
if transactions_df.empty:
    print("O DataFrame transactions_df está vazio. Verifique o pré-processamento.")
    sys.exit(1)

# 4. Calcular métricas locais (features)
print("Calculando métricas locais...")

def calculate_local_metrics(transactions_df):
    # Verificar se as colunas necessárias estão presentes
    required_columns = {'address', 'amount', 'is_anomalous'}
    if not required_columns.issubset(transactions_df.columns):
        print(f"As colunas necessárias não estão presentes em transactions_df: {required_columns}")
        return pd.DataFrame()

    # Agrupar por endereço e calcular métricas
    metrics_df = transactions_df.groupby('address').agg(
        total_amount=('amount', 'sum'),
        num_transactions=('amount', 'count'),
        is_anomalous=('is_anomalous', 'max')
    ).reset_index()

    return metrics_df

metrics_df = calculate_local_metrics(transactions_df)

# Verificar se metrics_df está vazio
if metrics_df.empty:
    print("O DataFrame metrics_df está vazio. Verifique o cálculo de métricas locais.")
    sys.exit(1)

# 5. Treinar modelo de Isolation Forest e avaliar
print("\nTreinando modelo de Isolation Forest...")
features = ['total_amount', 'num_transactions']

# Preencher valores nulos com zeros
metrics_df[features] = metrics_df[features].fillna(0)

model_if, processed_data_if = train_isolation_forest(metrics_df, features, contamination=0.01)

# Avaliar o modelo não supervisionado
print("\nAvaliação do Isolation Forest:")
evaluate_model(processed_data_if['is_anomalous'], processed_data_if['is_anomaly'], model_name='Isolation Forest')

# 6. Treinar modelo supervisionado (Random Forest)
print("\nTreinando modelo supervisionado (Random Forest)...")

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = metrics_df[features]
y = metrics_df['is_anomalous']

# Lidar com desbalanceamento usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Treinar o modelo Random Forest
model_rf = train_random_forest(X_train, y_train)

# Prever e avaliar
y_pred_rf = model_rf.predict(X_test)

print("\nAvaliação do Random Forest:")
evaluate_model(y_test, y_pred_rf, model_name='Random Forest')

# 7. Gerar gráficos solicitados
print("\nGerando gráficos solicitados...")

# Usaremos uma amostra das transações para os gráficos
sample_size = 1000000
if len(transactions_df) > sample_size:
    transactions_sample = transactions_df.sample(n=sample_size, random_state=42)
else:
    transactions_sample = transactions_df

# Gerar gráficos (ajuste as funções de visualização conforme necessário)
# ...

print("\nPipeline concluído!")

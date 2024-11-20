import os
import pandas as pd
from src.preprocess import preprocess_pipeline_chunked
from src.models import train_isolation_forest, detect_anomalies
from src.visualization import plot_feature_distributions
from src.models import evaluate_model
from src.models import explain_model_with_shap


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

# 2. Carregar dados processados
print("Carregando dados processados...")
processed_data = pd.read_csv(PROCESSED_FILE)

# Verificar as estatísticas básicas
numeric_columns = ['total_sent', 'total_received', 'net_flow']
print("Estatísticas das features numéricas:")
print(processed_data[numeric_columns].describe())

# Configurações
contamination_values = [0.001, 0.01, 0.05]
features = ['total_sent', 'total_received', 'net_flow']

for contamination in contamination_values:
    print(f"\nTreinando Isolation Forest com contamination={contamination}...")
    model, processed_data = train_isolation_forest(processed_data, features, contamination=contamination)
    
    # Salvar resultados intermediários para cada configuração
    results_file = f"results/anomaly_scores_contamination_{contamination}.csv"
    processed_data.to_csv(results_file, index=False)
    print(f"Resultados salvos em {results_file}")
    
    # Analisar transações anômalas
    anomalies = processed_data[processed_data['is_anomaly'] == 1]
    print(f"Transações anômalas detectadas (contamination={contamination}): {len(anomalies)}")
    print(anomalies[['total_sent', 'total_received', 'net_flow']].describe())


# 5. Visualização
# Plotar distribuições das features
print("Gerando visualizações das distribuições das features...")
plot_feature_distributions(processed_data, features, label_column='is_anomaly', save_path="results/visualizations")

# Avaliar o modelo
#print("Avaliando o modelo com métricas...")
#evaluate_model(processed_data, true_labels_column='true_labels', anomaly_score_column='anomaly_score', prediction_column='is_anomaly')
# Explicar o modelo com SHAP
print("Explicando o modelo com SHAP...")
explain_model_with_shap(model, processed_data, features)

print("Pipeline concluído!")

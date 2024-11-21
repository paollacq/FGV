# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_anomalous_transactions_per_month(anomalous_transactions, save_path):
    """
    Plota o número de transações anômalas por mês.
    """
    # Converter a coluna 'timestamp' para datetime
    anomalous_transactions['timestamp'] = pd.to_datetime(anomalous_transactions['timestamp'], errors='coerce')
    
    # Remover timestamps nulos
    anomalous_transactions = anomalous_transactions.dropna(subset=['timestamp'])
    
    # Extrair o ano e o mês
    anomalous_transactions['year_month'] = anomalous_transactions['timestamp'].dt.to_period('M')
    
    # Contar o número de transações por mês
    transactions_per_month = anomalous_transactions.groupby('year_month').size().reset_index(name='count')
    
    # Plotar
    plt.figure(figsize=(12,6))
    sns.barplot(x='year_month', y='count', data=transactions_per_month)
    plt.xticks(rotation=45)
    plt.title('Número de Transações Anômalas por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Número de Transações')
    plt.tight_layout()
    plt.savefig(f"{save_path}/anomalous_transactions_per_month.png")
    plt.close()

def plot_anomalous_volume_per_month(anomalous_transactions, save_path):
    """
    Plota o volume total de Bitcoin transferido por mês para transações anômalas.
    """
    # Converter a coluna 'timestamp' para datetime
    anomalous_transactions['timestamp'] = pd.to_datetime(anomalous_transactions['timestamp'], errors='coerce')
    
    # Remover timestamps nulos
    anomalous_transactions = anomalous_transactions.dropna(subset=['timestamp'])
    
    # Extrair o ano e o mês
    anomalous_transactions['year_month'] = anomalous_transactions['timestamp'].dt.to_period('M')
    
    # Calcular o volume total enviado e recebido por mês
    volume_per_month = anomalous_transactions.groupby('year_month')['amount'].sum().reset_index()
    
    # Plotar
    plt.figure(figsize=(12,6))
    sns.barplot(x='year_month', y='amount', data=volume_per_month)
    plt.xticks(rotation=45)
    plt.title('Volume de Bitcoin Transferido por Mês para Transações Anômalas')
    plt.xlabel('Mês')
    plt.ylabel('Volume de Bitcoin')
    plt.tight_layout()
    plt.savefig(f"{save_path}/anomalous_volume_per_month.png")
    plt.close()

def plot_transactions_per_address(transactions, save_path):
    """
    Plota um histograma da quantidade de transações por endereço.
    """
    # Contar o número de transações enviadas e recebidas por endereço
    sent_counts = transactions['sender'].value_counts()
    received_counts = transactions['receiver'].value_counts()
    
    # Combinar os contadores
    total_counts = sent_counts.add(received_counts, fill_value=0)
    
    # Plotar
    plt.figure(figsize=(12,6))
    sns.histplot(total_counts, bins=50, log_scale=(True, True))
    plt.title('Histograma da Quantidade de Transações por Endereço')
    plt.xlabel('Número de Transações')
    plt.ylabel('Contagem de Endereços')
    plt.tight_layout()
    plt.savefig(f"{save_path}/transactions_per_address_histogram.png")
    plt.close()

def plot_heatmap_top_addresses(transactions, save_path, top_n=50):
    """
    Plota um heatmap das transações entre os top N endereços por volume.
    """
    # Calcular o volume total por endereço
    sent_volumes = transactions.groupby('sender')['amount'].sum()
    received_volumes = transactions.groupby('receiver')['amount'].sum()
    total_volumes = sent_volumes.add(received_volumes, fill_value=0)
    
    # Selecionar os top N endereços
    top_addresses = total_volumes.nlargest(top_n).index.tolist()
    
    # Filtrar transações entre os top N endereços
    filtered_transactions = transactions[
        (transactions['sender'].isin(top_addresses)) &
        (transactions['receiver'].isin(top_addresses))
    ]
    
    # Criar matriz de transações
    pivot_table = filtered_transactions.pivot_table(
        index='sender',
        columns='receiver',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )
    
    # Plotar heatmap
    plt.figure(figsize=(14,10))
    sns.heatmap(pivot_table, cmap='viridis')
    plt.title(f'Heatmap das Transações entre os Top {top_n} Endereços')
    plt.xlabel('Destinatário')
    plt.ylabel('Remetente')
    plt.tight_layout()
    plt.savefig(f"{save_path}/heatmap_top_{top_n}_addresses.png")
    plt.close()

def plot_boxplot_transaction_values(transactions, anomalous_transactions, save_path):
    """
    Plota um boxplot comparando os valores transferidos em transações normais e anômalas.
    """
    # Marcar transações anômalas
    transactions['is_anomalous'] = 'Normal'
    anomalous_indices = anomalous_transactions.index
    transactions.loc[transactions.index.isin(anomalous_indices), 'is_anomalous'] = 'Anômala'
    
    # Amostrar dados para evitar sobrecarga de memória (opcional)
    sample_size = min(100000, len(transactions))
    transactions_sample = transactions.sample(n=sample_size, random_state=42)
    
    # Plotar boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x='is_anomalous', y='amount', data=transactions_sample)
    plt.yscale('log')
    plt.title('Distribuição dos Valores Transferidos por Tipo de Transação')
    plt.xlabel('Tipo de Transação')
    plt.ylabel('Valor Transferido (escala log)')
    plt.tight_layout()
    plt.savefig(f"{save_path}/boxplot_transaction_values.png")
    plt.close()

def plot_total_sent_anomaly_scores(processed_data, save_path):
    """
    Plota um boxplot de total_sent destacando transações com scores de anomalia elevados.
    """
    # Verificar se 'anomaly_score' está presente
    if 'anomaly_score' not in processed_data.columns:
        return
    
    # Definir um limiar para scores de anomalia elevados
    high_anomaly_threshold = processed_data['anomaly_score'].quantile(0.95)
    
    # Categorizar as transações
    processed_data['Anomaly Level'] = 'Baixo'
    processed_data.loc[processed_data['anomaly_score'] >= high_anomaly_threshold, 'Anomaly Level'] = 'Alto'
    
    # Plotar boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Anomaly Level', y='total_sent', data=processed_data)
    plt.yscale('log')
    plt.title('Total Sent por Nível de Anomalia')
    plt.xlabel('Nível de Anomalia')
    plt.ylabel('Total Sent (escala log)')
    plt.tight_layout()
    plt.savefig(f"{save_path}/boxplot_total_sent_anomaly_scores.png")
    plt.close()

def plot_kde_anomaly_scores(processed_data, save_path):
    """
    Plota o KDE dos scores de anomalia.
    """
    if 'anomaly_score' not in processed_data.columns:
        print("A coluna 'anomaly_score' não está presente em processed_data.")
        return

    plt.figure(figsize=(8,6))
    sns.kdeplot(processed_data['anomaly_score'], shade=True)
    plt.title('KDE dos Scores de Anomalia')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Densidade')
    plt.tight_layout()
    plt.savefig(f"{save_path}/kde_anomaly_scores.png")
    plt.close()

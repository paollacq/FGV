# src/preprocess.py

import pandas as pd
import numpy as np

def preprocess_pipeline_chunked(input_file, output_file, anomaly_hashes, chunk_size=100000):
    """
    Processa o arquivo de transações em chunks, adicionando labels de anomalias.

    Parameters:
    - input_file: Caminho para o arquivo CSV de entrada.
    - output_file: Caminho para o arquivo CSV de saída.
    - anomaly_hashes: Conjunto de hashes de transações anômalas.
    - chunk_size: Tamanho dos chunks para processamento.
    """
    column_names = ['tx_hash', 'address', 'timestamp', 'amount']

    for chunk_number, chunk in enumerate(pd.read_csv(input_file, header=None, names=column_names, chunksize=chunk_size)):
        # Converter timestamp para datetime
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')

        # Adicionar label de anomalia
        chunk['is_anomalous'] = chunk['tx_hash'].isin(anomaly_hashes).astype(int)

        # Remover transações com valores nulos ou negativos
        chunk = chunk.dropna(subset=['address', 'amount'])
        chunk = chunk[chunk['amount'] > 0]

        # Verificar se o chunk não está vazio
        if chunk.empty:
            continue

        # Salvar o chunk processado
        if chunk_number == 0:
            chunk.to_csv(output_file, index=False, mode='w')
        else:
            chunk.to_csv(output_file, index=False, mode='a', header=False)

    print("Pré-processamento concluído.")

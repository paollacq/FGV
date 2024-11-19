import pandas as pd

def process_chunk(chunk):
    """
    Processa um chunk do dataset bruto, criando features úteis.
    """
    # Criar features básicas
    chunk['value'] = chunk['value'].astype(float)
    
    # Total de valores enviados e recebidos por endereço
    chunk_grouped_from = chunk.groupby('from_address')['value'].sum().reset_index()
    chunk_grouped_from.columns = ['address', 'total_sent']
    
    chunk_grouped_to = chunk.groupby('to_address')['value'].sum().reset_index()
    chunk_grouped_to.columns = ['address', 'total_received']
    
    # Combinar os dados
    combined = pd.merge(chunk_grouped_from, chunk_grouped_to, on='address', how='outer').fillna(0)
    
    # Criar feature de saldo (opcional)
    combined['net_flow'] = combined['total_received'] - combined['total_sent']
    
    return combined

def preprocess_pipeline_chunked(input_file, output_file, chunk_size=1000000):
    """
    Processa o arquivo bruto em chunks e gera o dataset processado.
    """
    # Inicializar arquivo de saída
    header_written = False
    
    # Processar em chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, header=None, names=['from_address', 'to_address', 'timestamp', 'value'])):
        processed_chunk = process_chunk(chunk)
        
        # Salvar incrementalmente
        mode = 'w' if not header_written else 'a'
        processed_chunk.to_csv(output_file, mode=mode, header=not header_written, index=False)
        header_written = True

    print(f"Pipeline concluído. Dados salvos em {output_file}")

if __name__ == "__main__":
    # Arquivo de entrada e saída
    input_file = "data/raw/btc_tx_2011_2013.csv"
    output_file = "data/processed/btc_transactions_features.csv"
    
    # Processar o dataset em chunks
    preprocess_pipeline_chunked(input_file, output_file, chunk_size=100000)

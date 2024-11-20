import pandas as pd

def process_chunk(chunk):
    """
    Processa um chunk do dataset bruto, criando features úteis.
    """
    # Converter a coluna 'value' para float
    chunk['value'] = chunk['value'].astype(float)
    
    # Garantir que 'from_address' e 'to_address' sejam strings
    chunk['from_address'] = chunk['from_address'].astype(str)
    chunk['to_address'] = chunk['to_address'].astype(str)
    
    # Calcular o total enviado (group by 'from_address') e recebido (group by 'to_address')
    chunk_grouped_from = chunk.groupby('from_address')['value'].sum().reset_index()
    chunk_grouped_from.columns = ['address', 'total_sent']
    
    chunk_grouped_to = chunk.groupby('to_address')['value'].sum().reset_index()
    chunk_grouped_to.columns = ['address', 'total_received']
    
    # Garantir que a coluna 'address' seja do mesmo tipo antes de mesclar
    chunk_grouped_from['address'] = chunk_grouped_from['address'].astype(str)
    chunk_grouped_to['address'] = chunk_grouped_to['address'].astype(str)
    
    # Combinar os dados em um único DataFrame
    combined = pd.merge(chunk_grouped_from, chunk_grouped_to, on='address', how='outer').fillna(0)
    
    # Calcular o fluxo líquido (net flow)
    combined['net_flow'] = combined['total_received'] - combined['total_sent']
    
    return combined



def preprocess_pipeline_chunked(input_file, output_file, chunk_size=10):  # Teste com um chunk pequeno
    header_written = False
    
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, header=None, names=['from_address', 'to_address', 'timestamp', 'value'])):
        print(f"Processando chunk {i + 1}...")
        processed_chunk = process_chunk(chunk)
        
        # Salvar apenas o primeiro chunk para depuração
        if i == 0:
            processed_chunk.to_csv(output_file, index=False)
            break

    print(f"Pipeline concluído. Dados salvos em {output_file}")


if __name__ == "__main__":
    # Arquivo de entrada e saída
    input_file = "data/raw/btc_tx_2011_2013.csv"
    output_file = "data/processed/btc_transactions_processed.csv"
    
    # Processar o dataset em chunks
    preprocess_pipeline_chunked(input_file, output_file, chunk_size=100000)

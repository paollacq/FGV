import pandas as pd
from sklearn.cluster import KMeans

def balance_with_xgbclus(data, target_column, n_clusters=10):
    """
    Balanceia o dataset usando clustering para subamostrar a classe majoritária.
    
    :param data: DataFrame contendo o dataset.
    :param target_column: Nome da coluna com as classes (0 ou 1).
    :param n_clusters: Número de clusters para amostragem da classe majoritária.
    :return: DataFrame balanceado.
    """
    # Separar classes majoritária e minoritária
    minority_class = data[data[target_column] == 1]
    majority_class = data[data[target_column] == 0]
    
    # Ajustar n_clusters ao número de amostras disponíveis
    n_clusters = min(n_clusters, len(majority_class))
    if n_clusters < 2:
        raise ValueError("Número insuficiente de amostras na classe majoritária para clustering.")
    
    # Clustering na classe majoritária
    features = majority_class.drop(columns=[target_column])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    majority_class['cluster'] = kmeans.fit_predict(features)
    
    # Seleção de uma amostra de cada cluster
    balanced_majority = (
        majority_class.groupby('cluster')
        .apply(lambda x: x.sample(n=min(len(x), len(minority_class)), random_state=42))
        .reset_index(drop=True)
    )
    
    # Concatenar classes balanceadas
    balanced_data = pd.concat([minority_class, balanced_majority], axis=0).reset_index(drop=True)
    
    # Remover a coluna de clusters antes de retornar
    return balanced_data.drop(columns=['cluster'])

if __name__ == "__main__":
    # Exemplo de uso
    file_path = "data/processed/btc_transactions_processed.csv"
    target_column = "is_fraud"
    
    # Carregar os dados
    data = pd.read_csv(file_path)
    
    # Balancear os dados
    balanced_data = balance_with_xgbclus(data, target_column, n_clusters=5)
    
    # Salvar os dados balanceados
    balanced_data.to_csv("data/processed/btc_transactions_balanced.csv", index=False)

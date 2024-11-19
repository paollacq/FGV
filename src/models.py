import pandas as pd
from sklearn.ensemble import IsolationForest

def train_isolation_forest(data, numeric_columns=None):
    """
    Treina o Isolation Forest com base em colunas numéricas especificadas.
    
    :param data: DataFrame com os dados processados.
    :param numeric_columns: Lista de colunas numéricas para treinamento. Se None, usa todas as numéricas.
    :return: Modelo treinado.
    """
    # Filtrar apenas colunas numéricas relevantes
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    X = data[numeric_columns]
    
    print(f"Treinando modelo com as colunas: {numeric_columns}")
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    return model

def detect_anomalies(model, data):
    """
    Aplica o modelo treinado para detectar anomalias em novos dados.
    
    :param model: Modelo treinado.
    :param data: DataFrame com os dados a serem analisados.
    :return: Tuple (scores, predictions) com os resultados da detecção.
    """
    # Calcular os scores de anomalia
    scores = model.decision_function(data)
    
    # Previsões: -1 é anômalo, 1 é normal
    predictions = model.predict(data)
    predictions = (predictions == -1).astype(int)  # Converter -1 (anômalo) para 1, 1 (normal) para 0
    
    return scores, predictions

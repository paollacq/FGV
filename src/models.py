import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import shap

def explain_model_with_shap(model, data, features):
    """
    Explica o modelo Isolation Forest com SHAP.
    """
    explainer = shap.Explainer(model, data[features])
    shap_values = explainer(data[features])
    
    # Sumário da importância das features
    shap.summary_plot(shap_values, data[features])



def train_isolation_forest(data, features, contamination=0.01):
    """
    Treina o modelo Isolation Forest e retorna as predições e o modelo treinado.
    """
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination, random_state=42)
    model.fit(data[features])
    
    # Predições
    data['anomaly_score'] = model.decision_function(data[features])
    data['is_anomaly'] = model.predict(data[features]) == -1  # -1 indica anomalia
    
    return model, data


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


def evaluate_model(data, true_labels_column, anomaly_score_column, prediction_column):
    """
    Avalia o modelo com métricas de classificação e AUC-ROC.
    """
    print("Classification Report:")
    print(classification_report(data[true_labels_column], data[prediction_column]))
    
    roc_auc = roc_auc_score(data[true_labels_column], data[anomaly_score_column])
    print(f"ROC-AUC: {roc_auc}")


def explain_model_with_shap(model, data, features):
    """
    Explica o modelo Isolation Forest com SHAP.
    """
    explainer = shap.Explainer(model, data[features])
    shap_values = explainer(data[features])
    
    # Sumário da importância das features
    shap.summary_plot(shap_values, data[features])

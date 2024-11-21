# src/models.py

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_isolation_forest(data, features, contamination=0.01):
    X = data[features]

    # Verificar se há dados suficientes
    if X.empty or X.shape[0] == 0:
        print("Não há dados suficientes para treinar o Isolation Forest.")
        return None, data

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    data['anomaly_score'] = model.decision_function(X)
    data['is_anomaly'] = model.predict(X)
    # Converter labels para 0 (normal) e 1 (anômalo)
    data['is_anomaly'] = data['is_anomaly'].map({1: 0, -1: 1})
    return model, data

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=10000, random_state=420)
    model.fit(X_train, y_train)
    return model

def evaluate_model(true_labels, predicted_labels, model_name='Model'):
    print(f"{model_name} Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

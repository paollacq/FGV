# model_evaluation.py

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia o modelo usando várias métricas e plota a curva ROC.

    Parameters:
    - model: Modelo treinado.
    - X_test: Dados de teste.
    - y_test: Labels reais.
    - model_name: Nome do modelo para identificação.

    Returns:
    - metrics: Dicionário com as métricas calculadas.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    tpr = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Resultados para {model_name}:")
    print(f"Acurácia: {accuracy:.2f}")
    print(f"Taxa de Verdadeiro Positivo (TPR): {tpr:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    fpr, tpr_curve, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr_curve, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend()
    plt.show()

    metrics = {
        'accuracy': accuracy,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

    return metrics

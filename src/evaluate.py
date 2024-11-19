import pandas as pd
import matplotlib.pyplot as plt

def evaluate_scores(data, score_column, label_column='is_anomaly'):
    """
    Avalia a distribuição dos scores e a relação com os rótulos.
    """
    data.groupby(label_column)[score_column].plot(kind='kde', legend=True)
    plt.title("Distribuição dos Anomaly Scores")
    plt.xlabel(score_column)
    plt.show()

def evaluate_results(file_path):
    """
    Avalia os resultados de detecção de anomalias.
    """
    data = pd.read_csv(file_path)
    evaluate_scores(data, 'anomaly_score')

if __name__ == "__main__":
    # Avaliar os resultados salvos
    results_file = "results/anomaly_scores.csv"
    evaluate_results(results_file)

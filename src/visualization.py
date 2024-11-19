import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomaly_score_distribution(data, score_column='anomaly_score', label_column='is_anomaly', save_path=None):
    """
    Gera a distribuição dos scores de anomalias separando classes normais e anômalas.
    
    :param data: DataFrame contendo os scores e as classes.
    :param score_column: Nome da coluna com os scores de anomalia.
    :param label_column: Nome da coluna com os rótulos de anomalias.
    :param save_path: Caminho para salvar a visualização. Se None, exibe o gráfico.
    """
    # Configuração do estilo do gráfico
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Distribuição dos scores para transações normais
    sns.kdeplot(
        data=data[data[label_column] == 0][score_column],
        label='Transações Normais',
        fill=True,
        color='blue'
    )
    
    # Distribuição dos scores para transações anômalas
    sns.kdeplot(
        data=data[data[label_column] == 1][score_column],
        label='Transações Anômalas',
        fill=True,
        color='red'
    )
    
    # Configurações do gráfico
    plt.title('Distribuição dos Scores de Anomalias')
    plt.xlabel('Score de Anomalia')
    plt.ylabel('Densidade')
    plt.legend()
    
    # Salvar ou exibir o gráfico
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Gráfico salvo em {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Caminho para o arquivo de resultados
    scores_file = "results/btc_anomaly_scores.csv"
    save_path = "results/visualizations/btc_anomaly_score_distribution.png"
    
    # Carregar os dados
    data = pd.read_csv(scores_file)
    
    # Gerar visualização
    plot_anomaly_score_distribution(data, save_path=save_path)

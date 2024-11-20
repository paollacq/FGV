import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distributions(data, features, label_column, save_path=None):
    """
    Plota a distribuição das features para transações normais e anômalas.
    """
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[data[label_column] == 0][feature], label='Transações Normais', fill=True, color='blue')
        sns.kdeplot(data[data[label_column] == 1][feature], label='Transações Anômalas', fill=True, color='red')
        plt.title(f'Distribuição de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Densidade')
        plt.legend()
        
        # Salvar ou mostrar o gráfico
        if save_path:
            feature_plot_path = f"{save_path}/{feature}_distribution.png"
            plt.savefig(feature_plot_path, bbox_inches='tight')
            print(f"Gráfico salvo em {feature_plot_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Exemplo de uso
    scores_file = "data/processed/btc_transactions_processed.csv"
    save_path = "results/visualizations/feature_distribution.png"
    
    # Carregar os dados
    data = pd.read_csv(scores_file)
    
    # Colunas numéricas
    numeric_columns = ['total_sent', 'total_received', 'net_flow']
    

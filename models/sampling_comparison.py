# sampling_comparison.py

from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE

def apply_sampling_techniques(X_train, y_train):
    """
    Aplica técnicas de subamostragem e superamostragem nos dados de treinamento.

    Returns:
    - sampled_data: Dicionário com os dados amostrados.
    """
    # Subamostragem com ClusterCentroids
    cc = ClusterCentroids(random_state=42)
    X_resampled_under, y_resampled_under = cc.fit_resample(X_train, y_train)

    # Superamostragem com SMOTE
    smote = SMOTE(random_state=42)
    X_resampled_over, y_resampled_over = smote.fit_resample(X_train, y_train)

    sampled_data = {
        'UnderSampling': (X_resampled_under, y_resampled_under),
        'OverSampling': (X_resampled_over, y_resampled_over)
    }

    return sampled_data

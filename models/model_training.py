# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def prepare_data(features_df, label_df):
    """
    Prepara os dados para treinamento.

    Parameters:
    - features_df: DataFrame com as características extraídas.
    - label_df: DataFrame com as labels correspondentes aos endereços.

    Returns:
    - X_train, X_test, y_train, y_test: Dados divididos para treinamento e teste.
    """
    # Mesclar as labels com as features
    data = features_df.merge(label_df, on='address', how='left')
    data['label'] = data['label'].fillna(0)  # Assumindo que endereços sem label não são fraudulentos

    X = data.drop(['address', 'label'], axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

def train_classifiers(X_train, y_train):
    """
    Treina classificadores baseados em árvores.

    Returns:
    - modelos: Dicionário com os modelos treinados.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)

    adb = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    adb.fit(X_train, y_train)

    modelos = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'AdaBoost': adb
    }

    return modelos

def train_ensemble_models(X_train, y_train, base_models):
    """
    Treina modelos de ensemble.

    Parameters:
    - base_models: Dicionário com os modelos base.

    Returns:
    - ensemble_models: Dicionário com os modelos de ensemble treinados.
    """
    estimators = [(name, model) for name, model in base_models.items()]

    voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
    voting_ensemble.fit(X_train, y_train)

    stacked_ensemble = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacked_ensemble.fit(X_train, y_train)

    ensemble_models = {
        'Voting Ensemble': voting_ensemble,
        'Stacked Ensemble': stacked_ensemble
    }

    return ensemble_models

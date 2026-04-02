"""
Lógica de treinamento e re-treinamento para mitigação de drift.

Implementa as estratégias de re-treinamento discutidas na seção 'Saiba Mais':
  - Re-treinamento com janela deslizante (sliding window)
  - Estratégia champion/challenger (shadow mode)
  - Cross-validation temporal
  - Treinamento online incremental com gatilho de drift

A escolha da janela de atualização é central: "curta o suficiente para
captar a nova realidade, longa o suficiente para evitar decisões precipitadas"
(Lu et al., 2019; Provost & Fawcett, 2013).

Referências:
    Lu, J., et al. (2019). Learning under concept drift: A review.
    IEEE TKDE, 31(12), 2346-2363.

    Provost, F., & Fawcett, T. (2013). Data Science for Business.
    O'Reilly Media.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .model import AdaptiveEnsembleModel, BatchChurnModel, OnlineChurnModel


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic",
    **kwargs: Any,
) -> BatchChurnModel:
    """Treina modelo batch (baseline).

    Implementa o treinamento inicial no regime estável (Período 1),
    conforme discutido na seção 'Saiba Mais': o modelo aprende
    correlações coerentes com o período pré-competição.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        model_type: 'logistic' ou 'random_forest'.
        **kwargs: Parâmetros adicionais para o modelo.

    Returns:
        Modelo treinado.
    """
    model = BatchChurnModel(model_type=model_type, **kwargs)
    model.fit(X_train, y_train)
    return model


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "logistic",
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Validação cruzada estratificada.

    Implementa validação robusta conforme discutido na seção 'Saiba Mais':
    "re-treinar ou reponderar um modelo sem governança pode apenas trocar
    um problema por outro".

    Args:
        X: Features.
        y: Target.
        model_type: Tipo de modelo.
        n_folds: Número de folds.
        seed: Semente.

    Returns:
        Dicionário com scores por fold para cada métrica.
    """
    model = BatchChurnModel(model_type=model_type, random_state=seed)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        scores = cross_val_score(model.model, X, y, cv=cv, scoring=metric)
        results[metric] = scores

    return results


def retrain_sliding_window(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "churn",
    window_size: int = 2000,
    step_size: int = 500,
    model_type: str = "logistic",
) -> list[dict[str, Any]]:
    """Re-treinamento com janela deslizante.

    Implementa a estratégia discutida na seção 'Saiba Mais': "a escolha
    da janela de atualização é central — curta o suficiente para captar
    a nova realidade, longa o suficiente para evitar decisões precipitadas"
    (Lu et al., 2019).

    Args:
        df: DataFrame completo ordenado por timestamp.
        feature_cols: Colunas de features.
        target_col: Coluna target.
        window_size: Tamanho da janela de treino.
        step_size: Passo da janela deslizante.
        model_type: Tipo de modelo.

    Returns:
        Lista de dicionários com métricas em cada janela.
    """
    df_sorted = df.sort_values("timestamp_month").reset_index(drop=True)
    results = []

    for start in range(0, len(df_sorted) - window_size - step_size, step_size):
        train_end = start + window_size
        test_end = min(train_end + step_size, len(df_sorted))

        X_train = df_sorted.loc[start:train_end - 1, feature_cols]
        y_train = df_sorted.loc[start:train_end - 1, target_col]
        X_test = df_sorted.loc[train_end:test_end - 1, feature_cols]
        y_test = df_sorted.loc[train_end:test_end - 1, target_col]

        if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            continue

        model = train_model(X_train, y_train, model_type=model_type)
        metrics = model.score(X_test, y_test)
        metrics["window_start"] = start
        metrics["window_end"] = train_end
        metrics["test_start"] = train_end
        metrics["test_end"] = test_end
        results.append(metrics)

    return results


def champion_challenger(
    X_train_old: pd.DataFrame,
    y_train_old: pd.Series,
    X_train_new: pd.DataFrame,
    y_train_new: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Avaliação champion/challenger (shadow mode).

    Implementa a estratégia discutida na seção 'Saiba Mais' sobre validação
    segmentada e implantação gradual: "a pergunta correta não é 'o novo
    modelo está melhor em média?', mas sim 'ele está melhor nos segmentos
    críticos?'" (Provost & Fawcett, 2013).

    Args:
        X_train_old: Features do modelo champion (dados antigos).
        y_train_old: Target do champion.
        X_train_new: Features do challenger (dados recentes).
        y_train_new: Target do challenger.
        X_test: Features de teste.
        y_test: Target de teste.
        model_type: Tipo de modelo.

    Returns:
        Dicionário com métricas do champion e challenger.
    """
    champion = train_model(X_train_old, y_train_old, model_type=model_type)
    challenger = train_model(X_train_new, y_train_new, model_type=model_type)

    return {
        "champion_metrics": champion.score(X_test, y_test),
        "challenger_metrics": challenger.score(X_test, y_test),
        "champion_model": champion,
        "challenger_model": challenger,
    }


def train_online(
    stream: list[tuple[dict, int]],
    seed: int = 42,
) -> OnlineChurnModel:
    """Treina modelo online com detecção de drift via ADWIN.

    Implementa o aprendizado incremental da seção 'Hands On' (Snippet 1):
    o detector ADWIN sinaliza mudança e abre espaço para medidas como
    recalibração de limiar ou re-treinamento.

    Referência:
        Bifet, A., & Gavaldà, R. (2007). Learning from time-changing
        data with adaptive windowing. SIAM SDM.

    Args:
        stream: Lista de tuplas (features_dict, label).
        seed: Semente.

    Returns:
        Modelo online treinado.
    """
    model = OnlineChurnModel(seed=seed)
    for x, y in stream:
        model.learn_one(x, y)
    return model


def train_adaptive_ensemble(
    stream: list[tuple[dict, int]],
    n_models: int = 5,
    seed: int = 42,
) -> AdaptiveEnsembleModel:
    """Treina ensemble adaptativo ADWINBoostingClassifier.

    Implementa o conceito do Snippet 3 da seção 'Hands On': o ensemble
    incorpora monitoramento de mudança e substituição gradual dos
    membros mais fracos (Krawczyk et al., 2017).

    Args:
        stream: Lista de tuplas (features_dict, label).
        n_models: Número de modelos no ensemble.
        seed: Semente.

    Returns:
        Ensemble adaptativo treinado.
    """
    model = AdaptiveEnsembleModel(n_models=n_models, seed=seed)
    for x, y in stream:
        model.learn_one(x, y)
    return model


def df_to_stream(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "churn",
) -> list[tuple[dict, int]]:
    """Converte DataFrame para stream de tuplas (dict, label).

    Formato necessário para modelos online (River).

    Args:
        df: DataFrame ordenado por timestamp.
        feature_cols: Colunas de features.
        target_col: Coluna target.

    Returns:
        Lista de tuplas (features_dict, label).
    """
    records = df[feature_cols].to_dict(orient="records")
    labels = df[target_col].tolist()
    return list(zip(records, labels))


if __name__ == "__main__":
    from .data_preprocessing import DataPreprocessor

    # Gerar e preparar dados
    preprocessor = DataPreprocessor()
    df = preprocessor.generate_dataset()
    df = preprocessor.clean_data(df)

    # Treinar baseline no período 1
    periods = preprocessor.split_by_period(df)
    X_p1, y_p1 = preprocessor.prepare_features(periods[1])
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_p1, y_p1)

    model = train_model(X_train, y_train)
    print("Baseline (Período 1):", model.score(X_test, y_test))

    # Avaliar degradação no período 2
    X_p2, y_p2 = preprocessor.prepare_features(periods[2])
    print("Baseline → Período 2:", model.score(X_p2, y_p2))

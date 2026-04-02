"""
Módulo de treinamento para o pipeline de detecção de fraude.

Implementa funções de treinamento, validação cruzada e busca de
hiperparâmetros, integrando verificações de drift conforme o
pipeline end-to-end descrito no Documento 04 da Aula 8.

Referências:
    Sculley, D. et al. (NIPS 2015). Hidden technical debt in
        machine learning systems.
    Gama, J. et al. (2014). A survey on concept drift adaptation.
        ACM Computing Surveys, 46(4).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from .model import FraudDetector
from .utils import save_model, save_metrics, set_seed


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42,
    store_reference: bool = True,
    save_path: Optional[str] = None,
) -> FraudDetector:
    """
    Treina o modelo de detecção de fraude.

    Implementa o fluxo de treinamento conforme discutido no
    Documento 04 (seção 'Pipeline Integrado End-to-End'), que
    enfatiza a importância de armazenar distribuições de referência
    durante o treinamento para posterior detecção de drift.

    Args:
        X_train: Features de treinamento.
        y_train: Labels de treinamento (0/1).
        n_estimators: Número de árvores no Random Forest.
        max_depth: Profundidade máxima das árvores.
        random_state: Semente para reproduzibilidade.
        store_reference: Se True, armazena P_ref(X) para drift detection.
        save_path: Caminho para salvar modelo. Se None, não salva.

    Returns:
        Modelo FraudDetector treinado.
    """
    set_seed(random_state)

    detector = FraudDetector(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    detector.fit(X_train, y_train, store_reference=store_reference)

    if save_path:
        save_model(detector, save_path)

    return detector


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Realiza validação cruzada estratificada do modelo.

    Utiliza StratifiedKFold para manter a proporção de classes
    em cada fold, fundamental em cenários com desbalanceamento
    como detecção de fraude (~5% positivos).

    Conforme Documento 04, a validação cruzada fornece estimativa
    robusta de performance antes de implantar o modelo em produção.

    Args:
        X: Features.
        y: Labels.
        n_splits: Número de folds.
        scoring: Métrica de avaliação ('roc_auc', 'f1', 'accuracy').
        random_state: Semente para reproduzibilidade.

    Returns:
        Dicionário com scores por fold e estatísticas (média, std).
    """
    set_seed(random_state)

    detector = FraudDetector(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        detector.model, X, y, cv=cv, scoring=scoring, n_jobs=-1
    )

    return {
        "scoring": scoring,
        "n_splits": n_splits,
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """
    Busca de hiperparâmetros via GridSearchCV.

    Conforme discutido no Documento 04, o re-treino de modelos
    após detecção de drift pode incluir ajuste de hiperparâmetros
    para adaptar o modelo às novas distribuições.

    Args:
        X: Features de treinamento.
        y: Labels.
        param_grid: Grade de hiperparâmetros. Se None, usa grid padrão.
        cv: Número de folds na validação cruzada.
        scoring: Métrica de otimização.
        random_state: Semente para reproduzibilidade.

    Returns:
        Tupla (melhores_params, melhor_score).
    """
    set_seed(random_state)

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
        }

    base_model = FraudDetector(random_state=random_state).model

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    grid_search.fit(X, y)

    return grid_search.best_params_, float(grid_search.best_score_)


def retrain_on_drift(
    detector: FraudDetector,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    X_production: pd.DataFrame,
    drift_features_threshold: int = 3,
    save_path: Optional[str] = None,
) -> Tuple[FraudDetector, bool]:
    """
    Re-treina o modelo se drift significativo for detectado.

    Implementa a lógica de re-treino automático conforme o pipeline
    integrado do Documento 04 (Vídeo 3): detecta drift, avalia
    severidade e, se necessário, re-treina o modelo com dados recentes.

    O limiar de re-treino é baseado no número de features com drift
    detectado, seguindo a abordagem prática discutida por Müller
    et al. (2024) para decisões de re-treino.

    Args:
        detector: Modelo atual (treinado).
        X_new: Dados de treinamento atualizados.
        y_new: Labels atualizados.
        X_production: Dados de produção para verificação de drift.
        drift_features_threshold: Número mínimo de features com drift
            para acionar re-treino.
        save_path: Caminho para salvar modelo re-treinado.

    Returns:
        Tupla (modelo_atualizado, foi_re_treinado).
    """
    # Verificar drift
    drift_results = detector.check_drift(X_production)
    n_drifted = sum(1 for r in drift_results.values() if r["drift_detected"])

    if n_drifted >= drift_features_threshold:
        # Re-treinar com dados novos
        new_detector = train_model(
            X_new, y_new,
            n_estimators=detector.model.n_estimators,
            max_depth=detector.model.max_depth,
            random_state=42,
            store_reference=True,
            save_path=save_path,
        )
        return new_detector, True

    return detector, False

"""
Módulo de treinamento e validação para a Aula 1.

Implementa o ciclo de treinamento, validação cruzada e aprendizado
incremental, conectando-se ao conceito de MLOps para monitoramento
e atualização de modelos discutido no Vídeo 4 da aula.

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning
    Systems. NeurIPS 28.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .model import AdaptiveClassifier, DriftDetector, DriftResult


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss: str = "log_loss",
    random_state: int = 42,
) -> AdaptiveClassifier:
    """Treina um AdaptiveClassifier em modo batch.

    Conforme discutido na seção 'Saiba Mais — Adaptação e Mitigação',
    o treinamento batch é o ponto de partida antes de adotar estratégias
    de aprendizado incremental.

    Args:
        X_train: Features de treino.
        y_train: Labels de treino.
        loss: Função de perda.
        random_state: Semente.

    Returns:
        Modelo treinado.
    """
    clf = AdaptiveClassifier(loss=loss, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Validação cruzada estratificada.

    Args:
        X: Features.
        y: Labels.
        n_folds: Número de folds.
        random_state: Semente.

    Returns:
        Dicionário com listas de acurácias de treino e validação por fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_scores: list[float] = []
    val_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        clf = AdaptiveClassifier(random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        train_scores.append(clf.score(X[train_idx], y[train_idx]))
        val_scores.append(clf.score(X[val_idx], y[val_idx]))

    return {"train_scores": train_scores, "val_scores": val_scores}


def incremental_train(
    clf: AdaptiveClassifier,
    X_batches: list[np.ndarray],
    y_batches: list[np.ndarray],
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
) -> dict[str, list[float]]:
    """Treinamento incremental batch-a-batch com monitoramento.

    Implementa o conceito de aprendizado contínuo com ``partial_fit``
    conforme discutido na seção 'Hands On — Snippet 2' e no Vídeo 4
    (Aprendizado Incremental e Resposta ao Drift).

    A cada batch, o modelo é atualizado e a acurácia no conjunto de
    avaliação é registrada, simulando o ciclo de MLOps descrito no
    material: monitorar → detectar → adaptar.

    Args:
        clf: Classificador adaptativo (pode já estar treinado).
        X_batches: Lista de arrays de features por batch.
        y_batches: Lista de arrays de labels por batch.
        X_eval: Features de avaliação (opcional).
        y_eval: Labels de avaliação (opcional).

    Returns:
        Dicionário com histórico de acurácias por batch.
    """
    eval_scores: list[float] = []

    for X_batch, y_batch in zip(X_batches, y_batches):
        clf.partial_fit(X_batch, y_batch)

        if X_eval is not None and y_eval is not None:
            score = clf.score(X_eval, y_eval)
            eval_scores.append(score)

    return {"eval_scores": eval_scores, "batch_history": clf.history}


def train_with_drift_detection(
    clf: AdaptiveClassifier,
    detector: DriftDetector,
    X_reference: np.ndarray,
    X_batches: list[np.ndarray],
    y_batches: list[np.ndarray],
    feature_names: list[str] | None = None,
) -> dict[str, object]:
    """Treinamento com detecção de drift integrada.

    Combina o ciclo de MLOps completo: a cada batch, detecta drift e,
    se drift for encontrado, faz partial_fit no batch para adaptar o
    modelo.  Caso contrário, apenas registra as métricas.

    Conforme discutido na seção 'Saiba Mais — Monitoramento e Detecção
    de Drift', configurar alertas automáticos baseados em gatilhos
    é prática recomendada em MLOps.

    Args:
        clf: Classificador adaptativo.
        detector: Detector de drift (KS).
        X_reference: Dados de referência (distribuição de treino).
        X_batches: Batches de dados novos.
        y_batches: Batches de labels correspondentes.
        feature_names: Nomes das features.

    Returns:
        Dicionário com histórico de drift e atualizações.
    """
    drift_log: list[dict[str, object]] = []

    for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
        # Detecção de drift no batch
        results = detector.detect_multivariate(
            X_reference, X_batch, feature_names
        )
        summary = detector.summary(results)

        drift_detected = summary["features_with_drift"] > 0
        action = "partial_fit" if drift_detected else "skip"

        # Se drift detectado, atualiza o modelo
        if drift_detected:
            clf.partial_fit(X_batch, y_batch)

        drift_log.append({
            "batch": i + 1,
            "drift_detected": drift_detected,
            "features_with_drift": summary["features_with_drift"],
            "drifted_features": summary["drifted_features"],
            "action": action,
        })

    return {"drift_log": drift_log, "model": clf}

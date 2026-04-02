"""
Pipeline de treinamento com janelas deslizantes e detecção de drift.

Implementa pipeline de monitoramento contínuo com janelas temporais
conforme discutido na seção 'Saiba Mais' da Aula 7. Inclui lógica
de janelas deslizantes, amostragem e triggers de retraining.

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems.
    Gama, J. et al. (2014). A Survey on Concept Drift Adaptation.
    Carbone, P. et al. (2015). Apache Flink: Stream and Batch.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from .model import DriftMonitor, DriftResult, IncrementalPSI


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: Optional[Dict[str, Any]] = None,
) -> GradientBoostingClassifier:
    """
    Treina modelo de detecção de fraude (classificador).

    Utiliza GradientBoosting como baseline para o caso FinBank.
    O modelo é treinado com dados do período de baseline e
    seu desempenho é monitorado ao longo do tempo para detectar
    degradação causada por drift (Sculley et al., 2015).

    Args:
        X_train: Features de treinamento (n_samples, n_features).
        y_train: Rótulos de treinamento.
        model_params: Parâmetros do clasificador. Se None, usa defaults.

    Returns:
        Modelo treinado.
    """
    if model_params is None:
        model_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        }

    clf = GradientBoostingClassifier(**model_params)
    clf.fit(X_train, y_train)
    return clf


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "f1",
) -> Dict[str, float]:
    """
    Realiza validação cruzada do modelo de fraude.

    Args:
        X: Features.
        y: Rótulos.
        cv: Número de folds.
        scoring: Métrica de avaliação.

    Returns:
        Dicionário com mean e std do score.
    """
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return {"mean": float(scores.mean()), "std": float(scores.std())}


def hyperparameter_tuning(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    cv: int = 3,
) -> Tuple[Dict[str, Any], float]:
    """
    Busca de hiperparâmetros via GridSearch.

    Args:
        X: Features de treinamento.
        y: Rótulos.
        param_grid: Grade de parâmetros. Se None, usa grade padrão.
        cv: Número de folds.

    Returns:
        Tupla (melhores_parametros, melhor_score).
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
        }

    clf = GradientBoostingClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X, y)

    return grid.best_params_, float(grid.best_score_)


def train_sliding_window_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    monitor: DriftMonitor,
    window_size: int = 1,
    step_size: int = 1,
    baseline_months: Tuple[int, ...] = (1, 2, 3),
) -> List[Dict[str, Any]]:
    """
    Executa pipeline de monitoramento de drift com janelas deslizantes.

    Implementa a lógica de janelas discutida na seção 'Saiba Mais':
    janelas curtas aumentam sensibilidade e reduzem atraso, mas
    podem produzir estimativas ruidosas; janelas longas estabilizam
    estimativas mas podem esconder drifts súbitos.

    Simula pipeline de streaming em lote (Snippet 4 do Hands On):
    para cada janela, calcula métricas de drift e emite alerta
    quando PSI ≥ limiar ou K–S significativo.

    Args:
        df: DataFrame com coluna 'month' e features numéricas.
        feature_columns: Lista de features a monitorar.
        monitor: Instância de DriftMonitor já configurada.
        window_size: Tamanho da janela em meses.
        step_size: Passo da janela em meses.
        baseline_months: Meses usados como baseline.

    Returns:
        Lista de dicionários com resultados por janela.
    """
    # Construir baseline
    baseline_df = df[df["month"].isin(baseline_months)]
    baseline_data = {
        col: baseline_df[col].values for col in feature_columns
    }
    monitor.fit(baseline_data)

    # Meses disponíveis para janelas
    all_months = sorted(df["month"].unique())
    monitor_months = [m for m in all_months if m not in baseline_months]

    results = []

    for start_idx in range(0, len(monitor_months), step_size):
        window_months = monitor_months[start_idx: start_idx + window_size]
        if not window_months:
            break

        window_df = df[df["month"].isin(window_months)]
        current_data = {
            col: window_df[col].values for col in feature_columns
        }

        # Detectar drift
        drift_results = monitor.predict(current_data)
        score = monitor.score(current_data)

        # Regra de alerta (Snippet 4 do Hands On):
        # se PSI excede limiar OU K–S significativo → alerta
        any_critical = any(r.severity == "critical" for r in drift_results)
        any_warning = any(r.severity == "warning" for r in drift_results)

        results.append({
            "window_months": window_months,
            "n_samples": len(window_df),
            "drift_results": drift_results,
            "aggregate_score": score,
            "alert_level": (
                "CRITICAL" if any_critical
                else "WARNING" if any_warning
                else "OK"
            ),
            "should_retrain": any_critical,
        })

    return results


def simulate_streaming_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    monitor: DriftMonitor,
    batch_size: int = 500,
    baseline_months: Tuple[int, ...] = (1, 2, 3),
    on_drift_callback: Optional[Callable[[Dict], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Simula pipeline de streaming com processamento em mini-batches.

    Emula o comportamento de um pipeline orientado a eventos
    (Apache Kafka + Apache Flink) conforme descrito na Videoaula 2.
    Transações chegam em lotes (simulando micro-batches) e
    métricas de drift são calculadas incrementalmente.

    Referência:
        Kreps, J. et al. (2011). Kafka: A Distributed Messaging System.
        Carbone, P. et al. (2015). Apache Flink: Stream and Batch.

    Args:
        df: DataFrame ordenado por timestamp.
        feature_columns: Features a monitorar.
        monitor: DriftMonitor configurado.
        batch_size: Tamanho do mini-batch.
        baseline_months: Meses de baseline.
        on_drift_callback: Callback chamado quando drift é detectado.

    Returns:
        Lista de resultados de detecção por batch.
    """
    # Configurar baseline
    baseline_df = df[df["month"].isin(baseline_months)]
    baseline_data = {
        col: baseline_df[col].values for col in feature_columns
    }
    monitor.fit(baseline_data)

    # Dados de monitoramento (excluindo baseline)
    stream_df = df[~df["month"].isin(baseline_months)].sort_values("timestamp")

    results = []
    n_batches = len(stream_df) // batch_size + (1 if len(stream_df) % batch_size else 0)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(stream_df))
        batch = stream_df.iloc[start:end]

        if len(batch) == 0:
            break

        current_data = {
            col: batch[col].values for col in feature_columns
        }

        score = monitor.score(current_data)
        drift_results = monitor.predict(current_data)

        result = {
            "batch_idx": batch_idx,
            "n_samples": len(batch),
            "months_in_batch": sorted(batch["month"].unique().tolist()),
            "aggregate_score": score,
            "drift_detected": score["drift_fraction"] > 0,
            "alert_level": (
                "CRITICAL" if score.get("max_psi", 0) >= monitor.psi_threshold
                else "WARNING" if score.get("mean_psi", 0) >= monitor.psi_warning
                else "OK"
            ),
        }
        results.append(result)

        # Callback de drift (simula trigger de retraining)
        if on_drift_callback and result["drift_detected"]:
            on_drift_callback(result)

    return results

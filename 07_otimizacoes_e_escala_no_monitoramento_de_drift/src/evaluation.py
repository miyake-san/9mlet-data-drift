"""
Métricas e visualizações para avaliação de detecção de drift.

Implementa funções de avaliação da qualidade do monitoramento
de drift, incluindo visualizações de PSI ao longo do tempo,
comparação de distribuições e dashboards de alertas.

Referências:
    Gama, J. et al. (2014). A Survey on Concept Drift Adaptation.
    Kullback, S. & Leibler, R. A. (1951). On Information and Sufficiency.
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .model import DriftResult


def calculate_metrics(
    drift_results_timeline: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calcula métricas agregadas de uma timeline de detecção de drift.

    Agrega resultados de múltiplas janelas para fornecer visão
    consolidada do monitoramento, conforme discutido na seção
    'Saiba Mais' — drift, observabilidade e governança.

    Args:
        drift_results_timeline: Lista de resultados por janela
            (saída de train_sliding_window_pipeline).

    Returns:
        Dicionário com métricas agregadas.
    """
    if not drift_results_timeline:
        return {"n_windows": 0}

    n_windows = len(drift_results_timeline)
    critical_count = sum(
        1 for r in drift_results_timeline if r.get("alert_level") == "CRITICAL"
    )
    warning_count = sum(
        1 for r in drift_results_timeline if r.get("alert_level") == "WARNING"
    )
    ok_count = n_windows - critical_count - warning_count

    mean_psi_values = [
        r["aggregate_score"]["mean_psi"]
        for r in drift_results_timeline
        if "aggregate_score" in r
    ]

    return {
        "n_windows": n_windows,
        "critical_windows": critical_count,
        "warning_windows": warning_count,
        "ok_windows": ok_count,
        "critical_rate": critical_count / n_windows,
        "mean_psi_overall": float(np.mean(mean_psi_values)) if mean_psi_values else 0.0,
        "max_psi_overall": float(np.max(mean_psi_values)) if mean_psi_values else 0.0,
    }


def plot_psi_timeline(
    drift_results_timeline: List[Dict[str, Any]],
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    title: str = "PSI ao Longo do Tempo",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota série temporal de PSI médio com limiares de alerta.

    Visualiza a evolução do drift ao longo das janelas, com
    limiares de warning e critical conforme Snippet 4 do Hands On.

    Args:
        drift_results_timeline: Lista de resultados por janela.
        psi_warning: Limiar de warning.
        psi_critical: Limiar de critical.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    window_labels = []
    psi_values = []

    for r in drift_results_timeline:
        months = r.get("window_months", [])
        label = f"M{'-'.join(str(m) for m in months)}"
        window_labels.append(label)
        psi_values.append(r["aggregate_score"]["mean_psi"])

    x = range(len(psi_values))
    colors = [
        "red" if v >= psi_critical
        else "orange" if v >= psi_warning
        else "green"
        for v in psi_values
    ]

    ax.bar(x, psi_values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(y=psi_warning, color="orange", linestyle="--", label=f"Warning ({psi_warning})")
    ax.axhline(y=psi_critical, color="red", linestyle="--", label=f"Crítico ({psi_critical})")

    ax.set_xticks(list(x))
    ax.set_xticklabels(window_labels, rotation=45, ha="right")
    ax.set_xlabel("Janela Temporal")
    ax.set_ylabel("PSI Médio")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distribution_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "Feature",
    n_bins: int = 30,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compara visualmente distribuições de referência e corrente.

    Gera histogramas sobrepostos e CDFs empíricas lado a lado,
    conforme as métricas de comparação discutidas na seção
    'Saiba Mais' (PSI por histograma, K–S por CDF).

    Args:
        reference: Array da distribuição de referência.
        current: Array da distribuição corrente.
        feature_name: Nome da feature para título.
        n_bins: Número de bins do histograma.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    axes[0].hist(reference, bins=n_bins, alpha=0.5, density=True,
                 label="Referência", color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].hist(current, bins=n_bins, alpha=0.5, density=True,
                 label="Corrente", color="coral", edgecolor="black", linewidth=0.5)
    axes[0].set_title(f"Distribuição: {feature_name}")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Densidade")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # CDF empírica (K–S)
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    cur_cdf = np.arange(1, len(cur_sorted) + 1) / len(cur_sorted)

    axes[1].plot(ref_sorted, ref_cdf, label="Referência (F_n)", color="steelblue")
    axes[1].plot(cur_sorted, cur_cdf, label="Corrente (G_m)", color="coral")
    axes[1].set_title(f"CDF Empírica: {feature_name} (K–S)")
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("CDF")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_heatmap(
    drift_results_by_window: List[Dict[str, Any]],
    feature_columns: List[str],
    title: str = "Heatmap de PSI por Feature e Janela",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Gera heatmap de PSI por feature e janela temporal.

    Permite visualizar quais features e janelas concentram
    mais drift, apoiando priorização de monitoramento por
    criticidade de negócio (Videoaula 4).

    Args:
        drift_results_by_window: Lista de resultados por janela
            (cada item contém 'drift_results' com DriftResult).
        feature_columns: Nomes das features.
        title: Título do heatmap.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    matrix = np.zeros((len(feature_columns), len(drift_results_by_window)))

    for j, window in enumerate(drift_results_by_window):
        for dr in window.get("drift_results", []):
            if dr.feature in feature_columns:
                i = feature_columns.index(dr.feature)
                matrix[i, j] = dr.psi_value

    window_labels = [
        f"M{'-'.join(str(m) for m in w.get('window_months', []))}"
        for w in drift_results_by_window
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="PSI")

    ax.set_xticks(range(len(window_labels)))
    ax.set_xticklabels(window_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_columns)))
    ax.set_yticklabels(feature_columns)
    ax.set_title(title)
    ax.set_xlabel("Janela Temporal")
    ax.set_ylabel("Feature")

    # Anotar valores
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.15 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão — Modelo de Fraude",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota matriz de confusão para modelo de classificação de fraude.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    cm = sk_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legítima", "Fraude"],
                yticklabels=["Legítima", "Fraude"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Curva ROC — Modelo de Fraude",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota curva ROC para o modelo de classificação.

    Args:
        y_true: Rótulos verdadeiros.
        y_scores: Scores de probabilidade.
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def evaluate_drift_detection(
    drift_results_timeline: List[Dict[str, Any]],
    expected_drift_windows: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Avalia qualidade da detecção de drift contra ground truth.

    Compara janelas onde drift foi detectado vs. janelas onde
    drift era esperado (ground truth do dataset sintético).

    Args:
        drift_results_timeline: Resultados do pipeline de janelas.
        expected_drift_windows: Índices das janelas com drift real.

    Returns:
        Dicionário com precision, recall e F1 da detecção.
    """
    n_windows = len(drift_results_timeline)
    detected = set()
    for i, r in enumerate(drift_results_timeline):
        if r.get("alert_level") in ("CRITICAL", "WARNING"):
            detected.add(i)

    if expected_drift_windows is None:
        return {
            "n_detected": len(detected),
            "detected_windows": sorted(detected),
        }

    expected = set(expected_drift_windows)

    tp = len(detected & expected)
    fp = len(detected - expected)
    fn = len(expected - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

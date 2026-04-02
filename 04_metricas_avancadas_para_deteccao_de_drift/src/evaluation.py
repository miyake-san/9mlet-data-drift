# -*- coding: utf-8 -*-
"""
evaluation.py – Métricas de avaliação e visualização para detecção de drift.

Implementa funções de avaliação e comparação entre métricas de drift
conforme discutido no Documento 04 (Seção "Saiba Mais"):
  - Comparação PSI vs MMD (sensibilidade a drift multivariado).
  - Visualização de distribuições e resultados de drift.
  - Relatórios de métricas consolidados.

Referências:
    Gretton, A. et al. (2012). A Kernel Two-Sample Test. JMLR, 13, 723–773.
    Székely, G. J.; Rizzo, M. L. (2013). Energy statistics. JSPI, 143(8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .model import (
    DriftDetector,
    DriftResult,
    EnergyDistanceCalculator,
    MMDCalculator,
    PSICalculator,
    WassersteinCalculator,
)

# Estilo de figuras (modo não-interativo para notebooks e scripts)
matplotlib.rcParams.update({"figure.max_open_warning": 0})


def calculate_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calcula todas as métricas de drift e retorna resumo consolidado.

    Combina PSI (univariado), MMD, Wasserstein e Energy Distance
    para análise comparativa, conforme Vídeos 3 e 4 (Documento 04).

    Args:
        reference: Dados de referência (n, d).
        current: Dados atuais (m, d).
        feature_names: Nomes das features (opcional).

    Returns:
        Dicionário com resultados de cada métrica.
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # PSI por feature — Documento 04, Snippet 1
    psi_calc = PSICalculator()
    psi_values = psi_calc.calculate_multifeature(reference, current)

    # MMD multivariado — Documento 04, Snippet 2
    mmd_calc = MMDCalculator()
    mmd_value = mmd_calc.calculate(reference, current)

    # Wasserstein por feature — Documento 04, Saiba Mais
    wass_calc = WassersteinCalculator()
    wass_values = wass_calc.calculate_multifeature(reference, current)

    # Energy Distance — Documento 04, Saiba Mais (Székely e Rizzo, 2013)
    energy_calc = EnergyDistanceCalculator()
    energy_value = energy_calc.calculate(reference, current)

    return {
        "psi": {
            "per_feature": dict(zip(feature_names, psi_values)),
            "max": max(psi_values),
            "mean": float(np.mean(psi_values)),
        },
        "mmd": mmd_value,
        "wasserstein": {
            "per_feature": dict(zip(feature_names, wass_values)),
            "mean": float(np.mean(wass_values)),
        },
        "energy_distance": energy_value,
    }


def plot_distributions(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota histogramas comparativos de referência vs atual por feature.

    Visualiza as distribuições marginais conforme a Figura 1 do
    Documento 04: quando o drift é apenas multivariado, as marginais
    parecem similares.

    Args:
        reference: Dados de referência (n, d).
        current: Dados atuais (m, d).
        feature_names: Nomes das features.
        figsize: Tamanho da figura.
        save_path: Caminho para salvar a figura (opcional).

    Returns:
        Objeto Figure do matplotlib.
    """
    reference = np.asarray(reference)
    current = np.asarray(current)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    if n_features == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(reference[:, i], bins=30, alpha=0.5, label="Referência", density=True)
        ax.hist(current[:, i], bins=30, alpha=0.5, label="Atual", density=True)
        ax.set_title(feature_names[i])
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)

    fig.suptitle("Distribuições Marginais: Referência vs Atual", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scatter_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_x: int = 0,
    feature_y: int = 1,
    labels: Optional[Tuple[str, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota diagrama de dispersão para visualizar drift multivariado.

    Conforme Figura 1 do Documento 04: o gráfico de dispersão revela
    a inversão de correlação entre variáveis que os histogramas
    marginais não capturam.

    Args:
        reference: Dados de referência (n, d).
        current: Dados atuais (m, d).
        feature_x: Índice da feature no eixo X.
        feature_y: Índice da feature no eixo Y.
        labels: Rótulos das features (x, y).
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    reference = np.asarray(reference)
    current = np.asarray(current)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        reference[:, feature_x], reference[:, feature_y],
        alpha=0.3, s=10, label="Referência", color="steelblue",
    )
    ax.scatter(
        current[:, feature_x], current[:, feature_y],
        alpha=0.3, s=10, label="Atual", color="tomato",
    )

    x_label = labels[0] if labels else f"Feature {feature_x}"
    y_label = labels[1] if labels else f"Feature {feature_y}"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Drift Multivariado: Inversão de Correlação\n(Documento 04, Figura 1)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota comparação entre métricas de drift.

    Conforme Vídeo 3 (Documento 04): comparação de sensibilidade
    PSI vs MMD para drift multivariado sutil.

    Args:
        metrics: Dicionário retornado por calculate_metrics().
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: PSI por feature
    psi_data = metrics["psi"]["per_feature"]
    names = list(psi_data.keys())
    values = list(psi_data.values())
    colors = ["tomato" if v > 0.25 else "steelblue" for v in values]

    axes[0].barh(names, values, color=colors)
    axes[0].axvline(x=0.25, color="red", linestyle="--", label="Limiar (0.25)")
    axes[0].set_xlabel("PSI")
    axes[0].set_title("PSI por Feature (Univariado)")
    axes[0].legend(fontsize=9)

    # Gráfico 2: Métricas multivariadas
    multi_metrics = {
        "MMD": metrics["mmd"],
        "Wasserstein\n(média)": metrics["wasserstein"]["mean"],
        "Energy\nDistance": metrics["energy_distance"],
    }
    m_names = list(multi_metrics.keys())
    m_values = list(multi_metrics.values())

    axes[1].bar(m_names, m_values, color=["steelblue", "seagreen", "goldenrod"])
    axes[1].set_ylabel("Valor da Métrica")
    axes[1].set_title("Métricas Multivariadas de Drift")

    fig.suptitle(
        "Comparação: Métricas Univariadas x Multivariadas (Doc 04, Vídeo 3)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_heatmaps(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota heatmaps de correlação lado a lado.

    Conforme Documento 04, Vídeo 1: evidencia mudanças na estrutura
    de correlação entre features que caracterizam drift multivariado.

    Args:
        reference: Dados de referência.
        current: Dados atuais.
        feature_names: Nomes das features.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure.
    """
    import pandas as pd

    if feature_names is None:
        n = reference.shape[1] if reference.ndim > 1 else 1
        feature_names = [f"F{i}" for i in range(n)]

    df_ref = pd.DataFrame(reference, columns=feature_names)
    df_cur = pd.DataFrame(current, columns=feature_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        df_ref.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, ax=ax1, square=True,
    )
    ax1.set_title("Correlação – Referência")

    sns.heatmap(
        df_cur.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, ax=ax2, square=True,
    )
    ax2.set_title("Correlação – Atual")

    fig.suptitle(
        "Mudança na Estrutura de Correlação (Documento 04, Vídeo 1)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

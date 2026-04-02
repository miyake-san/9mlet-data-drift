"""
Módulo de avaliação e métricas para o pipeline de detecção de fraude.

Implementa funções de cálculo de métricas de performance, métricas de drift
e visualizações (confusion matrix, ROC curve, distribuições) conforme
discutido no Documento 04 da Aula 8.

As métricas de drift implementadas (KS, Wasserstein, Jensen-Shannon) são
as mesmas utilizadas pelo Evidently AI internamente, permitindo ao aluno
compreender o que ocorre "por baixo dos panos" da ferramenta.

Referências:
    Rabanser, S. et al. (NeurIPS 2019). Failing loudly: an empirical
        study of methods for detecting dataset shift.
    Müller, R. et al. (2024). Open-source drift detection tools in action.
        arXiv:2404.18673.
    Moreno-Torres, J. G. et al. (2012). A unifying view on dataset shift
        in classification. Pattern Recognition, 45(1), 521-530.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Métricas de Performance do Modelo
# ---------------------------------------------------------------------------


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calcula métricas de performance do modelo de fraude.

    Implementa as métricas discutidas no Documento 04 (seção 'Pipeline
    Integrado End-to-End'): accuracy, precision, recall, f1 e ROC AUC.
    Essas métricas são as mesmas que o NannyML CBPE tenta estimar sem
    acesso aos rótulos verdadeiros.

    Referência:
        Müller, R. et al. (2024). Open-source drift detection tools in
        action — mostra a eficácia do NannyML em estimar essas métricas.

    Args:
        y_true: Labels verdadeiros (0/1).
        y_pred: Predições binárias.
        y_proba: Probabilidades da classe positiva (para ROC AUC).

    Returns:
        Dicionário com métricas de performance.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Gera relatório de classificação formatado.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições.
        target_names: Nomes das classes. Padrão: ['Legítima', 'Fraude'].

    Returns:
        String com relatório de classificação.
    """
    if target_names is None:
        target_names = ["Legítima", "Fraude"]
    return classification_report(y_true, y_pred, target_names=target_names)


# ---------------------------------------------------------------------------
# Métricas de Drift
# ---------------------------------------------------------------------------


def calculate_drift_metrics(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 30,
) -> Dict[str, float]:
    """
    Calcula métricas de drift entre distribuição de referência e produção.

    Implementa os testes estatísticos discutidos no Documento 04 (seção
    'Detecção Estatística de Data Drift'):

    - **KS statistic**: D_{n,m} = sup_x |F_ref(x) - F_prod(x)|
    - **Wasserstein distance**: custo de transformar P em Q
    - **Jensen-Shannon divergence**: versão simétrica da KL divergence

    Essas métricas são as mesmas que o Evidently AI calcula internamente
    ao gerar relatórios de DataDriftPreset(), conforme Snippet 1 do
    Hands On do Documento 04.

    Referência:
        Rabanser, S. et al. (NeurIPS 2019). Failing loudly: an empirical
        study of methods for detecting dataset shift.

    Args:
        reference: Valores da distribuição de referência (baseline).
        production: Valores da distribuição de produção (atual).
        n_bins: Número de bins para histograma (Jensen-Shannon).

    Returns:
        Dicionário com métricas de drift e indicação de detecção.
    """
    # Teste KS — conforme Documento 04
    ks_stat, ks_pvalue = stats.ks_2samp(reference, production)

    # Wasserstein distance — "Earth Mover's Distance"
    w_dist = float(wasserstein_distance(reference, production))

    # Jensen-Shannon divergence — versão simétrica da KL
    # Criar histogramas normalizados para as duas distribuições
    all_data = np.concatenate([reference, production])
    bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    prod_hist, _ = np.histogram(production, bins=bins, density=True)

    # Adicionar epsilon para evitar log(0)
    ref_hist = ref_hist + 1e-10
    prod_hist = prod_hist + 1e-10

    js_div = float(jensenshannon(ref_hist, prod_hist) ** 2)

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "ks_drift_detected": ks_pvalue < 0.05,
        "wasserstein_distance": w_dist,
        "jensen_shannon_divergence": js_div,
    }


def calculate_all_features_drift(
    df_reference: pd.DataFrame,
    df_production: pd.DataFrame,
    numeric_features: List[str],
) -> pd.DataFrame:
    """
    Calcula métricas de drift para todas as features numéricas.

    Aplica calculate_drift_metrics para cada feature e retorna um
    DataFrame resumo, similar ao relatório gerado pelo Evidently AI
    (DataDriftPreset), conforme demonstrado no Documento 04.

    Args:
        df_reference: DataFrame de referência (baseline).
        df_production: DataFrame de produção.
        numeric_features: Lista de features numéricas a avaliar.

    Returns:
        DataFrame com métricas de drift por feature.
    """
    results = []
    for feat in numeric_features:
        if feat not in df_reference.columns or feat not in df_production.columns:
            continue
        ref_vals = df_reference[feat].dropna().values
        prod_vals = df_production[feat].dropna().values
        drift = calculate_drift_metrics(ref_vals, prod_vals)
        drift["feature"] = feat
        results.append(drift)

    return pd.DataFrame(results).set_index("feature")


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Matriz de Confusão — Detecção de Fraude",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota matriz de confusão do modelo de detecção de fraude.

    Conforme Documento 04 (seção 'Pipeline Integrado'), a matriz de
    confusão é essencial para compreender o trade-off entre falsos
    positivos (transações legítimas bloqueadas) e falsos negativos
    (fraudes não detectadas).

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições.
        labels: Nomes das classes. Padrão: ['Legítima', 'Fraude'].
        title: Título do gráfico.
        save_path: Caminho para salvar a figura. Se None, não salva.

    Returns:
        Objeto Figure do matplotlib.
    """
    if labels is None:
        labels = ["Legítima", "Fraude"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Curva ROC — Detecção de Fraude",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota curva ROC do modelo de detecção de fraude.

    O ROC AUC é a métrica principal estimada pelo NannyML CBPE
    sem acesso aos rótulos verdadeiros, conforme Snippet 2 do
    Hands On do Documento 04.

    Args:
        y_true: Labels verdadeiros (0/1).
        y_proba: Probabilidades da classe positiva.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Curva Precision-Recall — Detecção de Fraude",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota curva Precision-Recall.

    Em cenários de fraude com classes desbalanceadas (~5% positivos),
    a curva PR é mais informativa que ROC, conforme discutido
    implicitamente no Documento 04 ao abordar métricas de performance.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades da classe positiva.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#FF5722", lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_comparison(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota distribuições de referência vs produção lado a lado.

    Visualização similar ao relatório do Evidently AI, mostrando
    histogramas comparativos das distribuições P_ref(X) e P_prod(X),
    conforme discutido na seção 'Detecção Estatística' do Documento 04.

    Args:
        reference: Valores de referência (baseline).
        production: Valores de produção.
        feature_name: Nome da feature.
        title: Título personalizado. Se None, gera automaticamente.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    if title is None:
        title = f"Distribuição de '{feature_name}' — Referência vs Produção"

    # Calcular drift para anotação
    drift = calculate_drift_metrics(reference, production)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramas sobrepostos
    axes[0].hist(reference, bins=40, alpha=0.6, label="Referência", color="#2196F3", density=True)
    axes[0].hist(production, bins=40, alpha=0.6, label="Produção", color="#FF5722", density=True)
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Densidade")
    axes[0].set_title("Histograma Comparativo")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparativo
    data_for_box = pd.DataFrame({
        "Valor": np.concatenate([reference, production]),
        "Período": (["Referência"] * len(reference)) + (["Produção"] * len(production)),
    })
    sns.boxplot(data=data_for_box, x="Período", y="Valor", ax=axes[1], palette=["#2196F3", "#FF5722"])
    axes[1].set_title("Box Plot Comparativo")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"{title}\n"
        f"KS={drift['ks_statistic']:.4f} (p={drift['ks_pvalue']:.4e}) | "
        f"Wasserstein={drift['wasserstein_distance']:.4f} | "
        f"JS={drift['jensen_shannon_divergence']:.4f}",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_summary(
    drift_df: pd.DataFrame,
    title: str = "Resumo de Drift por Feature",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota resumo de drift para todas as features (heatmap de p-values).

    Visualização inspirada nos relatórios do Evidently AI, destacando
    quais features apresentam drift significativo (p < 0.05).

    Args:
        drift_df: DataFrame de calculate_all_features_drift().
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(drift_df) * 0.5)))

    # KS statistic por feature
    colors = ["#FF5722" if d else "#4CAF50" for d in drift_df["ks_drift_detected"]]
    axes[0].barh(drift_df.index, drift_df["ks_statistic"], color=colors)
    axes[0].axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="Threshold sugerido")
    axes[0].set_xlabel("KS Statistic")
    axes[0].set_title("Estatística KS por Feature")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="x")

    # Wasserstein distance por feature
    axes[1].barh(drift_df.index, drift_df["wasserstein_distance"], color="#2196F3")
    axes[1].set_xlabel("Wasserstein Distance")
    axes[1].set_title("Distância de Wasserstein por Feature")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

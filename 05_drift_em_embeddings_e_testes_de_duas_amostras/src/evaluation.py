"""
Módulo de avaliação e visualização para detecção de drift em embeddings.

Implementa funções de métricas, visualização de distribuições e geração
de relatórios de drift conforme os métodos discutidos na seção 'Saiba Mais'
do documento da Aula 5.

Inclui visualizações com UMAP e t-SNE para projeção de embeddings de alta
dimensionalidade, heatmaps de kernel, e gráficos de monitoramento temporal.

Referências:
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
    Massey Jr., F.J. (1951). The Kolmogorov-Smirnov Test for Goodness of Fit.
    Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier two-sample tests.
    Greco, S. et al. (2024). Unsupervised concept drift detection from
    deep learning representations in real-time. arXiv:2406.17813.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

from .model import EmbeddingDriftDetector


def calculate_drift_metrics(
    emb_ref: np.ndarray,
    emb_prod: np.ndarray,
    methods: Optional[List[str]] = None,
    gamma: float = 1.0,
    n_permutations: int = 100,
) -> Dict[str, Any]:
    """
    Calcula métricas de drift entre embeddings de referência e produção.

    Aplica múltiplos métodos de detecção conforme discutido na seção 'Saiba Mais'
    do documento da Aula 5: MMD (Gretton et al., 2012), KS (Massey, 1951) e
    classificador adversário (Lopez-Paz & Oquab, 2017).

    Args:
        emb_ref: Embeddings de referência (n, d).
        emb_prod: Embeddings de produção (m, d).
        methods: Lista de métodos a aplicar. Default: ['mmd', 'ks', 'adversarial'].
        gamma: Parâmetro do kernel RBF para MMD.
        n_permutations: Número de permutações para p-value do MMD.

    Returns:
        Dicionário com resultados de cada método.
    """
    if methods is None:
        methods = ["mmd", "ks", "adversarial"]

    results: Dict[str, Any] = {}

    for method in methods:
        detector = EmbeddingDriftDetector(
            method=method, gamma=gamma, n_permutations=n_permutations,
        )
        detector.fit(emb_ref)
        result = detector.predict(emb_prod)
        results[method] = result

    # Estatísticas descritivas dos embeddings
    results["embedding_stats"] = {
        "ref_mean_norm": float(np.mean(np.linalg.norm(emb_ref, axis=1))),
        "prod_mean_norm": float(np.mean(np.linalg.norm(emb_prod, axis=1))),
        "ref_std_norm": float(np.std(np.linalg.norm(emb_ref, axis=1))),
        "prod_std_norm": float(np.std(np.linalg.norm(emb_prod, axis=1))),
        "cosine_similarity_mean": float(_mean_cosine_similarity(emb_ref, emb_prod)),
    }

    return results


def plot_embedding_distributions(
    emb_ref: np.ndarray,
    emb_prod: np.ndarray,
    method: str = "umap",
    labels_ref: Optional[np.ndarray] = None,
    labels_prod: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Visualiza distribuições de embeddings usando redução de dimensionalidade.

    Conforme discutido na Aula 5, UMAP e t-SNE projetam embeddings de alta
    dimensionalidade em 2D para visualização, permitindo identificar visualmente
    se os clusters de referência e produção se sobrepõem (sem drift) ou se
    separam (drift detectado).

    Args:
        emb_ref: Embeddings de referência (n, d).
        emb_prod: Embeddings de produção (m, d).
        method: Método de redução ('umap' ou 'tsne').
        labels_ref: Rótulos de categoria para referência (opcional).
        labels_prod: Rótulos de categoria para produção (opcional).
        title: Título do gráfico.
        save_path: Caminho para salvar a figura (opcional).
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    # Combina dados para projeção conjunta
    combined = np.vstack([emb_ref, emb_prod])
    n_ref = len(emb_ref)

    # Redução de dimensionalidade
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            projected = reducer.fit_transform(combined)
        except ImportError:
            print("UMAP não disponível. Usando t-SNE como fallback.")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        projected = reducer.fit_transform(combined)

    proj_ref = projected[:n_ref]
    proj_prod = projected[n_ref:]

    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Referência vs Produção
    ax1 = axes[0]
    ax1.scatter(proj_ref[:, 0], proj_ref[:, 1], alpha=0.4, s=10,
                label="Referência", c="steelblue")
    ax1.scatter(proj_prod[:, 0], proj_prod[:, 1], alpha=0.4, s=10,
                label="Produção", c="coral")
    ax1.set_xlabel(f"{method.upper()} Dim 1")
    ax1.set_ylabel(f"{method.upper()} Dim 2")
    ax1.set_title("Referência vs Produção")
    ax1.legend()

    # Plot 2: Por categoria (se labels disponíveis)
    ax2 = axes[1]
    if labels_ref is not None and labels_prod is not None:
        all_labels = np.concatenate([labels_ref, labels_prod])
        unique_labels = np.unique(all_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            ax2.scatter(
                projected[mask, 0], projected[mask, 1],
                alpha=0.4, s=10, label=label, c=[colors[i]],
            )
        ax2.set_title("Distribuição por Categoria")
    else:
        ax2.scatter(proj_ref[:, 0], proj_ref[:, 1], alpha=0.3, s=10,
                    label="Referência", c="steelblue", marker="o")
        ax2.scatter(proj_prod[:, 0], proj_prod[:, 1], alpha=0.3, s=10,
                    label="Produção", c="coral", marker="x")
        ax2.set_title("Referência vs Produção (alternativo)")

    ax2.set_xlabel(f"{method.upper()} Dim 1")
    ax2.set_ylabel(f"{method.upper()} Dim 2")
    ax2.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_scores(
    scores: List[float],
    timestamps: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    title: str = "Drift Score ao Longo do Tempo",
    ylabel: str = "MMD²",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plota scores de drift ao longo do tempo com limiar de alerta.

    Conforme discutido no Vídeo 4 da Aula 5, o monitoramento contínuo
    de drift requer visualização temporal dos scores com limiares
    para disparo de alertas.

    Args:
        scores: Lista de scores de drift.
        timestamps: Labels temporais (opcional).
        threshold: Limiar de alerta (opcional).
        title: Título do gráfico.
        ylabel: Label do eixo Y.
        save_path: Caminho para salvar a figura.
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(scores))
    if timestamps is not None:
        x = timestamps

    ax.plot(x, scores, marker="o", linewidth=2, markersize=4,
            color="steelblue", label="Drift Score")

    if threshold is not None:
        ax.axhline(y=threshold, color="red", linestyle="--",
                   linewidth=1.5, label=f"Limiar ({threshold:.4f})")
        # Destaca pontos acima do limiar
        above = [s > threshold for s in scores]
        if any(above):
            ax.scatter(
                [i for i, a in enumerate(above) if a],
                [s for s, a in zip(scores, above) if a],
                color="red", s=60, zorder=5, label="Drift Detectado",
            )

    ax.set_xlabel("Tempo / Batch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_mmd_heatmap(
    emb_ref: np.ndarray,
    emb_prod: np.ndarray,
    gamma: float = 1.0,
    max_samples: int = 200,
    title: str = "Heatmap do Kernel RBF (Referência vs Produção)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plota heatmap da matriz de kernel entre referência e produção.

    Visualiza as similaridades medidas pelo kernel RBF (k(u,v) = exp(-γ||u-v||²))
    entre amostras de referência e produção. Blocos diagonais claros indicam
    alta similaridade intra-grupo; blocos off-diagonal escuros indicam drift.

    Args:
        emb_ref: Embeddings de referência.
        emb_prod: Embeddings de produção.
        gamma: Parâmetro do kernel RBF.
        max_samples: Máximo de amostras por grupo para visualização.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    # Subsample para visualização
    n_ref = min(len(emb_ref), max_samples)
    n_prod = min(len(emb_prod), max_samples)

    idx_ref = np.random.choice(len(emb_ref), n_ref, replace=False)
    idx_prod = np.random.choice(len(emb_prod), n_prod, replace=False)

    combined = np.vstack([emb_ref[idx_ref], emb_prod[idx_prod]])
    K = rbf_kernel(combined, combined, gamma=gamma)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(K, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Similaridade (kernel RBF)")

    # Linhas separadoras
    ax.axhline(y=n_ref - 0.5, color="red", linewidth=2, linestyle="--")
    ax.axvline(x=n_ref - 0.5, color="red", linewidth=2, linestyle="--")

    ax.set_xlabel("Amostras")
    ax.set_ylabel("Amostras")
    ax.set_title(title)

    # Anotações
    ax.text(n_ref / 2, -3, "Ref", ha="center", fontsize=10, color="blue")
    ax.text(n_ref + n_prod / 2, -3, "Prod", ha="center", fontsize=10, color="red")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_drift_report(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Gera relatório resumido de detecção de drift.

    Args:
        metrics: Dicionário retornado por calculate_drift_metrics().
        output_path: Caminho para salvar o relatório (opcional).

    Returns:
        String com o relatório formatado.
    """
    lines = [
        "=" * 60,
        "RELATÓRIO DE DETECÇÃO DE DRIFT EM EMBEDDINGS",
        "=" * 60,
        "",
    ]

    for method_name in ["mmd", "ks", "adversarial"]:
        if method_name not in metrics:
            continue
        result = metrics[method_name]
        lines.append(f"--- {method_name.upper()} ---")
        lines.append(f"  Score:           {result.get('score', 'N/A'):.6f}")
        if "p_value" in result:
            lines.append(f"  P-value:         {result['p_value']:.6f}")
        if "accuracy" in result:
            lines.append(f"  Accuracy:        {result['accuracy']:.4f}")
        lines.append(f"  Drift Detectado: {'SIM' if result.get('drift_detected') else 'NÃO'}")
        lines.append("")

    if "embedding_stats" in metrics:
        stats = metrics["embedding_stats"]
        lines.append("--- ESTATÍSTICAS DE EMBEDDINGS ---")
        lines.append(f"  Norma média (ref):   {stats['ref_mean_norm']:.4f}")
        lines.append(f"  Norma média (prod):  {stats['prod_mean_norm']:.4f}")
        lines.append(f"  Similaridade cosseno: {stats['cosine_similarity_mean']:.4f}")
        lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def _mean_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Calcula similaridade cosseno média entre dois conjuntos de embeddings."""
    # Normaliza os vetores
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)

    # Média dos centróides
    centroid_X = X_norm.mean(axis=0)
    centroid_Y = Y_norm.mean(axis=0)

    # Similaridade cosseno entre centróides
    cos_sim = np.dot(centroid_X, centroid_Y) / (
        np.linalg.norm(centroid_X) * np.linalg.norm(centroid_Y) + 1e-10
    )
    return float(cos_sim)

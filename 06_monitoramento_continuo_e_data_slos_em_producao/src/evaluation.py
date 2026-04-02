"""
Módulo de avaliação de modelos de ML para monitoramento contínuo.

Implementa métricas de avaliação, Brier Score para calibração e
visualizações para análise de desempenho do modelo de crédito,
conforme discutido no DOCUMENTO_AULA_6.md.

Referências:
    Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty?
    Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.

    Breck, E. et al. (2017). The ML Test Score: A Rubric for ML
    Production Readiness and Technical Debt Reduction. IEEE BigData.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Avaliador de modelos de ML com foco em monitoramento.

    Implementa cálculo de métricas, Brier Score e visualizações
    para avaliar o desempenho do modelo e sua calibração, conforme
    discutido na seção 'SLIs e Data SLOs em ML' do DOCUMENTO_AULA_6.md.

    O Brier Score mede calibração probabilística:
    BS = (1/N) Σ (p_i - y_i)²

    Conforme Ovadia et al. (2019), tanto a acurácia quanto a calibração
    se deterioram sob dataset shift.

    Attributes:
        figures_dir: Diretório para salvar gráficos.
    """

    def __init__(self, figures_dir: Optional[str] = None) -> None:
        """Inicializa o avaliador.

        Args:
            figures_dir: Diretório para salvar figuras. Se None, usa
                         outputs/figures/ relativo ao módulo.
        """
        if figures_dir is None:
            self.figures_dir = Path(__file__).resolve().parent.parent / "outputs" / "figures"
        else:
            self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calcula métricas de avaliação do modelo.

        Implementa métricas discutidas na seção 'SLIs e Data SLOs em ML'
        do DOCUMENTO_AULA_6.md, incluindo acurácia (SLI de desempenho)
        e Brier Score (calibração).

        BS = (1/N) Σ (p_i - y_i)²
        conforme fórmula da seção 'Saiba Mais' do documento 04.

        Args:
            y_true: Rótulos verdadeiros.
            y_pred: Previsões do modelo.
            y_proba: Probabilidades da classe positiva (para Brier/AUC).

        Returns:
            Dicionário com métricas calculadas.
        """
        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            # Brier Score: mede calibração probabilística
            # Conforme Ovadia et al. (2019): modelos deterioram calibração sob shift
            metrics["brier_score"] = brier_score_loss(y_true, y_proba)

        logger.info(f"Métricas calculadas: {metrics}")
        return metrics

    def compare_metrics(
        self,
        metrics_ref: Dict[str, float],
        metrics_prod: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compara métricas entre referência e produção.

        Permite identificar degradação de performance conforme
        discutido na seção 'Por Que Monitorar Modelos de ML' do doc 04.

        Args:
            metrics_ref: Métricas nos dados de referência.
            metrics_prod: Métricas nos dados de produção.

        Returns:
            Dicionário com comparação (ref, prod, delta).
        """
        comparison: Dict[str, Dict[str, float]] = {}

        for key in metrics_ref:
            if key in metrics_prod:
                delta = metrics_prod[key] - metrics_ref[key]
                comparison[key] = {
                    "reference": metrics_ref[key],
                    "production": metrics_prod[key],
                    "delta": delta,
                }

        return comparison

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Matriz de Confusão",
        labels: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "confusion_matrix.png",
    ) -> plt.Figure:
        """Gera gráfico de matriz de confusão.

        Args:
            y_true: Rótulos verdadeiros.
            y_pred: Previsões do modelo.
            title: Título do gráfico.
            labels: Nomes das classes.
            save: Se True, salva a figura.
            filename: Nome do arquivo.

        Returns:
            Objeto Figure do matplotlib.
        """
        if labels is None:
            labels = ["Adimplente", "Inadimplente"]

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Real")
        ax.set_title(title)
        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / filename, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva: {self.figures_dir / filename}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Curva ROC",
        save: bool = True,
        filename: str = "roc_curve.png",
    ) -> plt.Figure:
        """Gera gráfico da curva ROC.

        Args:
            y_true: Rótulos verdadeiros.
            y_proba: Probabilidades da classe positiva.
            title: Título do gráfico.
            save: Se True, salva a figura.
            filename: Nome do arquivo.

        Returns:
            Objeto Figure do matplotlib.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Aleatório")
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
        ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / filename, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva: {self.figures_dir / filename}")

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Curva de Calibração",
        save: bool = True,
        filename: str = "calibration_curve.png",
    ) -> plt.Figure:
        """Gera gráfico de calibração probabilística.

        Conforme discutido na seção 'SLIs e Data SLOs em ML' do
        DOCUMENTO_AULA_6.md, a calibração mede se as probabilidades
        previstas refletem fielmente a realidade.

        Ovadia et al. (2019) mostraram que modelos ficam 'super ou
        sub-confiantes quando expostos a dados cuja distribuição
        diverge daquela original'.

        Args:
            y_true: Rótulos verdadeiros.
            y_proba: Probabilidades da classe positiva.
            n_bins: Número de bins para calibração.
            title: Título do gráfico.
            save: Se True, salva a figura.
            filename: Nome do arquivo.

        Returns:
            Objeto Figure do matplotlib.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )

        brier = brier_score_loss(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            mean_predicted_value, fraction_of_positives,
            marker="o", color="blue", lw=2,
            label=f"Modelo (Brier={brier:.4f})",
        )
        ax.plot(
            [0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Perfeitamente calibrado",
        )
        ax.set_xlabel("Probabilidade prevista média")
        ax.set_ylabel("Fração de positivos observada")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / filename, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva: {self.figures_dir / filename}")

        return fig

    def plot_metrics_comparison(
        self,
        metrics_ref: Dict[str, float],
        metrics_prod: Dict[str, float],
        title: str = "Comparação: Referência vs Produção",
        save: bool = True,
        filename: str = "metrics_comparison.png",
    ) -> plt.Figure:
        """Gera gráfico comparativo de métricas referência vs produção.

        Visualiza a degradação de performance conforme discutido na
        seção 'Por Que Monitorar Modelos de ML' do DOCUMENTO_AULA_6.md.

        Args:
            metrics_ref: Métricas nos dados de referência.
            metrics_prod: Métricas nos dados de produção.
            title: Título do gráfico.
            save: Se True, salva a figura.
            filename: Nome do arquivo.

        Returns:
            Objeto Figure do matplotlib.
        """
        common_keys = [k for k in metrics_ref if k in metrics_prod]
        ref_vals = [metrics_ref[k] for k in common_keys]
        prod_vals = [metrics_prod[k] for k in common_keys]

        x = np.arange(len(common_keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, ref_vals, width, label="Referência", color="steelblue")
        bars2 = ax.bar(x + width / 2, prod_vals, width, label="Produção", color="coral")

        ax.set_xlabel("Métrica")
        ax.set_ylabel("Valor")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(common_keys, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Adicionar valores nas barras
        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_dir / filename, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva: {self.figures_dir / filename}")

        return fig

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Gera relatório de classificação textual.

        Args:
            y_true: Rótulos verdadeiros.
            y_pred: Previsões do modelo.
            target_names: Nomes das classes.

        Returns:
            String com relatório formatado.
        """
        if target_names is None:
            target_names = ["Adimplente", "Inadimplente"]

        report = classification_report(
            y_true, y_pred, target_names=target_names
        )
        return report


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Calcula o Brier Score para calibração probabilística.

    Implementa a fórmula do Brier Score conforme apresentada na seção
    'SLIs e Data SLOs em ML' do DOCUMENTO_AULA_6.md:

    BS = (1/N) Σ (p_i - y_i)²

    onde p_i é a probabilidade prevista e y_i é o resultado observado.

    Conforme Ovadia et al. (2019), a calibração dos modelos costuma
    se deteriorar sob dataset shift.

    Referência:
        Ovadia, Y. et al. (2019). Can You Trust Your Model's Uncertainty?
        Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS.

    Args:
        y_true: Rótulos verdadeiros (0 ou 1).
        y_proba: Probabilidades previstas para a classe positiva.

    Returns:
        Valor do Brier Score (quanto menor, melhor calibração).
    """
    return float(brier_score_loss(y_true, y_proba))


# ======================================================================
# Execução direta: avalia modelo salvo
# ======================================================================

if __name__ == "__main__":
    from src.data_preprocessing import DataPreprocessor
    from src.utils import load_model, save_metrics, setup_logging

    setup_logging()

    # Carregar dados
    preprocessor = DataPreprocessor(random_state=42)
    data = preprocessor.load_data()
    df_ref, df_prod = preprocessor.split_reference_production(data)

    df_ref_clean = preprocessor.clean_data(df_ref)
    df_prod_clean = preprocessor.clean_data(df_prod)

    X_ref, y_ref = preprocessor.prepare_features(df_ref_clean)
    X_prod, y_prod = preprocessor.prepare_features(df_prod_clean)

    # Carregar modelo
    model_path = str(
        Path(__file__).resolve().parent.parent / "outputs" / "models" / "credit_model.joblib"
    )
    model = load_model(model_path)

    # Avaliar
    evaluator = ModelEvaluator()

    y_pred_ref = model.predict(X_ref)
    y_proba_ref = model.predict_proba(X_ref)[:, 1]
    metrics_ref = evaluator.calculate_metrics(y_ref.values, y_pred_ref, y_proba_ref)

    y_pred_prod = model.predict(X_prod)
    y_proba_prod = model.predict_proba(X_prod)[:, 1]
    metrics_prod = evaluator.calculate_metrics(y_prod.values, y_pred_prod, y_proba_prod)

    # Comparar
    comparison = evaluator.compare_metrics(metrics_ref, metrics_prod)
    print("\n=== Comparação Referência vs Produção ===")
    for metric, vals in comparison.items():
        print(f"  {metric}: ref={vals['reference']:.4f} prod={vals['production']:.4f} Δ={vals['delta']:+.4f}")

    # Gráficos
    evaluator.plot_confusion_matrix(y_prod.values, y_pred_prod, title="Confusão — Produção")
    evaluator.plot_roc_curve(y_prod.values, y_proba_prod, title="ROC — Produção")
    evaluator.plot_calibration_curve(y_prod.values, y_proba_prod, title="Calibração — Produção")
    evaluator.plot_metrics_comparison(metrics_ref, metrics_prod)

    # Salvar métricas
    save_metrics({"reference": metrics_ref, "production": metrics_prod, "comparison": comparison})

    print("\nAvaliação concluída! Gráficos em outputs/figures/")

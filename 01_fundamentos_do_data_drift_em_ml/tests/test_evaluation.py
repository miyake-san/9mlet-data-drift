"""
Testes unitários para o módulo evaluation.

Valida cálculo de métricas, geração de gráficos e funções de
comparação de distribuições.
"""

import numpy as np
import pytest

from src.evaluation import (
    calculate_metrics,
    calculate_metrics_per_period,
    plot_confusion_matrix,
    plot_distribution_comparison,
    plot_roc_curve,
)

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para testes


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def binary_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixture: y_true, y_pred e y_proba para classificação binária."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_pred = y_true.copy()
    # Introduz 20% de erros
    flip_idx = rng.choice(200, 40, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_proba = rng.uniform(0.1, 0.9, 200)
    # Ajusta probas para serem consistentes (parcialmente)
    y_proba[y_true == 1] += 0.2
    y_proba = np.clip(y_proba, 0, 1)
    return y_true, y_pred, y_proba


# ---------------------------------------------------------------
# Testes de cálculo de métricas
# ---------------------------------------------------------------

class TestCalculateMetrics:
    """Testes da função calculate_metrics."""

    def test_metricas_basicas(self, binary_predictions: tuple) -> None:
        """Verifica que métricas retornam valores no range [0, 1]."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        for key in ["accuracy", "precision", "recall", "f1", "auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_acuracia_perfeita(self) -> None:
        """Acurácia perfeita quando y_true == y_pred."""
        y = np.array([0, 1, 1, 0, 1])
        metrics = calculate_metrics(y, y)
        assert metrics["accuracy"] == 1.0

    def test_sem_proba_nao_tem_auc(self) -> None:
        """Sem y_proba, AUC não deve aparecer no resultado."""
        y = np.array([0, 1, 1, 0])
        metrics = calculate_metrics(y, y)
        assert "auc" not in metrics

    def test_metrics_per_period(self, binary_predictions: tuple) -> None:
        """Verifica cálculo de métricas por período."""
        y_true, y_pred, _ = binary_predictions
        periods = np.repeat([1, 2, 3, 4], 50)
        result = calculate_metrics_per_period(y_true, y_pred, periods)
        assert len(result) == 4
        for p in [1, 2, 3, 4]:
            assert "accuracy" in result[p]


# ---------------------------------------------------------------
# Testes de visualização
# ---------------------------------------------------------------

class TestPlots:
    """Testes de geração de gráficos (não interativos)."""

    def test_plot_confusion_matrix_retorna_figure(self, binary_predictions: tuple) -> None:
        """Verifica que plot_confusion_matrix retorna Figure."""
        y_true, y_pred, _ = binary_predictions
        import matplotlib.pyplot as plt
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        plt.close(fig)

    def test_plot_roc_curve_retorna_figure(self, binary_predictions: tuple) -> None:
        """Verifica que plot_roc_curve retorna Figure."""
        y_true, _, y_proba = binary_predictions
        import matplotlib.pyplot as plt
        fig = plot_roc_curve(y_true, y_proba)
        assert fig is not None
        plt.close(fig)

    def test_plot_distribution_comparison_retorna_figure(self) -> None:
        """Verifica que plot_distribution_comparison retorna Figure."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        fig = plot_distribution_comparison(ref, cur, "idade")
        assert fig is not None
        plt.close(fig)

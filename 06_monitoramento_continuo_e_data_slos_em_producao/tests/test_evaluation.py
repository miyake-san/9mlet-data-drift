"""
Testes unitários para o módulo evaluation.py.

Testa cálculo de métricas, Brier Score, comparação de métricas
e geração de relatórios de classificação.
"""

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import ModelEvaluator, calculate_brier_score


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def evaluator(tmp_path: Path) -> ModelEvaluator:
    """Cria ModelEvaluator com diretório temporário."""
    return ModelEvaluator(figures_dir=str(tmp_path / "figures"))


@pytest.fixture
def binary_predictions() -> tuple:
    """Gera previsões binárias para testes."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.choice([0, 1], n, p=[0.7, 0.3])
    # Previsões razoavelmente boas
    y_pred = y_true.copy()
    # Introduzir ~15% de erros
    error_mask = rng.random(n) < 0.15
    y_pred[error_mask] = 1 - y_pred[error_mask]
    # Probabilidades baseadas no true com ruído
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, n), 0, 1)
    return y_true, y_pred, y_proba


# ======================================================================
# Testes
# ======================================================================


class TestCalculateMetrics:
    """Testes para calculate_metrics."""

    def test_metrics_keys(
        self, evaluator: ModelEvaluator, binary_predictions: tuple
    ) -> None:
        """Verifica que todas as métricas esperadas são retornadas."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
        expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc", "brier_score"}
        assert expected_keys == set(metrics.keys())

    def test_metrics_values_range(
        self, evaluator: ModelEvaluator, binary_predictions: tuple
    ) -> None:
        """Verifica que métricas estão no intervalo [0, 1]."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} fora do intervalo: {value}"

    def test_metrics_without_proba(
        self, evaluator: ModelEvaluator, binary_predictions: tuple
    ) -> None:
        """Verifica métricas sem probabilidades (sem AUC/Brier)."""
        y_true, y_pred, _ = binary_predictions
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        assert "roc_auc" not in metrics
        assert "brier_score" not in metrics
        assert "accuracy" in metrics

    def test_perfect_predictions(self, evaluator: ModelEvaluator) -> None:
        """Verifica métricas com previsões perfeitas."""
        y = np.array([0, 0, 1, 1, 0, 1])
        metrics = evaluator.calculate_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0


class TestBrierScore:
    """Testes para calculate_brier_score."""

    def test_brier_perfect_calibration(self) -> None:
        """Verifica Brier Score perfeito (0.0)."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        brier = calculate_brier_score(y_true, y_proba)
        assert brier == pytest.approx(0.0, abs=1e-6)

    def test_brier_worst_calibration(self) -> None:
        """Verifica Brier Score pior caso (1.0)."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([1.0, 1.0, 0.0, 0.0])
        brier = calculate_brier_score(y_true, y_proba)
        assert brier == pytest.approx(1.0, abs=1e-6)

    def test_brier_range(self, binary_predictions: tuple) -> None:
        """Verifica que Brier Score está no intervalo [0, 1]."""
        y_true, _, y_proba = binary_predictions
        brier = calculate_brier_score(y_true, y_proba)
        assert 0 <= brier <= 1


class TestCompareMetrics:
    """Testes para compare_metrics."""

    def test_compare_returns_delta(self, evaluator: ModelEvaluator) -> None:
        """Verifica comparação com deltas corretos."""
        metrics_ref = {"accuracy": 0.90, "f1_score": 0.85}
        metrics_prod = {"accuracy": 0.80, "f1_score": 0.75}
        comparison = evaluator.compare_metrics(metrics_ref, metrics_prod)
        assert comparison["accuracy"]["delta"] == pytest.approx(-0.10, abs=1e-6)
        assert comparison["f1_score"]["delta"] == pytest.approx(-0.10, abs=1e-6)


class TestPlots:
    """Testes para geração de gráficos."""

    def test_plot_confusion_matrix_saves(
        self, evaluator: ModelEvaluator, binary_predictions: tuple
    ) -> None:
        """Verifica que confusion matrix salva arquivo."""
        import matplotlib
        matplotlib.use("Agg")
        y_true, y_pred, _ = binary_predictions
        fig = evaluator.plot_confusion_matrix(
            y_true, y_pred, save=True, filename="test_cm.png"
        )
        assert (Path(evaluator.figures_dir) / "test_cm.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_roc_curve_saves(
        self, evaluator: ModelEvaluator, binary_predictions: tuple
    ) -> None:
        """Verifica que ROC curve salva arquivo."""
        import matplotlib
        matplotlib.use("Agg")
        y_true, _, y_proba = binary_predictions
        fig = evaluator.plot_roc_curve(
            y_true, y_proba, save=True, filename="test_roc.png"
        )
        assert (Path(evaluator.figures_dir) / "test_roc.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

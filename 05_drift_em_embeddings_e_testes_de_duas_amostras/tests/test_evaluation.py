"""
Testes unitários para o módulo evaluation.

Valida as funções de métricas de drift, geração de relatórios e funções auxiliares.
"""

import numpy as np
import pytest

from src.evaluation import (
    calculate_drift_metrics,
    generate_drift_report,
    _mean_cosine_similarity,
)
from src.utils import set_seed


@pytest.fixture(autouse=True)
def fix_seed() -> None:
    """Garante reproduzibilidade."""
    set_seed(42)


@pytest.fixture
def identical_embeddings() -> tuple:
    """Embeddings da mesma distribuição (sem drift)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 16)
    Y = rng.randn(200, 16)
    return X, Y


@pytest.fixture
def drifted_embeddings() -> tuple:
    """Embeddings de distribuições distintas (drift claro)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 16)
    Y = rng.randn(200, 16) + 3.0
    return X, Y


class TestCalculateDriftMetrics:
    """Testes para calculate_drift_metrics()."""

    def test_returns_expected_keys(
        self, identical_embeddings: tuple
    ) -> None:
        """Testa que o resultado contém chaves dos métodos solicitados."""
        X, Y = identical_embeddings
        result = calculate_drift_metrics(X, Y, methods=["mmd", "ks"])
        assert "mmd" in result
        assert "ks" in result
        assert "embedding_stats" in result

    def test_no_drift_detected_identical(
        self, identical_embeddings: tuple
    ) -> None:
        """Sem drift esperado para distribuições idênticas (MMD)."""
        X, Y = identical_embeddings
        result = calculate_drift_metrics(
            X, Y, methods=["mmd"], gamma=1.0, n_permutations=50,
        )
        # P-value deve ser alto (sem drift)
        assert result["mmd"]["p_value"] > 0.01

    def test_drift_detected_different(
        self, drifted_embeddings: tuple
    ) -> None:
        """Drift deve ser detectado para distribuições deslocadas."""
        X, Y = drifted_embeddings
        result = calculate_drift_metrics(
            X, Y, methods=["mmd", "ks"], gamma=0.1, n_permutations=50,
        )
        assert result["mmd"]["drift_detected"] is True
        assert result["ks"]["drift_detected"] is True

    def test_embedding_stats_present(
        self, identical_embeddings: tuple
    ) -> None:
        """Testa que estatísticas de embeddings são calculadas."""
        X, Y = identical_embeddings
        result = calculate_drift_metrics(X, Y, methods=["mmd"])
        stats = result["embedding_stats"]
        assert "ref_mean_norm" in stats
        assert "prod_mean_norm" in stats
        assert "cosine_similarity_mean" in stats


class TestGenerateDriftReport:
    """Testes para generate_drift_report()."""

    def test_report_is_nonempty_string(
        self, drifted_embeddings: tuple
    ) -> None:
        """Testa que o relatório é uma string não vazia."""
        X, Y = drifted_embeddings
        metrics = calculate_drift_metrics(
            X, Y, methods=["mmd"], gamma=0.1, n_permutations=20,
        )
        report = generate_drift_report(metrics)
        assert isinstance(report, str)
        assert len(report) > 50

    def test_report_contains_method_names(
        self, drifted_embeddings: tuple
    ) -> None:
        """Testa que o relatório menciona os métodos usados."""
        X, Y = drifted_embeddings
        metrics = calculate_drift_metrics(
            X, Y, methods=["mmd", "ks"], gamma=0.1, n_permutations=20,
        )
        report = generate_drift_report(metrics)
        assert "MMD" in report
        assert "KS" in report

    def test_report_save_to_file(
        self, drifted_embeddings: tuple, tmp_path
    ) -> None:
        """Testa salvamento do relatório em arquivo."""
        X, Y = drifted_embeddings
        metrics = calculate_drift_metrics(
            X, Y, methods=["mmd"], gamma=0.1, n_permutations=20,
        )
        out = str(tmp_path / "report.txt")
        generate_drift_report(metrics, output_path=out)
        assert (tmp_path / "report.txt").exists()


class TestCosineSimilarity:
    """Testes para _mean_cosine_similarity()."""

    def test_identical_embeddings_high_similarity(self) -> None:
        """Similaridade cosseno alta para distribuições iguais."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        sim = _mean_cosine_similarity(X, X)
        assert sim > 0.99

    def test_different_embeddings_lower_similarity(self) -> None:
        """Similaridade menor para distribuições deslocadas."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        Y = rng.randn(100, 8) + 5.0
        sim = _mean_cosine_similarity(X, Y)
        assert sim < 1.0

"""
Testes unitários para o módulo model.

Valida o EmbeddingDriftDetector: inicialização, MMD, KS, classificador adversário,
fit/predict workflow. Usa dados sintéticos com seed fixa para reproduzibilidade.
"""

import numpy as np
import pytest

from src.model import EmbeddingDriftDetector
from src.utils import set_seed


@pytest.fixture(autouse=True)
def fix_seed() -> None:
    """Garante reproduzibilidade em todos os testes."""
    set_seed(42)


@pytest.fixture
def identical_distributions() -> tuple:
    """Fixture com dois conjuntos vindos da mesma distribuição."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 16)
    Y = rng.randn(200, 16)
    return X, Y


@pytest.fixture
def different_distributions() -> tuple:
    """Fixture com dois conjuntos de distribuições diferentes (drift claro)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 16)
    Y = rng.randn(200, 16) + 3.0  # Deslocamento grande
    return X, Y


class TestInit:
    """Testes de inicialização do EmbeddingDriftDetector."""

    def test_valid_methods(self) -> None:
        """Testa inicialização com métodos válidos."""
        for method in ("mmd", "ks", "adversarial"):
            det = EmbeddingDriftDetector(method=method)
            assert det.method == method

    def test_invalid_method_raises(self) -> None:
        """Testa erro com método inválido."""
        with pytest.raises(ValueError, match="inválido"):
            EmbeddingDriftDetector(method="invalido")

    def test_default_parameters(self) -> None:
        """Testa parâmetros default."""
        det = EmbeddingDriftDetector()
        assert det.method == "mmd"
        assert det.gamma == 1.0
        assert det.n_permutations == 100
        assert det.is_fitted is False


class TestComputeMMD:
    """Testes para compute_mmd()."""

    def test_mmd_identical_distributions_near_zero(
        self, identical_distributions: tuple
    ) -> None:
        """MMD entre distribuições idênticas deve ser próximo de zero."""
        X, Y = identical_distributions
        det = EmbeddingDriftDetector(method="mmd", gamma=1.0)
        mmd = det.compute_mmd(X, Y)
        # Para amostras da mesma distribuição, MMD deve ser pequeno
        assert mmd >= 0
        assert mmd < 0.5

    def test_mmd_different_distributions_large(
        self, different_distributions: tuple
    ) -> None:
        """MMD entre distribuições diferentes deve ser positivo e significativo."""
        X, Y = different_distributions
        det = EmbeddingDriftDetector(method="mmd", gamma=0.1)
        mmd = det.compute_mmd(X, Y)
        assert mmd > 0.1

    def test_mmd_self_is_zero(self) -> None:
        """MMD de um conjunto consigo mesmo deve ser zero."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        det = EmbeddingDriftDetector(method="mmd")
        mmd = det.compute_mmd(X, X)
        assert np.isclose(mmd, 0.0, atol=1e-10)

    def test_mmd_custom_gamma(self, different_distributions: tuple) -> None:
        """Testa compute_mmd com gamma personalizado."""
        X, Y = different_distributions
        det = EmbeddingDriftDetector(method="mmd")
        mmd_low = det.compute_mmd(X, Y, gamma=0.001)
        mmd_high = det.compute_mmd(X, Y, gamma=10.0)
        # Ambos devem ser não-negativos
        assert mmd_low >= 0
        assert mmd_high >= 0


class TestComputeKS:
    """Testes para compute_ks_test()."""

    def test_ks_identical_high_pvalue(
        self, identical_distributions: tuple
    ) -> None:
        """KS entre distribuições iguais deve ter p-value alto."""
        X, Y = identical_distributions
        det = EmbeddingDriftDetector(method="ks")
        result = det.compute_ks_test(X, Y)
        assert result["adjusted_p_value"] > 0.01

    def test_ks_different_low_pvalue(
        self, different_distributions: tuple
    ) -> None:
        """KS entre distribuições diferentes deve ter p-value baixo."""
        X, Y = different_distributions
        det = EmbeddingDriftDetector(method="ks")
        result = det.compute_ks_test(X, Y)
        assert result["adjusted_p_value"] < 0.05

    def test_ks_returns_correct_keys(
        self, identical_distributions: tuple
    ) -> None:
        """Testa que o resultado contém todas as chaves esperadas."""
        X, Y = identical_distributions
        det = EmbeddingDriftDetector(method="ks")
        result = det.compute_ks_test(X, Y)
        assert "d_statistics" in result
        assert "p_values" in result
        assert "adjusted_p_value" in result
        assert "max_d" in result
        assert len(result["d_statistics"]) == 16


class TestFitPredict:
    """Testes para o workflow fit() → predict()."""

    def test_predict_without_fit_raises(self) -> None:
        """Testa erro ao chamar predict sem fit."""
        det = EmbeddingDriftDetector(method="mmd")
        X = np.random.randn(50, 8)
        with pytest.raises(RuntimeError, match="não ajustado"):
            det.predict(X)

    def test_fit_predict_mmd_no_drift(
        self, identical_distributions: tuple
    ) -> None:
        """Testa fit/predict com MMD em dados sem drift."""
        X, Y = identical_distributions
        det = EmbeddingDriftDetector(method="mmd", gamma=1.0, n_permutations=50)
        det.fit(X)
        result = det.predict(Y)
        assert "score" in result
        assert "p_value" in result
        assert "drift_detected" in result
        assert isinstance(result["drift_detected"], bool)

    def test_fit_predict_mmd_with_drift(
        self, different_distributions: tuple
    ) -> None:
        """Testa fit/predict com MMD em dados com drift."""
        X, Y = different_distributions
        det = EmbeddingDriftDetector(method="mmd", gamma=0.1, n_permutations=50)
        det.fit(X)
        result = det.predict(Y)
        assert result["drift_detected"] is True

    def test_score_returns_float(
        self, identical_distributions: tuple
    ) -> None:
        """Testa que score() retorna float."""
        X, Y = identical_distributions
        det = EmbeddingDriftDetector(method="mmd", n_permutations=20)
        det.fit(X)
        score = det.score(Y)
        assert isinstance(score, float)

    def test_fit_predict_adversarial(
        self, different_distributions: tuple
    ) -> None:
        """Testa fit/predict com classificador adversário."""
        X, Y = different_distributions
        det = EmbeddingDriftDetector(method="adversarial")
        det.fit(X)
        result = det.predict(Y)
        assert "accuracy" in result
        assert result["drift_detected"] is True

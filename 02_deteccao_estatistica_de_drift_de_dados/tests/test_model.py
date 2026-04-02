"""
Testes unitários para o módulo model.

Valida DriftDetector (KS, PSI, KL, JS, Chi2) e data classes
DriftResult e DriftReport, garantindo que as implementações são
consistentes com os conceitos do Documento da Aula 2.
"""

import numpy as np
import pytest

from src.model import DriftDetector, DriftReport, DriftResult


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector(alpha=0.05, psi_threshold=0.25, n_bins=10)


@pytest.fixture
def identical_samples(rng: np.random.RandomState):
    """Duas amostras da mesma distribuição (sem drift)."""
    a = rng.normal(0, 1, size=1000)
    b = rng.normal(0, 1, size=1000)
    return a, b


@pytest.fixture
def drifted_samples(rng: np.random.RandomState):
    """Duas amostras de distribuições diferentes (com drift)."""
    ref = rng.normal(40, 10, size=1000)
    cur = rng.normal(30, 8, size=1000)
    return ref, cur


# ---------------------------------------------------------------
# Tests: Inicialização
# ---------------------------------------------------------------

class TestDriftDetectorInit:
    """Testes de inicialização do DriftDetector."""

    def test_default_init(self) -> None:
        """Verifica inicialização com valores default."""
        d = DriftDetector()
        assert d.alpha == 0.05
        assert d.psi_threshold == 0.25
        assert d.n_bins == 10

    def test_custom_init(self) -> None:
        """Verifica inicialização com valores customizados."""
        d = DriftDetector(alpha=0.01, psi_threshold=0.10, n_bins=20)
        assert d.alpha == 0.01
        assert d.psi_threshold == 0.10
        assert d.n_bins == 20

    def test_invalid_alpha_raises(self) -> None:
        """Alpha fora de (0, 1) deve levantar ValueError."""
        with pytest.raises(ValueError):
            DriftDetector(alpha=0.0)
        with pytest.raises(ValueError):
            DriftDetector(alpha=1.0)

    def test_invalid_psi_threshold_raises(self) -> None:
        """psi_threshold <= 0 deve levantar ValueError."""
        with pytest.raises(ValueError):
            DriftDetector(psi_threshold=0.0)

    def test_invalid_n_bins_raises(self) -> None:
        """n_bins < 2 deve levantar ValueError."""
        with pytest.raises(ValueError):
            DriftDetector(n_bins=1)


# ---------------------------------------------------------------
# Tests: Teste KS
# ---------------------------------------------------------------

class TestKSTest:
    """Testes do teste Kolmogorov-Smirnov."""

    def test_ks_no_drift(self, detector: DriftDetector, identical_samples) -> None:
        """KS não deve detectar drift em amostras idênticas."""
        a, b = identical_samples
        result = detector.ks_test(a, b, feature_name="test_feature")
        assert isinstance(result, DriftResult)
        assert result.method == "ks"
        # p-valor deve ser alto (sem drift)
        assert result.p_value > 0.05

    def test_ks_with_drift(self, detector: DriftDetector, drifted_samples) -> None:
        """KS deve detectar drift em amostras com distribuições diferentes."""
        ref, cur = drifted_samples
        result = detector.ks_test(ref, cur, feature_name="idade")
        assert result.drift_detected is True
        assert result.p_value < 0.05
        assert result.statistic > 0

    def test_ks_returns_correct_fields(self, detector: DriftDetector, drifted_samples) -> None:
        """Verifica que DriftResult tem todos os campos corretos."""
        ref, cur = drifted_samples
        result = detector.ks_test(ref, cur, feature_name="my_feature")
        assert result.feature_name == "my_feature"
        assert result.method == "ks"
        assert 0 <= result.statistic <= 1
        assert 0 <= result.p_value <= 1
        assert result.threshold == 0.05


# ---------------------------------------------------------------
# Tests: PSI
# ---------------------------------------------------------------

class TestPSI:
    """Testes do Population Stability Index."""

    def test_psi_no_drift(self, detector: DriftDetector, identical_samples) -> None:
        """PSI deve ser baixo para amostras semelhantes."""
        a, b = identical_samples
        result = detector.psi(a, b, feature_name="test")
        assert result.method == "psi"
        assert result.statistic < 0.10  # estável
        assert result.drift_detected is False

    def test_psi_with_drift(self, detector: DriftDetector, drifted_samples) -> None:
        """PSI deve ser alto para amostras com drift."""
        ref, cur = drifted_samples
        result = detector.psi(ref, cur, feature_name="idade")
        assert result.statistic > 0.10  # pelo menos moderado
        assert result.p_value is None  # PSI não tem p-valor

    def test_psi_interpret_stable(self) -> None:
        """Verifica interpretação PSI < 0.10."""
        assert "estável" in DriftDetector.interpret_psi(0.05)

    def test_psi_interpret_moderate(self) -> None:
        """Verifica interpretação 0.10 ≤ PSI < 0.25."""
        assert "moderado" in DriftDetector.interpret_psi(0.15)

    def test_psi_interpret_severe(self) -> None:
        """Verifica interpretação PSI ≥ 0.25."""
        assert "severo" in DriftDetector.interpret_psi(0.30)


# ---------------------------------------------------------------
# Tests: KL e JS
# ---------------------------------------------------------------

class TestDivergences:
    """Testes das divergências KL e JS."""

    def test_kl_non_negative(self, detector: DriftDetector, drifted_samples) -> None:
        """KL divergence deve ser sempre ≥ 0."""
        ref, cur = drifted_samples
        result = detector.kl_divergence(ref, cur)
        assert result.statistic >= 0

    def test_kl_asymmetric(self, detector: DriftDetector, drifted_samples) -> None:
        """KL deve ser assimétrica: KL(P||Q) ≠ KL(Q||P)."""
        ref, cur = drifted_samples
        kl_pq = detector.kl_divergence(ref, cur)
        kl_qp = detector.kl_divergence(cur, ref)
        # Com drift real, devem ser diferentes
        assert abs(kl_pq.statistic - kl_qp.statistic) > 0.001

    def test_js_symmetric(self, detector: DriftDetector, drifted_samples) -> None:
        """JS deve ser simétrica: JS(P||Q) ≈ JS(Q||P)."""
        ref, cur = drifted_samples
        js_pq = detector.js_divergence(ref, cur)
        js_qp = detector.js_divergence(cur, ref)
        assert abs(js_pq.statistic - js_qp.statistic) < 0.01

    def test_js_bounded(self, detector: DriftDetector, drifted_samples) -> None:
        """JS deve estar entre 0 e 1 (com log base 2)."""
        ref, cur = drifted_samples
        result = detector.js_divergence(ref, cur)
        assert 0 <= result.statistic <= 1


# ---------------------------------------------------------------
# Tests: Chi-quadrado
# ---------------------------------------------------------------

class TestChi2:
    """Testes do teste Qui-quadrado para categóricas."""

    def test_chi2_no_drift(self, detector: DriftDetector) -> None:
        """Chi2 não deve detectar drift em distribuições categóricas iguais."""
        rng = np.random.RandomState(42)
        cats = np.array(["A", "B", "C"])
        probs = [0.5, 0.3, 0.2]
        ref = rng.choice(cats, size=1000, p=probs)
        cur = rng.choice(cats, size=1000, p=probs)
        result = detector.chi2_test(ref, cur, feature_name="cat_feature")
        assert result.method == "chi2"
        # Provavelmente não detecta drift
        assert result.p_value > 0.01

    def test_chi2_with_drift(self, detector: DriftDetector) -> None:
        """Chi2 deve detectar drift em distribuições categóricas diferentes."""
        rng = np.random.RandomState(42)
        cats = np.array(["A", "B", "C"])
        ref = rng.choice(cats, size=1000, p=[0.5, 0.3, 0.2])
        cur = rng.choice(cats, size=1000, p=[0.1, 0.2, 0.7])
        result = detector.chi2_test(ref, cur, feature_name="cat_feature")
        assert result.drift_detected is True
        assert result.p_value < 0.05


# ---------------------------------------------------------------
# Tests: detect_all
# ---------------------------------------------------------------

class TestDetectAll:
    """Testes da detecção completa multi-feature."""

    def test_detect_all_returns_report(self, detector: DriftDetector) -> None:
        """detect_all deve retornar um DriftReport."""
        import pandas as pd

        rng = np.random.RandomState(42)
        ref = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
            "cat": rng.choice(["A", "B"], 500),
        })
        cur = pd.DataFrame({
            "f1": rng.normal(2, 1, 500),
            "f2": rng.normal(0, 1, 500),
            "cat": rng.choice(["A", "B"], 500, p=[0.8, 0.2]),
        })

        report = detector.detect_all(ref, cur, methods=["ks", "psi"])
        assert isinstance(report, DriftReport)
        assert report.n_features_total > 0
        assert len(report.results) > 0

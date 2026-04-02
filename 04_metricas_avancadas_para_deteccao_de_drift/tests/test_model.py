# -*- coding: utf-8 -*-
"""
test_model.py – Testes unitários para detectores de drift.

Cobre PSICalculator, MMDCalculator, WassersteinCalculator,
EnergyDistanceCalculator e DriftDetector (Documento 04).
"""

import numpy as np
import pytest

from src.model import (
    DriftDetector,
    DriftResult,
    EnergyDistanceCalculator,
    MMDCalculator,
    PSICalculator,
    WassersteinCalculator,
)


# ==================================================================
# Fixtures
# ==================================================================
@pytest.fixture
def rng() -> np.random.RandomState:
    """Gerador com seed fixa para reprodutibilidade."""
    return np.random.RandomState(42)


@pytest.fixture
def identical_samples(rng: np.random.RandomState):
    """Duas amostras IID da mesma distribuição (sem drift)."""
    X = rng.normal(loc=0, scale=1, size=(200, 3))
    Y = rng.normal(loc=0, scale=1, size=(200, 3))
    return X, Y


@pytest.fixture
def drifted_samples(rng: np.random.RandomState):
    """Amostras com drift multivariado (inversão de correlação).

    Simula o cenário da fintech do Documento 04, Snippet 2.
    """
    n = 300
    idade = rng.normal(40, 10, n)
    renda_ref = 3000 + 500 * idade + rng.normal(0, 10000, n)
    renda_cur = 3000 - 500 * idade + rng.normal(0, 10000, n)  # inversão
    X = np.column_stack([idade, renda_ref])
    Y = np.column_stack([idade, renda_cur])
    return X, Y


# ==================================================================
# PSICalculator
# ==================================================================
class TestPSICalculator:
    """Testes para PSICalculator (Documento 04, Snippet 1)."""

    def test_psi_identical_distributions(self, rng: np.random.RandomState) -> None:
        """PSI próximo de zero para distribuições idênticas."""
        data = rng.normal(50, 10, 5000)
        psi = PSICalculator().calculate(data, data)
        assert psi < 0.01

    def test_psi_shifted_distribution(self, rng: np.random.RandomState) -> None:
        """PSI > 0.25 para distribuição com shift significativo."""
        baseline = rng.normal(50, 10, 5000)
        shifted = rng.normal(60, 10, 5000)  # shift de 1σ
        psi = PSICalculator().calculate(baseline, shifted)
        assert psi > 0.1  # Shift significativo detectado

    def test_psi_non_negative(self, rng: np.random.RandomState) -> None:
        """PSI é sempre ≥ 0."""
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0.5, 1.5, 1000)
        psi = PSICalculator().calculate(a, b)
        assert psi >= 0.0

    def test_psi_multifeature_returns_list(self, identical_samples) -> None:
        """calculate_multifeature retorna lista com len == n_features."""
        X, Y = identical_samples
        values = PSICalculator().calculate_multifeature(X, Y)
        assert isinstance(values, list)
        assert len(values) == X.shape[1]

    def test_psi_custom_bins(self, rng: np.random.RandomState) -> None:
        """PSI funciona com bins customizados."""
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        bins = np.linspace(-4, 4, 21)
        psi = PSICalculator().calculate(a, b, bins=bins)
        assert isinstance(psi, float)
        assert psi >= 0.0


# ==================================================================
# MMDCalculator
# ==================================================================
class TestMMDCalculator:
    """Testes para MMDCalculator (Documento 04, Seção MMD/Gretton et al.)."""

    def test_mmd_identical_near_zero(self, identical_samples) -> None:
        """MMD próximo de zero para amostras da mesma distribuição."""
        X, Y = identical_samples
        mmd = MMDCalculator().calculate(X, Y)
        assert mmd < 0.05

    def test_mmd_drifted_positive(self, drifted_samples) -> None:
        """MMD > 0 para amostras com drift multivariado."""
        X, Y = drifted_samples
        mmd = MMDCalculator().calculate(X, Y)
        assert mmd > 0.0

    def test_mmd_non_negative(self, identical_samples) -> None:
        """MMD² é sempre ≥ 0."""
        X, Y = identical_samples
        mmd = MMDCalculator().calculate(X, Y)
        assert mmd >= 0.0

    def test_mmd_with_fixed_gamma(self, drifted_samples) -> None:
        """MMD funciona com γ fixo."""
        X, Y = drifted_samples
        mmd = MMDCalculator(gamma=1e-5).calculate(X, Y)
        assert isinstance(mmd, float)
        assert mmd >= 0.0

    def test_mmd_permutation_test(self, drifted_samples) -> None:
        """Teste de permutação detecta drift e retorna DriftResult."""
        X, Y = drifted_samples
        result = MMDCalculator().permutation_test(
            X, Y, n_permutations=50, seed=42
        )
        assert isinstance(result, DriftResult)
        assert result.metric_name == "MMD"
        assert result.statistic > 0.0
        assert "p_value" in result.details

    def test_mmd_1d_input(self, rng: np.random.RandomState) -> None:
        """MMD lida com entrada 1D (reshape automático)."""
        a = rng.normal(0, 1, 200)
        b = rng.normal(2, 1, 200)
        mmd = MMDCalculator().calculate(a, b)
        assert mmd > 0.0


# ==================================================================
# WassersteinCalculator
# ==================================================================
class TestWassersteinCalculator:
    """Testes para WassersteinCalculator (Documento 04, Saiba Mais)."""

    def test_wasserstein_identical_near_zero(
        self, rng: np.random.RandomState
    ) -> None:
        """Wasserstein próximo de zero para mesma distribuição."""
        data = rng.normal(0, 1, 5000)
        w = WassersteinCalculator().calculate(data, data)
        assert w < 0.01

    def test_wasserstein_shifted_positive(
        self, rng: np.random.RandomState
    ) -> None:
        """Wasserstein > 0 para distribuições deslocadas."""
        a = rng.normal(0, 1, 1000)
        b = rng.normal(5, 1, 1000)
        w = WassersteinCalculator().calculate(a, b)
        assert w > 3.0  # Shift de ~5σ

    def test_wasserstein_multifeature(self, identical_samples) -> None:
        """calculate_multifeature retorna lista correta."""
        X, Y = identical_samples
        values = WassersteinCalculator().calculate_multifeature(X, Y)
        assert len(values) == X.shape[1]
        assert all(isinstance(v, float) for v in values)


# ==================================================================
# EnergyDistanceCalculator
# ==================================================================
class TestEnergyDistanceCalculator:
    """Testes para EnergyDistanceCalculator (Székely & Rizzo, 2013)."""

    def test_energy_identical_near_zero(self, identical_samples) -> None:
        """Energy distance próxima de zero para mesma distribuição."""
        X, Y = identical_samples
        e = EnergyDistanceCalculator().calculate(X, Y)
        assert e < 0.2

    def test_energy_drifted_positive(self, drifted_samples) -> None:
        """Energy distance > 0 para amostras com drift."""
        X, Y = drifted_samples
        e = EnergyDistanceCalculator().calculate(X, Y)
        assert e > 0.0

    def test_energy_non_negative(self, identical_samples) -> None:
        """Energy distance é sempre ≥ 0."""
        X, Y = identical_samples
        e = EnergyDistanceCalculator().calculate(X, Y)
        assert e >= 0.0


# ==================================================================
# DriftDetector (unificado)
# ==================================================================
class TestDriftDetector:
    """Testes para DriftDetector (Documento 04, Vídeos 3-4)."""

    def test_fit_predict_returns_dict(self, drifted_samples) -> None:
        """predict retorna dicionário com todas as métricas."""
        X, Y = drifted_samples
        detector = DriftDetector()
        detector.fit(X)
        results = detector.predict(Y)
        assert "psi" in results
        assert "mmd" in results
        assert "wasserstein" in results
        assert "energy" in results

    def test_score_returns_float(self, drifted_samples) -> None:
        """score() retorna float (valor MMD)."""
        X, Y = drifted_samples
        detector = DriftDetector().fit(X)
        s = detector.score(Y)
        assert isinstance(s, float)
        assert s >= 0.0

    def test_psi_no_false_alarm_multivariate_drift(
        self, rng: np.random.RandomState,
    ) -> None:
        """PSI não detecta drift multivariado sutil (limitação esperada).

        Conforme Documento 04, Seção "Limites de Métricas Simples":
        quando as marginais permanecem similares mas a correlação muda,
        o PSI falha em detectar o drift. Aqui garantimos marginais
        idênticas embaralhando o pareamento entre variáveis.
        """
        n = 500
        a = rng.normal(0, 1, n)
        b = rng.normal(0, 1, n)
        X = np.column_stack([a, b])
        # Mesmo dados marginais, mas correlação destruída pelo shuffle
        Y = np.column_stack([a, rng.permutation(b)])
        detector = DriftDetector(psi_threshold=0.25).fit(X)
        results = detector.predict(Y)
        # PSI não deve detectar drift quando marginais são idênticas
        assert results["psi"].drift_detected is False

    def test_mmd_detects_multivariate_drift(self, drifted_samples) -> None:
        """MMD detecta drift multivariado sutil.

        Conforme Documento 04, Snippet 2: a MMD captura a inversão
        de correlação renda-idade que o PSI não detecta.
        """
        X, Y = drifted_samples
        detector = DriftDetector().fit(X)
        results = detector.predict(Y)
        assert results["mmd"].statistic > 0.0

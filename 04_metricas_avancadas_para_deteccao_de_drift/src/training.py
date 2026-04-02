# -*- coding: utf-8 -*-
"""
training.py – Calibração de limiares e busca de hiperparâmetros
para detectores de drift.

Conforme Documento 04, Seção "Saiba Mais": para uso prático dos
detectores de drift, é necessário definir limiares de significância
estatística. Este módulo implementa calibração via permutação e
busca de hiperparâmetros (γ do kernel RBF).

Referências:
    Gretton, A. et al. (2012). A Kernel Two-Sample Test. JMLR, 13, 723–773.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import DriftDetector, MMDCalculator


def train_model(
    X_reference: np.ndarray,
    gamma: Optional[float] = None,
    psi_threshold: float = 0.25,
) -> DriftDetector:
    """Treina (ajusta) o detector de drift com dados de referência.

    Conforme Documento 04, Vídeo 4: o detector é inicializado com
    os dados do período de referência (baseline), que servem como
    distribuição esperada para comparação futura.

    Args:
        X_reference: Dados do período de referência (n_samples, n_features).
        gamma: Parâmetro γ do kernel RBF para MMD. None = auto (mediana).
        psi_threshold: Limiar para PSI. Default: 0.25.

    Returns:
        DriftDetector ajustado.
    """
    detector = DriftDetector(
        psi_threshold=psi_threshold,
        mmd_gamma=gamma,
    )
    detector.fit(X_reference)
    return detector


def cross_validate(
    X_reference: np.ndarray,
    n_folds: int = 5,
    n_permutations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validação cruzada para estimar limiar de MMD sob H0 (sem drift).

    Divide os dados de referência em folds, calcula MMD entre
    folds (que por construção não apresentam drift), e estima o
    limiar no percentil 95 das distribuições nulas.

    Conforme Documento 04, Seção "Saiba Mais": definir o limiar
    de significância via permutações (Gretton et al., 2012).

    Args:
        X_reference: Dados de referência (n, d).
        n_folds: Número de folds.
        n_permutations: Permutações por fold (não usado diretamente aqui).
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário com limiar estimado e valores nulos de MMD.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X_reference, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    indices = np.arange(len(X))
    rng.shuffle(indices)
    fold_size = len(X) // n_folds
    null_mmds: List[float] = []
    mmd_calc = MMDCalculator()

    for i in range(n_folds - 1):
        start = i * fold_size
        end = start + fold_size
        fold_a = X[indices[start:end]]
        fold_b = X[indices[end : end + fold_size]]
        null_mmds.append(mmd_calc.calculate(fold_a, fold_b))

    threshold_95 = float(np.percentile(null_mmds, 95))
    return {
        "threshold_95": threshold_95,
        "null_mmds": null_mmds,
        "mean_null": float(np.mean(null_mmds)),
        "std_null": float(np.std(null_mmds)),
    }


def hyperparameter_tuning(
    X_reference: np.ndarray,
    X_drifted: np.ndarray,
    gamma_candidates: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Busca o melhor γ para o kernel RBF da MMD.

    Avalia a separação entre MMD sob H0 (dados de referência vs
    referência) e MMD sob H1 (referência vs dados com drift),
    selecionando o γ que maximiza essa separação.

    Conforme Documento 04, Seção "Saiba Mais": a escolha do
    parâmetro do kernel é crítica para a sensibilidade da MMD.

    Args:
        X_reference: Dados de referência.
        X_drifted: Dados com drift conhecido.
        gamma_candidates: Lista de γ candidatos. None = auto.
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário com melhor γ e resultados por candidato.
    """
    rng = np.random.RandomState(seed)
    X_ref = np.asarray(X_reference, dtype=np.float64)
    X_drift = np.asarray(X_drifted, dtype=np.float64)
    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_drift.ndim == 1:
        X_drift = X_drift.reshape(-1, 1)

    if gamma_candidates is None:
        # Gera candidatos em escala logarítmica ao redor da mediana heuristic
        mmd_calc = MMDCalculator()
        auto_gamma = mmd_calc._compute_gamma(X_ref, X_drift)
        gamma_candidates = [
            auto_gamma * factor for factor in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
        ]

    results: List[Dict[str, float]] = []
    best_gamma = gamma_candidates[0]
    best_separation = -np.inf

    for gamma in gamma_candidates:
        mmd_calc = MMDCalculator(gamma=gamma)

        # MMD sob H0: ref vs ref (split metade)
        n_half = len(X_ref) // 2
        idx = rng.permutation(len(X_ref))
        mmd_h0 = mmd_calc.calculate(X_ref[idx[:n_half]], X_ref[idx[n_half: 2 * n_half]])

        # MMD sob H1: ref vs drift
        mmd_h1 = mmd_calc.calculate(X_ref, X_drift)

        separation = mmd_h1 - mmd_h0
        results.append(
            {"gamma": gamma, "mmd_h0": mmd_h0, "mmd_h1": mmd_h1, "separation": separation}
        )

        if separation > best_separation:
            best_separation = separation
            best_gamma = gamma

    return {
        "best_gamma": best_gamma,
        "best_separation": best_separation,
        "results": results,
    }

"""
Módulo de treinamento para detecção de drift em embeddings.

Implementa funções de treinamento do classificador adversário
(Lopez-Paz & Oquab, 2017), cálculo de estatísticas de referência
e tunagem de hiperparâmetros para os métodos de detecção de drift.

Conforme a seção 'Saiba Mais' da Aula 5, o classificador adversário
funciona treinando um modelo para distinguir dados de referência de
dados de produção. Se o classificador consegue distinguir com accuracy
significativamente acima de 50%, há evidência estatística de drift.

Referências:
    Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier two-sample tests. ICLR.
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from .model import EmbeddingDriftDetector


def train_adversarial_classifier(
    X_ref: np.ndarray,
    X_prod: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 3,
    test_size: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Treina classificador adversário para detecção de drift.

    Implementa a abordagem de Lopez-Paz & Oquab (2017) conforme discutida
    na seção 'Saiba Mais' do documento da Aula 5: combina dados de referência
    (label=0) e produção (label=1), treina um classificador e avalia se ele
    consegue distinguir os dois conjuntos.

    Se accuracy > 50% (aleatório), existe evidência de que as distribuições
    são diferentes (drift). Quanto maior a accuracy, maior o drift.

    Args:
        X_ref: Embeddings de referência (n, d).
        X_prod: Embeddings de produção (m, d).
        n_estimators: Número de estimadores do Gradient Boosting.
        max_depth: Profundidade máxima das árvores.
        test_size: Fração para teste.
        seed: Seed para reproduzibilidade.

    Returns:
        Dicionário com:
            - classifier: Modelo treinado.
            - train_accuracy: Accuracy no treino.
            - test_accuracy: Accuracy no teste.
            - cv_accuracy: Accuracy com cross-validation.
            - classification_report: Relatório de classificação.
            - drift_detected: Boolean.

    Referência:
        Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier
        two-sample tests. ICLR.
    """
    # Combina dados de referência (0) e produção (1)
    X_combined = np.vstack([X_ref, X_prod])
    y_combined = np.array([0] * len(X_ref) + [1] * len(X_prod))

    # Split treino/teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=test_size,
        random_state=seed, stratify=y_combined,
    )

    # Treina classificador
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Avaliação
    train_accuracy = float(accuracy_score(y_train, clf.predict(X_train)))
    test_accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    # Cross-validation completa
    cv_scores = cross_val_score(
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        ),
        X_combined,
        y_combined,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="accuracy",
    )

    report = classification_report(
        y_test, clf.predict(X_test),
        target_names=["referência", "produção"],
        output_dict=True,
    )

    return {
        "classifier": clf,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_accuracy": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "classification_report": report,
        "drift_detected": test_accuracy > 0.6,
    }


def compute_reference_statistics(
    X_ref: np.ndarray,
    gamma: float = 1.0,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Calcula estatísticas de baseline (referência) para calibração do detector.

    Divide os dados de referência em subconjuntos e calcula a distribuição
    de MMD sob a hipótese nula (sem drift). Isso permite definir limiares
    adequados para detecção.

    Conforme a seção 'Saiba Mais' da Aula 5, a decisão estatística é tomada
    comparando o valor observado da MMD com uma distribuição crítica derivada
    via simulações de permutação (Gretton et al., 2012).

    Args:
        X_ref: Embeddings de referência (n, d).
        gamma: Parâmetro do kernel RBF.
        n_bootstrap: Número de iterações de bootstrap.
        seed: Seed para reproduzibilidade.

    Returns:
        Dicionário com mmd_scores, threshold_95, threshold_99, mean_mmd, std_mmd.
    """
    rng = np.random.RandomState(seed)
    detector = EmbeddingDriftDetector(method="mmd", gamma=gamma)

    n = len(X_ref)
    half = n // 2
    mmd_scores: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.permutation(n)
        X1 = X_ref[idx[:half]]
        X2 = X_ref[idx[half : 2 * half]]
        mmd = detector.compute_mmd(X1, X2, gamma=gamma)
        mmd_scores.append(mmd)

    return {
        "mmd_scores": mmd_scores,
        "threshold_95": float(np.percentile(mmd_scores, 95)),
        "threshold_99": float(np.percentile(mmd_scores, 99)),
        "mean_mmd": float(np.mean(mmd_scores)),
        "std_mmd": float(np.std(mmd_scores)),
    }


def cross_validate_detector(
    detector: EmbeddingDriftDetector,
    X_ref: np.ndarray,
    X_prod: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Cross-valida o detector de drift usando múltiplos subconjuntos.

    Avalia a consistência da detecção dividindo os dados em folds e
    verificando se o drift é detectado consistentemente.

    Args:
        detector: Instância de EmbeddingDriftDetector.
        X_ref: Embeddings de referência (n, d).
        X_prod: Embeddings de produção (m, d).
        n_splits: Número de splits para cross-validation.
        seed: Seed para reproduzibilidade.

    Returns:
        Dicionário com scores, drift_detected_per_fold e consistency.
    """
    rng = np.random.RandomState(seed)
    n_ref = len(X_ref)
    n_prod = len(X_prod)

    fold_size_ref = n_ref // n_splits
    fold_size_prod = n_prod // n_splits

    scores: List[float] = []
    detected: List[bool] = []

    idx_ref = rng.permutation(n_ref)
    idx_prod = rng.permutation(n_prod)

    for i in range(n_splits):
        # Seleciona subconjuntos
        ref_fold = X_ref[idx_ref[i * fold_size_ref : (i + 1) * fold_size_ref]]
        prod_fold = X_prod[idx_prod[i * fold_size_prod : (i + 1) * fold_size_prod]]

        # Ajusta e prediz
        det = EmbeddingDriftDetector(
            method=detector.method,
            gamma=detector.gamma,
            n_permutations=detector.n_permutations,
        )
        det.fit(ref_fold)
        result = det.predict(prod_fold)

        scores.append(result["score"])
        detected.append(result["drift_detected"])

    consistency = sum(detected) / len(detected)

    return {
        "scores": scores,
        "drift_detected_per_fold": detected,
        "consistency": consistency,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }


def hyperparameter_tuning(
    X_ref: np.ndarray,
    X_prod: np.ndarray,
    method: str = "mmd",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Tunagem de hiperparâmetros para detecção de drift.

    Para MMD: busca o gamma ótimo do kernel RBF.
    Para adversarial: busca n_estimators e max_depth ótimos.

    Args:
        X_ref: Embeddings de referência.
        X_prod: Embeddings de produção.
        method: Método ('mmd' ou 'adversarial').
        param_grid: Grid de parâmetros (usa default se None).
        seed: Seed para reproduzibilidade.

    Returns:
        Dicionário com best_params, all_results e best_score.
    """
    if method == "mmd":
        if param_grid is None:
            param_grid = {"gamma": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

        results = []
        for gamma in param_grid["gamma"]:
            detector = EmbeddingDriftDetector(
                method="mmd", gamma=gamma, n_permutations=50,
            )
            detector.fit(X_ref)
            result = detector.predict(X_prod)
            results.append({
                "gamma": gamma,
                "mmd_score": result["score"],
                "p_value": result["p_value"],
                "drift_detected": result["drift_detected"],
            })

        best = max(results, key=lambda r: r["mmd_score"])
        return {
            "best_params": {"gamma": best["gamma"]},
            "all_results": results,
            "best_score": best["mmd_score"],
        }

    elif method == "adversarial":
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [2, 3, 5],
            }

        X_combined = np.vstack([X_ref, X_prod])
        y_combined = np.array([0] * len(X_ref) + [1] * len(X_prod))

        clf = GradientBoostingClassifier(random_state=seed)
        grid_search = GridSearchCV(
            clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1,
        )
        grid_search.fit(X_combined, y_combined)

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "all_results": [
                {
                    "params": grid_search.cv_results_["params"][i],
                    "mean_score": float(grid_search.cv_results_["mean_test_score"][i]),
                }
                for i in range(len(grid_search.cv_results_["params"]))
            ],
        }

    else:
        raise ValueError(f"Tunagem não suportada para método '{method}'")

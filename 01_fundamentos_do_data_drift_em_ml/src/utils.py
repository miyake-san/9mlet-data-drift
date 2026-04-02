"""
Módulo de utilitários para a Aula 1.

Funções auxiliares para persistência de modelos, métricas e configuração.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def get_project_root() -> Path:
    """Retorna o diretório raiz desta aula."""
    return Path(__file__).resolve().parent.parent


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura logging para a aula.

    Args:
        log_dir: Diretório de logs (default: outputs/logs/).
        level: Nível de logging.

    Returns:
        Logger configurado.
    """
    if log_dir is None:
        log_dir = get_project_root() / "outputs" / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger("aula01_data_drift")
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_model(model: Any, filepath: str | Path) -> Path:
    """Salva modelo com joblib.

    Args:
        model: Objeto do modelo (AdaptiveClassifier ou sklearn).
        filepath: Caminho de saída (recomendado: .joblib).

    Returns:
        Path do arquivo salvo.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    return filepath


def load_model(filepath: str | Path) -> Any:
    """Carrega modelo salvo com joblib.

    Args:
        filepath: Caminho do arquivo .joblib.

    Returns:
        Objeto do modelo.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    return joblib.load(filepath)


def save_metrics(metrics: dict[str, Any], filepath: str | Path) -> Path:
    """Salva métricas em JSON.

    Args:
        metrics: Dicionário de métricas.
        filepath: Caminho de saída (.json).

    Returns:
        Path do arquivo salvo.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Converter tipos numpy para Python nativo
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_convert(metrics), f, indent=2, ensure_ascii=False)

    return filepath


def load_metrics(filepath: str | Path) -> dict[str, Any]:
    """Carrega métricas salvas em JSON.

    Args:
        filepath: Caminho do arquivo .json.

    Returns:
        Dicionário de métricas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Métricas não encontradas: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    """Cria diretórios de saída se não existirem."""
    root = get_project_root()
    for subdir in ["outputs/models", "outputs/figures", "outputs/logs",
                    "data/raw", "data/processed"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

"""
Módulo de utilitários para a Aula 2.

Funções auxiliares para persistência de resultados, métricas e configuração.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

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

    logger = logging.getLogger("aula02_drift_detection")
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
    """Salva modelo ou objeto com json.

    Args:
        model: Dicionário ou objeto serializável.
        filepath: Caminho de saída (.json).

    Returns:
        Path do arquivo salvo.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_convert_numpy(model), f, indent=2, ensure_ascii=False)

    return filepath


def load_model(filepath: str | Path) -> Any:
    """Carrega resultados salvos em JSON.

    Args:
        filepath: Caminho do arquivo .json.

    Returns:
        Objeto carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics: dict[str, Any], filepath: str | Path) -> Path:
    """Salva métricas de drift em JSON.

    Args:
        metrics: Dicionário de métricas.
        filepath: Caminho de saída (.json).

    Returns:
        Path do arquivo salvo.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_convert_numpy(metrics), f, indent=2, ensure_ascii=False)

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


def load_config(filepath: str | Path | None = None) -> dict[str, Any]:
    """Carrega configuração de execução.

    Args:
        filepath: Caminho do arquivo de configuração JSON.
            Se None, retorna configuração padrão.

    Returns:
        Dicionário de configuração.
    """
    default_config: dict[str, Any] = {
        "alpha": 0.05,
        "psi_threshold": 0.25,
        "n_bins": 10,
        "n_samples": 5000,
        "seed": 42,
        "methods": ["ks", "psi"],
    }

    if filepath is None:
        return default_config

    filepath = Path(filepath)
    if not filepath.exists():
        return default_config

    with open(filepath, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    default_config.update(user_config)
    return default_config


def ensure_dirs() -> None:
    """Cria diretórios de saída se não existirem."""
    root = get_project_root()
    for subdir in [
        "outputs/models",
        "outputs/figures",
        "outputs/logs",
        "data/raw",
        "data/processed",
    ]:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def _convert_numpy(obj: Any) -> Any:
    """Converte tipos numpy para Python nativo (para JSON).

    Args:
        obj: Objeto a converter.

    Returns:
        Objeto com tipos Python nativos.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj

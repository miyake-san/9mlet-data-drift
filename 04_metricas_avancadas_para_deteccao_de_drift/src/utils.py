# -*- coding: utf-8 -*-
"""
utils.py – Funções utilitárias para a Aula 04.

Funções de I/O, configuração, logging e persistência de resultados
para detecção de drift multivariado.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def get_project_root() -> Path:
    """Retorna o diretório raiz do módulo (04_metricas_...).

    Returns:
        Path do diretório raiz.
    """
    return Path(__file__).resolve().parent.parent


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura logging com saída para arquivo e console.

    Args:
        log_dir: Diretório para logs. None = outputs/logs/.
        level: Nível de logging.

    Returns:
        Logger configurado.
    """
    if log_dir is None:
        log_dir = str(get_project_root() / "outputs" / "logs")

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"drift_detection_{timestamp}.log")

    logger = logging.getLogger("drift_aula04")
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        logger.addHandler(fh)

    return logger


def save_model(model: Any, path: Optional[str] = None) -> str:
    """Salva um detector de drift em disco via pickle.

    Args:
        model: Objeto DriftDetector (ou qualquer serializável).
        path: Caminho de saída. None = outputs/models/detector.pkl.

    Returns:
        Caminho do arquivo salvo.
    """
    if path is None:
        out_dir = str(get_project_root() / "outputs" / "models")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "detector.pkl")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: Optional[str] = None) -> Any:
    """Carrega um detector de drift salvo em disco.

    Args:
        path: Caminho do arquivo pickle. None = outputs/models/detector.pkl.

    Returns:
        Objeto deserializado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if path is None:
        path = str(get_project_root() / "outputs" / "models" / "detector.pkl")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(
    metrics: Dict[str, Any],
    path: Optional[str] = None,
) -> str:
    """Salva métricas em formato JSON.

    Args:
        metrics: Dicionário com resultados de métricas.
        path: Caminho de saída. None = outputs/logs/metrics.json.

    Returns:
        Caminho do arquivo salvo.
    """
    if path is None:
        out_dir = str(get_project_root() / "outputs" / "logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "metrics.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Serializa tipos numpy para JSON
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(metrics, default=_convert))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Carrega configuração JSON.

    Args:
        path: Caminho do arquivo de configuração.

    Returns:
        Dicionário de configuração.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if path is None:
        path = str(get_project_root() / "config.json")

    if not os.path.isfile(path):
        # Retorna configuração padrão
        return {
            "n_samples": 5000,
            "seed": 42,
            "psi_threshold": 0.25,
            "psi_n_bins": 10,
            "mmd_gamma": None,
            "n_permutations": 100,
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

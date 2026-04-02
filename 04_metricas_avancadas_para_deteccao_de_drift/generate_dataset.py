#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_dataset.py – Gera o dataset sintético para a Aula 04.

Uso:
    python generate_dataset.py
"""

import sys
from pathlib import Path

# Adiciona o diretório pai ao path para importação
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_preprocessing import DataPreprocessor


def main() -> None:
    dp = DataPreprocessor(n_samples=5000, seed=42)
    path = dp.generate_and_save()
    print(f"Dataset gerado em: {path}")

    df = dp.generate_dataset()
    print(f"Shape: {df.shape}")
    print(f"Períodos: {df['periodo'].value_counts().to_dict()}")
    print(f"\nPrimeiras linhas:\n{df.head()}")


if __name__ == "__main__":
    main()

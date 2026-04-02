#!/usr/bin/env python
"""
Script para gerar o dataset sintético da Aula 6.

Gera dados de crédito digital simulando cenário de fintech com drift,
conforme descrito no material da aula (DOCUMENTO_AULA_6.md).

Uso:
    python generate_dataset.py
    python generate_dataset.py --output data/raw/dataset.csv --seed 42
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório pai ao path para importar src/
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import DataPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera dataset sintético para Aula 6 - Monitoramento Contínuo e Data SLOs"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/dataset.csv",
        help="Caminho de saída do CSV (default: data/raw/dataset.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--n-reference", type=int, default=5000,
                        help="Amostras de referência (default: 5000)")
    parser.add_argument("--n-production", type=int, default=5000,
                        help="Amostras de produção (default: 5000)")

    args = parser.parse_args()

    print(f"Gerando dataset sintético de crédito digital (seed={args.seed})...")
    df = DataPreprocessor.generate_dataset(
        n_reference=args.n_reference,
        n_production=args.n_production,
        output_path=args.output,
        random_state=args.seed,
    )
    print(f"\nDataset gerado com {len(df)} amostras → {args.output}")
    print(f"  Referência: {(df['is_production'] == 0).sum()}")
    print(f"  Produção:   {(df['is_production'] == 1).sum()}")
    print(f"  Target=1:   {df['target'].mean():.1%}")
    print(f"  Missings:   {df.isnull().sum().sum()} valores ausentes no total")


if __name__ == "__main__":
    main()

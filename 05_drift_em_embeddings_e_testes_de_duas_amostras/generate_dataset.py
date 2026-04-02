#!/usr/bin/env python
"""
Script para gerar o dataset sintético da Aula 5.

Gera embeddings simulados representando textos de redes sociais com três
períodos: referência, produção estável e produção com drift semântico.

Uso:
    python generate_dataset.py
    python generate_dataset.py --output data/raw/dataset.csv --seed 42
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório pai ao path para importar src/
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera dataset sintético para Aula 5 - Drift em Embeddings"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/dataset.csv",
        help="Caminho de saída do CSV (default: data/raw/dataset.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--n-reference", type=int, default=5000)
    parser.add_argument("--n-stable", type=int, default=2500)
    parser.add_argument("--n-drift", type=int, default=2500)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--drift-magnitude", type=float, default=1.5)

    args = parser.parse_args()

    print(f"Gerando dataset sintético (seed={args.seed})...")
    df = generate_synthetic_dataset(
        n_reference=args.n_reference,
        n_production_stable=args.n_stable,
        n_production_drift=args.n_drift,
        embedding_dim=args.embedding_dim,
        drift_magnitude=args.drift_magnitude,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Dataset gerado com {len(df)} amostras → {args.output}")
    print(f"  Referência:        {(df['period'] == 'reference').sum()}")
    print(f"  Produção estável:  {(df['period'] == 'production_stable').sum()}")
    print(f"  Produção com drift:{(df['period'] == 'production_drift').sum()}")


if __name__ == "__main__":
    main()

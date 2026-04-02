#!/usr/bin/env python3
"""
Script para geração do dataset sintético da Aula 1.

Gera 10.000 instâncias de dados de clientes de loja online distribuídas
em 5 períodos temporais, com data drift injetado a partir do período 3.

Uso:
    python scripts/generate_dataset.py

O dataset é salvo em data/raw/dataset.csv.
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz da aula ao path para importar src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import DataPreprocessor


def main() -> None:
    """Gera e salva o dataset sintético."""
    base_dir = Path(__file__).resolve().parent.parent

    preprocessor = DataPreprocessor(random_state=42)

    print("Gerando dataset sintético (10.000 instâncias, 5 períodos)...")
    df = preprocessor.generate_synthetic_data(
        n_samples=10_000,
        n_periods=5,
        drift_start_period=3,
        drift_magnitude=1.5,
    )

    out_path = base_dir / "data" / "raw" / "dataset.csv"
    preprocessor.save_data(df, out_path)

    # Estatísticas rápidas
    print(f"\nDataset salvo em: {out_path}")
    print(f"  Total de linhas: {len(df)}")
    print(f"  Colunas: {list(df.columns)}")
    print(f"\n  Distribuição por período:")
    print(df["periodo"].value_counts().sort_index().to_string())
    print(f"\n  Distribuição do target:")
    print(df["target"].value_counts().to_string())
    print(f"\n  Primeiras 5 linhas:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()

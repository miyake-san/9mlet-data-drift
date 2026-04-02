#!/usr/bin/env python3
"""
Script para geração do dataset sintético da Aula 2.

Gera 10.000 instâncias de dados de clientes de fintech de crédito,
com 5.000 de referência (distribuição estável) e 5.000 de produção
(com drift injetado), simulando o cenário descrito no Documento da Aula 2.

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
    out_path = base_dir / "data" / "raw" / "dataset.csv"

    preprocessor = DataPreprocessor(seed=42, n_samples=5000)

    print("Gerando dataset sintético (10.000 instâncias: 5K ref + 5K prod)...")
    df = preprocessor.generate_and_save(str(out_path))

    # Estatísticas rápidas
    ref = df[df["origem"] == "referencia"]
    prod = df[df["origem"] == "producao"]
    print(f"\nReferência: {len(ref)} amostras")
    print(f"Produção:   {len(prod)} amostras")
    print(f"\nIdade média (ref):  {ref['idade'].mean():.1f}")
    print(f"Idade média (prod): {prod['idade'].mean():.1f}")
    print(f"\nInadimplência (ref):  {ref['inadimplente'].mean():.1%}")
    print(f"Inadimplência (prod): {prod['inadimplente'].mean():.1%}")
    print(f"\nDataset salvo em: {out_path}")


if __name__ == "__main__":
    main()

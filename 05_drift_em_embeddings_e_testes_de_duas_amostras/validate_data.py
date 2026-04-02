"""Validate dataset structure."""
import pandas as pd

df = pd.read_csv("data/raw/dataset.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"First 5 cols: {list(df.columns[:5])}")
print(f"Embedding cols: {len([c for c in df.columns if c.startswith('emb_')])}")
print(f"Periods: {df['period'].value_counts().to_dict()}")
print(f"Categories: {df['category'].unique().tolist()}")
print(f"Shape: {df.shape}")

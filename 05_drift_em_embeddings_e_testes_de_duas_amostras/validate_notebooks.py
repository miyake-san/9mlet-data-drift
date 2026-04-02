"""Validation script for notebook structure."""
import json
from pathlib import Path

def validate_notebook(path):
    nb = json.load(open(path, "r", encoding="utf-8"))
    cells = nb["cells"]
    md = [c for c in cells if c["cell_type"] == "markdown"]
    code = [c for c in cells if c["cell_type"] == "code"]
    name = Path(path).stem
    
    print(f"\n=== {name} ===")
    print(f"Total cells: {len(cells)} (MD: {len(md)}, Code: {len(code)})")
    print(f"First cell: {cells[0]['cell_type']}, Last cell: {cells[-1]['cell_type']}")
    
    # Check range 15-25
    in_range = 15 <= len(cells) <= 25
    print(f"Cell count in range [15-25]: {'YES' if in_range else 'NO'}")
    
    # Check alternating pattern
    all_text = " ".join("".join(c["source"]) for c in cells)
    refs = ["Gretton", "Massey", "Lopez-Paz", "documento", "Aula 5", "Saiba Mais", "BERT", "MMD", "TrendCast", "kernel"]
    found = [r for r in refs if r in all_text]
    print(f"Doc04 references: {found}")
    
    # Check imports in cell 2
    if len(cells) > 1 and cells[1]["cell_type"] == "code":
        src = "".join(cells[1]["source"])
        has_imports = "import" in src
        print(f"Cell 2 has imports: {has_imports}")
    
    return in_range

for nb in ["notebooks/01_exploracao.ipynb", "notebooks/02_treinamento.ipynb", "notebooks/03_avaliacao.ipynb"]:
    validate_notebook(nb)

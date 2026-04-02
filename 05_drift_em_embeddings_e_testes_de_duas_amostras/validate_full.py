"""Validate doc 04 connections in source code."""
from pathlib import Path
import re

# Academic references that should appear
refs = {
    "Gretton": "MMD kernel two-sample test",
    "Massey": "KS test",
    "Lopez-Paz": "Adversarial classifier",
    "Devlin": "BERT",
    "Mikolov": "Word embeddings",
    "Feldhans": "Drift in text data",
}

# Required doc references
doc_refs = [
    "Aula 5",
    "Saiba Mais",
    "documento",
]

src_dir = Path("src")
notebooks_dir = Path("notebooks")
tests_dir = Path("tests")

print("===== SOURCE CODE VALIDATION =====\n")

for py_file in sorted(src_dir.glob("*.py")):
    if py_file.name == "__pycache__":
        continue
    content = py_file.read_text(encoding="utf-8")
    
    print(f"--- {py_file.name} ---")
    
    # Check type hints
    func_count = len(re.findall(r"def \w+\(", content))
    typed = len(re.findall(r"def \w+\([^)]*:\s*\w+", content))
    print(f"  Functions: {func_count}, With type hints: {typed}")
    
    # Check docstrings
    docstrings = len(re.findall(r'"""[\s\S]*?"""', content))
    print(f"  Docstrings: {docstrings}")
    
    # Check references
    found_refs = [r for r in refs if r in content]
    found_doc = [r for r in doc_refs if r in content]
    print(f"  Academic refs: {found_refs}")
    print(f"  Doc04 refs: {found_doc}")
    print()

print("\n===== TEST FILE VALIDATION =====\n")
total_tests = 0
for py_file in sorted(tests_dir.glob("test_*.py")):
    content = py_file.read_text(encoding="utf-8")
    test_count = len(re.findall(r"def test_\w+", content))
    total_tests += test_count
    has_fixtures = "fixture" in content
    has_seed = "seed" in content.lower()
    print(f"--- {py_file.name} ---")
    print(f"  Tests: {test_count}")
    print(f"  Uses fixtures: {has_fixtures}")
    print(f"  Fixed seed: {has_seed}")

print(f"\nTotal tests: {total_tests}")
print(f"Min required: 10 -> {'PASS' if total_tests >= 10 else 'FAIL'}")

print("\n===== MISSING FILES CHECK =====\n")
required = [
    "README.md",
    "requirements.txt",
    "data/README.md",
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/model.py",
    "src/training.py",
    "src/evaluation.py",
    "src/utils.py",
    "notebooks/01_exploracao.ipynb",
    "notebooks/02_treinamento.ipynb",
    "notebooks/03_avaliacao.ipynb",
    "tests/__init__.py",
    "tests/test_preprocessing.py",
    "tests/test_model.py",
    "tests/test_evaluation.py",
    "data/raw/dataset.csv",
    "generate_dataset.py",
]

for f in required:
    exists = Path(f).exists()
    status = "EXISTS" if exists else "MISSING"
    print(f"  [{status}] {f}")

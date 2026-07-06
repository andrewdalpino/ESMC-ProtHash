# AGENTS.md

## Project Overview

**ProtHash** is a fast protein language model that outputs contextual embeddings aligned by biological properties (structure and function). It is a distilled (student) model of ESMC trained on UniRef50 (53M+ sequences), matching up to 98% of the ESMC embedding space at reduced cost.

## Commands

```sh
# Editable install
pip install -e .

# Install with dev/test deps
pip install -e ".[dev]"
pip install -e ".[test]"

# Build wheel
python -m build

# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_model -v
python -m unittest tests.test_loss -v
python -m unittest tests.test_metrics -v
```

## Project Structure

```
src/prothash/         # Installed package (model.py, __init__.py)
data.py               # Dataset classes (UniRef50, samplers)
loss.py               # Loss functions (DecomposedTokenRepresentationLoss, WeightedCombinedLoss)
metrics.py            # Metrics (CosineSimilarity, LinearCKA, Top1MacroF1)
distill.py            # Distillation training script
tests/                # Unit tests (unittest framework)
```

## Coding Conventions

- **Indentation:** 4 spaces
- **Naming:** PascalCase classes, snake_case functions/vars, UPPER_SNAKE_CASE constants
- **Imports:** stdlib → third-party → local, grouped with blank lines
- **Type hints:** use on all function signatures
- **Docstrings:** Google/NumPy style (`Args:`, `Returns:`)
- **Patterns:** `self.layer.forward(x)` (explicit forward), `super().__init__()`, `@property` for computed attrs, `@torch.inference_mode()` on inference methods
- **Input validation:** use `assert` statements
- **Private helpers:** prefix with underscore (`_helper`)

## Testing Guidelines

- Framework: `unittest.TestCase` (no pytest)
- Class naming: `Test<Component>`
- Method naming: `test_<scenario>`
- Use `setUp` with `torch.manual_seed(42)` for reproducibility
- Generate test data with `torch.randn`/`torch.randint`

## Notes

- `model.py` lives under `src/prothash/` (installed package), while `loss.py`, `metrics.py`, and `data.py` are at repo root
- `torch` version: 2.9.1 (production), 2.7.1 (training via requirements.txt)
- License: Apache 2.0

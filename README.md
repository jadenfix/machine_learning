# machine_learning

## Reproduce the checked research workflow

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_research.py 'Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb'
```

Five ensemble models train on the built-in Wisconsin Diagnostic Breast Cancer dataset, with a stratified 70/30 split. This is an educational fixed holdout evaluation, not a medical decision system or independent validation study. Figures and a cell execution manifest are written to `research-output/`. The runner executes the original code cells, stops on errors, seeds NumPy to 42, and uses a headless plotting backend.

Output publication is transactional: a run is assembled in a staging directory and replaces `research-output/` only after every notebook cell and artifact write succeeds. If a rerun fails, the last complete evidence directory remains byte-for-byte unchanged instead of being mixed with partial or stale results. The manifest records the executed cells and notebook SHA-256.

```bash
python -m unittest discover -s tests -v
```

Runnable notebooks: `Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb`.

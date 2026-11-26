# Episode 1: your 99% accurate model is lying

A model trained on every column of `data/loans.csv` scores a perfect AUC. This episode shows why that is
a bug, finds the columns carrying the answer, and builds the honest model.

```bash
uv sync
uv run python 01-leakage-intro/leakage_intro.py
```

Or open `leakage_intro.py` in VS Code or Jupyter and run it cell by cell (each `# %%` is a cell).
It takes about 10 seconds and writes `single_feature_auc.png` next to the script.

## What you will see

| Model | Accuracy | AUC |
| --- | --- | --- |
| Every column | 100.0% | 1.000 |
| Predict "no default" for everyone | 86.9% | 0.5 |
| Only columns known on the application date | 88.4% | 0.817 |

## Steps

1. Train on every column with a random 80/20 split: a perfect score.
2. Compare with a model that does nothing: accuracy alone fools you.
3. Single-feature AUC scan: which one column predicts default on its own?
4. Ask of every column: would the lender know this on the application date?
5. Drop what fails, retrain, get the honest number.

The honest number is still a little too high. Episode 2 shows why.

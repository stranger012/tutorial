# %% [markdown]
# # Episode 1: your 99% accurate model is lying
#
# We predict whether a loan defaults within 12 months, using the lender's export in `data/loans.csv`.
# The prediction is made on the **application date**. Anything the lender only learned after that day
# cannot be used, because it will not exist when the model runs for real.
#
# Run it as a script (`uv run python 01-leakage-intro/leakage_intro.py`) or cell by cell in
# VS Code / Jupyter: every `# %%` line starts a cell.

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the chart to a file; works without a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LABEL = "defaulted_12m"
SEED = 42

loans = pd.read_csv(DATA / "loans.csv", parse_dates=["application_date", "last_payment_date", "default_date"])
print(f"{len(loans):,} loans, {loans.shape[1]} columns")
print(f"Default rate: {loans[LABEL].mean():.1%}")


# %% [markdown]
# ## Step 1: the model everyone builds first
#
# Use every column, split 80/20 at random, train a gradient-boosted model.


# %%
def to_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Dates become day numbers, text becomes category codes, so any model can read them."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            out[col] = (df[col] - pd.Timestamp("2020-01-01")).dt.days
        elif df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            out[col] = df[col].astype("category").cat.codes
        else:
            out[col] = df[col]
    return out


def evaluate(features: list[str], name: str) -> dict:
    X = to_numbers(loans[features])
    y = loans[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = HistGradientBoostingClassifier(random_state=SEED).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    result = {
        "model": name,
        "features": len(features),
        "accuracy": accuracy_score(y_test, proba > 0.5),
        "auc": roc_auc_score(y_test, proba),
    }
    print(f"{name:28} accuracy {result['accuracy']:.1%}   AUC {result['auc']:.3f}")
    return result


all_columns = [c for c in loans.columns if c != LABEL]
naive = evaluate(all_columns, "Every column")

# %% [markdown]
# Almost perfect. Before celebrating, compare it with a model that does nothing.

# %%
always_no = 1 - loans[LABEL].mean()
print(f"Predict 'no default' for everyone: accuracy {always_no:.1%}")

# %% [markdown]
# Accuracy alone fools you: doing nothing already scores high, because most loans do not default.
# AUC is the better yardstick here: 0.5 is a coin flip, 1.0 is perfect.
# Credit models in the real world usually land around 0.70 to 0.85. An AUC this close to 1.0 is
# not a breakthrough; it is a bug report.
#
# ## Step 2: find what is carrying the answer
#
# Train a tiny model on **one column at a time**. Any single column that predicts the label almost
# perfectly is suspect.


# %%
def single_feature_auc(column: str) -> float:
    X = to_numbers(loans[[column]])
    y = loans[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=50, random_state=SEED).fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


scan = pd.Series({c: single_feature_auc(c) for c in all_columns}).sort_values(ascending=False)
print(scan.round(3).to_string())

# %%
colors = ["#c0392b" if auc >= 0.85 else "#7f8c8d" for auc in scan]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(scan.index[::-1], scan.values[::-1], color=colors[::-1])
ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":")
ax.set_xlim(0.4, 1.02)
ax.set_xlabel("AUC using this one column alone (0.5 = coin flip)")
ax.set_title("Which single column predicts default?")
fig.tight_layout()
fig.savefig(HERE / "single_feature_auc.png", dpi=150)
print(f"Chart saved to {HERE / 'single_feature_auc.png'}")

# %% [markdown]
# ## Step 3: ask one question of every suspect
#
# **Would the lender know this on the application date?**
#
# | Column | Known on the application date? |
# | --- | --- |
# | `months_paid_first_year` | No: counted over the 12 months after the loan starts |
# | `loan_status` | No: the status on the export date, years later |
# | `default_date` | No: it only exists if the loan defaulted |
# | `total_recovered` | No: collections happen after a default |
# | `collection_calls` | No: same |
# | `last_payment_date` | No: payments come after the application |
# | `hardship_flag` | No: set while the borrower is struggling |
# | `loan_id` | An ID should carry no information. If it predicts the label, something about how the data was stored leaks the outcome |
#
# This is **target leakage**: the feature is a consequence of the outcome, so the model reads the
# answer instead of learning the risk.
#
# Notice `last_payment_date` and `hardship_flag`: the scan did not flag them red, yet both are from
# after the application. The scan finds the loud leaks; the timeline question finds the quiet ones.
# Ask it of every column, not just the red bars.

# %%
after_the_fact = [
    "months_paid_first_year",
    "loan_status",
    "default_date",
    "total_recovered",
    "collection_calls",
    "last_payment_date",
    "hardship_flag",
]
identifiers = ["loan_id", "customer_id"]

# %% [markdown]
# ## Step 4: the honest model

# %%
honest_columns = [c for c in all_columns if c not in after_the_fact + identifiers]
print("Kept:", ", ".join(honest_columns))
honest = evaluate(honest_columns, "Known at application only")

# %%
summary = pd.DataFrame([naive, honest]).set_index("model")
print(summary.round(3).to_string())

# %% [markdown]
# The honest score is lower and it is the real one: this is roughly what the model will do on
# tomorrow's applications.
#
# ## Recap
#
# 1. Leakage = training on information you will not have when you predict.
# 2. Two families: a feature that carries the answer (**target leakage**, today), and test data that
#    touched training (**train-test contamination**, next episode).
# 3. Suspiciously high score? Run a single-feature scan, then ask of each suspect: *known at
#    prediction time?*
#
# We are not done. This data still hides more leaks; the honest number above is still a little too
# high. Next episode: splitting the data the right way.

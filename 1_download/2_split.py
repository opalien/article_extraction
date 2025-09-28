from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE = Path("data/brut/all_ai_models.csv")
DATA_DIR = Path("data/tables")
RATIOS = {"train": 0.7, "dev": 0.2, "test": 0.1}
RANDOM_STATE = 42


def split_dataframe(df):
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    total = len(shuffled)

    train_end = int(total * RATIOS["train"])
    dev_end = train_end + int(total * RATIOS["dev"])

    splits = {
        "train": shuffled.iloc[:train_end],
        "dev": shuffled.iloc[train_end:dev_end],
        "test": shuffled.iloc[dev_end:]
    }

    # Ensure all rows are accounted for by reallocating any rounding remainder to the test split.
    remainder = total - sum(len(v) for v in splits.values())
    if remainder > 0:
        extra = shuffled.iloc[-remainder:]
        splits["test"] = pd.concat([splits["test"], extra], ignore_index=True)

    return splits


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Le fichier source {SOURCE} est introuvable.")

    df = pd.read_csv(SOURCE)
    splits = split_dataframe(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, subset in splits.items():
        subset.to_csv(DATA_DIR / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()

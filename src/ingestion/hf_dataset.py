"""Download a public Q&A sample for later RAG benchmarking."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROCESSED_DIR = Path("data/processed")


def load_squad_sample(size: int = 500) -> pd.DataFrame:
    """Load a SQuAD v1.1 sample as a DataFrame."""
    dataset = load_dataset("rajpurkar/squad", split=f"train[:{size}]")
    return pd.DataFrame(
        {
            "context": dataset["context"],
            "question": dataset["question"],
            "answer": [answers["text"] for answers in dataset["answers"]],
        }
    )


if __name__ == "__main__":
    dataframe = load_squad_sample()
    output_path = PROCESSED_DIR / "squad_sample.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    print(f"Saved {len(dataframe)} Q&A pairs -> {output_path}")
    print(dataframe.head(3))
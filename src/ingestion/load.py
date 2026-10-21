"""Load PDFs into cleaned page-level text records."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))


@dataclass
class Document:
    doc_id: str
    source_file: str
    page_number: int
    text: str
    char_count: int


def load_pdf(pdf_path: Path) -> list[Document]:
    """Extract one Document record per readable page."""
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if len(text) < 30:
            continue
        documents.append(
            Document(
                doc_id=f"{pdf_path.stem}_p{page_number}",
                source_file=pdf_path.name,
                page_number=page_number,
                text=text,
                char_count=len(text),
            )
        )
    return documents


def load_all_pdfs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every PDF directly inside raw_dir."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Put PDF files there first."
        )

    all_documents: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name} ...")
        documents = load_pdf(pdf_path)
        print(f"  -> {len(documents)} pages with text")
        all_documents.extend(documents)

    dataframe = pd.DataFrame(asdict(document) for document in all_documents)
    print(f"\nTotal: {len(dataframe)} pages from {len(pdf_paths)} PDFs")
    return dataframe


def save_processed(
    dataframe: pd.DataFrame,
    output_path: Path = PROCESSED_DIR / "documents.parquet",
) -> Path:
    """Save extracted records as a Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    print(f"Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    save_processed(load_all_pdfs())
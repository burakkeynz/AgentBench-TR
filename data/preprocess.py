"""
Preprocessing raw Turkish documents into cleaned text files.
"""

import re
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    """Cleaning raw text by removing HTML, markdown artifacts, and noise."""
    # Removing PDF cid encoding artifacts
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Removing HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Removing HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Removing URLs
    text = re.sub(r"https?://\S+", "", text)
    # Removing image/badge references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Removing markdown links but keeping label
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Removing code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Removing inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Removing markdown table rows and separators
    text = re.sub(r"\|.*?\|", " ", text)
    text = re.sub(r"[-]{3,}", " ", text)
    # Removing markdown headers symbols
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Removing markdown bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Removing emoji unicode blocks
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\U0001F300-\U0001F9FF]", "", text, flags=re.UNICODE)
    # Removing leftover special characters (keep Turkish chars)
    text = re.sub(r"[^\w\s\.,;:!?()'\"\-\nğüşıöçĞÜŞİÖÇ]", " ", text)
    # Collapsing multiple whitespace/blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_txt(filepath: Path) -> str:
    """Reading and cleaning a .txt or .md file."""
    print(f"Processing text file: {filepath.name}...")
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return clean_text(raw)


def process_pdf(filepath: Path) -> str:
    """Extracting and cleaning text from a PDF file."""
    print(f"Processing PDF file: {filepath.name}...")
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return clean_text("\n\n".join(text_parts))
    except ImportError:
        import fitz
        doc = fitz.open(str(filepath))
        text_parts = [page.get_text() for page in doc]
        doc.close()
        return clean_text("\n\n".join(text_parts))


def process_docx(filepath: Path) -> str:
    """Extracting and cleaning text from a .docx file."""
    print(f"Processing Word document: {filepath.name}...")
    from docx import Document
    doc = Document(str(filepath))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n\n".join(paragraphs))


def process_all() -> None:
    """Processing all raw documents into cleaned text files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for filepath in sorted(RAW_DIR.iterdir()):
        suffix = filepath.suffix.lower()

        if suffix in (".txt", ".md"):
            cleaned = process_txt(filepath)
        elif suffix == ".pdf":
            cleaned = process_pdf(filepath)
        elif suffix == ".docx":
            cleaned = process_docx(filepath)
        else:
            print(f"Skipping unsupported file type: {filepath.name}")
            continue

        out_path = PROCESSED_DIR / (filepath.stem + "_cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        word_count = len(cleaned.split())
        print(f"  Saving {out_path.name} — {word_count} words.")
        processed_count += 1

    print(f"\nProcessing complete — {processed_count} documents saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    process_all()

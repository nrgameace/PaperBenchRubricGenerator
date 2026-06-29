"""Discover and load inputs from a structured data/input/{paper}/ directory."""

import json
from pathlib import Path


def discover_pdf(input_dir: Path) -> Path:
    """Return the first .pdf file in input_dir or raise FileNotFoundError."""
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {input_dir}")
    return pdfs[0]


def discover_mineru_dir(input_dir: Path) -> Path:
    """Return the first subdirectory of input_dir that contains content_list.json."""
    for candidate in sorted(input_dir.iterdir()):
        if candidate.is_dir() and (candidate / "content_list.json").exists():
            return candidate
    raise FileNotFoundError(f"No MinerU output folder (with content_list.json) found in {input_dir}")


def load_content_list(mineru_dir: Path) -> list:
    """Parse content_list.json from the MinerU output directory."""
    path = mineru_dir / "content_list.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

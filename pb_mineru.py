"""Convert MinerU content_list blocks to LLM-readable text and slice by section."""

import difflib


def _is_heading(block: dict) -> bool:
    """Return True if block is a section heading (has text_level)."""
    return block.get("type") == "text" and block.get("text_level") is not None


def _heading_level(block: dict) -> int:
    return block.get("text_level", 99)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def blocks_to_text(blocks: list) -> str:
    """Convert MinerU content blocks to a structured, LLM-readable string."""
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "").strip()
            if not text:
                continue
            level = block.get("text_level")
            parts.append(f"{'#' * min(level, 4)} {text}" if level else text)
        elif btype == "image":
            caption = block.get("img_caption", "").strip()
            parts.append(f"[Figure: {caption}]" if caption else "[Figure]")
        elif btype == "table":
            body = block.get("table_body", "").strip()
            caption = block.get("table_caption", "").strip()
            parts.append(f"[Table: {caption}]\n{body}" if caption else body)
        elif btype == "equation":
            eq = block.get("text", "").strip()
            parts.append(f"[Equation: {eq}]" if eq else "[Equation]")
    return "\n\n".join(p for p in parts if p)


def slice_section(content_list: list, hint: str) -> list:
    """Return the blocks belonging to the section whose heading best matches hint.

    Falls back to the full content_list when no heading scores above the similarity
    threshold (covers papers without explicit section structure).
    """
    headings = [(i, block) for i, block in enumerate(content_list) if _is_heading(block)]
    if not headings:
        return content_list

    hint_norm = _normalize(hint)
    scores = [
        (i, difflib.SequenceMatcher(None, hint_norm, _normalize(block["text"])).ratio())
        for i, block in headings
    ]
    best_idx, best_score = max(scores, key=lambda x: x[1])
    if best_score < 0.3:
        return content_list

    start_level = _heading_level(content_list[best_idx])
    end = len(content_list)
    for i in range(best_idx + 1, len(content_list)):
        if _is_heading(content_list[i]) and _heading_level(content_list[i]) <= start_level:
            end = i
            break
    return content_list[best_idx:end]

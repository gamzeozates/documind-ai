import re

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str) -> list[int]:
    """Extract numbered citations from an answer."""

    matches = CITATION_PATTERN.findall(answer)

    return [int(match) for match in matches]


def validate_citations(
    answer: str,
    sources: list[dict],
    require_citation: bool = False,
) -> dict:
    """Validate citation IDs and optionally require citations."""

    cited_ids = extract_citations(answer)

    valid_source_ids = {source["id"] for source in sources}

    invalid_ids = sorted(set(cited_ids) - valid_source_ids)

    missing_citations = require_citation and bool(sources) and not cited_ids

    return {
        "cited_ids": sorted(set(cited_ids)),
        "invalid_ids": invalid_ids,
        "missing_citations": missing_citations,
        "is_valid": (len(invalid_ids) == 0 and not missing_citations),
    }

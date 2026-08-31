"""Deterministic protein-sequence correspondence used by paired-structure analyses."""

from __future__ import annotations

from typing import Literal

from Bio.Align import PairwiseAligner

MAX_ENUMERATED_OPTIMAL_ALIGNMENTS = 10_000


def _aligned_pairs(alignment) -> tuple[tuple[int, int], ...]:
    return tuple(
        (reference_index, mobile_index)
        for (reference_start, reference_end), (mobile_start, mobile_end) in zip(
            alignment.aligned[0], alignment.aligned[1], strict=True
        )
        for reference_index, mobile_index in zip(
            range(int(reference_start), int(reference_end)),
            range(int(mobile_start), int(mobile_end)),
            strict=True,
        )
    )


def align_protein_sequences(
    reference_sequence: str,
    mobile_sequence: str,
    *,
    mode: Literal["semiglobal", "local"] = "semiglobal",
    max_optimal_alignments: int = MAX_ENUMERATED_OPTIMAL_ALIGNMENTS,
) -> tuple[list[tuple[int, int]], dict[str, float | int]]:
    """Align two coordinate-derived sequences with one frozen scoring scheme."""
    reference = "".join(str(reference_sequence).split()).upper()
    mobile = "".join(str(mobile_sequence).split()).upper()
    if not reference or not mobile:
        raise ValueError("Protein sequences must be non-empty.")
    if isinstance(max_optimal_alignments, bool) or int(max_optimal_alignments) != max_optimal_alignments:
        raise ValueError("max_optimal_alignments must be a positive integer.")
    if max_optimal_alignments <= 0:
        raise ValueError("max_optimal_alignments must be a positive integer.")

    aligner = PairwiseAligner()
    aligner.mode = "local" if mode == "local" else "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    if mode == "semiglobal":
        aligner.end_gap_score = 0.0
    alignments = aligner.align(reference, mobile)
    try:
        optimal_alignment_count = len(alignments)
    except OverflowError as exc:
        raise ValueError("The number of optimal sequence alignments is too large to characterize reliably.") from exc
    if optimal_alignment_count > max_optimal_alignments:
        raise ValueError(
            "Sequence correspondence is too ambiguous to characterize exhaustively: "
            f"{optimal_alignment_count} optimal alignments exceed the limit of {max_optimal_alignments}."
        )
    candidates = []
    for alignment in alignments:
        pairs = _aligned_pairs(alignment)
        identical = sum(reference[left] == mobile[right] for left, right in pairs)
        candidates.append(
            {
                "pairs": pairs,
                "identical": identical,
                "aligned": len(pairs),
                "gap_openings": max(0, len(alignment.aligned[0]) - 1),
                "score": float(alignment.score),
            }
        )
    if not candidates:
        return [], {
            "aligned_residue_count": 0,
            "alignment_identity": float("nan"),
            "reference_coverage": 0.0,
            "mobile_coverage": 0.0,
            "alignment_score": 0.0,
            "optimal_alignment_count": 0,
            "optimal_correspondence_count": 0,
            "consensus_pair_count": 0,
            "selected_pair_consensus_fraction": float("nan"),
            "selected_alignment_rule": (
                "maximum exact matches, aligned residues, and compactness; then lexicographic correspondence"
            ),
        }

    by_correspondence = {candidate["pairs"]: candidate for candidate in candidates}
    unique_candidates = list(by_correspondence.values())
    selected = min(
        unique_candidates,
        key=lambda candidate: (
            -int(candidate["identical"]),
            -int(candidate["aligned"]),
            int(candidate["gap_openings"]),
            candidate["pairs"],
        ),
    )
    selected_pairs = selected["pairs"]
    correspondence_sets = [set(candidate["pairs"]) for candidate in unique_candidates]
    consensus = set.intersection(*correspondence_sets)
    aligned = int(selected["aligned"])
    return list(selected_pairs), {
        "aligned_residue_count": aligned,
        "alignment_identity": int(selected["identical"]) / aligned if aligned else float("nan"),
        "reference_coverage": aligned / len(reference),
        "mobile_coverage": aligned / len(mobile),
        "alignment_score": float(selected["score"]),
        "optimal_alignment_count": int(optimal_alignment_count),
        "optimal_correspondence_count": int(len(unique_candidates)),
        "consensus_pair_count": int(len(consensus)),
        "selected_pair_consensus_fraction": len(consensus) / aligned if aligned else float("nan"),
        "selected_alignment_rule": (
            "maximum exact matches, aligned residues, and compactness; then lexicographic correspondence"
        ),
    }


__all__ = ["MAX_ENUMERATED_OPTIMAL_ALIGNMENTS", "align_protein_sequences"]

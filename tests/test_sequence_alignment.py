import pytest

from topoppi.sequence_alignment import align_protein_sequences


def test_semiglobal_alignment_preserves_fragmented_coordinate_sequence():
    observed = "HRYTPHAQRSTTPNV"
    full_segment = "HRYSTPHAFTFNTSSPSSEGSLSQRQRSTSTPNVHM"

    pairs, report = align_protein_sequences(observed, full_segment)

    assert len(pairs) == len(observed)
    assert report["alignment_identity"] == 1.0
    assert report["reference_coverage"] == 1.0


def test_semiglobal_alignment_does_not_count_unaligned_terminal_flanks():
    pairs, report = align_protein_sequences("ACDEFG", "MMMMACDEFGKKKK")

    assert pairs == [(index, index + 4) for index in range(6)]
    assert report["alignment_identity"] == 1.0
    assert report["reference_coverage"] == 1.0
    assert report["mobile_coverage"] == 6 / 14


def test_local_alignment_reports_an_empty_correspondence_without_positive_score():
    pairs, report = align_protein_sequences("AAAA", "RRRR", mode="local")

    assert pairs == []
    assert report["aligned_residue_count"] == 0
    assert report["reference_coverage"] == 0.0
    assert report["mobile_coverage"] == 0.0


def test_tied_optima_are_reported_and_resolved_deterministically():
    first_pairs, first_report = align_protein_sequences("AAAA", "AAAAA")
    second_pairs, second_report = align_protein_sequences("AAAA", "AAAAA")

    assert first_pairs == second_pairs
    assert first_report == second_report
    assert first_report["optimal_alignment_count"] > 1
    assert first_report["optimal_correspondence_count"] > 1
    assert first_report["selected_pair_consensus_fraction"] < 1.0


def test_excessive_alignment_ambiguity_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="too ambiguous to characterize exhaustively"):
        align_protein_sequences("AAAA", "AAAAA", max_optimal_alignments=1)

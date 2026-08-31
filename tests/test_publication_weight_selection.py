from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "tools" / "publication" / "select_residue_aware_optcuts_weight.py"
    spec = importlib.util.spec_from_file_location("test_select_residue_aware_optcuts_weight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SELECT = _load_script()


def _load_prepare_script():
    path = Path(__file__).parents[1] / "tools" / "publication" / "prepare_residue_aware_optcuts_weight_study.py"
    spec = importlib.util.spec_from_file_location("test_prepare_residue_aware_optcuts_weight_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load_prepare_script()


def _quality(value: float):
    return {
        "domain_hashes": ["domain-a"],
        "residue_footprint_fragmentation": {
            "objective_weighted_fragmentation": value,
        },
        "symmetric_dirichlet": {"mean": 2.0 + value},
        "seam": {"seam_length_3d_normalized": value},
        "flip_rate": 0.0,
    }


def _row(name: str, *, complete: bool = True, baseline: float = 0.5, treatment: float = 0.4):
    methods = (
        {
            SELECT.BASELINE: _quality(baseline),
            SELECT.TREATMENT: _quality(treatment),
        }
        if complete
        else {}
    )
    return {
        "manifest_record_id": name,
        "pdb": name,
        "input_sha256": f"hash-{name}",
        "chain_selection": {"chain_a": "A", "chain_b": "B"},
        "family_id": f"family-{name}",
        "sequence_cluster_a": f"cluster-a-{name}",
        "sequence_cluster_b": f"cluster-b-{name}",
        "analysis_split": "development",
        "residue_aware_comparison_domain": {"complete": complete},
        "residue_aware_pair_quality": {
            "complete": complete,
            "domain_signature": f"signature-{name}",
            "arms": {
                SELECT.BASELINE: {
                    "domain_complete": complete,
                    "metric_finite": complete,
                    "globally_injective": complete,
                    "usable": complete,
                },
                SELECT.TREATMENT: {
                    "domain_complete": complete,
                    "metric_finite": complete,
                    "globally_injective": complete,
                    "usable": complete,
                },
            },
            "methods": methods,
        },
    }


def test_indexed_pair_rows_separates_attempted_and_complete_domains():
    report = {"files": [_row("shared"), _row("failed", complete=False)]}

    attempted, complete = SELECT.indexed_pair_rows(report)

    assert len(attempted) == 2
    assert len(complete) == 1
    assert next(iter(complete))[0] == "shared"


def test_indexed_pair_rows_rejects_nonfinite_selection_endpoint():
    row = _row("bad")
    row["residue_aware_pair_quality"]["methods"][SELECT.TREATMENT]["symmetric_dirichlet"]["mean"] = float("nan")

    with pytest.raises(ValueError, match="Non-finite weight-selection endpoint"):
        SELECT.indexed_pair_rows({"files": [row]})


def test_baseline_fingerprint_detects_domain_or_endpoint_changes():
    _, first_rows = SELECT.indexed_pair_rows({"files": [_row("same")]})
    _, second_rows = SELECT.indexed_pair_rows({"files": [_row("same")]})
    first = next(iter(first_rows.values()))
    second = next(iter(second_rows.values()))
    assert SELECT.baseline_fingerprint(first) == SELECT.baseline_fingerprint(second)

    second[SELECT.BASELINE]["symmetric_dirichlet"]["mean"] += 1e-12
    assert SELECT.baseline_fingerprint(first) != SELECT.baseline_fingerprint(second)


def test_complete_identity_intersection_excludes_candidate_specific_successes():
    _, first = SELECT.indexed_pair_rows({"files": [_row("shared"), _row("only-first")]})
    _, second = SELECT.indexed_pair_rows({"files": [_row("shared"), _row("only-second")]})

    common = set.intersection(set(first), set(second))

    assert {identity[0] for identity in common} == {"shared"}


def test_pair_usability_requires_both_exact_arms_to_be_globally_usable():
    usable = _row("usable")
    invalid = _row("invalid")
    invalid["residue_aware_pair_quality"]["arms"][SELECT.TREATMENT]["usable"] = False

    assert SELECT.pair_is_usable(usable)
    assert not SELECT.pair_is_usable(invalid)


def test_complete_finite_rate_is_distinct_from_global_injectivity():
    invalid = _row("invalid")
    arm = invalid["residue_aware_pair_quality"]["arms"][SELECT.TREATMENT]
    arm["globally_injective"] = False
    arm["usable"] = False

    assert SELECT.pair_is_complete_finite(invalid)
    assert not SELECT.pair_is_usable(invalid)


@pytest.mark.parametrize("invalid", [[float("nan")], [float("inf")], [0.0], [-1.0], []])
def test_weight_grid_requires_finite_positive_values(invalid):
    with pytest.raises(ValueError, match="finite and positive"):
        PREPARE.validated_weights(invalid)


def test_weight_grid_is_sorted_and_deduplicated():
    assert PREPARE.validated_weights([5.0, 1.0, 5.0]) == [1.0, 5.0]

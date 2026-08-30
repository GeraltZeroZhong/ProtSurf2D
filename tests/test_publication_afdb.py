from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from topoppi.benchmarking.manifest_metadata import (
    ipsae_confidence_stratum,
    observed_sequence_metadata,
    plddt_confidence_stratum,
)
from topoppi.io.afdb_download import (
    download_sidecar_path,
    project_uniprot_intervals_to_coordinates,
    validated_cached_download,
)
from topoppi.io.io_loader import PDBLoader
from topoppi.io.pdb_records import selected_protein_atom_lines


def _load_script(name: str):
    path = Path(__file__).parents[1] / "tools" / "publication" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATCH = _load_script("match_afdb_complexes")
DOWNLOAD = _load_script("download_afdb_matches")
MONOMER = _load_script("build_afdb_monomer_replacements")
PAIRED_QC = _load_script("audit_paired_structures")
STRATIFY = _load_script("stratify_afdb_paired_geometry")
COORDINATE_AUDIT = _load_script("audit_manifest_coordinates")
SUBSET = _load_script("select_benchmark_subset")
STAGE = _load_script("stage_manifest_inputs")
FORMAL = _load_script("prepare_formal_benchmarks")
CLUSTER = _load_script("cluster_pdbbind_manifest")
PDBBIND = _load_script("prepare_pdbbind_r1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "mean_pLDDT_low_lt_70"),
        (69.999, "mean_pLDDT_low_lt_70"),
        (70.0, "mean_pLDDT_medium_ge_70_lt_90"),
        (89.999, "mean_pLDDT_medium_ge_70_lt_90"),
        (90.0, "mean_pLDDT_high_ge_90"),
        (100.0, "mean_pLDDT_high_ge_90"),
    ],
)
def test_plddt_confidence_stratum_has_explicit_boundaries(value, expected):
    assert plddt_confidence_stratum(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "ipSAE_low_lt_0.50"),
        (0.499, "ipSAE_low_lt_0.50"),
        (0.5, "ipSAE_medium_ge_0.50_lt_0.70"),
        (0.699, "ipSAE_medium_ge_0.50_lt_0.70"),
        (0.7, "ipSAE_high_ge_0.70"),
        (1.0, "ipSAE_high_ge_0.70"),
    ],
)
def test_ipsae_confidence_stratum_has_explicit_boundaries(value, expected):
    assert ipsae_confidence_stratum(value) == expected


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (plddt_confidence_stratum, -0.001),
        (plddt_confidence_stratum, 100.001),
        (plddt_confidence_stratum, float("nan")),
        (ipsae_confidence_stratum, -0.001),
        (ipsae_confidence_stratum, 1.001),
        (ipsae_confidence_stratum, float("inf")),
    ],
)
def test_confidence_strata_reject_invalid_values(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_afdb_dimer_manifest_does_not_conflate_plddt_with_ipsae():
    metadata = DOWNLOAD.confidence_manifest_metadata(76.0, 0.61)

    assert metadata == {
        "confidence_metric": "plddt_bfactor",
        "confidence_source": "AlphaFold DB model PDB B-factor field",
        "confidence_threshold": 70.0,
        "confidence_stratum": "mean_pLDDT_medium_ge_70_lt_90",
        "afdb_ipsae_stratum": "ipSAE_medium_ge_0.50_lt_0.70",
    }


def test_coordinate_audit_rejects_a_missing_analysis_split():
    row = {
        "analysis_split": "",
        "cluster_id": "component",
        "family_id": "family",
        "sequence_cluster_a": "sequence-a",
        "sequence_cluster_b": "sequence-b",
        "analysis_split_component_id": "component",
        "analysis_split_basis": "frozen-components",
    }

    with pytest.raises(ValueError, match="analysis_split"):
        COORDINATE_AUDIT.validate_dependency_splits([row])


def test_coordinate_audit_rejects_a_dependency_split_across_components():
    shared = {
        "analysis_split": "test",
        "analysis_split_basis": "frozen-components",
        "family_id": "family",
        "sequence_cluster_a": "sequence-a",
        "sequence_cluster_b": "sequence-b",
    }
    rows = [
        {**shared, "cluster_id": "cluster", "analysis_split_component_id": "component-1"},
        {**shared, "cluster_id": "cluster", "analysis_split_component_id": "component-2"},
    ]

    with pytest.raises(ValueError, match="multiple split components"):
        COORDINATE_AUDIT.validate_dependency_splits(rows)


def test_coordinate_audit_requires_explicit_structure_type():
    result = COORDINATE_AUDIT.audit_coordinate(
        (
            "experimental",
            {
                "record_id": "record-1",
                "structure_path": "unused.pdb",
            },
        )
    )

    assert result["status"] == "failed"
    assert "structure_type" in result["reason"]


def test_coordinate_audit_binds_declared_plddt_summary_and_stratum():
    loader = PDBLoader(Path(__file__).parent / "fixtures" / "tiny_complex.pdb")
    sequences = observed_sequence_metadata(
        COORDINATE_AUDIT.chain_sequence(loader, "A"),
        COORDINATE_AUDIT.chain_sequence(loader, "B"),
    )
    structure_path = Path(__file__).parent / "fixtures" / "tiny_complex.pdb"
    row = {
        "record_id": "predicted-1",
        "structure_path": str(structure_path),
        "input_sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        "chain_a": "A",
        "chain_b": "B",
        "structure_type": "predicted",
        "confidence_metric": "plddt_bfactor",
        **sequences,
    }
    baseline = COORDINATE_AUDIT.audit_coordinate(("predicted", row))
    assert baseline["status"] == "passed"
    declared = {
        **row,
        "confidence_threshold": "70",
        "confidence_stratum": baseline["plddt_confidence_stratum"],
        "crop_plddt_atom_count": str(baseline["plddt_atom_count"]),
        "crop_plddt_atom_minimum": str(baseline["plddt_atom_minimum"]),
        "crop_plddt_atom_mean": str(baseline["plddt_atom_weighted_mean"]),
        "crop_plddt_atom_maximum": str(baseline["plddt_atom_maximum"]),
        "crop_plddt_residue_count": str(baseline["plddt_residue_count"]),
        "crop_plddt_residue_minimum": str(baseline["plddt_minimum"]),
        "crop_plddt_residue_mean": str(baseline["plddt_mean"]),
        "crop_plddt_residue_maximum": str(baseline["plddt_maximum"]),
    }
    assert COORDINATE_AUDIT.audit_coordinate(("predicted", declared))["status"] == "passed"

    wrong_mean = {**declared, "crop_plddt_residue_mean": "99"}
    result = COORDINATE_AUDIT.audit_coordinate(("predicted", wrong_mean))
    assert result["status"] == "failed"
    assert "crop_plddt_residue_mean differs" in result["reason"]

    wrong_stratum = {**declared, "confidence_stratum": "mean_pLDDT_high_ge_90"}
    result = COORDINATE_AUDIT.audit_coordinate(("predicted", wrong_stratum))
    assert result["status"] == "failed"
    assert "confidence_stratum differs" in result["reason"]


@pytest.mark.parametrize("module", [SUBSET, STAGE])
def test_publication_selection_tools_reject_a_missing_analysis_split(module):
    with pytest.raises(ValueError, match="analysis_split"):
        module.analysis_split({})


@pytest.mark.parametrize(
    "row",
    [
        {"chain_a_residue_count": "10.5", "chain_b_residue_count": "20"},
        {"chain_a_residue_count": "nan", "chain_b_residue_count": "20"},
        {"chain_a_residue_count": "-1", "chain_b_residue_count": "20"},
    ],
)
def test_subset_size_requires_integral_positive_residue_counts(row):
    with pytest.raises(ValueError, match="finite positive integers"):
        SUBSET.structure_size(row)


def test_formal_protocol_requires_a_single_expected_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "record_id,include,analysis_split\nr1,1,test\nr2,0,development\n",
        encoding="utf-8",
    )

    assert FORMAL.require_manifest_split(manifest, "test") == 1
    with pytest.raises(ValueError, match="only included development rows"):
        FORMAL.require_manifest_split(manifest, "development")


def test_formal_protocol_recomputes_the_one_standard_error_rule(tmp_path):
    records = []
    for weight, effect, standard_error in ((1.0, 0.10, 0.01), (5.0, 0.12, 0.03)):
        report = tmp_path / f"report-{weight}.json"
        report.write_text("{}", encoding="utf-8")
        records.append(
            {
                "weight": weight,
                "eligible": True,
                "report": str(report),
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "primary": {
                    "mean_cluster_difference": effect,
                    "primary_standard_error_difference": standard_error,
                },
            }
        )

    selected = FORMAL.recompute_selected_weight({"records": records})

    assert selected == (1.0, "positive_effect_one_standard_error_rule", 0.12, 0.03)
    (tmp_path / "report-1.0.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FORMAL.recompute_selected_weight({"records": records})


def test_formal_protocol_rejects_unavailable_one_standard_error(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    records = [
        {
            "weight": 1.0,
            "eligible": True,
            "report": str(report),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "primary": {
                "mean_cluster_difference": 0.1,
                "primary_standard_error_difference": float("nan"),
            },
        }
    ]

    with pytest.raises(ValueError, match="dependence-aware standard error"):
        FORMAL.recompute_selected_weight({"records": records})


def test_formal_quality_config_separates_worker_and_invocation_timeouts(tmp_path):
    config = FORMAL.quality_config(
        input_folder=tmp_path,
        manifest=tmp_path / "manifest.csv",
        output_root=tmp_path / "output",
        binary=tmp_path / "OptCuts_bin",
        binary_sha256="0" * 64,
        weight=5.0,
        workers=8,
        expected_git_commit="1" * 40,
        coordinate_audit=tmp_path / "coordinate-audit.json",
        coordinate_audit_sha256="2" * 64,
        worker_timeout_sec=7200.0,
        optcuts_timeout_sec=600.0,
        include_topology_ablation=True,
    )

    assert config["worker_timeout_sec"] == 7200.0
    assert config["optcuts"]["timeout_sec"] == 600.0
    assert config["include_topology_ablation"] is True


def test_formal_protocol_validates_paired_inference_metadata_before_execution(tmp_path):
    fields = [
        "record_id",
        "paired_record_id",
        "paired_experimental_record_id",
        "paired_reference_record_id",
        "cluster_id",
        "family_id",
        "sequence_cluster_a",
        "sequence_cluster_b",
        "analysis_split",
        "analysis_split_component_id",
        "analysis_split_basis",
        "experimental_methods_json",
        "experimental_method_group",
        "experimental_method_contains_nmr",
        "inference_sequence_cluster_a",
        "inference_sequence_cluster_b",
        "inference_family_id",
        "inference_dependency_basis",
    ]
    shared = {
        "cluster_id": "cluster",
        "family_id": "family",
        "sequence_cluster_a": "sequence-a",
        "sequence_cluster_b": "sequence-b",
        "analysis_split": "test",
        "analysis_split_component_id": "component",
        "analysis_split_basis": STRATIFY.ANALYSIS_SPLIT_BASIS,
        "experimental_methods_json": '["X-RAY DIFFRACTION"]',
        "experimental_method_group": "x_ray_diffraction",
        "experimental_method_contains_nmr": "False",
        "inference_sequence_cluster_a": "pdep-a",
        "inference_sequence_cluster_b": "pdep-b",
        "inference_family_id": "pifam_pdep-a_pdep-b",
        "inference_dependency_basis": STRATIFY.INFERENCE_DEPENDENCY_BASIS,
    }
    experimental = tmp_path / "experimental.csv"
    predicted = tmp_path / "predicted.csv"
    STAGE.write_csv_atomic(
        experimental,
        [{**shared, "record_id": "experimental-1"}],
        fields,
    )
    STAGE.write_csv_atomic(
        predicted,
        [
            {
                **shared,
                "record_id": "predicted-1",
                "paired_record_id": "pair-1",
                "paired_experimental_record_id": "experimental-1",
                "paired_reference_record_id": "experimental-1",
            }
        ],
        fields,
    )

    assert FORMAL.validate_paired_protocol_manifests(experimental, [predicted]) == {str(predicted.resolve()): 1}
    rows = FORMAL.included_manifest_rows(experimental)
    rows[0]["inference_sequence_cluster_a"] = ""
    STAGE.write_csv_atomic(experimental, rows, fields)
    with pytest.raises(ValueError, match="inference_sequence_cluster_a"):
        FORMAL.validate_paired_protocol_manifests(experimental, [predicted])


def test_component_split_requires_two_independent_homology_components():
    with pytest.raises(ValueError, match="at least two homology components"):
        CLUSTER.choose_development_components({"only": 12}, 0.2, 7)


def test_component_split_uses_the_nearest_reachable_size():
    selected = CLUSTER.choose_development_components({"large": 8, "small": 2}, 0.2, 7)

    assert selected == {"small"}


def test_mmseqs_clustering_command_requests_single_step_exhaustive_candidates(tmp_path):
    args = Namespace(
        mmseqs=Path("mmseqs"),
        fasta=tmp_path / "input.fasta",
        output_dir=tmp_path,
        min_sequence_identity=0.3,
        coverage=0.8,
        sensitivity=7.5,
        maximum_evalue=1e-3,
        threads=8,
    )

    command = CLUSTER.mmseqs_command(args, tmp_path / "clusters", 5590)

    assert command[command.index("--cluster-mode") + 1] == "1"
    assert command[command.index("--single-step-clustering") + 1] == "1"
    assert command[command.index("--max-seqs") + 1] == "5590"
    assert command[command.index("-s") + 1] == "7.5"
    assert command[command.index("--cov-mode") + 1] == "0"


def test_rcsb_experiment_metadata_preserves_methods_and_resolution():
    record = PDBBIND.experiment_record_from_graphql(
        {
            "rcsb_id": "1ABC",
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": [2.1]},
        }
    )

    assert record["pdb_id"] == "1abc"
    assert record["experimental_methods"] == ["X-RAY DIFFRACTION"]
    assert record["resolution_combined_angstrom"] == [2.1]
    assert record["resolution_angstrom"] == 2.1
    assert record["experimental_method_group"] == "x_ray_diffraction"
    assert record["experimental_method_contains_nmr"] is False


def test_rcsb_experiment_metadata_marks_nmr_hybrid_without_collapsing_methods():
    record = PDBBIND.normalized_experiment_record(
        pdb_id="2xyz",
        methods=["solution nmr", "solution scattering"],
        resolutions=[],
        source="test",
    )

    assert record["experimental_methods"] == ["SOLUTION NMR", "SOLUTION SCATTERING"]
    assert record["resolution_angstrom"] == ""
    assert record["experimental_method_group"] == "multiple_or_other"
    assert record["experimental_method_contains_nmr"] is True


def test_rcsb_mmcif_fallback_extracts_experiment_metadata(tmp_path):
    path = tmp_path / "3pvm.cif"
    path.write_text(
        "data_3PVM\n_exptl.method 'X-RAY DIFFRACTION'\n_refine.ls_d_res_high 4.300\n",
        encoding="utf-8",
    )

    record = PDBBIND.experiment_record_from_mmcif(
        path,
        "3pvm",
        "https://files.rcsb.org/download/3PVM.cif",
    )

    assert record["experimental_methods"] == ["X-RAY DIFFRACTION"]
    assert record["resolution_combined_angstrom"] == [4.3]
    assert record["source"] == "rcsb_official_mmcif_fallback"
    assert record["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_rcsb_metadata_attachment_keeps_pdbbind_and_official_fields_separate():
    metadata = {
        "records": {
            "1abc": {
                "experimental_methods": ["SOLUTION NMR"],
                "resolution_combined_angstrom": [],
                "resolution_angstrom": "",
                "experimental_method_group": "solution_nmr",
                "experimental_method_contains_nmr": True,
                "source": "rcsb_data_api_graphql",
            }
        }
    }

    enriched = PDBBIND.attach_experiment_metadata(
        [{"pdb_id": "1abc", "resolution": "NMR"}],
        metadata,
    )[0]

    assert enriched["pdbbind_index_resolution_angstrom"] == "NMR"
    assert enriched["structure_method"] == "SOLUTION NMR"
    assert enriched["experimental_methods_json"] == '["SOLUTION NMR"]'
    assert enriched["rcsb_resolution_combined_angstrom_json"] == "[]"
    assert enriched["resolution_angstrom"] == ""
    assert enriched["resolution_angstrom_semantics"] == ("single_official_rcsb_resolution_combined_value_or_empty")


def test_monomer_source_metadata_removes_unrelated_complex_scores():
    source = {
        "record_id": "experimental:1abc",
        "afdb_accession_a": "P1",
        "afdb_accession_b": "P2",
        "afdb_match_status": "matched",
        "afdb_model_id": "AF-COMPLEX",
        "afdb_iptm": "0.8",
        "afdb_ipsae": "0.7",
        "afdb_ipsae_stratum": "ipSAE_high_ge_0.70",
        "afdb_pdockq": "0.6",
        "afdb_model_metadata_sha256": "digest",
    }

    cleaned = MONOMER.monomer_source_metadata(source)

    assert cleaned == {
        "record_id": "experimental:1abc",
        "afdb_accession_a": "P1",
        "afdb_accession_b": "P2",
    }


def test_sifts_selection_is_deterministic_and_json_serializable():
    payload = {
        "1abc": {
            "UniProt": {
                "P_SHORT": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 1,
                            "unp_end": 8,
                            "identity": 1.0,
                            "coverage": 1.0,
                            "start": {"author_residue_number": 1},
                            "end": {"author_residue_number": 8},
                        }
                    ]
                },
                "P_LONG": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 10,
                            "unp_end": 19,
                            "identity": 0.9,
                            "coverage": 0.8,
                            "start": {"author_residue_number": 2},
                            "end": {"author_residue_number": 11},
                        },
                        {
                            "chain_id": "A",
                            "unp_start": 20,
                            "unp_end": 24,
                            "identity": 0.8,
                            "coverage": 0.8,
                            "start": {"author_residue_number": 12},
                            "end": {"author_residue_number": 16},
                        },
                    ]
                },
            }
        }
    }

    selected = MATCH.select_chain_accession(
        payload,
        "1abc",
        "A",
        [(number, "") for number in range(1, 17)],
        "A" * 16,
        {
            "results": [
                {"primaryAccession": "P_SHORT", "sequence": {"value": "A" * 8}},
                {"primaryAccession": "P_LONG", "sequence": {"value": "M" * 9 + "A" * 15}},
            ]
        },
    )

    assert selected["accession"] == "P_LONG"
    assert selected["intervals"] == [(10, 24)]
    assert selected["candidate_accession_count"] == 2
    assert selected["mapped_residue_count"] == 15
    assert selected["experimental_sequence_coverage"] == 15 / 16
    assert json.loads(json.dumps(selected))["accession"] == "P_LONG"


def test_uniprot_search_cache_is_bound_to_the_candidate_limit(tmp_path):
    small = MATCH.uniprot_cache_path(tmp_path, "1abc", 25)
    complete = MATCH.uniprot_cache_path(tmp_path, "1abc", 500)

    assert small != complete
    assert complete.name == "1abc.size500.json"


def test_uniprot_search_rejects_a_candidate_set_at_the_request_limit():
    with pytest.raises(ValueError, match="may be truncated"):
        MATCH.require_complete_uniprot_search({"results": [{}, {}]}, 2)

    MATCH.require_complete_uniprot_search({"results": [{}]}, 2)


def test_afdb_matcher_binds_observed_residues_to_the_manifest_checksum(tmp_path):
    path = tmp_path / "experimental.pdb"
    path.write_text("ATOM\n", encoding="ascii")

    with pytest.raises(ValueError, match="checksum mismatch"):
        MATCH.observed_pair_residue_ids(("record", str(path), "0" * 64, "A", "B"))


def test_sifts_coverage_counts_observed_author_residues_and_never_exceeds_one():
    payload = {
        "1abc": {
            "UniProt": {
                "P_WIDE_UNIPROT": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 1,
                            "unp_end": 500,
                            "identity": 1.0,
                            "coverage": 1.0,
                            "start": {"author_residue_number": 10},
                            "end": {"author_residue_number": 10},
                        }
                    ]
                },
                "P_OBSERVED": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 30,
                            "unp_end": 32,
                            "identity": 0.95,
                            "coverage": 0.5,
                            "start": {"author_residue_number": 10},
                            "end": {
                                "author_residue_number": 11,
                                "author_insertion_code": "A",
                            },
                        }
                    ]
                },
            }
        }
    }

    selected = MATCH.select_chain_accession(
        payload,
        "1abc",
        "A",
        [(10, ""), (11, ""), (11, "A")],
        "AGS",
        {
            "results": [
                {"primaryAccession": "P_WIDE_UNIPROT", "sequence": {"value": "M" * 500}},
                {"primaryAccession": "P_OBSERVED", "sequence": {"value": "M" * 29 + "AGS"}},
            ]
        },
    )

    assert selected["accession"] == "P_OBSERVED"
    assert selected["mapped_residue_count"] == 3
    assert selected["experimental_sequence_coverage"] == 1.0
    assert selected["uniprot_interval_residue_count"] == 3


def test_exact_afdb_dimer_selection_respects_stoichiometry_and_ranking():
    candidates = [
        {
            "modelEntityId": "AF-low",
            "complexComposition": [
                {"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 1},
                {"identifierType": "uniprotAccession", "identifier": "P2", "stoichiometry": 1},
            ],
            "complexPredictionAccuracy_ipTM": 0.9,
            "complexPredictionAccuracy_ipSAE": 0.6,
        },
        {
            "modelEntityId": "AF-high",
            "complexComposition": [
                {"identifierType": "uniprotAccession", "identifier": "P2", "stoichiometry": 1},
                {"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 1},
            ],
            "complexPredictionAccuracy_ipTM": 0.8,
            "complexPredictionAccuracy_ipSAE": 0.7,
        },
        {
            "modelEntityId": "AF-wrong-stoichiometry",
            "complexComposition": [{"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 2}],
            "complexPredictionAccuracy_ipTM": 1.0,
            "complexPredictionAccuracy_ipSAE": 1.0,
        },
        {
            "modelEntityId": "AF-extra-non-uniprot-component",
            "complexComposition": [
                {"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 1},
                {"identifierType": "uniprotAccession", "identifier": "P2", "stoichiometry": 1},
                {"identifierType": "other", "identifier": "X", "stoichiometry": 1},
            ],
            "complexPredictionAccuracy_ipTM": 1.0,
            "complexPredictionAccuracy_ipSAE": 1.0,
        },
        {
            "modelEntityId": "AF-zero-stoichiometry",
            "complexComposition": [
                {"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 0},
                {"identifierType": "uniprotAccession", "identifier": "P2", "stoichiometry": 1},
            ],
            "complexPredictionAccuracy_ipTM": 1.0,
            "complexPredictionAccuracy_ipSAE": 1.0,
        },
        {
            "modelEntityId": "AF-fractional-stoichiometry",
            "complexComposition": [
                {"identifierType": "uniprotAccession", "identifier": "P1", "stoichiometry": 1.5},
                {"identifierType": "uniprotAccession", "identifier": "P2", "stoichiometry": 1},
            ],
            "complexPredictionAccuracy_ipTM": 1.0,
            "complexPredictionAccuracy_ipSAE": 1.0,
        },
        "malformed-candidate",
    ]

    selected, count = MATCH.select_exact_model("P1", "P2", [candidates, candidates])

    assert selected["modelEntityId"] == "AF-high"
    assert count == 2


def test_uniprot_fallback_selects_by_chain_sequence_and_maps_crop_interval():
    payload = {
        "results": [
            {
                "primaryAccession": "P_WRONG",
                "sequence": {"value": "MMMMMMMMMMMM", "length": 12},
            },
            {
                "primaryAccession": "P_RIGHT",
                "sequence": {"value": "TTACDEFGHIKQQ", "length": 13},
            },
        ]
    }

    selected = MATCH.select_sequence_matched_accession(
        payload,
        "ACDEFGHIK",
        minimum_aligned_residues=8,
        minimum_identity=0.9,
        minimum_chain_coverage=0.9,
    )

    assert selected["accession"] == "P_RIGHT"
    assert selected["intervals"] == [(3, 11)]
    assert selected["mapped_residue_count"] == 9
    assert selected["weighted_identity"] == 1.0
    assert selected["experimental_sequence_coverage"] == 1.0
    assert selected["mapping_method"] == "uniprot_pdb_xref_sequence_alignment"


def test_uniprot_fallback_rejects_short_local_matches():
    payload = {
        "results": [
            {
                "primaryAccession": "P_SHORT_MATCH",
                "sequence": {"value": "TTACDQQQQQQ", "length": 11},
            }
        ]
    }

    try:
        MATCH.select_sequence_matched_accession(
            payload,
            "ACDEFGHIK",
            minimum_aligned_residues=6,
            minimum_identity=0.9,
            minimum_chain_coverage=0.7,
        )
    except ValueError as exc:
        assert "passes the chain mapping thresholds" in str(exc)
    else:
        raise AssertionError("A short local match must not identify a partner chain")


def test_uniprot_fallback_rejects_a_positionally_ambiguous_repeat():
    payload = {
        "results": [
            {
                "primaryAccession": "P_REPEAT",
                "sequence": {"value": "ACDEFGTTTTACDEFG"},
            }
        ]
    }

    with pytest.raises(ValueError, match="pair_consensus"):
        MATCH.select_sequence_matched_accession(
            payload,
            "ACDEFG",
            minimum_aligned_residues=6,
            minimum_identity=1.0,
            minimum_chain_coverage=1.0,
            minimum_pair_consensus=0.9,
        )


def test_uniprot_fallback_can_trim_nonhomologous_construct_ends():
    observed = "ASAIVDYERKIQRIQQRVAELENTLKKLEHENRHLEQRAQELEQQIRAHAG"
    candidate = "M" * 355 + "ASVDYIRKLQREQQRAKELENRQKKLEHANRHLLLRIQELEMQARAH" + "K" * 20
    payload = {"results": [{"primaryAccession": "O75030", "sequence": {"value": candidate}}]}

    selected = MATCH.select_sequence_matched_accession(
        payload,
        observed,
        minimum_aligned_residues=10,
        minimum_identity=0.7,
        minimum_chain_coverage=0.7,
    )

    assert selected["accession"] == "O75030"
    assert selected["alignment_mode"] == "local"
    assert selected["experimental_sequence_coverage"] >= 0.9
    assert selected["weighted_identity"] >= 0.7


def test_monomer_metadata_prefers_canonical_gdm_record_over_community_model():
    payload = [
        {
            "entryId": "AF-P12345-F1",
            "modelEntityId": "AF-P12345-F1",
            "uniprotAccession": "P12345",
            "providerId": "GDM",
            "pdbUrl": "https://example.test/canonical.pdb",
        },
        {
            "entryId": "AF-0000000001",
            "modelEntityId": "AF-0000000001",
            "uniprotAccession": "P12345",
            "providerId": "ATBC",
            "pdbUrl": "https://example.test/community.pdb",
        },
    ]

    selected = MONOMER.prediction_record(payload, "P12345")

    assert selected["entryId"] == "AF-P12345-F1"


def test_monomer_builder_rejects_an_experimental_checksum_mismatch(tmp_path):
    path = tmp_path / "experimental.pdb"
    path.write_text("ATOM\n", encoding="ascii")

    with pytest.raises(ValueError, match="checksum mismatch"):
        MONOMER.verify_experimental_coordinate(path, "0" * 64)


def test_afdb_crop_uses_uniprot_number_intervals_and_renames_chains(tmp_path):
    raw = tmp_path / "raw.pdb"
    raw.write_text(
        "ATOM      1  CA  ALA X   1       0.000   0.000   0.000  1.00 61.00           C  \n"
        "ATOM      2  CA  GLY X   2       1.000   0.000   0.000  1.00 71.00           C  \n"
        "TER\n"
        "ATOM      3  CA  SER Y  10       0.000   2.000   0.000  1.00 81.00           C  \n"
        "ATOM      4  CA  THR Y  11       1.000   2.000   0.000  1.00 91.00           C  \n"
        "TER\nEND\n",
        encoding="ascii",
    )
    cropped = tmp_path / "cropped.pdb"

    report = DOWNLOAD.crop_model(raw, cropped, "X", "Y", [(2, 2)], [(10, 10)])
    model = PDBParser(QUIET=True).get_structure("crop", str(cropped))[0]

    assert list(model.child_dict) == ["A", "B"]
    assert [residue.id[1] for residue in model["A"]] == [2]
    assert [residue.id[1] for residue in model["B"]] == [10]
    assert report["crop_residue_count_a"] == 1
    assert report["crop_residue_count_b"] == 1
    assert report["sequence_a"] == "G"
    assert report["sequence_b"] == "S"
    assert report["sequence_a_sha256"] == hashlib.sha256(b"G").hexdigest()
    assert report["chain_a_residue_count"] == 1
    assert report["chain_b_residue_count"] == 1
    assert report["sequence_semantics"] == "observed_residues_in_current_coordinate_input"
    assert report["crop_plddt_atom_mean"] == 76.0
    assert report["crop_plddt_residue_count"] == 2
    assert report["crop_plddt_residue_mean"] == 76.0


def test_afdb_interval_projection_handles_number_offsets_and_model_truncation():
    coordinates, report = project_uniprot_intervals_to_coordinates(
        [(105, 110), (118, 125)],
        {
            "uniprotStart": 100,
            "uniprotEnd": 120,
            "sequenceStart": 1,
            "sequenceEnd": 21,
        },
    )

    assert coordinates == [(6, 11), (19, 21)]
    assert report["coordinate_residue_number_offset"] == -99
    assert report["requested_uniprot_residue_count"] == 14
    assert report["available_uniprot_residue_count"] == 9
    assert report["available_uniprot_fraction"] == 9 / 14
    assert report["requested_interval_truncated_to_model"] is True


def test_afdb_interval_projection_rejects_non_affine_metadata():
    with pytest.raises(ValueError, match="exact residue-number offset"):
        project_uniprot_intervals_to_coordinates(
            [(100, 110)],
            {
                "uniprotStart": 100,
                "uniprotEnd": 120,
                "sequenceStart": 1,
                "sequenceEnd": 20,
            },
        )


def test_predicted_manifest_separates_observed_and_paired_reference_sequences():
    row = {
        "record_id": "experimental:1abc",
        "structure_path": "/data/1abc.pdb",
        "input_sha256": "coordinate-hash",
        "chain_a": "X",
        "chain_b": "Y",
        "sequence_a": "AC",
        "sequence_b": "GG",
        "sequence_a_sha256": hashlib.sha256(b"AC").hexdigest(),
        "sequence_b_sha256": hashlib.sha256(b"GG").hexdigest(),
        "chain_a_residue_count": 2,
        "chain_b_residue_count": 2,
    }

    reference = MONOMER.paired_reference_metadata(row)
    observed = MONOMER.observed_sequence_metadata("ACD", "G")

    assert observed["sequence_a"] == "ACD"
    assert observed["chain_a_residue_count"] == 3
    assert reference["paired_reference_sequence_a"] == "AC"
    assert reference["paired_reference_chain_a_residue_count"] == 2
    assert reference["sequence_cluster_reference"] == "paired_experimental_observed_sequences"


def test_paired_qc_rejects_sequence_metadata_that_does_not_describe_coordinates():
    coordinate = np.zeros((1, 3), dtype=np.float64)
    chain_a = [PAIRED_QC.Residue("ALA", 1, "", coordinate, coordinate[0])]
    chain_b = [PAIRED_QC.Residue("GLY", 1, "", coordinate, coordinate[0])]
    row = {
        **MONOMER.observed_sequence_metadata("A", "G"),
        "sequence_a": "S",
    }

    with pytest.raises(ValueError, match="sequence_a differs"):
        PAIRED_QC.validate_observed_sequence_metadata(row, chain_a, chain_b, "Predicted")


def test_monomer_superposition_recovers_a_rigid_transform(tmp_path):
    experimental_path = tmp_path / "experimental.pdb"
    predicted_path = tmp_path / "predicted.pdb"
    residue_names = ("ALA", "GLY", "SER")

    def pdb_text(offset):
        lines = []
        for index, residue in enumerate(residue_names, start=1):
            x, y, z = (float(index) + offset[0], float(index % 2) + offset[1], offset[2])
            lines.append(
                f"ATOM  {index:5d}  CA  {residue} A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 85.00           C  "
            )
        return "\n".join(lines) + "\nTER\nEND\n"

    experimental_path.write_text(pdb_text((0.0, 0.0, 0.0)), encoding="ascii")
    predicted_path.write_text(pdb_text((10.0, -4.0, 3.0)), encoding="ascii")
    experimental = MONOMER.parse_protein_chains(experimental_path)["A"]
    predicted = MONOMER.parse_protein_chains(predicted_path)["A"]

    rotation, translation, report = MONOMER.superposition_transform(
        experimental,
        predicted,
        minimum_aligned_ca=3,
        minimum_identity=1.0,
    )

    assert report["alignment_ca_rmsd_angstrom"] < 1e-6
    assert report["reference_geometry_second_to_first_ratio"] > 0.0
    experimental_ca = [residue.ca for residue in experimental]
    predicted_ca = [residue.ca @ rotation + translation for residue in predicted]
    np.testing.assert_allclose(predicted_ca, experimental_ca, atol=1e-6)


def test_monomer_superposition_rejects_collinear_alignment_geometry():
    experimental = [MONOMER.ResidueAtoms("ALA", index, "", [], np.asarray([index, 0.0, 0.0])) for index in range(1, 5)]
    predicted = [
        MONOMER.ResidueAtoms("ALA", index, "", [], np.asarray([index + 10.0, 0.0, 0.0])) for index in range(1, 5)
    ]

    with pytest.raises(ValueError, match="too close to collinear"):
        MONOMER.superposition_transform(
            experimental,
            predicted,
            minimum_aligned_ca=3,
            minimum_identity=1.0,
        )


def test_monomer_superposition_records_but_does_not_select_on_chain_coverage():
    names = ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU")
    matched = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    experimental = [
        MONOMER.ResidueAtoms(
            name,
            index,
            "",
            [],
            matched[index - 1] if index <= len(matched) else np.asarray([float(index), 2.0, 1.0]),
        )
        for index, name in enumerate(names, start=1)
    ]
    predicted = [
        MONOMER.ResidueAtoms(name, index, "", [], coordinate + np.asarray([5.0, -2.0, 3.0]))
        for index, (name, coordinate) in enumerate(zip(names, matched, strict=False), start=1)
    ]

    _rotation, _translation, report = MONOMER.superposition_transform(
        experimental,
        predicted,
        minimum_aligned_ca=4,
        minimum_identity=1.0,
    )

    assert report["experimental_ca_coverage"] == 0.4
    assert report["alignment_ca_rmsd_angstrom"] < 1e-6


def test_afdb_coordinate_cache_requires_url_and_checksum_sidecar(tmp_path):
    coordinate = tmp_path / "model.pdb"
    coordinate.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C  \n",
        encoding="ascii",
    )
    url = "https://example.org/model.pdb"
    payload = {
        "url": url,
        "sha256": hashlib.sha256(coordinate.read_bytes()).hexdigest(),
        "size_bytes": coordinate.stat().st_size,
        "retrieved_at_utc": "2026-08-19T00:00:00Z",
    }
    download_sidecar_path(coordinate).write_text(json.dumps(payload), encoding="utf-8")

    assert validated_cached_download(url, coordinate) == payload
    assert validated_cached_download("https://example.org/other.pdb", coordinate) is None
    coordinate.write_text(coordinate.read_text(encoding="ascii") + "END\n", encoding="ascii")
    assert validated_cached_download(url, coordinate) is None


def test_publication_parsers_follow_pipeline_altloc_selection(tmp_path):
    path = tmp_path / "alternate_conformers.pdb"
    path.write_text(
        "ATOM      1  CA AALA A   1       1.000   0.000   0.000  0.40 10.00           C  \n"
        "ATOM      2  CA BALA A   1       2.000   0.000   0.000  0.60 20.00           C  \n"
        "ATOM      3  CB BALA A   1       3.000   0.000   0.000  0.60 30.00           C  \n"
        "ATOM      4  CA AGLY A   2       4.000   0.000   0.000  0.50 40.00           C  \n"
        "ATOM      5  CA BGLY A   2       5.000   0.000   0.000  0.50 50.00           C  \n"
        "TER\nEND\n",
        encoding="ascii",
    )

    lines = selected_protein_atom_lines(path)
    monomer = MONOMER.parse_protein_chains(path)["A"]
    paired = PAIRED_QC.parse_chains(path)["A"]
    pipeline_coordinates, pipeline_atoms = PDBLoader(path).get_chain_data("A")

    assert [line[16] for line in lines] == ["B", "B", "A"]
    assert [atom.get_altloc() for atom in pipeline_atoms] == ["B", "B", "A"]
    np.testing.assert_allclose(
        np.concatenate([residue.atoms for residue in paired]),
        pipeline_coordinates,
    )
    np.testing.assert_allclose([residue.ca for residue in monomer], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])


def test_publication_parsers_keep_recognized_modified_amino_acids(tmp_path):
    path = tmp_path / "modified.pdb"
    path.write_text(
        "HETATM    1  CA  MSE A   1       0.000   0.000   0.000  1.00 90.00           C  \n"
        "HETATM    2  SE  MSE A   1       1.000   0.000   0.000  1.00 90.00          SE  \n"
        "HETATM    3 ZN    ZN A   2       2.000   0.000   0.000  1.00 90.00          ZN  \n"
        "TER\nEND\n",
        encoding="ascii",
    )

    lines = selected_protein_atom_lines(path)
    monomer = MONOMER.parse_protein_chains(path)["A"]
    paired = PAIRED_QC.parse_chains(path)["A"]
    pipeline_coordinates, _pipeline_atoms = PDBLoader(path).get_chain_data("A")

    assert len(lines) == 2
    assert [residue.name for residue in monomer] == ["MSE"]
    assert MONOMER.AMINO_ACID_LETTERS[monomer[0].name] == "M"
    np.testing.assert_allclose(paired[0].atoms, pipeline_coordinates)
    loader = PDBLoader(path)
    assert MATCH._observed_chain_residue_ids(loader, "A") == [(1, "")]


def test_paired_parser_recognizes_digit_prefixed_hydrogen_without_element_column(tmp_path):
    path = tmp_path / "hydrogen.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00\n"
        "ATOM      2 1HG  ALA A   1       1.000   0.000   0.000  1.00 10.00\n"
        "TER\nEND\n",
        encoding="ascii",
    )

    residue = PAIRED_QC.parse_chains(path)["A"][0]

    assert residue.atoms.shape == (1, 3)
    assert len(selected_protein_atom_lines(path)) == 1


def test_monomer_plddt_summary_is_residue_weighted(tmp_path):
    path = tmp_path / "prediction.pdb"
    path.write_text(
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 60.00           C  \n"
        "ATOM      2  N   ALA A   2       1.000   0.000   0.000  1.00100.00           N  \n"
        "ATOM      3  CA  ALA A   2       2.000   0.000   0.000  1.00100.00           C  \n"
        "ATOM      4  C   ALA A   2       3.000   0.000   0.000  1.00100.00           C  \n"
        "TER\nEND\n",
        encoding="ascii",
    )
    residues = MONOMER.parse_protein_chains(path)["A"]

    report = MONOMER.residue_plddt_report(residues)

    assert report["crop_plddt_residue_count"] == 2
    assert report["crop_plddt_residue_mean"] == 80.0


def test_monomer_plddt_summary_rejects_inconsistent_atoms_within_a_residue(tmp_path):
    path = tmp_path / "prediction.pdb"
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 75.00           N  \n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 85.00           C  \n"
        "TER\nEND\n",
        encoding="ascii",
    )
    residues = MONOMER.parse_protein_chains(path)["A"]

    with pytest.raises(ValueError, match="inconsistent pLDDT"):
        MONOMER.residue_plddt_report(residues)


def test_paired_structure_qc_is_sequence_mapped_and_rigid_motion_invariant(tmp_path):
    experimental_path = tmp_path / "experimental.pdb"
    predicted_path = tmp_path / "predicted.pdb"

    def complex_text(translation):
        lines = []
        serial = 1
        for chain, y in (("A", 0.0), ("B", 3.0)):
            for number, residue in enumerate(("ALA", "GLY", "SER"), start=1):
                coordinate = np.array([float(number), y + float(number % 2), 0.0]) + translation
                lines.append(
                    f"ATOM  {serial:5d}  CA  {residue} {chain}{number:4d}    "
                    f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
                    "  1.00 90.00           C  "
                )
                serial += 1
            lines.append("TER")
        return "\n".join(lines) + "\nEND\n"

    experimental_path.write_text(complex_text(np.zeros(3)), encoding="ascii")
    predicted_path.write_text(complex_text(np.array([9.0, -4.0, 2.0])), encoding="ascii")
    experimental = PAIRED_QC.parse_chains(experimental_path)
    predicted = PAIRED_QC.parse_chains(predicted_path)
    map_a, _ = PAIRED_QC.align_residues(experimental["A"], predicted["A"])
    map_b, _ = PAIRED_QC.align_residues(experimental["B"], predicted["B"])

    contacts = PAIRED_QC.contact_comparison(
        experimental["A"],
        experimental["B"],
        predicted["A"],
        predicted["B"],
        map_a,
        map_b,
        3.1,
    )
    pose = PAIRED_QC.pose_comparison(
        experimental["A"],
        experimental["B"],
        predicted["A"],
        predicted["B"],
        map_a,
        map_b,
        3.1,
    )

    assert contacts["contact_recall_fnat"] == 1.0
    assert contacts["contact_precision"] == 1.0
    assert pose["receptor_fit_ca_rmsd_angstrom"] < 1e-6
    assert pose["ligand_ca_rmsd_after_receptor_fit_angstrom"] < 1e-6


def test_paired_qc_requires_identical_pair_id_sets():
    experimental = [{"paired_record_id": "pair-1"}, {"paired_record_id": "pair-2"}]
    predicted = [{"paired_record_id": "pair-1"}]

    with pytest.raises(ValueError, match="same paired_record_id set"):
        PAIRED_QC.paired_rows_by_id(experimental, predicted)


def test_paired_qc_requires_dependency_metadata_to_match():
    experimental = {
        "record_id": "experimental:1abc",
        "structure_path": "/data/1abc.pdb",
        "input_sha256": "coordinate-hash",
        "chain_a": "A",
        "chain_b": "B",
        "sequence_a": "A",
        "sequence_b": "G",
        "sequence_a_sha256": hashlib.sha256(b"A").hexdigest(),
        "sequence_b_sha256": hashlib.sha256(b"G").hexdigest(),
        "chain_a_residue_count": 1,
        "chain_b_residue_count": 1,
        "analysis_split": "test",
        "analysis_split_component_id": "component-1",
        "cluster_id": "cluster-1",
        "family_id": "family-1",
        "sequence_cluster_a": "sequence-a",
        "sequence_cluster_b": "sequence-b",
    }
    predicted = {
        **MONOMER.paired_reference_metadata(experimental),
        **{
            key: experimental[key]
            for key in (
                "analysis_split",
                "analysis_split_component_id",
                "cluster_id",
                "family_id",
                "sequence_cluster_a",
                "sequence_cluster_b",
            )
        },
        "paired_experimental_record_id": experimental["record_id"],
        "sequence_cluster_reference": "paired_experimental_observed_sequences",
    }
    predicted["family_id"] = "wrong-family"

    with pytest.raises(ValueError, match="family_id differs"):
        PAIRED_QC.validate_paired_reference_metadata(experimental, predicted)


def test_paired_contact_metrics_penalize_contacts_outside_the_sequence_correspondence():
    def residue(name: str, x: float) -> PAIRED_QC.Residue:
        coordinate = np.asarray([[x, 0.0, 0.0]], dtype=np.float64)
        return PAIRED_QC.Residue(name, int(x) + 1, "", coordinate, coordinate[0])

    experimental_a = [residue("ALA", value) for value in (0.0, 10.0, 20.0)]
    experimental_b = [residue("GLY", value) for value in (1.0, 11.0, 21.0)]
    predicted_a = experimental_a[:2]
    predicted_b = experimental_b[:2]

    report = PAIRED_QC.contact_comparison(
        experimental_a,
        experimental_b,
        predicted_a,
        predicted_b,
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        2.0,
    )

    assert report["experimental_contact_count_total"] == 3
    assert report["experimental_contact_count_comparable"] == 2
    assert report["experimental_contact_mapping_coverage"] == 2 / 3
    assert report["contact_recall_mapped_domain"] == 1.0
    assert report["contact_jaccard_mapped_domain"] == 1.0
    assert report["contact_recall_fnat"] == 2 / 3
    assert report["contact_jaccard"] == 2 / 3


def test_paired_interface_benchmark_retains_a_contactless_prediction():
    eligible, reason = PAIRED_QC.interface_benchmark_eligibility(
        {"predicted_contact_count_total": 0},
        {},
    )

    assert eligible
    assert reason == ""


def test_paired_interface_benchmark_excludes_role_ambiguous_afdb_homodimer():
    eligible, reason = PAIRED_QC.interface_benchmark_eligibility(
        {"predicted_contact_count_total": 12},
        {
            "structure_type": "afdb",
            "afdb_accession_a": "P12345",
            "afdb_accession_b": "P12345",
        },
    )

    assert not eligible
    assert reason == "homodimer_partner_role_is_not_identifiable"


def test_paired_geometry_strata_require_mapping_completeness():
    row = {
        "contact_jaccard": "0.5",
        "interface_ligand_ca_rmsd_after_receptor_fit_angstrom": "1.0",
        "predicted_cross_chain_clash_atom_fraction": "0.01",
        "experimental_contact_mapping_coverage": "1.0",
        "interface_ligand_ca_mapping_coverage": "1.0",
        "alignment_a_selected_pair_consensus_fraction": "1.0",
        "alignment_b_selected_pair_consensus_fraction": "1.0",
    }

    assert STRATIFY.geometry_stratum(row) == "high_fidelity"
    row["experimental_contact_mapping_coverage"] = "0.7"
    assert STRATIFY.geometry_stratum(row) == "moderate_fidelity"
    row["experimental_contact_mapping_coverage"] = "0.4"
    assert STRATIFY.geometry_stratum(row) == "geometry_stress_test"


def test_paired_geometry_strata_require_identifiable_sequence_correspondence():
    row = {
        "contact_jaccard": "0.8",
        "interface_ligand_ca_rmsd_after_receptor_fit_angstrom": "0.5",
        "predicted_cross_chain_clash_atom_fraction": "0.0",
        "experimental_contact_mapping_coverage": "1.0",
        "interface_ligand_ca_mapping_coverage": "1.0",
        "alignment_a_selected_pair_consensus_fraction": "0.899",
        "alignment_b_selected_pair_consensus_fraction": "1.0",
    }

    assert STRATIFY.geometry_stratum(row) == "geometry_stress_test"
    row["alignment_a_selected_pair_consensus_fraction"] = "0.9"
    assert STRATIFY.geometry_stratum(row) == "high_fidelity"


def test_prediction_dependencies_union_reference_homology_and_reused_accessions():
    rows = [
        {
            "paired_record_id": "pair-1",
            "analysis_split": "development",
            "analysis_split_component_id": "component-1",
            "analysis_split_basis": STRATIFY.ANALYSIS_SPLIT_BASIS,
            "sequence_cluster_a": "sequence-a1",
            "sequence_cluster_b": "sequence-b",
            "afdb_accession_a": "P_SHARED",
            "afdb_accession_b": "P_B",
        },
        {
            "paired_record_id": "pair-2",
            "analysis_split": "test",
            "analysis_split_component_id": "component-2",
            "analysis_split_basis": STRATIFY.ANALYSIS_SPLIT_BASIS,
            "sequence_cluster_a": "sequence-a2",
            "sequence_cluster_b": "sequence-c",
            "afdb_accession_a": "P_SHARED",
            "afdb_accession_b": "P_C",
        },
    ]

    enriched, summary = STRATIFY.add_prediction_dependencies(rows)

    assert enriched[0]["inference_sequence_cluster_a"] == enriched[1]["inference_sequence_cluster_a"]
    assert enriched[0]["inference_family_id"] != enriched[1]["inference_family_id"]
    assert summary["inference_sequence_cluster_cross_split_count"] == 1
    assert summary["inference_sequence_cluster_cross_component_count"] == 1

    with pytest.raises(RuntimeError, match="leak prediction-dependency"):
        STRATIFY.validate_prediction_dependency_splits(summary)


def test_split_reconciliation_keeps_reused_sources_in_one_partition():
    reference = [
        {"record_id": "r1", "cluster_id": "c1", "analysis_split": "development"},
        {"record_id": "r2", "cluster_id": "c2", "analysis_split": "test"},
        {"record_id": "r3", "cluster_id": "c3", "analysis_split": "test"},
    ]
    dependencies = [
        {"record_id": "r1", "afdb_accession_a": "P_SHARED", "afdb_accession_b": "P1"},
        {"record_id": "r2", "afdb_accession_a": "P_SHARED", "afdb_accession_b": "P2"},
        {"record_id": "r3", "afdb_accession_a": "P3", "afdb_accession_b": "P4"},
    ]

    assignments, summary = STRATIFY.reconcile_analysis_splits(
        reference,
        dependencies,
        development_fraction=1 / 3,
        seed=7,
    )

    assert assignments["r1"] == assignments["r2"]
    assert assignments["r3"][0] == "development"
    assert assignments["r1"][0] == "test"
    assert summary["analysis_split_component_count"] == 2
    assert summary["analysis_split_largest_component_structure_count"] == 2


def test_dependency_manifest_reader_unions_all_prediction_cohorts(tmp_path):
    monomer = tmp_path / "monomer.csv"
    dimer = tmp_path / "dimer.csv"
    monomer.write_text("record_id,afdb_accession_a\nr1,P1\n", encoding="utf-8")
    dimer.write_text("record_id,afdb_accession_a\nr2,P2\n", encoding="utf-8")

    rows, records = STRATIFY.read_dependency_manifests([monomer, dimer])

    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert [record["row_count"] for record in records] == [1, 1]
    assert [record["path"] for record in records] == [
        str(monomer.resolve()),
        str(dimer.resolve()),
    ]


def test_dependency_binding_rejects_stale_clusters_when_split_is_reused():
    reference = [
        {
            "record_id": "r1",
            "sequence_cluster_a": "sequence-a",
            "sequence_cluster_b": "sequence-b",
        }
    ]
    stale = [
        {
            "paired_reference_record_id": "r1",
            "sequence_cluster_a": "stale-sequence-a",
            "sequence_cluster_b": "sequence-b",
            "afdb_accession_a": "P1",
            "afdb_accession_b": "P2",
        }
    ]

    with pytest.raises(ValueError, match="changed sequence_cluster_a"):
        STRATIFY.bind_prediction_dependencies_to_reference(
            reference,
            stale,
            require_sequence_clusters=True,
        )


def test_dependency_binding_fills_clusters_from_authoritative_reference():
    reference = [
        {
            "record_id": "r1",
            "sequence_cluster_a": "sequence-a",
            "sequence_cluster_b": "sequence-b",
        }
    ]
    dependency = [
        {
            "paired_reference_record_id": "r1",
            "afdb_accession_a": "P1",
            "afdb_accession_b": "P2",
        }
    ]

    bound = STRATIFY.bind_prediction_dependencies_to_reference(
        reference,
        dependency,
        require_sequence_clusters=True,
    )

    assert bound[0]["sequence_cluster_a"] == "sequence-a"
    assert bound[0]["sequence_cluster_b"] == "sequence-b"


def test_split_reconciliation_resolves_predicted_rows_to_experimental_references():
    reference = [
        {"record_id": "r1", "cluster_id": "c1", "analysis_split": "development"},
        {"record_id": "r2", "cluster_id": "c2", "analysis_split": "test"},
        {"record_id": "r3", "cluster_id": "c3", "analysis_split": "test"},
    ]
    dependencies = [
        {
            "record_id": "predicted-1",
            "paired_reference_record_id": "r1",
            "paired_experimental_record_id": "r1",
            "afdb_accession_a": "P_SHARED",
            "afdb_accession_b": "P1",
        },
        {
            "record_id": "predicted-2",
            "paired_reference_record_id": "r2",
            "paired_experimental_record_id": "r2",
            "afdb_accession_a": "P_SHARED",
            "afdb_accession_b": "P2",
        },
    ]

    assignments, summary = STRATIFY.reconcile_analysis_splits(
        reference,
        dependencies,
        development_fraction=1 / 3,
        seed=7,
    )

    assert assignments["r1"] == assignments["r2"]
    assert summary["analysis_split_component_count"] == 2


def test_split_reconciliation_rejects_unbound_dependency_rows():
    reference = [{"record_id": "r1", "cluster_id": "c1", "analysis_split": "test"}]
    dependency = [{"record_id": "predicted-1", "afdb_accession_a": "P1"}]

    with pytest.raises(ValueError, match="unknown experimental record"):
        STRATIFY.reconcile_analysis_splits(
            reference,
            dependency,
            development_fraction=0.5,
            seed=7,
        )


def test_stratum_enrichment_preserves_zero_and_false_values():
    row = {"zero": 0, "false": False, "missing": None}

    assert STRATIFY.value_or_empty(row, "zero") == 0
    assert STRATIFY.value_or_empty(row, "false") is False
    assert STRATIFY.value_or_empty(row, "missing") == ""


def test_full_experimental_manifest_keeps_outcome_independent_dependencies():
    paired = {
        "inference_sequence_cluster_a": "pdep-a",
        "inference_family_id": "pifam-a-b",
        "paired_geometry_stratum": "geometry_stress_test",
        "paired_benchmark_eligible": False,
    }

    assert (
        STRATIFY.full_experimental_paired_value(
            paired,
            "inference_sequence_cluster_a",
            paired_is_eligible=False,
        )
        == "pdep-a"
    )
    assert (
        STRATIFY.full_experimental_paired_value(
            paired,
            "paired_benchmark_eligible",
            paired_is_eligible=False,
        )
        is False
    )
    assert (
        STRATIFY.full_experimental_paired_value(
            paired,
            "paired_geometry_stratum",
            paired_is_eligible=False,
        )
        == ""
    )


def test_geometry_strata_qc_is_bound_to_both_coordinate_inputs():
    predicted = [
        {
            "paired_record_id": "pair-1",
            "record_id": "predicted-1",
            "input_sha256": "predicted-sha",
        }
    ]
    experimental = [
        {
            "paired_record_id": "pair-1",
            "record_id": "experimental-1",
            "input_sha256": "experimental-sha",
        }
    ]
    qc = {
        "pair-1": {
            "predicted_record_id": "predicted-1",
            "experimental_record_id": "experimental-1",
            "predicted_input_sha256": "predicted-sha",
            "experimental_input_sha256": "experimental-sha",
        }
    }

    STRATIFY.validate_qc_bindings(predicted, experimental, qc)
    qc["pair-1"]["predicted_input_sha256"] = "stale-sha"
    with pytest.raises(ValueError, match="stale or belongs"):
        STRATIFY.validate_qc_bindings(predicted, experimental, qc)

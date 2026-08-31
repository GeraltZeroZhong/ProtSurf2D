#!/usr/bin/env python3
"""Attach frozen experimental/AFDB paired-geometry strata to manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from topoppi.benchmarking.manifest_metadata import (
    INFERENCE_DEPENDENCY_BASIS,
    INFERENCE_DEPENDENCY_FIELDS,
    inference_family_id,
)
from topoppi.file_utils import read_csv_rows, sha256_file, write_csv_atomic
from topoppi.json_utils import dump_json_atomic

INFERENCE_FIELDS = INFERENCE_DEPENDENCY_FIELDS
ANALYSIS_SPLIT_BASIS = "experimental_homology_and_reused_afdb_accession_component"
ALLOWED_ANALYSIS_SPLITS = {"development", "test", "exploratory"}


class DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            smaller, larger = sorted((root_left, root_right))
            self.parent[larger] = smaller


def add_prediction_dependencies(
    rows: list[dict[str, str]],
    dependency_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int | str]]:
    """Join experimental homology and reused AFDB accessions for inference."""

    nodes = set()
    links = []
    for row in dependency_rows if dependency_rows is not None else rows:
        for side in ("a", "b"):
            sequence_cluster = str(row.get(f"sequence_cluster_{side}") or "").strip()
            accession = str(row.get(f"afdb_accession_{side}") or "").strip()
            if not accession:
                continue
            if not sequence_cluster:
                raise ValueError("A prediction dependency has an accession but no sequence cluster.")
            sequence_node = f"experimental_sequence_cluster:{sequence_cluster}"
            accession_node = f"afdb_uniprot_accession:{accession}"
            nodes.update((sequence_node, accession_node))
            links.append((sequence_node, accession_node))
    for row in rows:
        for side in ("a", "b"):
            sequence_cluster = str(row.get(f"sequence_cluster_{side}") or "").strip()
            accession = str(row.get(f"afdb_accession_{side}") or "").strip()
            if not sequence_cluster or not accession:
                raise ValueError("Predicted rows require experimental sequence clusters and AFDB accessions.")
            sequence_node = f"experimental_sequence_cluster:{sequence_cluster}"
            accession_node = f"afdb_uniprot_accession:{accession}"
            nodes.update((sequence_node, accession_node))
            links.append((sequence_node, accession_node))

    disjoint = DisjointSet(nodes)
    for left, right in links:
        disjoint.union(left, right)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        members_by_root[disjoint.find(node)].append(node)
    component_id = {
        root: "pdep_" + hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }

    enriched = []
    for row in rows:
        partner_clusters = []
        for side in ("a", "b"):
            sequence_node = "experimental_sequence_cluster:" + str(row[f"sequence_cluster_{side}"]).strip()
            partner_clusters.append(component_id[disjoint.find(sequence_node)])
        enriched.append(
            {
                **row,
                "inference_sequence_cluster_a": partner_clusters[0],
                "inference_sequence_cluster_b": partner_clusters[1],
                "inference_family_id": inference_family_id(*partner_clusters),
                "inference_dependency_basis": INFERENCE_DEPENDENCY_BASIS,
            }
        )

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    cluster_components: dict[str, set[str]] = defaultdict(set)
    family_components: dict[str, set[str]] = defaultdict(set)
    split_bases: set[str] = set()
    for row in enriched:
        split = str(row.get("analysis_split") or "").strip().lower()
        component = str(row.get("analysis_split_component_id") or "").strip()
        split_basis = str(row.get("analysis_split_basis") or "").strip()
        if split not in ALLOWED_ANALYSIS_SPLITS or not component or not split_basis:
            raise ValueError(
                "Predicted rows require a valid analysis_split, analysis_split_component_id, "
                "and analysis_split_basis before dependency enrichment."
            )
        split_bases.add(split_basis)
        cluster_splits[str(row["inference_sequence_cluster_a"])].add(split)
        cluster_splits[str(row["inference_sequence_cluster_b"])].add(split)
        family_splits[str(row["inference_family_id"])].add(split)
        cluster_components[str(row["inference_sequence_cluster_a"])].add(component)
        cluster_components[str(row["inference_sequence_cluster_b"])].add(component)
        family_components[str(row["inference_family_id"])].add(component)
    if len(split_bases) != 1:
        raise ValueError("Predicted rows must use one frozen analysis_split_basis.")
    return enriched, {
        "inference_dependency_rule": (
            "prediction-set efficacy is aggregated by components that union experimental "
            "sequence-homology clusters with reused AFDB UniProt sources"
        ),
        "inference_sequence_cluster_count": len(set(component_id.values())),
        "inference_family_count": len({str(row["inference_family_id"]) for row in enriched}),
        "inference_sequence_cluster_cross_split_count": sum(len(splits) > 1 for splits in cluster_splits.values()),
        "inference_family_cross_split_count": sum(len(splits) > 1 for splits in family_splits.values()),
        "inference_sequence_cluster_cross_component_count": sum(
            len(components) > 1 for components in cluster_components.values()
        ),
        "inference_family_cross_component_count": sum(len(components) > 1 for components in family_components.values()),
    }


def validate_prediction_dependency_splits(summary: dict[str, int | str]) -> None:
    cluster_leaks = int(summary["inference_sequence_cluster_cross_split_count"])
    family_leaks = int(summary["inference_family_cross_split_count"])
    cluster_component_leaks = int(summary["inference_sequence_cluster_cross_component_count"])
    family_component_leaks = int(summary["inference_family_cross_component_count"])
    if cluster_leaks or family_leaks or cluster_component_leaks or family_component_leaks:
        raise RuntimeError(
            "Analysis splits or components leak prediction-dependency groups: "
            f"split_sequence_clusters={cluster_leaks}; split_families={family_leaks}; "
            f"component_sequence_clusters={cluster_component_leaks}; "
            f"component_families={family_component_leaks}."
        )


def dependency_reference_id(row: dict[str, str]) -> str:
    paired_ids = {
        str(row.get(field) or "").strip()
        for field in ("paired_reference_record_id", "paired_experimental_record_id")
        if str(row.get(field) or "").strip()
    }
    if len(paired_ids) > 1:
        raise ValueError("A dependency row names inconsistent experimental reference records.")
    if paired_ids:
        return paired_ids.pop()
    return str(row.get("record_id") or "").strip()


def bind_prediction_dependencies_to_reference(
    reference_rows: list[dict[str, str]],
    dependency_rows: list[dict[str, str]],
    *,
    require_sequence_clusters: bool = False,
) -> list[dict[str, str]]:
    """Bind every accession dependency to its authoritative experimental row."""

    record_ids = [str(row.get("record_id") or "").strip() for row in reference_rows]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Dependency-reference rows require unique, non-empty record_id values.")
    reference_by_id = dict(zip(record_ids, reference_rows, strict=True))
    accessions_by_record_side: dict[tuple[str, str], str] = {}
    bound = []
    link_count = 0
    for row in dependency_rows:
        record_id = dependency_reference_id(row)
        reference = reference_by_id.get(record_id)
        if reference is None:
            raise ValueError(f"Prediction dependency references an unknown experimental record: {record_id!r}.")
        normalized = dict(row)
        for side in ("a", "b"):
            field = f"sequence_cluster_{side}"
            dependency_cluster = str(row.get(field) or "").strip()
            reference_cluster = str(reference.get(field) or "").strip()
            if require_sequence_clusters and not reference_cluster:
                raise ValueError(f"Experimental dependency reference {record_id!r} lacks {field}.")
            if dependency_cluster and dependency_cluster != reference_cluster:
                raise ValueError(f"Prediction dependency {record_id!r} changed {field}.")
            if reference_cluster:
                normalized[field] = reference_cluster
            accession = str(row.get(f"afdb_accession_{side}") or "").strip()
            if not accession:
                continue
            link_count += 1
            key = (record_id, side)
            previous_accession = accessions_by_record_side.setdefault(key, accession)
            if previous_accession != accession:
                raise ValueError(
                    f"Prediction dependency {record_id!r} has conflicting AFDB accessions for partner {side}."
                )
        bound.append(normalized)
    if not link_count:
        raise ValueError("Prediction dependency manifest contributes no reference-bound AFDB accession links.")
    return bound


def _choose_development_components(
    component_sizes: dict[str, int],
    fraction: float,
    seed: int,
) -> set[str]:
    total = sum(component_sizes.values())
    target = int(round(total * fraction))
    ordered = list(component_sizes)
    random.Random(seed).shuffle(ordered)
    reachable = [False] * (total + 1)
    predecessor: list[tuple[int, str] | None] = [None] * (total + 1)
    reachable[0] = True
    for component in ordered:
        size = component_sizes[component]
        for current in range(total - size, -1, -1):
            destination = current + size
            if reachable[current] and not reachable[destination]:
                reachable[destination] = True
                predecessor[destination] = (current, component)
    candidates = [value for value in range(1, total) if reachable[value]]
    if not candidates:
        raise ValueError("Prediction-dependency components cannot form non-empty development and test splits.")
    selected_total = min(candidates, key=lambda value: (abs(value - target), value))
    selected = set()
    while selected_total:
        previous, component = predecessor[selected_total]
        selected.add(component)
        selected_total = previous
    return selected


def reconcile_analysis_splits(
    reference_rows: list[dict[str, str]],
    dependency_rows: list[dict[str, str]],
    *,
    development_fraction: float,
    seed: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, object]]:
    """Partition whole homology/accession components before outcome analysis."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one.")
    record_ids = [str(row.get("record_id") or "").strip() for row in reference_rows]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Split reference rows require unique, non-empty record_id values.")
    cluster_by_record = {
        record_id: str(row.get("cluster_id") or "").strip()
        for record_id, row in zip(record_ids, reference_rows, strict=True)
    }
    if any(not value for value in cluster_by_record.values()):
        raise ValueError("Split reference rows require cluster_id values.")
    dependency_rows = bind_prediction_dependencies_to_reference(
        reference_rows,
        dependency_rows,
    )

    homology_nodes = {f"homology_component:{cluster}" for cluster in cluster_by_record.values()}
    nodes = set(homology_nodes)
    links = []
    for row in dependency_rows:
        record_id = dependency_reference_id(row)
        cluster = cluster_by_record[record_id]
        component_node = f"homology_component:{cluster}"
        for side in ("a", "b"):
            accession = str(row.get(f"afdb_accession_{side}") or "").strip()
            if not accession:
                continue
            accession_node = f"afdb_uniprot_accession:{accession}"
            nodes.add(accession_node)
            links.append((component_node, accession_node))

    disjoint = DisjointSet(nodes)
    for left, right in links:
        disjoint.union(left, right)
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        members_by_root[disjoint.find(node)].append(node)
    stable_component_id = {
        root: "splitc_" + hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    split_component_by_homology = {
        node.removeprefix("homology_component:"): stable_component_id[disjoint.find(node)] for node in homology_nodes
    }
    component_sizes = Counter(split_component_by_homology[cluster_by_record[record_id]] for record_id in record_ids)
    development_components = _choose_development_components(
        dict(component_sizes),
        development_fraction,
        seed,
    )
    assignments = {}
    reassigned = 0
    for record_id, row in zip(record_ids, reference_rows, strict=True):
        component = split_component_by_homology[cluster_by_record[record_id]]
        split = "development" if component in development_components else "test"
        assignments[record_id] = (split, component)
        reassigned += int(str(row.get("analysis_split") or "").strip().lower() != split)
    split_counts = Counter(split for split, _component in assignments.values())
    return assignments, {
        "analysis_split_rule": (
            "whole connected components of experimental homology groups and reused AFDB UniProt sources; "
            "components are assigned before TopoPPI outcome analysis"
        ),
        "analysis_split_seed": int(seed),
        "analysis_split_target_development_fraction": float(development_fraction),
        "analysis_split_component_count": int(len(component_sizes)),
        "analysis_split_largest_component_structure_count": int(max(component_sizes.values())),
        "analysis_split_reassigned_structure_count": int(reassigned),
        "analysis_split_development_structure_count": int(split_counts["development"]),
        "analysis_split_test_structure_count": int(split_counts["test"]),
        "analysis_split_observed_development_fraction": float(split_counts["development"] / len(reference_rows)),
    }


def apply_split_assignments(
    rows: list[dict[str, object]],
    assignments: dict[str, tuple[str, str]],
    *,
    reference_field: str,
) -> list[dict[str, object]]:
    enriched = []
    for row in rows:
        reference_id = str(row.get(reference_field) or "").strip()
        if reference_id not in assignments:
            raise ValueError(f"No frozen split assignment for reference record {reference_id!r}.")
        split, component = assignments[reference_id]
        enriched.append(
            {
                **row,
                "analysis_split": split,
                "analysis_split_component_id": component,
                "analysis_split_basis": ANALYSIS_SPLIT_BASIS,
            }
        )
    return enriched


def read_dependency_manifests(
    paths: list[Path],
) -> tuple[list[dict[str, str]] | None, list[dict[str, object]]]:
    sources = [(path, read_csv_rows(path)) for path in paths]
    if any(not rows for _path, rows in sources):
        raise ValueError("Every dependency manifest must contain at least one data row.")
    records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "row_count": len(rows),
        }
        for path, rows in sources
    ]
    combined = [row for _path, rows in sources for row in rows]
    return (combined or None), records


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def boolean(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key) or "").strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Paired QC has invalid {key}: {row.get(key)!r}")


def value_or_empty(row: dict[str, object], key: str) -> object:
    value = row.get(key)
    return "" if value is None else value


def full_experimental_paired_value(
    paired: dict[str, object] | None,
    name: str,
    paired_is_eligible: bool,
) -> object:
    if paired is None:
        return ""
    if paired_is_eligible or name in INFERENCE_FIELDS or name.startswith("paired_benchmark_"):
        return paired.get(name, "")
    return ""


def validate_qc_bindings(
    predicted: list[dict[str, object]],
    experimental: list[dict[str, object]],
    qc_by_pair: dict[str, dict[str, str]],
) -> None:
    predicted_by_pair = {str(row["paired_record_id"]): row for row in predicted}
    experimental_by_pair = {str(row["paired_record_id"]): row for row in experimental}
    for pair_id, qc in qc_by_pair.items():
        predicted_row = predicted_by_pair[pair_id]
        experimental_row = experimental_by_pair[pair_id]
        expected = {
            "predicted_record_id": predicted_row.get("record_id"),
            "experimental_record_id": experimental_row.get("record_id"),
            "predicted_input_sha256": predicted_row.get("input_sha256"),
            "experimental_input_sha256": experimental_row.get("input_sha256"),
        }
        for field, value in expected.items():
            if str(qc.get(field) or "") != str(value or ""):
                raise ValueError(f"Paired QC {field} is stale or belongs to another manifest row.")


def geometry_stratum(row: dict[str, str]) -> str:
    jaccard = numeric(row, "contact_jaccard")
    interface_rmsd = numeric(row, "interface_ligand_ca_rmsd_after_receptor_fit_angstrom")
    clash_fraction = numeric(row, "predicted_cross_chain_clash_atom_fraction")
    contact_mapping = numeric(row, "experimental_contact_mapping_coverage")
    interface_mapping = numeric(row, "interface_ligand_ca_mapping_coverage")
    correspondence_consensus = min(
        numeric(row, "alignment_a_selected_pair_consensus_fraction"),
        numeric(row, "alignment_b_selected_pair_consensus_fraction"),
    )
    if (
        correspondence_consensus >= 0.9
        and contact_mapping >= 0.8
        and interface_mapping >= 0.8
        and jaccard >= 0.4
        and interface_rmsd <= 3.0
        and clash_fraction <= 0.03
    ):
        return "high_fidelity"
    if (
        correspondence_consensus >= 0.9
        and contact_mapping >= 0.5
        and interface_mapping >= 0.5
        and jaccard >= 0.2
        and interface_rmsd <= 5.0
        and clash_fraction <= 0.05
    ):
        return "moderate_fidelity"
    return "geometry_stress_test"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach paired-geometry strata and reconciled splits to manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predicted-manifest",
        required=True,
        type=Path,
        help="Predicted-structure manifest CSV.",
    )
    parser.add_argument(
        "--experimental-manifest",
        required=True,
        type=Path,
        help="Experimental manifest paired to the predicted cohort.",
    )
    parser.add_argument(
        "--full-experimental-manifest",
        type=Path,
        help="Full experimental manifest used when reconciling dependencies.",
    )
    parser.add_argument(
        "--dependency-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Rows carrying sequence-cluster/AFDB-accession links used to freeze splits; repeat the "
            "option to union dependencies from multiple prediction cohorts."
        ),
    )
    parser.add_argument(
        "--split-reference-manifest",
        type=Path,
        help="Previously reconciled full experimental manifest whose split assignments are reused.",
    )
    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.20,
        help="Target fraction assigned to the development split.",
    )
    parser.add_argument("--split-seed", type=int, default=20260817, help="Seed for component split assignment.")
    parser.add_argument(
        "--paired-qc",
        required=True,
        type=Path,
        help="Paired-structure audit CSV for geometry strata.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for reconciled manifests.")
    parser.add_argument(
        "--predicted-label",
        choices=("afdb_monomer", "afdb_dimer"),
        default="afdb_monomer",
        help="Label written for the predicted cohort.",
    )
    args = parser.parse_args()
    required_paths = (args.predicted_manifest, args.experimental_manifest, args.paired_qc)
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError("predicted-manifest, experimental-manifest, and paired-qc must exist.")
    for path in (
        args.full_experimental_manifest,
        args.split_reference_manifest,
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    for path in args.dependency_manifest:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predicted = read_csv_rows(args.predicted_manifest)
    experimental = read_csv_rows(args.experimental_manifest)
    if not predicted or not experimental:
        raise ValueError("Predicted and paired experimental manifests must be non-empty.")
    full_experimental = read_csv_rows(args.full_experimental_manifest) if args.full_experimental_manifest else None
    dependency_rows, dependency_manifest_records = read_dependency_manifests(args.dependency_manifest)
    if dependency_rows is not None:
        if full_experimental is None:
            raise ValueError(
                "dependency-manifest requires full-experimental-manifest, including when a "
                "split-reference-manifest is reused."
            )
        dependency_rows = bind_prediction_dependencies_to_reference(
            full_experimental,
            dependency_rows,
            require_sequence_clusters=True,
        )
    split_summary: dict[str, object] = {}
    assignments: dict[str, tuple[str, str]] | None = None
    if args.split_reference_manifest:
        split_reference = read_csv_rows(args.split_reference_manifest)
        assignments = {}
        for row in split_reference:
            record_id = str(row.get("record_id") or "").strip()
            split = str(row.get("analysis_split") or "").strip().lower()
            component = str(row.get("analysis_split_component_id") or "").strip()
            split_basis = str(row.get("analysis_split_basis") or "").strip()
            if (
                not record_id
                or split not in {"development", "test"}
                or not component
                or split_basis != ANALYSIS_SPLIT_BASIS
            ):
                raise ValueError(
                    "Split-reference rows require record_id, development/test, component ID, "
                    f"and analysis_split_basis={ANALYSIS_SPLIT_BASIS!r}."
                )
            if record_id in assignments:
                raise ValueError(f"Duplicate split-reference record_id: {record_id}")
            assignments[record_id] = (split, component)
        if full_experimental is not None:
            full_ids = {str(row.get("record_id") or "").strip() for row in full_experimental}
            if "" in full_ids or set(assignments) != full_ids:
                raise ValueError("Split-reference and full experimental manifests must contain the same record_id set.")
        split_summary = {
            "analysis_split_rule": "reused from a frozen prediction-dependency split reference",
            "analysis_split_reference_manifest": str(args.split_reference_manifest.resolve()),
            "analysis_split_reference_manifest_sha256": sha256_file(str(args.split_reference_manifest)),
        }
    elif dependency_rows is not None:
        assignments, split_summary = reconcile_analysis_splits(
            full_experimental,
            dependency_rows,
            development_fraction=args.development_fraction,
            seed=args.split_seed,
        )
    if assignments is not None:
        predicted = apply_split_assignments(
            predicted,
            assignments,
            reference_field="paired_reference_record_id",
        )
        experimental = apply_split_assignments(
            experimental,
            assignments,
            reference_field="record_id",
        )
        if full_experimental is not None:
            full_experimental = apply_split_assignments(
                full_experimental,
                assignments,
                reference_field="record_id",
            )
    else:
        split_summary = {
            "analysis_split_rule": "inherited from experimental homology components",
        }
    qc_rows = read_csv_rows(args.paired_qc)
    if not qc_rows:
        raise ValueError("Paired QC table must be non-empty.")
    qc_by_pair = {row["paired_record_id"]: row for row in qc_rows if row.get("status") == "accepted"}
    predicted_pair_ids = [str(row.get("paired_record_id") or "").strip() for row in predicted]
    experimental_pair_ids = [str(row.get("paired_record_id") or "").strip() for row in experimental]
    qc_pair_ids = [str(row.get("paired_record_id") or "").strip() for row in qc_rows]
    if any(not value for value in predicted_pair_ids + experimental_pair_ids + qc_pair_ids):
        raise ValueError("Predicted, experimental, and QC rows require non-empty paired_record_id values.")
    if len(set(predicted_pair_ids)) != len(predicted_pair_ids):
        raise ValueError("Predicted manifest contains duplicate paired_record_id values.")
    if len(set(experimental_pair_ids)) != len(experimental_pair_ids):
        raise ValueError("Experimental manifest contains duplicate paired_record_id values.")
    if len(set(qc_pair_ids)) != len(qc_pair_ids):
        raise ValueError("Paired QC contains duplicate paired_record_id values.")
    if set(predicted_pair_ids) != set(experimental_pair_ids):
        raise ValueError("Predicted and experimental manifests declare different paired records.")
    if set(qc_by_pair) != set(predicted_pair_ids):
        raise ValueError("Every predicted manifest row must have exactly one accepted paired-QC row.")
    validate_qc_bindings(predicted, experimental, qc_by_pair)

    predicted, dependency_summary = add_prediction_dependencies(predicted, dependency_rows)
    validate_prediction_dependency_splits(dependency_summary)
    dependency_by_pair = {
        str(row["paired_record_id"]): {field: row[field] for field in INFERENCE_FIELDS} for row in predicted
    }
    experimental = [{**row, **dependency_by_pair[str(row["paired_record_id"])]} for row in experimental]

    fields = {
        "paired_geometry_stratum": "",
        "paired_contact_cutoff_angstrom": "",
        "paired_predicted_contact_count_total": "",
        "paired_contact_recall_fnat": "",
        "paired_contact_precision": "",
        "paired_contact_jaccard": "",
        "paired_experimental_contact_mapping_coverage": "",
        "paired_interface_residue_a_mapping_coverage": "",
        "paired_interface_residue_b_mapping_coverage": "",
        "paired_interface_ligand_ca_mapping_coverage": "",
        "paired_interface_ligand_ca_rmsd_angstrom": "",
        "paired_cross_chain_clash_atom_fraction": "",
        "paired_alignment_a_optimal_correspondence_count": "",
        "paired_alignment_b_optimal_correspondence_count": "",
        "paired_alignment_a_selected_pair_consensus_fraction": "",
        "paired_alignment_b_selected_pair_consensus_fraction": "",
        "paired_benchmark_eligible": "",
        "paired_benchmark_exclusion_reason": "",
    }

    def enrich(row: dict[str, str]) -> dict[str, object]:
        qc = qc_by_pair[row["paired_record_id"]]
        return {
            **row,
            **fields,
            "paired_geometry_stratum": geometry_stratum(qc),
            "paired_contact_cutoff_angstrom": value_or_empty(qc, "contact_cutoff_angstrom"),
            "paired_predicted_contact_count_total": value_or_empty(qc, "predicted_contact_count_total"),
            "paired_contact_recall_fnat": value_or_empty(qc, "contact_recall_fnat"),
            "paired_contact_precision": value_or_empty(qc, "contact_precision"),
            "paired_contact_jaccard": value_or_empty(qc, "contact_jaccard"),
            "paired_experimental_contact_mapping_coverage": value_or_empty(qc, "experimental_contact_mapping_coverage"),
            "paired_interface_residue_a_mapping_coverage": value_or_empty(qc, "interface_residue_a_mapping_coverage"),
            "paired_interface_residue_b_mapping_coverage": value_or_empty(qc, "interface_residue_b_mapping_coverage"),
            "paired_interface_ligand_ca_mapping_coverage": value_or_empty(qc, "interface_ligand_ca_mapping_coverage"),
            "paired_interface_ligand_ca_rmsd_angstrom": value_or_empty(
                qc, "interface_ligand_ca_rmsd_after_receptor_fit_angstrom"
            ),
            "paired_cross_chain_clash_atom_fraction": value_or_empty(qc, "predicted_cross_chain_clash_atom_fraction"),
            "paired_alignment_a_optimal_correspondence_count": value_or_empty(
                qc, "alignment_a_optimal_correspondence_count"
            ),
            "paired_alignment_b_optimal_correspondence_count": value_or_empty(
                qc, "alignment_b_optimal_correspondence_count"
            ),
            "paired_alignment_a_selected_pair_consensus_fraction": value_or_empty(
                qc, "alignment_a_selected_pair_consensus_fraction"
            ),
            "paired_alignment_b_selected_pair_consensus_fraction": value_or_empty(
                qc, "alignment_b_selected_pair_consensus_fraction"
            ),
            "paired_benchmark_eligible": value_or_empty(qc, "benchmark_eligible"),
            "paired_benchmark_exclusion_reason": value_or_empty(qc, "benchmark_exclusion_reason"),
        }

    predicted_all = [enrich(row) for row in predicted]
    experimental_all = [enrich(row) for row in experimental]
    eligible_pair_ids = {pair_id for pair_id, qc in qc_by_pair.items() if boolean(qc, "benchmark_eligible")}
    if not eligible_pair_ids:
        raise ValueError("No paired structure is eligible for benchmark execution.")
    predicted_enriched = [row for row in predicted_all if str(row["paired_record_id"]) in eligible_pair_ids]
    experimental_enriched = [row for row in experimental_all if str(row["paired_record_id"]) in eligible_pair_ids]
    all_predicted_path = args.output_dir / f"{args.predicted_label}_all_aligned_manifest.csv"
    all_experimental_path = args.output_dir / "pdbbind_all_aligned_paired_manifest.csv"
    predicted_path = args.output_dir / f"{args.predicted_label}_benchmark_manifest.csv"
    experimental_path = args.output_dir / "pdbbind_paired_benchmark_manifest.csv"
    write_csv_atomic(all_predicted_path, predicted_all)
    write_csv_atomic(all_experimental_path, experimental_all)
    write_csv_atomic(predicted_path, predicted_enriched)
    write_csv_atomic(experimental_path, experimental_enriched)
    full_experimental_path = None
    full_experimental_count = None
    full_experimental_paired_count = None
    if full_experimental is not None:
        paired_by_record_id = {str(row.get("record_id") or ""): row for row in experimental_all}
        if "" in paired_by_record_id or len(paired_by_record_id) != len(experimental_all):
            raise ValueError("Paired experimental rows must have unique, non-empty record_id values.")
        full_record_ids = [str(row.get("record_id") or "") for row in full_experimental]
        if any(not value for value in full_record_ids) or len(set(full_record_ids)) != len(full_record_ids):
            raise ValueError("Full experimental rows must have unique, non-empty record_id values.")
        missing_full_rows = sorted(set(paired_by_record_id) - set(full_record_ids))
        if missing_full_rows:
            raise ValueError(
                "Paired experimental rows are absent from the full manifest: " + ", ".join(missing_full_rows[:8])
            )
        paired_columns = ("paired_record_id", *INFERENCE_FIELDS, *fields)
        full_experimental_enriched = []
        for row in full_experimental:
            paired = paired_by_record_id.get(str(row.get("record_id") or ""))
            paired_is_eligible = bool(
                paired is not None and str(paired.get("paired_record_id") or "") in eligible_pair_ids
            )
            full_experimental_enriched.append(
                {
                    **row,
                    **{
                        name: full_experimental_paired_value(
                            paired,
                            name,
                            paired_is_eligible,
                        )
                        for name in paired_columns
                    },
                }
            )
        full_experimental_path = args.output_dir / "pdbbind_full_benchmark_manifest.csv"
        write_csv_atomic(full_experimental_path, full_experimental_enriched)
        full_experimental_count = len(full_experimental_enriched)
        full_experimental_paired_count = len(eligible_pair_ids)
    all_stratum_counts = Counter(str(row["paired_geometry_stratum"]) for row in predicted_all)
    benchmark_exclusion_counts = Counter(
        str(row.get("paired_benchmark_exclusion_reason") or "unspecified")
        for row in predicted_all
        if str(row.get("paired_record_id") or "") not in eligible_pair_ids
    )
    summary = {
        "schema_version": 6,
        "predicted_label": args.predicted_label,
        "design": (
            "outcome-blind operational strata frozen before TopoPPI benchmarking; all aligned "
            "replacements with identifiable partner roles remain in the execution manifests, "
            "including predicted pairs without residue contacts at the frozen cutoff"
        ),
        "contact_metric_domain": (
            "all experimental and predicted contacts, with unmapped experimental contacts "
            "retained as false negatives and unmapped predicted contacts retained as false positives"
        ),
        **split_summary,
        **dependency_summary,
        "thresholds": {
            "high_fidelity": {
                "selected_pair_consensus_fraction_minimum": 0.9,
                "experimental_contact_mapping_coverage_minimum": 0.8,
                "interface_ligand_ca_mapping_coverage_minimum": 0.8,
                "contact_jaccard_minimum": 0.4,
                "interface_ligand_ca_rmsd_angstrom_maximum": 3.0,
                "cross_chain_clash_atom_fraction_maximum": 0.03,
            },
            "moderate_fidelity": {
                "selected_pair_consensus_fraction_minimum": 0.9,
                "experimental_contact_mapping_coverage_minimum": 0.5,
                "interface_ligand_ca_mapping_coverage_minimum": 0.5,
                "contact_jaccard_minimum": 0.2,
                "interface_ligand_ca_rmsd_angstrom_maximum": 5.0,
                "cross_chain_clash_atom_fraction_maximum": 0.05,
            },
            "geometry_stress_test": (
                "all remaining benchmark-eligible replacements, including sequence-correspondence-"
                "ambiguous, mapping-limited, and predicted-contact-absent cases"
            ),
        },
        "aligned_replacement_count": len(predicted_all),
        "benchmark_eligible_record_count": len(predicted_enriched),
        "benchmark_ineligible_record_count": len(predicted_all) - len(predicted_enriched),
        "benchmark_exclusion_reason_counts": dict(sorted(benchmark_exclusion_counts.items())),
        "predicted_contact_present_record_count": sum(
            int(float(str(row["paired_predicted_contact_count_total"]))) > 0 for row in predicted_all
        ),
        "predicted_contact_absent_record_count": sum(
            int(float(str(row["paired_predicted_contact_count_total"]))) == 0 for row in predicted_all
        ),
        "record_count": len(predicted_enriched),
        "stratum_counts": dict(
            sorted(Counter(str(row["paired_geometry_stratum"]) for row in predicted_enriched).items())
        ),
        "all_aligned_stratum_counts": dict(sorted(all_stratum_counts.items())),
        "ambiguous_sequence_correspondence_pair_count": sum(
            numeric(row, "paired_alignment_a_optimal_correspondence_count") > 1
            or numeric(row, "paired_alignment_b_optimal_correspondence_count") > 1
            for row in predicted_all
        ),
        "low_consensus_sequence_correspondence_pair_count": sum(
            min(
                numeric(row, "paired_alignment_a_selected_pair_consensus_fraction"),
                numeric(row, "paired_alignment_b_selected_pair_consensus_fraction"),
            )
            < 0.9
            for row in predicted_all
        ),
        "source_predicted_manifest_sha256": sha256_file(str(args.predicted_manifest)),
        "source_experimental_manifest_sha256": sha256_file(str(args.experimental_manifest)),
        "source_paired_qc_sha256": sha256_file(str(args.paired_qc)),
        "source_full_experimental_manifest_sha256": sha256_file(str(args.full_experimental_manifest))
        if args.full_experimental_manifest
        else None,
        "source_dependency_manifests": dependency_manifest_records,
        "predicted_manifest_sha256": sha256_file(str(predicted_path)),
        "experimental_manifest_sha256": sha256_file(str(experimental_path)),
        "all_aligned_predicted_manifest_sha256": sha256_file(str(all_predicted_path)),
        "all_aligned_experimental_manifest_sha256": sha256_file(str(all_experimental_path)),
        "full_experimental_manifest": str(full_experimental_path.resolve()) if full_experimental_path else None,
        "full_experimental_manifest_sha256": sha256_file(str(full_experimental_path))
        if full_experimental_path
        else None,
        "full_experimental_record_count": full_experimental_count,
        "full_experimental_paired_record_count": full_experimental_paired_count,
    }
    summary_path = args.output_dir / "paired_geometry_strata.json"
    dump_json_atomic(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

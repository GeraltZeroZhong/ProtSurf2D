#!/usr/bin/env python3
"""Cluster selected PDBbind chains and create leakage-safe benchmark splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from topoppi.file_utils import read_csv_rows, sha256_file, write_csv_atomic
from topoppi.json_utils import dump_json_atomic


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
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            smaller, larger = sorted((root_left, root_right))
            self.parent[larger] = smaller


def fasta_record_count(path: Path) -> int:
    return sum(line.startswith(">") for line in path.read_text(encoding="utf-8").splitlines())


def mmseqs_command(args, output_prefix: Path, sequence_count: int) -> list[str]:
    return [
        str(args.mmseqs),
        "easy-cluster",
        str(args.fasta),
        str(output_prefix),
        str(args.output_dir / "mmseqs_tmp"),
        "--min-seq-id",
        str(args.min_sequence_identity),
        "-c",
        str(args.coverage),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
        "--single-step-clustering",
        "1",
        "-s",
        str(args.sensitivity),
        "--max-seqs",
        str(sequence_count),
        "-e",
        str(args.maximum_evalue),
        "--seq-id-mode",
        "0",
        "--alignment-mode",
        "3",
        "--threads",
        str(args.threads),
    ]


def run_mmseqs(args, output_prefix: Path, sequence_count: int) -> tuple[Path, dict[str, object]]:
    cluster_tsv = Path(f"{output_prefix}_cluster.tsv")
    command = mmseqs_command(args, output_prefix, sequence_count)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    (args.output_dir / "mmseqs_easy_cluster.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    version = subprocess.run([str(args.mmseqs), "version"], check=True, capture_output=True, text=True).stdout.strip()
    return cluster_tsv, {"command": command, "version": version}


def parse_clusters(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    members_by_representative: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            members_by_representative[representative].append(member)
    member_to_cluster = {}
    stable_members = {}
    for members in members_by_representative.values():
        ordered = sorted(set(members))
        cluster_id = "seqc_" + hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:16]
        stable_members[cluster_id] = ordered
        for member in ordered:
            if member in member_to_cluster:
                raise ValueError(f"MMseqs member occurs in multiple clusters: {member}")
            member_to_cluster[member] = cluster_id
    return member_to_cluster, stable_members


def component_ids(
    rows: list[dict[str, str]], member_to_cluster: dict[str, str]
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    sequence_clusters = set(member_to_cluster.values())
    disjoint = DisjointSet(sequence_clusters)
    pair_clusters = {}
    for row in rows:
        left_member = f"{row['record_id']}|a|chain={row['chain_a']}"
        right_member = f"{row['record_id']}|b|chain={row['chain_b']}"
        if left_member not in member_to_cluster or right_member not in member_to_cluster:
            raise ValueError(f"Selected sequence is absent from MMseqs output: {row['record_id']}")
        left = member_to_cluster[left_member]
        right = member_to_cluster[right_member]
        disjoint.union(left, right)
        pair_clusters[row["record_id"]] = (left, right)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for sequence_cluster in sequence_clusters:
        members_by_root[disjoint.find(sequence_cluster)].append(sequence_cluster)
    stable_component_id = {
        root: "homc_" + hashlib.sha256("\n".join(sorted(members)).encode("ascii")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    return {
        record_id: stable_component_id[disjoint.find(pair[0])] for record_id, pair in pair_clusters.items()
    }, pair_clusters


def choose_development_components(component_sizes: dict[str, int], fraction: float, seed: int) -> set[str]:
    total = sum(component_sizes.values())
    if len(component_sizes) < 2 or total < 2:
        raise ValueError("A leakage-safe development/test split requires at least two homology components.")
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
    selected_total = min(candidates, key=lambda value: (abs(value - target), value))
    selected = set()
    while selected_total:
        previous, component = predecessor[selected_total]
        selected.add(component)
        selected_total = previous
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster PDBbind partner sequences and create benchmark splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected-pairs",
        required=True,
        type=Path,
        help="Dominant chain-pair CSV from prepare_pdbbind_r1.py.",
    )
    parser.add_argument("--fasta", required=True, type=Path, help="Partner-chain FASTA for the selected pairs.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for clusters and split manifests.")
    parser.add_argument("--mmseqs", type=Path, default=Path("mmseqs"), help="MMseqs2 executable path.")
    parser.add_argument(
        "--min-sequence-identity",
        type=float,
        default=0.30,
        help="Minimum sequence identity used by MMseqs2.",
    )
    parser.add_argument("--coverage", type=float, default=0.80, help="Minimum aligned sequence coverage.")
    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.20,
        help="Target fraction assigned to the development split.",
    )
    parser.add_argument("--sensitivity", type=float, default=7.5, help="MMseqs2 search sensitivity.")
    parser.add_argument("--maximum-evalue", type=float, default=1e-3, help="Maximum MMseqs2 E-value.")
    parser.add_argument("--seed", type=int, default=20260817, help="Random seed for component assignment.")
    parser.add_argument("--threads", type=int, default=8, help="MMseqs2 threads.")
    args = parser.parse_args()
    if not math.isfinite(args.development_fraction) or not 0.0 < args.development_fraction < 1.0:
        raise ValueError("development-fraction must be between zero and one.")
    if not math.isfinite(args.min_sequence_identity) or not 0.0 < args.min_sequence_identity <= 1.0:
        raise ValueError("min-sequence-identity must be in (0, 1].")
    if not math.isfinite(args.coverage) or not 0.0 < args.coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1].")
    if not math.isfinite(args.sensitivity) or not 1.0 <= args.sensitivity <= 7.5:
        raise ValueError("sensitivity must be in [1.0, 7.5].")
    if not math.isfinite(args.maximum_evalue) or args.maximum_evalue <= 0.0:
        raise ValueError("maximum-evalue must be finite and positive.")
    if args.threads <= 0:
        raise ValueError("threads must be positive.")
    if not args.selected_pairs.is_file() or not args.fasta.is_file():
        raise FileNotFoundError("selected-pairs and fasta must be existing files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(args.selected_pairs)
    if not rows:
        raise ValueError("Selected-pairs table contains no rows.")
    record_ids = [str(row.get("record_id") or "").strip() for row in rows]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Selected-pairs rows require unique, non-empty record_id values.")
    sequence_count = fasta_record_count(args.fasta)
    if sequence_count != 2 * len(rows):
        raise ValueError("Selected-pairs FASTA must contain exactly two partner sequences per record.")
    cluster_tsv, mmseqs_metadata = run_mmseqs(
        args,
        args.output_dir / "pdbbind_r1_seq30_cov80",
        sequence_count,
    )
    member_to_cluster, cluster_members = parse_clusters(cluster_tsv)
    record_to_component, pair_clusters = component_ids(rows, member_to_cluster)
    component_sizes = Counter(record_to_component.values())
    development_components = choose_development_components(dict(component_sizes), args.development_fraction, args.seed)

    final_rows = []
    for row in rows:
        left_cluster, right_cluster = pair_clusters[row["record_id"]]
        family_left, family_right = sorted((left_cluster, right_cluster))
        component = record_to_component[row["record_id"]]
        final_rows.append(
            {
                **row,
                "cluster_id": component,
                "family_id": f"pair_{family_left}_{family_right}",
                "sequence_cluster_a": left_cluster,
                "sequence_cluster_b": right_cluster,
                "analysis_split": "development" if component in development_components else "test",
                "analysis_split_component_id": component,
                "dataset_source": "PDBbind v2020R1 protein-protein archive",
                "source_accession": row["pdb_id"],
                "license_or_terms": "PDBbind+ access terms: https://pdbbind-plus.org.cn/",
                "structure_type": "experimental",
                "structure_method": row.get("structure_method") or "not_declared",
                "resolution_angstrom": row.get("resolution_angstrom") or "",
                "paired_record_id": "",
                "hotspot_residues_a": "",
                "prolif_file": "",
                "prolif_sha256": "",
            }
        )

    fields = list(final_rows[0])
    manifest_path = args.output_dir / "pdbbind_r1_experimental_manifest.csv"
    write_csv_atomic(manifest_path, final_rows, fields)
    split_counts = Counter(row["analysis_split"] for row in final_rows)
    sequence_cluster_splits: dict[str, set[str]] = defaultdict(set)
    for row in final_rows:
        sequence_cluster_splits[row["sequence_cluster_a"]].add(row["analysis_split"])
        sequence_cluster_splits[row["sequence_cluster_b"]].add(row["analysis_split"])
    leaking = [cluster for cluster, splits in sequence_cluster_splits.items() if len(splits) > 1]
    if leaking:
        raise RuntimeError(f"Homology leakage detected in {len(leaking)} sequence clusters.")

    summary = {
        "schema_version": 2,
        "selected_pair_count": len(rows),
        "sequence_cluster_count": len(cluster_members),
        "homology_component_count": len(component_sizes),
        "largest_homology_component_structure_count": max(component_sizes.values()),
        "development_structure_count": split_counts["development"],
        "test_structure_count": split_counts["test"],
        "development_fraction_observed": split_counts["development"] / len(final_rows),
        "homology_leakage_sequence_cluster_count": 0,
        "minimum_sequence_identity": args.min_sequence_identity,
        "bidirectional_coverage": args.coverage,
        "maximum_evalue": args.maximum_evalue,
        "search_sensitivity": args.sensitivity,
        "maximum_prefilter_results_per_query": sequence_count,
        "single_step_connected_components": True,
        "sequence_identity_denominator": "alignment_length",
        "random_seed": args.seed,
        "mmseqs": mmseqs_metadata,
        "selected_pairs_sha256": sha256_file(str(args.selected_pairs)),
        "fasta_sha256": sha256_file(str(args.fasta)),
        "manifest_sha256": sha256_file(str(manifest_path)),
    }
    dump_json_atomic(summary, args.output_dir / "pdbbind_r1_clustering_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

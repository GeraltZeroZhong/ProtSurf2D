"""Helpers for unambiguous paired benchmark manifest metadata."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

INFERENCE_DEPENDENCY_BASIS = "union_of_experimental_sequence_homology_clusters_and_reused_afdb_uniprot_accessions"
INFERENCE_DEPENDENCY_FIELDS = (
    "inference_sequence_cluster_a",
    "inference_sequence_cluster_b",
    "inference_family_id",
    "inference_dependency_basis",
)
PREDICTED_STRUCTURE_TYPES = frozenset(
    {
        "predicted",
        "alphafold",
        "afdb",
        "computed_model",
        "afdb_monomer_replacement",
    }
)
FORMAL_STRUCTURE_TYPES = frozenset({"experimental", *PREDICTED_STRUCTURE_TYPES})


def plddt_confidence_stratum(mean_plddt: float) -> str:
    """Bin a residue-mean pLDDT value on the conventional 70/90 scale."""

    value = float(mean_plddt)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("Mean pLDDT must be finite and lie in [0, 100].")
    if value >= 90.0:
        return "mean_pLDDT_high_ge_90"
    if value >= 70.0:
        return "mean_pLDDT_medium_ge_70_lt_90"
    return "mean_pLDDT_low_lt_70"


def ipsae_confidence_stratum(ipsae: float) -> str:
    """Bin AFDB complex ipSAE without conflating it with pLDDT."""

    value = float(ipsae)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ipSAE must be finite and lie in [0, 1].")
    if value >= 0.70:
        return "ipSAE_high_ge_0.70"
    if value >= 0.50:
        return "ipSAE_medium_ge_0.50_lt_0.70"
    return "ipSAE_low_lt_0.50"


def inference_family_id(left: str, right: str) -> str:
    """Return the role-invariant family identifier for two inference clusters."""

    first, second = sorted((left, right))
    return f"pifam_{first}_{second}"


def observed_sequence_metadata(sequence_a: str, sequence_b: str) -> dict[str, object]:
    """Describe the residue sequences observed in the current coordinate input."""

    return {
        "sequence_a": sequence_a,
        "sequence_b": sequence_b,
        "sequence_a_sha256": hashlib.sha256(sequence_a.encode("ascii")).hexdigest(),
        "sequence_b_sha256": hashlib.sha256(sequence_b.encode("ascii")).hexdigest(),
        "chain_a_residue_count": len(sequence_a),
        "chain_b_residue_count": len(sequence_b),
        "sequence_semantics": "observed_residues_in_current_coordinate_input",
    }


def paired_reference_metadata(row: Mapping[str, object]) -> dict[str, object]:
    """Retain the experimental sequence provenance used for paired grouping."""

    return {
        "paired_reference_record_id": row["record_id"],
        "paired_reference_structure_path": row["structure_path"],
        "paired_reference_input_sha256": row["input_sha256"],
        "paired_reference_chain_a": row["chain_a"],
        "paired_reference_chain_b": row["chain_b"],
        "paired_reference_sequence_a": row["sequence_a"],
        "paired_reference_sequence_b": row["sequence_b"],
        "paired_reference_sequence_a_sha256": row["sequence_a_sha256"],
        "paired_reference_sequence_b_sha256": row["sequence_b_sha256"],
        "paired_reference_chain_a_residue_count": row["chain_a_residue_count"],
        "paired_reference_chain_b_residue_count": row["chain_b_residue_count"],
        "sequence_cluster_reference": "paired_experimental_observed_sequences",
    }

"""Mesh operations that retain source vertex, face, and atom identities."""

from __future__ import annotations

import copy
from typing import Iterable, List

import numpy as np
import trimesh

SOURCE_VERTEX_IDS = "source_vertex_ids"
SOURCE_FACE_IDS = "source_face_ids"
SOURCE_ATOM_IDS = "source_atom_indices"
OPTCUTS_GEOMETRY_VERTEX_IDS = "optcuts_geometry_vertex_ids"
PROVENANCE_HISTORY = "provenance_history"


def initialize_provenance(mesh: trimesh.Trimesh, stage: str | None = None) -> trimesh.Trimesh:
    metadata = mesh.metadata
    if SOURCE_VERTEX_IDS not in metadata:
        metadata[SOURCE_VERTEX_IDS] = np.arange(len(mesh.vertices), dtype=np.int64)
    elif len(metadata[SOURCE_VERTEX_IDS]) != len(mesh.vertices):
        raise ValueError("source_vertex_ids length does not match mesh vertices.")
    if SOURCE_FACE_IDS not in metadata:
        metadata[SOURCE_FACE_IDS] = np.arange(len(mesh.faces), dtype=np.int64)
    elif len(metadata[SOURCE_FACE_IDS]) != len(mesh.faces):
        raise ValueError("source_face_ids length does not match mesh faces.")
    if SOURCE_ATOM_IDS in metadata and len(metadata[SOURCE_ATOM_IDS]) != len(mesh.vertices):
        raise ValueError("source_atom_indices length does not match mesh vertices.")
    if OPTCUTS_GEOMETRY_VERTEX_IDS in metadata and len(metadata[OPTCUTS_GEOMETRY_VERTEX_IDS]) != len(mesh.vertices):
        raise ValueError("optcuts_geometry_vertex_ids length does not match mesh vertices.")
    metadata.setdefault("root_vertex_count", int(len(mesh.vertices)))
    metadata.setdefault("root_face_count", int(len(mesh.faces)))
    metadata.setdefault("root_area", float(mesh.area))
    metadata.setdefault(PROVENANCE_HISTORY, [])
    if stage:
        record_stage(mesh, stage)
    return mesh


def _metadata_copy(mesh: trimesh.Trimesh) -> dict:
    source_keys = {
        SOURCE_VERTEX_IDS,
        SOURCE_FACE_IDS,
        SOURCE_ATOM_IDS,
        OPTCUTS_GEOMETRY_VERTEX_IDS,
        PROVENANCE_HISTORY,
    }
    result = {key: copy.deepcopy(value) for key, value in mesh.metadata.items() if key not in source_keys}
    result[PROVENANCE_HISTORY] = copy.deepcopy(mesh.metadata.get(PROVENANCE_HISTORY, []))
    return result


def tracked_submesh(
    mesh: trimesh.Trimesh,
    face_indices: Iterable[int],
    *,
    stage: str,
) -> trimesh.Trimesh:
    """Extract a face subset while preserving root identities."""

    initialize_provenance(mesh)
    if isinstance(face_indices, np.ndarray):
        face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    else:
        face_indices = np.fromiter(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
        empty.metadata.update(_metadata_copy(mesh))
        empty.metadata[SOURCE_VERTEX_IDS] = np.empty(0, dtype=np.int64)
        empty.metadata[SOURCE_FACE_IDS] = np.empty(0, dtype=np.int64)
        if SOURCE_ATOM_IDS in mesh.metadata:
            empty.metadata[SOURCE_ATOM_IDS] = np.empty(0, dtype=np.int64)
        if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata:
            empty.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.empty(0, dtype=np.int64)
        record_stage(empty, stage)
        return empty

    selected_faces = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    used_vertices = np.unique(selected_faces.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)

    result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[used_vertices],
        faces=remap[selected_faces],
        process=False,
    )
    result.metadata.update(_metadata_copy(mesh))
    result.metadata[SOURCE_VERTEX_IDS] = np.asarray(mesh.metadata[SOURCE_VERTEX_IDS], dtype=np.int64)[used_vertices]
    result.metadata[SOURCE_FACE_IDS] = np.asarray(mesh.metadata[SOURCE_FACE_IDS], dtype=np.int64)[face_indices]
    if SOURCE_ATOM_IDS in mesh.metadata and len(mesh.metadata[SOURCE_ATOM_IDS]) == len(mesh.vertices):
        result.metadata[SOURCE_ATOM_IDS] = np.asarray(mesh.metadata[SOURCE_ATOM_IDS], dtype=np.int64)[used_vertices]
    if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata:
        result.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.asarray(
            mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS], dtype=np.int64
        )[used_vertices]
    record_stage(result, stage)
    return result


def tracked_vertex_duplication(
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_to_input: np.ndarray,
    *,
    stage: str,
) -> trimesh.Trimesh:
    """Materialize a cut mesh while retaining every input face and root identity."""

    initialize_provenance(mesh)
    new_vertices = np.asarray(vertices, dtype=np.float64)
    new_faces = np.asarray(faces, dtype=np.int64)
    source_index = np.asarray(vertex_to_input, dtype=np.int64).reshape(-1)
    if new_faces.shape != np.asarray(mesh.faces).shape:
        raise ValueError("A topology cut must retain the complete input face array.")
    if len(new_vertices) != len(source_index):
        raise ValueError("vertex_to_input length does not match cut-mesh vertices.")
    if len(source_index) and (source_index.min() < 0 or source_index.max() >= len(mesh.vertices)):
        raise ValueError("vertex_to_input contains an invalid input vertex index.")
    if len(new_faces) and (new_faces.min() < 0 or new_faces.max() >= len(new_vertices)):
        raise ValueError("Cut-mesh faces contain an invalid output vertex index.")
    mapped_faces = source_index[new_faces]
    if not np.array_equal(mapped_faces, np.asarray(mesh.faces, dtype=np.int64)):
        raise ValueError("A topology cut must preserve the input face order and corner correspondence.")
    input_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    coordinate_scale = max(float(np.max(np.abs(input_vertices))), 1.0)
    coordinate_tolerance = 256.0 * np.finfo(np.float64).eps * coordinate_scale
    if not np.allclose(
        new_vertices,
        input_vertices[source_index],
        rtol=0.0,
        atol=coordinate_tolerance,
    ):
        raise ValueError("A topology cut must not change the input 3-D geometry.")

    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    result.metadata.update(_metadata_copy(mesh))
    result.metadata[SOURCE_VERTEX_IDS] = np.asarray(mesh.metadata[SOURCE_VERTEX_IDS], dtype=np.int64)[source_index]
    result.metadata[SOURCE_FACE_IDS] = np.asarray(mesh.metadata[SOURCE_FACE_IDS], dtype=np.int64).copy()
    if SOURCE_ATOM_IDS in mesh.metadata:
        result.metadata[SOURCE_ATOM_IDS] = np.asarray(mesh.metadata[SOURCE_ATOM_IDS], dtype=np.int64)[source_index]
    if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata:
        result.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.asarray(
            mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS], dtype=np.int64
        )[source_index]
    record_stage(result, stage)
    return result


def replace_mesh(target: trimesh.Trimesh, source: trimesh.Trimesh) -> None:
    target.vertices = np.asarray(source.vertices, dtype=np.float64).copy()
    target.faces = np.asarray(source.faces, dtype=np.int64).copy()
    # Geometry replacement invalidates trimesh's lazily cached vertex/face
    # colour arrays whenever a topology cut changes the vertex count.
    target.visual = trimesh.visual.ColorVisuals(mesh=target)
    target.metadata.clear()
    target.metadata.update(_metadata_copy(source))
    for key in (
        SOURCE_VERTEX_IDS,
        SOURCE_FACE_IDS,
        SOURCE_ATOM_IDS,
        OPTCUTS_GEOMETRY_VERTEX_IDS,
    ):
        if key in source.metadata:
            target.metadata[key] = np.asarray(source.metadata[key], dtype=np.int64).copy()


def filter_faces(mesh: trimesh.Trimesh, keep_mask: np.ndarray, *, stage: str) -> trimesh.Trimesh:
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.shape != (len(mesh.faces),):
        raise ValueError("Face mask length does not match mesh face count.")
    return tracked_submesh(mesh, np.flatnonzero(mask), stage=stage)


def connected_face_components(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    """Return deterministic face components linked by a shared edge."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_index)
    adjacency = [set() for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        for left in incident:
            adjacency[left].update(right for right in incident if right != left)

    components: List[np.ndarray] = []
    unseen = set(range(len(faces)))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def remove_duplicate_faces(mesh: trimesh.Trimesh, *, stage: str) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return tracked_submesh(mesh, [], stage=stage)
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    return tracked_submesh(mesh, np.sort(first), stage=stage)


def merge_duplicate_vertices(mesh: trimesh.Trimesh, *, stage: str, digits: int = 12) -> trimesh.Trimesh:
    """Merge coincident vertices deterministically and retain representative IDs."""

    initialize_provenance(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return tracked_submesh(mesh, [], stage=stage)
    quantized = np.round(vertices, decimals=digits)
    _, first, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    inverse_order = np.empty(len(order), dtype=np.int64)
    inverse_order[order] = np.arange(len(order), dtype=np.int64)
    new_faces = inverse_order[inverse][np.asarray(mesh.faces, dtype=np.int64)]
    representatives = first[order]

    result = trimesh.Trimesh(vertices=vertices[representatives], faces=new_faces, process=False)
    result.metadata.update(_metadata_copy(mesh))
    result.metadata[SOURCE_VERTEX_IDS] = np.asarray(mesh.metadata[SOURCE_VERTEX_IDS], dtype=np.int64)[representatives]
    result.metadata[SOURCE_FACE_IDS] = np.asarray(mesh.metadata[SOURCE_FACE_IDS], dtype=np.int64).copy()
    if SOURCE_ATOM_IDS in mesh.metadata and len(mesh.metadata[SOURCE_ATOM_IDS]) == len(mesh.vertices):
        result.metadata[SOURCE_ATOM_IDS] = np.asarray(mesh.metadata[SOURCE_ATOM_IDS], dtype=np.int64)[representatives]
    if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata:
        result.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.asarray(
            mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS], dtype=np.int64
        )[representatives]
    record_stage(result, stage)
    return result


def record_stage(mesh: trimesh.Trimesh, stage: str) -> dict:
    history = list(mesh.metadata.get(PROVENANCE_HISTORY, []))
    source_vertices = np.asarray(mesh.metadata.get(SOURCE_VERTEX_IDS, []), dtype=np.int64)
    source_faces = np.asarray(mesh.metadata.get(SOURCE_FACE_IDS, []), dtype=np.int64)
    source_atoms = np.asarray(mesh.metadata.get(SOURCE_ATOM_IDS, []), dtype=np.int64)
    record = {
        "stage": str(stage),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "area": float(mesh.area) if len(mesh.faces) else 0.0,
        "unique_source_vertex_count": int(len(np.unique(source_vertices))),
        "unique_source_face_count": int(len(np.unique(source_faces))),
        "unique_source_atom_count": int(len(np.unique(source_atoms[source_atoms >= 0]))) if len(source_atoms) else 0,
    }
    if history and history[-1].get("stage") == stage:
        history[-1] = record
    else:
        history.append(record)
    mesh.metadata[PROVENANCE_HISTORY] = history
    return record


def provenance_summary(mesh: trimesh.Trimesh) -> dict:
    initialize_provenance(mesh)
    source_vertices = np.asarray(mesh.metadata[SOURCE_VERTEX_IDS], dtype=np.int64)
    source_faces = np.asarray(mesh.metadata[SOURCE_FACE_IDS], dtype=np.int64)
    source_atoms = np.asarray(mesh.metadata.get(SOURCE_ATOM_IDS, []), dtype=np.int64)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "area": float(mesh.area) if len(mesh.faces) else 0.0,
        "source_vertex_ids": sorted(int(x) for x in np.unique(source_vertices)),
        "source_face_ids": sorted(int(x) for x in np.unique(source_faces)),
        "source_atom_indices": sorted(int(x) for x in np.unique(source_atoms[source_atoms >= 0]))
        if len(source_atoms)
        else [],
        "history": copy.deepcopy(mesh.metadata.get(PROVENANCE_HISTORY, [])),
    }

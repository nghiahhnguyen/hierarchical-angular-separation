"""Disjoint-split aggregation and null controls copied from the paper code."""
from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np

_PER_NODE_FLOAT_FIELDS: List[str] = [
    "theta_i_rad", "theta_i_deg", "num_descendants", "num_non_descendants",
    "num_non_descendant_pairs", "num_violations", "violation_proportion",
    "mean_exceeding_angle_violation_rad", "mean_exceeding_angle_violation_deg",
    "mean_exceeding_angle_margin_rad", "mean_exceeding_angle_margin_deg",
    "max_exceeding_angle_rad", "max_exceeding_angle_deg",
    "std_exceeding_angle_rad", "std_exceeding_angle_deg",
    "mean_relative_exceeding_angle_violation", "max_relative_exceeding_angle_violation",
    "std_relative_exceeding_angle_violation", "leaves_excl_count",
    "subtree_leaf_count", "anchor_leaf_total_count",
]


_META_EXCLUDE: frozenset = frozenset(_PER_NODE_FLOAT_FIELDS + ["top_violations", "max_desc_node_reverse"])


def split_even_odd_embeddings(all_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Split cached per-image embeddings into disjoint even/odd subsets."""
    embeddings = np.asarray(all_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D all_embeddings, got shape {embeddings.shape}")

    split_a = embeddings[0::2]
    split_b = embeddings[1::2]
    if split_a.shape[0] == 0 or split_b.shape[0] == 0:
        raise ValueError(
            f"Need both even and odd examples for disjoint split, got counts "
            f"{split_a.shape[0]} and {split_b.shape[0]}."
        )

    return {
        "A_embeddings": split_a,
        "B_embeddings": split_b,
        "A_mean": split_a.mean(axis=0),
        "B_mean": split_b.mean(axis=0),
        "A_count": int(split_a.shape[0]),
        "B_count": int(split_b.shape[0]),
        "total_count": int(embeddings.shape[0]),
    }


def aggregate_descendant_leaf_embeddings(
    G: nx.DiGraph,
    leaf_means: Dict[str, np.ndarray],
    leaf_counts: Dict[str, int],
    weighting: str = "count",
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Build node embeddings/counts by bottom-up aggregation over the tree.

    weighting="count" weights each child by its instance count, so an internal node
    equals the mean over the raw images of its descendant leaves.
    weighting="uniform_children" weights every direct child equally, so an internal
    node is the mean of its child prototypes regardless of subtree size. The returned
    counts are instance counts either way.
    """
    if weighting not in ("count", "uniform_children"):
        raise ValueError(f"Unknown weighting: {weighting}")

    node_embeddings: Dict[str, np.ndarray] = {}
    node_counts: Dict[str, int] = {}

    for node in reversed(list(nx.topological_sort(G))):
        if G.out_degree(node) == 0:
            if node not in leaf_means or node not in leaf_counts:
                continue
            node_embeddings[node] = np.asarray(leaf_means[node], dtype=np.float32)
            node_counts[node] = int(leaf_counts[node])
            continue

        weighted_sum = None
        total_weight = 0.0
        total_count = 0
        for child in G.successors(node):
            child_embedding = node_embeddings.get(child)
            child_count = node_counts.get(child, 0)
            if child_embedding is None or child_count <= 0:
                continue
            weight = 1.0 if weighting == "uniform_children" else float(child_count)
            contribution = child_embedding * weight
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
            total_weight += weight
            total_count += child_count

        if weighted_sum is None or total_weight <= 0:
            continue

        node_embeddings[node] = (weighted_sum / total_weight).astype(np.float32, copy=False)
        node_counts[node] = total_count

    return node_embeddings, node_counts


def _compute_branch_statistics(per_node_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    if not per_node_results:
        return {}, {}

    theta_values = np.array([r["theta_i_rad"] for r in per_node_results], dtype=np.float64)
    violation_percentages = np.array([r["violation_proportion"] for r in per_node_results], dtype=np.float64)
    mean_exceeding_violations = np.array(
        [r["mean_exceeding_angle_violation_rad"] for r in per_node_results], dtype=np.float64
    )
    mean_exceeding_margins = np.array(
        [r["mean_exceeding_angle_margin_rad"] for r in per_node_results], dtype=np.float64
    )
    max_exceeding_violations = np.array([r["max_exceeding_angle_rad"] for r in per_node_results], dtype=np.float64)
    std_exceeding_violations = np.array([r["std_exceeding_angle_rad"] for r in per_node_results], dtype=np.float64)
    mean_relative_exceeding = np.array(
        [r["mean_relative_exceeding_angle_violation"] for r in per_node_results], dtype=np.float64
    )
    max_relative_exceeding = np.array(
        [r["max_relative_exceeding_angle_violation"] for r in per_node_results], dtype=np.float64
    )
    std_relative_exceeding = np.array(
        [r["std_relative_exceeding_angle_violation"] for r in per_node_results], dtype=np.float64
    )

    aggregated_statistics = {
        "cone_radii": {
            "mean_rad": float(np.mean(theta_values)),
            "mean_deg": float(np.degrees(np.mean(theta_values))),
            "std_rad": float(np.std(theta_values)),
            "std_deg": float(np.degrees(np.std(theta_values))),
            "median_rad": float(np.median(theta_values)),
            "median_deg": float(np.degrees(np.median(theta_values))),
            "min_rad": float(np.min(theta_values)),
            "min_deg": float(np.degrees(np.min(theta_values))),
            "max_rad": float(np.max(theta_values)),
            "max_deg": float(np.degrees(np.max(theta_values))),
            "percentile_25_deg": float(np.degrees(np.percentile(theta_values, 25))),
            "percentile_75_deg": float(np.degrees(np.percentile(theta_values, 75))),
            "percentile_90_deg": float(np.degrees(np.percentile(theta_values, 90))),
        },
        "violation_proportion": {
            "mean": float(np.mean(violation_percentages)),
            "std": float(np.std(violation_percentages)),
            "median": float(np.median(violation_percentages)),
            "min": float(np.min(violation_percentages)),
            "max": float(np.max(violation_percentages)),
            "percentile_25": float(np.percentile(violation_percentages, 25)),
            "percentile_75": float(np.percentile(violation_percentages, 75)),
            "percentile_90": float(np.percentile(violation_percentages, 90)),
        },
        "mean_exceeding_angle_violation": {
            "mean_rad": float(np.mean(mean_exceeding_violations)),
            "mean_deg": float(np.degrees(np.mean(mean_exceeding_violations))),
            "std_rad": float(np.std(mean_exceeding_violations)),
            "std_deg": float(np.degrees(np.std(mean_exceeding_violations))),
            "median_rad": float(np.median(mean_exceeding_violations)),
            "median_deg": float(np.degrees(np.median(mean_exceeding_violations))),
            "max_rad": float(np.max(mean_exceeding_violations)),
            "max_deg": float(np.degrees(np.max(mean_exceeding_violations))),
        },
        "mean_exceeding_angle_margin": {
            "mean_rad": float(np.mean(mean_exceeding_margins)),
            "mean_deg": float(np.degrees(np.mean(mean_exceeding_margins))),
            "std_rad": float(np.std(mean_exceeding_margins)),
            "std_deg": float(np.degrees(np.std(mean_exceeding_margins))),
            "median_rad": float(np.median(mean_exceeding_margins)),
            "median_deg": float(np.degrees(np.median(mean_exceeding_margins))),
            "min_rad": float(np.min(mean_exceeding_margins)),
            "min_deg": float(np.degrees(np.min(mean_exceeding_margins))),
        },
        "max_exceeding_angle": {
            "mean_rad": float(np.mean(max_exceeding_violations)),
            "mean_deg": float(np.degrees(np.mean(max_exceeding_violations))),
            "max_rad": float(np.max(max_exceeding_violations)),
            "max_deg": float(np.degrees(np.max(max_exceeding_violations))),
        },
        "std_exceeding_angle": {
            "mean_rad": float(np.mean(std_exceeding_violations)),
            "mean_deg": float(np.degrees(np.mean(std_exceeding_violations))),
            "max_rad": float(np.max(std_exceeding_violations)),
            "max_deg": float(np.degrees(np.max(std_exceeding_violations))),
        },
        "mean_relative_exceeding_angle_violation": {
            "mean": float(np.mean(mean_relative_exceeding)),
            "std": float(np.std(mean_relative_exceeding)),
            "median": float(np.median(mean_relative_exceeding)),
            "max": float(np.max(mean_relative_exceeding)),
        },
        "max_relative_exceeding_angle_violation": {
            "mean": float(np.mean(max_relative_exceeding)),
            "std": float(np.std(max_relative_exceeding)),
            "median": float(np.median(max_relative_exceeding)),
            "max": float(np.max(max_relative_exceeding)),
        },
        "std_relative_exceeding_angle_violation": {
            "mean": float(np.mean(std_relative_exceeding)),
            "std": float(np.std(std_relative_exceeding)),
            "median": float(np.median(std_relative_exceeding)),
            "max": float(np.max(std_relative_exceeding)),
        },
        "num_nodes": len(per_node_results),
    }

    depth_stratified: Dict[int, Dict[str, Any]] = {}
    for depth in sorted({int(r["depth"]) for r in per_node_results}):
        depth_rows = [r for r in per_node_results if int(r["depth"]) == depth]
        depth_thetas = np.array([r["theta_i_rad"] for r in depth_rows], dtype=np.float64)
        depth_vps = np.array([r["violation_proportion"] for r in depth_rows], dtype=np.float64)
        depth_exc_viol = np.array([r["mean_exceeding_angle_violation_rad"] for r in depth_rows], dtype=np.float64)
        depth_exc_margin = np.array([r["mean_exceeding_angle_margin_rad"] for r in depth_rows], dtype=np.float64)
        depth_max_exc = np.array([r["max_exceeding_angle_rad"] for r in depth_rows], dtype=np.float64)
        depth_mean_rel = np.array(
            [r["mean_relative_exceeding_angle_violation"] for r in depth_rows], dtype=np.float64
        )
        depth_max_rel = np.array(
            [r["max_relative_exceeding_angle_violation"] for r in depth_rows], dtype=np.float64
        )
        depth_std_rel = np.array(
            [r["std_relative_exceeding_angle_violation"] for r in depth_rows], dtype=np.float64
        )
        depth_stratified[depth] = {
            "cone_radii_deg": {
                "mean": float(np.degrees(np.mean(depth_thetas))),
                "std": float(np.degrees(np.std(depth_thetas))),
                "median": float(np.degrees(np.median(depth_thetas))),
                "min": float(np.degrees(np.min(depth_thetas))),
                "max": float(np.degrees(np.max(depth_thetas))),
            },
            "violation_proportion": {
                "mean": float(np.mean(depth_vps)),
                "std": float(np.std(depth_vps)),
                "median": float(np.median(depth_vps)),
            },
            "mean_exceeding_angle_violation_deg": {
                "mean": float(np.degrees(np.mean(depth_exc_viol))),
                "std": float(np.degrees(np.std(depth_exc_viol))),
            },
            "mean_exceeding_angle_margin_deg": {
                "mean": float(np.degrees(np.mean(depth_exc_margin))),
                "std": float(np.degrees(np.std(depth_exc_margin))),
            },
            "max_exceeding_angle_deg": {
                "mean": float(np.degrees(np.mean(depth_max_exc))),
                "max": float(np.degrees(np.max(depth_max_exc))),
            },
            "mean_relative_exceeding_angle_violation": {
                "mean": float(np.mean(depth_mean_rel)),
                "std": float(np.std(depth_mean_rel)),
                "median": float(np.median(depth_mean_rel)),
            },
            "max_relative_exceeding_angle_violation": {
                "mean": float(np.mean(depth_max_rel)),
                "std": float(np.std(depth_max_rel)),
                "max": float(np.max(depth_max_rel)),
            },
            "std_relative_exceeding_angle_violation": {
                "mean": float(np.mean(depth_std_rel)),
                "std": float(np.std(depth_std_rel)),
            },
            "count": len(depth_rows),
        }

    return aggregated_statistics, depth_stratified


def average_branch_separation_results(
    result_a_to_b: Dict[str, Any],
    result_b_to_a: Dict[str, Any],
    model_key: str,
    embedding_type: str = "image",
    id_field: str = "node_id",
) -> Dict[str, Any]:
    """Average directional branch-separation results row-wise and recompute summaries."""
    per_node_a = {row[id_field]: row for row in result_a_to_b.get("per_node_results", [])}
    per_node_b = {row[id_field]: row for row in result_b_to_a.get("per_node_results", [])}
    shared_nodes = sorted(set(per_node_a) & set(per_node_b))
    if not shared_nodes:
        raise ValueError("No shared nodes between directional branch-separation results.")

    avg_rows: List[Dict[str, Any]] = []
    for node_id in shared_nodes:
        row_a = per_node_a[node_id]
        row_b = per_node_b[node_id]
        avg_row = {
            id_field: node_id,
            "node_name": row_a.get("node_name", row_b.get("node_name", str(node_id))),
            "depth": int(row_a.get("depth", row_b.get("depth", -1))),
            "max_desc_node": row_a.get("max_desc_node", row_b.get("max_desc_node")),
            "max_desc_node_reverse": row_b.get("max_desc_node", row_a.get("max_desc_node")),
            "top_violations": [],
        }
        for field in _PER_NODE_FLOAT_FIELDS:
            if field in row_a or field in row_b:
                avg_row[field] = float(0.5 * (row_a.get(field, 0.0) + row_b.get(field, 0.0)))
        for key, value in row_a.items():
            if key not in avg_row and key not in _META_EXCLUDE:
                avg_row[key] = value
        for key, value in row_b.items():
            if key not in avg_row and key not in _META_EXCLUDE:
                avg_row[key] = value
        avg_rows.append(avg_row)

    aggregated_statistics, depth_stratified = _compute_branch_statistics(avg_rows)
    return {
        "model_key": model_key,
        "embedding_type": embedding_type,
        "cone_boundary_percentile": result_a_to_b["cone_boundary_percentile"],
        "level_filtered_non_descendants": result_a_to_b["level_filtered_non_descendants"],
        "use_individual_template_embeddings": False,
        "evaluation_mode": "bidirectional_avg",
        "forward_label": "A_to_B",
        "reverse_label": "B_to_A",
        "per_node_results": avg_rows,
        "aggregated_statistics": aggregated_statistics,
        "depth_stratified": depth_stratified,
    }


def _gaussian_splits_from_counts(
    leaf_counts: Dict[str, int],
    gaussian_seed: int,
    dim: int = 768,
    normalize: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int], Dict[str, int]]:
    """Generate i.i.d. Gaussian even/odd splits from pre-read leaf counts (pure numpy; cheap)."""
    rng = np.random.default_rng(gaussian_seed)
    leaf_means_a: Dict[str, np.ndarray] = {}
    leaf_means_b: Dict[str, np.ndarray] = {}
    leaf_counts_a: Dict[str, int] = {}
    leaf_counts_b: Dict[str, int] = {}
    for leaf_id in sorted(leaf_counts):
        n = leaf_counts[leaf_id]
        vectors = rng.standard_normal(size=(n, dim)).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        split = split_even_odd_embeddings(vectors)
        leaf_means_a[leaf_id] = split["A_mean"]
        leaf_means_b[leaf_id] = split["B_mean"]
        leaf_counts_a[leaf_id] = split["A_count"]
        leaf_counts_b[leaf_id] = split["B_count"]
    return leaf_means_a, leaf_means_b, leaf_counts_a, leaf_counts_b


def shuffle_leaf_assignments(
    leaf_means_a: Dict[str, np.ndarray],
    leaf_means_b: Dict[str, np.ndarray],
    leaf_counts_a: Dict[str, int],
    leaf_counts_b: Dict[str, int],
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int], Dict[str, int]]:
    """Randomly permute which leaf node receives which cached embedding split.

    Keeps the tree topology and per-leaf image counts intact; only the
    synset-to-embedding association is scrambled.
    """
    if not (set(leaf_means_b) == set(leaf_means_a) == set(leaf_counts_a) == set(leaf_counts_b)):
        raise ValueError(
            "All four leaf dicts must have identical key sets; "
            f"got sizes {len(leaf_means_a)}, {len(leaf_means_b)}, "
            f"{len(leaf_counts_a)}, {len(leaf_counts_b)}"
        )
    leaf_ids = sorted(leaf_means_a.keys())
    perm_ids = list(rng.permutation(leaf_ids))
    return (
        {k: leaf_means_a[v] for k, v in zip(leaf_ids, perm_ids)},
        {k: leaf_means_b[v] for k, v in zip(leaf_ids, perm_ids)},
        {k: leaf_counts_a[v] for k, v in zip(leaf_ids, perm_ids)},
        {k: leaf_counts_b[v] for k, v in zip(leaf_ids, perm_ids)},
    )

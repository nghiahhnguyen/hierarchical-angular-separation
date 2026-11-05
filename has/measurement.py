"""The paper's percentile-based HAS measurement."""
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from tqdm import tqdm

def test_branch_separation(
    G: Any,
    embeddings: Dict[str, np.ndarray],
    model_key: str,
    embedding_type: str = "text",
    synset_to_name: Optional[Dict[str, str]] = None,
    save_all_violations: bool = False,
    use_first_name_only: bool = False,
    cone_boundary_percentile: float = 100.0,
    level_filtered_non_descendants: bool = False,
    individual_embeddings: Optional[Dict[str, np.ndarray]] = None,
    use_individual_template_embeddings: bool = False,
    comparison_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Test how well-separated tree branches are in the hierarchical structure.

    For each non-leaf node i with at least one child, computes:
    - theta_i = percentile of angles between a_i and descendants (max when percentile=100)
    - violation_proportion = percentage (0-100) of non-descendants with angle < theta_i
    - absolute violation angle = max(0, theta_i - angle) for violating non-descendants
    - relative violation angle = (theta_i - angle) / theta_i for violating non-descendants
      (normalized by cone size, 0=no violation, 1=at node center)
      Summary stats for exceeding-angle violations (mean/std/max) are computed over violating
      non-descendants only (conditional on violation). Margin stats are computed over all
      non-descendants.

    This measures whether the cone defined by a node and its descendants is well-separated
    from other branches in the tree, as described in Section 2 (Geometric Conditions) of
    the hierarchical model paper.

    Args:
        G: NetworkX hierarchy graph
        embeddings: Dictionary mapping node IDs to embedding vectors
        model_key: Model identifier for results
        embedding_type: Type of embeddings ("text" or "image")
        synset_to_name: Mapping from synset IDs to class names
        save_all_violations: If True, save all violating pairs (not just top 5)
        use_first_name_only: If True, only use first name before comma in violation reports
        cone_boundary_percentile: Percentile of descendant angles to use for cone boundary (default=100, i.e., max)
        level_filtered_non_descendants: If True, filter non-descendants to only include nodes at level > L
            (where L is the ancestor node's depth). This provides a more precise measurement by excluding
            nodes at shallower levels than the ancestor.
        individual_embeddings: Optional dict mapping node IDs to per-template embeddings with shape (T, D).
            When provided and use_individual_template_embeddings=True, descendants/non-descendants are evaluated
            at the (node, template) level rather than using the synset-averaged `embeddings` only.
        use_individual_template_embeddings: If True, compute descendant cone boundary and non-descendant
            violations using individual template embeddings (unit of analysis: (node, template) pair). The
            cone central axis remains the synset representation from `embeddings[node]`.
        comparison_embeddings: Optional dictionary used only for descendant/non-descendant comparisons.
            When provided, the cone axis for node `u` still comes from `embeddings[u]`, while descendant and
            non-descendant vectors come from `comparison_embeddings[v]`. This is intended for cross-split
            evaluations such as A->B or B->A using a shared hierarchy.

    Returns:
        Dictionary containing:
        - per_node_results: List of dicts with theta_i, violation_proportion, absolute/relative violations, etc.
        - aggregated_statistics: Overall statistics (mean, median, percentiles)
        - depth_stratified: Statistics grouped by depth level
        - all_violations: (if save_all_violations=True) List of all violating pairs
        - template_matched: (if use_individual_template_embeddings=True) per-template breakdown (template index
          matched across nodes) including mean violation % and mean theta per template
        - cone_boundary_percentile: The percentile used for cone boundary calculation
        - level_filtered_non_descendants: Whether level filtering was applied
        - use_individual_template_embeddings: Whether template-level evaluation was used
    """
    from .hierarchy import compute_node_depths

    level_filter_str = " (level-filtered)" if level_filtered_non_descendants else ""
    template_str = " (individual templates)" if use_individual_template_embeddings else ""
    print(f"Testing branch separation for {embedding_type} embeddings{level_filter_str}{template_str}...")

    results = {
        'model_key': model_key,
        'embedding_type': embedding_type,
        'cone_boundary_percentile': cone_boundary_percentile,
        'level_filtered_non_descendants': level_filtered_non_descendants,
        'use_individual_template_embeddings': use_individual_template_embeddings,
        'per_node_results': [],
        'aggregated_statistics': {},
        'depth_stratified': {}
    }

    if use_individual_template_embeddings and save_all_violations:
        raise ValueError(
            "save_all_violations=True is not supported when use_individual_template_embeddings=True "
            "(would produce a very large list of (node, template) pairs)."
        )
    if use_individual_template_embeddings and comparison_embeddings is not None:
        raise ValueError(
            "comparison_embeddings is not supported when use_individual_template_embeddings=True."
        )

    if save_all_violations:
        results['all_violations'] = []

    node_depths = compute_node_depths(G)
    all_nodes = set(G.nodes())
    comparison_embeddings = embeddings if comparison_embeddings is None else comparison_embeddings

    # Find non-leaf nodes with at least one child
    non_leaf_nodes = [n for n in G.nodes() if len(list(G.successors(n))) >= 1]
    print(f"Found {len(non_leaf_nodes)} non-leaf nodes with children")

    if not non_leaf_nodes:
        print("No non-leaf nodes with children found")
        return results

    violation_percentages = []
    theta_values = []  # cone radii (theta_i)
    mean_exceeding_violations = []
    mean_exceeding_margins = []
    max_exceeding_violations = []
    std_exceeding_violations = []
    mean_relative_exceeding_violations = []
    max_relative_exceeding_violations = []
    std_relative_exceeding_violations = []
    depth_violations = {}
    depth_theta_values = {}  # cone radii by depth
    depth_exceeding_violations = {}
    depth_exceeding_margins = {}
    depth_max_exceeding = {}
    depth_std_exceeding = {}
    depth_mean_relative_exceeding = {}
    depth_max_relative_exceeding = {}
    depth_std_relative_exceeding = {}

    template_matched_violation_percentages = []
    template_matched_theta_degs = []
    template_matched_mean_exceeding_violation_degs = []
    num_templates = None
    nodes_with_individual = None
    node_to_index = None
    individual_stack = None
    individual_norms = None

    if use_individual_template_embeddings:
        if individual_embeddings is None:
            raise ValueError("individual_embeddings must be provided when use_individual_template_embeddings=True")
        nodes_with_individual = sorted([n for n in G.nodes() if n in embeddings and n in individual_embeddings])
        if not nodes_with_individual:
            print("No nodes with individual embeddings found; returning empty results")
            return results
        first = np.asarray(individual_embeddings[nodes_with_individual[0]])
        if first.ndim != 2:
            raise ValueError(f"Expected individual_embeddings[{nodes_with_individual[0]}] to have shape (T, D)")
        num_templates = int(first.shape[0])
        template_dim = int(first.shape[1])
        for n in nodes_with_individual:
            arr = np.asarray(individual_embeddings[n])
            if arr.shape != (num_templates, template_dim):
                raise ValueError(
                    f"Inconsistent individual embedding shape for node {n}: expected {(num_templates, template_dim)}, got {arr.shape}"
                )
        node_to_index = {n: i for i, n in enumerate(nodes_with_individual)}
        individual_stack = np.stack(
            [np.asarray(individual_embeddings[n], dtype=np.float32) for n in nodes_with_individual],
            axis=0,
        )
        individual_norms = np.linalg.norm(individual_stack, axis=2)
        results['num_templates'] = int(num_templates)

    for node in tqdm(non_leaf_nodes, desc="Computing branch separation"):
        if node not in embeddings:
            continue

        node_vec = embeddings[node]
        node_norm = np.linalg.norm(node_vec)

        if node_norm <= 1e-8:
            continue

        # Get all descendants
        descendants = set(nx.descendants(G, node))
        if not descendants:
            continue

        theta_by_template = None
        if use_individual_template_embeddings:
            desc_nodes = sorted([d for d in descendants if d in node_to_index])
            if not desc_nodes:
                continue
            desc_indices = [node_to_index[d] for d in desc_nodes]
            desc_stack = individual_stack[desc_indices]
            desc_norms = individual_norms[desc_indices]
            valid_desc = desc_norms > 1e-8
            desc_cos = np.full(desc_norms.shape, np.nan, dtype=np.float32)
            desc_dots = np.tensordot(desc_stack, node_vec, axes=([2], [0]))
            desc_cos[valid_desc] = desc_dots[valid_desc] / (desc_norms[valid_desc] * node_norm)
            desc_angles = np.arccos(np.clip(desc_cos, -1.0, 1.0))
            desc_angles_flat = desc_angles[~np.isnan(desc_angles)]
            if desc_angles_flat.size == 0:
                continue
            max_angle_desc = float(np.percentile(desc_angles_flat, cone_boundary_percentile))
            max_flat = int(np.nanargmax(desc_angles))
            max_desc_row, _ = divmod(max_flat, num_templates)
            max_desc_node = desc_nodes[max_desc_row]
            theta_by_template = np.nanpercentile(desc_angles, cone_boundary_percentile, axis=0)
        else:
            # Collect all descendant angles first
            descendant_angles = []
            desc_angle_map = {}  # Map desc -> angle for finding max_desc_node
            for desc in descendants:
                if desc not in comparison_embeddings:
                    continue

                desc_vec = comparison_embeddings[desc]
                desc_norm = np.linalg.norm(desc_vec)

                if desc_norm > 1e-8:
                    cos_sim = np.dot(node_vec, desc_vec) / (node_norm * desc_norm)
                    angle = np.arccos(np.clip(cos_sim, -1, 1))
                    descendant_angles.append(angle)
                    desc_angle_map[desc] = angle

            # If no valid descendants found, skip
            if not descendant_angles:
                continue

            # Compute theta_i using percentile (100 = max for backwards compatibility)
            max_angle_desc = np.percentile(descendant_angles, cone_boundary_percentile)

            # Find which descendant is at/near the boundary angle (for reporting)
            max_desc_node = max(desc_angle_map, key=desc_angle_map.get)

        # Get non-descendants: all nodes - {node} - ancestors - descendants
        ancestors = set(nx.ancestors(G, node))
        non_descendants = all_nodes - {node} - ancestors - descendants

        # Filter to only nodes with embeddings
        non_descendants = [nd for nd in non_descendants if nd in comparison_embeddings]

        # If level filtering is enabled, only keep non-descendants at deeper levels
        if level_filtered_non_descendants:
            node_depth = node_depths.get(node, 0)
            non_descendants = [nd for nd in non_descendants if node_depths.get(nd, 0) > node_depth]

        if not non_descendants:
            continue

        # Compute metrics for non-descendants
        violations = 0
        violation_nodes = []
        total_pairs = None
        if use_individual_template_embeddings:
            non_desc_nodes = sorted([nd for nd in non_descendants if nd in node_to_index])
            if not non_desc_nodes:
                continue
            non_desc_indices = [node_to_index[nd] for nd in non_desc_nodes]
            non_desc_stack = individual_stack[non_desc_indices]
            non_desc_norms = individual_norms[non_desc_indices]
            valid_non_desc = non_desc_norms > 1e-8
            non_desc_cos = np.full(non_desc_norms.shape, np.nan, dtype=np.float32)
            non_desc_dots = np.tensordot(non_desc_stack, node_vec, axes=([2], [0]))
            non_desc_cos[valid_non_desc] = non_desc_dots[valid_non_desc] / (non_desc_norms[valid_non_desc] * node_norm)
            non_desc_angles = np.arccos(np.clip(non_desc_cos, -1.0, 1.0))
            non_desc_angles_flat = non_desc_angles[~np.isnan(non_desc_angles)]
            if non_desc_angles_flat.size == 0:
                continue

            violations = int(np.sum(non_desc_angles_flat < max_angle_desc))
            total_pairs = int(non_desc_angles_flat.size)
            violation_proportion = violations / total_pairs if total_pairs else 0.0
            violation_percentage = violation_proportion * 100.0
            violation_percentages.append(violation_percentage)

            violating_flat_mask = non_desc_angles_flat < max_angle_desc
            exceeding_angles_violation = max_angle_desc - non_desc_angles_flat[violating_flat_mask]
            exceeding_angles_margin = np.maximum(0.0, non_desc_angles_flat - max_angle_desc)
            if max_angle_desc > 1e-8:
                relative_exceeding_angles_violation = exceeding_angles_violation / max_angle_desc
            else:
                relative_exceeding_angles_violation = np.zeros_like(exceeding_angles_violation)

            # Store top-5 violating (node, template) pairs for quick inspection
            violating_mask = non_desc_angles < max_angle_desc
            if np.any(violating_mask):
                flat_angles = non_desc_angles.reshape(-1)
                violating_indices = np.nonzero(violating_mask.reshape(-1))[0]
                k = min(5, len(violating_indices))
                part = np.argpartition(flat_angles[violating_indices], k - 1)[:k]
                topk_flat = violating_indices[part]
                topk_flat = topk_flat[np.argsort(flat_angles[topk_flat])]
                for flat_idx in topk_flat:
                    row, template_idx = divmod(int(flat_idx), num_templates)
                    viol_node = non_desc_nodes[row]
                    non_desc_depth = node_depths.get(viol_node, -1)
                    violation_nodes.append({
                        'node_id': viol_node,
                        'node_name': synset_to_name.get(viol_node, viol_node).split(",")[0] if synset_to_name else viol_node,
                        'template_index': int(template_idx),
                        'angle_rad': float(non_desc_angles[row, template_idx]),
                        'angle_deg': float(np.degrees(non_desc_angles[row, template_idx])),
                        'violator_depth': non_desc_depth
                    })

            if theta_by_template is not None:
                theta_by_template = np.asarray(theta_by_template, dtype=np.float32)
                template_theta_deg = np.degrees(theta_by_template).astype(np.float32)
                template_matched_theta_degs.append(template_theta_deg)

                valid_by_template = ~np.isnan(non_desc_angles)
                violations_by_template = np.sum((non_desc_angles < theta_by_template[None, :]) & valid_by_template, axis=0)
                totals_by_template = np.sum(valid_by_template, axis=0)
                template_violation_pct = np.full((num_templates,), np.nan, dtype=np.float32)
                nonzero = totals_by_template > 0
                valid_theta = ~np.isnan(theta_by_template)
                eligible = nonzero & valid_theta
                template_violation_pct[eligible] = (100.0 * violations_by_template[eligible] / totals_by_template[eligible]).astype(np.float32)
                template_matched_violation_percentages.append(template_violation_pct)

                exceeding_by_template = np.maximum(0.0, theta_by_template[None, :] - non_desc_angles)
                template_mean_exc_violation = np.full((num_templates,), np.nan, dtype=np.float32)
                summed = np.nansum(exceeding_by_template, axis=0)
                violation_counts = violations_by_template.astype(np.float32)
                has_violations = violation_counts > 0
                eligible_with_violations = eligible & has_violations
                template_mean_exc_violation[eligible_with_violations] = (
                    summed[eligible_with_violations] / violation_counts[eligible_with_violations]
                ).astype(np.float32)
                template_mean_exc_violation[eligible & ~has_violations] = 0.0
                template_matched_mean_exceeding_violation_degs.append(np.degrees(template_mean_exc_violation).astype(np.float32))
        else:
            exceeding_angles_violation = []  # theta_i - angle over violating non-descendants only
            exceeding_angles_margin = []  # ReLU(angle - theta_i) over all non-descendants
            relative_exceeding_angles_violation = []  # (theta_i - angle) / theta_i over violating non-descendants only

            for non_desc in non_descendants:
                non_desc_vec = comparison_embeddings[non_desc]
                non_desc_norm = np.linalg.norm(non_desc_vec)

                if non_desc_norm > 1e-8:
                    cos_sim = np.dot(node_vec, non_desc_vec) / (node_norm * non_desc_norm)
                    angle = np.arccos(np.clip(cos_sim, -1, 1))

                    # Violation metrics
                    if angle < max_angle_desc:
                        violations += 1
                        non_desc_depth = node_depths.get(non_desc, -1)
                        exceeding_angle = max_angle_desc - angle
                        violation_data = {
                            'node_id': non_desc,
                            'node_name': synset_to_name.get(non_desc, non_desc).split(",")[0] if synset_to_name else non_desc,
                            'angle_rad': float(angle),
                            'angle_deg': float(np.degrees(angle)),
                            'violator_depth': non_desc_depth
                        }
                        violation_nodes.append(violation_data)
                        exceeding_angles_violation.append(exceeding_angle)

                        # Relative violation: normalize by cone size (theta_i)
                        if max_angle_desc > 1e-8:
                            relative_exceeding_angles_violation.append(exceeding_angle / max_angle_desc)
                        else:
                            relative_exceeding_angles_violation.append(0.0)

                        # If saving all violations, add full details
                        if save_all_violations:
                            # Get names and optionally trim to first part before comma
                            ancestor_name = synset_to_name.get(node, node) if synset_to_name else node
                            violator_name = synset_to_name.get(non_desc, non_desc) if synset_to_name else non_desc

                            if use_first_name_only:
                                if ',' in ancestor_name:
                                    ancestor_name = ancestor_name.split(',')[0].strip()
                                if ',' in violator_name:
                                    violator_name = violator_name.split(',')[0].strip()

                            results['all_violations'].append({
                                'ancestor_id': node,
                                'ancestor_name': ancestor_name,
                                'ancestor_depth': node_depths.get(node, -1),
                                'violator_id': non_desc,
                                'violator_name': violator_name,
                                'violator_depth': non_desc_depth,
                                'theta_i_deg': float(np.degrees(max_angle_desc)),
                                'angle_deg': float(np.degrees(angle)),
                                'exceeding_angle_deg': float(np.degrees(exceeding_angle))
                            })

                    # Exceeding angle metrics
                    exceeding_margin = max(0.0, angle - max_angle_desc)  # ReLU(angle - theta_i)
                    exceeding_angles_margin.append(exceeding_margin)

            violation_proportion = violations / len(non_descendants) if non_descendants else 0.0
            violation_percentage = violation_proportion * 100.0
            violation_percentages.append(violation_percentage)

        # Compute exceeding angle statistics
        exceeding_angles_violation = np.array(exceeding_angles_violation)
        exceeding_angles_margin = np.array(exceeding_angles_margin)
        relative_exceeding_angles_violation = np.array(relative_exceeding_angles_violation)

        mean_exceeding_violation = float(np.mean(exceeding_angles_violation)) if len(exceeding_angles_violation) > 0 else 0.0
        mean_exceeding_margin = float(np.mean(exceeding_angles_margin)) if len(exceeding_angles_margin) > 0 else 0.0
        max_exceeding_violation = float(np.max(exceeding_angles_violation)) if len(exceeding_angles_violation) > 0 else 0.0
        std_exceeding_violation = float(np.std(exceeding_angles_violation)) if len(exceeding_angles_violation) > 0 else 0.0
        mean_relative_exceeding_violation = float(np.mean(relative_exceeding_angles_violation)) if len(relative_exceeding_angles_violation) > 0 else 0.0
        max_relative_exceeding_violation = float(np.max(relative_exceeding_angles_violation)) if len(relative_exceeding_angles_violation) > 0 else 0.0
        std_relative_exceeding_violation = float(np.std(relative_exceeding_angles_violation)) if len(relative_exceeding_angles_violation) > 0 else 0.0

        # Collect for aggregation
        theta_values.append(max_angle_desc)
        mean_exceeding_violations.append(mean_exceeding_violation)
        mean_exceeding_margins.append(mean_exceeding_margin)
        max_exceeding_violations.append(max_exceeding_violation)
        std_exceeding_violations.append(std_exceeding_violation)
        mean_relative_exceeding_violations.append(mean_relative_exceeding_violation)
        max_relative_exceeding_violations.append(max_relative_exceeding_violation)
        std_relative_exceeding_violations.append(std_relative_exceeding_violation)

        # Store per-node results
        node_depth = node_depths.get(node, -1)
        short_node_name = synset_to_name.get(node, node).split(",")[0] if synset_to_name else node
        short_max_desc_name = synset_to_name.get(max_desc_node, max_desc_node).split(",")[0] if synset_to_name else max_desc_node

        node_result = {
            'node_id': node,
            'node_name': short_node_name,
            'depth': node_depth,
            'theta_i_rad': float(max_angle_desc),
            'theta_i_deg': float(np.degrees(max_angle_desc)),
            'max_desc_node': short_max_desc_name,
            'num_descendants': len(descendants),
            'num_non_descendants': len(non_descendants),
            'num_violations': violations,
            'violation_proportion': float(violation_percentage),
            'mean_exceeding_angle_violation_rad': mean_exceeding_violation,
            'mean_exceeding_angle_violation_deg': float(np.degrees(mean_exceeding_violation)),
            'mean_exceeding_angle_margin_rad': mean_exceeding_margin,
            'mean_exceeding_angle_margin_deg': float(np.degrees(mean_exceeding_margin)),
            'max_exceeding_angle_rad': max_exceeding_violation,
            'max_exceeding_angle_deg': float(np.degrees(max_exceeding_violation)),
            'std_exceeding_angle_rad': std_exceeding_violation,
            'std_exceeding_angle_deg': float(np.degrees(std_exceeding_violation)),
            'mean_relative_exceeding_angle_violation': mean_relative_exceeding_violation,
            'max_relative_exceeding_angle_violation': max_relative_exceeding_violation,
            'std_relative_exceeding_angle_violation': std_relative_exceeding_violation,
            'top_violations': violation_nodes[:5] if violation_nodes else []
        }
        if total_pairs is not None:
            node_result['num_non_descendant_pairs'] = int(total_pairs)
        results['per_node_results'].append(node_result)

        # Group by depth
        if node_depth not in depth_violations:
            depth_violations[node_depth] = []
            depth_theta_values[node_depth] = []
            depth_exceeding_violations[node_depth] = []
            depth_exceeding_margins[node_depth] = []
            depth_max_exceeding[node_depth] = []
            depth_std_exceeding[node_depth] = []
            depth_mean_relative_exceeding[node_depth] = []
            depth_max_relative_exceeding[node_depth] = []
            depth_std_relative_exceeding[node_depth] = []
        depth_violations[node_depth].append(violation_percentage)
        depth_theta_values[node_depth].append(max_angle_desc)
        depth_exceeding_violations[node_depth].append(mean_exceeding_violation)
        depth_exceeding_margins[node_depth].append(mean_exceeding_margin)
        depth_max_exceeding[node_depth].append(max_exceeding_violation)
        depth_std_exceeding[node_depth].append(std_exceeding_violation)
        depth_mean_relative_exceeding[node_depth].append(mean_relative_exceeding_violation)
        depth_max_relative_exceeding[node_depth].append(max_relative_exceeding_violation)
        depth_std_relative_exceeding[node_depth].append(std_relative_exceeding_violation)

    # Compute aggregated statistics
    if violation_percentages:
        violation_percentages = np.array(violation_percentages)
        theta_values = np.array(theta_values)
        mean_exceeding_violations = np.array(mean_exceeding_violations)
        mean_exceeding_margins = np.array(mean_exceeding_margins)
        max_exceeding_violations = np.array(max_exceeding_violations)
        std_exceeding_violations = np.array(std_exceeding_violations)
        mean_relative_exceeding_violations = np.array(mean_relative_exceeding_violations)
        max_relative_exceeding_violations = np.array(max_relative_exceeding_violations)
        std_relative_exceeding_violations = np.array(std_relative_exceeding_violations)

        results['aggregated_statistics'] = {
            'cone_radii': {
                'mean_rad': float(np.mean(theta_values)),
                'mean_deg': float(np.degrees(np.mean(theta_values))),
                'std_rad': float(np.std(theta_values)),
                'std_deg': float(np.degrees(np.std(theta_values))),
                'median_rad': float(np.median(theta_values)),
                'median_deg': float(np.degrees(np.median(theta_values))),
                'min_rad': float(np.min(theta_values)),
                'min_deg': float(np.degrees(np.min(theta_values))),
                'max_rad': float(np.max(theta_values)),
                'max_deg': float(np.degrees(np.max(theta_values))),
                'percentile_25_deg': float(np.degrees(np.percentile(theta_values, 25))),
                'percentile_75_deg': float(np.degrees(np.percentile(theta_values, 75))),
                'percentile_90_deg': float(np.degrees(np.percentile(theta_values, 90)))
            },
            'violation_proportion': {
                'mean': float(np.mean(violation_percentages)),
                'std': float(np.std(violation_percentages)),
                'median': float(np.median(violation_percentages)),
                'min': float(np.min(violation_percentages)),
                'max': float(np.max(violation_percentages)),
                'percentile_25': float(np.percentile(violation_percentages, 25)),
                'percentile_75': float(np.percentile(violation_percentages, 75)),
                'percentile_90': float(np.percentile(violation_percentages, 90))
            },
            'mean_exceeding_angle_violation': {
                'mean_rad': float(np.mean(mean_exceeding_violations)),
                'mean_deg': float(np.degrees(np.mean(mean_exceeding_violations))),
                'std_rad': float(np.std(mean_exceeding_violations)),
                'std_deg': float(np.degrees(np.std(mean_exceeding_violations))),
                'median_rad': float(np.median(mean_exceeding_violations)),
                'median_deg': float(np.degrees(np.median(mean_exceeding_violations))),
                'max_rad': float(np.max(mean_exceeding_violations)),
                'max_deg': float(np.degrees(np.max(mean_exceeding_violations)))
            },
            'mean_exceeding_angle_margin': {
                'mean_rad': float(np.mean(mean_exceeding_margins)),
                'mean_deg': float(np.degrees(np.mean(mean_exceeding_margins))),
                'std_rad': float(np.std(mean_exceeding_margins)),
                'std_deg': float(np.degrees(np.std(mean_exceeding_margins))),
                'median_rad': float(np.median(mean_exceeding_margins)),
                'median_deg': float(np.degrees(np.median(mean_exceeding_margins))),
                'min_rad': float(np.min(mean_exceeding_margins)),
                'min_deg': float(np.degrees(np.min(mean_exceeding_margins)))
            },
            'max_exceeding_angle': {
                'mean_rad': float(np.mean(max_exceeding_violations)),
                'mean_deg': float(np.degrees(np.mean(max_exceeding_violations))),
                'max_rad': float(np.max(max_exceeding_violations)),
                'max_deg': float(np.degrees(np.max(max_exceeding_violations)))
            },
            'std_exceeding_angle': {
                'mean_rad': float(np.mean(std_exceeding_violations)),
                'mean_deg': float(np.degrees(np.mean(std_exceeding_violations))),
                'max_rad': float(np.max(std_exceeding_violations)),
                'max_deg': float(np.degrees(np.max(std_exceeding_violations)))
            },
            'mean_relative_exceeding_angle_violation': {
                'mean': float(np.mean(mean_relative_exceeding_violations)),
                'std': float(np.std(mean_relative_exceeding_violations)),
                'median': float(np.median(mean_relative_exceeding_violations)),
                'max': float(np.max(mean_relative_exceeding_violations))
            },
            'max_relative_exceeding_angle_violation': {
                'mean': float(np.mean(max_relative_exceeding_violations)),
                'std': float(np.std(max_relative_exceeding_violations)),
                'median': float(np.median(max_relative_exceeding_violations)),
                'max': float(np.max(max_relative_exceeding_violations))
            },
            'std_relative_exceeding_angle_violation': {
                'mean': float(np.mean(std_relative_exceeding_violations)),
                'std': float(np.std(std_relative_exceeding_violations)),
                'median': float(np.median(std_relative_exceeding_violations)),
                'max': float(np.max(std_relative_exceeding_violations))
            },
            'num_nodes': len(violation_percentages)
        }

        print(f"Branch separation statistics:")
        print(f"  Cone radii (theta_i): {results['aggregated_statistics']['cone_radii']['mean_deg']:.2f}° ± {results['aggregated_statistics']['cone_radii']['std_deg']:.2f}°")
        print(f"  Violation proportion: {results['aggregated_statistics']['violation_proportion']['mean']:.2f}% ± {results['aggregated_statistics']['violation_proportion']['std']:.2f}%")
        print(f"  Mean exceeding angle (violations only): {results['aggregated_statistics']['mean_exceeding_angle_violation']['mean_deg']:.2f}° ± {results['aggregated_statistics']['mean_exceeding_angle_violation']['std_deg']:.2f}°")
        print(f"  Mean exceeding angle (margin): {results['aggregated_statistics']['mean_exceeding_angle_margin']['mean_deg']:.2f}° ± {results['aggregated_statistics']['mean_exceeding_angle_margin']['std_deg']:.2f}°")
        print(f"  Max exceeding angle (worst case): {results['aggregated_statistics']['max_exceeding_angle']['max_deg']:.2f}°")
        print(f"  Mean relative violation (violations only; exceed/theta_i): {results['aggregated_statistics']['mean_relative_exceeding_angle_violation']['mean']:.4f} ± {results['aggregated_statistics']['mean_relative_exceeding_angle_violation']['std']:.4f}")

    if use_individual_template_embeddings and template_matched_violation_percentages:
        template_violation_matrix = np.stack(template_matched_violation_percentages, axis=0)
        template_theta_matrix = np.stack(template_matched_theta_degs, axis=0)
        template_mean_exc_violation_matrix = None
        if template_matched_mean_exceeding_violation_degs:
            template_mean_exc_violation_matrix = np.stack(template_matched_mean_exceeding_violation_degs, axis=0)
        results['template_matched'] = {
            'num_templates': int(num_templates),
            'num_nodes': int(template_violation_matrix.shape[0]),
            'cone_boundary_percentile': float(cone_boundary_percentile),
            'violation_percentage_mean_by_template': np.nanmean(template_violation_matrix, axis=0).tolist(),
            'theta_deg_mean_by_template': np.nanmean(template_theta_matrix, axis=0).tolist(),
            'violation_percentage_by_node_template': template_violation_matrix.tolist(),
            'theta_deg_by_node_template': template_theta_matrix.tolist(),
        }
        if template_mean_exc_violation_matrix is not None:
            results['template_matched']['mean_exceeding_angle_violation_deg_mean_by_template'] = np.nanmean(template_mean_exc_violation_matrix, axis=0).tolist()
            results['template_matched']['mean_exceeding_angle_violation_deg_by_node_template'] = template_mean_exc_violation_matrix.tolist()

    # Compute depth-stratified statistics
    for depth in depth_violations.keys():
        depth_vps = np.array(depth_violations[depth])
        depth_thetas = np.array(depth_theta_values[depth])
        depth_exc_viol = np.array(depth_exceeding_violations[depth])
        depth_exc_marg = np.array(depth_exceeding_margins[depth])
        depth_max_exc = np.array(depth_max_exceeding[depth])
        depth_std_exc = np.array(depth_std_exceeding[depth])
        depth_mean_rel_exc = np.array(depth_mean_relative_exceeding[depth])
        depth_max_rel_exc = np.array(depth_max_relative_exceeding[depth])
        depth_std_rel_exc = np.array(depth_std_relative_exceeding[depth])

        results['depth_stratified'][depth] = {
            'cone_radii_deg': {
                'mean': float(np.degrees(np.mean(depth_thetas))),
                'std': float(np.degrees(np.std(depth_thetas))),
                'median': float(np.degrees(np.median(depth_thetas))),
                'min': float(np.degrees(np.min(depth_thetas))),
                'max': float(np.degrees(np.max(depth_thetas)))
            },
            'violation_proportion': {
                'mean': float(np.mean(depth_vps)),
                'std': float(np.std(depth_vps)),
                'median': float(np.median(depth_vps))
            },
            'mean_exceeding_angle_violation_deg': {
                'mean': float(np.degrees(np.mean(depth_exc_viol))),
                'std': float(np.degrees(np.std(depth_exc_viol)))
            },
            'mean_exceeding_angle_margin_deg': {
                'mean': float(np.degrees(np.mean(depth_exc_marg))),
                'std': float(np.degrees(np.std(depth_exc_marg)))
            },
            'max_exceeding_angle_deg': {
                'mean': float(np.degrees(np.mean(depth_max_exc))),
                'max': float(np.degrees(np.max(depth_max_exc)))
            },
            'mean_relative_exceeding_angle_violation': {
                'mean': float(np.mean(depth_mean_rel_exc)),
                'std': float(np.std(depth_mean_rel_exc)),
                'median': float(np.median(depth_mean_rel_exc))
            },
            'max_relative_exceeding_angle_violation': {
                'mean': float(np.mean(depth_max_rel_exc)),
                'std': float(np.std(depth_max_rel_exc)),
                'max': float(np.max(depth_max_rel_exc))
            },
            'std_relative_exceeding_angle_violation': {
                'mean': float(np.mean(depth_std_rel_exc)),
                'std': float(np.std(depth_std_rel_exc))
            },
            'count': len(depth_vps)
        }

    if results['depth_stratified']:
        print(f"  Depth-stratified analysis:")
        for depth in sorted(results['depth_stratified'].keys()):
            stats = results['depth_stratified'][depth]
            cone = stats['cone_radii_deg']
            vp = stats['violation_proportion']
            exc_viol = stats['mean_exceeding_angle_violation_deg']
            print(f"    Depth {depth}: {stats['count']} nodes, cone_radii={cone['mean']:.2f}°, viol_prop={vp['mean']:.2f}%, exc_angle={exc_viol['mean']:.2f}°")

    return results

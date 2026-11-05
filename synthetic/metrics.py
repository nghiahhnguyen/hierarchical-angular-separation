"""Metrics for measuring hierarchical learning signatures."""

import numpy as np
from functools import lru_cache
from hierarchy import Tree, Node
from model import Checkpoint


# =============================================================================
# Tree Structure Cache
# =============================================================================

def precompute_tree_structure(tree: Tree) -> dict:
    """
    Precompute tree relationships for all nodes in a single pass.

    Returns dict with:
        'descendants': {node_id: set of descendant ids}
        'ancestors': {node_id: set of ancestor ids}
        'non_descendants': {node_id: set of ids that are not self/ancestor/descendant}
        'leaf_descendant_ids': {node_id: list of leaf descendant ids}
        'descendant_arrays': {node_id: sorted np.ndarray of descendant ids}
        'non_descendant_arrays': {node_id: sorted np.ndarray of non-descendant ids}
        'internal_nodes_by_depth': {depth: list of non-leaf node ids}
        'internal_node_ids': list of all non-leaf node ids
    """
    all_nodes = tree.get_all_nodes()
    all_ids = set(n.id for n in all_nodes)
    max_depth = tree.get_max_depth()
    descendants = {}
    leaf_descendant_ids = {}

    # Bottom-up: process nodes in reverse BFS order (deepest first)
    for node in reversed(all_nodes):
        if node.is_leaf():
            descendants[node.id] = set()
            leaf_descendant_ids[node.id] = [node.id]
        else:
            desc = set()
            leaves = []
            for child in node.children:
                desc.add(child.id)
                desc.update(descendants[child.id])
                leaves.extend(leaf_descendant_ids[child.id])
            descendants[node.id] = desc
            leaf_descendant_ids[node.id] = leaves

    # Ancestors: walk up from each node
    ancestors = {}
    for node in all_nodes:
        anc = set()
        current = node.parent
        while current is not None:
            anc.add(current.id)
            current = current.parent
        ancestors[node.id] = anc

    # Non-descendants
    non_descendants = {}
    for node in all_nodes:
        non_descendants[node.id] = all_ids - {node.id} - ancestors[node.id] - descendants[node.id]

    descendant_arrays = {
        node_id: np.array(sorted(node_ids), dtype=np.int64)
        for node_id, node_ids in descendants.items()
    }
    non_descendant_arrays = {
        node_id: np.array(sorted(node_ids), dtype=np.int64)
        for node_id, node_ids in non_descendants.items()
    }
    internal_nodes_by_depth = {depth: [] for depth in range(max_depth)}
    internal_node_ids = []
    for node in all_nodes:
        if node.is_leaf():
            continue
        internal_nodes_by_depth[node.depth].append(node.id)
        internal_node_ids.append(node.id)

    return {
        'descendants': descendants,
        'ancestors': ancestors,
        'non_descendants': non_descendants,
        'leaf_descendant_ids': leaf_descendant_ids,
        'descendant_arrays': descendant_arrays,
        'non_descendant_arrays': non_descendant_arrays,
        'internal_nodes_by_depth': internal_nodes_by_depth,
        'internal_node_ids': internal_node_ids,
    }


def build_mean_descendant_matrix(W: np.ndarray, tree_cache: dict) -> np.ndarray:
    """
    Build matrix where each row is the mean of that node's leaf descendant embeddings.

    For leaves, returns the leaf embedding directly.
    """
    W_mean = np.empty_like(W)
    for node_id, leaf_ids in tree_cache['leaf_descendant_ids'].items():
        W_mean[node_id] = W[leaf_ids].mean(axis=0)
    return W_mean


def _prepare_leaf_image_stats(leaf_assignments: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute assignment index and per-node image counts for H aggregation."""
    leaf_ids = np.asarray(leaf_assignments, dtype=np.int64)
    counts = np.bincount(leaf_ids, minlength=num_nodes).astype(np.float64, copy=False)
    return leaf_ids, counts


def _build_h_by_node_from_images(
    H: np.ndarray,
    W_shape: tuple[int, int],
    leaf_ids: np.ndarray,
    leaf_counts: np.ndarray,
) -> np.ndarray:
    """Aggregate image embeddings to leaf nodes via indexed accumulation."""
    H_by_node = np.zeros(W_shape, dtype=H.dtype)
    np.add.at(H_by_node, leaf_ids, H.T)
    valid_nodes = leaf_counts > 0
    if np.any(valid_nodes):
        H_by_node[valid_nodes] /= leaf_counts[valid_nodes, None]
    return H_by_node


# =============================================================================
# Embedding Types: Direct (from W) vs Mean-of-Descendants (from W or H)
# =============================================================================

def compute_mean_descendant_embedding(W: np.ndarray, node: Node, tree: Tree) -> np.ndarray:
    """
    Compute embedding for a node as the mean of all its descendant leaf embeddings.

    For leaves, returns the leaf embedding directly.
    For internal nodes, returns mean of all leaf descendants.

    Args:
        W: Embedding matrix (V, d) - can be text (W) or image (H.T) embeddings
        node: Node to compute embedding for
        tree: Hierarchy tree

    Returns:
        Embedding vector (d,)
    """
    if node.is_leaf():
        return W[node.id, :]

    # Collect all leaf descendants
    def get_leaf_descendants(n: Node) -> list[Node]:
        if n.is_leaf():
            return [n]
        leaves = []
        for child in n.children:
            leaves.extend(get_leaf_descendants(child))
        return leaves

    leaves = get_leaf_descendants(node)
    leaf_embeddings = np.array([W[leaf.id, :] for leaf in leaves])
    return leaf_embeddings.mean(axis=0)


# =============================================================================
# Signature 1: Spectral / Coarse-to-Fine Learning Dynamics
# =============================================================================

def compute_svd_of_centered_support(S_bar: np.ndarray, k: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD of centered support matrix.

    Args:
        S_bar: Centered support matrix (V, m)
        k: Number of singular values to keep (None = all)

    Returns:
        U: Left singular vectors (V, k)
        sigma: Singular values (k,)
        Vt: Right singular vectors (k, m)
    """
    U, sigma, Vt = np.linalg.svd(S_bar, full_matrices=False)
    if k is not None:
        U, sigma, Vt = U[:, :k], sigma[:k], Vt[:k, :]
    return U, sigma, Vt


def compute_projection_onto_svd_subspace(H: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Project H onto top-k right singular subspace of S_bar.

    Args:
        H: Image embeddings (d, m)
        Vt: Right singular vectors from SVD of S_bar (k_max, m)
        k: Use top-k singular vectors

    Returns:
        Projection norms for each singular direction (k,)
    """
    # H is (d, m), Vt[:k] is (k, m)
    # For each singular direction i, compute ||H @ v_i||^2 where v_i is the i-th row of Vt
    projections = np.zeros(k)
    for i in range(k):
        v_i = Vt[i, :]  # (m,)
        proj = H @ v_i  # (d,)
        projections[i] = np.linalg.norm(proj)
    return projections


def track_spectral_alignment(checkpoints: list[Checkpoint], S_bar: np.ndarray, k: int = 10) -> dict:
    """
    Track alignment of H with top-k singular directions over training.

    Args:
        checkpoints: List of training checkpoints
        S_bar: Centered support matrix
        k: Number of singular directions to track

    Returns:
        Dictionary with:
            'epochs': list of epochs
            'projections': array (num_checkpoints, k) of projection norms
            'singular_values': array (k,) of singular values
    """
    _, sigma, Vt = compute_svd_of_centered_support(S_bar, k=k)

    epochs = []
    projections = []
    for ckpt in checkpoints:
        epochs.append(ckpt.epoch)
        proj = compute_projection_onto_svd_subspace(ckpt.H, Vt, k)
        projections.append(proj)

    return {
        'epochs': np.array(epochs),
        'projections': np.array(projections),  # (num_checkpoints, k)
        'singular_values': sigma
    }


# =============================================================================
# Signature 2: Hierarchical Orthogonality
# =============================================================================

def compute_concept_embedding(W: np.ndarray, node_id: int) -> np.ndarray:
    """Get embedding for a concept (row of W)."""
    return W[node_id, :]


def compute_orthogonality_score(W: np.ndarray, parent: Node, child: Node) -> float:
    """
    Compute orthogonality score: cos(p, c - p) where p is parent embedding, c is child embedding.

    Perfect orthogonality: cos = 0
    """
    p = compute_concept_embedding(W, parent.id)
    c = compute_concept_embedding(W, child.id)
    residual = c - p

    p_norm = np.linalg.norm(p)
    r_norm = np.linalg.norm(residual)
    if p_norm < 1e-10 or r_norm < 1e-10:
        return 0.0

    return np.dot(p, residual) / (p_norm * r_norm)


def aggregate_orthogonality_by_depth(W: np.ndarray, tree: Tree) -> dict[int, list[float]]:
    """
    Compute orthogonality scores for all parent-child pairs, grouped by child depth.

    Args:
        W: Concept embedding matrix (V, d)
        tree: Hierarchy tree

    Returns:
        Dictionary mapping depth -> list of orthogonality scores
    """
    scores_by_depth = {}
    max_depth = tree.get_max_depth()

    for depth in range(1, max_depth + 1):
        nodes = tree.get_nodes_at_depth(depth)
        scores = []
        for node in nodes:
            if node.parent is not None:
                score = compute_orthogonality_score(W, node.parent, node)
                scores.append(score)
        scores_by_depth[depth] = scores

    return scores_by_depth


def compute_orthogonality_score_mean_descendants(W: np.ndarray, parent: Node, child: Node, tree: Tree) -> float:
    """
    Compute orthogonality using mean-of-descendants definition for embeddings.

    Parent embedding = mean of all leaf descendants of parent
    Child embedding = mean of all leaf descendants of child
    """
    p = compute_mean_descendant_embedding(W, parent, tree)
    c = compute_mean_descendant_embedding(W, child, tree)
    residual = c - p

    p_norm = np.linalg.norm(p)
    r_norm = np.linalg.norm(residual)
    if p_norm < 1e-10 or r_norm < 1e-10:
        return 0.0
    return np.dot(p, residual) / (p_norm * r_norm)


def aggregate_orthogonality_by_depth_mean_descendants(W: np.ndarray, tree: Tree) -> dict[int, list[float]]:
    """Compute orthogonality using mean-of-descendants, grouped by child depth."""
    scores_by_depth = {}
    max_depth = tree.get_max_depth()

    for depth in range(1, max_depth + 1):
        nodes = tree.get_nodes_at_depth(depth)
        scores = []
        for node in nodes:
            if node.parent is not None:
                score = compute_orthogonality_score_mean_descendants(W, node.parent, node, tree)
                scores.append(score)
        scores_by_depth[depth] = scores
    return scores_by_depth


def aggregate_orthogonality_by_depth_vectorized(W: np.ndarray, tree: Tree) -> dict[int, list[float]]:
    """Vectorized orthogonality: batch parent/child indexing per depth."""
    scores_by_depth = {}
    max_depth = tree.get_max_depth()
    for depth in range(1, max_depth + 1):
        nodes = tree.get_nodes_at_depth(depth)
        pairs = [(node.parent.id, node.id) for node in nodes if node.parent is not None]
        if not pairs:
            scores_by_depth[depth] = []
            continue
        parent_ids, child_ids = zip(*pairs)
        P = W[list(parent_ids)]  # (n, d)
        C = W[list(child_ids)]   # (n, d)
        R = C - P
        p_norms = np.linalg.norm(P, axis=1)
        r_norms = np.linalg.norm(R, axis=1)
        valid = (p_norms >= 1e-10) & (r_norms >= 1e-10)
        scores = np.zeros(len(pairs))
        if np.any(valid):
            scores[valid] = np.sum(P[valid] * R[valid], axis=1) / (p_norms[valid] * r_norms[valid])
        scores_by_depth[depth] = scores.tolist()
    return scores_by_depth


def aggregate_cone_angle_by_depth_vectorized(W: np.ndarray, tree: Tree) -> dict[int, list[float]]:
    """Vectorized cone angles: batch parent/child indexing, pairwise angles per parent."""
    angles_by_depth = {}
    max_depth = tree.get_max_depth()
    for depth in range(1, max_depth + 1):
        parent_depth = depth - 1
        parents = tree.get_nodes_at_depth(parent_depth)
        angles = []
        for parent in parents:
            children = parent.children
            if len(children) < 2:
                continue
            child_ids = [c.id for c in children]
            C = W[child_ids]  # (b, d)
            p = W[parent.id]  # (d,)
            R = C - p  # (b, d)
            r_norms = np.linalg.norm(R, axis=1)
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    n1, n2 = r_norms[i], r_norms[j]
                    if n1 < 1e-10 or n2 < 1e-10:
                        angles.append(0.0)
                    else:
                        cos_angle = np.clip(np.dot(R[i], R[j]) / (n1 * n2), -1.0, 1.0)
                        angles.append(np.degrees(np.arccos(cos_angle)))
        angles_by_depth[depth] = angles
    return angles_by_depth


def track_orthogonality_over_training(checkpoints: list[Checkpoint], tree: Tree) -> dict:
    """
    Track orthogonality scores over training.

    Returns:
        Dictionary with:
            'epochs': list of epochs
            'scores_by_depth': dict mapping depth -> array (num_checkpoints,) of mean scores
    """
    max_depth = tree.get_max_depth()
    epochs = [ckpt.epoch for ckpt in checkpoints]
    scores_by_depth = {d: [] for d in range(1, max_depth + 1)}

    for ckpt in checkpoints:
        depth_scores = aggregate_orthogonality_by_depth(ckpt.W, tree)
        for d in range(1, max_depth + 1):
            mean_score = np.mean(depth_scores[d]) if depth_scores[d] else 0.0
            scores_by_depth[d].append(mean_score)

    return {
        'epochs': np.array(epochs),
        'scores_by_depth': {d: np.array(s) for d, s in scores_by_depth.items()}
    }


def track_orthogonality_over_training_mean_descendants(
    checkpoints: list[Checkpoint], tree: Tree, use_H: bool = False
) -> dict:
    """
    Track orthogonality using mean-of-descendants definition.

    Args:
        checkpoints: Training checkpoints
        tree: Hierarchy tree
        use_H: If True, use H.T (image embeddings) instead of W (text embeddings)

    Returns:
        Dictionary with epochs and scores_by_depth
    """
    max_depth = tree.get_max_depth()
    epochs = [ckpt.epoch for ckpt in checkpoints]
    scores_by_depth = {d: [] for d in range(1, max_depth + 1)}

    for ckpt in checkpoints:
        emb = ckpt.H.T if use_H else ckpt.W  # (m, d) if H, (V, d) if W
        depth_scores = aggregate_orthogonality_by_depth_mean_descendants(emb, tree)
        for d in range(1, max_depth + 1):
            mean_score = np.mean(depth_scores[d]) if depth_scores[d] else 0.0
            scores_by_depth[d].append(mean_score)

    return {
        'epochs': np.array(epochs),
        'scores_by_depth': {d: np.array(s) for d, s in scores_by_depth.items()}
    }


# =============================================================================
# Signature 3: Shrinking Cone Effect
# =============================================================================

def compute_residual(W: np.ndarray, parent: Node, child: Node) -> np.ndarray:
    """Compute residual vector c - p using direct embeddings."""
    p = compute_concept_embedding(W, parent.id)
    c = compute_concept_embedding(W, child.id)
    return c - p


def compute_residual_mean_descendants(W: np.ndarray, parent: Node, child: Node, tree: Tree) -> np.ndarray:
    """Compute residual vector c - p using mean-of-descendants embeddings."""
    p = compute_mean_descendant_embedding(W, parent, tree)
    c = compute_mean_descendant_embedding(W, child, tree)
    return c - p


def compute_pairwise_cone_angle(r1: np.ndarray, r2: np.ndarray) -> float:
    """Compute angle (in degrees) between two residual vectors."""
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_angle = np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_sibling_cone_angles(W: np.ndarray, parent: Node, tree: Tree = None, use_mean_descendants: bool = False) -> list[float]:
    """
    Compute pairwise angles between sibling residual vectors.

    Args:
        W: Embedding matrix
        parent: Parent node whose children are siblings
        tree: Hierarchy tree (required if use_mean_descendants=True)
        use_mean_descendants: If True, use mean-of-descendants for embeddings

    Returns:
        List of pairwise angles between siblings
    """
    children = parent.children
    if len(children) < 2:
        return []

    if use_mean_descendants:
        residuals = [compute_residual_mean_descendants(W, parent, child, tree) for child in children]
    else:
        residuals = [compute_residual(W, parent, child) for child in children]

    angles = []
    for i in range(len(residuals)):
        for j in range(i + 1, len(residuals)):
            angle = compute_pairwise_cone_angle(residuals[i], residuals[j])
            angles.append(angle)
    return angles


def aggregate_cone_angle_by_depth(W: np.ndarray, tree: Tree, use_mean_descendants: bool = False) -> dict[int, list[float]]:
    """
    Compute sibling cone angles grouped by depth.

    Args:
        W: Embedding matrix (V, d)
        tree: Hierarchy tree
        use_mean_descendants: Use mean-of-descendants for parent/child embeddings

    Returns:
        Dictionary mapping depth -> list of sibling cone angles
    """
    angles_by_depth = {}
    max_depth = tree.get_max_depth()

    for depth in range(1, max_depth + 1):
        angles = []
        parent_depth = depth - 1
        parents = tree.get_nodes_at_depth(parent_depth)
        for parent in parents:
            if len(parent.children) >= 2:
                angles.extend(compute_sibling_cone_angles(W, parent, tree, use_mean_descendants))
        angles_by_depth[depth] = angles

    return angles_by_depth


def track_cone_angles_over_training(checkpoints: list[Checkpoint], tree: Tree) -> dict:
    """Track cone angles using direct W embeddings."""
    max_depth = tree.get_max_depth()
    epochs = [ckpt.epoch for ckpt in checkpoints]
    angles_by_depth = {d: [] for d in range(1, max_depth + 1)}

    for ckpt in checkpoints:
        depth_angles = aggregate_cone_angle_by_depth(ckpt.W, tree, use_mean_descendants=False)
        for d in range(1, max_depth + 1):
            mean_angle = np.mean(depth_angles[d]) if depth_angles[d] else 0.0
            angles_by_depth[d].append(mean_angle)

    return {
        'epochs': np.array(epochs),
        'angles_by_depth': {d: np.array(a) for d, a in angles_by_depth.items()}
    }


# =============================================================================
# Unified tracking for all embedding types
# =============================================================================

# =============================================================================
# Signature 4: Branch Separation
# =============================================================================

def compute_branch_separation_for_node(
    embeddings: np.ndarray,
    node: Node,
    tree: Tree,
    node_to_images: dict[int, list[int]] = None,
    cone_boundary_percentile: float = 100.0,
    tree_cache: dict = None
) -> dict:
    """
    Compute branch separation metrics for a single node.

    For node n with embedding e_n:
    - theta_i = percentile of angles to descendants (max when percentile=100)
    - violation_rate = % of non-descendants with angle < theta_i
    - mean_relative_violation = avg((theta_i - angle) / theta_i) for violations

    Args:
        embeddings: Embedding matrix (V, d) or (m, d) for image embeddings
        node: The node to evaluate
        tree: Hierarchy tree
        node_to_images: Optional mapping from node_id to image indices (for H embeddings)
        cone_boundary_percentile: Percentile of descendant angles to use for cone boundary
        tree_cache: Precomputed tree structure from precompute_tree_structure()

    Returns:
        Dict with theta_i, violation_rate, mean_exceeding_angle, mean_relative_violation, cone_boundary_percentile
    """
    if node_to_images is not None:
        # Image embedding path: keep original scalar logic (rare, small overhead)
        return _compute_branch_separation_for_node_images(
            embeddings, node, tree, node_to_images, cone_boundary_percentile, tree_cache
        )

    # Fast path: direct embeddings with vectorized computation
    node_emb = embeddings[node.id]
    node_norm = np.linalg.norm(node_emb)
    if node_norm < 1e-8:
        return None

    # Get descendant/non-descendant sets
    if tree_cache is not None:
        desc_ids = tree_cache['descendants'][node.id]
        non_desc_ids = tree_cache['non_descendants'][node.id]
        desc_id_list = tree_cache.get('descendant_arrays', {}).get(node.id)
        nd_id_list = tree_cache.get('non_descendant_arrays', {}).get(node.id)
    else:
        def _get_descendants(n):
            desc = set()
            for child in n.children:
                desc.add(child.id)
                desc.update(_get_descendants(child))
            return desc
        def _get_ancestors(n):
            anc = set()
            current = n.parent
            while current is not None:
                anc.add(current.id)
                current = current.parent
            return anc
        desc_ids = _get_descendants(node)
        all_ids = set(n.id for n in tree.get_all_nodes())
        non_desc_ids = all_ids - {node.id} - _get_ancestors(node) - desc_ids
        desc_id_list = None
        nd_id_list = None

    if not desc_ids:
        return None

    # Batch descendant angles
    if desc_id_list is None:
        desc_id_list = list(desc_ids)
    desc_embs = embeddings[desc_id_list]  # (n_desc, d)
    desc_norms = np.linalg.norm(desc_embs, axis=1)  # (n_desc,)
    valid_desc = desc_norms >= 1e-8
    if not np.any(valid_desc):
        return None

    cos_sims_desc = np.clip(
        desc_embs[valid_desc] @ node_emb / (desc_norms[valid_desc] * node_norm), -1, 1
    )
    descendant_angles = np.arccos(cos_sims_desc)

    max_angle_desc = np.percentile(descendant_angles, cone_boundary_percentile)
    if max_angle_desc < 1e-8:
        return None

    if not non_desc_ids:
        return None

    # Batch non-descendant angles
    if nd_id_list is None:
        nd_id_list = list(non_desc_ids)
    nd_embs = embeddings[nd_id_list]  # (n_nd, d)
    nd_norms = np.linalg.norm(nd_embs, axis=1)
    valid_nd = nd_norms >= 1e-8
    n_non_desc = len(nd_id_list)

    if not np.any(valid_nd):
        return {
            'theta_i': np.degrees(max_angle_desc),
            'cone_boundary_percentile': cone_boundary_percentile,
            'violation_rate': 0.0,
            'mean_exceeding_angle': 0.0,
            'mean_relative_violation': 0.0,
            'num_violations': 0,
            'num_non_descendants': n_non_desc
        }

    cos_sims_nd = np.clip(
        nd_embs[valid_nd] @ node_emb / (nd_norms[valid_nd] * node_norm), -1, 1
    )
    nd_angles = np.arccos(cos_sims_nd)

    # Vectorized violation computation
    violation_mask = nd_angles < max_angle_desc
    violations = int(np.sum(violation_mask))
    if violations > 0:
        exceeding = max_angle_desc - nd_angles[violation_mask]
        mean_exceeding = float(np.mean(exceeding))
        mean_relative = float(np.mean(exceeding / max_angle_desc))
    else:
        mean_exceeding = 0.0
        mean_relative = 0.0

    return {
        'theta_i': np.degrees(max_angle_desc),
        'cone_boundary_percentile': cone_boundary_percentile,
        'violation_rate': violations / n_non_desc,
        'mean_exceeding_angle': np.degrees(mean_exceeding),
        'mean_relative_violation': mean_relative,
        'num_violations': violations,
        'num_non_descendants': n_non_desc
    }


def _compute_branch_separation_for_node_images(
    embeddings, node, tree, node_to_images, cone_boundary_percentile, tree_cache
):
    """Branch separation for image embeddings (node_to_images path)."""
    def _get_leaf_descendants(n):
        if n.is_leaf():
            return [n]
        leaves = []
        for child in n.children:
            leaves.extend(_get_leaf_descendants(child))
        return leaves

    def get_embedding(n):
        if tree_cache is not None:
            leaf_ids = tree_cache['leaf_descendant_ids'][n.id]
        else:
            leaf_ids = [leaf.id for leaf in _get_leaf_descendants(n)]
        img_indices = []
        for lid in leaf_ids:
            img_indices.extend(node_to_images.get(lid, []))
        if not img_indices:
            return None
        return embeddings[img_indices].mean(axis=0)

    node_emb = get_embedding(node)
    if node_emb is None:
        return None
    node_norm = np.linalg.norm(node_emb)
    if node_norm < 1e-8:
        return None

    if tree_cache is not None:
        desc_ids = tree_cache['descendants'][node.id]
        non_desc_ids = tree_cache['non_descendants'][node.id]
    else:
        def _get_desc(n):
            desc = set()
            for child in n.children:
                desc.add(child.id)
                desc.update(_get_desc(child))
            return desc
        def _get_anc(n):
            anc = set()
            current = n.parent
            while current is not None:
                anc.add(current.id)
                current = current.parent
            return anc
        desc_ids = _get_desc(node)
        all_ids = set(n.id for n in tree.get_all_nodes())
        non_desc_ids = all_ids - {node.id} - _get_anc(node) - desc_ids

    if not desc_ids:
        return None

    descendant_angles = []
    for desc_id in desc_ids:
        desc_node = tree.get_node(desc_id)
        desc_emb = get_embedding(desc_node)
        if desc_emb is None:
            continue
        desc_norm = np.linalg.norm(desc_emb)
        if desc_norm < 1e-8:
            continue
        cos_sim = np.clip(np.dot(node_emb, desc_emb) / (node_norm * desc_norm), -1, 1)
        descendant_angles.append(np.arccos(cos_sim))

    if not descendant_angles:
        return None

    max_angle_desc = np.percentile(descendant_angles, cone_boundary_percentile)
    if max_angle_desc < 1e-8:
        return None
    if not non_desc_ids:
        return None

    violations = 0
    exceeding_angles = []
    relative_violations = []
    for nd_id in non_desc_ids:
        nd_node = tree.get_node(nd_id)
        nd_emb = get_embedding(nd_node)
        if nd_emb is None:
            continue
        nd_norm = np.linalg.norm(nd_emb)
        if nd_norm < 1e-8:
            continue
        cos_sim = np.clip(np.dot(node_emb, nd_emb) / (node_norm * nd_norm), -1, 1)
        angle = np.arccos(cos_sim)
        if angle < max_angle_desc:
            violations += 1
            exceeding = max_angle_desc - angle
            exceeding_angles.append(exceeding)
            relative_violations.append(exceeding / max_angle_desc)

    violation_rate = violations / len(non_desc_ids) if non_desc_ids else 0.0
    mean_exceeding = np.mean(exceeding_angles) if exceeding_angles else 0.0
    mean_relative = np.mean(relative_violations) if relative_violations else 0.0

    return {
        'theta_i': np.degrees(max_angle_desc),
        'cone_boundary_percentile': cone_boundary_percentile,
        'violation_rate': violation_rate,
        'mean_exceeding_angle': np.degrees(mean_exceeding),
        'mean_relative_violation': mean_relative,
        'num_violations': violations,
        'num_non_descendants': len(non_desc_ids)
    }


def _compute_branch_separation_node_metrics_vectorized(
    embeddings: np.ndarray,
    tree_cache: dict,
    cone_boundary_percentile: float,
) -> dict[int, dict]:
    """Compute branch separation metrics for all internal nodes from one cosine matrix."""
    eps = 1e-8
    norms = np.linalg.norm(embeddings, axis=1)
    safe_norms = np.where(norms >= eps, norms, 1.0)
    normalized = embeddings / safe_norms[:, None]
    cos_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)

    metrics_by_node = {}
    for node_id in tree_cache['internal_node_ids']:
        if norms[node_id] < eps:
            continue

        desc_ids = tree_cache['descendant_arrays'][node_id]
        if desc_ids.size == 0:
            continue

        valid_desc = norms[desc_ids] >= eps
        if not np.any(valid_desc):
            continue
        descendant_angles = np.arccos(cos_sim[node_id, desc_ids[valid_desc]])
        max_angle_desc = np.percentile(descendant_angles, cone_boundary_percentile)
        if max_angle_desc < eps:
            continue

        non_desc_ids = tree_cache['non_descendant_arrays'][node_id]
        if non_desc_ids.size == 0:
            continue
        n_non_desc = int(non_desc_ids.size)

        valid_nd = norms[non_desc_ids] >= eps
        if not np.any(valid_nd):
            metrics_by_node[node_id] = {
                'theta_i': np.degrees(max_angle_desc),
                'cone_boundary_percentile': cone_boundary_percentile,
                'violation_rate': 0.0,
                'mean_exceeding_angle': 0.0,
                'mean_relative_violation': 0.0,
                'num_violations': 0,
                'num_non_descendants': n_non_desc
            }
            continue

        nd_angles = np.arccos(cos_sim[node_id, non_desc_ids[valid_nd]])
        violation_mask = nd_angles < max_angle_desc
        violations = int(np.sum(violation_mask))
        if violations > 0:
            exceeding = max_angle_desc - nd_angles[violation_mask]
            mean_exceeding = float(np.mean(exceeding))
            mean_relative = float(np.mean(exceeding / max_angle_desc))
        else:
            mean_exceeding = 0.0
            mean_relative = 0.0

        metrics_by_node[node_id] = {
            'theta_i': np.degrees(max_angle_desc),
            'cone_boundary_percentile': cone_boundary_percentile,
            'violation_rate': violations / n_non_desc,
            'mean_exceeding_angle': np.degrees(mean_exceeding),
            'mean_relative_violation': mean_relative,
            'num_violations': violations,
            'num_non_descendants': n_non_desc
        }

    return metrics_by_node


def _aggregate_branch_separation_by_depth_vectorized(
    embeddings: np.ndarray,
    tree_cache: dict,
    max_depth: int,
    cone_boundary_percentile: float,
    include_empty_depths: bool = False,
) -> dict[int, dict]:
    """Aggregate vectorized branch separation metrics by depth."""
    node_metrics = _compute_branch_separation_node_metrics_vectorized(
        embeddings, tree_cache, cone_boundary_percentile
    )
    results = {}
    for depth in range(max_depth):
        node_ids = tree_cache['internal_nodes_by_depth'][depth]
        if not node_ids:
            if include_empty_depths:
                results[depth] = {
                    'mean_violation_rate': 0.0,
                    'mean_exceeding_angle': 0.0,
                    'mean_relative_violation': 0.0,
                    'mean_theta_i': 0.0,
                    'count': 0
                }
            continue

        violation_rates = []
        exceeding_angles = []
        relative_violations = []
        theta_i_vals = []

        for node_id in node_ids:
            metrics = node_metrics.get(node_id)
            if metrics is None:
                continue
            violation_rates.append(metrics['violation_rate'])
            exceeding_angles.append(metrics['mean_exceeding_angle'])
            relative_violations.append(metrics['mean_relative_violation'])
            theta_i_vals.append(metrics['theta_i'])

        if violation_rates or include_empty_depths:
            results[depth] = {
                'mean_violation_rate': np.mean(violation_rates) if violation_rates else 0.0,
                'mean_exceeding_angle': np.mean(exceeding_angles) if exceeding_angles else 0.0,
                'mean_relative_violation': np.mean(relative_violations) if relative_violations else 0.0,
                'mean_theta_i': np.mean(theta_i_vals) if theta_i_vals else 0.0,
                'count': len(violation_rates)
            }
    return results


def compute_branch_separation_by_depth(
    embeddings: np.ndarray,
    tree: Tree,
    node_to_images: dict[int, list[int]] = None,
    cone_boundary_percentile: float = 100.0,
    tree_cache: dict = None
) -> dict[int, dict]:
    """
    Compute branch separation metrics aggregated by depth.

    Args:
        embeddings: Embedding matrix (V, d) or (m, d)
        tree: Hierarchy tree
        node_to_images: Optional mapping for image embeddings
        cone_boundary_percentile: Percentile of descendant angles to use for cone boundary
        tree_cache: Precomputed tree structure from precompute_tree_structure()

    Returns:
        Dict mapping depth -> {mean_violation_rate, mean_exceeding_angle, mean_relative_violation, count}
    """
    max_depth = tree.get_max_depth()
    if node_to_images is None:
        if tree_cache is None:
            tree_cache = precompute_tree_structure(tree)
        aggregated = _aggregate_branch_separation_by_depth_vectorized(
            embeddings=embeddings,
            tree_cache=tree_cache,
            max_depth=max_depth,
            cone_boundary_percentile=cone_boundary_percentile,
            include_empty_depths=False,
        )
        return {
            depth: {
                'mean_violation_rate': depth_metrics['mean_violation_rate'],
                'mean_exceeding_angle': depth_metrics['mean_exceeding_angle'],
                'mean_relative_violation': depth_metrics['mean_relative_violation'],
                'count': depth_metrics['count'],
            }
            for depth, depth_metrics in aggregated.items()
            if depth_metrics['count'] > 0
        }

    results = {}
    for depth in range(max_depth):
        nodes = tree.get_nodes_at_depth(depth)
        violation_rates = []
        exceeding_angles = []
        relative_violations = []

        for node in nodes:
            if node.is_leaf():
                continue
            metrics = compute_branch_separation_for_node(
                embeddings, node, tree, node_to_images, cone_boundary_percentile, tree_cache=tree_cache
            )
            if metrics is None:
                continue
            violation_rates.append(metrics['violation_rate'])
            exceeding_angles.append(metrics['mean_exceeding_angle'])
            relative_violations.append(metrics['mean_relative_violation'])

        if violation_rates:
            results[depth] = {
                'mean_violation_rate': np.mean(violation_rates),
                'mean_exceeding_angle': np.mean(exceeding_angles),
                'mean_relative_violation': np.mean(relative_violations),
                'count': len(violation_rates)
            }

    return results


def track_branch_separation_over_training(
    checkpoints: list[Checkpoint],
    tree: Tree,
    leaf_assignments: np.ndarray,
    cone_boundary_percentile: float = 100.0
) -> dict:
    """
    Track branch separation metrics over training for A, W_mean, H_mean.

    Args:
        checkpoints: Training checkpoints
        tree: Hierarchy tree
        leaf_assignments: Array mapping image index to leaf node id
        cone_boundary_percentile: Percentile of descendant angles to use for cone boundary (default=100, i.e., max)

    Returns:
        Dictionary with:
            'epochs': array of epochs
            'A': {depth: {violation_rate: [], mean_exceeding_angle: [], mean_relative_violation: []}}
            'W_mean': same structure
            'H_mean': same structure
    """
    max_depth = tree.get_max_depth()
    epochs = [ckpt.epoch for ckpt in checkpoints]
    tree_cache = precompute_tree_structure(tree)
    leaf_ids, leaf_counts = _prepare_leaf_image_stats(leaf_assignments, tree.num_nodes)

    results = {
        'epochs': np.array(epochs),
        'A': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
        'W_mean': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
        'H_mean': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
    }

    for ckpt in checkpoints:
        W = ckpt.W  # (V, d)
        H = ckpt.H  # (d, m)

        W_mean = build_mean_descendant_matrix(W, tree_cache)
        H_by_node = _build_h_by_node_from_images(H, W.shape, leaf_ids, leaf_counts)
        H_mean = build_mean_descendant_matrix(H_by_node, tree_cache)

        by_depth = {
            'A': _aggregate_branch_separation_by_depth_vectorized(
                W, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
            'W_mean': _aggregate_branch_separation_by_depth_vectorized(
                W_mean, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
            'H_mean': _aggregate_branch_separation_by_depth_vectorized(
                H_mean, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
        }

        for depth in range(max_depth):
            for emb_type in ['A', 'W_mean', 'H_mean']:
                depth_metrics = by_depth[emb_type][depth]
                results[emb_type][depth]['violation_rate'].append(depth_metrics['mean_violation_rate'])
                results[emb_type][depth]['mean_exceeding_angle'].append(depth_metrics['mean_exceeding_angle'])
                results[emb_type][depth]['mean_relative_violation'].append(depth_metrics['mean_relative_violation'])
                results[emb_type][depth]['mean_theta_i'].append(depth_metrics['mean_theta_i'])

    # Convert lists to arrays
    for emb_type in ['A', 'W_mean', 'H_mean']:
        for d in results[emb_type]:
            for metric in ['violation_rate', 'mean_exceeding_angle', 'mean_relative_violation', 'mean_theta_i']:
                results[emb_type][d][metric] = np.array(results[emb_type][d][metric])

    return results


def track_all_metrics_single_pass(
    checkpoints,
    tree: Tree,
    leaf_assignments: np.ndarray,
    cone_boundary_percentile: float = 100.0
) -> dict:
    """
    Compute both branch_sep and orthogonality/cone_angles in a single pass over checkpoints.

    Returns dict with keys 'branch_sep' and 'all_metrics', matching the return structures
    of track_branch_separation_over_training and track_all_metrics_over_training respectively.
    """
    max_depth = tree.get_max_depth()
    tree_cache = precompute_tree_structure(tree)
    leaf_ids, leaf_counts = _prepare_leaf_image_stats(leaf_assignments, tree.num_nodes)

    # Initialize branch_sep storage
    bs = {
        'epochs': [],
        'A': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
        'W_mean': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
        'H_mean': {d: {'violation_rate': [], 'mean_exceeding_angle': [], 'mean_relative_violation': [], 'mean_theta_i': []} for d in range(max_depth)},
    }

    # Initialize all_metrics storage
    am = {
        'epochs': [],
        'orthogonality': {
            'A': {d: [] for d in range(1, max_depth + 1)},
            'W_mean': {d: [] for d in range(1, max_depth + 1)},
            'H_mean': {d: [] for d in range(1, max_depth + 1)},
        },
        'cone_angles': {
            'A': {d: [] for d in range(1, max_depth + 1)},
            'W_mean': {d: [] for d in range(1, max_depth + 1)},
            'H_mean': {d: [] for d in range(1, max_depth + 1)},
        }
    }

    for ckpt in checkpoints:
        bs['epochs'].append(ckpt.epoch)
        am['epochs'].append(ckpt.epoch)

        W = ckpt.W  # (V, d)
        H = ckpt.H  # (d, m)

        W_mean = build_mean_descendant_matrix(W, tree_cache)
        H_by_node = _build_h_by_node_from_images(H, W.shape, leaf_ids, leaf_counts)
        H_mean = build_mean_descendant_matrix(H_by_node, tree_cache)

        # --- Branch separation ---
        by_depth = {
            'A': _aggregate_branch_separation_by_depth_vectorized(
                W, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
            'W_mean': _aggregate_branch_separation_by_depth_vectorized(
                W_mean, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
            'H_mean': _aggregate_branch_separation_by_depth_vectorized(
                H_mean, tree_cache, max_depth, cone_boundary_percentile, include_empty_depths=True
            ),
        }
        for depth in range(max_depth):
            for emb_type in ['A', 'W_mean', 'H_mean']:
                depth_metrics = by_depth[emb_type][depth]
                bs[emb_type][depth]['violation_rate'].append(depth_metrics['mean_violation_rate'])
                bs[emb_type][depth]['mean_exceeding_angle'].append(depth_metrics['mean_exceeding_angle'])
                bs[emb_type][depth]['mean_relative_violation'].append(depth_metrics['mean_relative_violation'])
                bs[emb_type][depth]['mean_theta_i'].append(depth_metrics['mean_theta_i'])

        # --- Orthogonality ---
        ortho_A = aggregate_orthogonality_by_depth_vectorized(W, tree)
        ortho_W_mean = aggregate_orthogonality_by_depth_vectorized(W_mean, tree)
        ortho_H_mean = aggregate_orthogonality_by_depth_vectorized(H_mean, tree)
        for d in range(1, max_depth + 1):
            for emb_label, ortho in [('A', ortho_A), ('W_mean', ortho_W_mean), ('H_mean', ortho_H_mean)]:
                am['orthogonality'][emb_label][d].append(np.mean(ortho[d]) if ortho[d] else 0.0)

        # --- Cone angles ---
        cone_A = aggregate_cone_angle_by_depth_vectorized(W, tree)
        cone_W_mean = aggregate_cone_angle_by_depth_vectorized(W_mean, tree)
        cone_H_mean = aggregate_cone_angle_by_depth_vectorized(H_mean, tree)
        for d in range(1, max_depth + 1):
            for emb_label, cone in [('A', cone_A), ('W_mean', cone_W_mean), ('H_mean', cone_H_mean)]:
                am['cone_angles'][emb_label][d].append(np.mean(cone[d]) if cone[d] else 0.0)

    # Convert lists to arrays
    bs['epochs'] = np.array(bs['epochs'])
    for emb_type in ['A', 'W_mean', 'H_mean']:
        for d in bs[emb_type]:
            for metric in ['violation_rate', 'mean_exceeding_angle', 'mean_relative_violation', 'mean_theta_i']:
                bs[emb_type][d][metric] = np.array(bs[emb_type][d][metric])

    am['epochs'] = np.array(am['epochs'])
    for metric in ['orthogonality', 'cone_angles']:
        for emb_type in ['A', 'W_mean', 'H_mean']:
            for d in am[metric][emb_type]:
                am[metric][emb_type][d] = np.array(am[metric][emb_type][d])

    return {'branch_sep': bs, 'all_metrics': am}


def track_all_metrics_over_training(
    checkpoints: list[Checkpoint],
    tree: Tree,
    leaf_assignments: np.ndarray
) -> dict:
    """
    Track orthogonality and cone angles for all embedding types:
    - A: Text embeddings with direct node access
    - W_mean: Text embeddings with mean-of-descendants
    - H_mean: Image embeddings with mean-of-descendants

    Args:
        checkpoints: Training checkpoints
        tree: Hierarchy tree
        leaf_assignments: Array mapping image index to leaf node id

    Returns:
        Dictionary with results for each embedding type
    """
    max_depth = tree.get_max_depth()
    epochs = [ckpt.epoch for ckpt in checkpoints]
    tree_cache = precompute_tree_structure(tree)
    leaf_ids, leaf_counts = _prepare_leaf_image_stats(leaf_assignments, tree.num_nodes)

    # Initialize storage
    results = {
        'epochs': np.array(epochs),
        'orthogonality': {
            'A': {d: [] for d in range(1, max_depth + 1)},
            'W_mean': {d: [] for d in range(1, max_depth + 1)},
            'H_mean': {d: [] for d in range(1, max_depth + 1)},
        },
        'cone_angles': {
            'A': {d: [] for d in range(1, max_depth + 1)},
            'W_mean': {d: [] for d in range(1, max_depth + 1)},
            'H_mean': {d: [] for d in range(1, max_depth + 1)},
        }
    }

    for ckpt in checkpoints:
        W = ckpt.W  # (V, d)
        H = ckpt.H  # (d, m)

        W_mean = build_mean_descendant_matrix(W, tree_cache)

        H_by_node = _build_h_by_node_from_images(H, W.shape, leaf_ids, leaf_counts)
        H_mean = build_mean_descendant_matrix(H_by_node, tree_cache)

        # Vectorized orthogonality
        ortho_A = aggregate_orthogonality_by_depth_vectorized(W, tree)
        ortho_W_mean = aggregate_orthogonality_by_depth_vectorized(W_mean, tree)
        ortho_H_mean = aggregate_orthogonality_by_depth_vectorized(H_mean, tree)

        for d in range(1, max_depth + 1):
            for emb_label, ortho in [('A', ortho_A), ('W_mean', ortho_W_mean), ('H_mean', ortho_H_mean)]:
                mean_score = np.mean(ortho[d]) if ortho[d] else 0.0
                results['orthogonality'][emb_label][d].append(mean_score)

        # Vectorized cone angles
        cone_A = aggregate_cone_angle_by_depth_vectorized(W, tree)
        cone_W_mean = aggregate_cone_angle_by_depth_vectorized(W_mean, tree)
        cone_H_mean = aggregate_cone_angle_by_depth_vectorized(H_mean, tree)

        for d in range(1, max_depth + 1):
            for emb_label, cone in [('A', cone_A), ('W_mean', cone_W_mean), ('H_mean', cone_H_mean)]:
                mean_angle = np.mean(cone[d]) if cone[d] else 0.0
                results['cone_angles'][emb_label][d].append(mean_angle)

    # Convert lists to arrays
    for metric in ['orthogonality', 'cone_angles']:
        for emb_type in ['A', 'W_mean', 'H_mean']:
            for d in results[metric][emb_type]:
                results[metric][emb_type][d] = np.array(results[metric][emb_type][d])

    return results

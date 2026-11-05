"""Synthetic data generation for hierarchical concept learning."""

import numpy as np
from hierarchy import Tree


def sample_leaf_assignments(tree: Tree, num_images: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Assign each image to a leaf node uniformly at random.

    Args:
        tree: Hierarchy tree
        num_images: Number of images
        rng: Random generator

    Returns:
        Array of shape (num_images,) with leaf node IDs
    """
    if rng is None:
        rng = np.random.default_rng()
    leaves = tree.get_leaves()
    leaf_ids = np.array([leaf.id for leaf in leaves])
    return rng.choice(leaf_ids, size=num_images)


def assign_images_per_leaf(tree: Tree, images_per_leaf: int) -> np.ndarray:
    """
    Assign exactly images_per_leaf images to each leaf node.

    Args:
        tree: Hierarchy tree
        images_per_leaf: Number of images per leaf

    Returns:
        Array of shape (num_leaves * images_per_leaf,) with leaf node IDs
    """
    leaves = tree.get_leaves()
    assignments = []
    for leaf in leaves:
        assignments.extend([leaf.id] * images_per_leaf)
    return np.array(assignments, dtype=np.int64)


def build_support_matrix(tree: Tree, leaf_assignments: np.ndarray) -> np.ndarray:
    """
    Build support matrix S where S[z, j] = 1 iff synset z is on path from root to leaf_j.

    Args:
        tree: Hierarchy tree with V nodes
        leaf_assignments: Array of shape (m,) with leaf node IDs for each image

    Returns:
        S: Binary matrix of shape (V, m)
    """
    V = tree.num_nodes
    m = len(leaf_assignments)
    S = np.zeros((V, m), dtype=np.float32)

    for j, leaf_id in enumerate(leaf_assignments):
        leaf = tree.get_node(leaf_id)
        path_ids = tree.get_path_ids_to_root(leaf)
        for z in path_ids:
            S[z, j] = 1.0

    return S


def build_caption_distribution(
    tree: Tree,
    leaf_assignments: np.ndarray,
    mode: str = 'uniform',
    alpha: float = 1.0
) -> np.ndarray:
    """
    Build caption probability distributions p_j for each image.

    Args:
        tree: Hierarchy tree with V nodes
        leaf_assignments: Array of shape (m,) with leaf node IDs
        mode: Caption distribution mode:
            - 'uniform': Equal probability along path from leaf to root
            - 'exponential': exp(-alpha * depth) weighted (root-biased)
            - 'leaf_only': All probability on leaf node only
            - 'linear': Linearly decreasing with depth (root-biased)
        alpha: Parameter for exponential mode

    Returns:
        P: Matrix of shape (V, m) where P[:, j] is the caption distribution for image j
    """
    V = tree.num_nodes
    m = len(leaf_assignments)
    P = np.zeros((V, m), dtype=np.float32)

    # 'balanced': deterministic round-robin along root-to-leaf path, per-leaf counter.
    # Produces a one-hot P column per image; used to satisfy the "balanced positive
    # sets" hypothesis of cor:large-lambda-hbs-transfer (body.tex:403-416).
    leaf_counter: dict = {}

    for j, leaf_id in enumerate(leaf_assignments):
        leaf = tree.get_node(leaf_id)
        path = tree.get_path_to_root(leaf)

        if mode == 'balanced':
            k = leaf_counter.get(int(leaf_id), 0)
            path_ids = list(reversed(tree.get_path_ids_to_root(leaf)))
            P[path_ids[k % len(path_ids)], j] = 1.0
            leaf_counter[int(leaf_id)] = k + 1
            continue

        if mode == 'uniform':
            prob = 1.0 / len(path)
            for node in path:
                P[node.id, j] = prob
        elif mode == 'exponential':
            weights = np.array([np.exp(-alpha * node.depth) for node in path])
            weights /= weights.sum()
            for i, node in enumerate(path):
                P[node.id, j] = weights[i]
        elif mode == 'leaf_only':
            P[leaf_id, j] = 1.0
        elif mode == 'linear':
            max_depth = tree.get_max_depth()
            weights = np.array([max_depth - node.depth + 1 for node in path], dtype=np.float32)
            weights /= weights.sum()
            for i, node in enumerate(path):
                P[node.id, j] = weights[i]
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: uniform, exponential, leaf_only, linear, balanced")

    return P


def compute_centered_support_matrix(S: np.ndarray) -> np.ndarray:
    """
    Compute centered support matrix: S_bar = (I - 1/V * 11^T) S

    Args:
        S: Support matrix of shape (V, m)

    Returns:
        S_bar: Centered support matrix of shape (V, m)
    """
    V = S.shape[0]
    col_means = S.mean(axis=0, keepdims=True)  # shape (1, m)
    S_bar = S - col_means
    return S_bar


def compute_image_weights(leaf_assignments: np.ndarray) -> np.ndarray:
    """
    Compute uniform image weights pi_j = 1/m.

    Args:
        leaf_assignments: Array of shape (m,) with leaf node IDs

    Returns:
        pi: Array of shape (m,) with uniform weights
    """
    m = len(leaf_assignments)
    return np.ones(m, dtype=np.float32) / m


def _validate_leaf_ids(tree: Tree, leaf_assignments: np.ndarray) -> np.ndarray:
    leaf_ids = np.asarray(leaf_assignments, dtype=np.int64)
    valid_leaf_ids = {leaf.id for leaf in tree.get_leaves()}
    if not all(int(leaf_id) in valid_leaf_ids for leaf_id in leaf_ids):
        raise ValueError("leaf_assignments must contain leaf node IDs only")
    return leaf_ids


def _canonical_simplex(branching_factor: int) -> np.ndarray:
    """Return centered simplex vertices e_i - (1/b)1 in R^b."""
    eye = np.eye(branching_factor, dtype=np.float64)
    return eye - (1.0 / branching_factor)


def build_hierarchical_prototypes(
    tree: Tree,
    d: int,
    theta_by_depth: dict[int, float],
    root_scale: float = 0.8,
    root_at_zero: int = 1,
) -> np.ndarray:
    """
    Build node prototypes using the recursive cone-constrained simplex construction.

    Args:
        tree: Balanced hierarchy tree
        d: Prototype dimension
        theta_by_depth: Mapping depth -> cone angle (radians), for depths 1..max_depth
        root_scale: Fallback scale when parent norm is 0 (matches source default 0.8)
        root_at_zero: Use zero root vector (1) or deterministic non-zero root (0)

    Returns:
        Prototypes matrix of shape (V, d), row i is prototype for node i
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    if root_scale <= 0:
        raise ValueError("root_scale must be > 0")

    V = tree.num_nodes
    max_depth = tree.get_max_depth()
    missing_depths = [depth for depth in range(1, max_depth + 1) if depth not in theta_by_depth]
    if missing_depths:
        raise ValueError(f"theta_by_depth missing values for depths: {missing_depths}")

    prototypes = np.zeros((V, d), dtype=np.float64)
    if not root_at_zero:
        prototypes[tree.root.id, 0] = 1.0

    for depth in range(1, max_depth + 1):
        theta_l = float(theta_by_depth[depth])
        parent_nodes = sorted(tree.get_nodes_at_depth(depth - 1), key=lambda node: node.id)
        for parent in parent_nodes:
            children = parent.children
            if not children:
                continue

            parent_vec = prototypes[parent.id]
            ancestor_ids = list(reversed(tree.get_path_ids_to_root(parent)))
            A_l = prototypes[ancestor_ids].T  # (d, num_ancestors)
            Q, _ = np.linalg.qr(A_l, mode='complete')
            U_l = Q[:, len(ancestor_ids):]  # orthonormal basis for null(A_l^T)

            simplex = _canonical_simplex(len(children))
            parent_norm = float(np.linalg.norm(parent_vec))
            lambda_l = root_scale if parent_norm == 0.0 else parent_norm * abs(np.tan(theta_l))

            for child_idx, child in enumerate(children):
                simplex_vertex = simplex[child_idx]
                if U_l.shape[1] >= simplex_vertex.shape[0]:
                    residual = U_l[:, :simplex_vertex.shape[0]] @ simplex_vertex
                else:
                    padded = np.zeros(U_l.shape[1], dtype=np.float64)
                    keep = min(U_l.shape[1], simplex_vertex.shape[0])
                    padded[:keep] = simplex_vertex[:keep]
                    residual = U_l @ padded
                child_vec = parent_vec + lambda_l * residual
                child_norm = float(np.linalg.norm(child_vec))
                if child_norm == 0.0:
                    raise ValueError("constructed child prototype has zero norm; cannot normalize")
                prototypes[child.id] = child_vec / child_norm

    return prototypes.astype(np.float32)


def sample_leaf_latents(
    tree: Tree,
    leaf_assignments: np.ndarray,
    prototypes: np.ndarray,
    sigma: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample latent vectors z_j ~ N(mu_leaf(j), sigma^2 I), returned as columns.

    Returns:
        Z of shape (d, m)
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if rng is None:
        rng = np.random.default_rng()
    if prototypes.ndim != 2 or prototypes.shape[0] != tree.num_nodes:
        raise ValueError("prototypes must have shape (V, d)")

    leaf_ids = _validate_leaf_ids(tree, leaf_assignments)
    m = leaf_ids.shape[0]
    d = prototypes.shape[1]

    means = prototypes[leaf_ids].T  # (d, m)
    noise = rng.normal(0.0, sigma, size=(d, m))
    return (means + noise).astype(np.float32)


def sample_observed_images(
    Z: np.ndarray,
    p_dim: int,
    rng: np.random.Generator = None,
    W_true: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate observed image features via X = W_true Z.

    Args:
        Z: Latent matrix (d, m)
        p_dim: Observed feature dimension p (ignored when W_true is provided)
        rng: Random generator
        W_true: Optional fixed projection matrix (p, d)

    Returns:
        X of shape (p, m), W_true of shape (p, d)
    """
    if rng is None:
        rng = np.random.default_rng()
    if Z.ndim != 2:
        raise ValueError("Z must have shape (d, m)")

    d, _ = Z.shape
    if W_true is None:
        if p_dim < 1:
            raise ValueError("p_dim must be >= 1 when W_true is not provided")
        W_true = rng.normal(size=(p_dim, d)).astype(np.float32) / np.sqrt(d)
    else:
        if W_true.ndim != 2 or W_true.shape[1] != d:
            raise ValueError("W_true must have shape (p, d) with d matching Z")
        W_true = W_true.astype(np.float32, copy=False)

    X = W_true @ Z
    return X.astype(np.float32), W_true


def build_leaf_only_distribution(tree: Tree, leaf_assignments: np.ndarray) -> np.ndarray:
    """
    Build deterministic one-hot labels over leaves: P[y_j, j] = 1.
    """
    V = tree.num_nodes
    leaf_ids = _validate_leaf_ids(tree, leaf_assignments)
    m = leaf_ids.shape[0]
    P = np.zeros((V, m), dtype=np.float32)
    P[leaf_ids, np.arange(m)] = 1.0
    return P

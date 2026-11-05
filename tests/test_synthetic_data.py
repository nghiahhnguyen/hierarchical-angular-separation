import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))

from hierarchy import build_balanced_tree
from synthetic_data import (
    assign_images_per_leaf,
    build_hierarchical_prototypes,
    sample_leaf_latents,
    sample_observed_images,
    build_leaf_only_distribution,
)


class TestStrictSyntheticData(unittest.TestCase):
    def test_build_hierarchical_prototypes_is_deterministic(self):
        tree = build_balanced_tree(depth=2, branching_factor=3)
        theta_by_depth = {1: np.deg2rad(40.0), 2: np.deg2rad(20.0)}

        p1 = build_hierarchical_prototypes(tree, d=16, theta_by_depth=theta_by_depth)
        p2 = build_hierarchical_prototypes(tree, d=16, theta_by_depth=theta_by_depth)

        self.assertEqual(p1.shape, (tree.num_nodes, 16))
        np.testing.assert_allclose(p1, p2, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(p1[tree.root.id], 0.0, atol=0.0, rtol=0.0)

        non_root_ids = [node.id for node in tree.get_all_nodes() if node.id != tree.root.id]
        norms = np.linalg.norm(p1[non_root_ids], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6, rtol=0.0)

    def test_sample_leaf_latents_matches_prototypes_when_sigma_zero(self):
        tree = build_balanced_tree(depth=2, branching_factor=2)
        theta_by_depth = {1: np.deg2rad(35.0), 2: np.deg2rad(20.0)}
        prototypes = build_hierarchical_prototypes(tree, d=12, theta_by_depth=theta_by_depth)
        leaf_assignments = assign_images_per_leaf(tree, images_per_leaf=3)

        Z = sample_leaf_latents(
            tree=tree,
            leaf_assignments=leaf_assignments,
            prototypes=prototypes,
            sigma=0.0,
            rng=np.random.default_rng(0),
        )
        expected = prototypes[leaf_assignments].T
        np.testing.assert_allclose(Z, expected, atol=0.0, rtol=0.0)

    def test_sample_observed_images_shapes(self):
        rng = np.random.default_rng(123)
        Z = rng.normal(size=(10, 25)).astype(np.float32)
        X, W_true = sample_observed_images(Z=Z, p_dim=20, rng=np.random.default_rng(1))
        self.assertEqual(X.shape, (20, 25))
        self.assertEqual(W_true.shape, (20, 10))

        W_fixed = rng.normal(size=(7, 10)).astype(np.float32)
        X_fixed, W_back = sample_observed_images(Z=Z, p_dim=1, W_true=W_fixed)
        np.testing.assert_allclose(W_back, W_fixed, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(X_fixed, W_fixed @ Z, atol=1e-6, rtol=1e-6)

    def test_build_leaf_only_distribution_is_one_hot(self):
        tree = build_balanced_tree(depth=2, branching_factor=2)
        leaf_assignments = assign_images_per_leaf(tree, images_per_leaf=2)
        P = build_leaf_only_distribution(tree, leaf_assignments)

        self.assertEqual(P.shape, (tree.num_nodes, len(leaf_assignments)))
        np.testing.assert_allclose(P.sum(axis=0), 1.0, atol=0.0, rtol=0.0)
        self.assertTrue(np.all((P == 0.0) | (P == 1.0)))
        np.testing.assert_array_equal(np.argmax(P, axis=0), leaf_assignments)


if __name__ == "__main__":
    unittest.main()

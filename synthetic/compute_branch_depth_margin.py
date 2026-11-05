"""Compute branch-only depth-collapsed margins for one strict identity-encoder run."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from hierarchy import build_balanced_tree
from synthetic_data import build_caption_distribution
from metrics import precompute_tree_structure, compute_branch_separation_by_depth


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict_data",
        type=str,
        help="Path to strict_data.npz or its containing directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Sweep results directory containing strict_data/ and MLflow params",
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        default="",
        help="Optional comma-separated run IDs to restrict --results_dir mode",
    )
    parser.add_argument("--lambda_reg", type=float, help="Required in --strict_data mode")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--branching_factor", type=int, default=3)
    parser.add_argument("--caption_mode", type=str, default="uniform")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_direction", type=str, default=None, choices=["both", "i2t", "t2i"],
                        help="Loss direction override. If omitted in --results_dir mode, read from MLflow params.")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=16)
    args = parser.parse_args()
    if bool(args.strict_data) == bool(args.results_dir):
        parser.error("provide exactly one of --strict_data or --results_dir")
    if args.strict_data is not None and args.lambda_reg is None:
        parser.error("--lambda_reg is required with --strict_data")
    return args


def _resolve_strict_data_path(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "strict_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"strict_data artifact not found: {path}")
    return path


def _resolve_results_dir(path: str) -> Path:
    results_dir = Path(path)
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    strict_data_dir = results_dir / "strict_data"
    if not strict_data_dir.is_dir():
        raise FileNotFoundError(f"strict_data directory not found under: {results_dir}")
    return results_dir


def _sample_concepts(tree, leaf_assignments: np.ndarray, caption_mode: str, seed: int) -> np.ndarray:
    P = build_caption_distribution(tree, leaf_assignments, mode=caption_mode)
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(tree.num_nodes, p=P[:, j]) for j in range(P.shape[1])], dtype=np.int64)


def _solve_identity_strict_run(
    X: np.ndarray,
    concept_indices: np.ndarray,
    V: int,
    temperature: float,
    lambda_reg: float,
    max_iter: int,
    num_threads: int,
    loss_direction: str = "both",
):
    d, m = X.shape
    torch.set_num_threads(min(num_threads, os.cpu_count() or 1))
    X_t = torch.tensor(X, dtype=torch.float64, device="cpu")
    concept_t = torch.tensor(concept_indices, dtype=torch.long, device="cpu")
    image_idx = torch.arange(m, device="cpu")
    W = torch.nn.Parameter(torch.zeros((V, d), dtype=torch.float64, device="cpu"))
    optimizer = torch.optim.LBFGS(
        [W],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter + 20,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    loss_history = []

    def loss_fn():
        sim = (W @ X_t) / temperature
        pos = sim[concept_t, image_idx]
        l_i2t = -(pos - torch.logsumexp(sim, dim=0)).mean()
        l_t2i = -(pos - torch.logsumexp(sim, dim=1)[concept_t]).mean()
        reg = 0.5 * lambda_reg * W.square().sum()
        if loss_direction == "i2t":
            return l_i2t + reg
        elif loss_direction == "t2i":
            return l_t2i + reg
        return 0.5 * (l_i2t + l_t2i) + reg

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        loss_history.append(float(loss.detach().cpu()))
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    post_loss = loss_fn()
    post_loss.backward()
    return {
        "W": W.detach().cpu().numpy().astype(np.float64),
        "post_loss": float(post_loss.detach().cpu()),
        "grad_norm": float(W.grad.norm().detach().cpu()),
        "num_closures": len(loss_history),
    }


def _compute_depth_collapsed_margins(W: np.ndarray, tree_cache: dict) -> dict[int, dict]:
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_norm = W / np.clip(row_norms, 1e-12, None)
    cosine = W_norm @ W_norm.T
    summaries = {}

    for depth, node_ids in tree_cache["internal_nodes_by_depth"].items():
        if not node_ids:
            continue

        min_desc_values = []
        max_offbranch_values = []
        node_margins = []

        for node_id in node_ids:
            desc_ids = tree_cache["descendant_arrays"][node_id]
            offbranch_ids = tree_cache["non_descendant_arrays"][node_id]
            if desc_ids.size == 0 or offbranch_ids.size == 0:
                continue

            min_desc = float(cosine[node_id, desc_ids].min())
            max_offbranch = float(cosine[node_id, offbranch_ids].max())
            min_desc_values.append(min_desc)
            max_offbranch_values.append(max_offbranch)
            node_margins.append(min_desc - max_offbranch)

        if not node_margins:
            summaries[depth] = {
                "internal_count": len(node_ids),
                "comparable_count": 0,
                "vacuous": True,
                "min_desc_cos": None,
                "max_offbranch_cos": None,
                "margin": None,
                "min_node_margin": None,
                "mean_node_margin": None,
                "frac_negative_node_margin": None,
            }
            continue

        node_margins_arr = np.asarray(node_margins, dtype=np.float64)
        summaries[depth] = {
            "internal_count": len(node_ids),
            "comparable_count": len(node_margins),
            "vacuous": False,
            "min_desc_cos": float(min(min_desc_values)),
            "max_offbranch_cos": float(max(max_offbranch_values)),
            "margin": float(min(min_desc_values) - max(max_offbranch_values)),
            "min_node_margin": float(node_margins_arr.min()),
            "mean_node_margin": float(node_margins_arr.mean()),
            "frac_negative_node_margin": float(np.mean(node_margins_arr < 0)),
        }

    return summaries


def _compute_softmax_uniformity(logits: np.ndarray, uniform_mass: float, axis: int) -> dict[str, float]:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=axis, keepdims=True)
    tv = 0.5 * np.abs(probs - uniform_mass).sum(axis=axis)
    log_uniform = np.log(uniform_mass)
    kl = (probs * (np.log(np.clip(probs, 1e-300, None)) - log_uniform)).sum(axis=axis)
    entropy = -(probs * np.log(np.clip(probs, 1e-300, None))).sum(axis=axis)
    return {
        "tv_mean": float(tv.mean()),
        "tv_max": float(tv.max()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "entropy_mean": float(entropy.mean()),
        "uniform_entropy": float(-np.log(uniform_mass)),
        "entropy_gap_mean": float(-np.log(uniform_mass) - entropy.mean()),
    }


def _compute_hierarchical_violation(W: np.ndarray, tree, tree_cache: dict) -> dict:
    by_depth = compute_branch_separation_by_depth(W, tree, tree_cache=tree_cache)
    rates = [stats["mean_violation_rate"] for stats in by_depth.values()]
    counts = [stats["count"] for stats in by_depth.values()]
    total = sum(counts)
    weighted_mean = (
        float(sum(r * c for r, c in zip(rates, counts)) / total) if total > 0 else 0.0
    )
    return {
        "by_depth": {depth: stats["mean_violation_rate"] for depth, stats in by_depth.items()},
        "node_weighted_mean": weighted_mean,
        "max_over_depths": float(max(rates)) if rates else 0.0,
    }


def _compute_prototype_alignment(W: np.ndarray, prototypes: np.ndarray, tree) -> dict:
    W_norm = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)
    proto_norm = prototypes / np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None)
    self_cos = np.sum(W_norm * proto_norm, axis=1)
    by_depth = {}

    for depth in range(1, tree.get_max_depth() + 1):
        node_ids = [node.id for node in tree.get_nodes_at_depth(depth)]
        if not node_ids:
            continue
        values = self_cos[node_ids]
        by_depth[depth] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
        }

    nonroot = self_cos[1:]
    return {
        "by_depth": by_depth,
        "global_mean_nonroot": float(nonroot.mean()),
        "global_min_nonroot": float(nonroot.min()),
    }


def _compute_cor_form_alignment(
    W: np.ndarray,
    prototypes: np.ndarray,
    concept_indices: np.ndarray,
    tree,
    loss_direction: str,
) -> dict:
    """Cosine alignment with the large-lambda first-order target for the chosen loss."""
    M = tree.num_nodes
    m = len(concept_indices)
    mu_root = prototypes[0]
    counts = np.bincount(concept_indices, minlength=M).astype(np.float64)  # m_k

    if loss_direction == "i2t":
        root_coeff = np.full(M, m / M, dtype=np.float64)
    elif loss_direction == "t2i":
        root_coeff = counts
    elif loss_direction == "both":
        root_coeff = (m / (2.0 * M)) + (counts / 2.0)
    else:
        raise ValueError(f"Unsupported loss_direction={loss_direction!r}")

    targets = counts[:, None] * prototypes - root_coeff[:, None] * mu_root  # (M, d)
    target_norms = np.linalg.norm(targets, axis=1)

    W_norm = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-12, None)
    target_norm = targets / np.clip(target_norms[:, None], 1e-12, None)
    cos_sim = np.sum(W_norm * target_norm, axis=1)  # (M,)

    by_depth = {}
    for depth in range(1, tree.get_max_depth() + 1):
        node_ids = [node.id for node in tree.get_nodes_at_depth(depth)]
        if not node_ids:
            continue
        values = cos_sim[node_ids]
        by_depth[depth] = {"mean": float(values.mean()), "min": float(values.min())}

    # exclude root (index 0) and any node whose target is degenerate
    valid = (target_norms > 1e-12)
    valid[0] = False
    nonroot_cos = cos_sim[valid]
    return {
        "by_depth": by_depth,
        "global_mean_nonroot": float(nonroot_cos.mean()),
        "global_min_nonroot": float(nonroot_cos.min()),
    }


def _build_single_summary(
    strict_data_path: str,
    depth: int,
    branching_factor: int,
    caption_mode: str,
    temperature: float,
    seed: int,
    lambda_reg: float,
    max_iter: int,
    num_threads: int,
    loss_direction: str,
) -> dict:
    strict_data_path = _resolve_strict_data_path(strict_data_path)
    data = np.load(strict_data_path)
    X = data["X"]
    leaf_assignments = data["leaf_assignments"]
    W_true = data["W_true"]

    if W_true.shape[0] != W_true.shape[1] or not np.allclose(W_true, np.eye(W_true.shape[0]), atol=1e-8):
        raise ValueError("This script only supports strict identity-encoder artifacts with W_true = I")

    tree = build_balanced_tree(depth, branching_factor)
    concept_indices = _sample_concepts(tree, leaf_assignments, caption_mode, seed)
    solve = _solve_identity_strict_run(
        X=X,
        concept_indices=concept_indices,
        V=tree.num_nodes,
        temperature=temperature,
        lambda_reg=lambda_reg,
        max_iter=max_iter,
        num_threads=num_threads,
        loss_direction=loss_direction,
    )
    tree_cache = precompute_tree_structure(tree)
    learned_by_depth = _compute_depth_collapsed_margins(solve["W"], tree_cache)
    prototype_by_depth = _compute_depth_collapsed_margins(data["prototypes"], tree_cache)
    nonvacuous_margins = [stats["margin"] for stats in learned_by_depth.values() if not stats["vacuous"]]
    logits = (solve["W"] @ X) / temperature
    margin_minus_prototype = {
        depth_: float(learned_by_depth[depth_]["margin"] - prototype_by_depth[depth_]["margin"])
        for depth_ in learned_by_depth
        if not learned_by_depth[depth_]["vacuous"]
    }

    return {
        "strict_data": strict_data_path,
        "temperature": temperature,
        "lambda_reg": lambda_reg,
        "seed": seed,
        "post_loss": solve["post_loss"],
        "grad_norm": solve["grad_norm"],
        "num_closures": solve["num_closures"],
        "depth_collapsed_margin_by_depth": learned_by_depth,
        "prototype_depth_collapsed_margin_by_depth": prototype_by_depth,
        "margin_minus_prototype_by_depth": margin_minus_prototype,
        "global_min_depth_collapsed_margin": float(min(nonvacuous_margins)) if nonvacuous_margins else None,
        "global_min_margin_minus_prototype": float(min(margin_minus_prototype.values())) if margin_minus_prototype else None,
        "all_nonnegative": bool(all(margin >= 0 for margin in nonvacuous_margins)),
        "i2t_softmax_uniformity": _compute_softmax_uniformity(logits, 1.0 / tree.num_nodes, axis=0),
        "t2i_softmax_uniformity": _compute_softmax_uniformity(logits, 1.0 / X.shape[1], axis=1),
        "hierarchical_violation": _compute_hierarchical_violation(solve["W"], tree, tree_cache),
        "prototype_alignment": _compute_prototype_alignment(solve["W"], data["prototypes"], tree),
        "cor_form_alignment": _compute_cor_form_alignment(
            solve["W"], data["prototypes"], concept_indices, tree, loss_direction=loss_direction
        ),
    }


def _find_run_params_dir(results_dir: Path, run_id: str) -> Path:
    patterns = [f"mlruns_*/*/{run_id}", f"mlruns_merged/*/{run_id}"]
    for pattern in patterns:
        for candidate in sorted(results_dir.glob(pattern)):
            if (candidate / "params").is_dir():
                return candidate / "params"
    raise FileNotFoundError(f"could not find MLflow params for run_id={run_id} under {results_dir}")


def _read_param(params_dir: Path, name: str, cast, default=None):
    path = params_dir / name
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"missing required param {name} in {params_dir}")
        return default
    return cast(path.read_text().strip())


def _read_run_config(results_dir: Path, run_id: str, args) -> dict:
    params_dir = _find_run_params_dir(results_dir, run_id)
    return {
        "run_id": run_id,
        "strict_data": str(results_dir / "strict_data" / run_id / "strict_data.npz"),
        "depth": _read_param(params_dir, "depth", int, args.depth),
        "branching_factor": _read_param(params_dir, "branching_factor", int, args.branching_factor),
        "caption_mode": _read_param(params_dir, "caption_mode", str, args.caption_mode),
        "temperature": _read_param(params_dir, "temperature", float, args.temperature),
        "seed": _read_param(params_dir, "seed", int, args.seed),
        "loss_direction": _read_param(params_dir, "loss_direction", str, args.loss_direction or "both"),
        "lambda_reg": _read_param(params_dir, "lambda_reg", float),
        "d_over_v": _read_param(params_dir, "d_over_v", float, None),
        "sigma": _read_param(params_dir, "strict_sigma", float, None),
        "images_per_leaf": _read_param(params_dir, "images_per_leaf", int, None),
    }


def _build_batch_record(run_config: dict, summary: dict) -> dict:
    depth4 = summary["depth_collapsed_margin_by_depth"][4]
    proto_depth4 = summary["prototype_depth_collapsed_margin_by_depth"][4]
    return {
        "run_id": run_config["run_id"],
        "d_over_v": run_config["d_over_v"],
        "lambda_reg": run_config["lambda_reg"],
        "sigma": run_config["sigma"],
        "images_per_leaf": run_config["images_per_leaf"],
        "global_min_margin": summary["global_min_depth_collapsed_margin"],
        "depth4_margin": depth4["margin"],
        "depth4_proto_margin": proto_depth4["margin"],
        "depth4_margin_minus_proto": summary["margin_minus_prototype_by_depth"][4],
        "all_nonnegative": summary["all_nonnegative"],
        "i2t_tv_mean": summary["i2t_softmax_uniformity"]["tv_mean"],
        "t2i_tv_mean": summary["t2i_softmax_uniformity"]["tv_mean"],
        "i2t_entropy_gap_mean": summary["i2t_softmax_uniformity"]["entropy_gap_mean"],
        "hier_violation_mean": summary["hierarchical_violation"]["node_weighted_mean"],
        "hier_violation_max_depth": summary["hierarchical_violation"]["max_over_depths"],
        "prototype_align_mean_nonroot": summary["prototype_alignment"]["global_mean_nonroot"],
        "prototype_align_min_nonroot": summary["prototype_alignment"]["global_min_nonroot"],
        "cor_form_align_mean_nonroot": summary["cor_form_alignment"]["global_mean_nonroot"],
        "cor_form_align_min_nonroot": summary["cor_form_alignment"]["global_min_nonroot"],
    }


def main():
    args = parse_args()
    if args.strict_data is not None:
        summary = _build_single_summary(
            strict_data_path=args.strict_data,
            depth=args.depth,
            branching_factor=args.branching_factor,
            caption_mode=args.caption_mode,
            temperature=args.temperature,
            seed=args.seed,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            num_threads=args.num_threads,
            loss_direction=args.loss_direction or "both",
        )
        print(json.dumps(summary, indent=2))
        return

    results_dir = _resolve_results_dir(args.results_dir)
    run_filter = {run_id.strip() for run_id in args.run_ids.split(",") if run_id.strip()}
    run_ids = sorted(path.name for path in (results_dir / "strict_data").iterdir() if path.is_dir())
    if run_filter:
        run_ids = [run_id for run_id in run_ids if run_id in run_filter]

    results = []
    for run_id in run_ids:
        run_config = _read_run_config(results_dir, run_id, args)
        summary = _build_single_summary(
            strict_data_path=run_config["strict_data"],
            depth=run_config["depth"],
            branching_factor=run_config["branching_factor"],
            caption_mode=run_config["caption_mode"],
            temperature=run_config["temperature"],
            seed=run_config["seed"],
            lambda_reg=run_config["lambda_reg"],
            max_iter=args.max_iter,
            num_threads=args.num_threads,
            loss_direction=run_config["loss_direction"],
        )
        results.append(_build_batch_record(run_config, summary))

    results.sort(key=lambda row: (row["lambda_reg"], row["sigma"], row["images_per_leaf"], row["d_over_v"]))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

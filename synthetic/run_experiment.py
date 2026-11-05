"""Experiment script for hierarchical learning dynamics (single run or sweep)."""

import sys
import os
import argparse
import gc
import itertools
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from hierarchy import build_balanced_tree
from synthetic_data import (
    assign_images_per_leaf,
    build_support_matrix,
    build_caption_distribution,
    build_hierarchical_prototypes,
    sample_leaf_latents,
    sample_observed_images,
    compute_centered_support_matrix,
    compute_image_weights,
)
from model import train_ufm, train_ufm_strict, Checkpoint, save_checkpoints, load_checkpoints, CheckpointDir
from metrics import (track_spectral_alignment, track_all_metrics_over_training,
                     track_branch_separation_over_training, track_all_metrics_single_pass)


def plot_loss(checkpoints: list, output_dir: str):
    """Plot training loss over epochs."""
    epochs = [ckpt.epoch for ckpt in checkpoints]
    losses = [ckpt.loss for ckpt in checkpoints]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Also plot log scale
    ax2 = ax.twinx()
    ax2.semilogy(epochs, losses, 'r--', alpha=0.5, linewidth=1)
    ax2.set_ylabel('Loss (log scale)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'loss.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def add_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--depth', type=int, default=3, help='Tree depth (root=0, leaves=depth)')
    parser.add_argument('--branching_factor', type=int, default=3, help='Children per internal node')
    parser.add_argument('--images_per_leaf', type=int, default=150, help='Number of images per leaf')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension d')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Checkpoint every N epochs')
    parser.add_argument('--caption_mode', type=str, default='uniform',
                        choices=['uniform', 'exponential', 'leaf_only', 'linear', 'balanced'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='files/hierarchical_dynamics', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--loss_type', type=str, default='softmax_ce',
                        choices=['softmax_ce', 'contrastive', 'contrastive_strict'])
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--track_branch_separation', type=int, default=0, choices=[0, 1],
                        help='Track branch separation metrics (1=yes, 0=no)')
    parser.add_argument('--strict_data_mode', type=int, default=0, choices=[0, 1],
                        help='Deprecated: strict mode now activates automatically for loss_type=contrastive_strict')
    parser.add_argument('--strict_sigma', type=float, default=0.1,
                        help='Latent Gaussian noise sigma for strict data generation')
    parser.add_argument('--strict_p_over_d', type=float, default=1.0,
                        help='Observed feature ratio p/d for strict data generation')
    parser.add_argument('--strict_theta_mode', type=str, default='geometric_decay',
                        choices=['constant', 'linear_decay', 'geometric_decay'],
                        help='Depth-wise theta schedule mode for strict prototypes')
    parser.add_argument('--strict_theta_values', type=str, default='40',
                        help='Theta values (constant: theta_deg, linear_decay: start_deg,end_deg, geometric_decay: theta0_deg[,r])')
    parser.add_argument('--strict_save_artifacts', type=int, default=1, choices=[0, 1],
                        help='Save strict synthetic data artifacts (1=yes, 0=no)')
    parser.add_argument('--learnable_temperature', type=int, default=0, choices=[0, 1],
                        help='Use learnable temperature (1=yes, 0=no)')
    parser.add_argument('--strict_identity_encoder', type=int, default=0, choices=[0, 1],
                        help='Use identity W_true and frozen identity W_img (forces p_over_d=1)')
    parser.add_argument('--strict_normalize', type=int, default=1, choices=[0, 1],
                        help='L2-normalize embeddings before loss (0=unnormalized, matching analytic model)')
    parser.add_argument('--loss_direction', type=str, default='both', choices=['both', 'i2t', 't2i'],
                        help='Loss direction: both (i2t+t2i), i2t only, or t2i only')
    parser.add_argument('--prototype_shift_norm', type=float, default=0.0,
                        help='Additive shift s*e_0 applied to every prototype before Z sampling '
                             '(shifts leaf centroid = paper mu_root). 0 = centered-root baseline.')


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 3, 4], help='Tree depths')
    parser.add_argument('--branching_factors', type=int, nargs='+', default=[2, 3, 4, 5], help='Branching factors')
    parser.add_argument('--caption_modes', type=str, nargs='+', default=['uniform', 'exponential', 'leaf_only'],
                        help='Caption modes')
    parser.add_argument('--loss_types', type=str, nargs='+', default=['softmax_ce', 'contrastive'],
                        help='Loss types')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.01, 0.07, 0.1],
                        help='Temperatures for contrastive loss')
    parser.add_argument('--lambda_regs', type=float, nargs='+', default=[0.0, 1e-4, 1e-3],
                        help='Regularization strengths')
    parser.add_argument('--images_per_leaf', type=int, default=150, help='Number of images per leaf')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension d')
    parser.add_argument('--d_over_v', type=float, nargs='*', default=[],
                        help='d/V ratios (overrides embedding_dim when provided)')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lrs', type=float, nargs='+', default=[0.01], help='Learning rates')
    parser.add_argument('--optimizers', type=str, nargs='+', default=['adam'], help='Optimizers (adam, adamw, sgd)')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output_dir', type=str, default='files/ufm_sweep', help='Output directory')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1], help='Verbose training output')
    parser.add_argument('--use_mlflow', type=int, default=1, choices=[0, 1], help='Use MLflow tracking (1=yes, 0=no)')
    parser.add_argument('--mlflow_experiment', type=str, default='ufm_branch_separation',
                        help='MLflow experiment name')
    parser.add_argument('--mlruns_subdir', type=str, default='mlruns',
                        help='Subdirectory name for local file-based MLflow store (e.g. mlruns_0 for per-worker isolation)')
    parser.add_argument('--strict_data_mode', type=int, default=0, choices=[0, 1],
                        help='Deprecated: strict mode now activates automatically for loss_type=contrastive_strict')
    parser.add_argument('--strict_sigma', type=float, nargs='+', default=[0.1],
                        help='Sigma values for strict data generation')
    parser.add_argument('--strict_p_over_d', type=float, nargs='+', default=[1.0],
                        help='p/d ratios for strict data generation')
    parser.add_argument('--strict_theta_mode', type=str, default='geometric_decay',
                        choices=['constant', 'linear_decay', 'geometric_decay'],
                        help='Depth-wise theta schedule mode for strict prototypes')
    parser.add_argument('--strict_theta_values', type=str, default='40',
                        help='Theta values (constant: theta_deg, linear_decay: start_deg,end_deg, geometric_decay: theta0_deg[,r])')
    parser.add_argument('--strict_save_artifacts', type=int, default=1, choices=[0, 1],
                        help='Save strict synthetic data artifacts (1=yes, 0=no)')
    parser.add_argument('--save_checkpoints', type=int, default=0, choices=[0, 1],
                        help='Stream checkpoints to disk during sweep (1=yes, 0=no). Disabled by default to save disk space.')
    parser.add_argument('--strict_identity_encoder', type=int, default=0, choices=[0, 1],
                        help='Use identity W_true and frozen identity W_img (forces p_over_d=1)')
    parser.add_argument('--strict_normalize', type=int, default=1, choices=[0, 1],
                        help='L2-normalize embeddings before loss (0=unnormalized, matching analytic model)')
    parser.add_argument('--loss_direction', type=str, default='both', choices=['both', 'i2t', 't2i'],
                        help='Loss direction: both (i2t+t2i), i2t only, or t2i only')
    parser.add_argument('--learnable_temperatures', type=int, nargs='+', default=[0],
                        help='0/1 for learnable temperature')
    parser.add_argument('--prototype_shift_norms', type=float, nargs='+', default=[0.0],
                        help='Additive shift s*e_0 applied to every prototype before Z sampling. '
                             'One or more values; sweeps as outer product with lambdas.')
    parser.add_argument('--config_index', type=int, default=-1,
                        help='Run only this config index (0-based). -1 = run all.')
    parser.add_argument('--count_configs', type=int, default=0, choices=[0, 1],
                        help='Print total config count and exit (1=yes, 0=no)')


def add_null_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--depth', type=str, default='3', help='Comma-separated tree depths')
    parser.add_argument('--branching_factor', type=int, default=3, help='Children per internal node')
    parser.add_argument('--images_per_leaf', type=int, default=150, help='Number of images per leaf')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension d')
    parser.add_argument('--d_over_v', type=str, default='',
                        help='Comma-separated d/V ratios (overrides embedding_dim when provided)')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='Checkpoint every N epochs')
    parser.add_argument('--n_permutations', type=int, default=100,
                        help='Number of permutations for null model test')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--output_dir', type=str, default='files/analytic_model/null_model', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--use_mlflow', type=int, default=0, choices=[0, 1], help='Use MLflow tracking')
    parser.add_argument('--mlflow_experiment', type=str, default='ufm_null_model',
                        help='MLflow experiment name')


def add_recompute_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Sweep output directory containing checkpoints/ and mlflow.db')
    parser.add_argument('--mlflow_experiment', type=str, default='ufm_branch_separation',
                        help='MLflow experiment name')
    parser.add_argument('--metrics', type=str, default='both',
                        choices=['branch_sep', 'all_metrics', 'both'],
                        help='Which metric groups to recompute')
    parser.add_argument('--run_ids', type=str, default='',
                        help='Comma-separated list of specific run IDs to recompute (default: all)')


def parse_args():
    parser = argparse.ArgumentParser(description='Hierarchical Learning Dynamics Experiments')
    subparsers = parser.add_subparsers(dest='command', required=True)
    single_parser = subparsers.add_parser('single', help='Run a single experiment')
    add_single_args(single_parser)
    sweep_parser = subparsers.add_parser('sweep', help='Run a sweep over parameters')
    add_sweep_args(sweep_parser)
    null_model_parser = subparsers.add_parser('null_model', help='Run null model permutation test')
    add_null_model_args(null_model_parser)
    recompute_parser = subparsers.add_parser('recompute_metrics',
                                              help='Recompute metrics from saved checkpoints')
    add_recompute_args(recompute_parser)
    argv = sys.argv[1:]
    if argv:
        if argv[0] not in {'single', 'sweep', 'null_model', 'recompute_metrics', '-h', '--help'}:
            argv = ['single'] + argv
    else:
        argv = ['single']
    return parser.parse_args(argv)


def plot_spectral_dynamics(spectral_data: dict, output_dir: str):
    """Plot spectral alignment over training."""
    epochs = spectral_data['epochs']
    projections = spectral_data['projections']
    singular_values = spectral_data['singular_values']
    k = projections.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        label = f'Direction {i+1} (sigma={singular_values[i]:.2f})'
        ax.plot(epochs, projections[:, i], label=label, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Projection Norm ||H @ v_i||', fontsize=12)
    ax.set_title('Spectral Dynamics: Alignment with Singular Directions of S_bar', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'spectral_dynamics.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_orthogonality(results: dict, output_dir: str):
    """Plot orthogonality for all embedding types (A, W_mean, H_mean)."""
    epochs = results['epochs']
    emb_types = ['A', 'W_mean', 'H_mean']
    emb_labels = {
        'A': 'Text (A) - Direct',
        'W_mean': 'Text (W) - Mean of Descendants',
        'H_mean': 'Image (H) - Mean of Descendants'
    }
    colors = {'A': 'tab:blue', 'W_mean': 'tab:orange', 'H_mean': 'tab:green'}

    # Get depths
    depths = sorted(results['orthogonality']['A'].keys())

    # Plot 1: All embedding types on same plot, one subplot per depth
    fig, axes = plt.subplots(1, len(depths), figsize=(5 * len(depths), 5), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        for emb_type in emb_types:
            scores = results['orthogonality'][emb_type][depth]
            ax.plot(epochs, scores, label=emb_labels[emb_type], color=colors[emb_type], linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_title(f'Depth {depth}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if depth == depths[0]:
            ax.set_ylabel('Mean cos(p, c-p)', fontsize=11)

    axes[-1].legend(loc='upper right', fontsize=9)
    fig.suptitle('Hierarchical Orthogonality by Embedding Type', fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'orthogonality_all_types.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Final orthogonality comparison (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(depths))
    width = 0.25

    for i, emb_type in enumerate(emb_types):
        final_scores = [results['orthogonality'][emb_type][d][-1] for d in depths]
        bars = ax.bar(x + i * width, final_scores, width, label=emb_labels[emb_type], color=colors[emb_type])
        for bar, score in zip(bars, final_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Mean cos(p, c-p)', fontsize=12)
    ax.set_title('Final Orthogonality by Depth and Embedding Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'orthogonality_final_comparison.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_cone_angles(results: dict, output_dir: str):
    """Plot cone angles for all embedding types."""
    epochs = results['epochs']
    emb_types = ['A', 'W_mean', 'H_mean']
    emb_labels = {
        'A': 'Text (A) - Direct',
        'W_mean': 'Text (W) - Mean of Descendants',
        'H_mean': 'Image (H) - Mean of Descendants'
    }
    colors = {'A': 'tab:blue', 'W_mean': 'tab:orange', 'H_mean': 'tab:green'}

    depths = sorted(results['cone_angles']['A'].keys())

    # Plot 1: Cone angles over training, one subplot per embedding type
    fig, axes = plt.subplots(1, len(emb_types), figsize=(5 * len(emb_types), 5), sharey=True)

    for ax, emb_type in zip(axes, emb_types):
        for depth in depths:
            angles = results['cone_angles'][emb_type][depth]
            ax.plot(epochs, angles, label=f'Depth {depth}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_title(emb_labels[emb_type], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if emb_type == emb_types[0]:
            ax.set_ylabel('Mean Sibling Cone Angle (deg)', fontsize=11)

    fig.suptitle('Cone Angles Over Training by Embedding Type', fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'cone_angles_over_training_all.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Final cone angles comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(depths))
    width = 0.25

    for i, emb_type in enumerate(emb_types):
        final_angles = [results['cone_angles'][emb_type][d][-1] for d in depths]
        bars = ax.bar(x + i * width, final_angles, width, label=emb_labels[emb_type], color=colors[emb_type])
        for bar, angle in zip(bars, final_angles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{angle:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Mean Sibling Cone Angle (degrees)', fontsize=12)
    ax.set_title('Final Cone Angles by Depth and Embedding Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'cone_angles_final_comparison.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Shrinking cone effect - final angles vs depth for each type
    fig, ax = plt.subplots(figsize=(8, 6))
    for emb_type in emb_types:
        final_angles = [results['cone_angles'][emb_type][d][-1] for d in depths]
        ax.plot(depths, final_angles, 'o-', label=emb_labels[emb_type], color=colors[emb_type],
                linewidth=2, markersize=8)

    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Mean Sibling Cone Angle (degrees)', fontsize=12)
    ax.set_title('Shrinking Cone Effect: Final Angles vs Depth', fontsize=14)
    ax.set_xticks(depths)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'shrinking_cone_effect.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_branch_separation_over_training(branch_sep_data: dict, output_dir: str):
    """Plot branch separation metrics over training."""
    epochs = branch_sep_data['epochs']
    emb_types = ['A', 'W_mean', 'H_mean']
    emb_labels = {
        'A': 'Text (A) - Direct',
        'W_mean': 'Text (W) - Mean of Descendants',
        'H_mean': 'Image (H) - Mean of Descendants'
    }
    colors = {'A': 'tab:blue', 'W_mean': 'tab:orange', 'H_mean': 'tab:green'}

    depths = sorted(branch_sep_data['A'].keys())

    # Plot 1: Violation rate over training, one subplot per embedding type
    fig, axes = plt.subplots(1, len(emb_types), figsize=(5 * len(emb_types), 5), sharey=True)

    for ax, emb_type in zip(axes, emb_types):
        for depth in depths:
            vr = branch_sep_data[emb_type][depth]['violation_rate'] * 100  # Convert to %
            ax.plot(epochs, vr, label=f'Depth {depth}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_title(emb_labels[emb_type], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if emb_type == emb_types[0]:
            ax.set_ylabel('Violation Rate (%)', fontsize=11)

    fig.suptitle('Branch Separation Violation Rate Over Training', fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'branch_separation_violation_rate.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Mean relative violation over training
    fig, axes = plt.subplots(1, len(emb_types), figsize=(5 * len(emb_types), 5), sharey=True)

    for ax, emb_type in zip(axes, emb_types):
        for depth in depths:
            mrv = branch_sep_data[emb_type][depth]['mean_relative_violation']
            ax.plot(epochs, mrv, label=f'Depth {depth}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_title(emb_labels[emb_type], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if emb_type == emb_types[0]:
            ax.set_ylabel('Mean Relative Violation', fontsize=11)

    fig.suptitle('Branch Separation Mean Relative Violation Over Training', fontsize=14, y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'branch_separation_relative_violation.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Final violation rate comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(depths))
    width = 0.25

    for i, emb_type in enumerate(emb_types):
        final_vr = [branch_sep_data[emb_type][d]['violation_rate'][-1] * 100 for d in depths]
        bars = ax.bar(x + i * width, final_vr, width, label=emb_labels[emb_type], color=colors[emb_type])
        for bar, vr in zip(bars, final_vr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{vr:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Violation Rate (%)', fontsize=12)
    ax.set_title('Final Branch Separation Violation Rate by Depth and Embedding Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'branch_separation_final_comparison.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_csv_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def _is_strict_loss(loss_type: str) -> bool:
    return loss_type == 'contrastive_strict'


def _build_theta_schedule(max_depth: int, mode: str, theta_values: str, branching_factor: int) -> dict[int, float]:
    values_deg = parse_csv_list(theta_values, float) if theta_values else []
    if mode == 'constant':
        theta = values_deg[0] if values_deg else 30.0
        values_deg = [theta] * max_depth
    elif mode == 'linear_decay':
        if not values_deg:
            start_deg, end_deg = 40.0, 20.0
        elif len(values_deg) == 1:
            start_deg = end_deg = values_deg[0]
        else:
            start_deg, end_deg = values_deg[0], values_deg[1]
        values_deg = np.linspace(start_deg, end_deg, max_depth).tolist()
    elif mode == 'geometric_decay':
        theta0_deg = values_deg[0] if values_deg else 40.0
        r = values_deg[1] if len(values_deg) > 1 else 0.4
        if r <= 0:
            raise ValueError("geometric_decay requires r > 0")
        r_effective = min(r, 1.0 / branching_factor)
        values_deg = [theta0_deg * (r_effective ** (depth - 1)) for depth in range(1, max_depth + 1)]
    else:
        raise ValueError(f"Unknown strict_theta_mode: {mode}")

    if any(theta_deg <= 0 or theta_deg >= 89.9 for theta_deg in values_deg):
        raise ValueError("All theta values must be in (0, 89.9) degrees")

    return {depth: np.deg2rad(values_deg[depth - 1]) for depth in range(1, max_depth + 1)}


def _prepare_strict_data(
    tree,
    leaf_assignments: np.ndarray,
    embedding_dim: int,
    sigma: float,
    p_over_d: float,
    theta_mode: str,
    theta_values: str,
    seed: int,
    identity_encoder: bool = False,
    prototype_shift_norm: float = 0.0,
) -> dict:
    if identity_encoder:
        p_over_d = 1.0
    if p_over_d <= 0:
        raise ValueError("strict_p_over_d must be > 0")

    p_dim = int(round(p_over_d * embedding_dim))
    if p_dim < 1:
        raise ValueError(f"strict_p_over_d={p_over_d} yields invalid p_dim={p_dim}")

    theta_by_depth = _build_theta_schedule(
        max_depth=tree.get_max_depth(),
        mode=theta_mode,
        theta_values=theta_values,
        branching_factor=len(tree.root.children),
    )
    prototypes = build_hierarchical_prototypes(
        tree=tree,
        d=embedding_dim,
        theta_by_depth=theta_by_depth,
        root_scale=0.8,
        root_at_zero=1,
    )
    if prototype_shift_norm != 0.0:
        # Additive shift on every prototype: moves leaf centroid (paper's mu_root)
        # off zero while leaving non-root cosines essentially unchanged for small s.
        prototypes[:, 0] += float(prototype_shift_norm)

    rng = np.random.default_rng(seed)
    Z = sample_leaf_latents(
        tree=tree,
        leaf_assignments=leaf_assignments,
        prototypes=prototypes,
        sigma=sigma,
        rng=rng,
    )
    W_true_arg = np.eye(embedding_dim, dtype=np.float32) if identity_encoder else None
    X, W_true = sample_observed_images(Z=Z, p_dim=p_dim, rng=rng, W_true=W_true_arg)

    return {
        'prototypes': prototypes,
        'Z': Z,
        'X': X,
        'W_true': W_true,
        'theta_by_depth': theta_by_depth,
        'p_dim': p_dim,
    }


def _save_strict_artifacts(output_dir: str, strict_data: dict, leaf_assignments: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, 'strict_data.npz'),
        prototypes=strict_data['prototypes'],
        Z=strict_data['Z'],
        X=strict_data['X'],
        W_true=strict_data['W_true'],
        leaf_assignments=leaf_assignments.astype(np.int64),
    )
    theta_deg = {int(depth): float(np.rad2deg(theta)) for depth, theta in strict_data['theta_by_depth'].items()}
    with open(os.path.join(output_dir, 'strict_config.json'), 'w') as f:
        json.dump(
            {
                'p_dim': int(strict_data['p_dim']),
                'theta_by_depth_deg': theta_deg,
                'sigma': float(np.std(strict_data['Z'] - strict_data['prototypes'][leaf_assignments].T)),
            },
            f,
            indent=2,
        )


def run_single_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    print(f"\n[1] Building balanced tree (depth={args.depth}, branching={args.branching_factor})")
    tree = build_balanced_tree(args.depth, args.branching_factor)
    V = tree.num_nodes
    num_leaves = len(tree.get_leaves())
    num_images = num_leaves * args.images_per_leaf
    print(f"    Total nodes (V): {V}, Leaves: {num_leaves}")

    # 2. Assign images to leaves (fixed number per leaf)
    print(f"\n[2] Assigning {args.images_per_leaf} images per leaf ({num_images} total)")
    leaf_assignments = assign_images_per_leaf(tree, args.images_per_leaf)

    # 3. Build support matrix and caption distributions
    print(f"\n[3] Building support matrix S and caption distribution P")
    S = build_support_matrix(tree, leaf_assignments)
    S_bar = compute_centered_support_matrix(S)
    strict_data = None
    if _is_strict_loss(args.loss_type):
        strict_data = _prepare_strict_data(
            tree=tree,
            leaf_assignments=leaf_assignments,
            embedding_dim=args.embedding_dim,
            sigma=args.strict_sigma,
            p_over_d=args.strict_p_over_d,
            theta_mode=args.strict_theta_mode,
            theta_values=args.strict_theta_values,
            seed=args.seed,
            identity_encoder=bool(args.strict_identity_encoder),
            prototype_shift_norm=float(getattr(args, 'prototype_shift_norm', 0.0)),
        )
        if args.strict_save_artifacts:
            _save_strict_artifacts(os.path.join(args.output_dir, 'strict_data'), strict_data, leaf_assignments)
    P = build_caption_distribution(tree, leaf_assignments, mode=args.caption_mode)
    pi = compute_image_weights(leaf_assignments)
    print(f"    S shape: {S.shape}, P shape: {P.shape}")
    if strict_data is not None:
        print(
            f"    Loss type: {args.loss_type}, Caption mode: {args.caption_mode}, "
            f"strict data: sigma={args.strict_sigma}, p/d={args.strict_p_over_d}, p_dim={strict_data['p_dim']}"
        )
    else:
        print(f"    Loss type: {args.loss_type}, Caption mode: {args.caption_mode}")

    # 4. Train UFM
    loss_info = f"loss={args.loss_type}"
    if args.loss_type in {'contrastive', 'contrastive_strict'}:
        loss_info += f", temp={args.temperature}"
    print(f"\n[4] Training UFM (d={args.embedding_dim}, epochs={args.epochs}, lr={args.lr}, lambda={args.lambda_reg}, {loss_info})")
    learn_temp = bool(args.learnable_temperature)
    if _is_strict_loss(args.loss_type):
        rng = np.random.default_rng(args.seed)
        concept_indices = np.array([rng.choice(V, p=P[:, j]) for j in range(num_images)])
        model, checkpoints = train_ufm_strict(
            V=V,
            d=args.embedding_dim,
            X=strict_data['X'],
            concept_indices=concept_indices,
            lambda_reg=args.lambda_reg,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
            verbose=True,
            temperature=args.temperature,
            optimizer_type='adam',
            learnable_temperature=learn_temp,
            freeze_w_img=bool(args.strict_identity_encoder),
            normalize=bool(args.strict_normalize),
            loss_direction=args.loss_direction,
        )
    else:
        model, checkpoints = train_ufm(
            V=V,
            d=args.embedding_dim,
            m=num_images,
            P=P,
            pi=pi,
            lambda_reg=args.lambda_reg,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
            verbose=True,
            loss_type=args.loss_type,
            temperature=args.temperature,
            S=S,
            learnable_temperature=learn_temp,
            loss_direction=args.loss_direction,
        )
    print(f"    Saved {len(checkpoints)} checkpoints")

    # 5. Compute metrics
    print(f"\n[5] Computing metrics")

    # Spectral dynamics
    k_spectral = min(10, V - 1)
    spectral_data = track_spectral_alignment(checkpoints, S_bar, k=k_spectral)
    print(f"    Spectral: tracking top-{k_spectral} singular directions")

    # All orthogonality and cone angle metrics
    print(f"    Computing orthogonality and cone angles for all embedding types...")
    all_metrics = track_all_metrics_over_training(checkpoints, tree, leaf_assignments)
    print(f"    Done. Embedding types: A, W_mean, H_mean")

    # Branch separation tracking (optional)
    branch_sep_data = None
    if args.track_branch_separation:
        print(f"    Computing branch separation metrics...")
        branch_sep_data = track_branch_separation_over_training(checkpoints, tree, leaf_assignments)
        print(f"    Done. Tracking violation rate, mean exceeding angle, mean relative violation")

    # 6. Generate plots
    print(f"\n[6] Generating plots")
    plot_loss(checkpoints, args.output_dir)
    plot_spectral_dynamics(spectral_data, args.output_dir)
    plot_all_orthogonality(all_metrics, args.output_dir)
    plot_all_cone_angles(all_metrics, args.output_dir)
    if branch_sep_data is not None:
        plot_branch_separation_over_training(branch_sep_data, args.output_dir)
    print(f"    Saved plots to {args.output_dir}")

    # 7. Print summary
    print(f"\n[7] Summary")
    final_loss = checkpoints[-1].loss
    print(f"    Final loss: {final_loss:.6f}")

    depths = sorted(all_metrics['orthogonality']['A'].keys())

    print(f"\n    Orthogonality (final epoch):")
    for emb_type in ['A', 'W_mean', 'H_mean']:
        print(f"      {emb_type}:")
        for d in depths:
            score = all_metrics['orthogonality'][emb_type][d][-1]
            print(f"        Depth {d}: cos = {score:.4f}")

    print(f"\n    Cone angles (final epoch):")
    for emb_type in ['A', 'W_mean', 'H_mean']:
        print(f"      {emb_type}:")
        for d in depths:
            angle = all_metrics['cone_angles'][emb_type][d][-1]
            print(f"        Depth {d}: {angle:.2f} deg")

    print(f"\n    Singular values of S_bar (top {k_spectral}):")
    for i, sv in enumerate(spectral_data['singular_values']):
        print(f"      sigma_{i+1} = {sv:.4f}")

    if branch_sep_data is not None:
        print(f"\n    Branch separation (final epoch):")
        for emb_type in ['A', 'W_mean', 'H_mean']:
            print(f"      {emb_type}:")
            for d in sorted(branch_sep_data[emb_type].keys()):
                vr = branch_sep_data[emb_type][d]['violation_rate'][-1] * 100
                mrv = branch_sep_data[emb_type][d]['mean_relative_violation'][-1]
                print(f"        Depth {d}: viol_rate = {vr:.2f}%, mean_rel_viol = {mrv:.4f}")

    print(f"\nExperiment complete. Results saved to {args.output_dir}")


def run_sweep_experiment(config: dict, use_mlflow: bool = False) -> dict:
    tree = build_balanced_tree(config['depth'], config['branching_factor'])
    V = tree.num_nodes
    num_leaves = len(tree.get_leaves())
    num_images = num_leaves * config['images_per_leaf']

    leaf_assignments = assign_images_per_leaf(tree, config['images_per_leaf'])
    S = build_support_matrix(tree, leaf_assignments)
    strict_data = None
    if _is_strict_loss(config['loss_type']):
        strict_data = _prepare_strict_data(
            tree=tree,
            leaf_assignments=leaf_assignments,
            embedding_dim=config['embedding_dim'],
            sigma=config['strict_sigma'],
            p_over_d=config['strict_p_over_d'],
            theta_mode=config['strict_theta_mode'],
            theta_values=config['strict_theta_values'],
            seed=config['seed'],
            identity_encoder=bool(config.get('strict_identity_encoder', 0)),
            prototype_shift_norm=float(config.get('prototype_shift_norm', 0.0)),
        )
    P = build_caption_distribution(tree, leaf_assignments, mode=config['caption_mode'])
    pi = compute_image_weights(leaf_assignments)

    if strict_data is not None and config.get('strict_save_artifacts'):
        run_tag = config.get('run_tag', 'run')
        if use_mlflow:
            run_tag = mlflow.active_run().info.run_id
        _save_strict_artifacts(
            os.path.join(config['output_dir'], 'strict_data', run_tag),
            strict_data,
            leaf_assignments,
        )

    # Determine checkpoint_dir for streaming checkpoints to disk
    ckpt_subdir = None
    if config.get('save_checkpoints') and use_mlflow and config.get('output_dir'):
        run_id = mlflow.active_run().info.run_id
        ckpt_subdir = os.path.join(config['output_dir'], 'checkpoints', run_id)

    learn_temp = bool(config.get('learnable_temperature', 0))
    if _is_strict_loss(config['loss_type']):
        rng = np.random.default_rng(config['seed'])
        concept_indices = np.array([rng.choice(V, p=P[:, j]) for j in range(num_images)])
        model, checkpoints = train_ufm_strict(
            V=V,
            d=config['embedding_dim'],
            X=strict_data['X'],
            concept_indices=concept_indices,
            lambda_reg=config['lambda_reg'],
            epochs=config['epochs'],
            lr=config['lr'],
            checkpoint_interval=config['checkpoint_interval'],
            device=config['device'],
            verbose=config['verbose'],
            temperature=config['temperature'],
            mlflow_log=False,
            optimizer_type=config.get('optimizer', 'adam'),
            checkpoint_dir=ckpt_subdir,
            learnable_temperature=learn_temp,
            freeze_w_img=bool(config.get('strict_identity_encoder', 0)),
            normalize=bool(config.get('strict_normalize', 1)),
            loss_direction=config.get('loss_direction', 'both'),
        )
    else:
        model, checkpoints = train_ufm(
            V=V,
            d=config['embedding_dim'],
            m=num_images,
            P=P,
            pi=pi,
            lambda_reg=config['lambda_reg'],
            epochs=config['epochs'],
            lr=config['lr'],
            checkpoint_interval=config['checkpoint_interval'],
            device=config['device'],
            verbose=config['verbose'],
            loss_type=config['loss_type'],
            temperature=config['temperature'],
            S=S,
            mlflow_log=False,
            optimizer_type=config.get('optimizer', 'adam'),
            checkpoint_dir=ckpt_subdir,
            learnable_temperature=learn_temp,
            loss_direction=config.get('loss_direction', 'both'),
        )

    del model

    # Single-pass metrics: loads one checkpoint at a time from disk
    combined = track_all_metrics_single_pass(checkpoints, tree, leaf_assignments)
    branch_sep = combined['branch_sep']
    all_metrics = combined['all_metrics']

    # Precompute reg_loss decomposition if lambda_reg > 0
    lambda_reg = config['lambda_reg']
    reg_losses = None
    if lambda_reg > 0:
        is_strict = _is_strict_loss(config['loss_type'])
        identity_enc = bool(config.get('strict_identity_encoder', 0))
        reg_losses = []
        for ckpt in checkpoints:
            W_norm_sq = float(np.sum(ckpt.W ** 2))
            if is_strict and identity_enc:
                # W_img = I (frozen), not regularized
                reg = lambda_reg / 2 * W_norm_sq
            elif is_strict:
                # W_img not stored in checkpoint; can't decompose
                reg = None
            else:
                reg = lambda_reg / 2 * (W_norm_sq + float(np.sum(ckpt.H ** 2)))
            reg_losses.append(reg)

    # Log metrics at each checkpoint to MLflow
    if use_mlflow:
        n_ckpts = len(checkpoints)
        for i in range(n_ckpts):
            epoch = checkpoints.epochs[i] if isinstance(checkpoints, CheckpointDir) else checkpoints[i].epoch
            loss = checkpoints.losses[i] if isinstance(checkpoints, CheckpointDir) else checkpoints[i].loss
            mlflow.log_metric("loss", loss, step=epoch)
            temp = checkpoints.temperatures[i] if isinstance(checkpoints, CheckpointDir) else checkpoints[i].temperature
            if temp is not None:
                mlflow.log_metric("temperature", temp, step=epoch)
            if reg_losses is not None and reg_losses[i] is not None:
                mlflow.log_metric("reg_loss", reg_losses[i], step=epoch)
                mlflow.log_metric("main_loss", loss - reg_losses[i], step=epoch)

            for emb_type in ['A', 'W_mean', 'H_mean']:
                for depth in branch_sep[emb_type]:
                    viol_rate = branch_sep[emb_type][depth]['violation_rate'][i]
                    mean_rel_viol = branch_sep[emb_type][depth]['mean_relative_violation'][i]
                    exc_angle = branch_sep[emb_type][depth]['mean_exceeding_angle'][i]
                    mean_theta = branch_sep[emb_type][depth]['mean_theta_i'][i]
                    mlflow.log_metric(f"viol_rate/{emb_type}/d{depth}", viol_rate, step=epoch)
                    mlflow.log_metric(f"mean_rel_viol/{emb_type}/d{depth}", mean_rel_viol, step=epoch)
                    mlflow.log_metric(f"exc_angle/{emb_type}/d{depth}", exc_angle, step=epoch)
                    mlflow.log_metric(f"mean_theta/{emb_type}/d{depth}", mean_theta, step=epoch)

            for emb_type in ['A', 'W_mean', 'H_mean']:
                for depth in all_metrics['orthogonality'][emb_type]:
                    ortho = all_metrics['orthogonality'][emb_type][depth][i]
                    mlflow.log_metric(f"ortho/{emb_type}/d{depth}", ortho, step=epoch)

    # Results dict for CSV (final values only)
    final_loss = checkpoints.losses[-1] if isinstance(checkpoints, CheckpointDir) else checkpoints[-1].loss
    results = {
        'final_loss': final_loss,
        'V': V,
        'num_leaves': num_leaves,
        'num_images': num_images,
    }

    for emb_type in ['A', 'W_mean', 'H_mean']:
        for d in branch_sep[emb_type]:
            results[f'branch_sep_{emb_type}_d{d}_viol_rate'] = branch_sep[emb_type][d]['violation_rate'][-1]
            results[f'branch_sep_{emb_type}_d{d}_mean_rel_viol'] = branch_sep[emb_type][d]['mean_relative_violation'][-1]
            results[f'branch_sep_{emb_type}_d{d}_mean_exc_angle'] = branch_sep[emb_type][d]['mean_exceeding_angle'][-1]
            results[f'branch_sep_{emb_type}_d{d}_mean_theta'] = branch_sep[emb_type][d]['mean_theta_i'][-1]

    for emb_type in ['A', 'W_mean', 'H_mean']:
        for d in all_metrics['orthogonality'][emb_type]:
            results[f'ortho_{emb_type}_d{d}'] = all_metrics['orthogonality'][emb_type][d][-1]

    return results


def run_sweep(args):
    depths = args.depths
    branching_factors = args.branching_factors
    caption_modes = args.caption_modes
    loss_types = args.loss_types
    temperatures = args.temperatures
    lambda_regs = args.lambda_regs
    d_over_v_values = args.d_over_v or []
    lrs = args.lrs
    optimizers = args.optimizers
    strict_p_over_d_values = args.strict_p_over_d or [1.0]
    strict_sigma_values = args.strict_sigma or [0.1]
    learnable_temps = args.learnable_temperatures
    prototype_shift_norms = args.prototype_shift_norms or [0.0]
    has_strict_loss = any(_is_strict_loss(lt) for lt in loss_types)
    if has_strict_loss and args.strict_data_mode:
        print("NOTE: --strict_data_mode is deprecated and ignored; strict mode is enabled by loss_type=contrastive_strict.")

    tree_sizes = {}
    if d_over_v_values:
        for depth, bf in itertools.product(depths, branching_factors):
            tree_sizes[(depth, bf)] = build_balanced_tree(depth, bf).num_nodes

    configs = []
    for depth, bf, lt in itertools.product(depths, branching_factors, loss_types):
        ratio_values = d_over_v_values if d_over_v_values else [None]
        temps_to_use = temperatures if lt in {'contrastive', 'contrastive_strict'} else [0.07]
        lt_values = learnable_temps if lt in {'contrastive', 'contrastive_strict'} else [0]
        caption_modes_for_loss = caption_modes
        p_over_d_values_for_loss = [1.0] if args.strict_identity_encoder else (strict_p_over_d_values if _is_strict_loss(lt) else [1.0])
        sigma_values_for_loss = strict_sigma_values if _is_strict_loss(lt) else [0.1]
        shift_values_for_loss = prototype_shift_norms if _is_strict_loss(lt) else [0.0]
        for cm in caption_modes_for_loss:
            for temp, lam, ratio, lr, opt, p_over_d, learn_temp, sigma, shift in itertools.product(
                temps_to_use, lambda_regs, ratio_values, lrs, optimizers, p_over_d_values_for_loss, lt_values, sigma_values_for_loss, shift_values_for_loss
            ):
                embedding_dim = args.embedding_dim
                if ratio is not None:
                    embedding_dim = int(round(ratio * tree_sizes[(depth, bf)]))
                    if embedding_dim < 1:
                        raise ValueError(f"d_over_v={ratio} yields embedding_dim={embedding_dim} for V={tree_sizes[(depth, bf)]}")
                configs.append({
                    'depth': depth,
                    'branching_factor': bf,
                    'caption_mode': cm,
                    'loss_type': lt,
                    'temperature': temp,
                    'learnable_temperature': learn_temp,
                    'lambda_reg': lam,
                    'images_per_leaf': args.images_per_leaf,
                    'embedding_dim': embedding_dim,
                    'd_over_v': ratio,
                    'strict_data_mode': int(_is_strict_loss(lt)),
                    'strict_sigma': sigma,
                    'strict_p_over_d': p_over_d,
                    'strict_theta_mode': args.strict_theta_mode,
                    'strict_theta_values': args.strict_theta_values,
                    'strict_save_artifacts': args.strict_save_artifacts,
                    'strict_identity_encoder': args.strict_identity_encoder,
                    'strict_normalize': args.strict_normalize,
                    'save_checkpoints': args.save_checkpoints,
                    'loss_direction': args.loss_direction,
                    'prototype_shift_norm': float(shift),
                    'epochs': args.epochs,
                    'lr': lr,
                    'optimizer': opt,
                    'checkpoint_interval': args.checkpoint_interval,
                    'device': args.device,
                    'verbose': args.verbose,
                    'output_dir': args.output_dir,
                    'seed': args.seed,
                })

    if args.count_configs:
        print(len(configs))
        sys.exit(0)

    if args.config_index >= 0:
        if args.config_index >= len(configs):
            print(f"ERROR: --config_index={args.config_index} but only {len(configs)} configs (0-{len(configs)-1})")
            sys.exit(1)
        configs = [configs[args.config_index]]
        print(f"Running single config index {args.config_index}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Setup MLflow if requested
    use_mlflow = args.use_mlflow and MLFLOW_AVAILABLE
    if args.use_mlflow and not MLFLOW_AVAILABLE:
        print("WARNING: MLflow requested but not installed. Install with: pip install mlflow")
        print("Continuing without MLflow tracking...")
        use_mlflow = False

    tracking_uri = None
    if use_mlflow:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            mlruns_dir = os.path.join(os.path.abspath(args.output_dir), args.mlruns_subdir)
            tracking_uri = f"file://{mlruns_dir}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        print(f"MLflow tracking enabled:")
        print(f"  Experiment: {args.mlflow_experiment}")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")

    print(f"Total configurations: {len(configs)}")
    print(f"Sweep dimensions:")
    print(f"  Depths: {depths}")
    print(f"  Branching factors: {branching_factors}")
    print(f"  Caption modes: {caption_modes}")
    print(f"  Loss types: {loss_types}")
    print(f"  Temperatures (contrastive/strict): {temperatures}")
    print(f"  Learnable temperatures: {learnable_temps}")
    print(f"  Lambda regularization: {lambda_regs}")
    print(f"  Learning rates: {lrs}")
    print(f"  Optimizers: {optimizers}")
    if has_strict_loss:
        print(f"  strict_data_mode: auto-by-loss_type")
        print(f"  strict_sigma: {args.strict_sigma}")
        print(f"  strict_p_over_d: {strict_p_over_d_values}")
        print(f"  strict_theta_mode: {args.strict_theta_mode}")
        print(f"  strict_theta_values: {args.strict_theta_values}")
    if d_over_v_values:
        print(f"  d/V ratios: {d_over_v_values}")
    else:
        print(f"  Embedding dim: {args.embedding_dim}")

    all_results = []
    from tqdm import tqdm
    for i, config in enumerate(tqdm(configs, desc="Running sweep experiments")):
        dim_tag = f"dv{config['d_over_v']}" if config['d_over_v'] is not None else f"dim{config['embedding_dim']}"
        strict_tag = f"_pod{config['strict_p_over_d']}" if _is_strict_loss(config['loss_type']) else ""
        identity_tag = "_idenc" if config.get('strict_identity_encoder') else ""
        unnorm_tag = "_unnorm" if not config.get('strict_normalize', 1) else ""
        lt_tag = "_lt" if config.get('learnable_temperature') else ""
        config_str = (f"d{config['depth']}_b{config['branching_factor']}_{config['caption_mode']}_"
                      f"{config['loss_type']}_{dim_tag}{strict_tag}{identity_tag}{unnorm_tag}_t{config['temperature']}{lt_tag}_"
                      f"l{config['lambda_reg']}_lr{config['lr']}_{config['optimizer']}")
        config['run_tag'] = config_str

        if use_mlflow:
            with mlflow.start_run(run_name=config_str):
                mlflow.set_tag('slurm_job_id', os.environ.get('SLURM_ARRAY_JOB_ID', 'local'))
                # Log parameters (format small floats <= 0.1 as scientific notation)
                _fmt = lambda v: f"{v:.0e}" if isinstance(v, float) and abs(v) <= 0.1 else v
                mlflow.log_params({
                    'seed': config['seed'],
                    'depth': config['depth'],
                    'branching_factor': config['branching_factor'],
                    'caption_mode': config['caption_mode'],
                    'loss_type': config['loss_type'],
                    'loss_direction': config.get('loss_direction', 'both'),
                    'temperature': _fmt(config['temperature']),
                    'learnable_temperature': config.get('learnable_temperature', 0),
                    'lambda_reg': _fmt(config['lambda_reg']),
                    'embedding_dim': config['embedding_dim'],
                    'd_over_v': config['d_over_v'] if config['d_over_v'] is not None else 'fixed',
                    'strict_data_mode': config['strict_data_mode'],
                    'strict_sigma': _fmt(config['strict_sigma']),
                    'strict_p_over_d': config['strict_p_over_d'],
                    'strict_theta_mode': config['strict_theta_mode'],
                    'strict_theta_values': config['strict_theta_values'],
                    'strict_identity_encoder': config.get('strict_identity_encoder', 0),
                    'strict_normalize': config.get('strict_normalize', 1),
                    'prototype_shift_norm': _fmt(config.get('prototype_shift_norm', 0.0)),
                    'epochs': config['epochs'],
                    'lr': _fmt(config['lr']),
                    'optimizer': config['optimizer'],
                    'images_per_leaf': config['images_per_leaf'],
                })
                results = run_sweep_experiment(config, use_mlflow=True)
        else:
            results = run_sweep_experiment(config, use_mlflow=False)

        results.update(config)
        all_results.append(results)

        if config['device'] == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        if (i + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(args.output_dir, 'sweep_results_partial.csv'), index=False)
            print(f"    Saved intermediate results ({len(all_results)} experiments)")

    df = pd.DataFrame(all_results)
    csv_name = f'sweep_results_{args.config_index}.csv' if args.config_index >= 0 else 'sweep_results.csv'
    df.to_csv(os.path.join(args.output_dir, csv_name), index=False)
    print(f"\nSaved final results to {args.output_dir}/{csv_name}")

    if args.config_index <= 0:
        config_summary = {
            'depths': depths,
            'branching_factors': branching_factors,
            'caption_modes': caption_modes,
            'loss_types': loss_types,
            'temperatures': temperatures,
            'learnable_temperatures': learnable_temps,
            'lambda_regs': lambda_regs,
            'lrs': lrs,
            'optimizers': optimizers,
            'd_over_v': d_over_v_values,
            'strict_data_mode': 'auto_by_loss_type',
            'strict_sigma': args.strict_sigma,
            'strict_p_over_d': strict_p_over_d_values if has_strict_loss else [1.0],
            'strict_theta_mode': args.strict_theta_mode,
            'strict_theta_values': args.strict_theta_values,
            'strict_identity_encoder': args.strict_identity_encoder,
            'strict_normalize': args.strict_normalize,
            'fixed_params': {
                'images_per_leaf': args.images_per_leaf,
                **({} if d_over_v_values else {'embedding_dim': args.embedding_dim}),
                'epochs': args.epochs,
            },
            'total_experiments': len(configs),
        }
        with open(os.path.join(args.output_dir, 'sweep_config.json'), 'w') as f:
            json.dump(config_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"\nFinal loss statistics:")
    print(f"  Mean: {df['final_loss'].mean():.6f}")
    print(f"  Std: {df['final_loss'].std():.6f}")
    print(f"  Min: {df['final_loss'].min():.6f}")
    print(f"  Max: {df['final_loss'].max():.6f}")

    print(f"\nBranch separation violation rate (depth 0, A) by loss type:")
    if 'branch_sep_A_d0_viol_rate' in df.columns:
        for lt in loss_types:
            lt_df = df[df['loss_type'] == lt]
            if len(lt_df) > 0:
                vr = lt_df['branch_sep_A_d0_viol_rate'].mean() * 100
                print(f"  {lt}: {vr:.2f}%")

    print(f"\nResults saved to {args.output_dir}")

    if use_mlflow:
        print(f"MLflow tracking enabled:")
        print(f"  Experiment: {args.mlflow_experiment}")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")


def _branch_sep_to_row(branch_sep, permutation_id, tree_depth, branching_factor, embedding_dim, V, final_loss):
    """Extract final-epoch branch separation metrics into a flat dict."""
    row = {
        'permutation_id': permutation_id,
        'depth': tree_depth,
        'branching_factor': branching_factor,
        'embedding_dim': embedding_dim,
        'V': V,
        'final_loss': final_loss,
    }
    for emb_type in ['A', 'W_mean', 'H_mean']:
        for d in branch_sep[emb_type]:
            row[f'viol_rate_{emb_type}_d{d}'] = branch_sep[emb_type][d]['violation_rate'][-1]
            row[f'mean_rel_viol_{emb_type}_d{d}'] = branch_sep[emb_type][d]['mean_relative_violation'][-1]
    return row


def run_null_model(args):
    """Run null model permutation test via cross-depth embedding permutation.

    For each config: train once, measure baseline branch separation, then
    generate null distribution by randomly permuting W rows across all depths
    and re-measuring branch separation (no retraining needed).
    """
    os.makedirs(args.output_dir, exist_ok=True)

    depths = parse_csv_list(args.depth, int)
    d_over_v_values = parse_csv_list(args.d_over_v, float) if args.d_over_v else []

    use_mlflow = args.use_mlflow and MLFLOW_AVAILABLE
    if args.use_mlflow and not MLFLOW_AVAILABLE:
        print("WARNING: MLflow not installed, continuing without tracking.")
        use_mlflow = False
    if use_mlflow:
        db_path = os.path.join(os.path.abspath(args.output_dir), "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(args.mlflow_experiment)

    configs = []
    for depth in depths:
        tree = build_balanced_tree(depth, args.branching_factor)
        V = tree.num_nodes
        if d_over_v_values:
            for dv in d_over_v_values:
                configs.append((depth, tree, V, int(round(dv * V)), dv))
        else:
            configs.append((depth, tree, V, args.embedding_dim, None))

    print(f"Null model test (cross-depth permutation): depths={depths}, bf={args.branching_factor}")
    if d_over_v_values:
        print(f"  d/V ratios: {d_over_v_values}")
    else:
        print(f"  embedding_dim: {args.embedding_dim}")
    print(f"  n_permutations={args.n_permutations}, seed={args.seed}")
    print(f"  Output: {args.output_dir}")
    print(f"  Total configs: {len(configs)}")

    all_results = []

    for ci, (depth, tree, V, embedding_dim, dv) in enumerate(configs):
        n_permutations = args.n_permutations
        dv_tag = f"dv{dv}" if dv is not None else f"dim{embedding_dim}"
        config_label = f"d{depth}_{dv_tag}"
        print(f"\n{'='*70}")
        print(f"Config {ci+1}/{len(configs)}: depth={depth}, V={V}, d={embedding_dim}, n_perms={n_permutations}")
        print(f"{'='*70}")

        # Train once
        num_leaves = len(tree.get_leaves())
        num_images = num_leaves * args.images_per_leaf
        leaf_assignments = assign_images_per_leaf(tree, args.images_per_leaf)
        S = build_support_matrix(tree, leaf_assignments)
        P = build_caption_distribution(tree, leaf_assignments, mode='uniform')
        pi = compute_image_weights(leaf_assignments)

        print(f"\n  Training model (V={V}, d={embedding_dim}, m={num_images})...")
        _, checkpoints = train_ufm(
            V=V, d=embedding_dim, m=num_images, P=P, pi=pi,
            lambda_reg=0.0, epochs=args.epochs, lr=args.lr,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device, verbose=False,
            loss_type='contrastive_strict', temperature=args.temperature,
            S=S, mlflow_log=False, optimizer_type=args.optimizer,
        )
        final_ckpt = checkpoints[-1]
        final_loss = final_ckpt.loss
        print(f"  Training done. Final loss: {final_loss:.4f}")

        # Baseline: measure branch separation on the actual trained embeddings
        print(f"\n  [0/{n_permutations}] Baseline (original embeddings)...")
        branch_sep = track_branch_separation_over_training([final_ckpt], tree, leaf_assignments)
        row = _branch_sep_to_row(branch_sep, 0, depth, args.branching_factor, embedding_dim, V, final_loss)
        row['d_over_v'] = dv

        if use_mlflow:
            with mlflow.start_run(run_name=f"baseline_{config_label}"):
                mlflow.log_params({'permutation_id': 0, 'depth': depth,
                                   'branching_factor': args.branching_factor,
                                   'embedding_dim': embedding_dim, 'd_over_v': dv or 'fixed',
                                   'is_baseline': True, 'n_permutations': n_permutations})
                mlflow.log_metric("final_loss", final_loss)
                for d in branch_sep['A']:
                    mlflow.log_metric(f"viol_rate/A/d{d}",
                                      branch_sep['A'][d]['violation_rate'][-1])

        all_results.append(row)
        print(f"    A viol_rate_d0: {row.get('viol_rate_A_d0', 'N/A'):.4f}")

        # Null permutations: permute W rows across all depths, re-measure
        W_orig = final_ckpt.W
        H_orig = final_ckpt.H
        for i in range(1, n_permutations + 1):
            rng = np.random.default_rng(args.seed + i)
            sigma = rng.permutation(V)
            perm_ckpt = Checkpoint(epoch=final_ckpt.epoch, W=W_orig[sigma], H=H_orig, loss=final_loss)
            branch_sep = track_branch_separation_over_training([perm_ckpt], tree, leaf_assignments)
            row = _branch_sep_to_row(branch_sep, i, depth, args.branching_factor, embedding_dim, V, final_loss)
            row['d_over_v'] = dv

            if use_mlflow:
                with mlflow.start_run(run_name=f"perm_{i}_{config_label}"):
                    mlflow.log_params({'permutation_id': i, 'depth': depth,
                                       'branching_factor': args.branching_factor,
                                       'embedding_dim': embedding_dim, 'd_over_v': dv or 'fixed',
                                       'is_baseline': False, 'seed': args.seed + i})
                    for d in branch_sep['A']:
                        mlflow.log_metric(f"viol_rate/A/d{d}",
                                          branch_sep['A'][d]['violation_rate'][-1])

            all_results.append(row)

            if i % 10 == 0:
                print(f"  [{i}/{n_permutations}] permutations done")
                pd.DataFrame(all_results).to_csv(
                    os.path.join(args.output_dir, 'null_model_results_partial.csv'), index=False)

        print(f"  All {n_permutations} permutations done.")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'null_model_results.csv'), index=False)

    # Summary per config
    print("\n" + "=" * 70)
    print("NULL MODEL SUMMARY")
    print("=" * 70)

    for depth, _, V, embedding_dim, dv in configs:
        cfg_df = df[(df['depth'] == depth) & (df['embedding_dim'] == embedding_dim)]
        baseline = cfg_df[cfg_df['permutation_id'] == 0]
        null = cfg_df[cfg_df['permutation_id'] > 0]
        dv_str = f", d/V={dv}" if dv is not None else ""
        print(f"\n  depth={depth}, V={V}, d={embedding_dim}{dv_str}:")

        viol_cols = [c for c in cfg_df.columns if c.startswith('viol_rate_A_')]
        for col in sorted(viol_cols):
            depth_str = col.split('_d')[-1]
            bl_val = baseline[col].values[0]
            if pd.isna(bl_val):
                continue
            null_vals = null[col].dropna()
            if null_vals.empty:
                continue
            null_mean = null_vals.mean()
            null_std = null_vals.std()
            p_val = (null_vals <= bl_val).mean()
            print(f"    Depth {depth_str}: baseline={bl_val:.4f}, null={null_mean:.4f}+/-{null_std:.4f}, p={p_val:.4f}")

    print(f"\nResults saved to {args.output_dir}/null_model_results.csv")


def recompute_metrics(args):
    """Recompute metrics from saved checkpoints and update existing MLflow runs."""
    if not MLFLOW_AVAILABLE:
        print("ERROR: mlflow is required. Install with: pip install mlflow")
        sys.exit(1)

    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: No checkpoints directory found at {ckpt_dir}")
        sys.exit(1)

    db_path = os.path.join(os.path.abspath(args.output_dir), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(args.mlflow_experiment)
    experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
    if experiment is None:
        print(f"ERROR: Experiment '{args.mlflow_experiment}' not found in {db_path}")
        sys.exit(1)

    # Get runs to recompute
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format='list')
    if args.run_ids:
        target_ids = set(args.run_ids.split(','))
        runs = [r for r in runs if r.info.run_id in target_ids]
    print(f"Found {len(runs)} runs to recompute in experiment '{args.mlflow_experiment}'")

    compute_branch_sep = args.metrics in ('branch_sep', 'both')
    compute_all = args.metrics in ('all_metrics', 'both')

    from tqdm import tqdm
    for run in tqdm(runs, desc="Recomputing metrics"):
        run_id = run.info.run_id

        # Support both new directory format and old stacked .npz format
        ckpt_subdir = os.path.join(ckpt_dir, run_id)
        ckpt_path = os.path.join(ckpt_dir, f'{run_id}.npz')
        if os.path.isdir(ckpt_subdir):
            checkpoints = CheckpointDir(ckpt_subdir)
        elif os.path.exists(ckpt_path):
            checkpoints = load_checkpoints(ckpt_path)
        else:
            print(f"  SKIP {run_id}: no checkpoint file or directory")
            continue

        # Reconstruct tree and leaf_assignments from logged params
        params = run.data.params
        depth = int(params['depth'])
        bf = int(params['branching_factor'])
        images_per_leaf = int(params['images_per_leaf'])
        tree = build_balanced_tree(depth, bf)
        leaf_assignments = assign_images_per_leaf(tree, images_per_leaf)

        # Recompute selected metrics
        branch_sep = None
        all_metrics = None
        if compute_branch_sep and compute_all:
            combined = track_all_metrics_single_pass(checkpoints, tree, leaf_assignments)
            branch_sep = combined['branch_sep']
            all_metrics = combined['all_metrics']
        elif compute_branch_sep:
            branch_sep = track_branch_separation_over_training(checkpoints, tree, leaf_assignments)
        elif compute_all:
            all_metrics = track_all_metrics_over_training(checkpoints, tree, leaf_assignments)

        # Reopen the existing run and overwrite metrics
        # Use epochs metadata to avoid loading full W/H for the logging loop
        n_ckpts = len(checkpoints)
        epochs_list = checkpoints.epochs if isinstance(checkpoints, CheckpointDir) else [c.epoch for c in checkpoints]
        with mlflow.start_run(run_id=run_id):
            for i in range(n_ckpts):
                epoch = epochs_list[i]
                if branch_sep is not None:
                    for emb_type in ['A', 'W_mean', 'H_mean']:
                        for d in branch_sep[emb_type]:
                            mlflow.log_metric(f"viol_rate/{emb_type}/d{d}",
                                              branch_sep[emb_type][d]['violation_rate'][i], step=epoch)
                            mlflow.log_metric(f"mean_rel_viol/{emb_type}/d{d}",
                                              branch_sep[emb_type][d]['mean_relative_violation'][i], step=epoch)
                            mlflow.log_metric(f"exc_angle/{emb_type}/d{d}",
                                              branch_sep[emb_type][d]['mean_exceeding_angle'][i], step=epoch)
                            mlflow.log_metric(f"mean_theta/{emb_type}/d{d}",
                                              branch_sep[emb_type][d]['mean_theta_i'][i], step=epoch)
                if all_metrics is not None:
                    for emb_type in ['A', 'W_mean', 'H_mean']:
                        for d in all_metrics['orthogonality'][emb_type]:
                            mlflow.log_metric(f"ortho/{emb_type}/d{d}",
                                              all_metrics['orthogonality'][emb_type][d][i], step=epoch)

    print(f"\nDone. Recomputed metrics for {len(runs)} runs.")
    print(f"View results with: mlflow ui --backend-store-uri sqlite:///{db_path} --host 0.0.0.0")
    import socket
    print(f"SSH tunnel: ssh -L 5000:{socket.gethostname()}:5000 nghianhh@<login-node>")


def main():
    args = parse_args()
    if args.command == 'sweep':
        run_sweep(args)
    elif args.command == 'null_model':
        run_null_model(args)
    elif args.command == 'recompute_metrics':
        recompute_metrics(args)
    else:
        run_single_experiment(args)


if __name__ == '__main__':
    main()

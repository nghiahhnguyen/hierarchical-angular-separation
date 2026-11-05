"""Unconstrained Features Model (UFM) for hierarchical concept learning."""

import os
import glob as glob_module
from functools import partial
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Checkpoint:
    """Training checkpoint with model parameters."""
    epoch: int
    W: np.ndarray  # (V, d)
    H: np.ndarray  # (d, m)
    loss: float
    temperature: Optional[float] = None


def save_checkpoints(checkpoints: list[Checkpoint], path: str):
    """Save list of checkpoints to a single .npz file."""
    kwargs = dict(
        epochs=np.array([c.epoch for c in checkpoints]),
        losses=np.array([c.loss for c in checkpoints]),
        W=np.stack([c.W for c in checkpoints]),  # (n_ckpt, V, d)
        H=np.stack([c.H for c in checkpoints]),  # (n_ckpt, d, m)
    )
    if any(c.temperature is not None for c in checkpoints):
        kwargs['temperatures'] = np.array([c.temperature for c in checkpoints])
    np.savez(path, **kwargs)


def load_checkpoints(path: str) -> list[Checkpoint]:
    """Load checkpoints from a .npz file."""
    data = np.load(path)
    temps = data['temperatures'] if 'temperatures' in data else [None] * len(data['epochs'])
    return [Checkpoint(epoch=int(e), W=W, H=H, loss=float(l), temperature=float(t) if t is not None else None)
            for e, W, H, l, t in zip(data['epochs'], data['W'], data['H'], data['losses'], temps)]


def save_checkpoint_individual(ckpt: Checkpoint, path: str):
    """Save a single checkpoint to a .npz file."""
    kwargs = dict(epoch=ckpt.epoch, W=ckpt.W, H=ckpt.H, loss=ckpt.loss)
    if ckpt.temperature is not None:
        kwargs['temperature'] = ckpt.temperature
    np.savez(path, **kwargs)


def load_checkpoint_individual(path: str) -> Checkpoint:
    """Load a single checkpoint from a .npz file."""
    data = np.load(path)
    temp = float(data['temperature']) if 'temperature' in data else None
    return Checkpoint(epoch=int(data['epoch']), W=data['W'], H=data['H'], loss=float(data['loss']), temperature=temp)


class CheckpointDir:
    """Lazy-loading container for checkpoints stored as individual files in a directory.

    Drop-in replacement for list[Checkpoint] -- supports __iter__, __len__, __getitem__.
    Loads one checkpoint at a time from disk during iteration to keep memory O(V*m).
    """

    def __init__(self, directory: str):
        self.directory = directory
        # Discover and sort checkpoint files by epoch number
        files = glob_module.glob(os.path.join(directory, '*.npz'))
        # Parse epoch from filename (e.g., "100.npz" -> 100)
        self._files = sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        # Load lightweight metadata (epochs/losses) without loading W/H
        self._epochs = []
        self._losses = []
        self._temperatures = []
        for f in self._files:
            data = np.load(f, mmap_mode='r')
            self._epochs.append(int(data['epoch']))
            self._losses.append(float(data['loss']))
            self._temperatures.append(float(data['temperature']) if 'temperature' in data else None)

    @property
    def epochs(self) -> list[int]:
        return self._epochs

    @property
    def losses(self) -> list[float]:
        return self._losses

    @property
    def temperatures(self) -> List[Optional[float]]:
        return self._temperatures

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [load_checkpoint_individual(self._files[i]) for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        return load_checkpoint_individual(self._files[idx])

    def __iter__(self):
        for f in self._files:
            yield load_checkpoint_individual(f)


class UFMModel(nn.Module):
    """
    Unconstrained Features Model.

    Learns embeddings W (concept embeddings) and H (image embeddings) directly.
    Logits = W @ H, trained with softmax cross-entropy.
    """

    def __init__(self, V: int, d: int, m: int,
                 learnable_temperature: bool = False, init_temperature: float = 0.07):
        """
        Args:
            V: Number of concepts (synsets)
            d: Embedding dimension
            m: Number of images
            learnable_temperature: If True, temperature is a learnable parameter in log-space
            init_temperature: Initial temperature value (used when learnable_temperature=True)
        """
        super().__init__()
        self.V, self.d, self.m = V, d, m
        # Xavier initialization
        self.W = nn.Parameter(torch.randn(V, d) / np.sqrt(d))
        self.H = nn.Parameter(torch.randn(d, m) / np.sqrt(d))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        else:
            self.log_temperature = None

    def get_temperature(self):
        if self.log_temperature is not None:
            return self.log_temperature.exp()
        return None

    def forward(self) -> torch.Tensor:
        """Compute logits = W @ H, shape (V, m)."""
        return self.W @ self.H

    def get_numpy_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return W and H as numpy arrays (copies to prevent aliasing)."""
        return self.W.detach().cpu().numpy().copy(), self.H.detach().cpu().numpy().copy()


class UFMStrictModel(nn.Module):
    """
    Strict contrastive model: learn text embeddings W and image encoder W_img.

    Image embeddings are parameterized as H = W_img @ X (not free parameters).
    """

    def __init__(self, V: int, d: int, X: torch.Tensor,
                 learnable_temperature: bool = False, init_temperature: float = 0.07,
                 freeze_w_img: bool = False):
        """
        Args:
            V: Number of concepts (synsets)
            d: Embedding dimension
            X: Observed image features of shape (p, m)
            learnable_temperature: If True, temperature is a learnable parameter in log-space
            init_temperature: Initial temperature value (used when learnable_temperature=True)
            freeze_w_img: If True, fix W_img to identity (requires p == d)
        """
        super().__init__()
        if X.ndim != 2:
            raise ValueError("X must have shape (p, m)")
        p, m = X.shape
        self.V, self.d, self.p, self.m = V, d, p, m
        self.W = nn.Parameter(torch.randn(V, d) / np.sqrt(d))
        if freeze_w_img:
            if p != d:
                raise ValueError(f"freeze_w_img requires p == d, got p={p}, d={d}")
            self.register_buffer("W_img", torch.eye(d))
        else:
            self.W_img = nn.Parameter(torch.randn(d, p) / np.sqrt(p))
        self.register_buffer("X", X)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        else:
            self.log_temperature = None

    def get_temperature(self):
        if self.log_temperature is not None:
            return self.log_temperature.exp()
        return None

    def get_H(self) -> torch.Tensor:
        """Compute image embeddings H = W_img @ X, shape (d, m)."""
        return self.W_img @ self.X

    def forward(self) -> torch.Tensor:
        """Compute logits = W @ (W_img @ X), shape (V, m)."""
        return self.W @ self.get_H()

    def get_numpy_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return W and H as numpy arrays (copies) for checkpoint compatibility."""
        H = self.get_H()
        return self.W.detach().cpu().numpy().copy(), H.detach().cpu().numpy().copy()


def compute_loss(
    model: UFMModel,
    P: torch.Tensor,
    pi: torch.Tensor,
    lambda_reg: float
) -> torch.Tensor:
    """
    Compute softmax cross-entropy loss with L2 regularization.

    Loss = sum_j pi_j * (- sum_z p_j[z] * log softmax(W @ h_j)[z])
           + lambda/2 * (||W||_F^2 + ||H||_F^2)

    Args:
        model: UFM model
        P: Caption distribution matrix (V, m)
        pi: Image weights (m,)
        lambda_reg: Regularization strength

    Returns:
        Scalar loss
    """
    logits = model()  # (V, m)
    log_probs = torch.log_softmax(logits, dim=0)  # (V, m)

    # Cross-entropy per image: -sum_z p_j[z] * log_softmax[z]
    ce_per_image = -(P * log_probs).sum(dim=0)  # (m,)
    ce_loss = (pi * ce_per_image).sum()

    # L2 regularization
    reg_loss = lambda_reg / 2 * (model.W.pow(2).sum() + model.H.pow(2).sum())

    return ce_loss + reg_loss


def compute_contrastive_loss(
    model: UFMModel,
    S: torch.Tensor,
    temperature: float,
    lambda_reg: float
) -> torch.Tensor:
    """
    Compute CLIP-style InfoNCE contrastive loss with L2 regularization.
    NOTE: This uses multiple positives per image (all ancestors). See
    compute_contrastive_loss_strict for standard CLIP one-to-one pairing.

    Uses L2-normalized embeddings (cosine similarity) and binary support matrix S
    to define positive pairs: (concept z, image j) where S[z,j] = 1.

    Loss = 0.5 * (L_i2t + L_t2i)

    L_i2t = (1/m) * sum_j [ -log( sum_{z: S[z,j]=1} exp(sim_zj) / sum_z exp(sim_zj) ) ]
    L_t2i = (1/V) * sum_z [ -log( sum_{j: S[z,j]=1} exp(sim_zj) / sum_j exp(sim_zj) ) ]

    Args:
        model: UFM model
        S: Binary support matrix (V, m) where S[z,j]=1 iff concept z is ancestor of image j's leaf
        temperature: Temperature scaling for softmax
        lambda_reg: L2 regularization strength

    Returns:
        Scalar loss
    """
    V, m = S.shape
    tau = model.get_temperature() if model.log_temperature is not None else temperature

    # L2-normalize embeddings
    W_norm = model.W / (model.W.norm(dim=1, keepdim=True) + 1e-8)  # (V, d)
    H_norm = model.H / (model.H.norm(dim=0, keepdim=True) + 1e-8)  # (d, m)

    # Compute similarity matrix (cosine similarity scaled by temperature)
    sim = (W_norm @ H_norm) / tau  # (V, m)

    # Precompute boolean mask and negative infinity tensor
    S_bool = S.bool()
    neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

    # Image-to-text loss: for each image j, the positive set is {z: S[z,j]=1}
    log_sum_exp_all = torch.logsumexp(sim, dim=0)  # (m,)
    masked_sim = torch.where(S_bool, sim, neg_inf)
    log_sum_exp_pos_i2t = torch.logsumexp(masked_sim, dim=0)  # (m,)
    L_i2t = -(log_sum_exp_pos_i2t - log_sum_exp_all).mean()

    # Text-to-image loss: for each concept z, the positive set is {j: S[z,j]=1}
    log_sum_exp_all_t2i = torch.logsumexp(sim, dim=1)  # (V,)
    log_sum_exp_pos_t2i = torch.logsumexp(masked_sim, dim=1)  # (V,)
    L_t2i = -(log_sum_exp_pos_t2i - log_sum_exp_all_t2i).mean()

    contrastive_loss = 0.5 * (L_i2t + L_t2i)

    # L2 regularization
    reg_loss = lambda_reg / 2 * (model.W.pow(2).sum() + model.H.pow(2).sum())

    return contrastive_loss + reg_loss


def compute_contrastive_loss_strict(
    model: UFMModel,
    concept_indices: torch.Tensor,
    temperature: float,
    lambda_reg: float,
    loss_direction: str = 'both',
) -> torch.Tensor:
    """
    Compute strict CLIP-style InfoNCE loss with exactly one positive per image.

    Each image j is paired with exactly one concept (pre-sampled and fixed).
    This matches the original CLIP formulation where each image has one caption.

    Args:
        model: UFM model
        concept_indices: Pre-sampled concept IDs per image (m,)
        temperature: Temperature scaling for softmax
        lambda_reg: L2 regularization strength

    Returns:
        Scalar loss
    """
    m = concept_indices.shape[0]
    tau = model.get_temperature() if model.log_temperature is not None else temperature

    # L2-normalize embeddings
    W_norm = model.W / (model.W.norm(dim=1, keepdim=True) + 1e-8)  # (V, d)
    H_norm = model.H / (model.H.norm(dim=0, keepdim=True) + 1e-8)  # (d, m)

    # Compute full similarity matrix
    sim = (W_norm @ H_norm) / tau  # (V, m)

    # Get positive similarities: sim[c_j, j] for each image j
    image_indices = torch.arange(m, device=concept_indices.device)
    pos_sim = sim[concept_indices, image_indices]  # (m,)

    # Image-to-text: for each image j, predict its concept c_j among all V concepts
    log_sum_exp_i2t = torch.logsumexp(sim, dim=0)  # (m,)
    L_i2t = -(pos_sim - log_sum_exp_i2t).mean()

    if loss_direction == 'i2t':
        contrastive_loss = L_i2t
    elif loss_direction == 't2i':
        # Text-to-image: for each sampled concept c_j, predict image j among all m images
        log_sum_exp_t2i = torch.logsumexp(sim, dim=1)[concept_indices]  # (m,)
        L_t2i = -(pos_sim - log_sum_exp_t2i).mean()
        contrastive_loss = L_t2i
    else:
        # Reuse sim (V, m) instead of materializing (m, m) sampled_concept_sims
        log_sum_exp_t2i = torch.logsumexp(sim, dim=1)[concept_indices]  # (m,)
        L_t2i = -(pos_sim - log_sum_exp_t2i).mean()
        contrastive_loss = 0.5 * (L_i2t + L_t2i)

    # L2 regularization
    reg_loss = lambda_reg / 2 * (model.W.pow(2).sum() + model.H.pow(2).sum())

    return contrastive_loss + reg_loss


def compute_contrastive_loss_strict_xwz(
    model: UFMStrictModel,
    concept_indices: torch.Tensor,
    temperature: float,
    lambda_reg: float,
    normalize: bool = True,
    loss_direction: str = 'both',
) -> torch.Tensor:
    """
    Compute strict CLIP-style InfoNCE where H is parameterized by W_img @ X.

    Regularization is applied to W (text embeddings) and W_img (image encoder).
    When normalize=False, uses raw (unnormalized) embeddings matching the analytic
    model formulation (eq. 3 in the paper).
    """
    m = concept_indices.shape[0]
    tau = model.get_temperature() if model.log_temperature is not None else temperature
    H = model.get_H()
    if H.shape[1] != m:
        raise ValueError("concept_indices length must match number of image columns in X")

    if normalize:
        W_emb = model.W / (model.W.norm(dim=1, keepdim=True) + 1e-8)  # (V, d)
        H_emb = H / (H.norm(dim=0, keepdim=True) + 1e-8)  # (d, m)
    else:
        W_emb = model.W
        H_emb = H
    sim = (W_emb @ H_emb) / tau  # (V, m)

    image_indices = torch.arange(m, device=concept_indices.device)
    pos_sim = sim[concept_indices, image_indices]  # (m,)

    log_sum_exp_i2t = torch.logsumexp(sim, dim=0)  # (m,)
    L_i2t = -(pos_sim - log_sum_exp_i2t).mean()

    if loss_direction == 'i2t':
        contrastive_loss = L_i2t
    elif loss_direction == 't2i':
        log_sum_exp_t2i = torch.logsumexp(sim, dim=1)[concept_indices]  # (m,)
        L_t2i = -(pos_sim - log_sum_exp_t2i).mean()
        contrastive_loss = L_t2i
    else:
        log_sum_exp_t2i = torch.logsumexp(sim, dim=1)[concept_indices]  # (m,)
        L_t2i = -(pos_sim - log_sum_exp_t2i).mean()
        contrastive_loss = 0.5 * (L_i2t + L_t2i)

    reg_loss = lambda_reg / 2 * model.W.pow(2).sum()
    if isinstance(model.W_img, nn.Parameter):
        reg_loss = reg_loss + lambda_reg / 2 * model.W_img.pow(2).sum()
    return contrastive_loss + reg_loss


def train_ufm(
    V: int,
    d: int,
    m: int,
    P: np.ndarray,
    pi: np.ndarray,
    lambda_reg: float,
    epochs: int,
    lr: float,
    checkpoint_interval: int = 100,
    device: str = 'cpu',
    verbose: bool = True,
    loss_type: str = 'softmax_ce',
    temperature: float = 0.07,
    S: np.ndarray = None,
    mlflow_log: bool = False,
    optimizer_type: str = 'adam',
    checkpoint_dir: str = None,
    learnable_temperature: bool = False,
    loss_direction: str = 'both',
) -> tuple[UFMModel, 'list[Checkpoint] | CheckpointDir']:
    """
    Train UFM model.

    Args:
        V: Number of concepts
        d: Embedding dimension
        m: Number of images
        P: Caption distribution (V, m) - required for softmax_ce and contrastive_strict
        pi: Image weights (m,) - required for softmax_ce loss
        lambda_reg: Regularization strength
        epochs: Number of training epochs
        lr: Learning rate
        checkpoint_interval: Save checkpoint every N epochs
        device: 'cpu' or 'cuda'
        verbose: Print progress
        loss_type: 'softmax_ce', 'contrastive', or 'contrastive_strict'
            - softmax_ce: Standard cross-entropy with soft labels from P
            - contrastive: Multi-positive InfoNCE using support matrix S
            - contrastive_strict: Standard CLIP one-to-one pairing, samples from P
        temperature: Temperature for contrastive losses (default 0.07)
        S: Support matrix (V, m) - required for contrastive loss
        mlflow_log: Whether to log metrics to MLflow (default False)
        optimizer_type: 'adam', 'adamw', or 'sgd' (default 'adam')
        checkpoint_dir: If set, save each checkpoint individually to this directory
            and return a CheckpointDir (lazy-loading) instead of a list

    Returns:
        Trained model and checkpoints (list[Checkpoint] or CheckpointDir)
    """
    if loss_type == 'contrastive' and S is None:
        raise ValueError("Support matrix S is required for contrastive loss")
    if loss_type == 'contrastive_strict' and P is None:
        raise ValueError("Caption distribution P is required for contrastive_strict loss")
    if learnable_temperature and loss_type == 'softmax_ce':
        raise ValueError("learnable_temperature is only supported for contrastive loss types")

    if mlflow_log:
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow is required for logging. Install with: pip install mlflow")

    model = UFMModel(V, d, m, learnable_temperature=learnable_temperature, init_temperature=temperature).to(device)
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    # Select loss function and prepare tensors
    if loss_type == 'softmax_ce':
        P_t = torch.tensor(P, dtype=torch.float32, device=device)
        pi_t = torch.tensor(pi, dtype=torch.float32, device=device)
        loss_fn = partial(compute_loss, P=P_t, pi=pi_t, lambda_reg=lambda_reg)
    elif loss_type == 'contrastive':
        S_t = torch.tensor(S, dtype=torch.float32, device=device)
        loss_fn = partial(compute_contrastive_loss, S=S_t, temperature=temperature, lambda_reg=lambda_reg)
    elif loss_type == 'contrastive_strict':
        P_t = torch.tensor(P, dtype=torch.float32, device=device)
        concept_indices = torch.multinomial(P_t.T, num_samples=1).squeeze(-1)  # (m,)
        loss_fn = partial(compute_contrastive_loss_strict, concept_indices=concept_indices, temperature=temperature, lambda_reg=lambda_reg, loss_direction=loss_direction)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    use_cuda = str(device).startswith('cuda')
    if use_cuda:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        loss_fn = torch.compile(loss_fn)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = []
    # Save initialization checkpoint (before any gradient steps)
    W_np, H_np = model.get_numpy_params()
    with torch.no_grad():
        init_loss = loss_fn(model).item()
    temp_val = model.get_temperature().item() if model.log_temperature is not None else None
    ckpt = Checkpoint(epoch=0, W=W_np, H=H_np, loss=init_loss, temperature=temp_val)
    if checkpoint_dir:
        save_checkpoint_individual(ckpt, os.path.join(checkpoint_dir, '0.npz'))
        del ckpt
    else:
        checkpoints.append(ckpt)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            W_np, H_np = model.get_numpy_params()
            temp_val = model.get_temperature().item() if model.log_temperature is not None else None
            ckpt = Checkpoint(epoch=epoch, W=W_np, H=H_np, loss=loss.item(), temperature=temp_val)
            if checkpoint_dir:
                save_checkpoint_individual(ckpt, os.path.join(checkpoint_dir, f'{epoch}.npz'))
                del ckpt
            else:
                checkpoints.append(ckpt)
            if verbose:
                temp_str = f", Temp: {temp_val:.6f}" if temp_val is not None else ""
                print(f"Epoch {epoch:5d}, Loss: {loss.item():.6f}{temp_str}")
            if mlflow_log:
                mlflow.log_metric("loss", loss.item(), step=epoch)
                if temp_val is not None:
                    mlflow.log_metric("temperature", temp_val, step=epoch)
                if lambda_reg > 0:
                    with torch.no_grad():
                        reg_A = lambda_reg / 2 * model.W.pow(2).sum()
                        reg_H = lambda_reg / 2 * model.H.pow(2).sum()
                    mlflow.log_metric("reg_loss_A", reg_A.item(), step=epoch)
                    mlflow.log_metric("reg_loss_H", reg_H.item(), step=epoch)
                    mlflow.log_metric("reg_loss", (reg_A + reg_H).item(), step=epoch)
                    mlflow.log_metric("main_loss", loss.item() - (reg_A + reg_H).item(), step=epoch)

    if checkpoint_dir:
        return model, CheckpointDir(checkpoint_dir)
    return model, checkpoints


def train_ufm_strict(
    V: int,
    d: int,
    X: np.ndarray,
    concept_indices: np.ndarray,
    lambda_reg: float,
    epochs: int,
    lr: float,
    checkpoint_interval: int = 100,
    device: str = 'cpu',
    verbose: bool = True,
    temperature: float = 0.07,
    mlflow_log: bool = False,
    optimizer_type: str = 'adam',
    checkpoint_dir: str = None,
    learnable_temperature: bool = False,
    freeze_w_img: bool = False,
    normalize: bool = True,
    loss_direction: str = 'both',
) -> tuple[UFMStrictModel, 'list[Checkpoint] | CheckpointDir']:
    """
    Train strict contrastive model with parameterized image embeddings H = W_img @ X.
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (p, m)")
    if concept_indices.ndim != 1:
        raise ValueError("concept_indices must have shape (m,)")
    m = X.shape[1]
    if concept_indices.shape[0] != m:
        raise ValueError("concept_indices length must match number of image columns in X")
    if np.any(concept_indices < 0) or np.any(concept_indices >= V):
        raise ValueError("concept_indices must be in [0, V)")

    if mlflow_log:
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow is required for logging. Install with: pip install mlflow")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    concept_t = torch.tensor(concept_indices, dtype=torch.long, device=device)
    model = UFMStrictModel(V, d, X_t, learnable_temperature=learnable_temperature,
                           init_temperature=temperature, freeze_w_img=freeze_w_img).to(device)
    use_lbfgs = optimizer_type == 'lbfgs'
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif use_lbfgs:
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=lr,
            history_size=10, max_iter=20, line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    loss_fn = partial(
        compute_contrastive_loss_strict_xwz,
        concept_indices=concept_t,
        temperature=temperature,
        lambda_reg=lambda_reg,
        normalize=normalize,
        loss_direction=loss_direction,
    )

    use_cuda = str(device).startswith('cuda')
    if use_cuda:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        loss_fn = torch.compile(loss_fn)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = []
    W_np, H_np = model.get_numpy_params()
    with torch.no_grad():
        init_loss = loss_fn(model).item()
    temp_val = model.get_temperature().item() if model.log_temperature is not None else None
    ckpt = Checkpoint(epoch=0, W=W_np, H=H_np, loss=init_loss, temperature=temp_val)
    if checkpoint_dir:
        save_checkpoint_individual(ckpt, os.path.join(checkpoint_dir, '0.npz'))
        del ckpt
    else:
        checkpoints.append(ckpt)

    for epoch in range(1, epochs + 1):
        if use_lbfgs:
            def closure():
                optimizer.zero_grad()
                l = loss_fn(model)
                l.backward()
                return l
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            optimizer.step()

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            W_np, H_np = model.get_numpy_params()
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            temp_val = model.get_temperature().item() if model.log_temperature is not None else None
            ckpt = Checkpoint(epoch=epoch, W=W_np, H=H_np, loss=loss_val, temperature=temp_val)
            if checkpoint_dir:
                save_checkpoint_individual(ckpt, os.path.join(checkpoint_dir, f'{epoch}.npz'))
                del ckpt
            else:
                checkpoints.append(ckpt)
            if verbose:
                temp_str = f", Temp: {temp_val:.6f}" if temp_val is not None else ""
                print(f"Epoch {epoch:5d}, Loss: {loss_val:.6f}{temp_str}")
            if mlflow_log:
                mlflow.log_metric("loss", loss_val, step=epoch)
                if temp_val is not None:
                    mlflow.log_metric("temperature", temp_val, step=epoch)
                if lambda_reg > 0:
                    with torch.no_grad():
                        reg_A = lambda_reg / 2 * model.W.pow(2).sum()
                    mlflow.log_metric("reg_loss_A", reg_A.item(), step=epoch)
                    reg_total = reg_A
                    if isinstance(model.W_img, nn.Parameter):
                        with torch.no_grad():
                            reg_Wimg = lambda_reg / 2 * model.W_img.pow(2).sum()
                        mlflow.log_metric("reg_loss_Wimg", reg_Wimg.item(), step=epoch)
                        reg_total = reg_total + reg_Wimg
                    mlflow.log_metric("reg_loss", reg_total.item(), step=epoch)
                    mlflow.log_metric("main_loss", loss_val - reg_total.item(), step=epoch)

    if checkpoint_dir:
        return model, CheckpointDir(checkpoint_dir)
    return model, checkpoints

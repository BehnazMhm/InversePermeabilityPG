"""
Inverse Diffusion Model for Material Parameter Generation
=========================================================

Goal:
-----

We have a dataset where each row contains:
  - physical material parameters (many columns)
  - permeabilities: Kxx, Kyy, Kzz

We want to learn two things:

1) Forward model (simple MLP):
   parameters → permeability (log10(Kxx,Kyy,Kzz) in standardized space)

2) Inverse diffusion model:
   permeability → parameters

   The inverse model is a diffusion model:
   - It adds noise to parameters x0 to get x_t = x0 + sigma(t) * eps
   - A neural net learns to predict the clean x0 from (x_t, cond, t)
   - At sampling time, we start from noise and iteratively "denoise"
     to generate plausible parameter sets that match the permeability.

Implementation is inspired by:
"Step-by-Step Diffusion: An Elementary Tutorial" (Nakkiran et al., Apple/Mila).

Main ideas:
-----------

- Forward diffusion:    x_t = x_0 + sigma(t) * epsilon
- Network predicts x0 directly (not epsilon).
- Sampling uses a DDIM-style deterministic update based on Claim 2:
  E[x_{t-dt} | x_t] ≈ (dt/t) * E[x_0 | x_t] + (1 - dt/t) * x_t

This file contains:
  - Data loading and scaling
  - Forward model definition and training
  - Inverse diffusion model definition and training
  - Test-time sampling and basic metrics
"""

import os
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# 1. General utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Data loading and preprocessing
# ============================================================

def load_dataframe_inverse(csv_path: str):
    """
    Load the CSV and identify:
      - param_cols: columns of physical parameters (targets for generation)
      - perm_cols:  permeability columns (Kxx, Kyy, Kzz), used as conditioning

    We assume the CSV has at least columns: "Kxx", "Kyy", "Kzz".
    """
    df = pd.read_csv(csv_path)

    assert set(["Kxx", "Kyy", "Kzz"]).issubset(df.columns), \
        "CSV must contain columns: Kxx, Kyy, Kzz."

    # These columns are NOT considered physical parameters
    exclude = {"id", "seed", "Kxx", "Kyy", "Kzz"}

    # All other columns are treated as physical parameters
    param_cols = [c for c in df.columns if c not in exclude]
    perm_cols = ["Kxx", "Kyy", "Kzz"]

    return df, param_cols, perm_cols


def split_dataframe(df: pd.DataFrame,
                    val_size: float,
                    test_size: float,
                    seed: int):
    """
    Split the full dataframe into train/val/test.

    - First, split off test set.
    - Then, split the remaining into train and val.

    val_size and test_size are fractions of the FULL dataset.
    """
    df_remain, df_test = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    # val_rel is the val fraction relative to the remaining data
    val_rel = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_remain, test_size=val_rel, random_state=seed
    )
    return df_train, df_val, df_test


def arrays_and_scalers_inverse(df_train: pd.DataFrame,
                               df_val: pd.DataFrame,
                               df_test: pd.DataFrame,
                               param_cols,
                               perm_cols):
    """
    Convert train/val/test dataframes to NumPy arrays, and fit scalers.

    We produce:
      - x0_*:   standardized physical parameters
      - cond_*: standardized log10(permeabilities)

    Also returns:
      - scaler_x0:   StandardScaler for parameters
      - scaler_cond: StandardScaler for log10(permeabilities)
    """

    def to_arrays(d: pd.DataFrame):
        # Physical parameters (targets for diffusion model)
        params = d[param_cols].astype(np.float32).values

        # Raw permeability values
        perms_lin = d[perm_cols].astype(np.float64).values

        # Drop rows with NaNs in permeability
        nan_rows = np.isnan(perms_lin).any(axis=1)
        if np.any(nan_rows):
            perms_lin = perms_lin[~nan_rows]
            params = params[~nan_rows]

        # Avoid log10(0) by clipping tiny values
        perms_lin = np.clip(perms_lin, 1e-30, None)

        # Normalize permeability scale before log10.
        # NOTE: Here we divide by (2e-6)^2 (this comes from your physical setup).
        perms_lin = perms_lin / ((2e-6) ** 2)

        # Work with log10(K) to keep dynamic range reasonable
        perms_log = np.log10(perms_lin).astype(np.float32)

        return params, perms_log

    # Convert each split
    x0_tr, cond_tr = to_arrays(df_train)
    x0_va, cond_va = to_arrays(df_val)
    x0_te, cond_te = to_arrays(df_test)

    # Fit scalers on training data
    scaler_x0 = StandardScaler().fit(x0_tr)
    scaler_cond = StandardScaler().fit(cond_tr)

    # Standardize parameters
    x0_tr = scaler_x0.transform(x0_tr).astype(np.float32)
    x0_va = scaler_x0.transform(x0_va).astype(np.float32)
    x0_te = scaler_x0.transform(x0_te).astype(np.float32)

    # Standardize log10(K)
    cond_tr = scaler_cond.transform(cond_tr).astype(np.float32)
    cond_va = scaler_cond.transform(cond_va).astype(np.float32)
    cond_te = scaler_cond.transform(cond_te).astype(np.float32)

    return (x0_tr, cond_tr,
            x0_va, cond_va,
            x0_te, cond_te,
            scaler_x0, scaler_cond)


def make_inverse_loaders(x0_tr, cond_tr,
                         x0_va, cond_va,
                         x0_te, cond_te,
                         batch_size: int,
                         num_workers: int = 0):
    """
    Create PyTorch DataLoaders for train/val/test.

    Each batch will contain:
      - x0:   standardized parameters
      - cond: standardized log10(Kxx,Kyy,Kzz)
    """
    train_ds = TensorDataset(
        torch.from_numpy(x0_tr), torch.from_numpy(cond_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(x0_va), torch.from_numpy(cond_va)
    )
    test_ds = TensorDataset(
        torch.from_numpy(x0_te), torch.from_numpy(cond_te)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 3. Model components
# ============================================================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (like positional encoding in Transformers).

    Input:
      t: shape [B] or [B,1], with values in [0,1]

    Output:
      emb: shape [B, dim]
    """

    def __init__(self, dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Ensure t has shape [B]
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        half = self.dim // 2

        # Frequencies for sine/cosine
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, device=t.device).float()
            / half
        )  # [half]

        # Broadcast: [B, half]
        args = t[:, None] * freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # If dim is odd, pad with one zero channel
        if self.dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros_like(emb[:, :1])], dim=-1
            )

        return emb


class ResBlock(nn.Module):
    """
    Simple residual block:
      y = x + Linear -> (optional LayerNorm) -> Act -> (optional Dropout)
    """

    def __init__(self,
                 dim: int,
                 Act,
                 layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(Act())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiffusionMLP(nn.Module):
    """
    Conditional denoiser network for diffusion.

    The network predicts clean parameters x0 from:
      - x_t:   noisy parameters  [B, param_dim]
      - cond:  conditioning info (log10(Kxx,Kyy,Kzz)) [B, cond_dim]
      - t:     time step in [0,1] [B] or [B,1]

    Forward pass:
      1) Embed t with TimeEmbedding
      2) Concatenate [x_t, cond, t_emb]
      3) Feed through MLP + residual blocks
      4) Output predicted x0
    """

    def __init__(self,
                 param_dim: int,
                 cond_dim: int = 3,
                 hidden=(256, 256, 128),
                 activation: str = "gelu",
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 time_dim: int = 32,
                 num_res_blocks: int = 2):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        Act = acts.get(activation, nn.GELU)

        self.time_emb = TimeEmbedding(dim=time_dim)

        # Input is [x_t, cond, time_embedding]
        in_dim = param_dim + cond_dim + time_dim

        # Backbone MLP (input → hidden[-1])
        backbone_layers = []
        prev = in_dim
        for h in hidden:
            backbone_layers.append(nn.Linear(prev, h))
            if layer_norm:
                backbone_layers.append(nn.LayerNorm(h))
            backbone_layers.append(Act())
            if dropout and dropout > 0:
                backbone_layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*backbone_layers)

        # Residual blocks at final hidden dimension
        block_dim = hidden[-1]
        self.res_blocks = nn.ModuleList([
            ResBlock(block_dim, Act, layer_norm=layer_norm, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Final linear layer to predict x0
        self.out = nn.Linear(block_dim, param_dim)

    def forward(self,
                x_t: torch.Tensor,
                cond: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # Normalize shapes for t
        if t.dim() == 1:
            t_in = t
        else:
            t_in = t.view(-1)

        t_emb = self.time_emb(t_in)  # [B, time_dim]

        # Concatenate everything
        x = torch.cat([x_t, cond, t_emb], dim=-1)

        # Backbone MLP
        h = self.backbone(x)

        # Residual refinement
        for blk in self.res_blocks:
            h = blk(h)

        # Predict clean parameters
        return self.out(h)


class ForwardMLP(nn.Module):
    """
    Simple forward model:
      parameters (x0) → cond (log10(Kxx,Kyy,Kzz))

    This is just a standard MLP regression network.
    """

    def __init__(self,
                 param_dim: int,
                 cond_dim: int = 3,
                 hidden=(256, 256),
                 activation: str = "gelu",
                 dropout: float = 0.0,
                 layer_norm: bool = True):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        Act = acts.get(activation, nn.GELU)

        layers = []
        prev = param_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(Act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # Final layer to cond_dim
        layers.append(nn.Linear(prev, cond_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4. Lightning modules (training logic)
# ============================================================

class ForwardLightning(pl.LightningModule):
    """
    Lightning wrapper for the forward MLP: parameters → log10(K).
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.crit = nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=8, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"},
        }

    def forward(self, x):
        return self.model(x)

    def _step_common(self, batch, stage: str):
        x0, cond = batch  # both already standardized
        pred = self.model(x0)
        loss = self.crit(pred, cond)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        with torch.no_grad():
            mae = (pred - cond).abs().mean()
            self.log(f"{stage}/mae", mae, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step_common(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step_common(batch, "test")


class InverseDiffusionLightning(pl.LightningModule):
    """
    LightningModule for the inverse diffusion model:

      p(x0 | Kxx,Kyy,Kzz)

    where:
      - x0 are standardized material parameters
      - cond is standardized log10(Kxx,Kyy,Kzz)

    Training:
      - Sample a noise level sigma(t) and add noise: x_t = x0 + sigma * eps
      - Model predicts x0_hat = f(x_t, cond, t)
      - Loss is (weighted) MSE between x0_hat and x0
      - Optionally also adds a "forward consistency" loss:
          fwd_loss = MSE( forward_model(x0_hat), cond )

    Sampling:
      - Start from pure noise at sigma_max
      - Use DDIM-style updates to progressively move towards x0_hat.
    """

    def __init__(self,
                 param_dim: int,
                 cond_dim: int,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 sigma_min: float = 0.01,
                 sigma_max: float = 0.8,
                 use_cosine_schedule: bool = True,
                 forward_model: nn.Module | None = None,
                 lambda_forward: float = 0.0,
                 **mlp_kwargs):
        super().__init__()

        # Do not try to save the forward_model in hyperparams (can be large)
        self.save_hyperparameters(ignore=["forward_model"])

        self.param_dim = param_dim
        self.cond_dim = cond_dim

        # Denoiser network
        self.model = DiffusionMLP(
            param_dim=param_dim,
            cond_dim=cond_dim,
            **mlp_kwargs,
        )

        self.crit = nn.MSELoss()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_cosine_schedule = use_cosine_schedule

        # Optional frozen forward model
        self.forward_model = forward_model
        self.lambda_forward = lambda_forward

        if self.forward_model is not None:
            self.forward_model.eval()
            for p in self.forward_model.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=8, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"},
        }

    def _sample_noise_level(self, batch_size: int, device: torch.device):
        """
        Sample a time t in [0,1] and corresponding noise scale sigma(t).

        We bias samples towards higher noise (t near 1) by squaring a uniform u.
        """
        u = torch.rand(batch_size, device=device)
        # t in [~1e-4, 1], with more mass near 1
        t = 1.0 - (1.0 - 1e-4) * (u ** 2)

        if self.use_cosine_schedule:
            cos_val = torch.cos(math.pi * t / 2.0)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_val) ** 2
        else:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t

        return t, sigma

    def _step_common(self, batch, stage: str):
        # x0:   standardized parameters
        # cond: standardized log10(K)
        x0, cond = batch
        B = x0.size(0)
        device = x0.device

        # Sample time and noise level
        t, sigma = self._sample_noise_level(B, device)  # t: [B], sigma: [B]
        sigma = sigma.view(B, 1)                        # [B,1] for broadcasting

        # Add noise
        eps = torch.randn_like(x0)
        x_t = x0 + sigma * eps

        # Predict clean x0
        x0_hat = self.model(x_t, cond, t)

        # Diffusion loss (weighted MSE)
        base_loss = (x0_hat - x0).pow(2).mean(dim=1)   # [B]
        w = (sigma.view(-1) / self.sigma_max) ** 2 + 1e-3
        diff_loss = (w * base_loss).mean()

        # Optional forward-consistency loss
        fwd_loss = torch.tensor(0.0, device=device)
        if self.forward_model is not None and self.lambda_forward > 0.0:
            pred_cond = self.forward_model(x0_hat)
            fwd_loss = F.mse_loss(pred_cond, cond)
            self.log(f"{stage}/forward_loss", fwd_loss,
                     prog_bar=False, on_epoch=True, on_step=False)

        loss = diff_loss + self.lambda_forward * fwd_loss

        # Log main loss and a simple MAE
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        with torch.no_grad():
            mae = (x0_hat - x0).abs().mean()
            self.log(f"{stage}/mae", mae, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step_common(batch, "val")

    def test_step(self, batch, batch_idx):
        """
        During test, we:
          - run one diffusion step loss
          - generate full samples and compute reconstruction metrics
        """
        x0, cond = batch
        # Single-step training-style loss
        loss = self._step_common(batch, "test")

        # Full sampling: cond → x0_gen
        with torch.no_grad():
            x0_gen = self.sample(cond, num_steps=500)

            # Metrics in standardized parameter space
            mse = (x0_gen - x0).pow(2).mean()
            mae = (x0_gen - x0).abs().mean()

            self.log("test/mse_gen", mse, prog_bar=True, on_epoch=True)
            self.log("test/mae_gen", mae, prog_bar=True, on_epoch=True)

            # Forward consistency for generated parameters
            if self.forward_model is not None:
                pred_cond_gen = self.forward_model(x0_gen)
                fwd_gen_mse = F.mse_loss(pred_cond_gen, cond)
                fwd_gen_mae = (pred_cond_gen - cond).abs().mean()

                self.log("test/fwd_mse_gen", fwd_gen_mse, prog_bar=True, on_epoch=True)
                self.log("test/fwd_mae_gen", fwd_gen_mae, prog_bar=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = 500, generator: torch.Generator = None) -> torch.Tensor:
        """
        DDIM-style sampling for generating parameters from permeability values.
        
        This function performs the reverse diffusion process: starting from random noise,
        it gradually denoises to generate parameter values that match the given permeability.

        Args:
            cond: [B, cond_dim] Standardized log10(Kxx, Kyy, Kzz) permeability values
            num_steps: Number of denoising steps (more steps = better quality but slower)
            generator: Optional PyTorch Generator for reproducible randomness.
                      If None, uses default random state.

        Returns:
            x0: [B, param_dim] Generated standardized parameter values
        """
        self.eval()

        # Get batch size from input condition tensor
        batch_size_from_input = cond.size(0)
        device = cond.device
        param_dim = self.param_dim

        # Initialize with random noise at maximum noise level
        # This is the starting point for the denoising process
        if generator is not None:
            # Use provided generator for reproducible randomness
            x = torch.randn(batch_size_from_input, param_dim, device=device, generator=generator) * self.sigma_max
        else:
            # Use default random state
            x = torch.randn(batch_size_from_input, param_dim, device=device) * self.sigma_max

        # Non-uniform time grid: more steps near high noise (t ~ 1)
        # This schedule concentrates more denoising steps at high noise levels where refinement is most needed
        t_eps = 1e-4
        s = torch.linspace(0, 1, steps=num_steps, device=device)
        times = 1.0 - (1.0 - t_eps) * s**2  # Quadratic schedule: more steps near t=1 (high noise)

        # Perform iterative denoising: gradually reduce noise to generate clean parameters
        for i, cur_t in enumerate(times):
            # Create time tensor for this step (same time for all samples in batch)
            t_batch = torch.full((batch_size_from_input,), cur_t.item(), device=device)

            # Predict clean x0 at this noise level
            x0_hat = self.model(x, cond, t_batch)

            # Last step: just take x0_hat
            if i == len(times) - 1:
                x = x0_hat
                break

            next_t = times[i + 1].item()
            cur_t_scalar = cur_t.item()

            # Compute sigma at current and next time
            if self.use_cosine_schedule:
                cos_cur = math.cos(math.pi * cur_t_scalar / 2.0)
                cos_next = math.cos(math.pi * next_t / 2.0)
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_cur) ** 2
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_next) ** 2
            else:
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * cur_t_scalar
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * next_t

            # DDIM update: move x closer to x0_hat based on sigma ratio
            if cur_sigma > 1e-5:
                sigma_ratio = next_sigma / cur_sigma
                sigma_ratio = max(0.0, min(1.0, sigma_ratio))
                x = x0_hat + sigma_ratio * (x - x0_hat)
            else:
                x = x0_hat

        return x


# ============================================================
# 5. Training helpers
# ============================================================

def train_forward_model(train_loader,
                        val_loader,
                        param_dim: int,
                        cond_dim: int,
                        outdir: str,
                        lr: float = 1e-3,
                        weight_decay: float = 1e-5,
                        max_epochs: int = 300,
                        patience: int = 20):
    """
    Train the forward model (parameters → log10(K)) and return the
    best-performing frozen nn.Module.
    """
    fwd_outdir = os.path.join(outdir, "forward_model")
    os.makedirs(fwd_outdir, exist_ok=True)

    fwd_mlp = ForwardMLP(param_dim=param_dim, cond_dim=cond_dim, hidden=(256, 256))
    fwd_lit = ForwardLightning(fwd_mlp, lr=lr, weight_decay=weight_decay)

    ckpt_cb = ModelCheckpoint(
        dirpath=fwd_outdir,
        filename="forward-{epoch:03d}-{val_loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val/loss", mode="min", patience=patience
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=10,
        default_root_dir=fwd_outdir,
    )

    trainer.fit(fwd_lit, train_loader, val_loader)

    # Load best forward checkpoint and return underlying nn.Module
    best_fwd = ForwardLightning.load_from_checkpoint(
        ckpt_cb.best_model_path, model=fwd_mlp
    )
    forward_model = best_fwd.model
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    return forward_model


def build_inverse_model_and_callbacks(param_dim: int,
                                      cond_dim: int,
                                      outdir: str,
                                      learning_rate: float,
                                      patience: int,
                                      forward_model: nn.Module,
                                      lambda_forward: float = 0.0):
    """
    Build the inverse diffusion LightningModule and associated callbacks.
    """
    model = InverseDiffusionLightning(
        param_dim=param_dim,
        cond_dim=cond_dim,
        lr=learning_rate,
        weight_decay=1e-5,
        hidden=(512, 512, 256),
        activation="gelu",
        dropout=0.0,
        layer_norm=True,
        time_dim=32,
        sigma_min=0.01,
        sigma_max=1.0,
        use_cosine_schedule=True,
        forward_model=forward_model,
        lambda_forward=lambda_forward,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=outdir,
        filename="invdiff-{epoch:03d}-{val_loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val/loss", mode="min", patience=patience
    )

    return model, ckpt_cb, es_cb


def build_trainer(max_epochs: int, outdir: str, callbacks):
    """
    Create a Trainer for the inverse diffusion model.
    """
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir=outdir,
    )
    return trainer


def train_and_validate(trainer, model, train_loader, val_loader, ckpt_cb):
    """
    Run training + validation and return path to best checkpoint.
    """
    trainer.fit(model, train_loader, val_loader)
    return ckpt_cb.best_model_path


def test_current_and_best(trainer,
                          model,
                          test_loader,
                          best_ckpt_path,
                          forward_model):
    """
    Test both:
      1) the current (last) model in memory
      2) the best checkpoint from training

    This will log loss/MAE/MSE metrics via Lightning.
    """
    # Test current in-memory model
    trainer.test(model, dataloaders=test_loader)

    # Load and test best checkpoint
    best_model = InverseDiffusionLightning.load_from_checkpoint(
        best_ckpt_path,
        forward_model=forward_model,
    )
    trainer.test(best_model, dataloaders=test_loader)

#===========================================================

def generate_samples_from_csv_file(
    model,
    input_csv_path: str,
    param_cols,
    perm_cols,
    scaler_x0,
    scaler_cond,
    output_csv_path: str,
    num_samples_per_row: int = 1000,
    num_steps: int = 500,
    device: str = "cpu",
    use_random_seed: bool = True,
    batch_size: int = 500,
):
    """
    Generate diverse parameter samples from permeability values in a CSV file.
    
    This function reads permeability values (Kxx, Kyy, Kzz) from a CSV file and generates
    multiple different parameter sets for each row. Each sample is generated using a unique
    random seed to ensure diversity in the outputs.
    
    Args:
        model: Trained InverseDiffusionLightning model for generating parameters
        input_csv_path: Path to CSV file containing Kxx, Kyy, Kzz columns
        param_cols: List of parameter column names to generate
        perm_cols: List of permeability column names (Kxx, Kyy, Kzz)
        scaler_x0: StandardScaler for normalizing/denormalizing parameters
        scaler_cond: StandardScaler for normalizing/denormalizing log10(permeabilities)
        output_csv_path: Path where the output CSV will be saved
        num_samples_per_row: Number of different samples to generate per input row (default: 1000)
        num_steps: Number of diffusion steps for each sample (default: 500, more = better quality)
        device: Device to run inference on ("cpu" or "cuda")
        use_random_seed: If True, each sample uses a unique random seed (default: True)
        batch_size: Internal batch size for processing (default: 500, larger = faster but more memory)
    """
    # Set model to evaluation mode and move to specified device
    model.eval()
    model = model.to(device)
    
    # Load and validate input CSV file
    print(f"\n{'='*70}")
    print(f"Loading input CSV from: {input_csv_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
    
    df_input = pd.read_csv(input_csv_path)
    print(f"✓ Loaded {len(df_input)} rows from input CSV")
    
    # Validate that required permeability columns exist
    missing_cols = [col for col in perm_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")
    
    # Extract permeability values and handle missing data
    permeability_values_linear = df_input[perm_cols].astype(np.float64).values
    
    # Remove rows with invalid (NaN) permeability values
    rows_with_nan = np.isnan(permeability_values_linear).any(axis=1)
    if np.any(rows_with_nan):
        num_dropped = np.sum(rows_with_nan)
        print(f"⚠ Warning: Dropping {num_dropped} rows with NaN permeability values")
        permeability_values_linear = permeability_values_linear[~rows_with_nan]
        df_input = df_input[~rows_with_nan].reset_index(drop=True)
    
    num_valid_rows = len(df_input)
    print(f"✓ Processing {num_valid_rows} valid rows")
    print(f"✓ Generating {num_samples_per_row} sample(s) per row")
    print(f"✓ Using {num_steps} diffusion steps per sample")
    print(f"✓ Batch processing size: {batch_size} samples per batch")
    
    # Preprocess permeability values: normalize scale, convert to log10, then standardize
    # This matches the preprocessing used during training
    permeability_values_linear = np.clip(permeability_values_linear, 1e-30, None)
    permeability_values_linear = permeability_values_linear / ((2e-6) ** 2)  # Normalize scale
    permeability_values_log10 = np.log10(permeability_values_linear).astype(np.float32)
    
    # Standardize log10(permeability) values using the same scaler from training
    condition_standardized = scaler_cond.transform(permeability_values_log10).astype(np.float32)
    
    # Convert to PyTorch tensor and move to device
    condition_tensor = torch.from_numpy(condition_standardized).to(device)
    
    # Initialize lists to collect all generated predictions
    all_predictions = []
    all_input_row_indices = []
    
    # Determine batch size for processing (smaller of requested batch size or samples per row)
    processing_batch_size = min(batch_size, num_samples_per_row)
    
    print(f"\n{'='*70}")
    print(f"Starting generation: {num_samples_per_row} samples × {num_valid_rows} rows = "
          f"{num_samples_per_row * num_valid_rows} total samples")
    print(f"Processing batch size: {processing_batch_size}")
    print(f"Note: Each sample requires {num_steps} diffusion steps.")
    print(f"This may take significant time for large numbers of samples.")
    print(f"{'='*70}\n")
    
    import time as time_module
    overall_start_time = time_module.time()
    
    for row_idx in range(num_valid_rows):
        # Get condition (permeability values) for this specific input row
        condition_for_this_row = condition_tensor[row_idx:row_idx+1]  # Shape: [1, 3]
        
        # Generate multiple samples for this row in batches
        row_predictions = []
        row_start_time = time_module.time()
        
        for batch_start in range(0, num_samples_per_row, processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, num_samples_per_row)
            current_batch_size = batch_end - batch_start
            
            # Repeat condition for batch processing (when not using random seeds)
            condition_batch = condition_for_this_row.repeat(current_batch_size, 1)  # Shape: [current_batch_size, 3]
            
            # Generate predictions for this batch
            # Each sample needs its own random number generator to ensure diversity
            with torch.no_grad():
                if use_random_seed:
                    # Generate each sample individually with a unique random generator
                    # This ensures each sample produces different results
                    batch_predictions = []
                    
                    for sample_idx in range(current_batch_size):
                        # Generate a truly random seed for this sample using system entropy
                        # This ensures each sample gets a different random seed, and each run produces different results
                        # We use os.urandom to get cryptographically secure random bytes, then convert to integer
                        # Also add time and process ID to ensure uniqueness across runs
                        import time
                        random_bytes = os.urandom(4)  # 4 bytes = 32 bits
                        time_component = int(time.time() * 1000000) % 1000000  # Microseconds
                        process_id = os.getpid() % 10000
                        sample_seed = (int.from_bytes(random_bytes, byteorder='big') + time_component + process_id * 1000) % (2**31)
                        
                        # Ensure seed is positive and non-zero
                        if sample_seed == 0:
                            # If by chance we get 0, use a fallback random seed
                            sample_seed = (int.from_bytes(os.urandom(4), byteorder='big') + time_component) % (2**31) + 1
                        
                        # Debug: Print first few seeds to verify they're different
                        if sample_idx < 3 and batch_start == 0 and row_idx == 0:
                            print(f"      Sample {sample_idx + 1}: Random seed = {sample_seed}")
                        
                        # Create a dedicated random number generator for this sample
                        # This ensures complete isolation from other samples
                        # Convert device string to torch.device if needed for generator
                        if isinstance(device, str):
                            torch_device_obj = torch.device(device)
                        else:
                            torch_device_obj = device
                        
                        # Create generator on the correct device
                        sample_generator = torch.Generator(device=torch_device_obj)
                        sample_generator.manual_seed(sample_seed)
                        
                        # Generate a single sample with its own random generator
                        # This guarantees unique randomness for each sample
                        single_condition = condition_for_this_row  # Shape: [1, 3] - permeability values for this row
                        single_prediction_std = model.sample(
                            single_condition, 
                            num_steps=num_steps,
                            generator=sample_generator
                        )
                        
                        # Extract the prediction (remove batch dimension)
                        batch_predictions.append(single_prediction_std.cpu().numpy()[0])
                    
                    # Convert list of predictions to numpy array for batch processing
                    x0_pred_std_array = np.array(batch_predictions)
                    
                else:
                    # Fixed seed mode: all samples will be similar/deterministic
                    # Only set seed once at the very beginning
                    if batch_start == 0 and row_idx == 0:
                        torch.manual_seed(42)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(42)
                        np.random.seed(42)
                        random.seed(42)
                    
                    # Generate batch of samples (all will be similar due to fixed seed)
                    x0_pred_std = model.sample(condition_batch, num_steps=num_steps)
                    x0_pred_std_array = x0_pred_std.cpu().numpy()
            
            # Convert standardized predictions back to original parameter scale
            predictions_original_scale = scaler_x0.inverse_transform(x0_pred_std_array)
            row_predictions.extend(predictions_original_scale)
            
            # Track which input row each sample came from (1-indexed for user-friendly output)
            all_input_row_indices.extend([row_idx + 1] * current_batch_size)
            
            # Progress update: show progress every 1000 samples or at the end of the row
            samples_completed_this_row = batch_start + current_batch_size
            if samples_completed_this_row % 1000 == 0 or samples_completed_this_row == num_samples_per_row:
                elapsed_time = time_module.time() - row_start_time
                samples_per_second = samples_completed_this_row / elapsed_time if elapsed_time > 0 else 0
                remaining_samples = num_samples_per_row - samples_completed_this_row
                estimated_seconds_remaining = remaining_samples / samples_per_second if samples_per_second > 0 else 0
                
                print(f"    Row {row_idx + 1}: {samples_completed_this_row}/{num_samples_per_row} samples "
                      f"({samples_per_second:.1f} samples/sec, ~{estimated_seconds_remaining:.0f}s remaining)")
        
        # Add all predictions from this row to the master list
        all_predictions.extend(row_predictions)
        
        # Progress update after completing each row
        row_elapsed_time = time_module.time() - row_start_time
        total_samples_generated = (row_idx + 1) * num_samples_per_row
        total_elapsed_time = time_module.time() - overall_start_time
        average_samples_per_second = total_samples_generated / total_elapsed_time if total_elapsed_time > 0 else 0
        remaining_rows = num_valid_rows - (row_idx + 1)
        estimated_minutes_remaining = (remaining_rows * num_samples_per_row) / average_samples_per_second / 60 if average_samples_per_second > 0 else 0
        
        print(f"  ✓ Row {row_idx + 1}/{num_valid_rows} complete: {num_samples_per_row} samples in {row_elapsed_time:.1f}s "
              f"(Total: {total_samples_generated} samples, Avg: {average_samples_per_second:.1f} samples/sec, "
              f"Est. remaining: {estimated_minutes_remaining:.1f} min)")
    
    # Convert list of predictions to numpy array for easier manipulation
    all_predictions_array = np.array(all_predictions)
    
    print(f"\n{'='*70}")
    print(f"Organizing output data...")
    print(f"{'='*70}")
    
    # Build output dataframe with organized structure
    output_data = {}
    
    # Add input row index (which input row each sample came from)
    output_data['input_row_index'] = all_input_row_indices
    
    # Add sample index within each input row (1, 2, 3, ..., num_samples_per_row)
    if num_samples_per_row > 1:
        sample_indices_list = []
        for row_idx in range(num_valid_rows):
            sample_indices_list.extend(range(1, num_samples_per_row + 1))
        output_data['sample_index'] = sample_indices_list
    
    # Add permeability columns (Kxx, Kyy, Kzz) - repeat original values for each sample
    for col_name in perm_cols:
        repeated_permeability_values = []
        for row_idx in range(num_valid_rows):
            # Repeat the same permeability value for all samples from this row
            repeated_permeability_values.extend([df_input[col_name].iloc[row_idx]] * num_samples_per_row)
        output_data[col_name] = repeated_permeability_values
    
    # Add predicted parameter columns (the generated values)
    for param_idx, param_col_name in enumerate(param_cols):
        output_data[param_col_name] = all_predictions_array[:, param_idx]
    
    # Add any other columns from input CSV (preserve original data)
    other_input_columns = [col for col in df_input.columns if col not in perm_cols]
    for col_name in other_input_columns:
        repeated_other_values = []
        for row_idx in range(num_valid_rows):
            # Repeat the same value for all samples from this row
            repeated_other_values.extend([df_input[col_name].iloc[row_idx]] * num_samples_per_row)
        output_data[col_name] = repeated_other_values
    
    # Create dataframe from organized data
    output_dataframe = pd.DataFrame(output_data)
    
    # Reorder columns according to user specification:
    # Order: Unnamed: 0, id, seed, [all parameter columns], Kxx, Kyy, Kzz
    
    # Define the exact desired column order as specified by user
    preferred_column_order = []
    
    # 1. Other input columns in specific order (Unnamed: 0, id, seed)
    specific_input_order = ['Unnamed: 0', 'id', 'seed']
    for col in specific_input_order:
        if col in output_dataframe.columns:
            preferred_column_order.append(col)
    
    # 2. Add remaining other input columns (excluding metadata and permeability)
    remaining_input_cols = [col for col in other_input_columns 
                            if col not in specific_input_order 
                            and col not in ['input_row_index', 'sample_index'] 
                            and col not in perm_cols]
    for col in remaining_input_cols:
        if col in output_dataframe.columns:
            preferred_column_order.append(col)
    
    # 3. Add all parameter columns (generated values) in their original order
    for param_col in param_cols:
        if param_col in output_dataframe.columns:
            preferred_column_order.append(param_col)
    
    # 4. Add permeability columns (Kxx, Kyy, Kzz) at the end
    for perm_col in perm_cols:
        if perm_col in output_dataframe.columns:
            preferred_column_order.append(perm_col)
    
    # Only include columns that actually exist in the dataframe
    final_column_order = [col for col in preferred_column_order if col in output_dataframe.columns]
    
    # Add any remaining columns that weren't in our preferred order (like input_row_index, sample_index if they exist)
    remaining_cols = [col for col in output_dataframe.columns if col not in final_column_order]
    if remaining_cols:
        # Add remaining columns at the beginning (metadata columns)
        final_column_order = remaining_cols + final_column_order
    
    output_dataframe = output_dataframe[final_column_order]
    
    # Save results to CSV file
    output_dataframe.to_csv(output_csv_path, index=False)
    
    # Print summary of results
    total_elapsed_time = time_module.time() - overall_start_time
    print(f"\n{'='*70}")
    print(f"✓ Generation complete!")
    print(f"{'='*70}")
    print(f"✓ Saved {len(output_dataframe)} samples to: {output_csv_path}")
    print(f"✓ Output shape: {output_dataframe.shape[0]} rows × {output_dataframe.shape[1]} columns")
    print(f"✓ Total time: {total_elapsed_time/60:.1f} minutes")
    print(f"✓ Average speed: {len(output_dataframe)/total_elapsed_time:.1f} samples/second")
    print(f"✓ Columns: {list(output_dataframe.columns)[:10]}..." if len(output_dataframe.columns) > 10 else f"✓ Columns: {list(output_dataframe.columns)}")
    print(f"{'='*70}\n")
    
    return output_dataframe


def generate_predictions_and_save_csv(model,
                                       df_data,
                                       param_cols,
                                       perm_cols,
                                       scaler_x0,
                                       scaler_cond,
                                       output_csv_path: str,
                                       num_steps: int = 500,
                                       device: str = "cpu"):
    """
    Generate predictions for the given dataframe and save to CSV with the same column order.
    
    Args:
        model: Trained InverseDiffusionLightning model
        df_data: DataFrame with the data (should have perm_cols and optionally other columns)
        param_cols: List of parameter column names
        perm_cols: List of permeability column names (Kxx, Kyy, Kzz)
        scaler_x0: StandardScaler for parameters
        scaler_cond: StandardScaler for log10(permeabilities)
        output_csv_path: Path to save the output CSV
        num_steps: Number of diffusion steps for sampling
        device: Device to run inference on
    """
    model.eval()
    model = model.to(device)
    
    # Process permeability values (same as in arrays_and_scalers_inverse)
    perms_lin = df_data[perm_cols].astype(np.float64).values
    
    # Drop rows with NaNs in permeability
    nan_rows = np.isnan(perms_lin).any(axis=1)
    valid_mask = ~nan_rows
    perms_lin = perms_lin[valid_mask]
    df_valid = df_data[valid_mask].copy().reset_index(drop=True)
    
    # Normalize permeability scale before log10
    perms_lin = np.clip(perms_lin, 1e-30, None)
    perms_lin = perms_lin / ((2e-6) ** 2)
    perms_log = np.log10(perms_lin).astype(np.float32)
    
    # Standardize log10(K)
    cond_std = scaler_cond.transform(perms_log).astype(np.float32)
    
    # Convert to tensor
    cond_tensor = torch.from_numpy(cond_std).to(device)
    
    # Generate predictions
    with torch.no_grad():
        x0_pred_std = model.sample(cond_tensor, num_steps=num_steps)
    
    # Inverse transform to original scale
    x0_pred = scaler_x0.inverse_transform(x0_pred_std.cpu().numpy())
    
    # Get the original column order from the input dataframe
    original_cols = df_data.columns.tolist()
    
    # Create a dictionary to build the output dataframe
    output_data = {}
    
    # Add permeability columns (Kxx, Kyy, Kzz) - use original values from valid rows
    for col in perm_cols:
        if col in original_cols:
            output_data[col] = df_valid[col].values
    
    # Add predicted parameter columns (these are the generated predictions)
    for i, col in enumerate(param_cols):
        if col in original_cols:
            output_data[col] = x0_pred[:, i]
    
    # Add any other columns that were in the original (like id, seed, etc.)
    other_cols = [c for c in original_cols if c not in param_cols and c not in perm_cols]
    for col in other_cols:
        if col in df_valid.columns:
            output_data[col] = df_valid[col].values
    
    # Create dataframe with data in the order we collected it
    output_df = pd.DataFrame(output_data)
    
    # Reorder columns to match original order exactly
    # Only include columns that exist in both
    cols_to_include = [c for c in original_cols if c in output_df.columns]
    output_df = output_df[cols_to_include]
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    print(f"Output shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")
    
    return output_df


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ============================================================
# 6. main()
# ============================================================

def main(
    checkpoint_path: str = None,
    resume_training: bool = False,
    inference_only: bool = False,
    input_csv_path: str = None,
    num_samples_per_row: int = 1000,
    num_steps: int = 500,
    use_random_seed: bool = True,
):
    """
    Main training/inference function.
    
    Args:
        checkpoint_path: Path to checkpoint file to load. If None, trains from scratch.
        resume_training: If True and checkpoint_path provided, resume training from checkpoint.
        inference_only: If True and checkpoint_path provided, skip training and only run inference.
        input_csv_path: Path to CSV file with Kxx, Kyy, Kzz columns for batch processing
        num_samples_per_row: Number of samples to generate per row in input CSV (default: 1000)
        num_steps: Number of diffusion steps for sampling (default: 500, lower = faster)
        use_random_seed: If True, use random seed for each sample (default: True)
    """
    # Only set seed if not using random seed for CSV inference
    # When using CSV input with random seeds, don't set a fixed seed at the start
    if input_csv_path is None or not use_random_seed:
        set_seed(42)
    else:
        # For CSV input with random seeds, ensure we start with a non-deterministic state
        # Don't set a fixed seed - let each sample use its own random seed
        # Also ensure PyTorch's default random state is not deterministic
        import time
        # Set a random initial state based on current time to ensure different runs
        initial_random_seed = int(time.time() * 1000000) % (2**31)
        torch.manual_seed(initial_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initial_random_seed)
        np.random.seed(initial_random_seed)
        random.seed(initial_random_seed)
        print(f"Using random seeds for each sample - initial random seed: {initial_random_seed}")

    # --- Paths and splits ---
    csv_path = "./result_no_duplicates_behnaz_remove_10.csv"
    val_size = 0.15
    test_size = 0.15
    seed = 42

    # --- Data loader config ---
    batch_size = 64
    num_workers = 15

    # --- Training config ---
    max_epochs = 500
    outdir = "./runs/perm_inverse_diffusion"
    os.makedirs(outdir, exist_ok=True)

    learning_rate = 3e-4
    patience = 40

    # 1) Load dataframe and split into train/val/test
    df, param_cols, perm_cols = load_dataframe_inverse(csv_path)
    df_train, df_val, df_test = split_dataframe(df, val_size, test_size, seed)

    # 2) Convert to arrays and scalers
    (x0_tr, cond_tr,
     x0_va, cond_va,
     x0_te, cond_te,
     scaler_x0, scaler_cond) = arrays_and_scalers_inverse(
        df_train, df_val, df_test,
        param_cols, perm_cols
    )

    # Save scalers for later use
    scaler_x0_path = os.path.join(outdir, "scaler_x0.joblib")
    scaler_cond_path = os.path.join(outdir, "scaler_cond.joblib")
    joblib.dump(scaler_x0, scaler_x0_path)
    joblib.dump(scaler_cond, scaler_cond_path)
    print(f"Saved scalers to {outdir}")

    # 3) Build DataLoaders
    train_loader, val_loader, test_loader = make_inverse_loaders(
        x0_tr, cond_tr,
        x0_va, cond_va,
        x0_te, cond_te,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Check if we should load from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        
        # Load the checkpoint to get model configuration
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})
        param_dim = hparams.get('param_dim', len(param_cols))
        cond_dim = hparams.get('cond_dim', len(perm_cols))
        
        # Load forward model if it exists
        forward_model = None
        forward_model_dir = os.path.join(outdir, "forward_model")
        if os.path.exists(forward_model_dir):
            # Try to find forward model checkpoints
            forward_checkpoints = [f for f in os.listdir(forward_model_dir) 
                                 if f.endswith('.ckpt')]
            if forward_checkpoints:
                # Try to load the best checkpoint (usually named with lowest val_loss)
                # Sort by filename which typically contains val_loss
                try:
                    forward_checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]) 
                                            if 'val_loss=' in x else float('inf'))
                    best_forward = os.path.join(forward_model_dir, forward_checkpoints[0])
                    print(f"Loading forward model from: {best_forward}")
                    # Reconstruct forward model architecture
                    fwd_mlp = ForwardMLP(
                        param_dim=len(param_cols),
                        cond_dim=len(perm_cols),
                        hidden=(256, 256)
                    )
                    forward_lit = ForwardLightning.load_from_checkpoint(
                        best_forward,
                        model=fwd_mlp
                    )
                    forward_model = forward_lit.model
                    forward_model.eval()
                    for p in forward_model.parameters():
                        p.requires_grad = False
                    print("Forward model loaded successfully")
                except Exception as e:
                    print(f"Error loading forward model: {e}")
                    print("Training new forward model...")
                    forward_model = train_forward_model(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        param_dim=len(param_cols),
                        cond_dim=len(perm_cols),
                        outdir=outdir,
                        lr=1e-3,
                        weight_decay=1e-5,
                        max_epochs=300,
                        patience=20,
                    )
            else:
                print("No forward model checkpoints found, training new one...")
                forward_model = train_forward_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    param_dim=len(param_cols),
                    cond_dim=len(perm_cols),
                    outdir=outdir,
                    lr=1e-3,
                    weight_decay=1e-5,
                    max_epochs=300,
                    patience=20,
                )
        else:
            # Train forward model if it doesn't exist
            print("Forward model directory not found, training new forward model...")
            forward_model = train_forward_model(
                train_loader=train_loader,
                val_loader=val_loader,
                param_dim=len(param_cols),
                cond_dim=len(perm_cols),
                outdir=outdir,
                lr=1e-3,
                weight_decay=1e-5,
                max_epochs=300,
                patience=20,
            )
        
        # Load inverse model from checkpoint
        best_model = InverseDiffusionLightning.load_from_checkpoint(
            checkpoint_path,
            forward_model=forward_model,
        )
        
        best_ckpt = checkpoint_path
        
        if inference_only:
            print("\nRunning inference only (skipping training)...")
        elif resume_training:
            print("\nResuming training from checkpoint...")
            # Build model and callbacks
            model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(
                param_dim=len(param_cols),
                cond_dim=len(perm_cols),
                outdir=outdir,
                learning_rate=learning_rate,
                patience=patience,
                forward_model=forward_model,
                lambda_forward=0.0,
            )
            # Load state from checkpoint
            model.load_state_dict(best_model.state_dict())
            trainer = build_trainer(
                max_epochs=max_epochs,
                outdir=outdir,
                callbacks=[ckpt_cb, es_cb],
            )
            # Resume training
            trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
            best_ckpt = ckpt_cb.best_model_path
            best_model = InverseDiffusionLightning.load_from_checkpoint(
                best_ckpt,
                forward_model=forward_model,
            )
    else:
        # Train from scratch
        if checkpoint_path:
            print(f"Warning: Checkpoint path provided but file not found: {checkpoint_path}")
            print("Training from scratch...")
        
        # 4) Train forward model (parameters → log10(K))
        forward_model = train_forward_model(
            train_loader=train_loader,
            val_loader=val_loader,
            param_dim=len(param_cols),
            cond_dim=len(perm_cols),
            outdir=outdir,
            lr=1e-3,
            weight_decay=1e-5,
            max_epochs=300,
            patience=20,
        )

        # 5) Build inverse diffusion model + callbacks
        model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(
            param_dim=len(param_cols),
            cond_dim=len(perm_cols),
            outdir=outdir,
            learning_rate=learning_rate,
            patience=patience,
            forward_model=forward_model,
            # Currently disabled; set >0.0 to enable forward consistency loss
            lambda_forward=0.0,
        )

        # 6) Trainer for inverse model
        trainer = build_trainer(
            max_epochs=max_epochs,
            outdir=outdir,
            callbacks=[ckpt_cb, es_cb],
        )

        # 7) Train inverse model and get best checkpoint
        best_ckpt = train_and_validate(
            trainer, model, train_loader, val_loader, ckpt_cb
        )

        # 8) Test current and best models
        test_current_and_best(
            trainer, model, test_loader, best_ckpt, forward_model
        )
        
        # Load best model for inference
        best_model = InverseDiffusionLightning.load_from_checkpoint(
            best_ckpt,
            forward_model=forward_model,
        )

    # 9) Generate predictions and save to CSV
    if input_csv_path is not None:
        # Generate samples from CSV file input
        output_csv_path = os.path.join(outdir, "predicted_from_file.csv")
        print(f"\n{'='*70}")
        print("Generating samples from CSV file input")
        print(f"{'='*70}")
        print(f"Checkpoint location: {best_ckpt}")
        print(f"Output will be saved to: {output_csv_path}")
        
        generate_samples_from_csv_file(
            model=best_model,
            input_csv_path=input_csv_path,
            param_cols=param_cols,
            perm_cols=perm_cols,
            scaler_x0=scaler_x0,
            scaler_cond=scaler_cond,
            output_csv_path=output_csv_path,
            num_samples_per_row=num_samples_per_row,
            num_steps=num_steps,
            device=device,
            use_random_seed=use_random_seed,
            batch_size=500,
        )
    else:
        # Generate predictions for test set
        # The output CSV will have the same column order as the input file, with:
        # - Kxx, Kyy, Kzz: original permeability values (from input)
        # - Parameter columns: predicted values (generated by the model)
        # - Other columns: preserved from input (e.g., id, seed, etc.)
        output_csv_path = os.path.join(outdir, "predictions.csv")
        print(f"\nGenerating predictions and saving to {output_csv_path}...")
        print(f"Using test set with {len(df_test)} samples")
        
        # You can change df_test to:
        # - df_test: test set predictions
        # - df_val: validation set predictions  
        # - df_train: training set predictions
        # - df: all data predictions
        generate_predictions_and_save_csv(
            model=best_model,
            df_data=df_test,  # Change this to generate predictions for different datasets
            param_cols=param_cols,
            perm_cols=perm_cols,
            scaler_x0=scaler_x0,
            scaler_cond=scaler_cond,
            output_csv_path=output_csv_path,
            num_steps=500,
            device=device,
        )




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or load inverse diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load (for inference or resuming training)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint (requires --checkpoint)"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training and only run inference (requires --checkpoint)"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file containing Kxx, Kyy, Kzz columns for batch processing"
    )
    parser.add_argument(
        "--samples-per-row",
        type=int,
        default=1000,
        help="Number of samples to generate per row in input CSV (default: 1000)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help="Number of diffusion steps for sampling (default: 500, lower = faster but potentially lower quality)"
    )
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use fixed seed (default: False, uses random seed for each sample)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.inference_only or args.resume:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --inference-only or --resume")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Validate CSV input arguments
    if args.input_csv is not None:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for CSV file input")
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
    
    main(
        checkpoint_path=args.checkpoint,
        resume_training=args.resume,
        inference_only=args.inference_only,
        input_csv_path=args.input_csv,
        num_samples_per_row=args.samples_per_row,
        num_steps=args.num_steps,
        use_random_seed=not args.fixed_seed,
    )

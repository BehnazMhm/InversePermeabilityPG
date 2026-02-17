"""
Inverse Diffusion Model for Material Parameter Generation (DiffusionMLP only)
=============================================================================

Goal:
-----

We have a dataset where each row contains:
  - physical material parameters (many columns)
  - permeabilities: Kxx, Kyy, Kzz

We learn an inverse diffusion model:
  permeability → parameters

  - It adds noise to parameters x0 to get x_t = x0 + sigma(t) * eps
  - A neural net (DiffusionMLP) learns to predict the clean x0 from (x_t, cond, t)
  - At sampling time, we start from noise and iteratively "denoise"
    to generate plausible parameter sets that match the permeability.

Implementation is inspired by:
"Step-by-Step Diffusion: An Elementary Tutorial" (Nakkiran et al., Apple/Mila).

Main ideas:
-----------

- Forward diffusion:    x_t = x_0 + sigma(t) * epsilon
- Network predicts x0 directly (not epsilon).
- Sampling uses a DDIM-style deterministic update.

This file contains:
  - Data loading and scaling
  - Inverse diffusion model (DiffusionMLP + Lightning)
  - Test-time sampling and basic metrics
"""

import os
import math
import random

import numpy as np
import pandas as pd

import time

import torch
import torch.nn as nn

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


# ============================================================
# 4. Lightning module (inverse diffusion only)
# ============================================================

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
                 forward_model=None,
                 lambda_forward: float = 0.0,
                 **mlp_kwargs):
        super().__init__()
        # forward_model / lambda_forward kept for backward compat when loading old checkpoints; ignored
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
        t, sigma = self._sample_noise_level(B, device)
        sigma = sigma.view(B, 1)

        # Add noise
        eps = torch.randn_like(x0)
        x_t = x0 + sigma * eps

        # Predict clean x0
        x0_hat = self.model(x_t, cond, t)

        # Diffusion loss (weighted MSE)
        base_loss = (x0_hat - x0).pow(2).mean(dim=1)
        w = (sigma.view(-1) / self.sigma_max) ** 2 + 1e-3
        loss = (w * base_loss).mean()

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
        x0, cond = batch
        loss = self._step_common(batch, "test")

        with torch.no_grad():
            x0_gen = self.sample(cond, num_steps=500)
            mse = (x0_gen - x0).pow(2).mean()
            mae = (x0_gen - x0).abs().mean()
            self.log("test/mse_gen", mse, prog_bar=True, on_epoch=True)
            self.log("test/mae_gen", mae, prog_bar=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = 500, generator: torch.Generator = None) -> torch.Tensor:
        """
        DDIM-style sampling for generating parameters from permeability values.
        """
        self.eval()

        batch_size_from_input = cond.size(0)
        device = cond.device
        param_dim = self.param_dim

        if generator is not None:
            x = torch.randn(batch_size_from_input, param_dim, device=device, generator=generator) * self.sigma_max
        else:
            x = torch.randn(batch_size_from_input, param_dim, device=device) * self.sigma_max

        t_eps = 1e-4
        s = torch.linspace(0, 1, steps=num_steps, device=device)
        times = 1.0 - (1.0 - t_eps) * s**2

        for i, cur_t in enumerate(times):
            t_batch = torch.full((batch_size_from_input,), cur_t.item(), device=device)
            x0_hat = self.model(x, cond, t_batch)

            if i == len(times) - 1:
                x = x0_hat
                break

            next_t = times[i + 1].item()
            cur_t_scalar = cur_t.item()

            if self.use_cosine_schedule:
                cos_cur = math.cos(math.pi * cur_t_scalar / 2.0)
                cos_next = math.cos(math.pi * next_t / 2.0)
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_cur) ** 2
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_next) ** 2
            else:
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * cur_t_scalar
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * next_t

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

def build_inverse_model_and_callbacks(param_dim: int,
                                      cond_dim: int,
                                      outdir: str,
                                      learning_rate: float,
                                      patience: int):
    """Build the inverse diffusion LightningModule and associated callbacks."""
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
    """Create a Trainer for the inverse diffusion model."""
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
    """Run training + validation and return path to best checkpoint."""
    trainer.fit(model, train_loader, val_loader)
    return ckpt_cb.best_model_path


def test_current_and_best(trainer, model, test_loader, best_ckpt_path):
    """Test current model and best checkpoint."""
    trainer.test(model, dataloaders=test_loader)
    best_model = InverseDiffusionLightning.load_from_checkpoint(best_ckpt_path)
    trainer.test(best_model, dataloaders=test_loader)


# ============================================================
# 6. Generation from CSV / predictions
# ============================================================

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
    """Generate diverse parameter samples from permeability values in a CSV file."""
    model.eval()
    model = model.to(device)

    # Load and preprocess input
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

    df_input = pd.read_csv(input_csv_path)
    missing_cols = [col for col in perm_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    permeability_values_linear = df_input[perm_cols].astype(np.float64).values
    rows_with_nan = np.isnan(permeability_values_linear).any(axis=1)
    if np.any(rows_with_nan):
        permeability_values_linear = permeability_values_linear[~rows_with_nan]
        df_input = df_input[~rows_with_nan].reset_index(drop=True)

    num_valid_rows = len(df_input)
    permeability_values_linear = np.clip(permeability_values_linear, 1e-30, None)
    permeability_values_linear = permeability_values_linear / ((2e-6) ** 2)
    permeability_values_log10 = np.log10(permeability_values_linear).astype(np.float32)
    condition_standardized = scaler_cond.transform(permeability_values_log10).astype(np.float32)
    condition_tensor = torch.from_numpy(condition_standardized).to(device)

    # Generate samples (one row at a time, in batches)
    all_predictions = []
    all_input_row_indices = []
    processing_batch_size = min(batch_size, num_samples_per_row)
    overall_start_time = time.time()

    for row_idx in range(num_valid_rows):
        condition_for_this_row = condition_tensor[row_idx:row_idx+1]
        row_predictions = []

        for batch_start in range(0, num_samples_per_row, processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, num_samples_per_row)
            current_batch_size = batch_end - batch_start
            condition_batch = condition_for_this_row.repeat(current_batch_size, 1)

            with torch.no_grad():
                if use_random_seed:
                    batch_predictions = []
                    for sample_idx in range(current_batch_size):
                        random_bytes = os.urandom(4)
                        time_component = int(time.time() * 1000000) % 1000000
                        process_id = os.getpid() % 10000
                        sample_seed = (int.from_bytes(random_bytes, byteorder='big') + time_component + process_id * 1000) % (2**31)
                        if sample_seed == 0:
                            sample_seed = (int.from_bytes(os.urandom(4), byteorder='big') + time_component) % (2**31) + 1
                        torch_device_obj = torch.device(device) if isinstance(device, str) else device
                        sample_generator = torch.Generator(device=torch_device_obj)
                        sample_generator.manual_seed(sample_seed)
                        single_prediction_std = model.sample(condition_for_this_row, num_steps=num_steps, generator=sample_generator)
                        batch_predictions.append(single_prediction_std.cpu().numpy()[0])
                    x0_pred_std_array = np.array(batch_predictions)
                else:
                    if batch_start == 0 and row_idx == 0:
                        torch.manual_seed(42)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(42)
                        np.random.seed(42)
                        random.seed(42)
                    x0_pred_std = model.sample(condition_batch, num_steps=num_steps)
                    x0_pred_std_array = x0_pred_std.cpu().numpy()

            predictions_original_scale = scaler_x0.inverse_transform(x0_pred_std_array)
            row_predictions.extend(predictions_original_scale)
            all_input_row_indices.extend([row_idx + 1] * current_batch_size)

        all_predictions.extend(row_predictions)

    # Build output dataframe: metadata, then param columns, then permeability
    all_predictions_array = np.array(all_predictions)
    output_data = {}
    output_data['input_row_index'] = all_input_row_indices
    if num_samples_per_row > 1:
        sample_indices_list = []
        for r in range(num_valid_rows):
            sample_indices_list.extend(range(1, num_samples_per_row + 1))
        output_data['sample_index'] = sample_indices_list
    for col_name in perm_cols:
        repeated = []
        for r in range(num_valid_rows):
            repeated.extend([df_input[col_name].iloc[r]] * num_samples_per_row)
        output_data[col_name] = repeated
    for param_idx, param_col_name in enumerate(param_cols):
        output_data[param_col_name] = all_predictions_array[:, param_idx]
    other_input_columns = [col for col in df_input.columns if col not in perm_cols]
    for col_name in other_input_columns:
        repeated = []
        for r in range(num_valid_rows):
            repeated.extend([df_input[col_name].iloc[r]] * num_samples_per_row)
        output_data[col_name] = repeated

    output_dataframe = pd.DataFrame(output_data)
    # Column order: metadata first, then params, then Kxx/Kyy/Kzz
    preferred_column_order = []
    for col in ['Unnamed: 0', 'id', 'seed']:
        if col in output_dataframe.columns:
            preferred_column_order.append(col)
    remaining_input_cols = [c for c in other_input_columns if c not in ['Unnamed: 0', 'id', 'seed', 'input_row_index', 'sample_index'] and c not in perm_cols]
    for col in remaining_input_cols:
        if col in output_dataframe.columns:
            preferred_column_order.append(col)
    for param_col in param_cols:
        if param_col in output_dataframe.columns:
            preferred_column_order.append(param_col)
    for perm_col in perm_cols:
        if perm_col in output_dataframe.columns:
            preferred_column_order.append(perm_col)
    final_column_order = [c for c in preferred_column_order if c in output_dataframe.columns]
    remaining_cols = [c for c in output_dataframe.columns if c not in final_column_order]
    if remaining_cols:
        final_column_order = remaining_cols + final_column_order
    output_dataframe = output_dataframe[final_column_order]
    output_dataframe.to_csv(output_csv_path, index=False)

    total_elapsed_time = time.time() - overall_start_time
    print(f"Saved {len(output_dataframe)} samples to {output_csv_path} ({total_elapsed_time/60:.1f} min)")
    return output_dataframe


def generate_predictions_and_save_csv(model, df_data, param_cols, perm_cols, scaler_x0, scaler_cond,
                                       output_csv_path: str, num_steps: int = 500, device: str = "cpu"):
    """Generate predictions for the given dataframe and save to CSV."""
    model.eval()
    model = model.to(device)
    perms_lin = df_data[perm_cols].astype(np.float64).values
    nan_rows = np.isnan(perms_lin).any(axis=1)
    valid_mask = ~nan_rows
    perms_lin = perms_lin[valid_mask]
    df_valid = df_data[valid_mask].copy().reset_index(drop=True)
    perms_lin = np.clip(perms_lin, 1e-30, None)
    perms_lin = perms_lin / ((2e-6) ** 2)
    perms_log = np.log10(perms_lin).astype(np.float32)
    cond_std = scaler_cond.transform(perms_log).astype(np.float32)
    cond_tensor = torch.from_numpy(cond_std).to(device)
    with torch.no_grad():
        x0_pred_std = model.sample(cond_tensor, num_steps=num_steps)
    x0_pred = scaler_x0.inverse_transform(x0_pred_std.cpu().numpy())
    original_cols = df_data.columns.tolist()
    output_data = {}
    for col in perm_cols:
        if col in original_cols:
            output_data[col] = df_valid[col].values
    for i, col in enumerate(param_cols):
        if col in original_cols:
            output_data[col] = x0_pred[:, i]
    other_cols = [c for c in original_cols if c not in param_cols and c not in perm_cols]
    for col in other_cols:
        if col in df_valid.columns:
            output_data[col] = df_valid[col].values
    output_df = pd.DataFrame(output_data)
    cols_to_include = [c for c in original_cols if c in output_df.columns]
    output_df = output_df[cols_to_include]
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    return output_df


# ============================================================
# 7. main()
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
    Train inverse diffusion from scratch, or load a checkpoint and optionally resume / run inference.
    If input_csv_path is given, generates samples from that CSV; otherwise generates on the test set.
    """
    # Reproducible vs diverse: fixed seed for training/eval, random seed per sample for CSV generation
    if input_csv_path is None or not use_random_seed:
        set_seed(42)
    else:
        initial_seed = int(time.time() * 1e6) % (2**31)
        set_seed(initial_seed)

    # --- Config ---
    csv_path = "./result_no_duplicates_behnaz_remove_10.csv"
    val_size, test_size, seed = 0.15, 0.15, 42
    batch_size, num_workers = 64, 15
    max_epochs = 500
    outdir = "./runs/perm_inverse_diffusion"
    os.makedirs(outdir, exist_ok=True)
    learning_rate, patience = 3e-4, 40

    # --- Data ---
    df, param_cols, perm_cols = load_dataframe_inverse(csv_path)
    df_train, df_val, df_test = split_dataframe(df, val_size, test_size, seed)
    (x0_tr, cond_tr, x0_va, cond_va, x0_te, cond_te, scaler_x0, scaler_cond) = arrays_and_scalers_inverse(
        df_train, df_val, df_test, param_cols, perm_cols)
    joblib.dump(scaler_x0, os.path.join(outdir, "scaler_x0.joblib"))
    joblib.dump(scaler_cond, os.path.join(outdir, "scaler_cond.joblib"))
    train_loader, val_loader, test_loader = make_inverse_loaders(
        x0_tr, cond_tr, x0_va, cond_va, x0_te, cond_te,
        batch_size=batch_size, num_workers=num_workers,
    )

    # --- Checkpoint: load and optionally resume; else train from scratch ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        best_model = InverseDiffusionLightning.load_from_checkpoint(checkpoint_path)
        best_ckpt = checkpoint_path
        if inference_only:
            print("\nRunning inference only (skipping training)...")
        elif resume_training:
            print("\nResuming training from checkpoint...")
            model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(
                param_dim=len(param_cols), cond_dim=len(perm_cols), outdir=outdir,
                learning_rate=learning_rate, patience=patience)
            model.load_state_dict(best_model.state_dict())
            trainer = build_trainer(max_epochs=max_epochs, outdir=outdir, callbacks=[ckpt_cb, es_cb])
            trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
            best_ckpt = ckpt_cb.best_model_path
            best_model = InverseDiffusionLightning.load_from_checkpoint(best_ckpt)
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint not found: {checkpoint_path}. Training from scratch.")
        model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(
            param_dim=len(param_cols), cond_dim=len(perm_cols), outdir=outdir,
            learning_rate=learning_rate, patience=patience)
        trainer = build_trainer(max_epochs=max_epochs, outdir=outdir, callbacks=[ckpt_cb, es_cb])
        best_ckpt = train_and_validate(trainer, model, train_loader, val_loader, ckpt_cb)
        test_current_and_best(trainer, model, test_loader, best_ckpt)
        best_model = InverseDiffusionLightning.load_from_checkpoint(best_ckpt)

    # --- Output: CSV from file or test set ---
    if input_csv_path is not None:
        output_csv_path = os.path.join(outdir, "predicted_from_file.csv")
        generate_samples_from_csv_file(
            model=best_model, input_csv_path=input_csv_path, param_cols=param_cols, perm_cols=perm_cols,
            scaler_x0=scaler_x0, scaler_cond=scaler_cond, output_csv_path=output_csv_path,
            num_samples_per_row=num_samples_per_row, num_steps=num_steps, device=device,
            use_random_seed=use_random_seed, batch_size=500)
    else:
        output_csv_path = os.path.join(outdir, "predictions.csv")
        generate_predictions_and_save_csv(
            model=best_model, df_data=df_test, param_cols=param_cols, perm_cols=perm_cols,
            scaler_x0=scaler_x0, scaler_cond=scaler_cond, output_csv_path=output_csv_path,
            num_steps=500, device=device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or load inverse diffusion model (DiffusionMLP only)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--inference-only", action="store_true", help="Skip training, only run inference")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to CSV with Kxx, Kyy, Kzz")
    parser.add_argument("--samples-per-row", type=int, default=1000, help="Samples per row (default: 1000)")
    parser.add_argument("--num-steps", type=int, default=500, help="Diffusion steps (default: 500)")
    parser.add_argument("--fixed-seed", action="store_true", help="Use fixed seed per sample")
    args = parser.parse_args()
    if args.inference_only or args.resume:
        if not args.checkpoint:
            raise ValueError("--checkpoint required with --inference-only or --resume")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.input_csv is not None:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for CSV input")
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    main(
        checkpoint_path=args.checkpoint, resume_training=args.resume, inference_only=args.inference_only,
        input_csv_path=args.input_csv, num_samples_per_row=args.samples_per_row, num_steps=args.num_steps,
        use_random_seed=not args.fixed_seed,
    )

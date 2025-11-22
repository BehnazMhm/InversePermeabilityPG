"""
Inverse Diffusion Model for Material Parameter Generation
==========================================================

This module implements a conditional diffusion model that learns to generate
physical material parameters given permeability values (Kxx, Kyy, Kzz).

The forward model: physical parameters -> permeability
The backward model: permeability -> physical parameters (using diffusion)

Since multiple physical parameters can map to the same permeability,
we use a diffusion model that takes random noise and conditions on permeability
to generate diverse parameter samples.

Implementation follows the "Step-by-Step Diffusion: An Elementary Tutorial"
by Nakkiran et al. (Apple, Mila), using:
- Forward diffusion: x_t = x_0 + sigma(t) * epsilon
- x0 prediction (predicting clean data directly)
- DDIM sampling with Claim 2: E[x_{t-dt} | x_t] = (dt/t) * E[x_0 | x_t] + (1 - dt/t) * x_t
"""

import os
import math
import numpy as np
import pandas as pd
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# ---------------------------
# Utilities / Config
# ---------------------------

def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Data prep helpers (inverse)
# ---------------------------

def load_dataframe_inverse(csv_path: str):
    """Load CSV and split columns into:
       - parameter columns (targets we want to generate)
       - permeability columns (conditioning inputs: Kxx, Kyy, Kzz)
    """
    df = pd.read_csv(csv_path)
    assert set(["Kxx", "Kyy", "Kzz"]).issubset(df.columns), "CSV must have Kxx, Kyy, Kzz."
    exclude = {"id", "seed", "Kxx", "Kyy", "Kzz"}
    param_cols = [c for c in df.columns if c not in exclude]  # physical parameters
    perm_cols = ["Kxx", "Kyy", "Kzz"]
    print("Num rows:", len(df))
    print("Parameter columns (to generate):", param_cols)
    print("Condition columns (permeabilities):", perm_cols)
    return df, param_cols, perm_cols

def split_dataframe(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Same splitting logic as forward MLP code."""
    df_remain, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(df_remain, test_size=val_rel, random_state=seed)
    return df_train, df_val, df_test

def arrays_and_scalers_inverse(df_train, df_val, df_test, param_cols, perm_cols):
    """
    Build numpy arrays for:
      x0:   standardized physical parameters (model output)
      cond: standardized log10-permeabilities (conditioning input)
    plus two StandardScalers to invert transforms later if needed.
    """
    def to_arrays(d: pd.DataFrame):
        params = d[param_cols].astype(np.float32).values
        # Use log10 of permeability (same trick as forward code) to keep scale reasonable
        perms_lin = d[perm_cols].astype(np.float64).values
        nan_rows = np.isnan(perms_lin).any(axis=1)
        if np.any(nan_rows):
            print(f"Warning: Dropping {np.sum(nan_rows)} rows with NaN permeability.")
            perms_lin = perms_lin[~nan_rows]
            params = params[~nan_rows]
        perms_lin = np.clip(perms_lin, 1e-30, None)
        perms_log = np.log10(perms_lin).astype(np.float32)
        return params, perms_log

    x0_tr, cond_tr = to_arrays(df_train)
    x0_va, cond_va = to_arrays(df_val)
    x0_te, cond_te = to_arrays(df_test)

    scaler_x0 = StandardScaler().fit(x0_tr)
    scaler_cond = StandardScaler().fit(cond_tr)

    x0_tr = scaler_x0.transform(x0_tr).astype(np.float32)
    x0_va = scaler_x0.transform(x0_va).astype(np.float32)
    x0_te = scaler_x0.transform(x0_te).astype(np.float32)

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
    train_ds = TensorDataset(torch.from_numpy(x0_tr), torch.from_numpy(cond_tr))
    val_ds   = TensorDataset(torch.from_numpy(x0_va), torch.from_numpy(cond_va))
    test_ds  = TensorDataset(torch.from_numpy(x0_te), torch.from_numpy(cond_te))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ---------------------------
# Model definitions
# ---------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding as in diffusion models."""
    def __init__(self, dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape [B] in [0,1] or [B,1]
        returns: shape [B, dim]
        """
        # Ensure shape [B]
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, device=t.device).float() / half
        )
        # [B, half]
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            # pad one zero if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class DiffusionMLP(nn.Module):
    """
    Conditional denoiser network that predicts x0 (clean parameters) given:
      - x_t: noisy parameters
      - cond: permeability conditioning (Kxx, Kyy, Kzz)
      - t: time step
    """
    def __init__(
        self,
        param_dim: int,
        cond_dim: int = 3,
        hidden=(256, 256, 128),
        activation: str = "gelu",
        dropout: float = 0.1,
        layer_norm: bool = True,
        time_dim: int = 32,
    ):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        Act = acts.get(activation, nn.GELU)
        self.time_emb = TimeEmbedding(dim=time_dim)
        in_dim = param_dim + cond_dim + time_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(Act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, param_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, param_dim]
        cond: [B, cond_dim]
        t: [B] or [B,1] in [0,1]
        """
        if t.dim() == 1:
            t_in = t
        else:
            t_in = t.view(-1)
        t_emb = self.time_emb(t_in)  # [B, time_dim]
        x = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(x)

class InverseDiffusionLightning(pl.LightningModule):
    """
    LightningModule that trains a conditional diffusion model
    p(x0 | Kxx,Kyy,Kzz), where x0 are material parameters.
    
    Uses a variance-preserving diffusion process with linear noise schedule.
    """
    def __init__(
        self,
        param_dim: int,
        cond_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        use_cosine_schedule: bool = True,
        **mlp_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.param_dim = param_dim
        self.cond_dim = cond_dim
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
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"}}

    def _sample_noise_level(self, batch_size: int, device: torch.device):
        """Sample noise level t ~ Uniform(0, 1] and compute sigma(t)."""
        # t ~ Uniform(0, 1]; avoid exactly 0 to prevent division issues
        t = torch.rand(batch_size, device=device) * (1.0 - 1e-4) + 1e-4
        
        if self.use_cosine_schedule:
            # Cosine schedule: often better for generation quality
            # sigma(t) = sigma_min + (sigma_max - sigma_min) * (1 - cos(π * t / 2))^2
            cos_val = torch.cos(math.pi * t / 2.0)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_val) ** 2
        else:
            # Linear schedule: sigma(t) = sigma_min + (sigma_max - sigma_min) * t
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        
        return t, sigma

    def _step_common(self, batch, stage: str):
        x0, cond = batch  # both already standardized
        B = x0.size(0)
        device = x0.device
        
        t, sigma = self._sample_noise_level(B, device)      # t: [B], sigma: [B]
        sigma = sigma.view(B, 1)                            # [B,1] for broadcasting
        eps = torch.randn_like(x0)                          # [B, param_dim]
        x_t = x0 + sigma * eps                              # forward diffusion: x_t = x0 + sigma * eps
        
        x0_hat = self.model(x_t, cond, t)                   # predict clean params
        loss = self.crit(x0_hat, x0)                        # MSE loss on x0 prediction
        
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # Additional metrics for monitoring
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
        # Compute both single-step and generation metrics
        loss = self._step_common(batch, "test")
        
        # Generate samples and compute reconstruction metrics
        # Use 200 steps (best quality) for final evaluation
        with torch.no_grad():
            x0_gen = self.sample(cond, num_steps=500)
            
            # Metrics in standardized space
            mse = (x0_gen - x0).pow(2).mean()
            mae = (x0_gen - x0).abs().mean()
            
            self.log("test/mse_gen", mse, prog_bar=True, on_epoch=True)
            self.log("test/mae_gen", mae, prog_bar=True, on_epoch=True)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int = 500) -> torch.Tensor:
        """
        DDIM-style deterministic sampler with improved stability.
        
        Uses non-uniform time grid (more steps at high noise) and robust update formula.
        Based on Claim 2 from "Step-by-Step Diffusion" tutorial.
        
        Given standardized cond (log10 Kxx,Kyy,Kzz), generate standardized parameters.
        Uses the same linear noise schedule as training for consistency.
        
        Args:
            cond: [B, cond_dim] standardized permeability values
            num_steps: number of denoising steps
            
        Returns:
            x0: [B, param_dim] generated standardized parameters
        """
        self.eval()
        B = cond.size(0)
        device = cond.device
        param_dim = self.param_dim

        # Start from pure noise at the largest noise level
        x = torch.randn(B, param_dim, device=device) * self.sigma_max

        # Use non-uniform time grid: more steps at high noise (t near 1), fewer at low noise (t near 0)
        # This helps because model needs more refinement at high noise levels
        t_eps = 1e-4
        # Quadratic schedule: concentrates more steps at the beginning
        # This is more efficient than uniform spacing
        s = torch.linspace(0, 1, steps=num_steps, device=device)
        times = t_eps + (1.0 - t_eps) * (1 - s ** 2)  # Quadratic: more steps near t=1
        times = torch.flip(times, [0])  # Reverse to go from 1.0 to t_eps

        for i, cur_t in enumerate(times):
            # Current time scalar -> [B]
            t_batch = torch.full((B,), cur_t.item(), device=device)
            
            # Predict clean x0 at time cur_t: E[x_0 | x_t]
            x0_hat = self.model(x, cond, t_batch)  # [B, param_dim]
            
            if i == len(times) - 1:
                # Last step: use prediction directly
                x = x0_hat
                break
            
            # Get next time step
            next_t = times[i + 1].item()
            cur_t_scalar = cur_t.item()
            
            # Compute current and next sigma values (matching training schedule)
            if self.use_cosine_schedule:
                # Cosine schedule (matching training)
                cos_cur = math.cos(math.pi * cur_t_scalar / 2.0)
                cos_next = math.cos(math.pi * next_t / 2.0)
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_cur) ** 2
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_next) ** 2
            else:
                # Linear schedule (matching training)
                cur_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * cur_t_scalar
                next_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * next_t
            
            # Improved DDIM update: direct interpolation based on sigma ratio
            # This is equivalent to Claim 2 but more numerically stable
            
            if cur_sigma > 1e-5:
                # Standard DDIM update: x_{next} = x0_hat + (sigma_next/sigma_cur) * (x - x0_hat)
                # This moves toward x0_hat while scaling the residual by sigma ratio
                sigma_ratio = next_sigma / cur_sigma
                sigma_ratio = max(0.0, min(1.0, sigma_ratio))  # Ensure valid range
                
                # Update: blend between current state and predicted x0
                x = x0_hat + sigma_ratio * (x - x0_hat)
            else:
                # For very small sigma, we're essentially at clean data
                x = x0_hat

        return x  # still standardized; inverse-transform with scaler_x0 later

# ---------------------------
# Training / Testing wrappers
# ---------------------------

def build_inverse_model_and_callbacks(param_dim: int,
                                      cond_dim: int,
                                      outdir: str,
                                      learning_rate: float,
                                      patience: int):
    model = InverseDiffusionLightning(
        param_dim=param_dim,
        cond_dim=cond_dim,
        lr=learning_rate,
        weight_decay=1e-4,
        hidden=(512, 512, 256),
        activation="gelu",
        dropout=0.1,
        layer_norm=True,
        time_dim=32,
        sigma_min=0.01,
        sigma_max=1.0,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=outdir,
        filename="invdiff-{epoch:03d}-{val_loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=patience)
    return model, ckpt_cb, es_cb

def build_trainer(max_epochs: int, outdir: str, callbacks):
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
    trainer.fit(model, train_loader, val_loader)
    print("Best checkpoint:", ckpt_cb.best_model_path)
    return ckpt_cb.best_model_path

def test_current_and_best(trainer, model, test_loader, best_ckpt_path):
    print("\n== Test (current model) ==")
    trainer.test(model, dataloaders=test_loader)
    
    # print("\n== Test (best checkpoint) ==")
    best_model = InverseDiffusionLightning.load_from_checkpoint(best_ckpt_path)
    trainer.test(best_model, dataloaders=test_loader)

# ---------------------------
# main()
# ---------------------------

def main():
    set_seed(42)

    # Paths / splits / loader params
    csv_path = "./result_no_duplicates_behnaz.csv"
    val_size = 0.15
    test_size = 0.15
    seed = 42
    batch_size = 64
    num_workers = 15

    # Training config
    max_epochs = 500
    outdir = "./runs/perm_inverse_diffusion"
    os.makedirs(outdir, exist_ok=True)
    learning_rate = 1e-3
    patience = 20

    # Data
    df, param_cols, perm_cols = load_dataframe_inverse(csv_path)
    df_train, df_val, df_test = split_dataframe(df, val_size, test_size, seed)
    (x0_tr, cond_tr,
     x0_va, cond_va,
     x0_te, cond_te,
     scaler_x0, scaler_cond) = arrays_and_scalers_inverse(df_train, df_val, df_test,
                                                          param_cols, perm_cols)

    train_loader, val_loader, test_loader = make_inverse_loaders(
        x0_tr, cond_tr,
        x0_va, cond_va,
        x0_te, cond_te,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model + callbacks + trainer
    model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(
        param_dim=len(param_cols),
        cond_dim=len(perm_cols),
        outdir=outdir,
        learning_rate=learning_rate,
        patience=patience,
    )
    trainer = build_trainer(max_epochs=max_epochs, outdir=outdir, callbacks=[ckpt_cb, es_cb])

    # Train / Validate / Test
    best_ckpt = train_and_validate(trainer, model, train_loader, val_loader, ckpt_cb)
    test_current_and_best(trainer, model, test_loader, best_ckpt)

    # Save scalers for later sampling / inverse-transform
    # joblib.dump(scaler_x0, os.path.join(outdir, "scaler_x0.joblib"))
    # joblib.dump(scaler_cond, os.path.join(outdir, "scaler_cond.joblib"))
    # print("Saved scalers to:", outdir)

if __name__ == "__main__":
    main()


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

import torch.nn.functional as F  # add this near the top if not already there

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")

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
        
        # Normalize permeability by (2e-6)^2 before taking log10
        perms_lin = perms_lin / ((2e-6) ** 2)

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

class ResBlock(nn.Module):

    def __init__(self, dim: int, Act, layer_norm: bool = True, dropout: float = 0.0):

        super().__init__()

        layers = [nn.Linear(dim, dim)]

        if layer_norm:

            layers.append(nn.LayerNorm(dim))

        layers.append(Act())

        if dropout and dropout > 0:

            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Simple residual: x + f(x)

        return x + self.net(x)

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

        dropout: float = 0.0,

        layer_norm: bool = True,

        time_dim: int = 32,

        num_res_blocks: int = 2,   # new: how many residual blocks at the end

    ):

        super().__init__()

        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}

        Act = acts.get(activation, nn.GELU)

        self.time_emb = TimeEmbedding(dim=time_dim)

        in_dim = param_dim + cond_dim + time_dim

        # ---- Backbone: map input → last hidden dim ----

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

        # ---- Residual blocks at constant dimension (hidden[-1]) ----

        block_dim = hidden[-1]

        self.res_blocks = nn.ModuleList([

            ResBlock(block_dim, Act, layer_norm=layer_norm, dropout=dropout)

            for _ in range(num_res_blocks)

        ])

        # ---- Final output layer ----

        self.out = nn.Linear(block_dim, param_dim)

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

        # Backbone: basic MLP

        h = self.backbone(x)

        # Residual refinement

        for blk in self.res_blocks:

            h = blk(h)

        # Predict clean parameters

        return self.out(h)

class ForwardMLP(nn.Module):

    """

    Simple MLP: maps standardized parameters x0 -> standardized cond (log10 Kxx,Kyy,Kzz).

    """

    def __init__(

        self,

        param_dim: int,

        cond_dim: int = 3,

        hidden=(256, 256),

        activation: str = "gelu",

        dropout: float = 0.0,

        layer_norm: bool = True,

    ):

        super().__init__()

        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}

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

        layers.append(nn.Linear(prev, cond_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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

        weight_decay: float = 1e-5,

        sigma_min: float = 0.01,

        sigma_max: float = 0.8,

        use_cosine_schedule: bool = True,

        forward_model: nn.Module | None = None,   # <--- new

        lambda_forward: float = 0.0,             # <--- new

        **mlp_kwargs,

    ):

        super().__init__()

        self.save_hyperparameters(ignore=["forward_model"])  # don't try to pickle the whole model

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

        # Forward model (frozen)

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

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"}}

    def _sample_noise_level(self, batch_size: int, device: torch.device):

        u = torch.rand(batch_size, device=device)

        # Push mass towards t ~ 1 (high noise)

        t = 1.0 - (1.0 - 1e-4) * (u ** 2)  # or u**3

        if self.use_cosine_schedule:

            cos_val = torch.cos(math.pi * t / 2.0)

            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - cos_val) ** 2

        else:

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

        # --- Diffusion loss (noise-level-weighted MSE) ---

        base_loss = (x0_hat - x0).pow(2).mean(dim=1)        # [B]

        w = (sigma.view(-1) / self.sigma_max) ** 2 + 1e-3   # focus more on larger sigma

        diff_loss = (w * base_loss).mean()

        # --- Forward consistency loss (using frozen forward model) ---

        fwd_loss = torch.tensor(0.0, device=device)

        if self.forward_model is not None and self.lambda_forward > 0.0:

            # with torch.no_grad():

            pred_cond = self.forward_model(x0_hat)      # [B, cond_dim] in standardized cond space

            fwd_loss = F.mse_loss(pred_cond, cond)

            # You can log it for monitoring:

            self.log(f"{stage}/forward_loss", fwd_loss, prog_bar=False, on_epoch=True, on_step=False)

        loss = diff_loss + self.lambda_forward * fwd_loss

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        

        # Additional metrics for monitoring (on x0)

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

        # Single-step + diffusion/forward loss

        loss = self._step_common(batch, "test")

        

        # Generate samples and compute reconstruction + forward metrics

        with torch.no_grad():

            x0_gen = self.sample(cond, num_steps=500)

            # Metrics in standardized parameter space

            mse = (x0_gen - x0).pow(2).mean()

            mae = (x0_gen - x0).abs().mean()

            self.log("test/mse_gen", mse, prog_bar=True, on_epoch=True)

            self.log("test/mae_gen", mae, prog_bar=True, on_epoch=True)

            # Forward consistency for generated parameters (if forward model exists)

            if self.forward_model is not None:

                pred_cond_gen = self.forward_model(x0_gen)   # standardized cond

                fwd_gen_mse = F.mse_loss(pred_cond_gen, cond)

                fwd_gen_mae = (pred_cond_gen - cond).abs().mean()

                self.log("test/fwd_mse_gen", fwd_gen_mse, prog_bar=True, on_epoch=True)

                self.log("test/fwd_mae_gen", fwd_gen_mae, prog_bar=True, on_epoch=True)

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

        times = 1.0 - (1.0 - t_eps) * s**2   # Quadratic: more steps near t=1

        # times = torch.flip(times, [0])  # Reverse to go from 1.0 to t_eps

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

class ForwardLightning(pl.LightningModule):

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-5):

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

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"}}

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

def train_forward_model(

    train_loader,

    val_loader,

    param_dim: int,

    cond_dim: int,

    outdir: str,

    lr: float = 1e-3,

    weight_decay: float = 1e-5,

    max_epochs: int = 300,

    patience: int = 20,

):

    print("== Training forward model (parameters -> permeability) ==")

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

    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=patience)

    trainer = pl.Trainer(

        max_epochs=max_epochs,

        accelerator="auto",

        devices="auto",

        callbacks=[ckpt_cb, es_cb],

        log_every_n_steps=10,

        default_root_dir=fwd_outdir,

    )

    trainer.fit(fwd_lit, train_loader, val_loader)

    print("Best forward checkpoint:", ckpt_cb.best_model_path)

    # Load the best checkpoint and return the underlying nn.Module

    best_fwd = ForwardLightning.load_from_checkpoint(ckpt_cb.best_model_path, model=fwd_mlp)

    forward_model = best_fwd.model

    forward_model.eval()

    for p in forward_model.parameters():

        p.requires_grad = False  # freeze

    return forward_model

def build_inverse_model_and_callbacks(param_dim: int,

                                      cond_dim: int,

                                      outdir: str,

                                      learning_rate: float,

                                      patience: int,

                                      forward_model: nn.Module,

                                      lambda_forward: float = 0.1):

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

        forward_model=forward_model,         # <--- pass it in

        lambda_forward=lambda_forward,       # <--- weight for forward-consistency

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

def generate_structure_from_params(params_dict, size=(250, 250)):
    """
    Generate material structure image from physical parameters.
    
    This is a placeholder function - you need to implement your actual structure generation logic here.
    The function should take a dictionary of parameters and return a 2D numpy array representing the structure.
    
    Args:
        params_dict: Dictionary mapping parameter names to values (in original scale)
        size: Tuple of (height, width) for the output structure image
    
    Returns:
        structure: 2D numpy array of shape (height, width) representing the material structure
                  Values should be in [0, 1] range (0 = one material, 1 = other material)
    """
    # TODO: Replace this with your actual structure generation code
    # This is a placeholder that creates a simple pattern based on parameters
    
    # Example: You might have parameters like:
    # - structure_type: 'blobs', 'checkerboard', 'cylinders', etc.
    # - porosity, pore_size, etc.
    
    # For now, create a simple random pattern as placeholder
    # You should replace this with your actual structure generation function
    structure = np.random.rand(size[0], size[1])
    
    # If you have a function that generates structures, call it here:
    # structure = your_structure_generator(params_dict, size)
    
    return structure

def plot_structure_comparisons(model, test_loader, scaler_x0, param_cols, outdir, model_name="best", max_samples=20):
    """
    Create side-by-side comparisons of actual vs predicted material structures.
    
    Args:
        model: Trained InverseDiffusionLightning model
        test_loader: DataLoader for test set
        scaler_x0: StandardScaler for parameters (to inverse transform)
        param_cols: List of parameter column names
        outdir: Output directory to save plots
        model_name: Name identifier for the model (for filename)
        max_samples: Maximum number of samples to visualize
    """
    if not HAS_MATPLOTLIB:
        print("Skipping structure plotting: matplotlib is not installed.")
        return
    
    model.eval()
    all_x0_actual = []
    all_x0_pred = []
    all_cond = []
    
    with torch.no_grad():
        for batch in test_loader:
            x0, cond = batch
            x0_gen = model.sample(cond, num_steps=500)
            
            all_x0_actual.append(x0.cpu().numpy())
            all_x0_pred.append(x0_gen.cpu().numpy())
            all_cond.append(cond.cpu().numpy())
    
    # Concatenate all batches
    x0_actual = np.concatenate(all_x0_actual, axis=0)
    x0_pred = np.concatenate(all_x0_pred, axis=0)
    cond_all = np.concatenate(all_cond, axis=0)
    
    # Inverse transform to original scale
    x0_actual_orig = scaler_x0.inverse_transform(x0_actual)
    x0_pred_orig = scaler_x0.inverse_transform(x0_pred)
    
    # Limit number of samples----------------------------------------------
    n_samples = 10000
    #------------------------------------------------------------------
    # Create output directory
    structure_plots_dir = os.path.join(outdir, f'structure_comparisons_{model_name}')
    os.makedirs(structure_plots_dir, exist_ok=True)
    
    print(f"\nGenerating structure comparison plots (showing {n_samples} samples)...")
    
    # Create individual structure comparison plots
    for sample_idx in range(n_samples):
        actual_params = x0_actual_orig[sample_idx]
        pred_params = x0_pred_orig[sample_idx]
        perm_values = cond_all[sample_idx]
        
        # Convert parameters to dictionary
        actual_params_dict = {name: val for name, val in zip(param_cols, actual_params)}
        pred_params_dict = {name: val for name, val in zip(param_cols, pred_params)}
        
        # Generate structures
        actual_structure = generate_structure_from_params(actual_params_dict)
        pred_structure = generate_structure_from_params(pred_params_dict)
        
        # Create side-by-side comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Actual structure
        im1 = axes[0].imshow(actual_structure, cmap='viridis', origin='lower', extent=[0, 250, 0, 250])
        axes[0].set_title('Actual Structure', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X', fontsize=10)
        axes[0].set_ylabel('Y', fontsize=10)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Predicted structure
        im2 = axes[1].imshow(pred_structure, cmap='viridis', origin='lower', extent=[0, 250, 0, 250])
        axes[1].set_title('Predicted Structure', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X', fontsize=10)
        axes[1].set_ylabel('Y', fontsize=10)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Add permeability info
        perm_info = f'Permeability (std): Kxx={perm_values[0]:.3f}, Kyy={perm_values[1]:.3f}, Kzz={perm_values[2]:.3f}'
        fig.suptitle(f'Sample {sample_idx + 1}: Actual vs Predicted Structure\n{perm_info}', 
                     fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(structure_plots_dir, f'structure_comparison_{sample_idx + 1:04d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (sample_idx + 1) % 5 == 0:
            print(f"  Generated {sample_idx + 1}/{n_samples} structure comparisons...")
    
    print(f"Saved {n_samples} structure comparison plots to: {structure_plots_dir}")
    
    # Create a grid summary plot
    n_cols = 4
    n_rows = (n_samples + 1) // 2  # 2 structures per sample (actual + predicted)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for sample_idx in range(min(n_samples, len(axes) // 2)):
        actual_params = x0_actual_orig[sample_idx]
        pred_params = x0_pred_orig[sample_idx]
        
        actual_params_dict = {name: val for name, val in zip(param_cols, actual_params)}
        pred_params_dict = {name: val for name, val in zip(param_cols, pred_params)}
        
        actual_structure = generate_structure_from_params(actual_params_dict)
        pred_structure = generate_structure_from_params(pred_params_dict)
        
        # Actual
        ax_actual = axes[sample_idx * 2]
        ax_actual.imshow(actual_structure, cmap='viridis', origin='lower', extent=[0, 250, 0, 250])
        ax_actual.set_title(f'Sample {sample_idx + 1}\nActual', fontsize=9, fontweight='bold')
        ax_actual.axis('off')
        
        # Predicted
        ax_pred = axes[sample_idx * 2 + 1]
        ax_pred.imshow(pred_structure, cmap='viridis', origin='lower', extent=[0, 250, 0, 250])
        ax_pred.set_title(f'Sample {sample_idx + 1}\nPredicted', fontsize=9, fontweight='bold')
        ax_pred.axis('off')
    
    # Hide unused subplots
    for i in range(min(n_samples * 2, len(axes)), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Structure Comparisons: Actual vs Predicted', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    summary_path = os.path.join(outdir, f'structure_comparisons_summary_{model_name}.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved structure comparison summary to: {summary_path}")
    plt.close()

def plot_individual_samples(model, test_loader, scaler_x0, param_cols, outdir, model_name="best", max_samples=50):
    """
    Create individual comparison plots for each test sample, showing all parameters
    (actual vs predicted) side-by-side.
    
    Args:
        model: Trained InverseDiffusionLightning model
        test_loader: DataLoader for test set
        scaler_x0: StandardScaler for parameters (to inverse transform)
        param_cols: List of parameter column names
        outdir: Output directory to save plots
        model_name: Name identifier for the model (for filename)
        max_samples: Maximum number of samples to plot (to avoid too many files)
    """
    if not HAS_MATPLOTLIB:
        print("Skipping individual sample plotting: matplotlib is not installed.")
        return
    
    model.eval()
    all_x0_actual = []
    all_x0_pred = []
    all_cond = []
    
    with torch.no_grad():
        for batch in test_loader:
            x0, cond = batch
            x0_gen = model.sample(cond, num_steps=500)
            
            all_x0_actual.append(x0.cpu().numpy())
            all_x0_pred.append(x0_gen.cpu().numpy())
            all_cond.append(cond.cpu().numpy())
    
    # Concatenate all batches
    x0_actual = np.concatenate(all_x0_actual, axis=0)
    x0_pred = np.concatenate(all_x0_pred, axis=0)
    cond_all = np.concatenate(all_cond, axis=0)
    
    # Inverse transform to original scale
    x0_actual_orig = scaler_x0.inverse_transform(x0_actual)
    x0_pred_orig = scaler_x0.inverse_transform(x0_pred)
    
    # Limit number of samples to plot
    n_samples = min(len(x0_actual_orig), max_samples)
    
    # Create output directory for individual plots
    individual_plots_dir = os.path.join(outdir, f'individual_samples_{model_name}')
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    print(f"\nGenerating individual sample plots (showing {n_samples} samples)...")
    
    for sample_idx in range(n_samples):
        actual_params = x0_actual_orig[sample_idx]
        pred_params = x0_pred_orig[sample_idx]
        perm_values = cond_all[sample_idx]
        
        # Create bar chart comparison
        fig, ax = plt.subplots(figsize=(max(12, len(param_cols) * 0.5), 6))
        
        x_pos = np.arange(len(param_cols))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, actual_params, width, label='Actual', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x_pos + width/2, pred_params, width, label='Predicted', alpha=0.8, color='coral')
        
        ax.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parameter Values', fontsize=12, fontweight='bold')
        ax.set_title(f'Sample {sample_idx + 1}: Actual vs Predicted Parameters\n'
                    f'Permeability (std): Kxx={perm_values[0]:.3f}, Kyy={perm_values[1]:.3f}, Kzz={perm_values[2]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_cols, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            # Only label if bars are tall enough
            if abs(height1) > (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02:
                ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                       f'{height1:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
            if abs(height2) > (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02:
                ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                       f'{height2:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        plt.tight_layout()
        plot_path = os.path.join(individual_plots_dir, f'sample_{sample_idx + 1:04d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (sample_idx + 1) % 10 == 0:
            print(f"  Generated {sample_idx + 1}/{n_samples} plots...")
    
    print(f"Saved {n_samples} individual sample plots to: {individual_plots_dir}")
    
    # Also create a summary plot showing a few representative samples
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Select 6 representative samples (first, last, and evenly spaced)
    sample_indices = [0, n_samples//5, 2*n_samples//5, 3*n_samples//5, 4*n_samples//5, n_samples-1]
    sample_indices = [i for i in sample_indices if i < n_samples][:6]
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        actual_params = x0_actual_orig[sample_idx]
        pred_params = x0_pred_orig[sample_idx]
        
        x_pos = np.arange(len(param_cols))
        width = 0.35
        
        ax.bar(x_pos - width/2, actual_params, width, label='Actual', alpha=0.8, color='steelblue')
        ax.bar(x_pos + width/2, pred_params, width, label='Predicted', alpha=0.8, color='coral')
        
        ax.set_title(f'Sample {sample_idx + 1}', fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_cols, rotation=45, ha='right', fontsize=7)
        if plot_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for plot_idx in range(len(sample_indices), len(axes)):
        axes[plot_idx].axis('off')
    
    plt.suptitle('Representative Samples: Actual vs Predicted Parameters', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    summary_path = os.path.join(outdir, f'individual_samples_summary_{model_name}.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to: {summary_path}")
    plt.close()

def test_current_and_best(trainer, model, test_loader, best_ckpt_path, forward_model, scaler_x0, param_cols, outdir):

    print("\n== Test (current model) ==")

    trainer.test(model, dataloaders=test_loader)

    

    print("\n== Test (best checkpoint) ==")

    best_model = InverseDiffusionLightning.load_from_checkpoint(

        best_ckpt_path,

        forward_model=forward_model,

    )

    trainer.test(best_model, dataloaders=test_loader)
    
    # Create structure comparison plots (actual vs predicted structures)
    print("\n== Generating structure comparison plots ==")
    plot_structure_comparisons(best_model, test_loader, scaler_x0, param_cols, outdir, model_name="best", max_samples=20)

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

    # learning_rate = 1e-3

    learning_rate = 3e-4

    patience = 40

    # Data

    df, param_cols, perm_cols = load_dataframe_inverse(csv_path)

    df_train, df_val, df_test = split_dataframe(df, val_size, test_size, seed)

    (x0_tr, cond_tr,

     x0_va, cond_va,

     x0_te, cond_te,

     scaler_x0, scaler_cond) = arrays_and_scalers_inverse(df_train, df_val, df_test,

                                                          param_cols, perm_cols)

    # --- Debug: log10(K) stats on the whole dataset ---

    

    print("scaler_cond.mean_ (log10 K):", scaler_cond.mean_)

    print("scaler_cond.scale_ (std-dev of log10 K):", scaler_cond.scale_)

    train_loader, val_loader, test_loader = make_inverse_loaders(

        x0_tr, cond_tr,

        x0_va, cond_va,

        x0_te, cond_te,

        batch_size=batch_size,

        num_workers=num_workers,

    )

    # 1) Train forward model (parameters -> permeability)

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

    # Model + callbacks + trainer

    model, ckpt_cb, es_cb = build_inverse_model_and_callbacks(

        param_dim=len(param_cols),

        cond_dim=len(perm_cols),

        outdir=outdir,

        learning_rate=learning_rate,

        patience=patience,

        forward_model=forward_model,         # <-- from above

        # Has been changed

        lambda_forward=0.0,                  # Disabled by default (set to 0.01-0.5 to enable forward consistency loss)

    )

    

    trainer = build_trainer(max_epochs=max_epochs, outdir=outdir, callbacks=[ckpt_cb, es_cb])

    # Train / Validate / Test

    best_ckpt = train_and_validate(trainer, model, train_loader, val_loader, ckpt_cb)

    test_current_and_best(trainer, model, test_loader, best_ckpt, forward_model, scaler_x0, param_cols, outdir)

if __name__ == "__main__":

    main()


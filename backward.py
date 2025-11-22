import os
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
from sklearn.metrics import r2_score


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed=42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Data prep helpers
# ---------------------------

def load_dataframe(csv_path: str):
    df = pd.read_csv(csv_path)
    exclude = {"id", "seed", "Kxx", "Kyy", "Kzz"}
    param_cols = [c for c in df.columns if c not in exclude]
    cond_cols = ["Kxx", "Kyy", "Kzz"]

    assert set(cond_cols).issubset(df.columns), "CSV must have Kxx, Kyy, Kzz."
    print("Num rows:", len(df))
    print("Parameter columns (to generate):", param_cols)
    print("Conditioning columns:", cond_cols)

    return df, param_cols, cond_cols


def split_dataframe(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    df_remain, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(df_remain, test_size=val_rel, random_state=seed)
    return df_train, df_val, df_test


def to_xy_inverse(d: pd.DataFrame, param_cols, cond_cols):
    """
    For inverse problem:
    X = parameters (to be generated)
    y = permeability (conditioning)
    """
    X = d[param_cols].astype(np.float32).values
    
    y_lin = d[cond_cols].astype(np.float64).values
    nan_rows = np.isnan(y_lin).any(axis=1)
    if np.any(nan_rows):
        print(f"Warning: Dropping {np.sum(nan_rows)} rows with NaN permeability.")
        y_lin = y_lin[~nan_rows]
        X = X[~nan_rows]
    
    y_lin = np.clip(y_lin, 1e-30, None)
    y_log = np.log10(y_lin).astype(np.float32)
    
    return X, y_log


def arrays_and_scalers(df_train, df_val, df_test, param_cols, cond_cols):
    Xtr, Ctr = to_xy_inverse(df_train, param_cols, cond_cols)
    Xva, Cva = to_xy_inverse(df_val, param_cols, cond_cols)
    Xte, Cte = to_xy_inverse(df_test, param_cols, cond_cols)

    # Separate scalers for parameters and permeability
    x_scaler = StandardScaler().fit(Xtr)
    c_scaler = StandardScaler().fit(Ctr)
    
    Xtr = x_scaler.transform(Xtr).astype(np.float32)
    Xva = x_scaler.transform(Xva).astype(np.float32)
    Xte = x_scaler.transform(Xte).astype(np.float32)
    
    Ctr = c_scaler.transform(Ctr).astype(np.float32)
    Cva = c_scaler.transform(Cva).astype(np.float32)
    Cte = c_scaler.transform(Cte).astype(np.float32)
    
    return (Xtr, Ctr, Xva, Cva, Xte, Cte, x_scaler, c_scaler)


def make_loaders(Xtr, Ctr, Xva, Cva, Xte, Cte, batch_size, num_workers):
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ctr))
    val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Cva))
    test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Cte))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------------------------
# Diffusion Model Components
# ---------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalNoisePredictor(nn.Module):
    """
    Predicts noise eps given:
      x_t : noised parameters
      c   : conditioning (log permeability)
      t   : timestep
    """
    def __init__(self, x_dim: int, c_dim: int, time_dim: int = 64, 
                 hidden=(256, 256, 128), activation="gelu", dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        Act = acts.get(activation, nn.GELU)
        
        in_dim = x_dim + c_dim + time_dim
        layers = []
        prev = in_dim
        
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        layers.append(nn.Linear(prev, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, c, t):
        t_emb = self.time_mlp(t)
        h = torch.cat([x_t, c, t_emb], dim=-1)
        return self.net(h)


class DiffusionSchedule(nn.Module):
    """Manages the noise schedule for diffusion"""
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.T = T
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        # Register as buffers so they move to correct device automatically
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))
        
        # For reverse process
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar)
        self.register_buffer('posterior_variance', posterior_variance)


# ---------------------------
# Lightning Module
# ---------------------------

class LightningDiffusion(pl.LightningModule):
    def __init__(self, x_dim, c_dim, T=1000, lr=1e-3, weight_decay=1e-4, 
                 beta_start=1e-4, beta_end=2e-2, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ConditionalNoisePredictor(x_dim=x_dim, c_dim=c_dim, **model_kwargs)
        self.schedule = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end)
        self.T = T
        
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar = self.schedule.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def p_sample(self, x_t, c, t):
        """Single reverse diffusion step"""
        betas = self.schedule.betas
        sqrt_recip_alphas = self.schedule.sqrt_recip_alphas
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_bar
        posterior_variance = self.schedule.posterior_variance
        
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        eps_pred = self.model(x_t, c, t_tensor)
        
        # Compute mean
        mean = sqrt_recip_alphas[t] * (
            x_t - betas[t] * eps_pred / sqrt_one_minus_alpha_bar[t]
        )
        
        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_variance[t]) * noise
    
    @torch.no_grad()
    def sample(self, c, n_samples=None):
        """Generate samples given conditioning c"""
        self.eval()
        
        if n_samples is None:
            n_samples = c.shape[0]
        
        x_dim = self.hparams.x_dim
        device = c.device
        
        # Start from pure noise
        x = torch.randn(n_samples, x_dim, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.T)):
            x = self.p_sample(x, c, t)
        
        return x
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), 
                               lr=self.hparams.lr, 
                               weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", 
                                                         patience=10, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/loss"}}
    
    def step_common(self, batch, stage):
        x0, c = batch
        B = x0.size(0)
        
        # Random timestep for each sample
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        
        # Forward diffusion
        x_t, noise = self.q_sample(x0, t)
        
        # Predict noise
        eps_pred = self.model(x_t, c, t)
        
        # MSE loss on noise prediction
        loss = F.mse_loss(eps_pred, noise)
        
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step_common(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        self.step_common(batch, "val")
    
    def test_step(self, batch, batch_idx):
        x0, c = batch
        loss = self.step_common(batch, "test")
        
        # Generate samples and compute reconstruction metrics
        with torch.no_grad():
            x_gen = self.sample(c)
            
            # MSE in parameter space
            mse_params = F.mse_loss(x_gen, x0)
            mae_params = (x_gen - x0).abs().mean()
            
            self.log("test/mse_params", mse_params, prog_bar=True, on_epoch=True)
            self.log("test/mae_params", mae_params, prog_bar=True, on_epoch=True)


# ---------------------------
# Training functions
# ---------------------------

def build_model_and_callbacks(x_dim, c_dim, outdir, learning_rate, patience, T=1000):
    model = LightningDiffusion(
        x_dim=x_dim,
        c_dim=c_dim,
        T=T,
        lr=learning_rate,
        weight_decay=1e-4,
        hidden=(256, 256, 128),
        activation="gelu",
        dropout=0.1
    )
    
    ckpt_cb = ModelCheckpoint(
        dirpath=outdir,
        filename="diffusion-{epoch:03d}-{val_loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=patience)
    
    return model, ckpt_cb, es_cb


def build_trainer(max_epochs, outdir, callbacks):
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

    print("\n== Test (best checkpoint) ==")
    best_model = LightningDiffusion.load_from_checkpoint(best_ckpt_path)
    trainer.test(best_model, dataloaders=test_loader)


# ---------------------------
# Main
# ---------------------------

def main():
    set_seed(42)

    # Paths and config
    csv_path = "./result_no_duplicates_behnaz.csv"
    val_size = 0.15
    test_size = 0.15
    seed = 42
    batch_size = 128
    num_workers = 0

    # Training config
    max_epochs = 500
    T = 1000  # diffusion timesteps
    outdir = "./runs/inverse_diffusion"
    os.makedirs(outdir, exist_ok=True)
    learning_rate = 1e-3
    patience = 30

    # Data
    df, param_cols, cond_cols = load_dataframe(csv_path)
    df_train, df_val, df_test = split_dataframe(df, val_size, test_size, seed)
    Xtr, Ctr, Xva, Cva, Xte, Cte, x_scaler, c_scaler = arrays_and_scalers(
        df_train, df_val, df_test, param_cols, cond_cols
    )
    train_loader, val_loader, test_loader = make_loaders(
        Xtr, Ctr, Xva, Cva, Xte, Cte, batch_size, num_workers
    )

    # Model + callbacks + trainer
    x_dim = Xtr.shape[1]
    c_dim = Ctr.shape[1]
    
    model, ckpt_cb, es_cb = build_model_and_callbacks(
        x_dim=x_dim,
        c_dim=c_dim,
        outdir=outdir,
        learning_rate=learning_rate,
        patience=patience,
        T=T
    )
    
    trainer = build_trainer(max_epochs=max_epochs, outdir=outdir, 
                          callbacks=[ckpt_cb, es_cb])

    # Train / Validate / Test
    best_ckpt = train_and_validate(trainer, model, train_loader, val_loader, ckpt_cb)
    test_current_and_best(trainer, model, test_loader, best_ckpt)
    
    print("\n== Training Complete ==")
    print(f"Best model saved at: {best_ckpt}")
    print(f"Parameter scaler saved in memory: x_scaler")
    print(f"Conditioning scaler saved in memory: c_scaler")


if __name__ == "__main__":
    main()
# %%
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ---------------------------
# Paths
# ---------------------------
LATENT_NPZ      = r"./GAUSSIAN_GROUP_VAE/group_tokens.npz"
VAE_CKPT_PATH   = r"./GAUSSIAN_GROUP_VAE/gaussian_group_vae_best.pt"
WORLD_STATS_NPZ = r"./GAUSSIAN_GROUP_VAE/world_stats.npz"

OUT_DIR = r"./GAUSSIAN_LATENT_DYNAMICS_REFINER"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DYN = os.path.join(OUT_DIR, "latent_dynamics.pt")
CKPT_REF = os.path.join(OUT_DIR, "diffusion_refiner.pt")

# ---------------------------
# Core dims
# ---------------------------
GROUP_SIZE = 32
LATENT_DIM = 128
GAUSS_DIM = 11
D_MODEL = 256
STATE_DIM = 128

# ---------------------------
# Frame padding
# ---------------------------
MAX_PATCHES_PER_FRAME = 128
MIN_PATCHES_PER_FRAME = 4

# ---------------------------
# Sequence setup
# ---------------------------
SEQ_FRAMES = 8           # prompt window
FUTURE_HORIZON = 4        # non-AR future chunk for dynamics training
REFINE_T_MAX = 80         # low-noise residual refine
BATCH_SIZE = 2
NUM_WORKERS = 0

# ---------------------------
# Training
# ---------------------------
EPOCHS_DYN = 3
EPOCHS_REF = 8
LR_DYN = 2e-4
LR_REF = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LOG_EVERY = 50

# ---------------------------
# Diffusion schedule
# ---------------------------
T_STEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2

# ---------------------------
# Inference
# ---------------------------
ROLL_PROMPT_START = 0
ROLL_OUT_STEPS = 1000
ROLL_REFINE_EVERY = 4      # refine every N states, not every step
ONE_STEP_T_EVAL = 20
ONE_STEP_INIT_NOISE_STD = 0.01
DELTA_CLIP_INFER = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %%
z = np.load(LATENT_NPZ)
required = ["latents", "frame_ids", "pos_ids", "mask"]
for k in required:
    if k not in z.files:
        raise RuntimeError(f"{LATENT_NPZ} missing required key: {k}")

latents = z["latents"].astype(np.float32)
frame_ids = z["frame_ids"].astype(np.int32)
pos_ids = z["pos_ids"].astype(np.int64)
group_mask = z["mask"].astype(np.float32)

valid_slot_mask = group_mask > 0.5
valid_latents = latents[valid_slot_mask]
lat_mean = valid_latents.mean(axis=0).astype(np.float32)
lat_std = valid_latents.std(axis=0).astype(np.float32)
lat_std = np.maximum(lat_std, 1e-4)

lat_norm_3d = np.zeros_like(latents, dtype=np.float32)
lat_norm_3d[valid_slot_mask] = (latents[valid_slot_mask] - lat_mean[None, :]) / lat_std[None, :]

frame_to_items = {}
for i, fid in enumerate(frame_ids):
    frame_to_items.setdefault(int(fid), []).append(i)

frame_keys = sorted(frame_to_items.keys())
frame_lat_list, frame_pos_list = [], []
frame_patch_valid_list, frame_slot_valid_list, frame_id_list = [], [], []

for fid in frame_keys:
    idxs = sorted(frame_to_items[fid], key=lambda ii: int(pos_ids[ii]))
    if len(idxs) < MIN_PATCHES_PER_FRAME:
        continue
    idxs = idxs[:MAX_PATCHES_PER_FRAME]
    n = len(idxs)

    arr_lat = np.zeros((MAX_PATCHES_PER_FRAME, GROUP_SIZE, LATENT_DIM), dtype=np.float32)
    arr_pos = np.zeros((MAX_PATCHES_PER_FRAME,), dtype=np.int64)
    arr_patch_valid = np.zeros((MAX_PATCHES_PER_FRAME,), dtype=np.float32)
    arr_slot_valid = np.zeros((MAX_PATCHES_PER_FRAME, GROUP_SIZE), dtype=np.float32)

    arr_lat[:n] = lat_norm_3d[idxs]
    arr_pos[:n] = pos_ids[idxs]
    arr_patch_valid[:n] = 1.0
    arr_slot_valid[:n] = group_mask[idxs]

    frame_lat_list.append(arr_lat)
    frame_pos_list.append(arr_pos)
    frame_patch_valid_list.append(arr_patch_valid)
    frame_slot_valid_list.append(arr_slot_valid)
    frame_id_list.append(int(fid))

frame_lat_all = np.asarray(frame_lat_list, dtype=np.float32)
frame_pos_all = np.asarray(frame_pos_list, dtype=np.int64)
frame_patch_valid_all = np.asarray(frame_patch_valid_list, dtype=np.float32)
frame_slot_valid_all = np.asarray(frame_slot_valid_list, dtype=np.float32)
frame_id_all = np.asarray(frame_id_list, dtype=np.int32)

print("frame_lat_all:", frame_lat_all.shape)
print("frame_pos_all:", frame_pos_all.shape)
print("frame_patch_valid_all:", frame_patch_valid_all.shape)
print("frame_slot_valid_all:", frame_slot_valid_all.shape)
print("frame_id_all:", frame_id_all.shape)


class SequenceChunkDataset(Dataset):
    """
    Input:
      history window  [t-K+1 ... t]
    Targets:
      future chunk    [t+1 ... t+H]

    This is non-AR chunk training for dynamics.
    """
    def __init__(
        self,
        frame_lat_all,
        frame_pos_all,
        frame_patch_valid_all,
        frame_slot_valid_all,
        frame_id_all,
        seq_frames=16,
        future_horizon=8,
    ):
        self.x = frame_lat_all
        self.pos = frame_pos_all
        self.pv = frame_patch_valid_all
        self.sv = frame_slot_valid_all
        self.fid = frame_id_all
        self.K = int(seq_frames)
        self.H = int(future_horizon)

        valid_starts = []
        need = self.K + self.H
        for s in range(self.x.shape[0] - need + 1):
            ids = self.fid[s:s + need]
            if np.all(np.diff(ids) > 0):
                valid_starts.append(s)
        if len(valid_starts) == 0:
            raise RuntimeError("No valid sequence chunks found.")
        self.valid_starts = np.asarray(valid_starts, dtype=np.int64)

    def __len__(self):
        return int(self.valid_starts.shape[0])

    def __getitem__(self, idx):
        s = int(self.valid_starts[idx])
        k = s + self.K
        h = k + self.H
        return (
            torch.from_numpy(self.x[s:k]),     # (K,P,32,128)
            torch.from_numpy(self.pos[s:k]),   # (K,P)
            torch.from_numpy(self.pv[s:k]),    # (K,P)
            torch.from_numpy(self.sv[s:k]),    # (K,P,32)
            torch.from_numpy(self.x[k:h]),     # (H,P,32,128)
            torch.from_numpy(self.pos[k:h]),   # (H,P)
            torch.from_numpy(self.pv[k:h]),    # (H,P)
            torch.from_numpy(self.sv[k:h]),    # (H,P,32)
        )

ds = SequenceChunkDataset(
    frame_lat_all, frame_pos_all, frame_patch_valid_all, frame_slot_valid_all, frame_id_all,
    seq_frames=SEQ_FRAMES, future_horizon=FUTURE_HORIZON
)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
print("sequence chunks:", len(ds))

# %%
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PosEmbedding3D(nn.Module):
    def __init__(self, d_model, vocab=1024):
        super().__init__()
        self.ex = nn.Embedding(vocab, d_model)
        self.ey = nn.Embedding(vocab, d_model)
        self.ez = nn.Embedding(vocab, d_model)

    def forward(self, pos_ids):
        ix = (pos_ids & 1023).long()
        iy = ((pos_ids >> 10) & 1023).long()
        iz = ((pos_ids >> 20) & 1023).long()
        return self.ex(ix) + self.ey(iy) + self.ez(iz)


class SlotDenoiseBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class DiffusionRefiner(nn.Module):
    """
    Refine residual on top of coarse latent.
    """
    def __init__(self, latent_dim=LATENT_DIM, d_model=D_MODEL, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(latent_dim, d_model)
        self.ctx_proj = nn.Linear(latent_dim, d_model)
        self.ctx_patch_proj = nn.Linear(latent_dim, d_model)

        self.pos_emb = PosEmbedding3D(d_model)
        self.slot_emb = nn.Embedding(GROUP_SIZE, d_model)

        self.valid_slot_emb = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.empty_slot_emb = nn.Parameter(torch.zeros(1, 1, 1, d_model))

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            SlotDenoiseBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x_t, pos_ids, patch_valid_mask, slot_valid_mask, t, ctx_latent=None, ctx_slot_valid_mask=None):
        B, P, S, D = x_t.shape
        h = self.in_proj(x_t)

        if ctx_latent is not None:
            if ctx_slot_valid_mask is None:
                ctx_slot_valid_mask = slot_valid_mask

            ctx_latent = ctx_latent * ctx_slot_valid_mask.unsqueeze(-1)
            h = h + self.ctx_proj(ctx_latent)

            denom = ctx_slot_valid_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
            ctx_patch = (ctx_latent * ctx_slot_valid_mask.unsqueeze(-1)).sum(dim=2) / denom
            h = h + self.ctx_patch_proj(ctx_patch).unsqueeze(2)

        h = h + self.pos_emb(pos_ids).unsqueeze(2)

        slot_ids = torch.arange(GROUP_SIZE, device=x_t.device).long()
        h = h + self.slot_emb(slot_ids).view(1, 1, GROUP_SIZE, self.d_model)

        slot_valid_bool = slot_valid_mask > 0.5
        h = h + torch.where(
            slot_valid_bool.unsqueeze(-1),
            self.valid_slot_emb.expand(B, P, S, -1),
            self.empty_slot_emb.expand(B, P, S, -1),
        )

        t_emb = self.t_mlp(timestep_embedding(t, self.d_model)).view(B, 1, 1, self.d_model)
        h = h + t_emb

        h = h.reshape(B * P, S, self.d_model)
        kpm = (~slot_valid_bool).reshape(B * P, S)
        all_empty = kpm.all(dim=1)
        if all_empty.any():
            kpm = kpm.clone()
            kpm[all_empty, 0] = False

        for blk in self.blocks:
            h = blk(h, key_padding_mask=kpm)

        eps = self.out_proj(self.ln_f(h)).reshape(B, P, S, D)
        eps = eps * patch_valid_mask.unsqueeze(-1).unsqueeze(-1)
        eps = eps * slot_valid_mask.unsqueeze(-1)
        return eps


class FrameStateEncoder(nn.Module):
    """
    (B,P,32,128) -> (B,P,STATE_DIM)
    Patch-level scene state encoder.
    """
    def __init__(self, latent_dim=LATENT_DIM, state_dim=STATE_DIM):
        super().__init__()
        self.patch_mlp = nn.Sequential(
            nn.Linear(latent_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, x_frame, patch_valid, slot_valid):
        # x_frame: (B,P,32,128)
        mask4 = slot_valid.unsqueeze(-1)
        slot_denom = slot_valid.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        patch_feat = (x_frame * mask4).sum(dim=2) / slot_denom      # (B,P,128)
        patch_feat = self.patch_mlp(patch_feat)                     # (B,P,STATE)
        patch_feat = patch_feat * patch_valid.unsqueeze(-1)
        return patch_feat


class LatentDynamicsTransformer(nn.Module):
    """
    Use:
      - short-term history window h_hist
      - persistent scene memory scene_mem
    to predict future patch-state chunk.

    Input:
      h_hist:    (B,K,P,STATE)
      pos_hist:  (B,K,P)
      scene_mem: (B,P,STATE)

    Output:
      h_future:  (B,H,P,STATE)
    """
    def __init__(self, state_dim=STATE_DIM, seq_frames=SEQ_FRAMES, future_horizon=FUTURE_HORIZON, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.seq_frames = seq_frames
        self.future_horizon = future_horizon

        self.in_proj = nn.Linear(state_dim, state_dim)
        self.mem_proj = nn.Linear(state_dim, state_dim)
        self.time_emb = nn.Parameter(torch.randn(1, seq_frames, 1, state_dim) * 0.02)
        self.pos_emb = PosEmbedding3D(state_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=n_heads,
            dim_feedforward=4 * state_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(2 * state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Linear(2 * state_dim, future_horizon * state_dim),
        )

    def forward(self, h_hist, pos_hist, scene_mem):
        B, K, P, D = h_hist.shape

        spatial_emb = self.pos_emb(pos_hist)               # (B,K,P,D)
        mem_bias = self.mem_proj(scene_mem).unsqueeze(1)   # (B,1,P,D)

        x = self.in_proj(h_hist) + self.time_emb[:, :K] + spatial_emb + mem_bias
        x = x.view(B, K * P, D)
        x = self.encoder(x)
        x = x.view(B, K, P, D)

        last_feat = x[:, -1]                               # (B,P,D)
        fused = torch.cat([last_feat, scene_mem], dim=-1)  # (B,P,2D)

        out = self.head(fused)                             # (B,P,H*D)
        out = out.view(B, P, self.future_horizon, D).transpose(1, 2)  # (B,H,P,D)
        return out


class FrameStateToCoarseLatent(nn.Module):
    """
    (B,P,STATE) + scene memory -> coarse latent (B,P,32,128)
    """
    def __init__(self, state_dim=STATE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.state_to_slot = nn.Sequential(
            nn.Linear(2 * state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Linear(2 * state_dim, latent_dim),
        )
        self.patch_gate = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, 1),
        )
        self.pos_proj = nn.Linear(D_MODEL, latent_dim)
        self.pos_emb = PosEmbedding3D(D_MODEL)

    def forward(self, h_patch, scene_mem, pos_ids, patch_valid, slot_valid):
        # h_patch:   (B,P,STATE)
        # scene_mem: (B,P,STATE)
        B, P = pos_ids.shape

        fused = torch.cat([h_patch, scene_mem], dim=-1)         # (B,P,2*STATE)
        slot_base = self.state_to_slot(fused)                   # (B,P,LATENT)
        slot_base = slot_base.unsqueeze(2).expand(B, P, GROUP_SIZE, LATENT_DIM)

        pos_h = self.pos_proj(self.pos_emb(pos_ids)).unsqueeze(2)  # (B,P,1,LATENT)
        gate = torch.sigmoid(self.patch_gate(fused)).unsqueeze(2)  # (B,P,1,1)

        z = slot_base + gate * pos_h
        z = z * patch_valid.unsqueeze(-1).unsqueeze(-1)
        z = z * slot_valid.unsqueeze(-1)
        return z


def init_scene_memory_from_history(h_hist):
    """
    h_hist: (B,K,P,STATE)
    -> scene_mem: (B,P,STATE)
    """
    return h_hist.mean(dim=1)


def update_scene_memory(scene_mem, h_new, momentum=0.98):
    """
    scene_mem: (B,P,STATE)
    h_new:     (B,P,STATE)
    """
    return momentum * scene_mem + (1.0 - momentum) * h_new


betas = torch.linspace(BETA_START, BETA_END, T_STEPS, dtype=torch.float32, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

def q_sample(x0, t, noise):
    c1 = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    c2 = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return c1 * x0 + c2 * noise


state_encoder = FrameStateEncoder().to(device)
dynamics = LatentDynamicsTransformer().to(device)
state_to_latent = FrameStateToCoarseLatent().to(device)
refiner = DiffusionRefiner(
    d_model=min(D_MODEL, 256),  # safer
    n_layers=4,
    n_heads=4
).to(device)

print("state_encoder params (M):", sum(p.numel() for p in state_encoder.parameters()) / 1e6)
print("dynamics params (M):", sum(p.numel() for p in dynamics.parameters()) / 1e6)
print("state_to_latent params (M):", sum(p.numel() for p in state_to_latent.parameters()) / 1e6)
print("refiner params (M):", sum(p.numel() for p in refiner.parameters()) / 1e6)

# %%
# %%
# Stage A: train state_encoder + dynamics + state_to_latent jointly
# 🌟 CRITICAL FIX: Added Reconstruction Loss to prevent Representation Collapse!

opt_dyn = torch.optim.AdamW(
    list(state_encoder.parameters()) + 
    list(dynamics.parameters()) + 
    list(state_to_latent.parameters()),  # 🌟 加入 state_to_latent
    lr=LR_DYN,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95)
)
scaler_dyn = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
best_dyn = 1e9
global_step = 0

SCENE_MEM_MOMENTUM = 0.98

for epoch in range(1, EPOCHS_DYN + 1):
    state_encoder.train()
    dynamics.train()
    state_to_latent.train()  # 🌟 開啟訓練模式
    losses = []
    t0 = time.time()

    for batch in dl:
        x_hist, pos_hist, patch_hist, slot_hist, x_fut, pos_fut, patch_fut, slot_fut = batch

        x_hist = x_hist.to(device, non_blocking=True)
        pos_hist = pos_hist.to(device, non_blocking=True)
        patch_hist = patch_hist.to(device, non_blocking=True)
        slot_hist = slot_hist.to(device, non_blocking=True)

        x_fut = x_fut.to(device, non_blocking=True)
        pos_fut = pos_fut.to(device, non_blocking=True)
        patch_fut = patch_fut.to(device, non_blocking=True)
        slot_fut = slot_fut.to(device, non_blocking=True)

        B = x_hist.shape[0]
        K = x_hist.shape[1]
        P = x_hist.shape[2]    # 🌟 新增呢行：定義 P (空間 Patch 數量)
        H = x_fut.shape[1]

        opt_dyn.zero_grad(set_to_none=True)

        # history states WITH grad
        hist_states = []
        for k in range(K):
            h = state_encoder(x_hist[:, k], patch_hist[:, k], slot_hist[:, k])   # (B,P,STATE)
            hist_states.append(h)
        h_hist = torch.stack(hist_states, dim=1)                                  # (B,K,P,STATE)

        # future states as target WITHOUT grad
        with torch.no_grad():
            fut_states_gt = []
            for hidx in range(H):
                h = state_encoder(x_fut[:, hidx], patch_fut[:, hidx], slot_fut[:, hidx])
                fut_states_gt.append(h)
            h_fut_gt = torch.stack(fut_states_gt, dim=1)                          # (B,H,P,STATE)

        scene_mem = init_scene_memory_from_history(h_hist)                        # (B,P,STATE)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            h_fut_pred = dynamics(h_hist, pos_hist, scene_mem)                    # (B,H,P,STATE)

            # 1. 軌跡預測 Loss
            loss_pred = F.mse_loss(h_fut_pred, h_fut_gt)

            # 2. 場景記憶一致性 Loss
            future_mean_gt = h_fut_gt.mean(dim=1)                                 
            loss_mem = F.mse_loss(scene_mem, future_mean_gt)

            # 3. 🌟 新增：歷史狀態重建 Loss (強制 state_encoder 保留真實幾何)
            # 將 (B, K, P) 壓平成 (B*K, P) 餵入 state_to_latent 進行解碼驗證
            z_coarse_hist = state_to_latent(
                h_hist.view(B*K, P, STATE_DIM),
                scene_mem.unsqueeze(1).expand(B, K, P, STATE_DIM).reshape(B*K, P, STATE_DIM),
                pos_hist.view(B*K, P),
                patch_hist.view(B*K, P),
                slot_hist.view(B*K, P, GROUP_SIZE)
            )
            
            mask_hist = slot_hist.view(B*K, P, GROUP_SIZE).unsqueeze(-1)
            x_hist_flat = x_hist.view(B*K, P, GROUP_SIZE, LATENT_DIM)
            
            # 確保還原出來嘅 Coarse Latent 貼近真實嘅 x_hist
            loss_recon = (((z_coarse_hist - x_hist_flat) ** 2) * mask_hist).sum() / mask_hist.sum().clamp_min(1.0)

            # 總 Loss 包含 預測 + 記憶 + 重建 (權重可微調)
            loss = loss_pred + 0.1 * loss_mem + 1.0 * loss_recon

        scaler_dyn.scale(loss).backward()
        scaler_dyn.unscale_(opt_dyn)
        torch.nn.utils.clip_grad_norm_(
            list(state_encoder.parameters()) + list(dynamics.parameters()) + list(state_to_latent.parameters()),
            GRAD_CLIP
        )
        scaler_dyn.step(opt_dyn)
        scaler_dyn.update()

        global_step += 1
        losses.append(float(loss.detach().cpu()))

        if global_step % LOG_EVERY == 0:
            print(f"[DYN ep{epoch} step{global_step}] loss={np.mean(losses[-LOG_EVERY:]):.6f}")

        del x_hist, pos_hist, patch_hist, slot_hist
        del x_fut, pos_fut, patch_fut, slot_fut
        del h_hist, h_fut_gt, h_fut_pred, scene_mem, loss_pred, loss_mem, loss_recon, loss
        if device == "cuda":
            torch.cuda.empty_cache()

    ep_loss = float(np.mean(losses))
    print(f"Dynamics Epoch {epoch}/{EPOCHS_DYN} | loss={ep_loss:.6f} | time={time.time()-t0:.1f}s")

    if ep_loss < best_dyn:
        best_dyn = ep_loss
        torch.save({
            "state_encoder": state_encoder.state_dict(),
            "dynamics": dynamics.state_dict(),
            "state_to_latent": state_to_latent.state_dict(), # 🌟 一併保存
            "loss": ep_loss,
            "config": {
                "SEQ_FRAMES": SEQ_FRAMES,
                "FUTURE_HORIZON": FUTURE_HORIZON,
                "STATE_DIM": STATE_DIM,
                "SCENE_MEM_MOMENTUM": SCENE_MEM_MOMENTUM,
                "TRAIN_MODE": "latent_dynamics_with_recon",
            }
        }, CKPT_DYN)
        print("saved best dynamics:", CKPT_DYN)

print("Best dynamics loss:", best_dyn)

# %%
# Stage B: train state_to_latent + refiner
# - freeze state_encoder + dynamics after Stage A
# - use scene memory when building coarse latent
# - add state feedback loss: refined latent re-encoded back to patch-state

dyn_ckpt = torch.load(CKPT_DYN, map_location="cpu")
state_encoder.load_state_dict(dyn_ckpt["state_encoder"], strict=True)
dynamics.load_state_dict(dyn_ckpt["dynamics"], strict=True)

for p in state_encoder.parameters():
    p.requires_grad = False
for p in dynamics.parameters():
    p.requires_grad = False

state_encoder.eval()
dynamics.eval()

opt_ref = torch.optim.AdamW(
    list(state_to_latent.parameters()) + list(refiner.parameters()),
    lr=LR_REF,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95)
)
scaler_ref = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
best_ref = 1e9
global_step = 0

STATE_FEEDBACK_W = 0.25
SCENE_MEM_MOMENTUM = 0.98

for epoch in range(1, EPOCHS_REF + 1):
    state_to_latent.train()
    refiner.train()

    losses = []
    diff_losses = []
    recon_losses = []
    state_losses = []
    t0 = time.time()

    for batch in dl:
        x_hist, pos_hist, patch_hist, slot_hist, x_fut, pos_fut, patch_fut, slot_fut = batch

        x_hist = x_hist.to(device, non_blocking=True)
        pos_hist = pos_hist.to(device, non_blocking=True)
        patch_hist = patch_hist.to(device, non_blocking=True)
        slot_hist = slot_hist.to(device, non_blocking=True)

        x_fut = x_fut.to(device, non_blocking=True)
        pos_fut = pos_fut.to(device, non_blocking=True)
        patch_fut = patch_fut.to(device, non_blocking=True)
        slot_fut = slot_fut.to(device, non_blocking=True)

        B = x_hist.shape[0]
        K = x_hist.shape[1]
        H = x_fut.shape[1]

        # frozen history states
        with torch.no_grad():
            hist_states = []
            for k in range(K):
                h = state_encoder(x_hist[:, k], patch_hist[:, k], slot_hist[:, k])   # (B,P,STATE)
                hist_states.append(h)
            h_hist = torch.stack(hist_states, dim=1)                                  # (B,K,P,STATE)
            scene_mem = init_scene_memory_from_history(h_hist)                        # (B,P,STATE)
            h_fut_pred = dynamics(h_hist, pos_hist, scene_mem)                        # (B,H,P,STATE)

        opt_ref.zero_grad(set_to_none=True)

        batch_loss_sum = 0.0
        batch_diff_sum = 0.0
        batch_recon_sum = 0.0
        batch_state_sum = 0.0

        for hidx in range(H):
            x_target = x_fut[:, hidx]
            pos_target = pos_fut[:, hidx]
            patch_target = patch_fut[:, hidx]
            slot_target = slot_fut[:, hidx]
            mask4 = slot_target.unsqueeze(-1)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                z_coarse = state_to_latent(
                    h_fut_pred[:, hidx],
                    scene_mem,
                    pos_target,
                    patch_target,
                    slot_target
                )

                mask4 = slot_target.unsqueeze(-1)
                denom = slot_target.sum().clamp_min(1.0) * LATENT_DIM

                # 🌟 新增：直接約束 z_coarse，令 0-shot 預測變得可靠
                coarse_loss = (((z_coarse - x_target) ** 2) * mask4).sum() / denom

                residual_gt = (x_target - z_coarse) * mask4
                t = torch.randint(low=0, high=REFINE_T_MAX, size=(B,), device=device)
                noise = torch.randn_like(residual_gt) * mask4
                residual_t = q_sample(residual_gt, t, noise)

                eps_pred = refiner(
                    residual_t,
                    pos_target,
                    patch_target,
                    slot_target,
                    t,
                    ctx_latent=z_coarse.detach(), # 🌟 detach 避免梯度衝突
                    ctx_slot_valid_mask=slot_target,
                )

                diff_loss = (((eps_pred - noise) ** 2) * mask4).sum() / denom

                c1 = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                c2 = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                residual0_hat = (residual_t - c2 * eps_pred) / c1.clamp_min(1e-8)
                residual0_hat = torch.clamp(residual0_hat, -1.5, 1.5) * mask4

                z_refined = (z_coarse + residual0_hat) * mask4
                recon_loss = (((z_refined - x_target) ** 2) * mask4).sum() / denom

                h_refined = state_encoder(z_refined, patch_target, slot_target)
                state_loss = F.mse_loss(h_refined, h_fut_pred[:, hidx])

                # 🌟 將 coarse_loss 加入總 Loss
                loss_h = (diff_loss + 0.5 * recon_loss + 1.0 * coarse_loss + STATE_FEEDBACK_W * state_loss) / H

            scaler_ref.scale(loss_h).backward()

            batch_loss_sum += float(loss_h.detach().cpu())
            batch_diff_sum += float(diff_loss.detach().cpu())
            # 🌟 記錄 recon_loss + coarse_loss 方便觀察
            batch_recon_sum += float((recon_loss + coarse_loss).detach().cpu())
            batch_state_sum += float(state_loss.detach().cpu())

            del x_target, pos_target, patch_target, slot_target, mask4
            del z_coarse, residual_gt, t, noise, residual_t, eps_pred
            del denom, diff_loss, c1, c2, residual0_hat, z_refined, recon_loss
            del h_refined, state_loss, loss_h

        scaler_ref.unscale_(opt_ref)
        torch.nn.utils.clip_grad_norm_(
            list(state_to_latent.parameters()) + list(refiner.parameters()),
            GRAD_CLIP
        )
        scaler_ref.step(opt_ref)
        scaler_ref.update()

        global_step += 1
        losses.append(batch_loss_sum)
        diff_losses.append(batch_diff_sum)
        recon_losses.append(batch_recon_sum)
        state_losses.append(batch_state_sum)

        if global_step % LOG_EVERY == 0:
            print(
                f"[REF ep{epoch} step{global_step}] "
                f"loss={np.mean(losses[-LOG_EVERY:]):.6f} | "
                f"diff={np.mean(diff_losses[-LOG_EVERY:]):.6f} | "
                f"recon={np.mean(recon_losses[-LOG_EVERY:]):.6f} | "
                f"state={np.mean(state_losses[-LOG_EVERY:]):.6f}"
            )

        del x_hist, pos_hist, patch_hist, slot_hist
        del x_fut, pos_fut, patch_fut, slot_fut
        del h_hist, scene_mem, h_fut_pred
        if device == "cuda":
            torch.cuda.empty_cache()

    ep_loss = float(np.mean(losses))
    print(f"Refiner Epoch {epoch}/{EPOCHS_REF} | loss={ep_loss:.6f} | time={time.time()-t0:.1f}s")

    if ep_loss < best_ref:
        best_ref = ep_loss
        torch.save({
            "state_to_latent": state_to_latent.state_dict(),
            "refiner": refiner.state_dict(),
            "loss": ep_loss,
            "config": {
                "STATE_DIM": STATE_DIM,
                "REFINE_T_MAX": REFINE_T_MAX,
                "STATE_FEEDBACK_W": STATE_FEEDBACK_W,
                "TRAIN_MODE": "diffusion_refiner_with_state_feedback",
            }
        }, CKPT_REF)
        print("saved best refiner:", CKPT_REF)

print("Best refiner loss:", best_ref)

# %%
# Long rollout in PATCH-STATE space with:
# 1) persistent scene memory
# 2) refined latent -> re-encode -> update h_hist
# This is the key fix for long-sequence stability.

dyn_ckpt = torch.load(CKPT_DYN, map_location="cpu")
state_encoder.load_state_dict(dyn_ckpt["state_encoder"], strict=True)
dynamics.load_state_dict(dyn_ckpt["dynamics"], strict=True)

ref_ckpt = torch.load(CKPT_REF, map_location="cpu")
state_to_latent.load_state_dict(ref_ckpt["state_to_latent"], strict=True)
refiner.load_state_dict(ref_ckpt["refiner"], strict=True)

state_encoder.eval()
dynamics.eval()
state_to_latent.eval()
refiner.eval()

SCENE_MEM_MOMENTUM = 0.98

@torch.no_grad()
def refine_multi_step_residual(z_coarse, pos_target, patch_target, slot_target, steps=20):
    """
    Multi-step residual denoising from noise.
    """
    B = z_coarse.shape[0]
    residual = torch.randn_like(z_coarse) * slot_target.unsqueeze(-1)

    # use early timesteps only; consistent with low-noise residual regime
    max_step = min(steps, REFINE_T_MAX)

    for i in reversed(range(max_step)):
        t = torch.full((B,), i, device=z_coarse.device, dtype=torch.long)

        eps_pred = refiner(
            residual,
            pos_target,
            patch_target,
            slot_target,
            t,
            ctx_latent=z_coarse,
            ctx_slot_valid_mask=slot_target,
        )

        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        beta_t = betas[i]

        if i > 0:
            noise = torch.randn_like(residual) * slot_target.unsqueeze(-1)
        else:
            noise = torch.zeros_like(residual)

        c1 = 1.0 / torch.sqrt(alpha_t)
        c2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

        residual = c1 * (residual - c2 * eps_pred) + torch.sqrt(beta_t) * noise
        residual = residual * slot_target.unsqueeze(-1)

    residual = torch.clamp(residual, -DELTA_CLIP_INFER, DELTA_CLIP_INFER)
    z_refined = (z_coarse + residual) * slot_target.unsqueeze(-1)
    return z_refined, residual


@torch.no_grad()
def rollout_states_and_latents(
    start_idx=0,
    rollout_steps=1000,
    refine_every=4,
    refine_steps=20,
):
    max_start = max(0, frame_lat_all.shape[0] - (SEQ_FRAMES + rollout_steps))
    start = int(np.clip(start_idx, 0, max_start))
    end = start + SEQ_FRAMES

    x_hist = torch.from_numpy(frame_lat_all[start:end]).unsqueeze(0).to(device)               # (1,K,P,32,128)
    pos_hist = torch.from_numpy(frame_pos_all[start:end]).unsqueeze(0).to(device)             # (1,K,P)
    pv_hist = torch.from_numpy(frame_patch_valid_all[start:end]).unsqueeze(0).to(device)      # (1,K,P)
    sv_hist = torch.from_numpy(frame_slot_valid_all[start:end]).unsqueeze(0).to(device)       # (1,K,P,32)

    # initial patch-state history
    h_hist_list = []
    for k in range(SEQ_FRAMES):
        h = state_encoder(x_hist[:, k], pv_hist[:, k], sv_hist[:, k])                         # (1,P,STATE)
        h_hist_list.append(h)
    h_hist = torch.stack(h_hist_list, dim=1)                                                  # (1,K,P,STATE)

    # initialize persistent scene memory
    scene_mem = init_scene_memory_from_history(h_hist)                                         # (1,P,STATE)

    pred_latents = []
    gt_latents = []

    for step in range(rollout_steps):
        tgt_idx = end + step

        # predict future patch states using BOTH h_hist and scene_mem
        h_future = dynamics(h_hist, pos_hist, scene_mem)                                       # (1,H,P,STATE)
        h_next_coarse = h_future[:, 0]                                                         # (1,P,STATE)

        # still using known target layout assumption (same as your 06/08 practical setup)
        pos_target = torch.from_numpy(frame_pos_all[tgt_idx:tgt_idx+1]).to(device)
        patch_target = torch.from_numpy(frame_patch_valid_all[tgt_idx:tgt_idx+1]).to(device)
        slot_target = torch.from_numpy(frame_slot_valid_all[tgt_idx:tgt_idx+1]).to(device)

        # coarse latent from state + scene memory
        z_coarse = state_to_latent(
            h_next_coarse,
            scene_mem,
            pos_target,
            patch_target,
            slot_target
        )

        if (step % refine_every) == 0:
            z_next, residual_hat = refine_multi_step_residual(
                z_coarse,
                pos_target,
                patch_target,
                slot_target,
                steps=refine_steps
            )
        else:
            z_next = z_coarse

        pred_latents.append(z_next[0].detach().cpu().numpy())
        gt_latents.append(frame_lat_all[tgt_idx].copy())

        # CRITICAL FIX:
        # re-encode REFINED latent, not raw h_next_coarse
        h_next_refined = state_encoder(z_next, patch_target, slot_target)                      # (1,P,STATE)

        # update state history with refined state
        h_hist = torch.cat([h_hist[:, 1:], h_next_refined.unsqueeze(1)], dim=1)
        pos_hist = torch.cat([pos_hist[:, 1:], pos_target.unsqueeze(1)], dim=1)

        # update persistent scene memory slowly
        scene_mem = update_scene_memory(scene_mem, h_next_refined, momentum=SCENE_MEM_MOMENTUM)

        if (step + 1) == 1 or (step + 1) == rollout_steps or ((step + 1) % 100 == 0):
            pred_np = z_next[0].detach().cpu().numpy()
            valid = frame_slot_valid_all[tgt_idx] > 0.5
            vals = pred_np[valid]
            print(
                f"[rollout {step+1}/{rollout_steps}] "
                f"latent mean/std/min/max = "
                f"{float(vals.mean()):.6f} / {float(vals.std()):.6f} / "
                f"{float(vals.min()):.6f} / {float(vals.max()):.6f}"
            )

    return start, end, pred_latents, gt_latents

ROLL_OUT_STEPS = 2
# run rollout
start, end, pred_latents, gt_latents = rollout_states_and_latents(
    start_idx=ROLL_PROMPT_START,
    rollout_steps=ROLL_OUT_STEPS,
    refine_every=ROLL_REFINE_EVERY,
    refine_steps=20,
)

print("rollout prompt window:", (start, end - 1))
print("generated frames:", len(pred_latents))

# final-step latent comparison
final_pred = pred_latents[-1]
final_gt = gt_latents[-1]
final_tgt_idx = end + ROLL_OUT_STEPS - 1

valid_patch_np = frame_patch_valid_all[final_tgt_idx] > 0.5
slot_valid_np = frame_slot_valid_all[final_tgt_idx][valid_patch_np]
valid_idx = slot_valid_np > 0.5

pred_np = final_pred[valid_patch_np]
gt_np = final_gt[valid_patch_np]

pred_vals = pred_np[valid_idx]
gt_vals = gt_np[valid_idx]

mae = np.mean(np.abs(pred_vals - gt_vals))
rmse = np.sqrt(np.mean((pred_vals - gt_vals) ** 2))

print("\n--- final latent compare ---")
print("final target idx:", final_tgt_idx)
print("latent MAE (normalized):", float(mae))
print("latent RMSE (normalized):", float(rmse))

np.savez_compressed(
    os.path.join(OUT_DIR, f"rollout_state_refiner_final_{final_tgt_idx:04d}.npz"),
    pred_norm=pred_np.astype(np.float32),
    gt_norm=gt_np.astype(np.float32),
    slot_valid=slot_valid_np.astype(np.float32),
    final_target_idx=np.asarray([final_tgt_idx], dtype=np.int32),
    rollout_steps=np.asarray([ROLL_OUT_STEPS], dtype=np.int32),
)
print("saved latent compare npz.")
print("You can now reuse your decode cell to export GT / PRED PLY.")

# %%
# Decode saved rollout_state_refiner_final_XXXX.npz into GT / PRED PLY
# Standalone cell: can run after your rollout cell.

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
# If you want to force a specific file, set it here.
COMPARE_NPZ_PATH = None   # e.g. r"./GAUSSIAN_LATENT_DYNAMICS_REFINER/rollout_state_refiner_final_0123.npz"

# fallback defaults (used if not already defined in notebook)
if "LATENT_NPZ" not in globals():
    LATENT_NPZ = r"./GAUSSIAN_GROUP_VAE/group_tokens.npz"
if "VAE_CKPT_PATH" not in globals():
    VAE_CKPT_PATH = r"./GAUSSIAN_GROUP_VAE/gaussian_group_vae_best.pt"
if "WORLD_STATS_NPZ" not in globals():
    WORLD_STATS_NPZ = r"./GAUSSIAN_GROUP_VAE/world_stats.npz"
if "OUT_DIR" not in globals():
    OUT_DIR = r"./GAUSSIAN_LATENT_DYNAMICS_REFINER"

PLY_OUT_DIR = os.path.join(OUT_DIR, "ply_decode")
os.makedirs(PLY_OUT_DIR, exist_ok=True)

if "GROUP_SIZE" not in globals():
    GROUP_SIZE = 32
if "LATENT_DIM" not in globals():
    LATENT_DIM = 128
if "GAUSS_DIM" not in globals():
    GAUSS_DIM = 11

# IMPORTANT: must match VAE checkpoint
VAE_D_MODEL = 256
EMPTY_EPS = 1e-8

# decode / export knobs
MASK_THR = globals().get("MASK_THR", 0.75)
OPACITY_THR = globals().get("OPACITY_THR", 0.15)
MIN_SCALE_CM = globals().get("MIN_SCALE_CM", 3.0)
MAX_SCALE_CM = globals().get("MAX_SCALE_CM", 80.0)

SURFACE_POINTS_PER_GAUSSIAN = globals().get("SURFACE_POINTS_PER_GAUSSIAN", 120)
INNER_POINTS_PER_GAUSSIAN = globals().get("INNER_POINTS_PER_GAUSSIAN", 60)
SURFACE_SIGMA_LEVEL = globals().get("SURFACE_SIGMA_LEVEL", 2.0)
MAX_GAUSS_BALL_POINTS_EXPORT = globals().get("MAX_GAUSS_BALL_POINTS_EXPORT", 400000)

device = globals().get("device", "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------------------------------------
# locate compare npz
# ---------------------------------------------------------
if COMPARE_NPZ_PATH is None:
    if "final_tgt_idx" in globals():
        candidate = os.path.join(OUT_DIR, f"rollout_state_refiner_final_{int(final_tgt_idx):04d}.npz")
        if os.path.exists(candidate):
            COMPARE_NPZ_PATH = candidate

if COMPARE_NPZ_PATH is None:
    cands = sorted(glob.glob(os.path.join(OUT_DIR, "rollout_state_refiner_final_*.npz")))
    if len(cands) == 0:
        raise FileNotFoundError(
            "No rollout_state_refiner_final_*.npz found. "
            "Please set COMPARE_NPZ_PATH manually."
        )
    COMPARE_NPZ_PATH = cands[-1]

print("COMPARE_NPZ_PATH:", COMPARE_NPZ_PATH)

# ---------------------------------------------------------
# load / prepare latent normalization stats
# ---------------------------------------------------------
if ("lat_mean" not in globals()) or ("lat_std" not in globals()):
    print("lat_mean / lat_std not found in memory, recomputing from LATENT_NPZ...")
    z = np.load(LATENT_NPZ)
    latents = z["latents"].astype(np.float32)   # (M,32,128)
    group_mask = z["mask"].astype(np.float32)   # (M,32)
    valid_slot_mask = group_mask > 0.5
    valid_latents = latents[valid_slot_mask]    # (N_valid,128)
    lat_mean = valid_latents.mean(axis=0).astype(np.float32)
    lat_std = valid_latents.std(axis=0).astype(np.float32)
    lat_std = np.maximum(lat_std, 1e-4)
else:
    print("Using in-memory lat_mean / lat_std")

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def safe_load(path):
    t0 = time.time()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("loaded:", path, "time:", f"{time.time()-t0:.2f}s")
    return ckpt

def quat_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)

def sample_gaussian_ellipsoid_points(
    mu_cm, sc_cm, quat, opacity,
    surface_n=120, inner_n=60, sigma_level=2.0, seed=123
):
    rng = np.random.default_rng(seed)
    all_pts, all_rgb = [], []

    for i in range(mu_cm.shape[0]):
        op = float(np.clip(opacity[i], 0.0, 1.0))
        s_n = int(surface_n * op)
        i_n = int(inner_n * op)
        if s_n + i_n <= 0:
            continue

        mu = mu_cm[i].astype(np.float64)
        sc = np.maximum(sc_cm[i].astype(np.float64), 1e-4)
        R = quat_to_rotmat(quat[i])

        if s_n > 0:
            v = rng.normal(size=(s_n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
            local = v * (sigma_level * sc[None, :])
            world = local @ R.T + mu[None, :]
            all_pts.append(world.astype(np.float32))

        if i_n > 0:
            v = rng.normal(size=(i_n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
            r = rng.random((i_n, 1)) ** (1.0 / 3.0)
            local = v * r * (sigma_level * sc[None, :])
            world = local @ R.T + mu[None, :]
            all_pts.append(world.astype(np.float32))

        n_i = s_n + i_n
        color = np.array([255 * op, 80, 255 * (1.0 - op)], dtype=np.uint8)
        all_rgb.append(np.repeat(color.reshape(1, 3), n_i, axis=0))

    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), np.float32)
    rgb = np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), np.uint8)
    return pts, rgb

def write_ply_points(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def make_uniform_rgb(n, color):
    c = np.asarray(color, dtype=np.uint8).reshape(1, 3)
    return np.repeat(c, n, axis=0)

def downsample_pts(pts, max_pts, rgb=None, seed=123):
    if pts.shape[0] <= max_pts:
        return (pts, rgb) if rgb is not None else pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
    if rgb is None:
        return pts[idx]
    return pts[idx], rgb[idx]

def infer_empty_mask_from_feat(feat, mask=None):
    if mask is not None:
        return (mask <= 0.5)
    return (feat.abs().sum(dim=-1) <= EMPTY_EPS)

def export_gaussians_pair(prefix, mu_cm, sc_cm, quat, op, color=(255, 0, 0), seed=123):
    p_mu = os.path.join(PLY_OUT_DIR, f"{prefix}_centers.ply")
    write_ply_points(p_mu, mu_cm, make_uniform_rgb(mu_cm.shape[0], list(color)))

    ball_pts, ball_rgb = sample_gaussian_ellipsoid_points(
        mu_cm, sc_cm, quat, op,
        surface_n=SURFACE_POINTS_PER_GAUSSIAN,
        inner_n=INNER_POINTS_PER_GAUSSIAN,
        sigma_level=SURFACE_SIGMA_LEVEL,
        seed=seed,
    )
    ball_pts, ball_rgb = downsample_pts(
        ball_pts, MAX_GAUSS_BALL_POINTS_EXPORT, rgb=ball_rgb, seed=seed + 1
    )

    p_ball = os.path.join(PLY_OUT_DIR, f"{prefix}_balls.ply")
    write_ply_points(p_ball, ball_pts, ball_rgb)

    print("saved:", p_mu)
    print("saved:", p_ball)

# ---------------------------------------------------------
# VAE decoder classes (checkpoint-compatible)
# ---------------------------------------------------------
class SlotContextEncoder(nn.Module):
    def __init__(self, in_dim=GAUSS_DIM, d_model=VAE_D_MODEL, latent_dim=LATENT_DIM):
        super().__init__()
        self.slot = nn.Sequential(
            nn.Linear(in_dim + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.to_latent = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, feat, mask):
        x = torch.cat([feat, mask.unsqueeze(-1)], dim=-1)
        h = self.slot(x)
        valid = mask.unsqueeze(-1)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        ctx = (h * valid).sum(dim=1, keepdim=True) / denom
        ctx = ctx.expand_as(h)
        z = self.to_latent(torch.cat([h, ctx], dim=-1))
        z = z * valid
        return z


class SlotContextDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, d_model=VAE_D_MODEL):
        super().__init__()

        self.slot = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.ctx = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.out_xyz = nn.Linear(d_model, 3)
        self.out_sc  = nn.Linear(d_model, 3)
        self.out_q   = nn.Linear(d_model, 4)
        self.out_op  = nn.Linear(d_model, 1)
        self.out_mask= nn.Linear(d_model, 1)

        nn.init.constant_(self.out_sc.bias, -4.0)
        nn.init.constant_(self.out_q.bias[0], 1.0)
        nn.init.constant_(self.out_q.bias[1], 0.0)
        nn.init.constant_(self.out_q.bias[2], 0.0)
        nn.init.constant_(self.out_q.bias[3], 0.0)

    def forward(self, z, slot_valid_mask=None):
        B, N, D = z.shape

        if slot_valid_mask is None:
            z_ctx = z.mean(dim=1, keepdim=True).expand(B, N, D)
        else:
            m = slot_valid_mask.float().unsqueeze(-1)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            z_ctx_single = (z * m).sum(dim=1, keepdim=True) / denom
            z_ctx = z_ctx_single.expand(B, N, D)

        h_slot = self.slot(z)
        h_ctx  = self.ctx(z_ctx)
        h = self.fuse(torch.cat([h_slot, h_ctx], dim=-1))

        xyz = torch.clamp(self.out_xyz(h), min=-3.0, max=3.0)
        log_sc = torch.clamp(self.out_sc(h), min=-12.0, max=4.0)

        quat = self.out_q(h)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        opacity = torch.sigmoid(self.out_op(h))
        mask_logit = self.out_mask(h).squeeze(-1)

        feat_out = torch.cat([xyz, log_sc, quat, opacity], dim=-1)
        return feat_out, mask_logit


class GaussianGroupContAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SlotContextEncoder(d_model=VAE_D_MODEL)
        self.decoder = SlotContextDecoder(d_model=VAE_D_MODEL)


# ---------------------------------------------------------
# load VAE + world stats
# ---------------------------------------------------------
world_stats = np.load(WORLD_STATS_NPZ)
world_center = world_stats["world_center"].astype(np.float32)
world_scale = float(world_stats["world_scale"])

vae_model = GaussianGroupContAE().to(device)
vae_ckpt = safe_load(VAE_CKPT_PATH)
vae_model.load_state_dict(vae_ckpt["model"], strict=True)
vae_model.eval()
print("VAE loaded.")

# ---------------------------------------------------------
# decode helper
# ---------------------------------------------------------
@torch.no_grad()
def decode_frame_latents_to_world_gaussians(z_frame_unnorm_4d, slot_valid_mask_2d):
    """
    z_frame_unnorm_4d: (Pv,32,128)   unnormalized latent
    slot_valid_mask_2d: (Pv,32)
    """
    P = z_frame_unnorm_4d.shape[0]
    if P == 0:
        return (
            np.zeros((0,3), np.float32),
            np.zeros((0,3), np.float32),
            np.zeros((0,4), np.float32),
            np.zeros((0,), np.float32),
        )

    z_in = z_frame_unnorm_4d.copy()
    z_in[slot_valid_mask_2d <= 0.5] = 0.0

    z_t = torch.from_numpy(z_in).to(device).float()               # (Pv,32,128)
    m_t = torch.from_numpy(slot_valid_mask_2d).to(device).float() # (Pv,32)

    feat_hat, mask_logit = vae_model.decoder(z_t, m_t)
    mask_prob = torch.sigmoid(mask_logit)

    feat = feat_hat.detach().cpu().numpy().astype(np.float32)     # (Pv,32,11)
    mprob = mask_prob.detach().cpu().numpy().astype(np.float32)   # (Pv,32)

    valid = (slot_valid_mask_2d > 0.5) & (mprob >= MASK_THR)
    feat = feat[valid]

    if feat.shape[0] == 0:
        return (
            np.zeros((0,3), np.float32),
            np.zeros((0,3), np.float32),
            np.zeros((0,4), np.float32),
            np.zeros((0,), np.float32),
        )

    xyz_n = feat[:, 0:3]
    mu_cm = xyz_n * world_scale + world_center[None, :]

    sc_cm = np.exp(np.clip(feat[:, 3:6], -12.0, 4.0)) * world_scale
    sc_cm = np.clip(sc_cm, MIN_SCALE_CM, MAX_SCALE_CM)

    quat = feat[:, 6:10]
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)

    op = np.clip(feat[:, 10], 0.0, 1.0)
    keep = op >= OPACITY_THR
    mu_cm, sc_cm, quat, op = mu_cm[keep], sc_cm[keep], quat[keep], op[keep]

    return mu_cm.astype(np.float32), sc_cm.astype(np.float32), quat.astype(np.float32), op.astype(np.float32)

# ---------------------------------------------------------
# load compare npz
# ---------------------------------------------------------
cmpz = np.load(COMPARE_NPZ_PATH)
pred_norm = cmpz["pred_norm"].astype(np.float32)     # (Pv,32,128), normalized
gt_norm = cmpz["gt_norm"].astype(np.float32)         # (Pv,32,128), normalized
slot_valid = cmpz["slot_valid"].astype(np.float32)   # (Pv,32)

final_target_idx = int(cmpz["final_target_idx"][0]) if "final_target_idx" in cmpz.files else -1
rollout_steps = int(cmpz["rollout_steps"][0]) if "rollout_steps" in cmpz.files else -1

print("pred_norm shape:", pred_norm.shape)
print("gt_norm shape  :", gt_norm.shape)
print("slot_valid shape:", slot_valid.shape)
print("final_target_idx:", final_target_idx)
print("rollout_steps:", rollout_steps)

# ---------------------------------------------------------
# unnormalize
# ---------------------------------------------------------
pred_unnorm = np.zeros_like(pred_norm, dtype=np.float32)
gt_unnorm = np.zeros_like(gt_norm, dtype=np.float32)

valid_idx = slot_valid > 0.5
pred_unnorm[valid_idx] = pred_norm[valid_idx] * lat_std[None, :] + lat_mean[None, :]
gt_unnorm[valid_idx] = gt_norm[valid_idx] * lat_std[None, :] + lat_mean[None, :]

print("valid latent count:", int(valid_idx.sum()))
if valid_idx.any():
    print("PRED unnorm mean/std:", float(pred_unnorm[valid_idx].mean()), float(pred_unnorm[valid_idx].std()))
    print("GT   unnorm mean/std:", float(gt_unnorm[valid_idx].mean()), float(gt_unnorm[valid_idx].std()))

# ---------------------------------------------------------
# decode
# ---------------------------------------------------------
mu_pred, sc_pred, quat_pred, op_pred = decode_frame_latents_to_world_gaussians(pred_unnorm, slot_valid)
mu_gt, sc_gt, quat_gt, op_gt = decode_frame_latents_to_world_gaussians(gt_unnorm, slot_valid)

print("\nDecoded:")
print("PRED gaussians:", mu_pred.shape[0])
print("GT   gaussians:", mu_gt.shape[0])

if mu_pred.shape[0] > 0:
    print("PRED xyz mean/std:", mu_pred.mean(axis=0), mu_pred.std(axis=0))
if mu_gt.shape[0] > 0:
    print("GT   xyz mean/std:", mu_gt.mean(axis=0), mu_gt.std(axis=0))

# ---------------------------------------------------------
# export
# ---------------------------------------------------------
tag_base = f"decoded_final_{final_target_idx:04d}" if final_target_idx >= 0 else "decoded_final"

export_gaussians_pair(
    prefix=f"{tag_base}_pred",
    mu_cm=mu_pred, sc_cm=sc_pred, quat=quat_pred, op=op_pred,
    color=(0, 255, 0), seed=5000
)

export_gaussians_pair(
    prefix=f"{tag_base}_gt",
    mu_cm=mu_gt, sc_cm=sc_gt, quat=quat_gt, op=op_gt,
    color=(255, 0, 0), seed=6000
)

print("\nDone.")
print("Output folder:", PLY_OUT_DIR)



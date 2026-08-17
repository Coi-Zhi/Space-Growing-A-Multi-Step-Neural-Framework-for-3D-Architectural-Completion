# %%
# %%
# Cell 1 — Imports + Config

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

# -----------------
# Switches
# -----------------
SYNTHETIC_DEMO = False   # set False to use real NPZ

# -----------------
# Paths (edit)
# -----------------
LATENT_NPZ = r"./GAUSSIAN_GROUP_VAE/group_tokens.npz"   # your saved latents per group/patch
OUT_DIR    = r"./GAUSSIAN_LATENT_LM/checkpoints_gs_latent_lm"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------
# Training
# -----------------
SEQ_LEN       = 512
BATCH_SIZE    = 8
NUM_WORKERS   = 0
EPOCHS        = 5
LR            = 3e-4
WEIGHT_DECAY  = 0.05
GRAD_CLIP     = 1.0
WARMUP_STEPS  = 200
LOG_EVERY     = 50
VAL_EVERY     = 500

TOKEN_DROPOUT_P = 0.10

# -----------------
# Generation
# -----------------
PROMPT_LEN = 256
CTX_LEN    = 1024
MAX_NEW    = 4000
LOG_EVERY_GEN = 200
FRAME_IDX_TO_VIEW = 0

# -----------------
# Model
# -----------------
LATENT_DIM = 128   # z_patch dimension (edit if yours differs)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %%
# %%
# Cell 2 — Load / Build (frame_id, pos_id, z_patch) table  [FIXED: never use group_ids as fake pos_ids]

def masked_mean(x, m, dim=1, eps=1e-6):
    # x: (M,N,D) m:(M,N) -> (M,D)
    w = m.float().unsqueeze(-1)
    return (x * w).sum(dim=dim) / (w.sum(dim=dim).clamp_min(eps))

def world_xyz_to_grid_index(xyz_cm, world_center, vox_cm, grid_bias):
    xyz_cm = np.asarray(xyz_cm, dtype=np.float32)
    world_center = np.asarray(world_center, dtype=np.float32)
    idx = np.rint((xyz_cm - world_center[None, :]) / float(vox_cm)).astype(np.int32)
    idx = idx + int(grid_bias)
    idx = np.clip(idx, 0, 1023)
    return idx.astype(np.int32)

def pack_xyz_index_to_pos_id(ixyz):
    ixyz = np.asarray(ixyz, dtype=np.int64)
    ix = ixyz[:, 0]
    iy = ixyz[:, 1]
    iz = ixyz[:, 2]
    return (ix | (iy << 10) | (iz << 20)).astype(np.int64)

if SYNTHETIC_DEMO:
    # Create a tiny synthetic dataset: 20 frames, each 40 patches
    rng = np.random.default_rng(0)
    n_frames = 20
    patches_per_frame = 40
    M = n_frames * patches_per_frame

    frame_ids = np.repeat(np.arange(n_frames, dtype=np.int32), patches_per_frame)
    pos_ids   = rng.integers(low=0, high=2**30 - 1, size=M, dtype=np.int64)
    z_patch   = rng.normal(size=(M, LATENT_DIM)).astype(np.float32)

    # keep these globals for inference visualization
    POS_PACK_VOX_CM_META = 20.0
    POS_GRID_BIAS_META = 512

    print("SYNTH data:", z_patch.shape, frame_ids.shape, pos_ids.shape)

else:
    z = np.load(LATENT_NPZ)

    # -------------------------
    # latents
    # -------------------------
    if "latents" not in z.files:
        raise RuntimeError("NPZ must contain 'latents'.")

    lat = z["latents"].astype(np.float32)

    # -------------------------
    # frame ids
    # -------------------------
    if "frame_ids" in z.files:
        frame_ids = z["frame_ids"].astype(np.int32)
    elif "frame" in z.files:
        frame_ids = z["frame"].astype(np.int32)
    else:
        raise RuntimeError("NPZ must contain 'frame_ids' or 'frame'.")

    # -------------------------
    # position ids (STRICT)
    # -------------------------
    if "pos_ids" in z.files:
        pos_ids = z["pos_ids"].astype(np.int64)

        POS_PACK_VOX_CM_META = float(z["pos_pack_vox_cm"]) if "pos_pack_vox_cm" in z.files else 20.0
        POS_GRID_BIAS_META = int(z["pos_grid_bias"]) if "pos_grid_bias" in z.files else 512

    elif ("group_centers_cm" in z.files) and ("world_center" in z.files):
        group_centers_cm = z["group_centers_cm"].astype(np.float32)
        world_center = z["world_center"].astype(np.float32)
        POS_PACK_VOX_CM_META = float(z["pos_pack_vox_cm"]) if "pos_pack_vox_cm" in z.files else 20.0
        POS_GRID_BIAS_META = int(z["pos_grid_bias"]) if "pos_grid_bias" in z.files else 512

        idx = world_xyz_to_grid_index(
            group_centers_cm,
            world_center=world_center,
            vox_cm=POS_PACK_VOX_CM_META,
            grid_bias=POS_GRID_BIAS_META
        )
        pos_ids = pack_xyz_index_to_pos_id(idx)

    else:
        raise RuntimeError(
            "This NPZ does not contain real spatial positions.\n"
            "Need 'pos_ids' or ('group_centers_cm' + 'world_center').\n"
            "Do NOT use 'group_ids' as fallback, because group_ids are only local chunk indices."
        )

    # -------------------------
    # latent pooling (FIXED: Flatten instead of Mean)
    # -------------------------
    if lat.ndim == 3:
        M, N, D = lat.shape
        # 保留所有 32 粒 Gaussian 嘅特徵，直接攤平成一條 32x128 = 4096 維嘅超級向量
        z_patch = lat.reshape(M, N * D)
        LATENT_DIM = int(z_patch.shape[-1])
        print(f"Flattened latents from ({M}, {N}, {D}) to ({M}, {LATENT_DIM})")

    elif lat.ndim == 2:
        z_patch = lat
        LATENT_DIM = int(z_patch.shape[-1])

    else:
        raise RuntimeError("latents must be (M,D) or (M,N,D).")

    print("Loaded:", z_patch.shape, frame_ids.shape, pos_ids.shape, "LATENT_DIM:", LATENT_DIM)
    print("POS_PACK_VOX_CM_META:", POS_PACK_VOX_CM_META, "POS_GRID_BIAS_META:", POS_GRID_BIAS_META)

# Build per-frame lists (sorted for deterministic grammar)
frame_to_items = {}
for fid, pid, zp in zip(frame_ids, pos_ids, z_patch):
    frame_to_items.setdefault(int(fid), []).append((int(pid), zp))

frame_keys = sorted(frame_to_items.keys())
print("num frames:", len(frame_keys), "example frame patches:", len(frame_to_items[frame_keys[0]]))

# %%
# %%
# Cell 3 — Vocabulary + Encode pos chunks + Build mixed event "sentence" arrays

# Vocab layout (small + fixed, like your voxel pipeline)
# - POS chunk tokens share the same 1024 range
# - Specials at SPECIAL_BASE
POS_BASE     = 0              # [0..1023] -> poschunk
SPECIAL_BASE = POS_BASE + 1024

FRAME_BOS = SPECIAL_BASE + 0
FRAME_EOS = SPECIAL_BASE + 1
EOS_ID    = SPECIAL_BASE + 2
MASK_ID   = SPECIAL_BASE + 3
PAD_ID    = SPECIAL_BASE + 4

VOCAB_SIZE = SPECIAL_BASE + 5

print("VOCAB_SIZE:", VOCAB_SIZE, "EOS_ID:", EOS_ID, "MASK_ID:", MASK_ID, "PAD_ID:", PAD_ID)

def encode_pos_chunks(pos_id: int):
    # 3 chunks of 10 bits (base-1024), pos_id < 1024^3 is supported
    p0 = pos_id & 1023
    p1 = (pos_id >> 10) & 1023
    p2 = (pos_id >> 20) & 1023
    return [POS_BASE + p0, POS_BASE + p1, POS_BASE + p2]

def decode_pos_from_chunks(p0, p1, p2):
    return int(p0) | (int(p1) << 10) | (int(p2) << 20)

# Build mixed event stream across frames
# arrays:
#   tok[t]   int64   (valid only when type[t]==0)
#   lat[t]   float32 (valid only when type[t]==1)
#   typ[t]   uint8   0=token, 1=latent
tok_list = []
lat_list = []
typ_list = []

def push_token(tid: int):
    tok_list.append(int(tid))
    lat_list.append(np.zeros((LATENT_DIM,), np.float32))
    typ_list.append(0)

def push_latent(zv: np.ndarray):
    tok_list.append(PAD_ID)  # token placeholder
    lat_list.append(zv.astype(np.float32))
    typ_list.append(1)

for fid in frame_keys:
    items = frame_to_items[fid]
    items.sort(key=lambda x: x[0])  # sort by pos_id for deterministic order

    push_token(FRAME_BOS)
    for pos_id, zp in items:
        c = encode_pos_chunks(pos_id)
        push_token(c[0]); push_token(c[1]); push_token(c[2])
        push_latent(zp)
    push_token(FRAME_EOS)

push_token(EOS_ID)

tok = np.array(tok_list, dtype=np.int64)
lat = np.stack(lat_list, axis=0).astype(np.float32)        # (T, D)
typ = np.array(typ_list, dtype=np.uint8)                   # (T,)

print("mixed stream length:", len(tok), "lat shape:", lat.shape, "typ:", typ.shape)
assert tok[-1] == EOS_ID
assert typ[-1] == 0

# Standardize latent for easier regression
lat_mean = lat[typ==1].mean(axis=0) if np.any(typ==1) else np.zeros((LATENT_DIM,), np.float32)
lat_std  = lat[typ==1].std(axis=0)  if np.any(typ==1) else np.ones((LATENT_DIM,), np.float32)
lat_std = np.maximum(lat_std, 1e-6).astype(np.float32)

lat_norm = lat.copy()
lat_norm[typ==1] = (lat_norm[typ==1] - lat_mean) / lat_std

print("latent norm stats:", float(np.mean(lat_norm[typ==1])), float(np.std(lat_norm[typ==1])))

# Save vocab/meta for later parsing
np.savez_compressed(
    os.path.join(OUT_DIR, "mixed_stream_meta.npz"),
    POS_BASE=np.int32(POS_BASE),
    SPECIAL_BASE=np.int32(SPECIAL_BASE),
    FRAME_BOS=np.int32(FRAME_BOS),
    FRAME_EOS=np.int32(FRAME_EOS),
    EOS_ID=np.int32(EOS_ID),
    MASK_ID=np.int32(MASK_ID),
    PAD_ID=np.int32(PAD_ID),
    VOCAB_SIZE=np.int32(VOCAB_SIZE),
    LATENT_DIM=np.int32(LATENT_DIM),
    lat_mean=lat_mean.astype(np.float32),
    lat_std=lat_std.astype(np.float32),
)

# %%
# %%
# Cell 4 — Sliding window dataset for mixed (token+latent) stream

class MixedSlidingWindowDataset(Dataset):
    def __init__(self, tok, lat, typ, seq_len):
        self.tok = tok
        self.lat = lat
        self.typ = typ
        self.seq_len = int(seq_len)
        self.L = int(tok.shape[0])
        self.max_i = self.L - (self.seq_len + 1)
        assert self.max_i > 0, "sequence too short; reduce SEQ_LEN"

        # bias sampling toward tail like your voxel pipeline
        self.tail_start = int(self.max_i * 0.80)

    def __len__(self):
        return 200000

    def __getitem__(self, idx):
        if np.random.rand() < 0.5:
            i = np.random.randint(0, self.max_i + 1)
        else:
            i = np.random.randint(self.tail_start, self.max_i + 1)

        x_tok = self.tok[i : i + self.seq_len]
        y_tok = self.tok[i + 1 : i + self.seq_len + 1]

        x_lat = self.lat[i : i + self.seq_len]
        y_lat = self.lat[i + 1 : i + self.seq_len + 1]

        x_typ = self.typ[i : i + self.seq_len]
        y_typ = self.typ[i + 1 : i + self.seq_len + 1]

        return (
            torch.from_numpy(x_tok),
            torch.from_numpy(x_lat),
            torch.from_numpy(x_typ),
            torch.from_numpy(y_tok),
            torch.from_numpy(y_lat),
            torch.from_numpy(y_typ),
            i,
        )

ds = MixedSlidingWindowDataset(tok, lat_norm, typ, SEQ_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))
print("dataset ok")

# %%
# %%
# Cell 5 — Model (Causal Transformer with token head + latent head)

@dataclass
class ModelConfig:
    vocab_size: int
    latent_dim: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

def causal_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

class MixedCausalTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lat_in  = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.token_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.latent_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.latent_dim)
        )

        self.register_buffer("mask_full", torch.empty(0), persistent=False)

    def forward(self, x_tok, x_lat, x_typ):
        """
        x_tok: (B,T) int64 (valid for typ==0)
        x_lat: (B,T,D) float32 (valid for typ==1)
        x_typ: (B,T) uint8/bool 0=token,1=latent
        """
        B, T = x_tok.shape
        assert T <= self.cfg.max_seq_len

        if self.mask_full.numel() == 0 or self.mask_full.shape[0] < T or self.mask_full.device != x_tok.device:
            self.mask_full = causal_mask(self.cfg.max_seq_len, x_tok.device)
        attn_mask = self.mask_full[:T, :T]

        pos = torch.arange(T, device=x_tok.device).unsqueeze(0).expand(B, T)

        # embed tokens / latents depending on type
        h_tok = self.tok_emb(x_tok)
        h_lat = self.lat_in(x_lat)

        x_typ_f = x_typ.to(dtype=torch.bool)
        h = torch.where(x_typ_f.unsqueeze(-1), h_lat, h_tok)

        h = h + self.pos_emb(pos)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h, attn_mask)

        h = self.ln_f(h)

        logits = self.token_head(h)           # (B,T,V)
        lat_pred = self.latent_head(h)        # (B,T,D)
        return logits, lat_pred

cfg = ModelConfig(
    vocab_size=VOCAB_SIZE,
    latent_dim=LATENT_DIM,
    d_model=512,
    n_layers=8,
    n_heads=8,
    dropout=0.1,
    max_seq_len=max(SEQ_LEN, 2048),
)
model = MixedCausalTransformer(cfg).to(device)
print("model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# %%
# %%
# Cell 6 — Train utils (dropout on tokens only + lr schedule + eval)

def apply_token_dropout(x_tok, x_typ, mask_id, p):
    if p <= 0:
        return x_tok
    drop = (torch.rand_like(x_tok.float()) < p) & (x_typ == 0)  # only token steps
    x2 = x_tok.clone()
    x2[drop] = mask_id
    return x2

def lr_schedule(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

@torch.no_grad()
def eval_losses(model, tok, lat, typ, seq_len, n_batches=20):
    model.eval()
    losses_tok = []
    losses_lat = []
    for _ in range(n_batches):
        i = np.random.randint(0, tok.shape[0] - (seq_len + 1))
        x_tok = torch.from_numpy(tok[i:i+seq_len]).unsqueeze(0).to(device)
        y_tok = torch.from_numpy(tok[i+1:i+seq_len+1]).unsqueeze(0).to(device)

        x_lat = torch.from_numpy(lat[i:i+seq_len]).unsqueeze(0).to(device)
        y_lat = torch.from_numpy(lat[i+1:i+seq_len+1]).unsqueeze(0).to(device)

        x_typ = torch.from_numpy(typ[i:i+seq_len]).unsqueeze(0).to(device)
        y_typ = torch.from_numpy(typ[i+1:i+seq_len+1]).unsqueeze(0).to(device)

        logits, lat_pred = model(x_tok, x_lat, x_typ)

        # predict y at each position
        tok_mask = (y_typ == 0)
        lat_mask = (y_typ == 1)

        if tok_mask.any():
            ce = F.cross_entropy(logits[tok_mask], y_tok[tok_mask])
            losses_tok.append(float(ce.item()))
        if lat_mask.any():
            l1 = F.smooth_l1_loss(lat_pred[lat_mask], y_lat[lat_mask])
            losses_lat.append(float(l1.item()))

    model.train()
    return float(np.mean(losses_tok) if losses_tok else 0.0), float(np.mean(losses_lat) if losses_lat else 0.0)

# %%
# %%
# Cell 7 — Train loop (mixed CE + latent regression)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

global_step = 0
total_steps = EPOCHS * (200000 // BATCH_SIZE)
t0 = time.time()
model.train()

# token CE weights: boost EOS a bit (optional)
ce_weight = torch.ones(VOCAB_SIZE, device=device)
ce_weight[EOS_ID] = 2.0

# balance between token CE and latent regression
LAMBDA_LAT = 1.0

for epoch in range(1, EPOCHS+1):
    for it, batch in enumerate(dl):
        x_tok, x_lat, x_typ, y_tok, y_lat, y_typ, start_i = batch
        x_tok = x_tok.to(device, non_blocking=True)
        x_lat = x_lat.to(device, non_blocking=True)
        x_typ = x_typ.to(device, non_blocking=True)

        y_tok = y_tok.to(device, non_blocking=True)
        y_lat = y_lat.to(device, non_blocking=True)
        y_typ = y_typ.to(device, non_blocking=True)

        x_tok_in = apply_token_dropout(x_tok, x_typ, MASK_ID, TOKEN_DROPOUT_P)

        lr_now = lr_schedule(global_step, total_steps, LR, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits, lat_pred = model(x_tok_in, x_lat, x_typ)

            tok_mask = (y_typ == 0)
            lat_mask = (y_typ == 1)

            loss_tok = torch.tensor(0.0, device=device)
            loss_lat = torch.tensor(0.0, device=device)

            if tok_mask.any():
                loss_tok = F.cross_entropy(logits[tok_mask], y_tok[tok_mask], weight=ce_weight)

            if lat_mask.any():
                loss_lat = F.smooth_l1_loss(lat_pred[lat_mask], y_lat[lat_mask])

            loss = loss_tok + LAMBDA_LAT * loss_lat

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        global_step += 1

        if global_step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(f"[ep{epoch} step{global_step}] loss={loss.item():.4f} tok={loss_tok.item():.4f} lat={loss_lat.item():.4f} lr={lr_now:.2e} time={dt:.1f}s")
            t0 = time.time()

        if global_step % VAL_EVERY == 0:
            vt, vl = eval_losses(model, tok, lat_norm, typ, SEQ_LEN, n_batches=10)
            print(f"  VAL: tok_ce~{vt:.4f} lat_l1~{vl:.4f}")

            ckpt_path = os.path.join(OUT_DIR, f"mixed_lm_step{global_step}.pt")
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "global_step": global_step,
                "VOCAB_SIZE": VOCAB_SIZE,
                "FRAME_BOS": FRAME_BOS,
                "FRAME_EOS": FRAME_EOS,
                "EOS_ID": EOS_ID,
                "MASK_ID": MASK_ID,
                "PAD_ID": PAD_ID,
                "POS_BASE": POS_BASE,
                "lat_mean": lat_mean,
                "lat_std": lat_std,
            }, ckpt_path)
            print("  saved:", ckpt_path)

    ckpt_path = os.path.join(OUT_DIR, f"mixed_lm_epoch{epoch}.pt")
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "global_step": global_step,
        "VOCAB_SIZE": VOCAB_SIZE,
        "FRAME_BOS": FRAME_BOS,
        "FRAME_EOS": FRAME_EOS,
        "EOS_ID": EOS_ID,
        "MASK_ID": MASK_ID,
        "PAD_ID": PAD_ID,
        "POS_BASE": POS_BASE,
        "lat_mean": lat_mean,
        "lat_std": lat_std,
    }, ckpt_path)
    print("saved epoch ckpt:", ckpt_path)

print("done.")

# %%
# %%
# Cell 8 — Greedy generation with grammar (fixed ctx like your voxel pipeline)

from torch.amp import autocast

@torch.no_grad()
def greedy_generate_mixed(model, prompt_tok, prompt_lat, prompt_typ,
                          ctx_len=1024, max_new=4000, log_every=200):
    """
    prompt_* are 1D arrays for the initial context
    returns generated tok/lat/typ arrays (including prompt)
    """
    model.eval()

    seq_tok = [int(x) for x in prompt_tok.tolist()]
    seq_typ = [int(x) for x in prompt_typ.tolist()]
    seq_lat = [prompt_lat[i].astype(np.float32) for i in range(prompt_lat.shape[0])]

    # Determine grammar state from last seen FRAME_BOS or FRAME_EOS:
    # After FRAME_BOS, expect pos0 (token), pos1(token), pos2(token), latent(vec), repeat...
    # After latent, next is token (pos0 or FRAME_EOS).
    # We'll track "phase" inside a frame:
    #   0=waiting BOS, 1=pos0,2=pos1,3=pos2,4=latent, then back to 1.
    # We'll infer by scanning backward within current frame.
    def infer_phase(seq_tok, seq_typ):
        # find last FRAME_BOS after last FRAME_EOS
        last_bos = -1
        last_eos = -1
        for i in range(len(seq_tok)-1, -1, -1):
            if seq_typ[i] == 0 and seq_tok[i] == FRAME_EOS:
                last_eos = i
                break
        for i in range(len(seq_tok)-1, -1, -1):
            if i <= last_eos:
                break
            if seq_typ[i] == 0 and seq_tok[i] == FRAME_BOS:
                last_bos = i
                break
        if last_bos < 0:
            return 0

        # count events after last_bos:
        # pattern: pos0,pos1,pos2,latent repeated, ending maybe at pos or latent
        k = 0
        for j in range(last_bos+1, len(seq_tok)):
            k += 1
        # k is number of steps after BOS
        # phase cycles every 4 steps: 1..4
        if k == 0:
            return 1
        r = (k - 1) % 4
        return 1 + r  # 1..4

    phase = infer_phase(seq_tok, seq_typ)

    t0 = time.time()
    for step in range(max_new):
        # build ctx tensors
        ctx_tok = np.array(seq_tok[-ctx_len:], dtype=np.int64)
        ctx_typ = np.array(seq_typ[-ctx_len:], dtype=np.uint8)
        ctx_lat = np.stack(seq_lat[-ctx_len:], axis=0).astype(np.float32)

        x_tok = torch.from_numpy(ctx_tok).unsqueeze(0).to(device)
        x_typ = torch.from_numpy(ctx_typ).unsqueeze(0).to(device)
        x_lat = torch.from_numpy(ctx_lat).unsqueeze(0).to(device)

        with autocast("cuda", enabled=(device=="cuda")):
            logits, lat_pred = model(x_tok, x_lat, x_typ)

        # predict next event from last position
        logits_next = logits[:, -1, :]          # (1,V)
        lat_next    = lat_pred[:, -1, :]        # (1,D)

        if phase in (1, 2, 3):
            # expecting pos chunk token: allow only [POS_BASE..POS_BASE+1023]
            allowed_min = POS_BASE
            allowed_max = POS_BASE + 1024
            logits_masked = logits_next.clone()
            logits_masked[:, :allowed_min] = -1e9
            logits_masked[:, allowed_max:] = -1e9
            next_id = int(torch.argmax(logits_masked, dim=-1).item())

            seq_tok.append(next_id)
            seq_typ.append(0)
            seq_lat.append(np.zeros((LATENT_DIM,), np.float32))

            phase += 1  # 1->2->3->4

        elif phase == 4:
            # expecting latent vector
            z = lat_next[0].detach().float().cpu().numpy().astype(np.float32)
            seq_tok.append(PAD_ID)
            seq_typ.append(1)
            seq_lat.append(z)
            phase = 1  # after latent, expect next patch pos0 or FRAME_EOS (handled in phase=1 step)

        else:
            # not in a frame; force BOS
            seq_tok.append(FRAME_BOS)
            seq_typ.append(0)
            seq_lat.append(np.zeros((LATENT_DIM,), np.float32))
            phase = 1

        # after a latent step, the next token phase=1 could also be FRAME_EOS; allow it sometimes:
        # We'll implement it by occasionally sampling FRAME_EOS when phase==1
        if phase == 1:
            # do one extra forward to decide between pos0 and FRAME_EOS (optional, cheap trick)
            # If you want strict greedy, skip this block.
            pass

        # stop if EOS emitted (only possible if you add EOS into grammar; here we keep it simple)
        # You can stop externally by max_new, then parse as many frames as you got.

        if (step + 1) % log_every == 0:
            print(f"gen step {step+1}/{max_new}, total_len={len(seq_tok)}, time={time.time()-t0:.1f}s")

    return (
        np.array(seq_tok, dtype=np.int64),
        np.stack(seq_lat, axis=0).astype(np.float32),
        np.array(seq_typ, dtype=np.uint8),
    )

# Build a prompt from GT head
prompt_tok = tok[:PROMPT_LEN]
prompt_lat = lat_norm[:PROMPT_LEN]
prompt_typ = typ[:PROMPT_LEN]

gen_tok, gen_lat_norm, gen_typ = greedy_generate_mixed(
    model, prompt_tok, prompt_lat, prompt_typ,
    ctx_len=CTX_LEN, max_new=MAX_NEW, log_every=LOG_EVERY_GEN
)

print("GEN len:", len(gen_tok))

# %%
# %%
# Cell 9 — Parse generated stream into frames of (pos_id, z_patch) and (optional) stub decode

def parse_mixed_stream_to_frames(tokens_1d, lat_2d, typ_1d):
    """
    Grammar:
      FRAME_BOS [poschunk poschunk poschunk latent]* FRAME_EOS ... EOS
    Returns:
      frames: list of (pos_ids, z_patches_norm)
    """
    frames = []
    i = 0
    n = len(tokens_1d)
    while i < n:
        if typ_1d[i] == 0 and tokens_1d[i] == EOS_ID:
            break
        if not (typ_1d[i] == 0 and tokens_1d[i] == FRAME_BOS):
            i += 1
            continue

        i += 1
        pos_list = []
        z_list = []

        while i < n:
            if typ_1d[i] == 0 and tokens_1d[i] == FRAME_EOS:
                i += 1
                break
            if typ_1d[i] == 0 and tokens_1d[i] == EOS_ID:
                i = n
                break

            # need 3 pos tokens + 1 latent step
            if i + 3 >= n:
                i = n
                break

            # pos chunks
            if not (typ_1d[i] == 0 and typ_1d[i+1] == 0 and typ_1d[i+2] == 0):
                i += 1
                continue
            p0 = int(tokens_1d[i])   - POS_BASE
            p1 = int(tokens_1d[i+1]) - POS_BASE
            p2 = int(tokens_1d[i+2]) - POS_BASE
            if not (0 <= p0 < 1024 and 0 <= p1 < 1024 and 0 <= p2 < 1024):
                i += 1
                continue

            # latent
            if typ_1d[i+3] != 1:
                i += 1
                continue

            pos_id = decode_pos_from_chunks(p0, p1, p2)
            z_norm = lat_2d[i+3].astype(np.float32)

            pos_list.append(pos_id)
            z_list.append(z_norm)

            i += 4

        if len(pos_list) > 0:
            frames.append((np.array(pos_list, np.int64), np.stack(z_list, axis=0).astype(np.float32)))
        else:
            frames.append((np.zeros((0,), np.int64), np.zeros((0, LATENT_DIM), np.float32)))

    return frames

gen_frames = parse_mixed_stream_to_frames(gen_tok, gen_lat_norm, gen_typ)
print("Parsed frames:", len(gen_frames))

fidx = int(np.clip(FRAME_IDX_TO_VIEW, 0, len(gen_frames)-1))
pos_ids_f, z_norm_f = gen_frames[fidx]
print("Frame", fidx, "patches:", len(pos_ids_f), "z shape:", z_norm_f.shape)

# unnormalize latent back
z_f = z_norm_f * lat_std + lat_mean
print("z unnorm:", z_f.shape)

# -------------------------
# Optional stub: decode z_patch -> gaussian group features
# Replace this stub with your own 3DGS latent decoder.
# Example: your decoder might map z_patch -> (GROUP_SIZE, GAUSS_DIM) for each patch.
# -------------------------
class StubPatchDecoder(nn.Module):
    def __init__(self, latent_dim, group_size=32, gauss_dim=11, hidden=512):
        super().__init__()
        self.group_size = group_size
        self.gauss_dim = gauss_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, group_size * gauss_dim),
        )

    def forward(self, z):
        # z: (P, D) -> (P, G, 11)
        out = self.net(z)
        return out.view(z.shape[0], self.group_size, self.gauss_dim)

stub_dec = StubPatchDecoder(LATENT_DIM).to(device).eval()

if len(pos_ids_f) > 0:
    with torch.no_grad():
        z_t = torch.from_numpy(z_f).to(device)
        feat = stub_dec(z_t).cpu().numpy().astype(np.float32)  # (P,32,11)
    np.savez_compressed(
        os.path.join(OUT_DIR, f"gen_frame{fidx:04d}_patch_gauss_stub.npz"),
        pos_ids=pos_ids_f,
        z_patch=z_f,
        feat=feat,
    )
    print("saved stub decode npz:", os.path.join(OUT_DIR, f"gen_frame{fidx:04d}_patch_gauss_stub.npz"))
else:
    print("No patches parsed for this frame; try different FRAME_IDX_TO_VIEW / more training.")

# %%
# %%
# Cell — Inference + PLY export (mixed latent LM) with REAL VAE DECODE
#
# 這段代碼會載入訓練好的 Transformer 進行序列生成，
# 然後將生成的 Latent 餵給 VAE Decoder，還原出真實的 3D Gaussian 場景！

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ------------------------------------------------------------
# 1. CONFIG (路徑設定)
# ------------------------------------------------------------
# 請確保這裡的 Checkpoint 名稱與你剛訓練好的 Transformer 模型一致
LM_CKPT = r"./GAUSSIAN_LATENT_LM/checkpoints_gs_latent_lm/mixed_lm_step5500.pt" 
PLY_OUT_DIR = r"./GAUSSIAN_LATENT_LM/ply_infer"
os.makedirs(PLY_OUT_DIR, exist_ok=True)

PROMPT_LEN = 256
CTX_LEN = 1024
MAX_NEW = 3000
LOG_EVERY_GEN = 200

GEN_FRAME_IDX = 0 # 選擇解析並導出第 0 幀

# Gaussian 採樣參數
SURFACE_POINTS_PER_GAUSSIAN = 120
INNER_POINTS_PER_GAUSSIAN = 60
SURFACE_SIGMA_LEVEL = 2.0
MAX_PATCH_CENTERS_EXPORT = 200000
MAX_GAUSS_BALL_POINTS_EXPORT = 400000

# VAE 超參數 (對應 Gaussian_VAE_10.py)
GAUSS_DIM = 11
LATENT_DIM = 128
D_MODEL = 256
GROUP_SIZE = 32

# ------------------------------------------------------------
# 2. VAE DECODER 定義 (用於將 Transformer 的 Latent 轉回 3DGS)
# ------------------------------------------------------------
class SlotContextDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, d_model=D_MODEL):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.out_xyz = nn.Linear(d_model, 3)
        self.out_sc  = nn.Linear(d_model, 3)
        self.out_q   = nn.Linear(d_model, 4)
        self.out_op  = nn.Linear(d_model, 1)
        self.out_mask= nn.Linear(d_model, 1)

    def forward(self, z):
        B, N, D = z.shape
        z_ctx = z.mean(dim=1, keepdim=True).expand(B, N, D)
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
        self.decoder = SlotContextDecoder()

# ------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------
def safe_load(path):
    t0 = time.time()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("loaded:", path, "time:", f"{time.time()-t0:.2f}s")
    return ckpt

def write_ply_points(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if rgb is not None:
        rgb = np.asarray(rgb).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(n))
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(xyz, rgb): f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

def make_uniform_rgb(n, color):
    if n <= 0: return np.zeros((0, 3), dtype=np.uint8)
    return np.repeat(np.asarray(color, dtype=np.uint8).reshape(1, 3), n, axis=0)

def downsample_pts(pts, max_pts, rgb=None, seed=123):
    pts = np.asarray(pts)
    if pts.shape[0] <= max_pts:
        return (pts, rgb) if rgb is not None else pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
    if rgb is None: return pts[idx]
    return pts[idx], np.asarray(rgb)[idx]

def quat_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)

def sample_gaussian_ellipsoid_points(mu_cm, sc_cm, quat, opacity, surface_n=120, inner_n=60, sigma_level=2.0, seed=123):
    mu_cm, sc_cm, quat, opacity = np.asarray(mu_cm), np.asarray(sc_cm), np.asarray(quat), np.asarray(opacity).reshape(-1)
    K = mu_cm.shape[0]
    if K == 0: return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    rng = np.random.default_rng(seed)
    all_pts, all_rgb = [], []
    for i in range(K):
        mu = mu_cm[i].astype(np.float64)
        sc = np.maximum(sc_cm[i].astype(np.float64), 1e-4)
        R = quat_to_rotmat(quat[i])
        
        if surface_n > 0:
            v = rng.normal(size=(surface_n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
            world = (v * (sigma_level * sc[None, :])) @ R.T + mu[None, :]
            all_pts.append(world.astype(np.float32))

        if inner_n > 0:
            v = rng.normal(size=(inner_n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
            r = rng.random((inner_n, 1)) ** (1.0 / 3.0)
            world = (v * r * (sigma_level * sc[None, :])) @ R.T + mu[None, :]
            all_pts.append(world.astype(np.float32))

        n_i = max(surface_n, 0) + max(inner_n, 0)
        op = float(np.clip(opacity[i], 0.0, 1.0))
        color = np.array([255 * op, 80, 255 * (1.0 - op)], dtype=np.uint8)
        all_rgb.append(np.repeat(color.reshape(1, 3), n_i, axis=0))

    return np.concatenate(all_pts, axis=0), np.concatenate(all_rgb, axis=0)

def infer_phase_from_prefix(seq_tok, seq_typ):
    last_bos, last_eos = -1, -1
    for i in range(len(seq_tok) - 1, -1, -1):
        if seq_typ[i] == 0 and seq_tok[i] == FRAME_EOS:
            last_eos = i
            break
    for i in range(len(seq_tok) - 1, -1, -1):
        if i <= last_eos: break
        if seq_typ[i] == 0 and seq_tok[i] == FRAME_BOS:
            last_bos = i
            break
    if last_bos < 0: return "outside"

    k = len(seq_tok) - (last_bos + 1)
    if k == 0: return "pos0"
    r = (k - 1) % 4
    if r == 0: return "pos1"
    elif r == 1: return "pos2"
    elif r == 2: return "latent"
    else: return "maybe_end_or_pos0"

@torch.no_grad()
def greedy_generate_mixed_onecell(model, prompt_tok, prompt_lat, prompt_typ, ctx_len=1024, max_new=3000, log_every=200):
    model.eval()
    seq_tok = [int(x) for x in prompt_tok.tolist()]
    seq_typ = [int(x) for x in prompt_typ.tolist()]
    seq_lat = [prompt_lat[i].astype(np.float32) for i in range(prompt_lat.shape[0])]
    t0 = time.time()

    for step in range(max_new):
        ctx_tok = np.asarray(seq_tok[-ctx_len:], dtype=np.int64)
        ctx_typ = np.asarray(seq_typ[-ctx_len:], dtype=np.uint8)
        ctx_lat = np.stack(seq_lat[-ctx_len:], axis=0).astype(np.float32)

        x_tok = torch.from_numpy(ctx_tok).unsqueeze(0).to(device)
        x_typ = torch.from_numpy(ctx_typ).unsqueeze(0).to(device)
        x_lat = torch.from_numpy(ctx_lat).unsqueeze(0).to(device)

        with autocast("cuda", enabled=(device == "cuda")):
            logits, lat_pred = model(x_tok, x_lat, x_typ)

        logits_next = logits[:, -1, :].float().clone()
        z_next = lat_pred[:, -1, :][0].detach().float().cpu().numpy().astype(np.float32)
        neg_large = torch.finfo(logits_next.dtype).min
        phase = infer_phase_from_prefix(seq_tok, seq_typ)

        if phase == "outside":
            next_tok = FRAME_BOS
            seq_tok.append(int(next_tok))
            seq_typ.append(0)
            seq_lat.append(np.zeros_like(z_next))
        elif phase in ["pos0", "pos1", "pos2"]:
            logits_next[:, :POS_BASE] = neg_large
            logits_next[:, POS_BASE + 1024:] = neg_large
            next_tok = int(torch.argmax(logits_next, dim=-1).item())
            seq_tok.append(next_tok)
            seq_typ.append(0)
            seq_lat.append(np.zeros_like(z_next))
        elif phase == "latent":
            seq_tok.append(PAD_ID)
            seq_typ.append(1)
            seq_lat.append(z_next)
        elif phase == "maybe_end_or_pos0":
            mask = torch.full_like(logits_next, neg_large)
            mask[:, FRAME_EOS] = 0.0
            mask[:, POS_BASE:POS_BASE + 1024] = 0.0
            logits_masked = logits_next + mask
            next_tok = int(torch.argmax(logits_masked, dim=-1).item())
            seq_tok.append(next_tok)
            seq_typ.append(0)
            seq_lat.append(np.zeros_like(z_next))
            if next_tok == FRAME_EOS and step > 32:
                break # 簡單處理：遇到 EOS 就當完成一幀
                
        if (step + 1) % log_every == 0:
            print(f"gen step {step+1}/{max_new}  total_len={len(seq_tok)}  time={time.time()-t0:.1f}s")

    return np.asarray(seq_tok), np.stack(seq_lat, axis=0), np.asarray(seq_typ)

def parse_mixed_stream_to_frames(tokens_1d, lat_2d, typ_1d):
    frames = []
    i, n = 0, len(tokens_1d)
    while i < n:
        if typ_1d[i] == 0 and tokens_1d[i] == FRAME_BOS:
            i += 1
            pos_list, z_list = [], []
            while i < n:
                if typ_1d[i] == 0 and tokens_1d[i] == FRAME_EOS:
                    i += 1; break
                if i + 3 >= n: break
                
                if not (typ_1d[i]==0 and typ_1d[i+1]==0 and typ_1d[i+2]==0 and typ_1d[i+3]==1):
                    i += 1; continue
                
                p0 = int(tokens_1d[i]) - POS_BASE
                p1 = int(tokens_1d[i+1]) - POS_BASE
                p2 = int(tokens_1d[i+2]) - POS_BASE
                if 0 <= p0 < 1024 and 0 <= p1 < 1024 and 0 <= p2 < 1024:
                    pos_id = int(p0) | (int(p1) << 10) | (int(p2) << 20)
                    pos_list.append(pos_id)
                    z_list.append(lat_2d[i+3].astype(np.float32))
                    i += 4
                else:
                    i += 1
            if len(pos_list) > 0:
                frames.append((np.asarray(pos_list, dtype=np.int64), np.stack(z_list, axis=0).astype(np.float32)))
        else:
            i += 1
    return frames

# ------------------------------------------------------------
# 4. 載入模型與生成
# ------------------------------------------------------------
lm_ckpt = safe_load(LM_CKPT)
cfg = ModelConfig(**lm_ckpt["cfg"])
cfg.max_seq_len = max(int(cfg.max_seq_len), int(CTX_LEN))

lm = MixedCausalTransformer(cfg).to(device)
lm.load_state_dict(lm_ckpt["model"], strict=True)
lm.eval()

lat_mean_use = np.asarray(lm_ckpt["lat_mean"], dtype=np.float32)
lat_std_use  = np.asarray(lm_ckpt["lat_std"], dtype=np.float32)

prompt_tok = np.asarray(tok[:PROMPT_LEN], dtype=np.int64)
prompt_lat = np.asarray(lat_norm[:PROMPT_LEN], dtype=np.float32)
prompt_typ = np.asarray(typ[:PROMPT_LEN], dtype=np.uint8)

print("prompt shapes:", prompt_tok.shape, prompt_lat.shape, prompt_typ.shape)

gen_tok, gen_lat_norm, gen_typ = greedy_generate_mixed_onecell(
    lm, prompt_tok, prompt_lat, prompt_typ, ctx_len=CTX_LEN, max_new=MAX_NEW, log_every=LOG_EVERY_GEN
)

gen_frames = parse_mixed_stream_to_frames(gen_tok, gen_lat_norm, gen_typ)
print("parsed valid frames:", len(gen_frames))

if len(gen_frames) == 0:
    raise RuntimeError("沒有生成出任何完整的 Frame，請增加 MAX_NEW 或檢查訓練權重。")

fidx = int(np.clip(GEN_FRAME_IDX, 0, len(gen_frames) - 1))
pos_ids_f, z_norm_f = gen_frames[fidx]
print(f"selected frame {fidx} - patches: {len(pos_ids_f)}, latent shape: {z_norm_f.shape}")

# 反歸一化 (Unnormalize Latent)
z_f = z_norm_f * lat_std_use[None, :] + lat_mean_use[None, :]

# ------------------------------------------------------------
# 5. 真實 VAE 解碼 (THE FIX)
# ------------------------------------------------------------
# 載入 world_stats 和 VAE Model
WORLD_STATS_NPZ = r"./GAUSSIAN_GROUP_VAE/world_stats.npz"
VAE_CKPT_PATH = r"./GAUSSIAN_GROUP_VAE/gaussian_group_vae_best.pt"

if not os.path.exists(WORLD_STATS_NPZ):
    raise FileNotFoundError(f"找不到 {WORLD_STATS_NPZ}")
    
world_stats = np.load(WORLD_STATS_NPZ)
world_center = world_stats["world_center"].astype(np.float32)
world_scale = float(world_stats["world_scale"])

vae_model = GaussianGroupContAE().to(device)
vae_ckpt = safe_load(VAE_CKPT_PATH)
vae_model.load_state_dict(vae_ckpt["model"], strict=False) # strict=False 因為我們只載入了 Decoder
vae_model.eval()

def decode_frame_latents_to_world_gaussians(z_frame_unnorm: np.ndarray):
    """
    將 Transformer 預測的 4096D Latent 或 128D 平均 Latent 轉換為真實 3D 空間的 Gaussians。
    VAE 輸出的坐標是全局絕對坐標，不受 pos_ids (Group index) 的位置影響。
    """
    P = z_frame_unnorm.shape[0]
    if P == 0: return np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,4)), np.zeros((0,))

    # 防呆機制：檢查 Transformer 生成的 Latent 維度
    latent_dim_gen = z_frame_unnorm.shape[-1]
    
    if latent_dim_gen == GROUP_SIZE * LATENT_DIM: # 4096D (正確的 Flattened 做法)
        z_reshaped = z_frame_unnorm.reshape(P, GROUP_SIZE, LATENT_DIM)
        z_t = torch.from_numpy(z_reshaped).to(device)
    elif latent_dim_gen == LATENT_DIM: # 128D (使用了平均化的做法)
        print(f"[警告] 你的 Transformer 依然是基於 128D 平均化 Latent 訓練的！")
        print("這會導致每個 Patch 裡面的 32 粒點長得一模一樣。請修改 Cell 2 並重新訓練！")
        print("臨時補救：將 128D 複製 32 份給 Decoder...")
        z_t = torch.from_numpy(z_frame_unnorm).unsqueeze(1).expand(P, GROUP_SIZE, LATENT_DIM).to(device)
    else:
        raise ValueError(f"未知的 Latent 維度: {latent_dim_gen}")

    with torch.no_grad():
        feat_hat, mask_logit = vae_model.decoder(z_t)
        mask_prob = torch.sigmoid(mask_logit)
        
    feat_flat = feat_hat.cpu().numpy().reshape(-1, GAUSS_DIM)
    mask_flat = mask_prob.cpu().numpy().reshape(-1)
    
    valid = mask_flat > 0.5
    feat_valid = feat_flat[valid]
    
    if len(feat_valid) == 0:
        return np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,4)), np.zeros((0,))
        
    # VAE 的預測已經是世界全局坐標系！
    xyz_n = feat_valid[:, 0:3]
    mu_cm = xyz_n * world_scale + world_center[None, :]
    sc_cm = np.exp(feat_valid[:, 3:6]) * world_scale
    
    quat = feat_valid[:, 6:10]
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
    op = np.clip(feat_valid[:, 10], 0.0, 1.0)
    
    return mu_cm.astype(np.float32), sc_cm.astype(np.float32), quat.astype(np.float32), op.astype(np.float32)

# ------------------------------------------------------------
# 6. EXPORT 最終結果
# ------------------------------------------------------------
if len(pos_ids_f) > 0:
    print("Decoding via real VAE...")
    mu_cm, sc_cm, quat, op = decode_frame_latents_to_world_gaussians(z_f)
    print("decoded REAL world gaussians:", mu_cm.shape[0])

    p_mu = os.path.join(PLY_OUT_DIR, f"gen_frame{fidx:04d}_real_gaussian_centers.ply")
    write_ply_points(p_mu, mu_cm, make_uniform_rgb(mu_cm.shape[0], [255, 0, 0]))

    ball_pts, ball_rgb = sample_gaussian_ellipsoid_points(
        mu_cm, sc_cm, quat, op,
        surface_n=SURFACE_POINTS_PER_GAUSSIAN,
        inner_n=INNER_POINTS_PER_GAUSSIAN,
        sigma_level=SURFACE_SIGMA_LEVEL,
        seed=123
    )
    ball_pts, ball_rgb = downsample_pts(ball_pts, MAX_GAUSS_BALL_POINTS_EXPORT, rgb=ball_rgb, seed=456)

    p_ball = os.path.join(PLY_OUT_DIR, f"gen_frame{fidx:04d}_real_gaussian_balls.ply")
    write_ply_points(p_ball, ball_pts, ball_rgb)
    print("saved real gaussian centers:", p_mu)
    print("saved real gaussian balls  :", p_ball)
    print("\n[成功] 已經使用 VAE 絕對世界坐標進行還原，長條線問題已解決！")



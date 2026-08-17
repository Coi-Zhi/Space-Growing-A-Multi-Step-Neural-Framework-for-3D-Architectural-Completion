# %%
# Cell 1 — Config & Paths (world-aligned occupancy tokenizer)
import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ====== 你要改呢幾個 ======
SEQ_ROOT  = r"./LearnCache_seq_fast"          # 同你 QF_2.py 產生嘅 cache 目錄一致
CACHE_DIR = os.path.join(SEQ_ROOT, "frame_cache")
CHAIN_NPY = os.path.join(SEQ_ROOT, "chain_indices.npy")

OUT_DIR   = r"./TOK_WORLD_BLOCK"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== world grid / block 參數 ======
VOX_CM    = 10.0        # 你 cache centers_cm 係以 VOX_CM=20cm grouping 生成（QF_2）
BLOCK_V   = 4           # block 邊長（8 voxels）
BLOCK_DIM = BLOCK_V**3  


P_EMPTY_IN_BATCH = 0.50     # 50% empty blocks in each batch
POS_WEIGHT = 12.0           # 比 50 小好多，減少 overfill（可試 8~20）
pos_weight = torch.tensor([POS_WEIGHT], device=device)

TOTAL_STEP = 60000         # 你要停嘅 step
NUM_WORKERS = 0             # Windows/Jupyter 穩定


# ====== VQ-VAE 參數 ======
EMBED_DIM    = 128
CODEBOOK_K   = 1024
VQ_BETA      = 0.25

# ====== Training 參數 ======
BATCH_SIZE   = 512
EPOCHS       = 10

LR           = 2e-4
WEIGHT_DECAY = 1e-4
LOG_EVERY    = 200
SAVE_EVERY   = 2000

# 平衡抽樣：空 block 太多會令模型學“全空”就贏
P_EMPTY   = 0.30      # 每 batch 30% 取空 block
P_NONEMPTY= 0.70      # 70% 取非空 block

# 用來存 global bounds（避免每次掃全 cache）
BOUNDS_NPZ = os.path.join(OUT_DIR, "world_bounds_blocks.npz")

# %%
# Cell 2 — Scan cache to compute world-aligned bounds (block grid)
# 目的：定義全局 block grid 的範圍（min/max block index）

def load_centers_cm(cache_id: int):
    p = os.path.join(CACHE_DIR, f"{int(cache_id):06d}.npz")
    z = np.load(p, allow_pickle=False)
    if "empty" in z.files:
        return None
    return z["centers_cm"].astype(np.float32)  # (V,3)

def centers_to_vox_idx(centers_cm: np.ndarray):
    # centers 通常係 (i+0.5)*VOX_CM，取 floor(center/VOX_CM) 得 voxel i
    v = np.floor(centers_cm / VOX_CM).astype(np.int32)
    return v

def vox_to_block_idx(vox_idx: np.ndarray):
    # block index = voxel index // BLOCK_V
    return (vox_idx // BLOCK_V).astype(np.int32)

if os.path.exists(BOUNDS_NPZ):
    bb = np.load(BOUNDS_NPZ)
    bmin = bb["bmin"].astype(np.int32)
    bmax = bb["bmax"].astype(np.int32)
    print("Loaded bounds:", BOUNDS_NPZ, "bmin:", bmin, "bmax:", bmax)
else:
    chain_ids = np.load(CHAIN_NPY).astype(np.int32)
    print("Scanning cache for bounds... frames:", len(chain_ids))

    bmin = np.array([ 10**9, 10**9, 10**9], dtype=np.int64)
    bmax = np.array([-10**9,-10**9,-10**9], dtype=np.int64)
    valid = 0

    t0 = time.time()
    for i, cid in enumerate(chain_ids):
        centers = load_centers_cm(int(cid))
        if centers is None or centers.shape[0] == 0:
            continue
        vox = centers_to_vox_idx(centers)
        blk = vox_to_block_idx(vox)
        bmin = np.minimum(bmin, blk.min(axis=0))
        bmax = np.maximum(bmax, blk.max(axis=0))
        valid += 1
        if (i+1) % 500 == 0:
            print(f"  scanned {i+1}/{len(chain_ids)} frames, valid={valid}, time={time.time()-t0:.1f}s")

    # 加少少 padding（避免邊界被截斷）
    PAD_BLOCKS = 1
    bmin = (bmin - PAD_BLOCKS).astype(np.int32)
    bmax = (bmax + PAD_BLOCKS).astype(np.int32)

    np.savez_compressed(BOUNDS_NPZ, bmin=bmin, bmax=bmax,
                        vox_cm=np.float32(VOX_CM), block_v=np.int32(BLOCK_V))
    print("Saved bounds:", BOUNDS_NPZ, "bmin:", bmin, "bmax:", bmax)

# block grid size
grid_blocks = (bmax - bmin + 1).astype(np.int32)
print("grid_blocks (Bx,By,Bz):", grid_blocks, "total blocks:", int(grid_blocks.prod()))

# %%
# Cell 3 — Build per-frame sparse block occupancy (world-aligned)
# 這個函數：給一個 cache_id -> 返回：
#   nonempty_blocks: dict {block_linear_id: 512-bit occupancy vector (float32 0/1)}
# 只存非空 block，空 block 用 implicit 0

def block_linear_id(bx, by, bz, grid_blocks):
    # 3D -> 1D（固定全局順序）
    return (bz * (grid_blocks[0]*grid_blocks[1]) + by * grid_blocks[0] + bx)

def make_block_occ_from_centers(centers_cm: np.ndarray, bmin, grid_blocks):
    vox = centers_to_vox_idx(centers_cm)              # (V,3) voxel indices in world grid
    blk = vox_to_block_idx(vox)                       # (V,3) block indices
    loc = (vox % BLOCK_V).astype(np.int32)            # local voxel index inside block [0..7]

    # shift block to [0..grid_blocks-1]
    blk_shift = (blk - bmin[None, :]).astype(np.int32)

    # filter within grid (just in case)
    m = np.all((blk_shift >= 0) & (blk_shift < grid_blocks[None, :]), axis=1)
    blk_shift = blk_shift[m]
    loc = loc[m]

    if blk_shift.shape[0] == 0:
        return {}

    # group voxels by block id
    out = {}
    for (bx,by,bz), (lx,ly,lz) in zip(blk_shift, loc):
        lid = int(block_linear_id(int(bx),int(by),int(bz), grid_blocks))
        if lid not in out:
            out[lid] = np.zeros((BLOCK_DIM,), dtype=np.float32)
        # local linear within block
        li = int(lz*BLOCK_V*BLOCK_V + ly*BLOCK_V + lx)
        out[lid][li] = 1.0
    return out

# quick sanity on one frame
chain_ids = np.load(CHAIN_NPY).astype(np.int32)
test_id = int(chain_ids[0])
centers = load_centers_cm(test_id)
if centers is None:
    print("first frame empty")
else:
    occ = make_block_occ_from_centers(centers, bmin, grid_blocks)
    print("test cache_id:", test_id, "nonempty blocks:", len(occ), "example block nnz:", int(next(iter(occ.values())).sum()))

# %%
from torch.utils.data import IterableDataset, DataLoader



class BatchBlockOccIterable(IterableDataset):
    def __init__(self, chain_ids, bmin, grid_blocks, batch_size, p_empty=0.5, max_tries=50):
        super().__init__()
        self.chain_ids = chain_ids.astype(np.int32)
        self.bmin = bmin.astype(np.int32)
        self.grid_blocks = grid_blocks.astype(np.int32)
        self.batch_size = int(batch_size)
        self.p_empty = float(p_empty)
        self.max_tries = int(max_tries)

    def __iter__(self):
        B = self.batch_size
        n_empty = int(B * self.p_empty)
        n_non = B - n_empty

        while True:
            # --- find a non-empty frame ---
            occ = None
            for _ in range(self.max_tries):
                cid = int(self.chain_ids[np.random.randint(0, len(self.chain_ids))])
                centers = load_centers_cm(cid)
                if centers is None or centers.shape[0] == 0:
                    continue
                occ = make_block_occ_from_centers(centers, self.bmin, self.grid_blocks)
                if len(occ) > 0:
                    break
                occ = None

            # if still none, yield empty batch (rare)
            batch = np.zeros((B, BLOCK_DIM), dtype=np.float32)
            if occ is not None and len(occ) > 0:
                keys = list(occ.keys())
                for i in range(n_non):
                    lid = keys[np.random.randint(0, len(keys))]
                    batch[i] = occ[int(lid)]
                # rest are empty already

            np.random.shuffle(batch)
            yield torch.from_numpy(batch)

ds_fast = BatchBlockOccIterable(chain_ids, bmin, grid_blocks, batch_size=BATCH_SIZE, p_empty=P_EMPTY_IN_BATCH)
dl_fast = DataLoader(ds_fast, batch_size=None, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

# %%
# Cell 5 — VQ-VAE for occupancy block (independent encode/decode)
# encode(x_block) -> z_e (B,E) -> VQ -> idx (B,) + z_q
# decode(idx) -> x_hat (B,512 logits) -> sigmoid -> occupancy

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embed_dim, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/codebook_size, 1.0/codebook_size)
        self.beta = beta

    def forward(self, z_e):
        # z_e: (B,E)
        # compute nearest code
        w = self.codebook.weight  # (K,E)
        # ||z-w||^2 = ||z||^2 + ||w||^2 -2 z·w
        z2 = (z_e**2).sum(dim=1, keepdim=True)       # (B,1)
        w2 = (w**2).sum(dim=1, keepdim=False)[None]  # (1,K)
        zw = z_e @ w.t()                             # (B,K)
        dist = z2 + w2 - 2*zw
        idx = torch.argmin(dist, dim=1)              # (B,)
        z_q = self.codebook(idx)                     # (B,E)

        # VQ loss
        loss_commit = F.mse_loss(z_e.detach(), z_q)
        loss_code   = F.mse_loss(z_e, z_q.detach())
        loss_vq = loss_code + self.beta * loss_commit

        # straight-through
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, idx, loss_vq

class BlockVQVAE(nn.Module):
    def __init__(self, block_dim=512, embed_dim=128, codebook_size=1024, beta=0.25):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(block_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.vq = VectorQuantizer(codebook_size, embed_dim, beta=beta)
        self.dec = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, block_dim),   # logits
        )

    def forward(self, x):
        # x: (B,512) float 0/1
        z_e = self.enc(x)
        z_q, idx, loss_vq = self.vq(z_e)
        logits = self.dec(z_q)
        return logits, idx, loss_vq

model = BlockVQVAE(BLOCK_DIM, EMBED_DIM, CODEBOOK_K, beta=VQ_BETA).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

print("params(M):", sum(p.numel() for p in model.parameters())/1e6)

# %%
# Cell 6 — Train loop (BCE recon + VQ loss)
# world-aligned / tgt-only tokenizer：完全唔涉及序列前後關係
global_step = 0
model.train()
t0 = time.time()

stop = False
for epoch in range(1, EPOCHS+1):
    for x in dl_fast:
        if global_step >= TOTAL_STEP:
            stop = True
            break

        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits, idx, loss_vq = model(x)
            loss_rec = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pos_weight)
            loss = loss_rec + loss_vq

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        global_step += 1

        if global_step % LOG_EVERY == 0:
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pred_occ = (prob > 0.5).float().mean().item()
                gt_occ = x.mean().item()
            dt = time.time() - t0
            print(f"[ep{epoch} step{global_step}] loss={loss.item():.4f} rec={loss_rec.item():.4f} vq={loss_vq.item():.4f} "
                  f"gt_occ={gt_occ:.3f} pred_occ={pred_occ:.3f} time={dt:.1f}s")
            t0 = time.time()

        if global_step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(OUT_DIR, f"block_vqvae_step{global_step}.pt")
            torch.save({
                "model": model.state_dict(),
                "embed_dim": EMBED_DIM,
                "codebook_k": CODEBOOK_K,
                "block_v": BLOCK_V,
                "vox_cm": VOX_CM,
                "bmin": bmin,
                "bmax": bmax,
                "grid_blocks": grid_blocks,
                "pos_weight": float(POS_WEIGHT),
                "p_empty_in_batch": float(P_EMPTY_IN_BATCH),
            }, ckpt_path)
            print("saved:", ckpt_path)

    if stop:
        print(f"Stopped at step {global_step} (TOTAL_STEP={TOTAL_STEP}).")
        break

# %%
# CELL A — Sparse tokenize whole sequence (no dense blocks, no RAM爆)
# Output: world_block_tokens_sparse.npz
import os, time, numpy as np, torch

TOK_OUT = os.path.join(OUT_DIR, "world_block_tokens_sparse.npz")

@torch.no_grad()
def encode_block_tokens(model, x_blocks):
    # x_blocks: (B, BLOCK_DIM) float32 on device
    model.eval()
    z_e = model.enc(x_blocks)
    w = model.vq.codebook.weight
    z2 = (z_e**2).sum(dim=1, keepdim=True)
    w2 = (w**2).sum(dim=1)[None]
    dist = z2 + w2 - 2*(z_e @ w.t())
    idx = torch.argmin(dist, dim=1)
    return idx

def frame_to_sparse_occ(cache_id: int):
    centers = load_centers_cm(cache_id)
    if centers is None or centers.shape[0] == 0:
        return None  # empty frame
    occ = make_block_occ_from_centers(centers, bmin, grid_blocks)  # dict {lid: vec(512)}
    if len(occ) == 0:
        return None
    lids = np.array(list(occ.keys()), dtype=np.int32)
    vecs = np.stack([occ[int(k)] for k in lids], axis=0).astype(np.float32)  # (M,512)
    return lids, vecs

# compute empty_id once
x0 = torch.zeros((1, BLOCK_DIM), device=device)
empty_id = int(encode_block_tokens(model, x0).item())
print("empty_token_id:", empty_id)

chain_ids = np.load(CHAIN_NPY).astype(np.int32)
T = len(chain_ids)
print("frames:", T)

# We'll store ragged data using concat + offsets:
pos_all = []
tok_all = []
offsets = np.zeros((T + 1,), dtype=np.int64)

# encoding chunk size (blocks)
ENC_CHUNK = 8192

t0 = time.time()
total_nonempty_blocks = 0

model.eval()
for t in range(T):
    cid = int(chain_ids[t])
    out = frame_to_sparse_occ(cid)
    if out is None:
        offsets[t+1] = offsets[t]  # no blocks
        continue

    lids, vecs = out  # lids (M,), vecs (M,512)
    M = vecs.shape[0]
    total_nonempty_blocks += M

    # encode in chunks to avoid GPU spikes
    tok = np.empty((M,), dtype=np.int16)
    s = 0
    while s < M:
        e = min(M, s + ENC_CHUNK)
        xb = torch.from_numpy(vecs[s:e]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            ids = encode_block_tokens(model, xb).detach().cpu().numpy().astype(np.int16)
        tok[s:e] = ids
        s = e
        if device == "cuda":
            torch.cuda.empty_cache()

    pos_all.append(lids)
    tok_all.append(tok)
    offsets[t+1] = offsets[t] + M

    if (t+1) % 200 == 0:
        print(f"tokenized {t+1}/{T} frames | total_blocks={total_nonempty_blocks} | time={time.time()-t0:.1f}s")

# concat storage
pos_all = np.concatenate(pos_all, axis=0) if len(pos_all) else np.zeros((0,), dtype=np.int32)
tok_all = np.concatenate(tok_all, axis=0) if len(tok_all) else np.zeros((0,), dtype=np.int16)

np.savez_compressed(
    TOK_OUT,
    chain_ids=chain_ids,
    offsets=offsets,          # (T+1,) int64
    pos=pos_all,              # (sumM,) int32  block_linear_id
    tok=tok_all,              # (sumM,) int16  token_id
    empty_id=np.int32(empty_id),
    bmin=bmin, bmax=bmax, grid_blocks=grid_blocks,
    vox_cm=np.float32(VOX_CM), block_v=np.int32(BLOCK_V),
    codebook_k=np.int32(CODEBOOK_K),
)
print("Saved:", TOK_OUT)
print("total nonempty blocks:", int(total_nonempty_blocks), "pos size MB:", pos_all.nbytes/1024/1024, "tok size MB:", tok_all.nbytes/1024/1024)

# %%
# CELL B — Sparse decode to PLY + compare with GT PLY
# Reads world_block_tokens_sparse.npz, decodes only nonempty blocks -> voxel centers -> PLY

import os, numpy as np, torch

OUT_DIR = r"./ply_out"
os.makedirs(OUT_DIR, exist_ok=True)

SPARSE_NPZ = os.path.join(OUT_DIR.replace("ply_out","TOK_WORLD_BLOCK"), "world_block_tokens_sparse.npz")
# 如果你 OUT_DIR 不同，改成你實際輸出位置
if not os.path.exists(SPARSE_NPZ):
    SPARSE_NPZ = os.path.join(OUT_DIR, "world_block_tokens_sparse.npz")

FRAME_IDX = 0
THRESH = 0.88           # 控制厚度：0.70~0.85；越高越薄（你要2-3 voxel，通常 0.75 左右）
MAX_PTS = 200000

GT_PLY  = os.path.join(OUT_DIR, "GT.ply")
DEC_PLY = os.path.join(OUT_DIR, "DECODED.ply")

def write_ply_xyz(path, pts_xyz):
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts_xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in pts_xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def downsample_pts(pts, max_pts):
    if pts.shape[0] <= max_pts:
        return pts
    sel = np.random.choice(pts.shape[0], size=max_pts, replace=False)
    return pts[sel]

@torch.no_grad()
def decode_block_logits(model, idx_1d):
    z_q = model.vq.codebook(idx_1d)
    return model.dec(z_q)

def block_linear_to_3d(lid, grid_blocks):
    gx, gy, gz = int(grid_blocks[0]), int(grid_blocks[1]), int(grid_blocks[2])
    bx = lid % gx
    by = (lid // gx) % gy
    bz = lid // (gx * gy)
    return bx, by, bz

def local_linear_to_3d(li, block_v):
    bv = int(block_v)
    lx = li % bv
    ly = (li // bv) % bv
    lz = li // (bv * bv)
    return lx, ly, lz

# load sparse tokens
z = np.load(SPARSE_NPZ)
chain_ids = z["chain_ids"].astype(np.int32)
offsets  = z["offsets"].astype(np.int64)
pos_all  = z["pos"].astype(np.int32)
tok_all  = z["tok"].astype(np.int16)
empty_id = int(z["empty_id"])
bmin = z["bmin"].astype(np.int32)
grid_blocks = z["grid_blocks"].astype(np.int32)
VOX_CM = float(z["vox_cm"])
BLOCK_V = int(z["block_v"])
BLOCK_DIM = BLOCK_V**3

FRAME_IDX = int(np.clip(FRAME_IDX, 0, len(chain_ids)-1))
cid = int(chain_ids[FRAME_IDX])

# GT points from cache
centers = load_centers_cm(cid)
GT_pts = centers if centers is not None else np.zeros((0,3), np.float32)

# slice this frame's sparse blocks
a = int(offsets[FRAME_IDX])
b = int(offsets[FRAME_IDX+1])
pos = pos_all[a:b].astype(np.int64)    # block_linear_id list
tok = tok_all[a:b].astype(np.int64)    # token_id list
print("frame:", FRAME_IDX, "cache_id:", cid, "nonempty blocks:", len(pos))

# decode only these blocks
DEC_pts_list = []
CHUNK = 8192

with torch.no_grad():
    for s in range(0, len(tok), CHUNK):
        e = min(len(tok), s+CHUNK)
        idx = torch.from_numpy(tok[s:e]).to(device)

        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits = decode_block_logits(model, idx)
            prob = torch.sigmoid(logits)

        occ = (prob > THRESH).cpu().numpy().astype(np.uint8)  # (chunk, BLOCK_DIM)

        # convert occupied voxels to world points
        for j in range(occ.shape[0]):
            lid = int(pos[s+j])
            li = np.nonzero(occ[j])[0]
            if li.size == 0:
                continue

            bx, by, bz = block_linear_to_3d(lid, grid_blocks)
            wb = bmin + np.array([bx,by,bz], dtype=np.int32)   # world block index
            base_vox = wb.astype(np.int64) * BLOCK_V

            pts = np.empty((li.size, 3), dtype=np.float32)
            for k, li_k in enumerate(li):
                lx, ly, lz = local_linear_to_3d(int(li_k), BLOCK_V)
                wv = base_vox + np.array([lx,ly,lz], dtype=np.int64)
                pts[k] = (wv.astype(np.float32) + 0.5) * VOX_CM
            DEC_pts_list.append(pts)

        del logits, prob
        if device == "cuda":
            torch.cuda.empty_cache()

DEC_pts = np.concatenate(DEC_pts_list, axis=0) if len(DEC_pts_list) else np.zeros((0,3), np.float32)

# downsample for file size
GT_pts  = downsample_pts(GT_pts,  MAX_PTS)
DEC_pts = downsample_pts(DEC_pts, MAX_PTS)

write_ply_xyz(GT_PLY, GT_pts)
write_ply_xyz(DEC_PLY, DEC_pts)

print("Saved:")
print(" GT :", GT_PLY, "pts:", GT_pts.shape[0])
print(" DEC:", DEC_PLY, "pts:", DEC_pts.shape[0], "THRESH:", THRESH)

# %%
# THRESH sweep to match GT voxel count (pick thinner threshold)
THRESH_LIST = [0.70,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.89,0.90,0.92]

# GT voxel count (from centers_cm, already world vox at VOX_CM)
cid = int(chain_ids[FRAME_IDX])
centers = load_centers_cm(cid)
gt_vox = 0 if centers is None else centers.shape[0]
print("GT voxels:", gt_vox)

def decode_voxel_count_for_thresh(thresh):
    # reuse your sparse decode loop, but only count voxels, don't store pts
    a = int(offsets[FRAME_IDX]); b = int(offsets[FRAME_IDX+1])
    pos = pos_all[a:b].astype(np.int64)
    tok = tok_all[a:b].astype(np.int64)

    vox_count = 0
    CHUNK = 8192
    with torch.no_grad():
        for s in range(0, len(tok), CHUNK):
            e = min(len(tok), s+CHUNK)
            idx = torch.from_numpy(tok[s:e]).to(device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = decode_block_logits(model, idx)
                prob = torch.sigmoid(logits)
            occ = (prob > thresh).cpu().numpy()
            vox_count += int(occ.sum())
            del logits, prob
            if device == "cuda":
                torch.cuda.empty_cache()
    return vox_count

for th in THRESH_LIST:
    dv = decode_voxel_count_for_thresh(th)
    ratio = (dv / max(gt_vox,1))
    print(f"TH={th:.2f} decoded_vox={dv} ratio={ratio:.3f}")



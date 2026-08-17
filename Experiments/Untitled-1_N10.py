# %%
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== main params (must match QF_2 meta) =====
VOX_CM = 20.0
PTS_PER_VOX = 256
VOX_PER_SAMPLE = 128  # M = query + ctx

# token budget split
Q_TOKENS = 96          # query tokens (empty input, predict occ+shape)
CTX_TOKENS = VOX_PER_SAMPLE - Q_TOKENS

# QF_2 outputs (long-seq cache)
SEQ_ROOT = r"K:\MasterEssay\Attempt_source_03\LearnCache_seq_fast"
CACHE_DIR = os.path.join(SEQ_ROOT, "frame_cache")
CHAIN_NPY = os.path.join(SEQ_ROOT, "chain_indices.npy")
SEQIDX_NPY = os.path.join(SEQ_ROOT, "seq_index.npy")
META_JSON = os.path.join(SEQ_ROOT, "meta.json")

assert os.path.exists(CACHE_DIR), CACHE_DIR
assert os.path.exists(CHAIN_NPY), CHAIN_NPY
assert os.path.exists(SEQIDX_NPY), SEQIDX_NPY

if os.path.exists(META_JSON):
    meta = json.load(open(META_JSON, "r", encoding="utf-8"))
    print("meta:", meta)

chain_ids = np.load(CHAIN_NPY).astype(np.int32)      # cache ids
seq_pairs = np.load(SEQIDX_NPY).astype(np.int32)     # (start, prefix_len)
print("chain len:", len(chain_ids), "seq pairs:", len(seq_pairs))
print("device:", device)

# %%
# %% [Cell: Seq Dataset Definition (Prefix->Next) with Relative Centering]

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache

# ---- Helper Functions (Packing/Unpacking Keys) ----
BIAS = 1 << 20
MASK = (1 << 21) - 1

def pack_key(x: int, y: int, z: int) -> np.int64:
    return np.int64(((x + BIAS) << 42) | ((y + BIAS) << 21) | (z + BIAS))

def unpack_key(k: np.int64):
    kk = int(k)
    x = ((kk >> 42) & MASK) - BIAS
    y = ((kk >> 21) & MASK) - BIAS
    z = (kk & MASK) - BIAS
    return x, y, z

def center_from_key(k: np.int64, vox_cm: float):
    x, y, z = unpack_key(k)
    return (np.array([x, y, z], np.float32) + 0.5) * float(vox_cm)

def neighbors_6_pack(k: np.int64):
    x, y, z = unpack_key(k)
    return [
        pack_key(x+1,y,z), pack_key(x-1,y,z),
        pack_key(x,y+1,z), pack_key(x,y-1,z),
        pack_key(x,y,z+1), pack_key(x,y,z-1),
    ]

# ---- Frame Loading with Cache ----
@lru_cache(maxsize=128)
def load_frame(cache_id: int):
    p = os.path.join(CACHE_DIR, f"{int(cache_id):06d}.npz")
    if not os.path.exists(p): 
        return None
    try:
        z = np.load(p, allow_pickle=False)
        if "empty" in z.files: 
            return None
        keys = z["keys_pack"].astype(np.int64)
        local = z["local_pts"].astype(np.float32)

        order = np.argsort(keys)
        return keys[order], local[order]
    except Exception as e:
        print(f"Error loading {p}: {e}")
        return None

def get_local_sorted(keys_sorted, local_sorted, k):
    idx = int(np.searchsorted(keys_sorted, k))
    if 0 <= idx < keys_sorted.shape[0] and keys_sorted[idx] == k:
        return local_sorted[idx]
    return None

def occ_nb6_count(k, occ_set):
    c = 0
    for nb in neighbors_6_pack(np.int64(k)):
        if np.int64(nb) in occ_set:
            c += 1
    return c

def weighted_sample_wo(items, weights, k, rng):
    # Efraimidis-Spirakis: key = U^(1/w)
    out = []
    keys = []
    for it, w in zip(items, weights):
        w = float(max(1e-6, w))
        u = rng.random()
        key = u ** (1.0 / w)
        keys.append((key, it))
    keys.sort(reverse=True, key=lambda t: t[0])
    return [it for _, it in keys[:k]]


# ---- Main Dataset Class ----
# %% [Cell 1] Dataset — quota + guaranteed hard negatives + returns soon_mask

import random
import numpy as np
import torch
from torch.utils.data import Dataset

# 你可以调呢几个数（建议先用呢套稳）
K_FUTURE = 3

MIN_NEG_Q   = 64            # ✅ 最少留几多个 hard neg
MAX_NEW_Q   = 64            # ✅ new positives 上限（防止 pos_ratio 爆）
MIN_NEW_Q   = 24            # ✅ new positives 下限（确保 occ 有信号）
MAX_SOON_Q  = 32            # ✅ soon-only 上限（可为 0；先保守）
MIN_SOON_Q  = 0

FRONTIER_BIAS = 0.7
FRONTIER2_MAX = 4096

def occ_nb6_count(k, occ_set):
    c = 0
    for nb in neighbors_6_pack(np.int64(k)):
        if np.int64(nb) in occ_set:
            c += 1
    return c

def weighted_sample_wo(items, weights, k, rng):
    out = []
    keys = []
    for it, w in zip(items, weights):
        w = float(max(1e-6, w))
        u = rng.random()
        key = u ** (1.0 / w)
        keys.append((key, it))
    keys.sort(reverse=True, key=lambda t: t[0])
    return [it for _, it in keys[:k]]

class PrefixNextVoxelDataset(Dataset):
    """
    returns:
      mem_local [M,N,3]
      tgt_local [M,N,3]
      centers_cm_rel [M,3]
      occ [M]
      qmsk [M]     (1=query)
      new_mask [M] (query & new_set)
      soon_mask[M] (query & (future_union - occ_set))
    """
    def __init__(self, chain_ids, seq_pairs, vox_cm=20.0, pts_per_vox=256,
                 M=256, Q=192, seed=1234):
        super().__init__()
        self.chain = chain_ids.astype(np.int32)
        self.pairs = seq_pairs.astype(np.int32)
        self.vox_cm = float(vox_cm)
        self.N = int(pts_per_vox)
        self.M = int(M)
        self.Q = int(Q)
        self.rng = random.Random(seed)

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        start, L = map(int, self.pairs[i])
        L = max(1, L)

        prefix_ids = self.chain[start : start+L]
        tgt_index  = start + L
        if tgt_index >= len(self.chain):
            return self._empty_sample()

        target_id  = int(self.chain[tgt_index])

        # prefix occ_set + frames
        occ_set = set()
        prefix_frames = []
        for cid in prefix_ids:
            pack = load_frame(int(cid))
            if pack:
                k_s, l_s = pack
                occ_set.update(k_s.tolist())
                prefix_frames.append((k_s, l_s))

        if not prefix_frames:
            return self._empty_sample()

        occ_set = set(np.int64(list(occ_set)))

        # target
        tgt_pack = load_frame(target_id)
        if not tgt_pack:
            return self._empty_sample()
        tgt_keys, tgt_local = tgt_pack
        tgt_set = set(np.int64(list(set(tgt_keys.tolist()))))

        new_set = set(np.int64(list(tgt_set - occ_set)))

        # future union
        future_union = set(tgt_set)
        for t in range(1, K_FUTURE):
            j = tgt_index + t
            if j >= len(self.chain):
                break
            pid = int(self.chain[j])
            ppack = load_frame(pid)
            if not ppack:
                continue
            pk, _ = ppack
            future_union.update(pk.tolist())
        future_union = set(np.int64(list(future_union)))

        soon_set = set(np.int64(list(future_union - occ_set)))
        soon_only = set(np.int64(list(soon_set - new_set)))

        # frontier pools for hard neg candidates
        frontier1 = set()
        for ok in occ_set:
            for nb in neighbors_6_pack(np.int64(ok)):
                nb = np.int64(nb)
                if nb not in occ_set:
                    frontier1.add(nb)

        frontier2 = set()
        if len(frontier1) > 0:
            f1_list = list(frontier1)
            self.rng.shuffle(f1_list)
            for k in f1_list[:min(len(f1_list), 2048)]:
                for nb in neighbors_6_pack(np.int64(k)):
                    nb = np.int64(nb)
                    if (nb not in occ_set) and (nb not in frontier1):
                        frontier2.add(nb)
                        if len(frontier2) >= FRONTIER2_MAX:
                            break
                if len(frontier2) >= FRONTIER2_MAX:
                    break

        # ---------- A) pick new queries (cap) ----------
        new_list = list(new_set)
        self.rng.shuffle(new_list)
        n_new = min(len(new_list), MAX_NEW_Q)
        n_new = max(n_new, min(MIN_NEW_Q, len(new_list)))

        # 先预留 neg 配额
        n_new = min(n_new, max(0, self.Q - MIN_NEG_Q - MIN_SOON_Q))
        new_q = [np.int64(k) for k in new_list[:n_new]]
        new_qset = set(new_q)

        # ---------- B) pick soon-only (small, optional) ----------
        soon_list = list(soon_only)
        self.rng.shuffle(soon_list)
        n_soon = min(len(soon_list), MAX_SOON_Q)
        n_soon = max(n_soon, min(MIN_SOON_Q, len(soon_list)))

        # 仍然要保留 neg quota
        n_soon = min(n_soon, max(0, self.Q - len(new_q) - MIN_NEG_Q))
        soon_q = [np.int64(k) for k in soon_list[:n_soon]]
        soon_qset = set(soon_q)

        # ---------- C) pick hard negatives: exclude soon_set entirely ----------
        pos_qset = new_qset | soon_qset
        neg_need = self.Q - len(new_q) - len(soon_q)
        neg_need = max(neg_need, MIN_NEG_Q)  # ✅ 强制留够 neg
        neg_need = min(neg_need, self.Q - len(new_q) - len(soon_q))

        f1 = [np.int64(k) for k in frontier1 if (np.int64(k) not in pos_qset) and (np.int64(k) not in soon_set)]
        f2 = [np.int64(k) for k in frontier2 if (np.int64(k) not in pos_qset) and (np.int64(k) not in soon_set)]

        w1 = [ (occ_nb6_count(k, occ_set) + 1) ** 2 for k in f1 ]
        w2 = [ (occ_nb6_count(k, occ_set) + 1) ** 2 for k in f2 ]

        def pick(items, weights, k):
            if k <= 0 or len(items) == 0:
                return []
            kk = min(len(items), k)
            out = weighted_sample_wo(items, weights, kk, self.rng)
            while len(out) < k:
                out.append(weighted_sample_wo(items, weights, 1, self.rng)[0])
            return out[:k]

        n1 = int(round(neg_need * FRONTIER_BIAS))
        n2 = neg_need - n1
        if len(f1) == 0:
            n1, n2 = 0, neg_need
        if len(f2) == 0:
            n1, n2 = neg_need, 0

        neg_q = pick(f1, w1, n1) + pick(f2, w2, n2)

        # fallback：如果 frontier 不够 hard neg，就 bbox random（排除 soon_set）
        if len(neg_q) < neg_need:
            occ_list = list(occ_set)
            if len(occ_list) > 0:
                xs, ys, zs = zip(*[unpack_key(k) for k in occ_list[:min(2000, len(occ_list))]])
                xmin, xmax = min(xs)-8, max(xs)+8
                ymin, ymax = min(ys)-8, max(ys)+8
                zmin, zmax = min(zs)-4, max(zs)+4
            else:
                xmin=ymin=zmin=-16
                xmax=ymax=zmax=16

            pool = set(neg_q)
            tries = 0
            while len(pool) < neg_need and tries < neg_need * 200:
                tries += 1
                rx = self.rng.randint(xmin, xmax)
                ry = self.rng.randint(ymin, ymax)
                rz = self.rng.randint(zmin, zmax)
                kk = np.int64(pack_key(rx, ry, rz))
                if (kk not in occ_set) and (kk not in soon_set) and (kk not in pos_qset):
                    pool.add(kk)
            neg_q = list(pool)
            if len(neg_q) > neg_need:
                neg_q = self.rng.sample(neg_q, neg_need)

        # final queries
        queries = new_q + soon_q + [np.int64(k) for k in neg_q[:neg_need]]
        queries = queries[:self.Q]
        self.rng.shuffle(queries)
        qset = set(queries)

        # context tokens（一定要有 ctx，不建议 M==Q）
        local_ctx = set()
        for qk in queries:
            for nb in neighbors_6_pack(np.int64(qk)):
                if np.int64(nb) in occ_set:
                    local_ctx.add(np.int64(nb))

        local_ctx = list(local_ctx)
        needed_c = self.M - len(queries)
        occ_list = list(occ_set - qset)

        if needed_c <= 0:
            tokens = queries[:self.M]
        else:
            if len(local_ctx) >= needed_c:
                final_ctx = self.rng.sample(local_ctx, needed_c)
            else:
                final_ctx = local_ctx
                rem = needed_c - len(final_ctx)
                if len(occ_list) >= rem:
                    final_ctx += self.rng.sample(occ_list, rem)
                else:
                    final_ctx += self.rng.choices(occ_list, k=rem) if occ_list else [np.int64(pack_key(0,0,0))]*rem
            tokens = queries + final_ctx
            self.rng.shuffle(tokens)

        # build tensors
        mem = np.zeros((self.M, self.N, 3), np.float32)
        tgt = np.zeros((self.M, self.N, 3), np.float32)
        cen = np.zeros((self.M, 3), np.float32)
        occ = np.zeros((self.M,), np.float32)
        qmsk = np.zeros((self.M,), np.float32)
        newm = np.zeros((self.M,), np.float32)
        soonm = np.zeros((self.M,), np.float32)

        prefix_frames_rev = list(reversed(prefix_frames))

        for ti, k_int in enumerate(tokens):
            k = np.int64(k_int)
            cen[ti] = center_from_key(k, self.vox_cm)

            is_query = (k in qset)
            qmsk[ti] = 1.0 if is_query else 0.0

            if not is_query:
                for ks, ls in prefix_frames_rev:
                    loc = get_local_sorted(ks, ls, k)
                    if loc is not None:
                        mem[ti] = loc
                        break

            loc_t = get_local_sorted(tgt_keys, tgt_local, k)
            if loc_t is not None:
                tgt[ti] = loc_t
                occ[ti] = 1.0
            if (not is_query) and (k in occ_set):
                occ[ti] = 1.0

            if is_query and (k in new_set):
                newm[ti] = 1.0
            if is_query and (k in soon_set):
                soonm[ti] = 1.0

        valid_mask = (qmsk < 0.5)
        centroid = np.mean(cen[valid_mask], axis=0) if valid_mask.any() else np.mean(cen, axis=0)
        cen_relative = cen - centroid[None, :]

        return (torch.from_numpy(mem),
                torch.from_numpy(tgt),
                torch.from_numpy(cen_relative),
                torch.from_numpy(occ),
                torch.from_numpy(qmsk),
                torch.from_numpy(newm),
                torch.from_numpy(soonm))

    def _empty_sample(self):
        return (torch.zeros((self.M, self.N, 3)),
                torch.zeros((self.M, self.N, 3)),
                torch.zeros((self.M, 3)),
                torch.zeros((self.M,)),
                torch.zeros((self.M,)),
                torch.zeros((self.M,)),
                torch.zeros((self.M,)))


# %%
# %% [Cell 2] Build dataset + dataloader

from torch.utils.data import DataLoader, Subset
import numpy as np

M = 256
Q = 192

ds_all = PrefixNextVoxelDataset(
    chain_ids=chain_ids,
    seq_pairs=seq_pairs,
    vox_cm=VOX_CM,
    pts_per_vox=PTS_PER_VOX,
    M=M,
    Q=Q,
    seed=1234,
)

idx = np.arange(len(ds_all))
np.random.shuffle(idx)
split = int(len(idx) * 0.9)
tr_idx, va_idx = idx[:split], idx[split:]

dl_train = DataLoader(Subset(ds_all, tr_idx), batch_size=2, shuffle=True, num_workers=0, drop_last=True)
dl_val   = DataLoader(Subset(ds_all, va_idx), batch_size=2, shuffle=False, num_workers=0, drop_last=False)

print("train batches:", len(dl_train), "val batches:", len(dl_val))


# %%
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------- Model ----------------
class PointNetBlockEncoder(nn.Module):
    def __init__(self, in_dim=3, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + 6, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )

    def forward(self, x):
        h = self.mlp(x)
        feat = h.max(dim=2).values
        mu = x.mean(dim=2)
        sd = x.std(dim=2)
        feat = self.fuse(torch.cat([feat, mu, sd], dim=-1))
        return feat

class VoxelPosEmbed(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
    def forward(self, centers_cm, vox_cm):
        x = centers_cm / float(vox_cm)
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + h)
        h = self.ff(x)
        x = self.ln2(x + h)
        return x

class FoldingDecoder2D(nn.Module):
    def __init__(self, feat_dim=128, n_out=256):
        super().__init__()
        self.n_out = n_out
        side = int(round(n_out ** 0.5))
        assert side * side == n_out, "PTS_PER_VOX 建议用平方数，如 256"
        u = torch.linspace(-1.0, 1.0, side)
        v = torch.linspace(-1.0, 1.0, side)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        grid = torch.stack([uu, vv], dim=-1).view(-1, 2)
        self.register_buffer("grid", grid)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, feat):
        BM, Fdim = feat.shape
        grid = self.grid.unsqueeze(0).expand(BM, -1, -1)
        feats = feat.unsqueeze(1).expand(-1, self.n_out, -1)
        x = torch.cat([feats, grid], dim=-1).reshape(BM*self.n_out, -1)
        out = self.mlp(x).view(BM, self.n_out, 3)
        return out

class VoxelTransformerFoldingNet(nn.Module):
    def __init__(self, feat_dim=128, n_vox=128, n_pts=256, depth=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_vox = n_vox
        self.n_pts = n_pts

        self.enc = PointNetBlockEncoder(in_dim=3, feat_dim=feat_dim)
        self.pos = VoxelPosEmbed(feat_dim=feat_dim)

        self.type_emb = nn.Embedding(2, feat_dim)  # 0=ctx(has pts), 1=empty/query
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model=feat_dim) for _ in range(depth)])

        # stage-1 selector (decide position)
        self.sel_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # stage-2 occ + shape (after warp)
        self.occ_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.dec = FoldingDecoder2D(feat_dim=feat_dim, n_out=n_pts)

    def forward(self, mem_local, centers_cm, qmsk=None, occ_mask=None, vox_cm=20.0):
        B, M, N, _ = mem_local.shape

        # ✅ 用 qmsk 决定 query / ctx（不要靠 mem 是否全0）
        if qmsk is None:
            is_query = (mem_local.abs().sum(dim=(2,3)) < 1e-6)  # fallback
        else:
            is_query = (qmsk > 0.5)

        type_ids = is_query.long()  # 0=ctx, 1=query

        enc_feat = self.enc(mem_local)

        # 只对 query token 用 mask_token
        enc_feat = torch.where(
            is_query.unsqueeze(-1),
            self.mask_token.expand(B, M, self.feat_dim),
            enc_feat
        )

        feat = enc_feat + self.pos(centers_cm, vox_cm) + self.type_emb(type_ids)

        if occ_mask is not None:
            all_pad = occ_mask.all(dim=1)
            if all_pad.any():
                occ_mask = occ_mask.clone()
                occ_mask[all_pad, :] = False

        x = feat
        for blk in self.blocks:
            x = blk(x, key_padding_mask=occ_mask)

        # stage-1 selector
        sel_logits = self.sel_head(x).squeeze(-1)
        gate = torch.sigmoid(sel_logits).unsqueeze(-1)
        gate = 0.2 + 0.8 * gate
        x_warp = x * gate

        # stage-2
        pred_occ = self.occ_head(x).squeeze(-1)  # ✅ changed
        pred_local = self.dec(x_warp.reshape(B*M, self.feat_dim)).reshape(B, M, self.n_pts, 3)


        return pred_local, pred_occ, sel_logits



net = VoxelTransformerFoldingNet(
    feat_dim=128,
    n_vox=VOX_PER_SAMPLE, n_pts=PTS_PER_VOX, depth=2
).to(device)
print("model params(M):", sum(p.numel() for p in net.parameters())/1e6)

@torch.no_grad()
def diag_occ(pred_occ_logits, occ, shape_mask):
    pred_occ_logits = pred_occ_logits.float()
    occ = occ.float()
    shape_mask = shape_mask.float()

    q = (shape_mask > 0.5)
    if q.sum() == 0:
        print("diag_occ: no query tokens")
        return

    logits_q = pred_occ_logits[q].float()
    occ_q = occ[q].float()

    p_q = torch.sigmoid(logits_q)
    p_pos = p_q[occ_q > 0.5]
    p_neg = p_q[occ_q < 0.5]

    n_q = int(q.sum())
    n_pos = int((occ_q > 0.5).sum())
    n_neg = int((occ_q < 0.5).sum())
    pos_ratio = n_pos / max(1, n_q)

    msg = f"Q={n_q} pos={n_pos} neg={n_neg} pos_ratio={pos_ratio:.3f}"
    if p_pos.numel() > 0:
        msg += f" | p_pos mean={p_pos.mean().item():.3f} p95={torch.quantile(p_pos,0.95).item():.3f} max={p_pos.max().item():.3f}"
    if p_neg.numel() > 0:
        msg += f" | p_neg mean={p_neg.mean().item():.3f} p95={torch.quantile(p_neg,0.95).item():.3f} max={p_neg.max().item():.3f}"
    if p_pos.numel() > 5 and p_neg.numel() > 5:
        thr = 0.5 * (torch.quantile(p_pos, 0.50) + torch.quantile(p_neg, 0.95))
        msg += f" | suggest occ_thresh≈{thr.item():.3f}"
    print(msg)

# %%
# %%


# %%
# %% [Cell] Training (v6.4) — direct q95 tail + mild max/mean-topk, earlier tail ramp, lower pos_weight cap, optional cap anneal

import math
import torch
import torch.nn.functional as F
from torch import optim

steps = 4000
print("🔥 start training steps:", steps)

base_lr = 1e-4
min_lr  = 5e-6
opt = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.05)
scaler = torch.amp.GradScaler('cuda')

warmup = 200
def lr_at(step):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * t))
    return max(min_lr, lr)

device = next(net.parameters()).device

def logit(p: float):
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))

# ---------------- weights / schedules ----------------
beta_sel_base = 1.0
beta_sel_late = 0.55

beta_occ_lo = 1.1
beta_occ_hi = 1.8
beta_occ_switch = 1600

lambda_rank_max = 0.28
rank_ramp_start = 300
rank_ramp_len   = 900
rank_sample_k   = 96
rank_margin     = 0.06

# v6.4: stronger + earlier tail
lambda_tail_max = 0.42
tail_ramp_start = 100
tail_ramp_len   = 800

# caps (with optional anneal)
cap_hard_start = 0.62
cap_hard_end   = 0.58
cap_soon_start = 0.57
cap_soon_end   = 0.53
cap_anneal_start = 1400
cap_anneal_len   = 1200

# suppress (soon-only neg)
occ_sel_suppress_k = 0.6
sel_gate_logit = 0.85
sup_start = 1700
sup_ramp  = 800

soonneg_w_lo = 0.50
soonneg_w_hi = 1.00
soonneg_w_switch = 2000

mix_start = 1000
mix_end   = 2000

def ramp(step, start, length, maxv=1.0):
    return max(0.0, min(maxv, (step - start) / float(max(1, length)) * maxv))

def lerp(a, b, t):
    return a + (b - a) * t

def cap_schedule(step, p_start, p_end, start, length):
    t = ramp(step, start, length, 1.0)
    return lerp(p_start, p_end, t)

# v6.4 tail settings (q95-based)
tail_frac = 0.05
tail_mink = 32
tail_maxk = 512

# weights inside tail term
tail_w_q   = 1.0   # main: push p95 down
tail_w_max = 0.15  # prevent rare blow-ups
tail_w_mean= 0.25  # keep top tail smooth

def tail_q95_penalty(logits_1d, cap_logit, frac=0.05, mink=32, maxk=512,scale=4.0):
    n = logits_1d.numel()
    if n == 0:
        return torch.zeros((), device=logits_1d.device)

    k = int(max(mink, min(maxk, math.ceil(frac * n))))
    k = min(k, n)

    top = logits_1d.topk(k, largest=True).values  # [k], sorted desc
    z_max = top[0]
    q95   = top[-1]  # smallest among top-k ≈ quantile(0.95)

    over_q   = ((q95   - cap_logit) * scale).clamp_min(0.0).clamp_max(6.0)
    over_max = ((z_max - cap_logit) * scale).clamp_min(0.0).clamp_max(6.0)
    over_all = ((top   - cap_logit) * scale).clamp_min(0.0).clamp_max(6.0)

    return (
        tail_w_q   * (over_q ** 2)
        + tail_w_max * (over_max ** 2)
        + tail_w_mean * (over_all ** 2).mean()
    )

net.train()
global_step = 0

for epoch in range(999999):
    for mem_local, tgt_local, centers_cm, occ, qmsk, new_mask, soon_mask in dl_train:
        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        qmsk       = qmsk.to(device).float()
        new_mask   = new_mask.to(device).float()
        soon_mask  = soon_mask.to(device).float()

        pad_mask = torch.zeros_like(new_mask, dtype=torch.bool, device=device)

        for g in opt.param_groups:
            g["lr"] = lr_at(global_step)

        # -------- mix --------
        if global_step < mix_start:
            mix = 0.0
        elif global_step < mix_end:
            mix = (global_step - mix_start) / float(mix_end - mix_start)
        else:
            mix = 1.0

        sel_target = (1.0 - mix) * new_mask + mix * soon_mask

        # -------- scheduled weights --------
        beta_occ = beta_occ_lo if global_step < beta_occ_switch else beta_occ_hi
        lambda_rank = ramp(global_step, rank_ramp_start, rank_ramp_len, lambda_rank_max)
        lambda_tail = ramp(global_step, tail_ramp_start, tail_ramp_len, lambda_tail_max)
        sup_w = ramp(global_step, sup_start, sup_ramp, 1.0)
        soonneg_w = soonneg_w_lo if global_step < soonneg_w_switch else soonneg_w_hi

        beta_sel = beta_sel_base * (1.0 - mix) + beta_sel_late * mix

        # caps (anneal)
        cap_hard_p = cap_schedule(global_step, cap_hard_start, cap_hard_end, cap_anneal_start, cap_anneal_len)
        cap_soon_p = cap_schedule(global_step, cap_soon_start, cap_soon_end, cap_anneal_start, cap_anneal_len)
        cap_hard_logit = torch.tensor(logit(cap_hard_p), device=device, dtype=torch.float32)
        cap_soon_logit = torch.tensor(logit(cap_soon_p), device=device, dtype=torch.float32)

        with torch.amp.autocast('cuda'):
            _, pred_occ_logits, sel_logits = net(
                mem_local, centers_cm, qmsk=qmsk, occ_mask=pad_mask, vox_cm=VOX_CM
            )

            is_query = (qmsk > 0.5)

            # ---------------- selector loss ----------------
            is_pos_sel = is_query & (sel_target > 0.5)
            is_neg_sel = is_query & (sel_target < 0.5)

            sel_raw = F.binary_cross_entropy_with_logits(sel_logits, sel_target, reduction="none")
            pos_sel_n = is_pos_sel.sum().float().clamp_min(1.0)
            neg_sel_n = is_neg_sel.sum().float().clamp_min(1.0)
            w_pos_sel = (neg_sel_n / pos_sel_n).clamp(1.0, 8.0)

            w_sel = torch.zeros_like(sel_raw)
            w_sel[is_pos_sel] = w_pos_sel
            w_sel[is_neg_sel] = 1.0
            loss_sel = (sel_raw * w_sel).sum() / (w_sel.sum().clamp_min(1.0))

            # ---------------- occ loss ----------------
            q = is_query
            y = new_mask

            pos = q & (y > 0.5)
            neg = q & (y < 0.5)

            hard_neg = q & (y < 0.5) & (soon_mask < 0.5)
            soon_neg = q & (y < 0.5) & (soon_mask > 0.5)

            pos_n = pos.sum().float().clamp_min(1.0)
            neg_n = neg.sum().float().clamp_min(1.0)

            # v6.4: reduce pos weight cap to avoid early blow-ups
            w_pos = (neg_n / pos_n).clamp(1.0, 16.0)

            occ_raw = F.binary_cross_entropy_with_logits(pred_occ_logits, y, reduction="none")

            w_occ = torch.zeros_like(occ_raw)
            w_occ[pos]      = w_pos
            w_occ[hard_neg] = 1.0
            w_occ[soon_neg] = soonneg_w
            any_neg = neg & ~(hard_neg | soon_neg)
            w_occ[any_neg] = 1.0

            loss_occ = (occ_raw * w_occ).sum() / (w_occ.sum().clamp_min(1.0))

            # ---------------- rank ----------------
            pos_logits = pred_occ_logits[pos]
            if mix >= 0.7:
                rank_neg_mask = (hard_neg | soon_neg)
            else:
                rank_neg_mask = hard_neg if hard_neg.any() else neg
            neg_logits = pred_occ_logits[rank_neg_mask]

            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                kp = min(rank_sample_k, pos_logits.numel())
                kn = min(rank_sample_k, neg_logits.numel())
                pos_s = pos_logits.topk(kp, largest=False).values
                neg_s = neg_logits.topk(kn, largest=True).values
                loss_rank = F.softplus(rank_margin + neg_s.unsqueeze(0) - pos_s.unsqueeze(1)).mean()
            else:
                loss_rank = torch.zeros((), device=device)

            # ---------------- occ_eff ----------------
            sel_gate = F.relu(sel_logits.detach() - sel_gate_logit)
            occ_eff_logits = pred_occ_logits.clone()
            if soon_neg.any():
                occ_eff_logits[soon_neg] = occ_eff_logits[soon_neg] - sup_w * occ_sel_suppress_k * sel_gate[soon_neg]

            # ---------------- tail (q95-driven) ----------------
            tail = torch.zeros((), device=device)
            if hard_neg.any():
                tail = tail + tail_q95_penalty(
                    occ_eff_logits[hard_neg], cap_hard_logit,
                    frac=tail_frac, mink=tail_mink, maxk=tail_maxk,scale= 4.0
                )
            if soon_neg.any():
                tail = tail + tail_q95_penalty(
                    occ_eff_logits[soon_neg], cap_soon_logit,
                    frac=tail_frac, mink=tail_mink, maxk=tail_maxk,scale= 3.0
                )

            loss = (
                beta_sel * loss_sel
                + beta_occ * loss_occ
                + lambda_rank * loss_rank
                + lambda_tail * tail
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if global_step % 50 == 0:
            with torch.no_grad():
                def qtl(x, p):
                    return torch.quantile(x.float(), p).item() if x.numel() else float("nan")

                p_occ     = torch.sigmoid(pred_occ_logits)
                p_occ_eff = torch.sigmoid(occ_eff_logits)
                p_sel     = torch.sigmoid(sel_logits)

                pos_ct = int(pos.sum().item())
                neg_ct = int(neg.sum().item())
                p_pos = p_occ[pos].mean().item() if pos_ct else float("nan")
                p_neg = p_occ[neg].mean().item() if neg_ct else float("nan")

                soon_pos = q & (soon_mask > 0.5)
                soon_neg_all = q & (soon_mask < 0.5)
                sp = int(soon_pos.sum().item())
                sn = int(soon_neg_all.sum().item())
                sel_sp = p_sel[soon_pos].mean().item() if sp else float("nan")
                sel_sn = p_sel[soon_neg_all].mean().item() if sn else float("nan")

                neg_all = q & (new_mask < 0.5)
                print(
                    f"step {global_step:04d} | loss {loss.item():.4f} "
                    f"[sel {loss_sel.item():.4f}, occ {loss_occ.item():.4f}, tail {tail.item():.4f}, rank {loss_rank.item():.4f}] | "
                    f"p_query(occ) {p_occ[q].mean().item():.3f}  p_query(occ_eff) {p_occ_eff[q].mean().item():.3f}  p_query(sel) {p_sel[q].mean().item():.3f} | "
                    f"p_pos(new) {p_pos:.3f}  p_neg(new) {p_neg:.3f} | "
                    f"sel_pos(soon) {sel_sp:.3f}  sel_neg(soon) {sel_sn:.3f} | "
                    f"p95_eff(all) {qtl(p_occ_eff[neg_all],0.95):.3f}  hard {qtl(p_occ_eff[hard_neg],0.95):.3f}  soon {qtl(p_occ_eff[soon_neg],0.95):.3f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | mix {mix:.2f} | sup_w {sup_w:.2f} | w_soonneg {soonneg_w:.2f} | "
                    f"β_sel {beta_sel:.2f} λ_rank {lambda_rank:.2f} λ_tail {lambda_tail:.2f} β_occ {beta_occ:.2f} | "
                    f"cap_hard {cap_hard_p:.2f} cap_soon {cap_soon_p:.2f}"
                )

        global_step += 1
        if global_step >= steps:
            break
    if global_step >= steps:
        break

print("🏁 Done.")


# %% [markdown]
# Net Save

# %%
# 1. 定义保存路径
save_path = "my_trained_model_final.pth"

# 2. 保存模型参数 (State Dict) - 这是最标准的做法
torch.save(net.state_dict(), save_path)

print(f"✅ 模型已保存到: {os.path.abspath(save_path)}")
print("现在你可以放心重启电脑了。")

# %%
# 1. 重新实例化模型结构 (必须与训练时参数一致)
# 假设你之前的参数是这些 (参考你的 N6 代码)
net = VoxelTransformerFoldingNet(
    feat_dim=128, 
    n_vox=VOX_PER_SAMPLE, 
    n_pts=PTS_PER_VOX, 
    depth=2
).to(device)

# 2. 加载权重
save_path = "my_trained_model_final.pth"
if os.path.exists(save_path):
    net.load_state_dict(torch.load(save_path, map_location=device))
    print("✅ 成功加载训练好的模型权重！")
else:
    print("❌ 找不到模型文件！")

# 3. 切换到评估模式 (如果你要跑 Rollout)
net.eval() 

# 此时你的 net 就恢复到训练完成时的状态了，可以拿去跑 Rollout

# %% [markdown]
# Roll Out
# 

# %%
# =========================
# Rollout / Verify Export Cell (paste-to-run)
# - Blue: input/prefix points
# - Red : net-added points (final - init)
# - Uses SAME pack/unpack + center_from_key as your training code style
# =========================

import os
import numpy as np

def _try_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

torch = _try_import_torch()

# ---- packing (match Untitled-1_N9.py) ----
B = (1 << 20)  # bias for signed -> unsigned

def unpack_key(k: int):
    k = int(k)
    x = ((k >> 42) & ((1 << 21) - 1)) - B
    y = ((k >> 21) & ((1 << 21) - 1)) - B
    z = ((k >>  0) & ((1 << 21) - 1)) - B
    return int(x), int(y), int(z)

def center_from_key(k: int, vox_cm: float, origin_cm=(0.0, 0.0, 0.0)):
    x, y, z = unpack_key(k)
    ox, oy, oz = origin_cm
    # voxel center
    return np.array([(x + 0.5) * vox_cm + ox,
                     (y + 0.5) * vox_cm + oy,
                     (z + 0.5) * vox_cm + oz], dtype=np.float32)

# ---- robust extraction of local points from mem_dict entry ----
def _to_numpy(a):
    if a is None:
        return None
    if isinstance(a, np.ndarray):
        return a
    if torch is not None and hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    # list/tuple
    try:
        return np.asarray(a)
    except Exception:
        return None

def extract_local_pts(mem_val):
    """
    Accepts:
      - np.ndarray / torch.Tensor shaped (N,3)
      - dict with keys: 'local_pts', 'local', 'pts', 'points'
      - tuple/list where first element is (N,3)
    Returns: np.ndarray (N,3) float32 or None
    """
    if mem_val is None:
        return None

    # direct array/tensor
    arr = _to_numpy(mem_val)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32)

    # dict wrapper
    if isinstance(mem_val, dict):
        for kk in ["local_pts", "local", "pts", "points", "xyz"]:
            if kk in mem_val:
                arr = _to_numpy(mem_val[kk])
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
                    return arr.astype(np.float32)

    # tuple/list wrapper
    if isinstance(mem_val, (list, tuple)) and len(mem_val) > 0:
        arr = _to_numpy(mem_val[0])
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.float32)

    return None

def local_to_world(local_pts, vox_center_cm, vox_cm):
    """
    Detect whether local_pts already in cm, or normalized.
    Heuristics:
      - if max_abs <= 1.5: treat as normalized in [-0.5,0.5] or [-1,1] => scale by vox_cm
      - if range mostly [0,1]: treat as [0,1] => (p-0.5)*vox_cm
      - else: treat as already cm offsets
    """
    if local_pts is None or len(local_pts) == 0:
        return None

    lp = local_pts.astype(np.float32)
    mx = float(np.max(lp))
    mn = float(np.min(lp))
    max_abs = float(np.max(np.abs(lp)))

    # case A: looks like [0,1]
    if mn >= -1e-3 and mx <= 1.0 + 1e-3:
        lp_cm = (lp - 0.5) * float(vox_cm)
        return vox_center_cm[None, :] + lp_cm

    # case B: looks like [-0.5,0.5] or [-1,1] normalized
    if max_abs <= 1.5:
        lp_cm = lp * float(vox_cm)
        return vox_center_cm[None, :] + lp_cm

    # case C: already cm offsets
    return vox_center_cm[None, :] + lp

def write_ply_xyzrgb(path, xyz, rgb=None):
    """
    xyz: (N,3) float
    rgb: (N,3) uint8 or None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    if rgb is None:
        header = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % (xyz.shape[0])
        with open(path, "w") as f:
            f.write(header)
            for p in xyz:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
    else:
        rgb = np.asarray(rgb, dtype=np.uint8)
        header = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % (xyz.shape[0])
        with open(path, "w") as f:
            f.write(header)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def export_input_vs_net_added(init_occ_set, init_mem_dict, final_occ_set, final_mem_dict,
                             vox_cm, origin_cm=(0,0,0),
                             out_dir="./rollout_vis", sample_limit_per_vox=64):
    """
    Output:
      - input_prefix_only.ply (blue)
      - net_added_only.ply (red)
      - input_vs_added_colored.ply (blue+red)
    """
    os.makedirs(out_dir, exist_ok=True)

    init_occ_set  = set(init_occ_set)
    final_occ_set = set(final_occ_set)

    added_occ = [k for k in final_occ_set if k not in init_occ_set]
    print("== basic counts ==")
    print(" init_occ:", len(init_occ_set), " final_occ:", len(final_occ_set), " added_occ:", len(added_occ))

    # debug: voxel coordinate bbox
    init_xyz = np.array([unpack_key(k) for k in list(init_occ_set)[:min(2000, len(init_occ_set))]], dtype=np.int32)
    fin_xyz  = np.array([unpack_key(k) for k in list(final_occ_set)[:min(2000, len(final_occ_set))]], dtype=np.int32)
    print(" sample init unpack bbox(vox):", init_xyz.min(0), init_xyz.max(0))
    print(" sample fin  unpack bbox(vox):", fin_xyz.min(0),  fin_xyz.max(0))

    def gather_points(occ_keys, mem_dict, tag):
        pts_all = []
        local_stats = []
        missing = 0

        for k in occ_keys:
            mv = mem_dict.get(k, None)
            lp = extract_local_pts(mv)
            if lp is None or len(lp) == 0:
                missing += 1
                continue
            if sample_limit_per_vox is not None and lp.shape[0] > sample_limit_per_vox:
                lp = lp[:sample_limit_per_vox]

            cen = center_from_key(k, vox_cm=float(vox_cm), origin_cm=origin_cm)
            wp = local_to_world(lp, cen, vox_cm=float(vox_cm))
            if wp is None or len(wp) == 0:
                missing += 1
                continue

            pts_all.append(wp)
            local_stats.append([float(lp.min()), float(lp.max()), float(np.max(np.abs(lp)))])

        if len(pts_all) == 0:
            print(f"[{tag}] gathered EMPTY points. missing_vox={missing}/{len(occ_keys)}")
            return np.zeros((0,3), dtype=np.float32)

        pts = np.concatenate(pts_all, axis=0)
        ls = np.array(local_stats, dtype=np.float32) if len(local_stats) else None
        if ls is not None:
            print(f"[{tag}] local_pts stats per-voxel (min,max,max_abs):")
            print("   min(min):", ls[:,0].min(), " max(max):", ls[:,1].max(), " mean(max_abs):", ls[:,2].mean())
        print(f"[{tag}] points:", pts.shape[0], " missing_vox:", missing, "/", len(occ_keys))
        amin, amax = pts.min(0), pts.max(0)
        print(f"[{tag}] bbox(cm):", amin, amax, " extent:", (amax-amin))
        return pts

    # gather points
    init_pts  = gather_points(list(init_occ_set),  init_mem_dict,  "INIT/PREFIX")
    added_pts = gather_points(added_occ,           final_mem_dict, "ADDED/NET")

    # write separate
    blue = np.array([0, 120, 255], dtype=np.uint8)
    red  = np.array([255, 60, 60], dtype=np.uint8)

    write_ply_xyzrgb(os.path.join(out_dir, "input_prefix_only.ply"),
                     init_pts, np.repeat(blue[None,:], init_pts.shape[0], axis=0) if init_pts.shape[0] else None)

    write_ply_xyzrgb(os.path.join(out_dir, "net_added_only.ply"),
                     added_pts, np.repeat(red[None,:], added_pts.shape[0], axis=0) if added_pts.shape[0] else None)

    # write combined colored
    comb_xyz = np.concatenate([init_pts, added_pts], axis=0) if (len(init_pts) + len(added_pts)) else np.zeros((0,3), np.float32)
    comb_rgb = None
    if comb_xyz.shape[0] > 0:
        comb_rgb = np.concatenate([
            np.repeat(blue[None,:], init_pts.shape[0], axis=0),
            np.repeat(red[None,:],  added_pts.shape[0], axis=0)
        ], axis=0)

    write_ply_xyzrgb(os.path.join(out_dir, "input_vs_added_colored.ply"), comb_xyz, comb_rgb)

    print("\nSaved PLY to:", os.path.abspath(out_dir))
    print(" - input_prefix_only.ply (blue)")
    print(" - net_added_only.ply    (red)")
    print(" - input_vs_added_colored.ply (blue+red)")
    return {
        "out_dir": out_dir,
        "n_init_pts": int(init_pts.shape[0]),
        "n_added_pts": int(added_pts.shape[0]),
        "n_added_vox": int(len(added_occ)),
    }

# ------------------------------
# Auto-detect your variables (no crash)
# ------------------------------
g = globals()

# candidates
init_occ = g.get("init_occ_set", g.get("init_occ", None))
init_mem = g.get("init_mem_dict", g.get("init_mem", None))
final_occ = g.get("final_occ_set", g.get("occ_set", g.get("final_occ", None)))
final_mem = g.get("final_mem_dict", g.get("mem_dict", g.get("final_mem", None)))

vox_cm = g.get("VOX_CM", g.get("vox_cm", None))
origin_cm = g.get("origin_cm", (0.0,0.0,0.0))

print("Auto-detected variables:")
print("  init_occ:", "OK" if init_occ is not None else "MISSING")
print("  init_mem:", "OK" if init_mem is not None else "MISSING")
print(" final_occ:", "OK" if final_occ is not None else "MISSING")
print(" final_mem:", "OK" if final_mem is not None else "MISSING")
print("   vox_cm :", vox_cm)
print(" origin_cm:", origin_cm)

if (init_occ is None) or (init_mem is None) or (final_occ is None) or (final_mem is None) or (vox_cm is None):
    print("\nMissing required variables -> I will NOT crash.")
    print("Please make sure you have: init_occ_set, init_mem_dict, occ_set(or final_occ_set), mem_dict(or final_mem_dict), VOX_CM(or vox_cm).")
else:
    # quick key sanity
    ex = next(iter(init_occ))
    print("\n== key inspection ==")
    print(" init_occ example:", ex, " type:", type(ex))
    print(" unpack_key(example):", unpack_key(ex))

    vis = export_input_vs_net_added(
        init_occ_set=set(init_occ), init_mem_dict=init_mem,
        final_occ_set=set(final_occ), final_mem_dict=final_mem,
        vox_cm=float(vox_cm), origin_cm=tuple(origin_cm),
        out_dir="./rollout_vis", sample_limit_per_vox=64
    )
    print("\nvis:", vis)


# %% [markdown]
# Continue

# %%
# %% [Cell: Sanity Check (VAL) — occ metrics + distributions + optional shape chamfer]
import torch
import torch.nn.functional as F

def chamfer_distance(a, b):
    d = torch.cdist(a, b)
    d1 = d.min(dim=2).values.mean(dim=1)
    d2 = d.min(dim=1).values.mean(dim=1)
    return (d1 + d2).mean()

@torch.no_grad()
def prf(prob, y, thr=0.5):
    pred = (prob >= thr).float()
    tp = ((pred==1) & (y==1)).sum().item()
    fp = ((pred==1) & (y==0)).sum().item()
    fn = ((pred==0) & (y==1)).sum().item()
    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    return prec, rec, f1, tp, fp, fn

# --- OPTIONAL: peek what dl_val actually returns (run once if needed) ---
@torch.no_grad()
def peek_val_batch(dl_val):
    batch = next(iter(dl_val))
    print("batch type:", type(batch))
    if isinstance(batch, dict):
        print("dict keys:", list(batch.keys()))
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print("batch len:", len(batch))
        for i, v in enumerate(batch):
            if torch.is_tensor(v):
                print(f"  [{i}]: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  [{i}]: {type(v)}")

@torch.no_grad()
def val_sanity(net, dl_val, thr=0.5, max_batches=10):
    net.eval()
    all_sel=[]
    all_occ=[]
    all_y=[]
    nb=0

    for batch in dl_val:
        # --- make it robust to "too many values" ---
        if isinstance(batch, dict):
            # adjust keys here if your dataset uses different names
            mem_local  = batch["mem_local"]
            centers_cm = batch["centers_cm"]
            shape_mask = batch["shape_mask"]
            new_mask   = batch["new_mask"]
            # optional / unused:
            # tgt_local = batch.get("tgt_local", None)
            # occ      = batch.get("occ", None)
        else:
            # ignore any extra fields safely
            mem_local, tgt_local, centers_cm, occ, shape_mask, new_mask, *extra = batch

        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        shape_mask = shape_mask.to(device).float()
        new_mask   = new_mask.to(device).float()

        # occ_mask in your forward is used as "mask", keep it bool and same shape as queries
        pad_mask = torch.zeros_like(new_mask, dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, pred_occ_logits, sel_logits = net(
                mem_local, centers_cm,
                qmsk=shape_mask, occ_mask=pad_mask, vox_cm=VOX_CM
            )

        is_query = (shape_mask > 0.5)

        # flatten by query-mask
        y     = new_mask[is_query]
        p_occ = torch.sigmoid(pred_occ_logits)[is_query]
        p_sel = torch.sigmoid(sel_logits)[is_query]

        all_y.append(y.detach().cpu())
        all_occ.append(p_occ.detach().cpu())
        all_sel.append(p_sel.detach().cpu())

        nb += 1
        if nb >= max_batches:
            break

    y = torch.cat(all_y, dim=0)
    p_occ = torch.cat(all_occ, dim=0)
    p_sel = torch.cat(all_sel, dim=0)

    prec, rec, f1, tp, fp, fn = prf(p_sel, y, thr)
    print(f"[VAL selector] thr={thr:.2f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} tp={tp} fp={fp} fn={fn}")

    prec, rec, f1, tp, fp, fn = prf(p_occ, y, thr)
    print(f"[VAL occ     ] thr={thr:.2f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} tp={tp} fp={fp} fn={fn}")

    net.train()

# If you're unsure about the batch format, run this once:
# peek_val_batch(dl_val)

val_sanity(net, dl_val, thr=0.5, max_batches=10)


# %%
# %% [Cell: VAL Top-K per-sample (correct) + micro/macro PRF]
import torch, numpy as np

@torch.no_grad()
def topk_per_sample_metrics(net, dl_val, K_list=(1,2,4,8,16,32,64), max_batches=50):
    net.eval()

    # accumulate micro counts for each K
    micro = {K: {"tp":0,"fp":0,"fn":0,"n_pos":0,"n_pred":0} for K in K_list}
    macro = {K: [] for K in K_list}  # list of (P,R,F1) per sample

    pos_counts = []

    for b, batch in enumerate(dl_val):
        if isinstance(batch, dict):
            mem_local  = batch["mem_local"]
            centers_cm = batch["centers_cm"]
            shape_mask = batch["shape_mask"]
            new_mask   = batch["new_mask"]
        else:
            mem_local  = batch[0]
            centers_cm = batch[2]
            shape_mask = batch[4]
            new_mask   = batch[5]

        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        shape_mask = shape_mask.to(device).float()
        new_mask   = new_mask.to(device).float()

        pad_mask = torch.zeros_like(new_mask, dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            _, occ_logits, sel_logits = net(
                mem_local, centers_cm,
                qmsk=shape_mask, occ_mask=pad_mask, vox_cm=VOX_CM
            )

        # pick which score to evaluate for Top-K
        score = torch.sigmoid(occ_logits)  # or sel_logits if you want

        # assume shapes like [B,Q] for masks and logits
        B = shape_mask.shape[0]
        for i in range(B):
            is_q = (shape_mask[i] > 0.5)
            if is_q.sum().item() == 0:
                continue

            y = new_mask[i][is_q].detach().cpu().numpy().astype(np.int32)
            s = score[i][is_q].detach().cpu().numpy()

            n_pos = int((y==1).sum())
            pos_counts.append(n_pos)

            order = np.argsort(-s)

            for K in K_list:
                kk = min(K, len(order))
                pred = np.zeros_like(y)
                pred[order[:kk]] = 1

                tp = int(((pred==1) & (y==1)).sum())
                fp = int(((pred==1) & (y==0)).sum())
                fn = int(((pred==0) & (y==1)).sum())

                P = tp/(tp+fp+1e-9)
                R = tp/(tp+fn+1e-9)
                F = 2*P*R/(P+R+1e-9)

                micro[K]["tp"] += tp
                micro[K]["fp"] += fp
                micro[K]["fn"] += fn
                micro[K]["n_pos"] += n_pos
                micro[K]["n_pred"] += kk

                macro[K].append((P,R,F))

        if b+1 >= max_batches:
            break

    print("== per-sample positive count stats ==")
    if len(pos_counts):
        pc = np.array(pos_counts)
        print(f"samples={len(pc)}  pos_mean={pc.mean():.2f}  pos_p50={np.percentile(pc,50):.0f}  pos_p90={np.percentile(pc,90):.0f}  max={pc.max()}")
    else:
        print("no samples processed")

    print("\n== Top-K per-sample (MICRO over samples) ==")
    for K in K_list:
        tp,fp,fn = micro[K]["tp"], micro[K]["fp"], micro[K]["fn"]
        P = tp/(tp+fp+1e-9)
        R = tp/(tp+fn+1e-9)
        F = 2*P*R/(P+R+1e-9)
        print(f"K={K:>3}  P={P:.3f} R={R:.3f} F1={F:.3f}  tp={tp} fp={fp} fn={fn}")

    print("\n== Top-K per-sample (MACRO average over samples) ==")
    for K in K_list:
        if len(macro[K]) == 0:
            print(f"K={K:>3}  (no data)")
            continue
        arr = np.array(macro[K])
        P,R,F = arr[:,0].mean(), arr[:,1].mean(), arr[:,2].mean()
        print(f"K={K:>3}  P={P:.3f} R={R:.3f} F1={F:.3f}  n={len(arr)}")

    net.train()

topk_per_sample_metrics(net, dl_val, K_list=(1,2,4,8,16,32,64), max_batches=50)


# %%
batch = next(iter(dl_val))
print("batch type:", type(batch))
print("num items:", len(batch))

for i, x in enumerate(batch):
    if torch.is_tensor(x):
        print(i, "Tensor", tuple(x.shape), x.dtype, x.device)
    else:
        print(i, type(x), x)


for batch in dl_val:
    mem_local, tgt_local, centers_cm, occ, shape_mask, new_mask, *rest = batch
    # rest 里就是多出来的东西
    break
is_query = (shape_mask > 0.5)
print("shape_mask.shape =", tuple(shape_mask.shape))
print("is_query.shape   =", tuple(is_query.shape))
print("len(is_query)    =", len(is_query), " (this is batch size)")

# 下面才是你要的“每样本 query 个数”
if is_query.ndim == 2:
    q_per_sample = is_query.sum(dim=1)
elif is_query.ndim == 3:
    # 常见情况： [B, 1, Q] 或 [B, Q, 1]
    q_per_sample = is_query.sum(dim=tuple(range(1, is_query.ndim)))
else:
    q_per_sample = is_query.sum()

print("q_per_sample =", q_per_sample.detach().cpu().tolist())
print("q_total      =", int(is_query.sum().item()))


# %%
# %% [Cell: audit NaN/INF + per-sample hard negatives dump]
import numpy as np, pandas as pd, torch

@torch.no_grad()
def dump_hardneg_per_sample(net, dl_val, max_batches=50, topn=20, path="hardneg_by_sample.csv"):
    net.eval()
    rows = []

    for b, batch in enumerate(dl_val):
        if isinstance(batch, dict):
            mem_local  = batch["mem_local"]
            centers_cm = batch["centers_cm"]
            shape_mask = batch["shape_mask"]
            new_mask   = batch["new_mask"]
        else:
            mem_local  = batch[0]
            centers_cm = batch[2]
            shape_mask = batch[4]
            new_mask   = batch[5]

        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        shape_mask = shape_mask.to(device).float()
        new_mask   = new_mask.to(device).float()
        pad_mask = torch.zeros_like(new_mask, dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            _, occ_logits, sel_logits = net(mem_local, centers_cm, qmsk=shape_mask, occ_mask=pad_mask, vox_cm=VOX_CM)

        p_occ = torch.sigmoid(occ_logits).detach().cpu().numpy()
        p_sel = torch.sigmoid(sel_logits).detach().cpu().numpy()
        y     = new_mask.detach().cpu().numpy()
        q     = (shape_mask.detach().cpu().numpy() > 0.5)

        # NaN/INF check
        if (not np.isfinite(p_occ).all()) or (not np.isfinite(p_sel).all()):
            print(f"[warn] non-finite scores in batch {b}")

        B = q.shape[0]
        for i in range(B):
            idx = np.where(q[i].reshape(-1) == 1)[0]
            if len(idx) == 0: 
                continue
            yi = y[i].reshape(-1)[idx].astype(np.int32)
            so = p_occ[i].reshape(-1)[idx]
            ss = p_sel[i].reshape(-1)[idx]

            neg = np.where(yi == 0)[0]
            if len(neg) == 0:
                continue

            # pick top negs by occ score (change to ss if needed)
            order = neg[np.argsort(-so[neg])][:topn]

            for r, j in enumerate(order):
                row = {
                    "batch_id": b, "sample_id": i, "rank": r,
                    "p_occ": float(so[j]), "p_sel": float(ss[j]),
                    "y": int(yi[j]),
                }
                rows.append(row)

        if b+1 >= max_batches:
            break

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"saved: {path}  rows={len(df)}")
    net.train()
    return df

df_hn2 = dump_hardneg_per_sample(net, dl_val, max_batches=50, topn=20, path="hardneg_by_sample.csv")
df_hn2.head()


# %%
# %% [Cell: VAL visualize (occ) + threshold sweep + top FP/FN dump]
import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def quick_hist(net, dl_val, which="sel", max_batches=2):
    net.eval()
    ps=[]
    ys=[]
    for i,(mem_local, tgt_local, centers_cm, occ, shape_mask, new_mask) in enumerate(dl_val):
        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        shape_mask = shape_mask.to(device).float()
        new_mask   = new_mask.to(device).float()

        pad_mask = torch.zeros_like(new_mask, dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            _, pred_occ_logits, sel_logits = net(mem_local, centers_cm, occ_mask=pad_mask, vox_cm=VOX_CM)

        is_query = (shape_mask > 0.5)
        y = new_mask[is_query].detach().cpu().numpy()

        if which == "sel":
            p = torch.sigmoid(sel_logits)[is_query].detach().cpu().numpy()
        else:
            p = torch.sigmoid(pred_occ_logits)[is_query].detach().cpu().numpy()

        ps.append(p); ys.append(y)
        if i+1 >= max_batches:
            break

    p = np.concatenate(ps); y = np.concatenate(ys)
    plt.figure()
    plt.hist(p[y<0.5], bins=30, alpha=0.7, label="neg")
    plt.hist(p[y>0.5], bins=30, alpha=0.7, label="pos")
    plt.title(f"{which} prob hist (query tokens)")
    plt.legend()
    plt.show()

    net.train()

quick_hist(net, dl_val, which="sel")
quick_hist(net, dl_val, which="occ")


# %% [markdown]
# Result Visualize
# 

# %%
# %% [Cell: VAL visualize (occ) + threshold sweep + top FP/FN dump]
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def _collect_val_occ(
    net, dl_val, device,
    max_batches=50,
    VOX_CM=None,
):
    net.eval()
    all_p = []
    all_y = []
    all_isq = []
    all_meta = []  # (batch_idx, i_flat) for trace-back

    bcount = 0
    for batch_idx, (mem_local, tgt_local, centers_cm, occ, shape_mask) in enumerate(dl_val):
        mem_local  = mem_local.to(device)
        centers_cm = centers_cm.to(device)
        occ        = occ.to(device).float()
        shape_mask = shape_mask.to(device).float()

        pad_mask = torch.zeros_like(occ, dtype=torch.bool, device=device)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            _, pred_occ_logits = net(mem_local, centers_cm, occ_mask=pad_mask, vox_cm=VOX_CM)

        prob = torch.sigmoid(pred_occ_logits).detach()

        is_query = (shape_mask > 0.5)
        # keep only query positions
        p_q = prob[is_query].float().flatten().cpu()
        y_q = occ[is_query].float().flatten().cpu()
        isq = torch.ones_like(y_q, dtype=torch.bool)

        all_p.append(p_q)
        all_y.append(y_q)
        all_isq.append(isq)

        # meta index mapping: flatten within query subset
        # We store (batch_idx, global_linear_index_in_full_tensor) for later trace-back.
        # Compute global linear indices in full (B*M) space:
        # occ shape is (B,M) usually; if different, still works if same mask shape.
        full_idx = torch.nonzero(is_query.view(-1), as_tuple=False).view(-1).cpu()
        # Map each query element to its global index in is_query.view(-1) space
        # (not the dataset index, but enough to locate within that batch if needed)
        all_meta.append((batch_idx, full_idx.numpy()))

        bcount += 1
        if bcount >= max_batches:
            break

    p = torch.cat(all_p).numpy()
    y = torch.cat(all_y).numpy()
    return p, y

def _pr_curve(p, y, num_thr=400):
    # y in {0,1}
    thr = np.linspace(0.0, 1.0, num_thr)
    P = np.zeros_like(thr)
    R = np.zeros_like(thr)
    F1 = np.zeros_like(thr)
    FPR = np.zeros_like(thr)

    y = y.astype(np.int32)
    pos = (y == 1)
    neg = (y == 0)
    npos = max(pos.sum(), 1)
    nneg = max(neg.sum(), 1)

    for i, t in enumerate(thr):
        pred = (p >= t)
        TP = np.sum(pred & pos)
        FP = np.sum(pred & neg)
        FN = np.sum((~pred) & pos)
        TN = np.sum((~pred) & neg)

        prec = TP / max(TP + FP, 1)
        rec  = TP / npos
        f1   = (2 * prec * rec) / max(prec + rec, 1e-12)
        fpr  = FP / nneg

        P[i], R[i], F1[i], FPR[i] = prec, rec, f1, fpr

    # AP (PR-AUC) via simple trapezoid integration on recall-sorted points
    # Sort by recall ascending to integrate precision(recall)
    order = np.argsort(R)
    ap = np.trapz(P[order], R[order])
    return thr, P, R, F1, FPR, ap

def _plot_hist(p_pos, p_neg, bins=50, title="occ prob hist"):
    plt.figure(figsize=(7,4))
    plt.hist(p_neg, bins=bins, alpha=0.6, density=True, label="neg(query)")
    plt.hist(p_pos, bins=bins, alpha=0.6, density=True, label="pos(query)")
    plt.xlabel("p_occ")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

def _plot_pr(R, P, title="PR curve"):
    plt.figure(figsize=(6,5))
    plt.plot(R, P)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

def _plot_sweep(thr, P, R, F1, FPR, title="threshold sweep"):
    plt.figure(figsize=(8,4))
    plt.plot(thr, P, label="precision")
    plt.plot(thr, R, label="recall")
    plt.plot(thr, F1, label="f1")
    plt.plot(thr, FPR, label="fpr")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

def _top_cases(p, y, thr, k=24):
    # false positives: y=0 but p>=thr
    fp_idx = np.where((y == 0) & (p >= thr))[0]
    fn_idx = np.where((y == 1) & (p <  thr))[0]

    fp_sorted = fp_idx[np.argsort(-p[fp_idx])] if fp_idx.size else np.array([], dtype=np.int64)
    fn_sorted = fn_idx[np.argsort(p[fn_idx])]  if fn_idx.size else np.array([], dtype=np.int64)

    return fp_sorted[:k], fn_sorted[:k]

# ---------------- RUN ----------------
# You likely already have: net, dl_val (or dl_train), device, VOX_CM
# Set which loader to visualize:
dl_vis = dl_val

max_batches = 30   # bump if you want smoother curves
num_thr = 401
thr_pick = 0.75    # pick any threshold to inspect FP/FN

p, y = _collect_val_occ(net, dl_vis, device, max_batches=max_batches, VOX_CM=VOX_CM)

p_pos = p[y == 1]
p_neg = p[y == 0]

print(f"[vis] Q_total={len(p)} | pos_ratio={y.mean():.3f}")
print(f"p_pos: n={len(p_pos)} mean={p_pos.mean():.3f} p95={np.quantile(p_pos,0.95):.3f} max={p_pos.max():.3f}")
print(f"p_neg: n={len(p_neg)} mean={p_neg.mean():.3f} p95={np.quantile(p_neg,0.95):.3f} max={p_neg.max():.3f}")
print(f"neg>0.5 {(p_neg>0.5).mean():.3f} | neg>0.8 {(p_neg>0.8).mean():.3f}")

thr, P, R, F1, FPR, ap = _pr_curve(p, y, num_thr=num_thr)
best = np.argmax(F1)
print(f"PR-AUC (trapz) ~ {ap:.4f}")
print(f"Best F1 @thr={thr[best]:.3f}: P={P[best]:.3f} R={R[best]:.3f} F1={F1[best]:.3f} FPR={FPR[best]:.3f}")

# plots
_plot_hist(p_pos, p_neg, bins=60, title=f"occ prob hist (batches={max_batches})")
_plot_pr(R, P, title=f"PR curve (AP~{ap:.3f})")
_plot_sweep(thr, P, R, F1, FPR, title="threshold sweep")

plt.show()

# top FP / FN indices within this collected sample (for manual inspection)
fp_top, fn_top = _top_cases(p, y, thr=thr_pick, k=30)
print(f"\n--- Top FP (y=0, p>= {thr_pick}) count={len(fp_top)} (show up to 30) ---")
for i in fp_top[:30]:
    print(f"idx={i:6d} | p={p[i]:.4f} | y={int(y[i])}")

print(f"\n--- Top FN (y=1, p< {thr_pick}) count={len(fn_top)} (show up to 30) ---")
for i in fn_top[:30]:
    print(f"idx={i:6d} | p={p[i]:.4f} | y={int(y[i])}")

# Optional: print a few quantiles to see tail behavior
for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"neg p{int(q*100):02d}={np.quantile(p_neg,q):.3f} | pos p{int(q*100):02d}={np.quantile(p_pos,q):.3f}")


# %%
# %% [Cell: Rhino PLY debug (centers prob + GT vs Pred, multi-scale)]
import os
import numpy as np
import torch
from pathlib import Path

export_dir = Path("./ply_debug")
export_dir.mkdir(parents=True, exist_ok=True)

# --- selection ---
use_topk = True
topk = 400          # 用 topk 更稳定；如果你想用阈值就 set use_topk=False
thr = 0.55          # 只在 use_topk=False 时用

# --- units ---
rhino_unit = "mm"   # "cm" | "mm" | "m"
if rhino_unit == "cm":
    unit_mul = 1.0
elif rhino_unit == "mm":
    unit_mul = 10.0
elif rhino_unit == "m":
    unit_mul = 0.01
else:
    raise ValueError("rhino_unit must be one of: cm, mm, m")

scale_modes = ["cm", "vox", "vox_half"]  # 一次出 3 份对比

def scale_cm_from_mode(mode: str):
    if mode == "cm":
        return 1.0
    if mode == "vox":
        return float(VOX_CM)
    if mode == "vox_half":
        return float(VOX_CM) * 0.5
    raise ValueError(mode)

def prob_to_rgb(p):
    p = np.clip(p, 0.0, 1.0)
    r = (255 * p).astype(np.uint8)
    g = (255 * (1.0 - np.abs(p - 0.5) * 2.0)).astype(np.uint8)
    b = (255 * (1.0 - p)).astype(np.uint8)
    return r, g, b

def write_ply_ascii(path, xyz, prob=None):
    xyz = xyz.astype(np.float32)
    K = xyz.shape[0]
    has_color = prob is not None
    if has_color:
        prob = prob.astype(np.float32).reshape(-1)
        r, g, b = prob_to_rgb(prob)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {K}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if not has_color:
            for i in range(K):
                f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]}\n")
        else:
            for i in range(K):
                f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {int(r[i])} {int(g[i])} {int(b[i])}\n")

net.eval()

batch = next(iter(dl_val))
mem_local, tgt_local, centers_cm, occ, shape_mask = batch
mem_local  = mem_local.to(device)
tgt_local  = tgt_local.to(device)
centers_cm = centers_cm.to(device)
occ        = occ.to(device).float()
shape_mask = shape_mask.to(device).float()
pad_mask = torch.zeros_like(occ, dtype=torch.bool, device=device)

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        pred_local, pred_occ_logits = net(mem_local, centers_cm, occ_mask=pad_mask, vox_cm=VOX_CM)
        prob = torch.sigmoid(pred_occ_logits)

is_query = (shape_mask > 0.5)
is_posq  = is_query & (occ > 0.5)
is_negq  = is_query & (occ < 0.5)

B, M, N, _ = pred_local.shape
pred_f = pred_local.detach().float().view(B*M, N, 3)
tgt_f  = tgt_local.detach().float().view(B*M, N, 3)
cen_f  = centers_cm.detach().float().view(B*M, 3)
prob_f = prob.detach().float().view(B*M)
q_f    = is_query.view(B*M)
pos_f  = is_posq.view(B*M)

# ---- quick diag: local scale check ----
sel_for_diag = q_f.nonzero().squeeze(1)[:min(int(q_f.sum().item()), 400)]
pl_min = pred_f[sel_for_diag].amin(dim=(0,1)).cpu().numpy()
pl_max = pred_f[sel_for_diag].amax(dim=(0,1)).cpu().numpy()
tl_min = tgt_f[sel_for_diag].amin(dim=(0,1)).cpu().numpy()
tl_max = tgt_f[sel_for_diag].amax(dim=(0,1)).cpu().numpy()
print(f"[diag] pred_local min={pl_min} max={pl_max}")
print(f"[diag] tgt_local  min={tl_min} max={tl_max}")
print(f"[diag] centers_cm min={cen_f[q_f].amin(0).cpu().numpy()} max={cen_f[q_f].amax(0).cpu().numpy()}")
print(f"[diag] VOX_CM={VOX_CM} | rhino_unit={rhino_unit} mul={unit_mul}")

# ---- A) export centers with prob color (shows occupancy field) ----
centers_xyz = (cen_f[q_f] * unit_mul).cpu().numpy()
centers_prob = prob_f[q_f].cpu().numpy()
outA = export_dir / f"centers_prob_{rhino_unit}.ply"
write_ply_ascii(outA, centers_xyz, centers_prob)
print("[export A]", outA, "points=", centers_xyz.shape[0])

# ---- choose predicted queries (sel) ----
q_idx = torch.nonzero(q_f).squeeze(1)
if use_topk:
    q_probs = prob_f[q_idx]
    k = min(topk, q_idx.numel())
    top_idx = torch.topk(q_probs, k=k, largest=True).indices
    sel_idx = q_idx[top_idx]
else:
    sel_idx = torch.nonzero(q_f & (prob_f >= thr)).squeeze(1)

# cap (avoid crazy large)
sel_idx = sel_idx[:min(sel_idx.numel(), 5000)]
pos_idx = torch.nonzero(pos_f).squeeze(1)[:min(int(pos_f.sum().item()), 5000)]

print(f"[select] use_topk={use_topk} topk={topk} thr={thr} | sel_queries={sel_idx.numel()} | gt_pos_queries={pos_idx.numel()}")

# ---- B/C) export GT pos and Pred sel with multiple scale modes ----
for mode in scale_modes:
    s = scale_cm_from_mode(mode)

    # GT: only true pos queries
    if pos_idx.numel() > 0:
        gt_world = (cen_f[pos_idx].unsqueeze(1) + tgt_f[pos_idx] * s) * unit_mul
        gt_xyz = gt_world.reshape(-1,3).cpu().numpy()
        outB = export_dir / f"gt_pos_{mode}_{rhino_unit}.ply"
        write_ply_ascii(outB, gt_xyz, None)
        print("[export B]", outB, "points=", gt_xyz.shape[0])

    # Pred: selected queries
    if sel_idx.numel() > 0:
        pr_world = (cen_f[sel_idx].unsqueeze(1) + pred_f[sel_idx] * s) * unit_mul
        pr_xyz = pr_world.reshape(-1,3).cpu().numpy()
        pr_prob = prob_f[sel_idx].repeat_interleave(N).cpu().numpy()
        outC = export_dir / f"pred_sel_{mode}_{rhino_unit}.ply"
        write_ply_ascii(outC, pr_xyz, pr_prob)
        print("[export C]", outC, "points=", pr_xyz.shape[0])

print("Done. Export dir:", export_dir.resolve())


# %%
# %% [Cell: Confirm pred != gt + shape error stats + overlay PLY (GT green, Pred red)]
import os
import numpy as np
import torch
from pathlib import Path

dbg_dir = Path("./ply_debug_overlay")
dbg_dir.mkdir(parents=True, exist_ok=True)

rhino_unit = "mm"   # "cm" | "mm" | "m"
if rhino_unit == "cm":
    unit_mul = 1.0
elif rhino_unit == "mm":
    unit_mul = 10.0
elif rhino_unit == "m":
    unit_mul = 0.01
else:
    raise ValueError("rhino_unit must be one of: cm, mm, m")

scale_mode = "vox"   # "cm" | "vox" | "vox_half"
def scale_cm_from_mode(mode: str):
    if mode == "cm":
        return 1.0
    if mode == "vox":
        return float(VOX_CM)
    if mode == "vox_half":
        return float(VOX_CM) * 0.5
    raise ValueError(mode)

S = scale_cm_from_mode(scale_mode)

def write_ply_rgb_ascii(path, xyz, rgb):
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    K = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {K}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(K):
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")

net.eval()

# ---- load one val batch ----
mem_local, tgt_local, centers_cm, occ, shape_mask = next(iter(dl_val))
mem_local  = mem_local.to(device)
tgt_local  = tgt_local.to(device)
centers_cm = centers_cm.to(device)
occ        = occ.to(device).float()
shape_mask = shape_mask.to(device).float()
pad_mask   = torch.zeros_like(occ, dtype=torch.bool, device=device)

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        pred_local, pred_occ_logits = net(mem_local, centers_cm, occ_mask=pad_mask, vox_cm=VOX_CM)
        prob = torch.sigmoid(pred_occ_logits)

is_query = (shape_mask > 0.5)
is_posq  = is_query & (occ > 0.5)
is_negq  = is_query & (occ < 0.5)

B, M, N, _ = pred_local.shape
pred_f = pred_local.detach().float().view(B*M, N, 3)
tgt_f  = tgt_local.detach().float().view(B*M, N, 3)
cen_f  = centers_cm.detach().float().view(B*M, 3)
prob_f = prob.detach().float().view(B*M)
pos_f  = is_posq.view(B*M)

# ---- 1) HARD PROOF: are they identical? ----
if pos_f.any():
    dif = (pred_f[pos_f] - tgt_f[pos_f]).abs()
    max_abs = dif.max().item()
    mean_abs = dif.mean().item()
    rms = torch.sqrt((dif**2).mean()).item()
    print(f"[check] pos queries: max|pred-gt|={max_abs:.6f} | mean|.|={mean_abs:.6f} | rms={rms:.6f}")
    print(f"[check] allclose(1e-4)? {torch.allclose(pred_f[pos_f], tgt_f[pos_f], atol=1e-4, rtol=0)}")
else:
    print("[check] no pos queries in this batch?")

# ---- pick a few pos queries: best & worst by L2 error ----
K = min(int(pos_f.sum().item()), 40)
pos_idx = torch.nonzero(pos_f).squeeze(1)
err_q = torch.sqrt(((pred_f[pos_idx] - tgt_f[pos_idx])**2).mean(dim=(1,2)))  # (num_pos,)
order = torch.argsort(err_q)
take = torch.cat([order[:K//2], order[-K//2:]], dim=0)
sel_idx = pos_idx[take]

print(f"[pick] exporting {sel_idx.numel()} pos queries | err_q min={err_q[order[0]].item():.6f} max={err_q[order[-1]].item():.6f}")

# ---- 2) OVERLAY export: GT green, Pred red (same queries) ----
gt_world = (cen_f[sel_idx].unsqueeze(1) + tgt_f[sel_idx]  * S) * unit_mul
pr_world = (cen_f[sel_idx].unsqueeze(1) + pred_f[sel_idx] * S) * unit_mul

gt_xyz = gt_world.reshape(-1,3).cpu().numpy()
pr_xyz = pr_world.reshape(-1,3).cpu().numpy()

# stack into one cloud with colors
xyz = np.concatenate([gt_xyz, pr_xyz], axis=0)
rgb = np.concatenate([
    np.tile(np.array([[0,255,0]], dtype=np.uint8), (gt_xyz.shape[0],1)),   # GT green
    np.tile(np.array([[255,0,0]], dtype=np.uint8), (pr_xyz.shape[0],1)),   # Pred red
], axis=0)

out = dbg_dir / f"overlay_gt_green_pred_red_{scale_mode}_{rhino_unit}.ply"
write_ply_rgb_ascii(out, xyz, rgb)
print("[export]", out, "points=", xyz.shape[0])

# ---- 3) also export centers colored by prob (for context) ----
# (blue=low, red=high)
cent_q = is_query.view(B*M)
cent_xyz = (cen_f[cent_q] * unit_mul).cpu().numpy()
cent_p = prob_f[cent_q].cpu().numpy()
r = (255*cent_p).astype(np.uint8)
g = np.zeros_like(r, dtype=np.uint8)
b = (255*(1.0-cent_p)).astype(np.uint8)
cent_rgb = np.stack([r,g,b], axis=1)
out2 = dbg_dir / f"centers_prob_redblue_{rhino_unit}.ply"
write_ply_rgb_ascii(out2, cent_xyz, cent_rgb)
print("[export]", out2, "points=", cent_xyz.shape[0])

print("Done. Dir:", dbg_dir.resolve())


# %% [markdown]
# Visualize Check

# %%
# %%
# Visualize frame_cache/*.npz  (reconstruct world points from centers_cm + local_pts)

import numpy as np
import pathlib

VOX_CM = 20.0          # ✅ 必须同你 cache 时用的一样
CAP_POINTS = 800_000   # ✅ 防止太大卡 Rhino，可自行调

def save_ply(filename, points, color=(255, 255, 255)):
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x","property float y","property float z",
        "property uchar red","property uchar green","property uchar blue",
        "end_header"
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")

def frame_npz_to_world_points(npz_path, vox_cm=20.0):
    z = np.load(npz_path, allow_pickle=True)
    if "empty" in z.files:
        return np.zeros((0,3), np.float32)

    centers = z["centers_cm"].astype(np.float32)           # (V,3)
    local   = z["local_pts"].astype(np.float32)            # (V,N,3) in ~[-1,1]
    half = float(vox_cm) / 2.0

    pts = centers[:, None, :] + local * half               # (V,N,3)
    pts = pts.reshape(-1, 3).astype(np.float32)

    # remove NaN/Inf
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

data_path = r"K:\MasterEssay\Attempt_source_03\LearnCache_seq_fast\frame_cache\000000.npz"
npz_dir = pathlib.Path(data_path).parent
file_list = sorted(npz_dir.glob("*.npz"))

print(f"Found {len(file_list)} npz files in: {npz_dir}")

n = min(30, len(file_list))

for i in range(n):
    npz_file = file_list[i]
    pts = frame_npz_to_world_points(str(npz_file), vox_cm=VOX_CM)

    # cap
    if pts.shape[0] > CAP_POINTS:
        idx = np.random.choice(pts.shape[0], CAP_POINTS, replace=False)
        pts = pts[idx]

    stem = npz_file.stem
    out_ply = npz_dir / f"frame_world_{i:03d}_{stem}.ply"
    save_ply(str(out_ply), pts, color=(200, 200, 200))
    print(f"[{i+1}/{n}] {npz_file.name} -> {out_ply.name} | pts={pts.shape[0]}")


# %% [markdown]
# Visualize Dataset
# 

# %%
# %% [Cell: Universal NPZ Inspector - Auto-detect Format]

import numpy as np
import os

# --- 1. PLY 導出函數 ---
def save_ply(filename, points, color=(255, 255, 255)):
    if points is None or len(points) == 0:
        print(f"⚠️ Warning: No points to save for {os.path.basename(filename)}")
        return
    points = points.astype(np.float32).reshape(-1, 3)
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    
    header = [
        "ply", "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
    print(f"✅ Saved: {filename} ({len(points)} pts)")

# --- 2. 核心：格式解碼器 ---
def inspect_and_convert(npz_path, vox_cm=20.0):
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}"); return

    print(f"\n🔍 Inspecting: {os.path.basename(npz_path)}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.files)
        print(f"📂 Keys: {keys}")
    except Exception as e:
        print(f"❌ Load failed: {e}"); return

    # === Case A: 新版 Cache 格式 (centers_cm + local_pts) ===
    if 'centers_cm' in keys and 'local_pts' in keys:
        print("💡 Detected Format: Sequence Token Cache (Compressed)")
        
        # 1. Load Data
        centers = data['centers_cm'].astype(np.float32) # (V, 3)
        local   = data['local_pts'].astype(np.float32)  # (V, N, 3)
        
        print(f"   - Voxel Count: {centers.shape[0]}")
        print(f"   - Points per Voxel: {local.shape[1]}")
        
        # 2. Decode to World Coordinates
        # Formula: World = Center + Local * (VoxSize / 2)
        half = float(vox_cm) / 2.0
        # shape broadcast: (V, 1, 3) + (V, N, 3) -> (V, N, 3)
        world_pts = centers[:, None, :] + local * half
        
        # 3. Flatten and Save
        world_pts = world_pts.reshape(-1, 3)
        
        # 簡單過濾一下無效點 (local=0 的 padding)
        # 如果 local 全是 0，说明是 padding 点，通常可以过滤，或者全部导出来看看
        save_ply(npz_path.replace(".npz", "_decoded.ply"), world_pts, color=(0, 200, 255))
        
    # === Case B: 舊版 Chain 格式 (mem_pts + target_vis_pts) ===
    elif 'mem_pts' in keys:
        print("💡 Detected Format: Raw Point Chain")
        save_ply(npz_path.replace(".npz", "_mem.ply"), data['mem_pts'], color=(180,180,180))
        if 'target_vis_pts' in keys:
            save_ply(npz_path.replace(".npz", "_tgt.ply"), data['target_vis_pts'], color=(0,255,100))
            
    else:
        print("❓ Unknown format. Cannot visualize.")

# --- 執行 ---
# 替換成你剛剛輸出的那個 npz 路徑
TARGET = r"K:\MasterEssay\Attempt_source_03\LearnCache_seq_fast\frame_cache\000000.npz"

inspect_and_convert(TARGET, vox_cm=20.0) # 確保 VOX_CM 與生成時一致

# %%
import numpy as np
import os

def save_ply(filename, points, color=(255, 255, 255)):
    """修正后的 PLY 写入，适配 Rhino 8"""
    # 确保颜色是整数
    r, g, b = int(color[0]), int(color[1]), int(color[2])
    
    # 标头必须严格左对齐，不能有前导空格
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            # 保证坐标精度，并确保颜色在 0-255 之间
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti

def apply_T(pts, T):
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
    return (T @ pts_h.T).T[:, :3]

# 1. 加载文件
data_path = r"K:\MasterEssay\Attempt_source_03\LearnCache_vis_fast_fixed\chain_00000_ovlp29_step22.npz"
data = np.load(data_path)

import numpy as np

def apply_T_col(pts, T):
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
    return (T @ pts_h.T).T[:, :3]

def apply_T_row(pts, T):
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
    return (pts_h @ T.T)[:, :3]

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti

def nn_mean_dist(A, B, n=800):
    # 粗暴最近邻（抽样 + O(n^2)），但够用来判对/错
    rng = np.random.default_rng(0)
    ia = rng.choice(len(A), size=min(n, len(A)), replace=False)
    ib = rng.choice(len(B), size=min(n, len(B)), replace=False)
    As = A[ia].astype(np.float32)
    Bs = B[ib].astype(np.float32)
    # (na, nb, 3)
    d2 = np.sum((As[:, None, :] - Bs[None, :, :])**2, axis=2)
    return float(np.mean(np.sqrt(np.min(d2, axis=1))))

T = T_rel.astype(np.float32)

# 基本检查：R 是否正交、det 是否 ~1
R = T[:3,:3]
print("det(R) =", np.linalg.det(R))
print("R^T R deviation =", np.linalg.norm(R.T @ R - np.eye(3), ord='fro'))
print("t =", T[:3,3])

# 四种候选对齐方式：方向 x 乘法约定
cand = []

# 1) 列向量 + curr->next：mem -> next 应该贴 target
cand.append(("col, mem->T", nn_mean_dist(apply_T_col(mem_pts, T), target_pts)))

# 2) 列向量 + next->curr：target -> curr 应该贴 mem
cand.append(("col, target->invT", nn_mean_dist(apply_T_col(target_pts, inv_T(T)), mem_pts)))

# 3) 行向量约定 + curr->next
cand.append(("row, mem->T", nn_mean_dist(apply_T_row(mem_pts, T), target_pts)))

# 4) 行向量约定 + next->curr
cand.append(("row, target->invT", nn_mean_dist(apply_T_row(target_pts, inv_T(T)), mem_pts)))

for name, err in cand:
    print(name, "mean NN dist =", err)


mem_pts = data['mem_pts']           # 当前帧点云 (Camera 1 视角)
target_pts = data['target_vis_pts'] # 下一帧点云 (Camera 2 视角)
T_rel = data['T_curr_to_next']      # 从 1 到 2 的变换矩阵

# 2. 核心步骤：将下一帧点云转换回当前帧的坐标系
# 逻辑：P_next = T_rel * P_curr  =>  P_curr = inv(T_rel) * P_next
T_next_to_curr = inv_T(T_rel)
target_pts_in_curr_frame = target_pts
# 3. 导出为 PLY 文件
save_ply("points_current.ply", mem_pts, color=(200, 200, 200))    # 灰色：当前点
save_ply("points_next_relative.ply", target_pts_in_curr_frame, color=(0, 255, 100)) # 绿色：新视野点

print("转换完成！请将生成的两个 .ply 文件拖入 Rhino 8。")

# %%
import numpy as np
from scipy.spatial import cKDTree

def calculate_overlap_and_new_points(npz_path, threshold=5.0):
    """
    计算重叠度并返回新增点的索引
    threshold: 距离阈值，建议设为你的 VOXEL 大小 (如 5.0 cm)
    """
    data = np.load(npz_path)
    mem_pts = data['mem_pts']           # P1: 当前帧 (Reference)
    target_pts_local = data['target_vis_pts'] # P2_local
    T_rel = data['T_curr_to_next']      # T_1_to_2

    # 1. 坐标变换：把下一帧转回当前帧坐标系
    # P_next_in_curr = inv(T_rel) @ P_next_local
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    # 逆变换矩阵的快速计算
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    
    # 执行变换
    pts_h = np.concatenate([target_pts_local, np.ones((len(target_pts_local), 1))], axis=1)
    target_pts_in_curr = (T_inv @ pts_h.T).T[:, :3]

    if len(mem_pts) == 0 or len(target_pts_in_curr) == 0:
        return 0.0, target_pts_in_curr

    # 2. 使用 KDTree 寻找近邻
    tree = cKDTree(mem_pts)
    
    # 对 target_pts_in_curr 中的每个点，去 mem_pts 里找最近距离
    distances, _ = tree.query(target_pts_in_curr, k=1)

    # 3. 计算统计数据
    is_overlap = distances < threshold
    overlap_count = np.sum(is_overlap)
    total_count = len(target_pts_in_curr)
    
    overlap_rate = overlap_count / total_count
    new_points = target_pts_in_curr[~is_overlap] # 提取新增点

    return overlap_rate, new_points

# 运行示例
path = r"K:\MasterEssay\Attempt_source_03\LearnCache_vis_fast_fixed\chain_00002_ovlp12_step22.npz"
rate, new_pts = calculate_overlap_and_new_points(path, threshold=5.0)

print(f"文件: {path}")
print(f"重叠度 (Overlap Rate): {rate:.2%}")
print(f"新增点比例 (New Info Rate): {1-rate:.2%}")
print(f"新增点数量: {len(new_pts)}")



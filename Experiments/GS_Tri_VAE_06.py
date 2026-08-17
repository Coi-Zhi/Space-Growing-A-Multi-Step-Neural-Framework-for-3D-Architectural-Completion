# %%
# Cell 1 — Config / Imports

import os
import glob
import math
import json
import time
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Repro
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Paths (EDIT THESE)
# -----------------------------
# QF_GS.py 輸出的 per-frame cache npz 所在資料夾
ROOT_DIR  = r".\LearnCache_gs"
CACHE_DIR = os.path.join(ROOT_DIR, "gaussian_frame_cache")   # 這個先係每幀 npz 所在
OUT_DIR   = r".\GS_Triplane_vae_out"
os.makedirs(OUT_DIR, exist_ok=True)

# 預先轉好嘅 triplane cache
TRIPLANE_CACHE_NPZ = os.path.join(OUT_DIR, "triplane_dataset_cache.npz")
TRIPLANE_META_JSON = os.path.join(OUT_DIR, "triplane_dataset_meta.json")

# checkpoint
CKPT_PATH_BEST = os.path.join(OUT_DIR, "triplane_2d_vae_best.pt")
CKPT_PATH_LAST = os.path.join(OUT_DIR, "triplane_2d_vae_last.pt")

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------
# Geometry / source-data assumptions
# -----------------------------
# 要同你 QF_GS 產生 local_pts 時一致，QF_GS 預設係 voxel 內 local point normalize 到 [-1, 1] 左右
VOX_CM = 20.0
LOCAL_PTS_CLIP = 1.25

# fit patch gaussian 時嘅數值穩定參數
COV_REG_CM2 = 1e-2
SCALE_STD_FACTOR = 2.0
MIN_SCALE_CM = 1.0
MAX_SCALE_CM = 250.0
MIN_VALID_PATCH_PTS = 4
MIN_OPACITY = 0.05
MAX_OPACITY = 1.0

# -----------------------------
# Triplane shape
# -----------------------------
TRIPLANE_RES = 128        # 每張 plane: 128 x 128
PLANE_CH = 32            # 每張 plane 32 channels
TRIPLANE_CH = PLANE_CH * 3   # 96 channels total

# 目標 latent shape: (8,16,16)
LATENT_CH = 32
LATENT_RES = 16

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
BETA_KL = 1e-5
GRAD_CLIP = 1.0

# 可以先限制訓練 frame 數做 debug；None = 全部
MAX_FRAMES = None

print("TRIPLANE:", (TRIPLANE_CH, TRIPLANE_RES, TRIPLANE_RES))
print("LATENT  :", (LATENT_CH, LATENT_RES, LATENT_RES))

# %%
# Cell 2 — Helpers: load cache / reconstruct patch points / fit patch gaussians

def load_cache_npz(npz_path):
    """
    讀 QF_GS 輸出嘅單幀 cache。
    最少要求:
      centers_cm, local_pts, local_count
    """
    z = np.load(npz_path, allow_pickle=True)

    if "empty" in z.files:
        return None

    required = ["centers_cm", "local_pts", "local_count"]
    for k in required:
        if k not in z.files:
            return None

    out = {
        "centers_cm": z["centers_cm"].astype(np.float32),
        "local_pts": z["local_pts"].astype(np.float32),
        "local_count": z["local_count"].astype(np.int32),
    }

    if "frame_id" in z.files:
        out["frame_id"] = int(z["frame_id"])
    else:
        out["frame_id"] = -1

    return out


def ensure_right_handed(R):
    R = np.asarray(R, dtype=np.float32).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def quat_from_rotmat(R):
    """
    輸出 wxyz
    """
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-8)
    return q


def reconstruct_patch_points(centers_cm, local_pts, local_count, vox_cm=VOX_CM, local_pts_clip=LOCAL_PTS_CLIP):
    """
    將 QF_GS cache 內嘅局部 normalized points 重建番世界座標點。
    返回:
      patches: list[(K,3)]
      raw_pts: (sumK, 3)
    """
    centers_cm = np.asarray(centers_cm, dtype=np.float32)
    local_pts = np.asarray(local_pts, dtype=np.float32)
    local_count = np.asarray(local_count, dtype=np.int32)

    half = float(vox_cm) / 2.0
    patches = []
    raw_all = []

    V = centers_cm.shape[0]
    max_local = local_pts.shape[1]

    for i in range(V):
        cen = centers_cm[i]
        n = int(local_count[i]) if i < len(local_count) else max_local
        n = max(0, min(n, max_local))

        loc = local_pts[i, :n]
        if loc.shape[0] == 0:
            patches.append(np.zeros((0, 3), dtype=np.float32))
            continue

        valid = np.isfinite(loc).all(axis=1)
        valid &= (np.max(np.abs(loc), axis=1) <= float(local_pts_clip))
        loc = loc[valid]

        if loc.shape[0] == 0:
            patches.append(np.zeros((0, 3), dtype=np.float32))
            continue

        pts = cen[None, :] + loc * half
        pts = pts.astype(np.float32)

        patches.append(pts)
        raw_all.append(pts)

    if len(raw_all) == 0:
        raw_pts = np.zeros((0, 3), dtype=np.float32)
    else:
        raw_pts = np.concatenate(raw_all, axis=0).astype(np.float32)

    return patches, raw_pts


def fit_patch_gaussian(arr_cm):
    """
    一個 patch points -> 一粒 patch gaussian
    返回:
      mu(3), scales_cm(3), quat_wxyz(4), opacity(1)
    """
    arr_cm = np.asarray(arr_cm, dtype=np.float32)
    K = arr_cm.shape[0]

    if K == 0:
        mu = np.zeros((3,), dtype=np.float32)
        sc = np.full((3,), VOX_CM * 0.5, dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        op = np.float32(MIN_OPACITY)
        return mu, sc, q, op

    mu = arr_cm.mean(axis=0).astype(np.float32)

    if K <= 1:
        scales = np.array([VOX_CM * 0.5, VOX_CM * 0.5, VOX_CM * 0.5], dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        centered = arr_cm - mu[None, :]
        cov = (centered.T @ centered) / float(max(K - 1, 1))
        cov = cov.astype(np.float32)
        cov += np.eye(3, dtype=np.float32) * float(COV_REG_CM2)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        R = ensure_right_handed(eigvecs)
        quat = quat_from_rotmat(R)

        std = np.sqrt(np.clip(eigvals, 1e-6, None)).astype(np.float32)
        scales = std * float(SCALE_STD_FACTOR)
        scales = np.clip(scales, MIN_SCALE_CM, MAX_SCALE_CM).astype(np.float32)

    opacity = np.float32(
        np.clip(K / max(1.0, float(MIN_VALID_PATCH_PTS) * 2.0), MIN_OPACITY, MAX_OPACITY)
    )
    return mu, scales, quat, opacity


def build_full_patch_gaussians(pack):
    """
    單幀 cache -> 所有 patch gaussians
    返回:
      mu_all (M,3), sc_all (M,3), q_all (M,4), op_all (M,), count_all (M,)
    """
    patches, _ = reconstruct_patch_points(
        pack["centers_cm"], pack["local_pts"], pack["local_count"],
        vox_cm=VOX_CM, local_pts_clip=LOCAL_PTS_CLIP
    )

    mu_all, sc_all, q_all, op_all, count_all = [], [], [], [], []

    for arr in patches:
        if arr.shape[0] == 0:
            continue
        mu, sc, q, op = fit_patch_gaussian(arr)
        mu_all.append(mu)
        sc_all.append(sc)
        q_all.append(q)
        op_all.append(op)
        count_all.append(arr.shape[0])

    if len(mu_all) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(mu_all, axis=0).astype(np.float32),
        np.stack(sc_all, axis=0).astype(np.float32),
        np.stack(q_all, axis=0).astype(np.float32),
        np.asarray(op_all, dtype=np.float32),
        np.asarray(count_all, dtype=np.float32),
    )


def compute_world_stats_from_cache_paths(cache_paths):
    """
    用 centers_cm 粗略估全資料集 world_center / world_scale。
    只用嚟做 triplane normalization。
    """
    all_centers = []

    for p in cache_paths:
        pack = load_cache_npz(p)
        if pack is None:
            continue
        c = pack["centers_cm"]
        if c.shape[0] > 0:
            all_centers.append(c.astype(np.float32))

    if len(all_centers) == 0:
        raise RuntimeError("No valid frame cache found in CACHE_DIR.")

    cat = np.concatenate(all_centers, axis=0).astype(np.float32)
    world_min = cat.min(axis=0)
    world_max = cat.max(axis=0)
    world_center = 0.5 * (world_min + world_max)
    half_extent = 0.5 * (world_max - world_min)
    world_scale = float(max(half_extent.max(), 1.0))

    return {
        "world_min": world_min.astype(np.float32),
        "world_max": world_max.astype(np.float32),
        "world_center": world_center.astype(np.float32),
        "world_scale": np.float32(world_scale),
    }


print("helpers ready")

# %%
# Cell 3 — Triplane builder: patch gaussians -> (96, 128, 128)

def normalize_xyz_world(xyz_cm, world_center, world_scale):
    return ((xyz_cm - world_center[None, :]) / float(world_scale)).astype(np.float32)


def _to_pixel01(v, res):
    """
    v in [-1,1] -> pixel float in [0, res-1]
    """
    u = (v * 0.5) + 0.5
    u = np.clip(u, 0.0, 1.0)
    return u * float(res - 1)


def make_patch_feature32(mu_cm, sc_cm, quat_wxyz, opacity, count, world_center, world_scale):
    """
    為每個 patch gaussian 構建 32 維 feature，之後投影到各 plane。
    """
    xyz_n = ((mu_cm - world_center) / float(world_scale)).astype(np.float32)  # [-?, ?]
    xyz_n = np.clip(xyz_n, -3.0, 3.0)

    log_sc = np.log(np.clip(sc_cm / float(world_scale), 1e-4, 10.0)).astype(np.float32)
    sc_n = np.clip(sc_cm / float(world_scale), 0.0, 10.0).astype(np.float32)

    q = np.asarray(quat_wxyz, dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-8)

    count_norm = np.float32(np.clip(count / 64.0, 0.0, 4.0))
    volume_norm = np.float32(np.clip((sc_n[0] * sc_n[1] * sc_n[2]), 0.0, 10.0))

    sc_sorted = np.sort(sc_cm.astype(np.float32))[::-1]
    aniso1 = np.float32(sc_sorted[0] / max(sc_sorted[1], 1e-6))
    aniso2 = np.float32(sc_sorted[1] / max(sc_sorted[2], 1e-6))
    aniso1 = np.clip(aniso1, 1.0, 20.0)
    aniso2 = np.clip(aniso2, 1.0, 20.0)

    feat = np.zeros((32,), dtype=np.float32)
    feat[0:3] = xyz_n
    feat[3:6] = log_sc
    feat[6:10] = q
    feat[10] = float(opacity)
    feat[11] = count_norm
    feat[12:15] = np.abs(xyz_n)
    feat[15:18] = sc_n
    feat[18:21] = xyz_n * xyz_n
    feat[21:24] = sc_n * sc_n
    feat[24] = q[0] * float(opacity)
    feat[25] = np.sqrt(np.clip(q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 0.0, 1.0)).astype(np.float32)
    feat[26] = volume_norm
    feat[27] = aniso1 / 20.0
    feat[28] = aniso2 / 20.0
    feat[29] = 1.0
    feat[30] = np.float32(np.clip(float(opacity) * count_norm, 0.0, 4.0))
    feat[31] = np.float32(np.clip(float(opacity) * volume_norm, 0.0, 10.0))
    return feat.astype(np.float32)


def _accumulate_bilinear(plane_sum, plane_den, x, y, feat, weight):
    """
    plane_sum: (C,H,W)
    plane_den: (1,H,W)
    x,y: float pixel coord
    feat: (C,)
    """
    H = plane_sum.shape[1]
    W = plane_sum.shape[2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    dx = float(x - x0)
    dy = float(y - y0)

    w00 = (1.0 - dx) * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w10 = dx * (1.0 - dy)
    w11 = dx * dy

    taps = [
        (x0, y0, w00),
        (x0, y1, w01),
        (x1, y0, w10),
        (x1, y1, w11),
    ]

    for px, py, ww in taps:
        if ww <= 0.0:
            continue
        a = float(weight) * ww
        plane_sum[:, py, px] += feat * a
        plane_den[0, py, px] += a


def _finalize_plane(plane_sum, plane_den):
    """
    plane_sum / plane_den -> average features
    並將部分 channel 改成更穩定嘅 occupancy / density
    """
    out = plane_sum.copy()
    den = np.clip(plane_den, 1e-8, None)
    out = out / den

    density = np.log1p(plane_den[0])
    occ = (plane_den[0] > 0).astype(np.float32)

    # 用兩個 channel 明確放 occupancy / density
    out[29] = occ
    out[30] = np.clip(density, 0.0, 10.0)
    out[31] = np.clip(plane_den[0], 0.0, 10.0)

    # 沒有覆蓋嘅位置直接歸零
    zero_mask = (plane_den[0] <= 1e-8)
    out[:, zero_mask] = 0.0
    return out.astype(np.float32)


def build_triplane_from_patch_gaussians(mu_all, sc_all, q_all, op_all, count_all, world_center, world_scale, res=TRIPLANE_RES):
    """
    輸出:
      triplane_flat: (96, res, res)
    """
    plane_xy_sum = np.zeros((PLANE_CH, res, res), dtype=np.float32)
    plane_xz_sum = np.zeros((PLANE_CH, res, res), dtype=np.float32)
    plane_yz_sum = np.zeros((PLANE_CH, res, res), dtype=np.float32)

    plane_xy_den = np.zeros((1, res, res), dtype=np.float32)
    plane_xz_den = np.zeros((1, res, res), dtype=np.float32)
    plane_yz_den = np.zeros((1, res, res), dtype=np.float32)

    M = mu_all.shape[0]
    if M == 0:
        return np.zeros((TRIPLANE_CH, res, res), dtype=np.float32)

    xyz_n = normalize_xyz_world(mu_all, world_center, world_scale)  # (M,3)

    for i in range(M):
        mu = mu_all[i]
        sc = sc_all[i]
        q = q_all[i]
        op = float(op_all[i])
        cnt = float(count_all[i])

        feat32 = make_patch_feature32(mu, sc, q, op, cnt, world_center, world_scale)

        # weight：高 opacity + 點數多嘅 patch 佔比更高
        weight = float(op) * float(np.clip(cnt / 16.0, 0.25, 4.0))

        x_n, y_n, z_n = xyz_n[i]
        px = _to_pixel01(x_n, res)
        py = _to_pixel01(y_n, res)
        pz = _to_pixel01(z_n, res)

        _accumulate_bilinear(plane_xy_sum, plane_xy_den, px, py, feat32, weight)
        _accumulate_bilinear(plane_xz_sum, plane_xz_den, px, pz, feat32, weight)
        _accumulate_bilinear(plane_yz_sum, plane_yz_den, py, pz, feat32, weight)

    plane_xy = _finalize_plane(plane_xy_sum, plane_xy_den)
    plane_xz = _finalize_plane(plane_xz_sum, plane_xz_den)
    plane_yz = _finalize_plane(plane_yz_sum, plane_yz_den)

    triplane_flat = np.concatenate([plane_xy, plane_xz, plane_yz], axis=0).astype(np.float32)
    return triplane_flat


def build_triplane_from_cache_pack(pack, world_center, world_scale, res=TRIPLANE_RES):
    mu_all, sc_all, q_all, op_all, count_all = build_full_patch_gaussians(pack)
    tri = build_triplane_from_patch_gaussians(
        mu_all, sc_all, q_all, op_all, count_all,
        world_center=world_center,
        world_scale=world_scale,
        res=res,
    )
    return tri


print("triplane builder ready")

# %%
# Cell 4A — Robust load chain_indices / seq_index and build selected cache path list

# 建議：明確指定 index 檔所在資料夾
# 如果 chain_indices.npy / seq_index.npy 同 LearnCache_gs 同一層，就設成 "."
# 如果佢哋喺 LearnCache_gs 入面，就設成 CACHE_DIR
# 如果唔確定，可以先設 None，程式會自動搜尋
INDEX_DIR = ROOT_DIR

# 選擇模式：
#   "all"        -> 全部 cache（不依賴 chain/seq）
#   "chain"      -> 只用 chain_indices 裏面所有關鍵幀
#   "seq_unique" -> 用 seq_index 展開後，取所有出現過嘅 chain frame（推薦）
SELECT_MODE = "seq_unique"

# 如果想限制只取前 N 個 seq samples（debug 用）
MAX_SEQ_SAMPLES = None

# 如果想限制最後選中 frame 數（debug 用）
MAX_SELECTED_FRAMES = None


def norm_path(p):
    return os.path.normpath(os.path.abspath(p))


def cache_idx_to_npz_path(cache_idx, cache_dir):
    return os.path.join(cache_dir, f"{int(cache_idx):06d}.npz")


def candidate_index_dirs(cache_dir, index_dir=None):
    """
    依次嘗試：
      1) 用戶明確指定 INDEX_DIR
      2) CACHE_DIR
      3) CACHE_DIR 的父資料夾
      4) 當前工作目錄
    """
    cand = []

    if index_dir is not None:
        cand.append(index_dir)

    cand.append(cache_dir)
    cand.append(os.path.dirname(cache_dir))
    cand.append(".")

    # 去重 + 規範化
    out = []
    seen = set()
    for p in cand:
        if p is None:
            continue
        q = norm_path(p)
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def find_existing_file(filename, search_dirs):
    tried = []
    for d in search_dirs:
        p = os.path.join(d, filename)
        tried.append(norm_path(p))
        if os.path.exists(p):
            return norm_path(p), tried
    return None, tried


def resolve_index_files(cache_dir, index_dir=None, require_seq=True):
    search_dirs = candidate_index_dirs(cache_dir, index_dir=index_dir)

    chain_path, tried_chain = find_existing_file("chain_indices.npy", search_dirs)
    if chain_path is None:
        raise FileNotFoundError(
            "Cannot find chain_indices.npy.\n"
            "Searched:\n  - " + "\n  - ".join(tried_chain)
        )

    seq_path = None
    tried_seq = []
    if require_seq:
        seq_path, tried_seq = find_existing_file("seq_index.npy", search_dirs)
        if seq_path is None:
            raise FileNotFoundError(
                "Cannot find seq_index.npy.\n"
                "Searched:\n  - " + "\n  - ".join(tried_seq)
            )

    return chain_path, seq_path, search_dirs


def load_chain_indices(chain_index_npy):
    arr = np.load(chain_index_npy)
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"chain_indices.npy is empty: {chain_index_npy}")
    return arr


def load_seq_index(seq_index_npy):
    arr = np.load(seq_index_npy)
    arr = np.asarray(arr, dtype=np.int32)

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise RuntimeError(f"seq_index.npy has invalid shape: {arr.shape}")
        arr = arr.reshape(-1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"seq_index.npy expected shape (N,2), got {arr.shape}")

    if arr.shape[0] == 0:
        raise RuntimeError(f"seq_index.npy is empty: {seq_index_npy}")

    return arr


def expand_seq_pairs_to_chain_positions(seq_pairs, chain_len):
    """
    seq_pairs: (N,2), each row is (start, L)

    這裡按你之前 pipeline 的假設處理：
    覆蓋 chain position [start, start+L]
    即共 L+1 個 frame
    """
    used = set()

    for start, L in seq_pairs:
        start = int(start)
        L = int(L)

        if start < 0:
            continue
        if L < 1:
            continue
        if start >= chain_len:
            continue

        end = min(start + L, chain_len - 1)
        for p in range(start, end + 1):
            used.add(p)

    return np.asarray(sorted(used), dtype=np.int32)


def build_selected_cache_paths(cache_dir, select_mode="seq_unique", index_dir=None):
    """
    返回:
      selected_cache_indices: (K,)
      selected_cache_paths: list[str]
      select_info: dict
    """
    cache_dir = norm_path(cache_dir)

    if select_mode == "all":
        cache_paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        cache_indices = []

        for p in cache_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                cache_indices.append(int(stem))
            except Exception:
                continue

        cache_indices = np.asarray(sorted(set(cache_indices)), dtype=np.int32)
        cache_paths = [cache_idx_to_npz_path(i, cache_dir) for i in cache_indices]

        if MAX_SELECTED_FRAMES is not None:
            cache_indices = cache_indices[: int(MAX_SELECTED_FRAMES)]
            cache_paths = cache_paths[: int(MAX_SELECTED_FRAMES)]

        info = {
            "mode": "all",
            "cache_dir": cache_dir,
            "num_selected": int(len(cache_indices)),
        }
        return cache_indices, cache_paths, info

    require_seq = (select_mode == "seq_unique")
    chain_path, seq_path, searched_dirs = resolve_index_files(
        cache_dir,
        index_dir=index_dir,
        require_seq=require_seq
    )

    chain_cache_ids = load_chain_indices(chain_path)

    info = {
        "mode": select_mode,
        "cache_dir": cache_dir,
        "chain_index_npy": chain_path,
        "seq_index_npy": seq_path,
        "searched_dirs": searched_dirs,
        "chain_len": int(len(chain_cache_ids)),
    }

    if select_mode == "chain":
        selected_cache_indices = chain_cache_ids.copy()
        info["num_selected_before_dedup"] = int(len(selected_cache_indices))

    elif select_mode == "seq_unique":
        seq_pairs = load_seq_index(seq_path)

        if MAX_SEQ_SAMPLES is not None:
            seq_pairs = seq_pairs[: int(MAX_SEQ_SAMPLES)]

        used_chain_pos = expand_seq_pairs_to_chain_positions(seq_pairs, len(chain_cache_ids))
        selected_cache_indices = chain_cache_ids[used_chain_pos]

        info["num_seq_pairs"] = int(seq_pairs.shape[0])
        info["num_used_chain_pos"] = int(len(used_chain_pos))
        info["num_selected_before_dedup"] = int(len(selected_cache_indices))

    else:
        raise ValueError(f"Unknown SELECT_MODE: {select_mode}")

    # 去重 + 排序（保持 cache index 時間順序）
    selected_cache_indices = np.asarray(sorted(set(map(int, selected_cache_indices))), dtype=np.int32)

    if MAX_SELECTED_FRAMES is not None:
        selected_cache_indices = selected_cache_indices[: int(MAX_SELECTED_FRAMES)]

    selected_cache_paths = [cache_idx_to_npz_path(i, cache_dir) for i in selected_cache_indices]

    final_indices = []
    final_paths = []
    missing = []

    for idx, p in zip(selected_cache_indices, selected_cache_paths):
        if os.path.exists(p):
            final_indices.append(int(idx))
            final_paths.append(p)
        else:
            missing.append(int(idx))

    final_indices = np.asarray(final_indices, dtype=np.int32)

    info["num_selected"] = int(len(selected_cache_indices))
    info["num_existing"] = int(len(final_indices))
    info["num_missing"] = int(len(missing))
    info["missing_preview"] = missing[:20]

    return final_indices, final_paths, info


selected_cache_indices, cache_paths, select_info = build_selected_cache_paths(
    CACHE_DIR,
    select_mode=SELECT_MODE,
    index_dir=INDEX_DIR,
)

print("SELECT_MODE:", SELECT_MODE)
print("selected cache files:", len(cache_paths))
print("selection info:")
for k, v in select_info.items():
    print(f"  {k}: {v}")

print("selected index preview:", selected_cache_indices[: min(20, len(selected_cache_indices))])

# %%
# Cell 4B — Compute world stats and precompute triplanes ONLY for selected cache files
# 如果已有相同設定的輸出，就直接沿用，不重新跑

def _normalize_for_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: _normalize_for_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_normalize_for_json(v) for v in x]
    return x


def build_triplane_build_signature():
    """
    只包含會影響 triplane cache 內容的關鍵設定。
    如果這些設定完全相同，就可視為可重用。
    """
    sig = {
        "cache_dir": os.path.normpath(os.path.abspath(CACHE_DIR)),
        "select_mode": SELECT_MODE,
        "selection_info": _normalize_for_json(select_info),
        "selected_cache_indices": _normalize_for_json(selected_cache_indices.tolist()),
        "num_files": int(len(cache_paths)),
        "triplane_res": int(TRIPLANE_RES),
        "plane_ch": int(PLANE_CH),
        "triplane_ch": int(TRIPLANE_CH),
        "latent_ch": int(LATENT_CH),
        "latent_res": int(LATENT_RES),
        "vox_cm": float(VOX_CM),
        "local_pts_clip": float(LOCAL_PTS_CLIP),
        "cov_reg_cm2": float(COV_REG_CM2),
        "scale_std_factor": float(SCALE_STD_FACTOR),
        "min_scale_cm": float(MIN_SCALE_CM),
        "max_scale_cm": float(MAX_SCALE_CM),
        "min_valid_patch_pts": int(MIN_VALID_PATCH_PTS),
        "min_opacity": float(MIN_OPACITY),
        "max_opacity": float(MAX_OPACITY),
    }
    return sig


def can_reuse_existing_triplane_cache(npz_path, meta_json_path, expected_sig):
    """
    判斷是否可直接重用舊的 triplane cache
    """
    if not os.path.exists(npz_path):
        return False, "npz cache file not found"
    if not os.path.exists(meta_json_path):
        return False, "meta json file not found"

    try:
        with open(meta_json_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
    except Exception as e:
        return False, f"failed to read meta json: {e}"

    old_sig = old_meta.get("build_signature", None)
    if old_sig is None:
        return False, "meta json missing build_signature"

    if old_sig != expected_sig:
        return False, "build_signature mismatch"

    try:
        z = np.load(npz_path, allow_pickle=True)

        required_keys = ["triplane_all", "frame_id_all", "cache_idx_all", "path_all",
                         "world_min", "world_max", "world_center", "world_scale"]
        for k in required_keys:
            if k not in z.files:
                return False, f"npz missing key: {k}"

        triplane_all = z["triplane_all"]
        frame_id_all = z["frame_id_all"]
        cache_idx_all = z["cache_idx_all"]

        if triplane_all.ndim != 4:
            return False, f"triplane_all ndim invalid: {triplane_all.ndim}"
        if triplane_all.shape[1:] != (TRIPLANE_CH, TRIPLANE_RES, TRIPLANE_RES):
            return False, f"triplane_all shape mismatch: {triplane_all.shape}"
        if frame_id_all.shape[0] != triplane_all.shape[0]:
            return False, "frame_id_all length mismatch"
        if cache_idx_all.shape[0] != triplane_all.shape[0]:
            return False, "cache_idx_all length mismatch"

        # 再確認 cache_idx_all 跟目前 selection 完全一致
        expected_idx = np.asarray(selected_cache_indices, dtype=np.int32)
        existing_idx = np.asarray(cache_idx_all, dtype=np.int32)

        if expected_idx.shape != existing_idx.shape:
            return False, "cache_idx_all shape mismatch with current selection"
        if not np.array_equal(expected_idx, existing_idx):
            return False, "cache_idx_all content mismatch with current selection"

    except Exception as e:
        return False, f"failed to validate npz: {e}"

    return True, "matched existing cache"


if len(cache_paths) == 0:
    raise RuntimeError("No selected cache paths. Check CACHE_DIR / INDEX_DIR / SELECT_MODE / chain_indices / seq_index.")

build_signature = build_triplane_build_signature()

reuse_ok, reuse_reason = can_reuse_existing_triplane_cache(
    TRIPLANE_CACHE_NPZ,
    TRIPLANE_META_JSON,
    build_signature
)

if reuse_ok:
    print("[Reuse] Found existing triplane cache with same settings.")
    print("[Reuse] Reason:", reuse_reason)

    z = np.load(TRIPLANE_CACHE_NPZ, allow_pickle=True)
    triplane_all = z["triplane_all"].astype(np.float32)
    frame_id_all = z["frame_id_all"].astype(np.int32)
    cache_idx_all = z["cache_idx_all"].astype(np.int32)
    path_all = z["path_all"]

    world_min = z["world_min"].astype(np.float32)
    world_max = z["world_max"].astype(np.float32)
    world_center = z["world_center"].astype(np.float32)
    world_scale = float(z["world_scale"])

    print("loaded:", TRIPLANE_CACHE_NPZ)
    print("triplane_all shape:", triplane_all.shape)
    print("frame_id_all shape:", frame_id_all.shape)
    print("cache_idx_all shape:", cache_idx_all.shape)
    print("world_min   :", world_min)
    print("world_max   :", world_max)
    print("world_center:", world_center)
    print("world_scale :", world_scale)

else:
    print("[Rebuild] Existing cache cannot be reused.")
    print("[Rebuild] Reason:", reuse_reason)

    # 先用「被選中」幀估 world stats
    stats = compute_world_stats_from_cache_paths(cache_paths)
    world_min = stats["world_min"]
    world_max = stats["world_max"]
    world_center = stats["world_center"]
    world_scale = float(stats["world_scale"])

    print("world_min   :", world_min)
    print("world_max   :", world_max)
    print("world_center:", world_center)
    print("world_scale :", world_scale)

    meta = {
        "build_signature": build_signature,
        "cache_dir": os.path.normpath(os.path.abspath(CACHE_DIR)),
        "select_mode": SELECT_MODE,
        "selection_info": _normalize_for_json(select_info),
        "num_files": len(cache_paths),
        "selected_cache_indices": selected_cache_indices.tolist(),
        "triplane_res": int(TRIPLANE_RES),
        "plane_ch": int(PLANE_CH),
        "triplane_ch": int(TRIPLANE_CH),
        "latent_ch": int(LATENT_CH),
        "latent_res": int(LATENT_RES),
        "vox_cm": float(VOX_CM),
        "local_pts_clip": float(LOCAL_PTS_CLIP),
        "cov_reg_cm2": float(COV_REG_CM2),
        "scale_std_factor": float(SCALE_STD_FACTOR),
        "min_scale_cm": float(MIN_SCALE_CM),
        "max_scale_cm": float(MAX_SCALE_CM),
        "min_valid_patch_pts": int(MIN_VALID_PATCH_PTS),
        "min_opacity": float(MIN_OPACITY),
        "max_opacity": float(MAX_OPACITY),
        "world_min": world_min.tolist(),
        "world_max": world_max.tolist(),
        "world_center": world_center.tolist(),
        "world_scale": float(world_scale),
    }

    with open(TRIPLANE_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    tri_list = []
    frame_id_list = []
    cache_idx_list = []
    valid_path_list = []

    t0 = time.time()

    for i, (cache_idx, p) in enumerate(zip(selected_cache_indices, cache_paths)):
        pack = load_cache_npz(p)
        if pack is None:
            continue

        tri = build_triplane_from_cache_pack(pack, world_center, world_scale, res=TRIPLANE_RES)

        tri_list.append(tri.astype(np.float32))
        frame_id_list.append(int(pack["frame_id"]))
        cache_idx_list.append(int(cache_idx))
        valid_path_list.append(p)

        if ((i + 1) % 20 == 0) or (i == len(cache_paths) - 1):
            print(f"[{i+1}/{len(cache_paths)}] processed")

    if len(tri_list) == 0:
        raise RuntimeError("No valid triplane built from selected cache files.")

    triplane_all = np.stack(tri_list, axis=0).astype(np.float32)   # (N,96,128,128)
    frame_id_all = np.asarray(frame_id_list, dtype=np.int32)
    cache_idx_all = np.asarray(cache_idx_list, dtype=np.int32)
    path_all = np.asarray(valid_path_list, dtype=object)

    np.savez_compressed(
        TRIPLANE_CACHE_NPZ,
        triplane_all=triplane_all,
        frame_id_all=frame_id_all,
        cache_idx_all=cache_idx_all,
        path_all=path_all,
        world_min=world_min.astype(np.float32),
        world_max=world_max.astype(np.float32),
        world_center=world_center.astype(np.float32),
        world_scale=np.float32(world_scale),
    )

    print("saved:", TRIPLANE_CACHE_NPZ)
    print("triplane_all shape:", triplane_all.shape)
    print("frame_id_all shape:", frame_id_all.shape)
    print("cache_idx_all shape:", cache_idx_all.shape)
    print("time:", time.time() - t0, "sec")

# %%
# Cell 5 — Dataset / DataLoader

class PrecomputedTriplaneDataset(Dataset):
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.triplane_all = z["triplane_all"].astype(np.float32)
        self.frame_id_all = z["frame_id_all"].astype(np.int32)

    def __len__(self):
        return self.triplane_all.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.triplane_all[idx])  # (96,128,128)
        fid = int(self.frame_id_all[idx])
        return x, fid


ds = PrecomputedTriplaneDataset(TRIPLANE_CACHE_NPZ)
dl = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
    drop_last=False,
)

print("dataset size:", len(ds))
print("batch count :", len(dl))

xb, fidb = next(iter(dl))
print("batch x:", xb.shape, xb.dtype)
print("batch frame ids:", fidb[: min(4, len(fidb))])

# %%
# Cell 6 — 2D Triplane VAE model (FIXED: train/inference use the same output domain)

class ResBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder2D(nn.Module):
    def __init__(self, in_ch=TRIPLANE_CH, latent_ch=LATENT_CH):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, 4, stride=2, padding=1),   # 128 -> 64
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            ResBlock2D(96),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(96, 128, 4, stride=2, padding=1),  # 64 -> 32
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            ResBlock2D(128),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 160, 4, stride=2, padding=1), # 32 -> 16
            nn.GroupNorm(8, 160),
            nn.SiLU(inplace=True),
            ResBlock2D(160),
            ResBlock2D(160),
        )

        self.to_mu = nn.Conv2d(160, latent_ch, 3, padding=1)
        self.to_logvar = nn.Conv2d(160, latent_ch, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class Decoder2D(nn.Module):
    def __init__(self, out_ch=TRIPLANE_CH, latent_ch=LATENT_CH):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_ch, 160, 3, padding=1),
            nn.GroupNorm(8, 160),
            nn.SiLU(inplace=True),
            ResBlock2D(160),
            ResBlock2D(160),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(160, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            ResBlock2D(128),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            ResBlock2D(96),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResBlock2D(64),
        )

        self.out = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.in_proj(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        x_hat_raw = self.out(h)
        return x_hat_raw


class Triplane2DVAE(nn.Module):
    """
    關鍵修正：
    - decoder 永遠先輸出 raw logits
    - 用同一個 soft_decode 規則，給 train/inference 共用
    - 暫時不做 hard mask，先消除 train/inference mismatch
    """
    def __init__(self, in_ch=TRIPLANE_CH, latent_ch=LATENT_CH):
        super().__init__()
        self.encoder = Encoder2D(in_ch=in_ch, latent_ch=latent_ch)
        self.decoder = Decoder2D(out_ch=in_ch, latent_ch=latent_ch)

        # 三個 plane 的 occupancy channel
        self.occ_idx = [29, 61, 93]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def soft_decode_triplane(self, x_hat_raw):
        """
        統一的 soft decode：
        - 只把 occupancy logits -> sigmoid
        - 不做 hard mask
        - 這個輸出同時給 training 的 feature loss 同 inference 使用
        """
        out = x_hat_raw.clone()
        out[:, self.occ_idx, :, :] = torch.sigmoid(out[:, self.occ_idx, :, :])
        return out

    def hard_decode_triplane(self, x_hat_raw, threshold=0.1):
        """
        只做診斷/可視化時用，不參與訓練主路徑
        """
        out = self.soft_decode_triplane(x_hat_raw)
        mask = (out[:, self.occ_idx, :, :] > float(threshold)).float()
        mask_expanded = torch.repeat_interleave(mask, PLANE_CH, dim=1)
        out = out * mask_expanded
        return out

    def forward(self, x, sample_posterior=True):
        """
        返回：
          mu, logvar, z, x_hat_raw, x_hat_soft
        """
        mu, logvar = self.encoder(x)

        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        x_hat_raw = self.decoder(z)
        x_hat_soft = self.soft_decode_triplane(x_hat_raw)
        return mu, logvar, z, x_hat_raw, x_hat_soft

    @torch.no_grad()
    def encode_to_latent(self, x, deterministic=True):
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device).float()
        mu, logvar = self.encoder(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return mu, logvar, z

    @torch.no_grad()
    def decode_from_latent_raw(self, z):
        self.eval()
        if z.ndim == 3:
            z = z.unsqueeze(0)
        z = z.to(next(self.parameters()).device).float()
        return self.decoder(z)

    @torch.no_grad()
    def decode_from_latent(self, z, mode="soft", hard_threshold=0.1):
        """
        mode:
          - "raw"  : 不做任何處理
          - "soft" : occupancy sigmoid（推薦，train/inference 一致）
          - "hard" : soft + hard mask（只做診斷）
        """
        self.eval()
        if z.ndim == 3:
            z = z.unsqueeze(0)
        z = z.to(next(self.parameters()).device).float()

        x_hat_raw = self.decoder(z)

        if mode == "raw":
            return x_hat_raw
        elif mode == "soft":
            return self.soft_decode_triplane(x_hat_raw)
        elif mode == "hard":
            return self.hard_decode_triplane(x_hat_raw, threshold=hard_threshold)
        else:
            raise ValueError(f"Unknown decode mode: {mode}")


model = Triplane2DVAE().to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print("model params (M):", round(n_params, 3))

# sanity
x = torch.randn(2, TRIPLANE_CH, TRIPLANE_RES, TRIPLANE_RES).to(device)
mu, logvar, z, x_hat_raw, x_hat_soft = model(x)
print("mu        :", mu.shape)
print("logvar    :", logvar.shape)
print("z         :", z.shape)
print("x_hat_raw :", x_hat_raw.shape)
print("x_hat_soft:", x_hat_soft.shape)

# %%
# Cell 7 — Loss functions (FIXED: occupancy 用 raw logits，feature 用 soft decode)

def kl_loss_2d(mu, logvar):
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def charbonnier_loss(x_hat, x, eps=1e-3):
    diff = x_hat - x
    return torch.sqrt(diff * diff + eps * eps).mean()


def reconstruction_loss_triplane(x_hat_raw, x_hat_soft, x):
    """
    修正後：
    - Occupancy loss：仍然對 raw logits 做 BCEWithLogits（數值穩定）
    - Feature loss：改用 soft decode 後的輸出，確保 train 與 inference 同域
    - 暫時不做 hard mask
    """
    occ_idx = [29, 61, 93]

    # 1) Occupancy loss on raw logits
    occ_hat_logits = x_hat_raw[:, occ_idx, :, :]
    occ_gt = x[:, occ_idx, :, :]
    loss_occ = F.binary_cross_entropy_with_logits(occ_hat_logits, occ_gt)

    # 2) Feature loss on soft-decoded tensor
    #    為避免 occupancy channel 重複計算，只對非 occupancy 通道做 feature loss
    feat_mask = torch.ones((TRIPLANE_CH,), dtype=torch.bool, device=x.device)
    feat_mask[occ_idx] = False

    feat_hat = x_hat_soft[:, feat_mask, :, :]
    feat_gt = x[:, feat_mask, :, :]

    # 只在 GT 有物體位置計算 feature loss，但用 soft mask，避免太硬
    occ_gt_softmask = occ_gt.mean(dim=1, keepdim=True)  # (B,1,H,W), 範圍約 0~1
    feat_mask_expanded = occ_gt_softmask.repeat(1, feat_hat.shape[1], 1, 1)

    feat_hat_masked = feat_hat * feat_mask_expanded
    feat_gt_masked = feat_gt * feat_mask_expanded

    l1 = F.l1_loss(feat_hat_masked, feat_gt_masked)
    ch = charbonnier_loss(feat_hat_masked, feat_gt_masked)
    loss_feat = 0.7 * l1 + 0.3 * ch

    # 權重先維持你上一版的大方向，但已消除 mismatch
    total_recon = 5.0 * loss_occ + 1.0 * loss_feat

    return total_recon, {
        "loss_occ": float(loss_occ.detach().cpu()),
        "loss_feat": float(loss_feat.detach().cpu()),
        "loss_l1_feat": float(l1.detach().cpu()),
        "loss_charb_feat": float(ch.detach().cpu()),
    }


def vae_total_loss(x_hat_raw, x_hat_soft, x, mu, logvar, beta=BETA_KL):
    recon, recon_dict = reconstruction_loss_triplane(x_hat_raw, x_hat_soft, x)
    kl = kl_loss_2d(mu, logvar)
    loss = recon + float(beta) * kl

    out = {
        "loss_total": float(loss.detach().cpu()),
        "loss_recon": float(recon.detach().cpu()),
        "loss_kl": float(kl.detach().cpu()),
    }
    out.update(recon_dict)
    return loss, out


print("loss helpers ready (train/inference aligned)")

# %%
# Cell 8 — Train loop (FIXED: training uses x_hat_soft for feature domain, same as inference)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_loss = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    loss_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    occ_sum = 0.0
    feat_sum = 0.0
    step_count = 0

    t0 = time.time()

    for step, (x, frame_ids) in enumerate(dl, start=1):
        x = x.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        # 訓練仍然 sample posterior
        mu, logvar, z, x_hat_raw, x_hat_soft = model(x, sample_posterior=True)

        loss, info = vae_total_loss(
            x_hat_raw=x_hat_raw,
            x_hat_soft=x_hat_soft,
            x=x,
            mu=mu,
            logvar=logvar,
            beta=BETA_KL,
        )

        loss.backward()

        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        loss_sum += info["loss_total"]
        recon_sum += info["loss_recon"]
        kl_sum += info["loss_kl"]
        occ_sum += info["loss_occ"]
        feat_sum += info["loss_feat"]
        step_count += 1

        if (step % 20 == 0) or (step == len(dl)):
            print(
                f"epoch {epoch:03d} | step {step:04d}/{len(dl):04d} | "
                f"loss {info['loss_total']:.6f} | recon {info['loss_recon']:.6f} | "
                f"occ {info['loss_occ']:.6f} | feat {info['loss_feat']:.6f} | kl {info['loss_kl']:.6f}"
            )

    loss_mean = loss_sum / max(step_count, 1)
    recon_mean = recon_sum / max(step_count, 1)
    kl_mean = kl_sum / max(step_count, 1)
    occ_mean = occ_sum / max(step_count, 1)
    feat_mean = feat_sum / max(step_count, 1)
    dt = time.time() - t0

    history.append({
        "epoch": epoch,
        "loss": loss_mean,
        "recon": recon_mean,
        "kl": kl_mean,
        "occ": occ_mean,
        "feat": feat_mean,
        "time_sec": dt,
    })

    print(
        f"[epoch {epoch:03d}] "
        f"loss={loss_mean:.6f} recon={recon_mean:.6f} "
        f"occ={occ_mean:.6f} feat={feat_mean:.6f} kl={kl_mean:.6f} "
        f"time={dt:.2f}s"
    )

    # save last
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "config": {
                "triplane_ch": TRIPLANE_CH,
                "triplane_res": TRIPLANE_RES,
                "plane_ch": PLANE_CH,
                "latent_ch": LATENT_CH,
                "latent_res": LATENT_RES,
                "beta_kl": BETA_KL,
                "vox_cm": VOX_CM,
                "decode_mode_train_infer": "soft",
                "hard_mask_used_in_train": False,
            },
            "world_center": np.asarray(world_center, dtype=np.float32),
            "world_scale": np.float32(world_scale),
        },
        CKPT_PATH_LAST,
    )

    # save best
    if loss_mean < best_loss:
        best_loss = loss_mean
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_loss": best_loss,
                "config": {
                    "triplane_ch": TRIPLANE_CH,
                    "triplane_res": TRIPLANE_RES,
                    "plane_ch": PLANE_CH,
                    "latent_ch": LATENT_CH,
                    "latent_res": LATENT_RES,
                    "beta_kl": BETA_KL,
                    "vox_cm": VOX_CM,
                    "decode_mode_train_infer": "soft",
                    "hard_mask_used_in_train": False,
                },
                "world_center": np.asarray(world_center, dtype=np.float32),
                "world_scale": np.float32(world_scale),
            },
            CKPT_PATH_BEST,
        )
        print("saved BEST:", CKPT_PATH_BEST)

print("saved LAST:", CKPT_PATH_LAST)
print("training done")

# %%
# Cell 9 — Quick sanity check: encode/decode one batch and inspect shapes / numeric range
# 已對齊新版 model.forward()：返回 5 個值
#   mu, logvar, z, x_hat_raw, x_hat_soft

# load best ckpt if needed
if os.path.exists(CKPT_PATH_BEST):
    ckpt = torch.load(CKPT_PATH_BEST, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("loaded best ckpt from:", CKPT_PATH_BEST)
elif os.path.exists(CKPT_PATH_LAST):
    ckpt = torch.load(CKPT_PATH_LAST, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("loaded last ckpt from:", CKPT_PATH_LAST)

model.eval()

batch = next(iter(dl))
if len(batch) == 2:
    x, fids = batch
elif len(batch) == 3:
    x, fids, cidx = batch
else:
    raise RuntimeError(f"Unexpected batch format: len(batch)={len(batch)}")

x = x.to(device).float()

with torch.no_grad():
    # 新版 forward：5 outputs
    mu, logvar, z, x_hat_raw, x_hat_soft = model(x, sample_posterior=False)

    # 額外再看 hard decode（只做診斷）
    x_hat_hard = model.hard_decode_triplane(x_hat_raw, threshold=0.1)

occ_idx = [29, 61, 93]

print("input      :", x.shape)
print("mu         :", mu.shape)
print("logvar     :", logvar.shape)
print("latent     :", z.shape)
print("x_hat_raw  :", x_hat_raw.shape)
print("x_hat_soft :", x_hat_soft.shape)
print("x_hat_hard :", x_hat_hard.shape)

print("input range       :", float(x.min().cpu()), float(x.max().cpu()))
print("raw range         :", float(x_hat_raw.min().cpu()), float(x_hat_raw.max().cpu()))
print("soft range        :", float(x_hat_soft.min().cpu()), float(x_hat_soft.max().cpu()))
print("hard range        :", float(x_hat_hard.min().cpu()), float(x_hat_hard.max().cpu()))
print("latent mean       :", float(z.mean().cpu()))
print("latent std        :", float(z.std().cpu()))

print("raw occ range     :", float(x_hat_raw[:, occ_idx].min().cpu()), float(x_hat_raw[:, occ_idx].max().cpu()))
print("soft occ range    :", float(x_hat_soft[:, occ_idx].min().cpu()), float(x_hat_soft[:, occ_idx].max().cpu()))
print("hard occ range    :", float(x_hat_hard[:, occ_idx].min().cpu()), float(x_hat_hard[:, occ_idx].max().cpu()))

l1_raw_vs_gt = float(torch.mean(torch.abs(x_hat_raw - x)).cpu())
l1_soft_vs_gt = float(torch.mean(torch.abs(x_hat_soft - x)).cpu())
l1_hard_vs_gt = float(torch.mean(torch.abs(x_hat_hard - x)).cpu())

print("L1(raw , gt):", l1_raw_vs_gt)
print("L1(soft, gt):", l1_soft_vs_gt)
print("L1(hard, gt):", l1_hard_vs_gt)

# occupancy 命中率（方便看 soft decode 是否合理）
occ_gt = x[:, occ_idx, :, :]
occ_pred_soft = x_hat_soft[:, occ_idx, :, :]

occ_gt_ratio = float((occ_gt > 0.5).float().mean().cpu())
occ_pred_ratio_01 = float((occ_pred_soft > 0.1).float().mean().cpu())
occ_pred_ratio_05 = float((occ_pred_soft > 0.5).float().mean().cpu())

print("GT occ ratio (>0.5)     :", occ_gt_ratio)
print("Pred occ ratio (>0.1)   :", occ_pred_ratio_01)
print("Pred occ ratio (>0.5)   :", occ_pred_ratio_05)

# %%
# Cell 9.5 — Diagnostic: compare raw / soft / hard decode on the SAME latent
# 用這格確認 train/inference mismatch 是否已經被消掉

# load best ckpt if exists
if os.path.exists(CKPT_PATH_BEST):
    ckpt = torch.load(CKPT_PATH_BEST, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("loaded best ckpt:", CKPT_PATH_BEST)

model.eval()

x, fids = next(iter(dl))
x = x.to(device).float()

with torch.no_grad():
    # 用 deterministic MU，避免隨機噪聲影響判斷
    mu, logvar = model.encoder(x)
    z = mu

    x_hat_raw = model.decode_from_latent(z, mode="raw")
    x_hat_soft = model.decode_from_latent(z, mode="soft")
    x_hat_hard = model.decode_from_latent(z, mode="hard", hard_threshold=0.1)

occ_idx = [29, 61, 93]

# 檢查三種輸出差異
raw_vs_soft = float(torch.mean(torch.abs(x_hat_raw[:, occ_idx] - x_hat_soft[:, occ_idx])).cpu())
soft_vs_hard = float(torch.mean(torch.abs(x_hat_soft - x_hat_hard)).cpu())

print("x target    :", x.shape)
print("x_hat_raw   :", x_hat_raw.shape)
print("x_hat_soft  :", x_hat_soft.shape)
print("x_hat_hard  :", x_hat_hard.shape)

print("raw occupancy range :", float(x_hat_raw[:, occ_idx].min().cpu()), float(x_hat_raw[:, occ_idx].max().cpu()))
print("soft occupancy range:", float(x_hat_soft[:, occ_idx].min().cpu()), float(x_hat_soft[:, occ_idx].max().cpu()))
print("hard occupancy range:", float(x_hat_hard[:, occ_idx].min().cpu()), float(x_hat_hard[:, occ_idx].max().cpu()))

print("mean |raw_occ - soft_occ| :", raw_vs_soft)
print("mean |soft - hard|        :", soft_vs_hard)

# 最重要：訓練與推理現在都應該用 soft
l1_soft_vs_gt = float(torch.mean(torch.abs(x_hat_soft - x)).cpu())
l1_hard_vs_gt = float(torch.mean(torch.abs(x_hat_hard - x)).cpu())

print("L1(soft, gt):", l1_soft_vs_gt)
print("L1(hard, gt):", l1_hard_vs_gt)
print("If hard is much worse than soft, the hard mask is still damaging the result.")

# %%
# Cell 10 — Inference helpers for later AR stage (aligned with new soft/raw/hard decode API)

@torch.no_grad()
def encode_cache_npz_to_triplane_latent(npz_path, model, world_center, world_scale, deterministic=True):
    pack = load_cache_npz(npz_path)
    if pack is None:
        raise RuntimeError(f"Invalid cache npz: {npz_path}")

    tri = build_triplane_from_cache_pack(pack, world_center, world_scale, res=TRIPLANE_RES)
    x = torch.from_numpy(tri).unsqueeze(0).to(device).float()  # (1,96,128,128)

    mu, logvar, z = model.encode_to_latent(x, deterministic=deterministic)
    return {
        "triplane": tri.astype(np.float32),
        "mu": mu.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "logvar": logvar.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "latent": z.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "frame_id": int(pack.get("frame_id", -1)),
    }


@torch.no_grad()
def decode_latent_to_triplane(latent_2d, model, mode="soft", hard_threshold=0.1):
    """
    latent_2d: (8,16,16) or (1,8,16,16)
    mode:
      - "raw"
      - "soft"  (推薦，train/inference 一致)
      - "hard"  (只做診斷)
    """
    if isinstance(latent_2d, np.ndarray):
        z = torch.from_numpy(latent_2d)
    else:
        z = latent_2d

    if z.ndim == 3:
        z = z.unsqueeze(0)

    z = z.to(device).float()
    x_hat = model.decode_from_latent(z, mode=mode, hard_threshold=hard_threshold)
    return x_hat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# example
sample_candidates = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
if len(sample_candidates) == 0:
    raise RuntimeError(f"No npz files found in CACHE_DIR: {CACHE_DIR}")

sample_npz = sample_candidates[0]
ret = encode_cache_npz_to_triplane_latent(
    sample_npz,
    model,
    world_center,
    world_scale,
    deterministic=True,   # 預設用 MU
)

print("sample frame_id:", ret["frame_id"])
print("triplane shape :", ret["triplane"].shape)
print("mu shape       :", ret["mu"].shape)
print("latent shape   :", ret["latent"].shape)

tri_hat_soft = decode_latent_to_triplane(ret["mu"], model, mode="soft")
tri_hat_raw  = decode_latent_to_triplane(ret["mu"], model, mode="raw")
tri_hat_hard = decode_latent_to_triplane(ret["mu"], model, mode="hard", hard_threshold=0.1)

print("decoded triplane (raw)  :", tri_hat_raw.shape)
print("decoded triplane (soft) :", tri_hat_soft.shape)
print("decoded triplane (hard) :", tri_hat_hard.shape)

print("L1(raw , gt):", float(np.mean(np.abs(tri_hat_raw  - ret["triplane"]))))
print("L1(soft, gt):", float(np.mean(np.abs(tri_hat_soft - ret["triplane"]))))
print("L1(hard, gt):", float(np.mean(np.abs(tri_hat_hard - ret["triplane"]))))

# %%
# Cell 11 — Export occupancy-based 3D query point cloud PLY (REPLACE old proxy-xyz version)
# 目的：
#   不再把 triplane 的 feature channel 0:3 當作 xyz。
#   改用三張 plane 的 occupancy channel，在 3D query grid 上融合成 occupancy score，
#   再輸出更可信的 GT / Recon occupancy point cloud。
#
# 前提：
#   - model 已載入 checkpoint
#   - world_center, world_scale 已存在
#   - build_triplane_from_cache_pack / load_cache_npz 已存在
#   - 新版 model.decode_from_latent(..., mode="soft") 可用

import os
import glob
import numpy as np
import torch

PLY_CHECK_DIR = os.path.join(OUT_DIR, "ply_check_occ3d")
os.makedirs(PLY_CHECK_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
EXPORT_SAMPLE_IDX = 0          # 如果 triplane_all 在記憶體，就按 dataset index 取
USE_DATASET_CACHE = True       # True: 優先用 triplane_all；False: 直接從某個 npz 建 triplane
USE_MU_DECODE = True           # 建議 True，deterministic

# 3D query grid 設定
QUERY_GRID_RES = 48            # 48/56/64 自行取捨；越大越密，越慢
QUERY_RANGE_MIN = -1.0
QUERY_RANGE_MAX =  1.0

# occupancy 融合與門檻
# fuse_mode:
#   "mean"    -> (xy + xz + yz)/3
#   "product" -> xy * xz * yz
#   "min"     -> min(xy, xz, yz)
FUSE_MODE = "mean"
PLY_OCC_THRESHOLD = 0.22

# 可選：只保留最多幾多個點（避免 ply 太大）
MAX_EXPORT_POINTS = 120000

# 若要同時輸出真 GT raw points（方便對照）
EXPORT_TRUE_GT_RAW_POINTS = True


# -----------------------------
# Basic PLY writer
# -----------------------------
def write_ply_xyzrgb(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")

    n = xyz.shape[0]

    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape != (n, 3):
            raise ValueError(f"rgb must be (N,3), got {rgb.shape}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_uniform_rgb(n, color):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.array([color], dtype=np.uint8), (n, 1))


# -----------------------------
# Sampling helpers
# -----------------------------
def split_triplane_occ(tri_flat):
    """
    tri_flat: (TRIPLANE_CH, H, W)
    returns:
      occ_xy, occ_xz, occ_yz   each: (H, W)
    """
    tri = np.asarray(tri_flat, dtype=np.float32)
    if tri.ndim != 3 or tri.shape[0] != TRIPLANE_CH:
        raise ValueError(f"Expected triplane shape ({TRIPLANE_CH},H,W), got {tri.shape}")

    occ_xy = tri[29]
    occ_xz = tri[61]
    occ_yz = tri[93]
    return occ_xy.astype(np.float32), occ_xz.astype(np.float32), occ_yz.astype(np.float32)


def world_to_norm(xyz_world, world_center, world_scale):
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    wc = np.asarray(world_center, dtype=np.float32).reshape(1, 3)
    ws = float(world_scale)
    return ((xyz_world - wc) / max(ws, 1e-8)).astype(np.float32)


def norm_to_world(xyz_norm, world_center, world_scale):
    xyz_norm = np.asarray(xyz_norm, dtype=np.float32)
    wc = np.asarray(world_center, dtype=np.float32).reshape(1, 3)
    ws = float(world_scale)
    return (xyz_norm * ws + wc).astype(np.float32)


def coord_norm_to_pixel(v_norm, res):
    """
    [-1,1] -> [0, res-1]
    """
    u = (v_norm * 0.5) + 0.5
    u = np.clip(u, 0.0, 1.0)
    return u * float(res - 1)


def bilinear_sample_2d(feat_2d, px, py):
    """
    feat_2d: (H,W)
    px, py:  same shape float arrays in pixel coords
    return:  sampled values same shape
    """
    H, W = feat_2d.shape

    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)

    dx = px - x0.astype(np.float32)
    dy = py - y0.astype(np.float32)

    v00 = feat_2d[y0, x0]
    v01 = feat_2d[y1, x0]
    v10 = feat_2d[y0, x1]
    v11 = feat_2d[y1, x1]

    out = (
        v00 * (1.0 - dx) * (1.0 - dy) +
        v01 * (1.0 - dx) * dy +
        v10 * dx * (1.0 - dy) +
        v11 * dx * dy
    )
    return out.astype(np.float32)


def fuse_occ(o_xy, o_xz, o_yz, mode="mean"):
    if mode == "mean":
        return ((o_xy + o_xz + o_yz) / 3.0).astype(np.float32)
    elif mode == "product":
        return (o_xy * o_xz * o_yz).astype(np.float32)
    elif mode == "min":
        return np.minimum(np.minimum(o_xy, o_xz), o_yz).astype(np.float32)
    else:
        raise ValueError(f"Unknown FUSE_MODE: {mode}")


def triplane_occ_to_pointcloud(tri_flat, world_center, world_scale,
                               grid_res=48, occ_threshold=0.22,
                               fuse_mode="mean", max_points=120000):
    """
    用 3D query grid + 三平面 occupancy 融合，把 triplane 轉成 3D occupancy 點雲。
    """
    occ_xy, occ_xz, occ_yz = split_triplane_occ(tri_flat)
    H, W = occ_xy.shape

    # 生成 normalized query grid
    xs = np.linspace(QUERY_RANGE_MIN, QUERY_RANGE_MAX, grid_res, dtype=np.float32)
    ys = np.linspace(QUERY_RANGE_MIN, QUERY_RANGE_MAX, grid_res, dtype=np.float32)
    zs = np.linspace(QUERY_RANGE_MIN, QUERY_RANGE_MAX, grid_res, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")  # shape: (grid_res, grid_res, grid_res)

    # 三個 plane 對應取樣
    px_xy = coord_norm_to_pixel(X, W)
    py_xy = coord_norm_to_pixel(Y, H)

    px_xz = coord_norm_to_pixel(X, W)
    py_xz = coord_norm_to_pixel(Z, H)

    px_yz = coord_norm_to_pixel(Y, W)
    py_yz = coord_norm_to_pixel(Z, H)

    s_xy = bilinear_sample_2d(occ_xy, px_xy, py_xy)
    s_xz = bilinear_sample_2d(occ_xz, px_xz, py_xz)
    s_yz = bilinear_sample_2d(occ_yz, px_yz, py_yz)

    occ3d = fuse_occ(s_xy, s_xz, s_yz, mode=fuse_mode)

    mask = occ3d > float(occ_threshold)
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {
                "num_points": 0,
                "grid_res": int(grid_res),
                "occ_threshold": float(occ_threshold),
                "fuse_mode": fuse_mode,
                "score_min": float(occ3d.min()),
                "score_max": float(occ3d.max()),
                "score_mean": float(occ3d.mean()),
            }
        )

    pts_norm = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float32)
    scores = occ3d[mask].astype(np.float32)

    # 如果點太多，保留高分
    if pts_norm.shape[0] > int(max_points):
        order = np.argsort(scores)[::-1]
        keep = order[: int(max_points)]
        pts_norm = pts_norm[keep]
        scores = scores[keep]

    pts_world = norm_to_world(pts_norm, world_center, world_scale)

    info = {
        "num_points": int(pts_world.shape[0]),
        "grid_res": int(grid_res),
        "occ_threshold": float(occ_threshold),
        "fuse_mode": fuse_mode,
        "score_min": float(scores.min()) if scores.size > 0 else 0.0,
        "score_max": float(scores.max()) if scores.size > 0 else 0.0,
        "score_mean": float(scores.mean()) if scores.size > 0 else 0.0,
    }
    return pts_world.astype(np.float32), scores.astype(np.float32), info


def scores_to_rgb(scores, base_color):
    """
    用 score 調亮度，方便觀察 occupancy 強弱
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    s = scores.copy()
    s = s - s.min()
    s = s / max(float(s.max()), 1e-8)

    base = np.asarray(base_color, dtype=np.float32).reshape(1, 3)
    rgb = base * (0.35 + 0.65 * s[:, None])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# -----------------------------
# 1) Load model ckpt
# -----------------------------
if os.path.exists(CKPT_PATH_BEST):
    ckpt = torch.load(CKPT_PATH_BEST, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("loaded best ckpt:", CKPT_PATH_BEST)
elif os.path.exists(CKPT_PATH_LAST):
    ckpt = torch.load(CKPT_PATH_LAST, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("loaded last ckpt:", CKPT_PATH_LAST)
else:
    print("warning: no checkpoint found, using current in-memory model weights")

model.eval()


# -----------------------------
# 2) Get sample (GT triplane)
# -----------------------------
sample_idx = int(EXPORT_SAMPLE_IDX)
frame_id = -1
sample_npz = None

if USE_DATASET_CACHE and ("triplane_all" in globals()) and ("frame_id_all" in globals()):
    sample_idx = int(np.clip(sample_idx, 0, len(triplane_all) - 1))
    tri_gt = np.asarray(triplane_all[sample_idx], dtype=np.float32)
    frame_id = int(frame_id_all[sample_idx])

    # 如果有 cache_idx_all，可映射回原始 npz
    if ("cache_idx_all" in globals()) and (len(cache_idx_all) > sample_idx):
        sample_cache_idx = int(cache_idx_all[sample_idx])
        candidate_npz = os.path.join(CACHE_DIR, f"{sample_cache_idx:06d}.npz")
        if os.path.exists(candidate_npz):
            sample_npz = candidate_npz
else:
    # fallback: 從 cache 直接讀第一個樣本
    sample_candidates = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if len(sample_candidates) == 0:
        raise RuntimeError(f"No npz files found in CACHE_DIR: {CACHE_DIR}")

    sample_idx = int(np.clip(sample_idx, 0, len(sample_candidates) - 1))
    sample_npz = sample_candidates[sample_idx]

    pack = load_cache_npz(sample_npz)
    if pack is None:
        raise RuntimeError(f"Invalid cache npz: {sample_npz}")

    tri_gt = build_triplane_from_cache_pack(pack, world_center, world_scale, res=TRIPLANE_RES)
    frame_id = int(pack.get("frame_id", -1))

print("sample_idx:", sample_idx)
print("frame_id  :", frame_id)
print("sample_npz:", sample_npz)
print("tri_gt shape:", tri_gt.shape)


# -----------------------------
# 3) VAE encode -> decode
# -----------------------------
x = torch.from_numpy(tri_gt).unsqueeze(0).to(device).float()

with torch.no_grad():
    mu, logvar = model.encoder(x)

    if USE_MU_DECODE:
        z_use = mu
        print("decode mode: MU (deterministic)")
    else:
        z_use = model.reparameterize(mu, logvar)
        print("decode mode: sampled Z (stochastic)")

    # 關鍵：用 soft，與目前 train/inference 主路徑一致
    x_hat = model.decode_from_latent(z_use, mode="soft")

tri_recon = x_hat.squeeze(0).detach().cpu().numpy().astype(np.float32)

# tensor-level metric
abs_diff = np.abs(tri_recon - tri_gt)
l1_mean = float(abs_diff.mean())
rmse = float(np.sqrt(np.mean((tri_recon - tri_gt) ** 2)))
print("triplane L1 mean:", l1_mean)
print("triplane RMSE   :", rmse)


# -----------------------------
# 4) Convert GT/Reconstructed triplane to occupancy 3D point clouds
# -----------------------------
gt_xyz, gt_scores, gt_info = triplane_occ_to_pointcloud(
    tri_gt,
    world_center=world_center,
    world_scale=world_scale,
    grid_res=QUERY_GRID_RES,
    occ_threshold=PLY_OCC_THRESHOLD,
    fuse_mode=FUSE_MODE,
    max_points=MAX_EXPORT_POINTS,
)

rc_xyz, rc_scores, rc_info = triplane_occ_to_pointcloud(
    tri_recon,
    world_center=world_center,
    world_scale=world_scale,
    grid_res=QUERY_GRID_RES,
    occ_threshold=PLY_OCC_THRESHOLD,
    fuse_mode=FUSE_MODE,
    max_points=MAX_EXPORT_POINTS,
)

print("GT occ pointcloud   :", gt_info)
print("Recon occ pointcloud:", rc_info)

gt_rgb = scores_to_rgb(gt_scores, [ 50, 220,  80])   # green
rc_rgb = scores_to_rgb(rc_scores, [220,  60,  60])   # red

gt_ply = os.path.join(PLY_CHECK_DIR, f"sample_{sample_idx:04d}_fid_{frame_id:06d}_gt_occ3d.ply")
rc_ply = os.path.join(PLY_CHECK_DIR, f"sample_{sample_idx:04d}_fid_{frame_id:06d}_recon_occ3d.ply")
mix_ply = os.path.join(PLY_CHECK_DIR, f"sample_{sample_idx:04d}_fid_{frame_id:06d}_compare_occ3d_gt_vs_recon.ply")

write_ply_xyzrgb(gt_ply, gt_xyz, gt_rgb)
write_ply_xyzrgb(rc_ply, rc_xyz, rc_rgb)

mix_xyz = np.concatenate([gt_xyz, rc_xyz], axis=0) if (len(gt_xyz) + len(rc_xyz)) > 0 else np.zeros((0, 3), dtype=np.float32)
mix_rgb = np.concatenate([
    make_uniform_rgb(len(gt_xyz), [ 40, 220,  80]),   # GT green
    make_uniform_rgb(len(rc_xyz), [220,  60,  60]),   # Recon red
], axis=0) if mix_xyz.shape[0] > 0 else np.zeros((0, 3), dtype=np.uint8)

write_ply_xyzrgb(mix_ply, mix_xyz, mix_rgb)

print("saved GT occ3d   :", gt_ply)
print("saved Recon occ3d:", rc_ply)
print("saved Compare    :", mix_ply)


# -----------------------------
# 5) Optional: export TRUE GT raw points from original cache for reference
# -----------------------------
if EXPORT_TRUE_GT_RAW_POINTS and (sample_npz is not None) and os.path.exists(sample_npz):
    pack_true = load_cache_npz(sample_npz)
    if pack_true is not None:
        patches, raw_pts = reconstruct_patch_points(
            pack_true["centers_cm"],
            pack_true["local_pts"],
            pack_true["local_count"],
            vox_cm=VOX_CM,
            local_pts_clip=LOCAL_PTS_CLIP,
        )

        if raw_pts.shape[0] > 0:
            gt_true_ply = os.path.join(PLY_CHECK_DIR, f"sample_{sample_idx:04d}_fid_{frame_id:06d}_true_gt_rawpts.ply")
            true_rgb = make_uniform_rgb(len(raw_pts), [120, 160, 255])  # blue-ish
            write_ply_xyzrgb(gt_true_ply, raw_pts.astype(np.float32), true_rgb)
            print("saved TRUE GT raw points:", gt_true_ply)
        else:
            print("TRUE GT raw points export skipped: raw_pts empty")
    else:
        print("TRUE GT raw points export skipped: invalid cache pack")
else:
    print("TRUE GT raw points export skipped")

# %%
# Cell X — Export TRUE GT point cloud from original cache (not triplane proxy)

GT_PLY_DIR = os.path.join(OUT_DIR, "ply_check_gt_true")
os.makedirs(GT_PLY_DIR, exist_ok=True)

EXPORT_SAMPLE_IDX = 0
EXPORT_USE_RAW_POINTS = True   # True: 輸出所有 raw points；False: 只輸出 patch centroids

def write_ply_xyzrgb(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")

    n = xyz.shape[0]
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape != (n, 3):
            raise ValueError(f"rgb must be (N,3), got {rgb.shape}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def uniform_rgb(n, color):
    if n <= 0:
        return np.zeros((0,3), dtype=np.uint8)
    return np.tile(np.array(color, dtype=np.uint8)[None, :], (n, 1))

# 由 selected cache 決定 sample
sample_idx = int(np.clip(EXPORT_SAMPLE_IDX, 0, len(selected_cache_indices) - 1))
cache_idx = int(selected_cache_indices[sample_idx])
npz_path = cache_idx_to_npz_path(cache_idx, CACHE_DIR)

pack = load_cache_npz(npz_path)
if pack is None:
    raise RuntimeError(f"Invalid cache npz: {npz_path}")

patches, raw_pts = reconstruct_patch_points(
    pack["centers_cm"], pack["local_pts"], pack["local_count"],
    vox_cm=VOX_CM, local_pts_clip=LOCAL_PTS_CLIP
)

# patch centroids
centroids = []
for arr in patches:
    if arr.shape[0] > 0:
        centroids.append(arr.mean(axis=0))
centroids = np.asarray(centroids, dtype=np.float32) if len(centroids) > 0 else np.zeros((0,3), dtype=np.float32)

frame_id = int(pack.get("frame_id", -1))

if EXPORT_USE_RAW_POINTS:
    xyz = raw_pts
    rgb = uniform_rgb(len(xyz), [60, 220, 80])
    out_ply = os.path.join(GT_PLY_DIR, f"sample_{sample_idx:04d}_cache_{cache_idx:06d}_fid_{frame_id:06d}_gt_raw_points.ply")
else:
    xyz = centroids
    rgb = uniform_rgb(len(xyz), [80, 160, 255])
    out_ply = os.path.join(GT_PLY_DIR, f"sample_{sample_idx:04d}_cache_{cache_idx:06d}_fid_{frame_id:06d}_gt_patch_centroids.ply")

write_ply_xyzrgb(out_ply, xyz, rgb)

print("cache_idx :", cache_idx)
print("frame_id  :", frame_id)
print("raw_pts   :", raw_pts.shape)
print("centroids :", centroids.shape)
print("saved GT  :", out_ply)

# %%




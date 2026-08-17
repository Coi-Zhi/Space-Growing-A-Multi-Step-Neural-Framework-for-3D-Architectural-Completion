# %%
# %%
# Cell 1 — Config & Imports

import os
import math
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ===== paths =====
ROOT_DIR   = r"./LearnCache_gs"
CACHE_DIR  = os.path.join(ROOT_DIR, "gaussian_frame_cache")
OUT_DIR    = r"./GAUSSIAN_GROUP_VAE"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== source cache params =====
VOX_CM = 20.0
LOCAL_PTS_CLIP = 1.25

# ===== patch gaussian fit params =====
MIN_VALID_PATCH_PTS = 6
COV_REG_CM2 = 4.0
SCALE_STD_FACTOR = 1.5
MIN_SCALE_CM = 3.0
MAX_SCALE_CM = 80.0
MIN_OPACITY = 0.05
MAX_OPACITY = 0.98

# ===== grouping =====
GROUP_SIZE = 32        # one local Gaussian group = one VAE sample
MIN_GROUP_VALID = 4    # skip groups that are too empty

# ===== model =====
GAUSS_DIM = 11         # [x,y,z, log_sx,log_sy,log_sz, qw,qx,qy,qz, opacity]
D_MODEL = 256
LATENT_DIM = 128
KL_WEIGHT = 1e-4

# ===== training =====
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-4
WD = 1e-4
NUM_WORKERS = 0
LOG_EVERY = 50

# ===== files =====
WORLD_STATS_NPZ = os.path.join(OUT_DIR, "world_stats.npz")
INDEX_NPZ       = os.path.join(OUT_DIR, "group_index.npz")
CKPT_PATH       = os.path.join(OUT_DIR, "gaussian_group_vae.pt")
LATENT_OUT_NPZ  = os.path.join(OUT_DIR, "group_latents.npz")

# %%
# %%
# Cell 2 — Geometry helpers (reconstruct patch points -> fit patch Gaussian -> group features)

def load_cache_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    if "empty" in z.files:
        return None
    required = ["centers_cm", "local_pts", "local_count"]
    for k in required:
        if k not in z.files:
            return None
    return {
        "centers_cm": z["centers_cm"].astype(np.float32),
        "local_pts": z["local_pts"].astype(np.float32),
        "local_count": z["local_count"].astype(np.int32),
        "frame_id": int(z["frame_id"]) if "frame_id" in z.files else -1,
    }

def quat_from_rotmat(R):
    R = R.astype(np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-8)
    return q

def ensure_right_handed(R):
    R = R.copy().astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R

def reconstruct_patch_points(centers_cm, local_pts, local_count, vox_cm=VOX_CM, local_pts_clip=LOCAL_PTS_CLIP):
    """
    returns list of (K,3) arrays, one per patch
    """
    half = float(vox_cm) / 2.0
    patches = []

    V = centers_cm.shape[0]
    for i in range(V):
        cen = centers_cm[i]
        n = int(local_count[i]) if i < len(local_count) else local_pts.shape[1]
        n = max(0, min(n, local_pts.shape[1]))

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
        patches.append(pts.astype(np.float32))

    return patches

def fit_patch_gaussian(arr_cm):
    """
    arr_cm: (K,3)
    returns:
      mu(3), scales(3), quat(4), opacity(1)
    """
    K = arr_cm.shape[0]
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

    opacity = np.float32(np.clip(K / max(1.0, MIN_VALID_PATCH_PTS * 2.0), MIN_OPACITY, MAX_OPACITY))
    return mu, scales, quat, opacity

def build_full_patch_gaussians(pack):
    """
    convert one frame cache -> full list of fitted patch gaussians
    returns:
      mu_all (M,3), sc_all (M,3), q_all (M,4), op_all (M,), count_all (M,)
    """
    patches = reconstruct_patch_points(pack["centers_cm"], pack["local_pts"], pack["local_count"])

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

def normalize_xyz_world(xyz_cm, world_center, world_scale):
    return ((xyz_cm - world_center[None, :]) / world_scale).astype(np.float32)

def scale_cm_to_log_norm(sc_cm, world_scale):
    sc_n = np.clip(sc_cm / world_scale, 1e-4, 10.0)
    return np.log(sc_n).astype(np.float32)

def build_group_features(mu_all, sc_all, q_all, op_all, world_center, world_scale, group_size=GROUP_SIZE):
    """
    Full patch gaussians -> multiple fixed-size groups.
    Deterministic spatial sort, then chunk into groups.
    returns list of (feat, mask)
    """
    M = mu_all.shape[0]
    if M == 0:
        return []

    xyz_n = normalize_xyz_world(mu_all, world_center, world_scale)
    log_sc = scale_cm_to_log_norm(sc_all, world_scale)
    feat_all = np.concatenate([xyz_n, log_sc, q_all, op_all[:, None]], axis=1).astype(np.float32)  # (M,11)

    # deterministic spatial ordering
    order = np.lexsort((mu_all[:, 2], mu_all[:, 1], mu_all[:, 0]))
    feat_all = feat_all[order]

    groups = []
    num_groups = int(math.ceil(M / float(group_size)))

    for gi in range(num_groups):
        s = gi * group_size
        e = min((gi + 1) * group_size, M)
        sub = feat_all[s:e]
        valid = sub.shape[0]

        if valid < MIN_GROUP_VALID:
            continue

        feat = np.zeros((group_size, GAUSS_DIM), dtype=np.float32)
        mask = np.zeros((group_size,), dtype=np.float32)
        feat[:valid] = sub
        mask[:valid] = 1.0

        # pad quat as identity for empty slots
        if valid < group_size:
            feat[valid:, 6] = 1.0

        groups.append((feat, mask))

    return groups

# %%
# %%
# Cell 2B — Position helpers + build_group_features with packed pos_id

import numpy as np
import math

# pack world xyz into 10-bit x / 10-bit y / 10-bit z
# so later Transformer can keep using encode_pos_chunks() unchanged
POS_PACK_VOX_CM = float(VOX_CM)   # usually same as your source voxel size
POS_GRID_BIAS   = 512             # center world_center near middle of [0..1023]
POS_GRID_MIN    = 0
POS_GRID_MAX    = 1023


def world_xyz_to_grid_index(xyz_cm, world_center, vox_cm=POS_PACK_VOX_CM, grid_bias=POS_GRID_BIAS):
    """
    xyz_cm: (..., 3) float32
    world_center: (3,)
    return:
        idx: (..., 3) int32 in [0, 1023]
    """
    xyz_cm = np.asarray(xyz_cm, dtype=np.float32)
    world_center = np.asarray(world_center, dtype=np.float32)

    idx = np.rint((xyz_cm - world_center[None, :]) / float(vox_cm)).astype(np.int32)
    idx = idx + int(grid_bias)
    idx = np.clip(idx, POS_GRID_MIN, POS_GRID_MAX)
    return idx.astype(np.int32)


def pack_xyz_index_to_pos_id(ixyz):
    """
    ixyz: (..., 3) int32, each in [0, 1023]
    return:
        pos_id: (...) int64
    """
    ixyz = np.asarray(ixyz, dtype=np.int64)
    ix = ixyz[..., 0]
    iy = ixyz[..., 1]
    iz = ixyz[..., 2]
    return (ix | (iy << 10) | (iz << 20)).astype(np.int64)


def pack_world_xyz_to_pos_id(xyz_cm, world_center, vox_cm=POS_PACK_VOX_CM, grid_bias=POS_GRID_BIAS):
    idx = world_xyz_to_grid_index(xyz_cm, world_center, vox_cm=vox_cm, grid_bias=grid_bias)
    return pack_xyz_index_to_pos_id(idx)


def build_group_features(mu_all, sc_all, q_all, op_all, world_center, world_scale, group_size=GROUP_SIZE):
    """
    Full patch gaussians -> multiple fixed-size groups.
    Deterministic spatial sort, then chunk into groups.

    returns list of:
        (feat, mask, pos_id, group_center_cm)
    """
    M = mu_all.shape[0]
    if M == 0:
        return []

    xyz_n = normalize_xyz_world(mu_all, world_center, world_scale)
    log_sc = scale_cm_to_log_norm(sc_all, world_scale)
    feat_all = np.concatenate([xyz_n, log_sc, q_all, op_all[:, None]], axis=1).astype(np.float32)  # (M,11)

    # deterministic spatial ordering
    order = np.lexsort((mu_all[:, 2], mu_all[:, 1], mu_all[:, 0]))
    feat_all = feat_all[order]
    mu_sorted = mu_all[order].astype(np.float32)

    groups = []
    num_groups = int(math.ceil(M / float(group_size)))

    for gi in range(num_groups):
        s = gi * group_size
        e = min((gi + 1) * group_size, M)

        sub = feat_all[s:e]
        sub_mu = mu_sorted[s:e]
        valid = sub.shape[0]

        if valid < MIN_GROUP_VALID:
            continue

        feat = np.zeros((group_size, GAUSS_DIM), dtype=np.float32)
        mask = np.zeros((group_size,), dtype=np.float32)
        feat[:valid] = sub
        mask[:valid] = 1.0

        # pad quat as identity for empty slots
        if valid < group_size:
            feat[valid:, 6] = 1.0

        group_center_cm = sub_mu.mean(axis=0).astype(np.float32)
        pos_id = int(pack_world_xyz_to_pos_id(group_center_cm[None, :], world_center)[0])

        groups.append((feat, mask, np.int64(pos_id), group_center_cm))

    return groups

# %%
# %%
# Cell 3 — Scan cache, estimate world stats, and precompute ALL group tensors (with pos_id)

import os
import json
import time
import glob
import numpy as np

GROUP_TENSOR_CACHE_NPZ = os.path.join(OUT_DIR, "group_tensor_cache.npz")
GROUP_TENSOR_META_JSON = os.path.join(OUT_DIR, "group_tensor_meta.json")
GROUP_TENSOR_PROGRESS_NPZ = os.path.join(OUT_DIR, "group_tensor_progress.npz")

SAVE_PROGRESS_EVERY = 25   # save progress every N frames

cache_paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
print("cache files:", len(cache_paths))

valid_cache_paths = []
all_centers = []

for p in cache_paths:
    pack = load_cache_npz(p)
    if pack is None:
        continue
    valid_cache_paths.append(p)
    all_centers.append(pack["centers_cm"])

print("valid cache files:", len(valid_cache_paths))

if len(valid_cache_paths) == 0:
    raise RuntimeError("No valid cache files found.")

# ---------------------------------------------------
# world stats
# ---------------------------------------------------
if os.path.exists(WORLD_STATS_NPZ):
    zz = np.load(WORLD_STATS_NPZ)
    world_center = zz["world_center"].astype(np.float32)
    world_scale = float(zz["world_scale"])
    print("loaded world stats")
else:
    cat_centers = np.concatenate(all_centers, axis=0).astype(np.float32)
    world_min = cat_centers.min(axis=0)
    world_max = cat_centers.max(axis=0)
    world_center = 0.5 * (world_min + world_max)
    half_extent = 0.5 * (world_max - world_min)
    world_scale = float(max(half_extent.max(), 1.0))

    np.savez_compressed(
        WORLD_STATS_NPZ,
        world_center=world_center.astype(np.float32),
        world_scale=np.float32(world_scale),
    )
    print("saved world stats")

print("world_center:", world_center)
print("world_scale:", world_scale)

# ---------------------------------------------------
# meta signature for cache consistency
# ---------------------------------------------------
current_meta = {
    "CACHE_FORMAT_VERSION": 2,          # bump format so old cache won't be reused
    "GROUP_SIZE": int(GROUP_SIZE),
    "MIN_GROUP_VALID": int(MIN_GROUP_VALID),
    "GAUSS_DIM": int(GAUSS_DIM),
    "VOX_CM": float(VOX_CM),
    "LOCAL_PTS_CLIP": float(LOCAL_PTS_CLIP),
    "MIN_VALID_PATCH_PTS": int(MIN_VALID_PATCH_PTS),
    "COV_REG_CM2": float(COV_REG_CM2),
    "SCALE_STD_FACTOR": float(SCALE_STD_FACTOR),
    "MIN_SCALE_CM": float(MIN_SCALE_CM),
    "MAX_SCALE_CM": float(MAX_SCALE_CM),
    "MIN_OPACITY": float(MIN_OPACITY),
    "MAX_OPACITY": float(MAX_OPACITY),
    "POS_PACK_VOX_CM": float(POS_PACK_VOX_CM),
    "POS_GRID_BIAS": int(POS_GRID_BIAS),
    "num_valid_cache_files": int(len(valid_cache_paths)),
}

def meta_matches(meta_a, meta_b):
    keys = sorted(set(meta_a.keys()) | set(meta_b.keys()))
    for k in keys:
        if meta_a.get(k, None) != meta_b.get(k, None):
            return False
    return True

# ---------------------------------------------------
# direct-load check
# ---------------------------------------------------
can_direct_load = False
if os.path.exists(GROUP_TENSOR_CACHE_NPZ) and os.path.exists(GROUP_TENSOR_META_JSON):
    try:
        with open(GROUP_TENSOR_META_JSON, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)

        if meta_matches(saved_meta, current_meta):
            zz = np.load(GROUP_TENSOR_CACHE_NPZ, allow_pickle=False)
            required = [
                "group_feat_all",
                "group_mask_all",
                "group_frame_id_all",
                "group_local_id_all",
                "group_pos_id_all",
                "group_center_all",
            ]
            if all(k in zz.files for k in required):
                can_direct_load = True
            else:
                print("existing tensor cache missing new pos arrays -> will rebuild / resume")
        else:
            print("existing tensor cache meta mismatch -> will rebuild / resume")
    except Exception as e:
        print("failed to read tensor cache meta, will rebuild:", e)

if can_direct_load:
    zz = np.load(GROUP_TENSOR_CACHE_NPZ, allow_pickle=False)
    group_feat_all     = zz["group_feat_all"].astype(np.float32)
    group_mask_all     = zz["group_mask_all"].astype(np.float32)
    group_frame_id_all = zz["group_frame_id_all"].astype(np.int32)
    group_local_id_all = zz["group_local_id_all"].astype(np.int32)
    group_pos_id_all   = zz["group_pos_id_all"].astype(np.int64)
    group_center_all   = zz["group_center_all"].astype(np.float32)
    print("loaded finished group tensor cache:", group_feat_all.shape[0])

else:
    # ---------------------------------------------------
    # resume from progress if possible
    # ---------------------------------------------------
    resume_from = 0
    feat_list = []
    mask_list = []
    frame_id_list = []
    local_id_list = []
    pos_id_list = []
    center_list = []

    can_resume = False
    if os.path.exists(GROUP_TENSOR_PROGRESS_NPZ) and os.path.exists(GROUP_TENSOR_META_JSON):
        try:
            with open(GROUP_TENSOR_META_JSON, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)

            if meta_matches(saved_meta, current_meta):
                zp = np.load(GROUP_TENSOR_PROGRESS_NPZ, allow_pickle=False)
                required = [
                    "group_feat_all",
                    "group_mask_all",
                    "group_frame_id_all",
                    "group_local_id_all",
                    "group_pos_id_all",
                    "group_center_all",
                    "resume_from",
                ]
                if all(k in zp.files for k in required):
                    feat_list = list(zp["group_feat_all"])
                    mask_list = list(zp["group_mask_all"])
                    frame_id_list = list(zp["group_frame_id_all"].astype(np.int32))
                    local_id_list = list(zp["group_local_id_all"].astype(np.int32))
                    pos_id_list = list(zp["group_pos_id_all"].astype(np.int64))
                    center_list = list(zp["group_center_all"].astype(np.float32))
                    resume_from = int(zp["resume_from"])
                    can_resume = True
                    print(f"resuming tensor cache build from frame {resume_from}/{len(valid_cache_paths)}")
                else:
                    print("progress file missing new pos arrays -> starting fresh")
            else:
                print("progress meta mismatch -> starting fresh")
        except Exception as e:
            print("failed to load tensor progress, starting fresh:", e)

    if not can_resume:
        feat_list = []
        mask_list = []
        frame_id_list = []
        local_id_list = []
        pos_id_list = []
        center_list = []
        resume_from = 0

        with open(GROUP_TENSOR_META_JSON, "w", encoding="utf-8") as f:
            json.dump(current_meta, f, ensure_ascii=False, indent=2)

        print("starting fresh tensor cache build")

    # ---------------------------------------------------
    # build / resume all group tensors
    # ---------------------------------------------------
    t0 = time.time()

    for i in range(resume_from, len(valid_cache_paths)):
        p = valid_cache_paths[i]
        pack = load_cache_npz(p)
        if pack is None:
            continue

        mu_all, sc_all, q_all, op_all, count_all = build_full_patch_gaussians(pack)
        groups = build_group_features(
            mu_all, sc_all, q_all, op_all,
            world_center, world_scale,
            group_size=GROUP_SIZE
        )

        local_counter = 0
        for item in groups:
            feat, mask, pos_id, group_center_cm = item

            feat_list.append(feat.astype(np.float32))
            mask_list.append(mask.astype(np.float32))
            frame_id_list.append(int(pack["frame_id"]))
            local_id_list.append(int(local_counter))
            pos_id_list.append(np.int64(pos_id))
            center_list.append(group_center_cm.astype(np.float32))
            local_counter += 1

        # periodic progress save
        if ((i + 1) % SAVE_PROGRESS_EVERY == 0) or (i == len(valid_cache_paths) - 1):
            np.savez_compressed(
                GROUP_TENSOR_PROGRESS_NPZ,
                group_feat_all=np.asarray(feat_list, dtype=np.float32),
                group_mask_all=np.asarray(mask_list, dtype=np.float32),
                group_frame_id_all=np.asarray(frame_id_list, dtype=np.int32),
                group_local_id_all=np.asarray(local_id_list, dtype=np.int32),
                group_pos_id_all=np.asarray(pos_id_list, dtype=np.int64),
                group_center_all=np.asarray(center_list, dtype=np.float32),
                resume_from=np.int32(i + 1),
            )
            print(f"progress saved: {i+1}/{len(valid_cache_paths)}")

    # ---------------------------------------------------
    # finalize tensor cache
    # ---------------------------------------------------
    group_feat_all     = np.asarray(feat_list, dtype=np.float32)
    group_mask_all     = np.asarray(mask_list, dtype=np.float32)
    group_frame_id_all = np.asarray(frame_id_list, dtype=np.int32)
    group_local_id_all = np.asarray(local_id_list, dtype=np.int32)
    group_pos_id_all   = np.asarray(pos_id_list, dtype=np.int64)
    group_center_all   = np.asarray(center_list, dtype=np.float32)

    np.savez_compressed(
        GROUP_TENSOR_CACHE_NPZ,
        group_feat_all=group_feat_all,
        group_mask_all=group_mask_all,
        group_frame_id_all=group_frame_id_all,
        group_local_id_all=group_local_id_all,
        group_pos_id_all=group_pos_id_all,
        group_center_all=group_center_all,
    )

    # cleanup progress
    if os.path.exists(GROUP_TENSOR_PROGRESS_NPZ):
        os.remove(GROUP_TENSOR_PROGRESS_NPZ)

    with open(GROUP_TENSOR_META_JSON, "w", encoding="utf-8") as f:
        json.dump(current_meta, f, ensure_ascii=False, indent=2)

    print("saved final tensor cache:", group_feat_all.shape[0], "samples", "time:", time.time() - t0)

print("group_feat_all    :", group_feat_all.shape)
print("group_mask_all    :", group_mask_all.shape)
print("group_frame_id_all:", group_frame_id_all.shape)
print("group_local_id_all:", group_local_id_all.shape)
print("group_pos_id_all  :", group_pos_id_all.shape)
print("group_center_all  :", group_center_all.shape)

# for compatibility with later cells
sample_frame_ids = group_frame_id_all
sample_group_ids = group_local_id_all
sample_pos_ids = group_pos_id_all
sample_group_centers_cm = group_center_all

# %%
# %%
# Cell 4 — Dataset (FAST: reads precomputed group tensors directly)

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class PrecomputedGaussianGroupDataset(Dataset):
    def __init__(self, group_feat_all, group_mask_all):
        self.group_feat_all = group_feat_all.astype(np.float32)
        self.group_mask_all = group_mask_all.astype(np.float32)

    def __len__(self):
        return self.group_feat_all.shape[0]

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.group_feat_all[idx])   # (GROUP_SIZE, 11)
        mask = torch.from_numpy(self.group_mask_all[idx])   # (GROUP_SIZE,)
        return feat, mask

ds = PrecomputedGaussianGroupDataset(group_feat_all, group_mask_all)

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

# %%
# %%
# Cell 5 — Gaussian Group Continuous Latent AE (slot-wise + self-pooled context, no codebook)

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# optional regularization weight (keep 0 first for pure AE sanity check)
LATENT_REG_WEIGHT = 0.0
EMPTY_EPS = 1e-8


def infer_empty_mask_from_feat(feat, mask=None):
    """
    Returns:
        empty_mask: (B, N) bool
    Priority:
        1) use provided mask
        2) otherwise infer from all-zero padded rows
    """
    if mask is not None:
        return (mask <= 0.5)
    return (feat.abs().sum(dim=-1) <= EMPTY_EPS)


class SlotContextEncoder(nn.Module):
    """
    Per-slot encoder, no global flatten, no Transformer, no VQ.
    Each slot gets a latent; latent already contains some group context.
    """
    def __init__(self, in_dim=GAUSS_DIM, d_model=D_MODEL, latent_dim=LATENT_DIM):
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
        """
        feat: (B, N, 11)
        mask: (B, N) float in {0,1}
        returns:
            z: (B, N, LATENT_DIM)
        """
        x = torch.cat([feat, mask.unsqueeze(-1)], dim=-1)   # (B,N,12)
        h = self.slot(x)                                    # (B,N,d_model)

        valid = mask.unsqueeze(-1)                          # (B,N,1)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        ctx = (h * valid).sum(dim=1, keepdim=True) / denom  # (B,1,d_model)
        ctx = ctx.expand_as(h)                              # (B,N,d_model)

        z = self.to_latent(torch.cat([h, ctx], dim=-1))     # (B,N,latent_dim)

        # hard-zero empty slots so decode can later work without extra mask input
        z = z * valid
        return z


class SlotContextDecoder(nn.Module):
    """
    Decode from continuous latent only.
    Group context is re-pooled from latent itself, so no extra input is needed.
    """
    def __init__(self, latent_dim=LATENT_DIM, d_model=D_MODEL):
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

        # stable init
        nn.init.constant_(self.out_sc.bias, -4.0)
        nn.init.constant_(self.out_q.bias[0], 1.0)
        nn.init.constant_(self.out_q.bias[1], 0.0)
        nn.init.constant_(self.out_q.bias[2], 0.0)
        nn.init.constant_(self.out_q.bias[3], 0.0)

    def forward(self, z):
        """
        z: (B, N, LATENT_DIM)
        """
        B, N, D = z.shape

        # re-pool context from latent itself (independent decode path)
        # because empty slots were zeroed in encoder, simple mean is workable
        z_ctx = z.mean(dim=1, keepdim=True).expand(B, N, D)   # (B,N,D)

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
    """
    Continuous latent AE with VQAE-compatible forward signature.
    forward(...) returns:
        z_e, z_q_st, token_ids, vq_loss, feat_hat, mask_logit
    Here:
        z_e      = latent
        z_q_st   = latent
        token_ids= latent (kept only for compatibility; not discrete!)
        vq_loss  = 0
    """
    def __init__(self):
        super().__init__()
        self.encoder = SlotContextEncoder()
        self.decoder = SlotContextDecoder()

    def forward(self, feat, mask=None):
        empty_mask = infer_empty_mask_from_feat(feat, mask)
        valid_mask = (~empty_mask).float()

        z = self.encoder(feat, valid_mask)
        feat_hat, mask_logit = self.decoder(z)

        # no VQ now; keep same interface so Cell 6 can still run
        zero_reg = torch.zeros((), device=feat.device, dtype=feat.dtype)

        # keep outputs in old shape contract
        z_e = z
        z_q_st = z
        token_ids = z   # compatibility placeholder only; no longer discrete tokens

        return z_e, z_q_st, token_ids, zero_reg, feat_hat, mask_logit

    @torch.no_grad()
    def encode_to_latent(self, feat, mask=None):
        self.eval()

        if feat.ndim == 2:
            feat = feat.unsqueeze(0)
        if (mask is not None) and (mask.ndim == 1):
            mask = mask.unsqueeze(0)

        empty_mask = infer_empty_mask_from_feat(feat, mask)
        valid_mask = (~empty_mask).float()

        z = self.encoder(feat, valid_mask)
        return z

    @torch.no_grad()
    def decode_from_latent(self, z):
        self.eval()

        if z.ndim == 2:
            z = z.unsqueeze(0)

        z = z.to(next(self.parameters()).device).float()
        feat_hat, mask_logit = self.decoder(z)
        mask_prob = torch.sigmoid(mask_logit)
        return feat_hat, (mask_prob > 0.5).float(), mask_prob

    # backward-compatible aliases (but now these are latents, not tokens)
    @torch.no_grad()
    def encode_to_token(self, feat, mask=None):
        return self.encode_to_latent(feat, mask)

    @torch.no_grad()
    def decode_from_token(self, token_ids):
        return self.decode_from_latent(token_ids)


def gaussian_group_vqae_loss(feat_gt, mask_gt, feat_hat, mask_logit, vq_loss):
    """
    Same signature as before, but vq_loss is now zero.
    """
    mask_gt = mask_gt.float()
    valid = mask_gt.unsqueeze(-1)

    denom = valid.sum().clamp_min(1.0)
    denom_mask = mask_gt.sum().clamp_min(1.0)

    loss_xyz = (
        F.smooth_l1_loss(feat_hat[..., 0:3], feat_gt[..., 0:3], reduction='none') * valid
    ).sum() / denom

    loss_sc = (
        F.smooth_l1_loss(feat_hat[..., 3:6], feat_gt[..., 3:6], reduction='none') * valid
    ).sum() / denom

    qh = feat_hat[..., 6:10]
    qg = feat_gt[..., 6:10]
    qh = qh / qh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    qg = qg / qg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    qdot = (qh * qg).sum(dim=-1).abs()
    loss_q = ((1.0 - qdot) * mask_gt).sum() / denom_mask

    loss_op = (
        F.smooth_l1_loss(feat_hat[..., 10:11], feat_gt[..., 10:11], reduction='none') * valid
    ).sum() / denom

    loss_mask = F.binary_cross_entropy_with_logits(mask_logit, mask_gt)

    # optional tiny latent regularization (keep 0 first)
    latent_reg = vq_loss * LATENT_REG_WEIGHT

    recon = 10.0 * loss_xyz + 5.0 * loss_sc + 2.0 * loss_q + 1.0 * loss_op + 1.0 * loss_mask
    loss = recon + latent_reg

    return loss, {
        "loss_xyz": float(loss_xyz.detach().cpu()),
        "loss_sc": float(loss_sc.detach().cpu()),
        "loss_mask": float(loss_mask.detach().cpu()),
        "loss_vq": float(latent_reg.detach().cpu()),
    }


model = GaussianGroupContAE().to(device)
print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
print("continuous latent mode: no codebook, no discrete tokens")
print("latent shape:", f"(B, {GROUP_SIZE}, {LATENT_DIM})")

# %%
# %%
# Cell 6 — Train Gaussian Group VQ-AE (best/last/manual stop)

import os
import time
import numpy as np
import torch

# you can lower LR if needed
VQ_LR = 1e-4
VQ_WD = WD

BEST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_best.pt"
LAST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_last.pt"
INTERRUPT_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_interrupt.pt"
EARLY_STOP_FILE = os.path.join(OUT_DIR, "STOP_TRAINING")

USE_PATIENCE = True
PATIENCE_EPOCHS = 5
MIN_DELTA = 1e-4
CHECK_STOP_EVERY_N_ITERS = 50

opt = torch.optim.AdamW(model.parameters(), lr=VQ_LR, weight_decay=VQ_WD)

best_loss = 1e9
best_epoch = 0
epochs_no_improve = 0
stop_requested = False

def build_ckpt(epoch, ep_loss):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "loss": ep_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "config": {
            "GROUP_SIZE": GROUP_SIZE,
            "LATENT_DIM": LATENT_DIM,
            "GAUSS_DIM": GAUSS_DIM,
            "D_MODEL": D_MODEL,
            "WORLD_CENTER": world_center,
            "WORLD_SCALE": world_scale,
            "LR": VQ_LR,
            "WD": VQ_WD,
        },
    }

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        t0 = time.time()

        for it, (feat, mask) in enumerate(dl, start=1):
            feat = feat.to(device)
            mask = mask.to(device)

            z_e, z_q_st, token_ids, vq_loss, feat_hat, mask_logit = model(feat, mask)
            loss, st = gaussian_group_vqae_loss(feat, mask, feat_hat, mask_logit, vq_loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

            if it % LOG_EVERY == 0:
                print(
                    f"[VQAE] epoch {epoch}/{EPOCHS} it {it}/{len(dl)} "
                    f"loss={np.mean(losses[-LOG_EVERY:]):.4f} "
                    f"xyz={st['loss_xyz']:.4f} sc={st['loss_sc']:.4f} "
                    f"mask={st['loss_mask']:.4f} vq={st['loss_vq']:.4f}"
                )

            if (it % CHECK_STOP_EVERY_N_ITERS == 0) and os.path.exists(EARLY_STOP_FILE):
                print(f"[VQAE] stop file detected: {EARLY_STOP_FILE}")
                stop_requested = True
                break

        ep_loss = float(np.mean(losses)) if len(losses) > 0 else np.inf
        print(f"[VQAE] epoch {epoch} done loss={ep_loss:.6f} time={time.time()-t0:.1f}s")

        # save last
        torch.save(build_ckpt(epoch, ep_loss), LAST_CKPT_PATH)
        print("saved last:", LAST_CKPT_PATH)

        improved = (best_loss - ep_loss) > MIN_DELTA
        if improved:
            best_loss = ep_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(build_ckpt(epoch, ep_loss), BEST_CKPT_PATH)
            print("saved best:", BEST_CKPT_PATH, "loss:", best_loss)
        else:
            epochs_no_improve += 1

        if stop_requested:
            print("[VQAE] graceful stop completed.")
            break

        if USE_PATIENCE and (epochs_no_improve >= PATIENCE_EPOCHS):
            print(
                f"[VQAE] early stopping: no improvement for {epochs_no_improve} epochs "
                f"(best epoch={best_epoch}, best loss={best_loss:.6f})"
            )
            break

except KeyboardInterrupt:
    current_epoch = epoch if "epoch" in locals() else -1
    current_loss = ep_loss if "ep_loss" in locals() else np.inf
    torch.save(build_ckpt(current_epoch, current_loss), INTERRUPT_CKPT_PATH)
    print("\n[VQAE] KeyboardInterrupt caught.")
    print("saved interrupt checkpoint:", INTERRUPT_CKPT_PATH)

print("training finished.")
print("best loss:", best_loss)
print("best epoch:", best_epoch)
print("best ckpt:", BEST_CKPT_PATH)
print("last ckpt:", LAST_CKPT_PATH)

# %%
# %%
# Cell 7 — Export continuous latents with REAL pos_ids (not just group_ids)

import os
import numpy as np
import torch

BEST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_best.pt"
LAST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_last.pt"

TOKEN_OUT_NPZ = os.path.join(OUT_DIR, "group_tokens.npz")

if os.path.exists(BEST_CKPT_PATH):
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("loaded:", BEST_CKPT_PATH, "epoch:", ckpt["epoch"], "loss:", ckpt["loss"])
elif os.path.exists(LAST_CKPT_PATH):
    ckpt = torch.load(LAST_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("loaded:", LAST_CKPT_PATH, "epoch:", ckpt["epoch"], "loss:", ckpt["loss"])
else:
    raise FileNotFoundError("No trained checkpoint found.")

model.eval()

latent_all = []
with torch.no_grad():
    for i in range(len(ds)):
        feat, mask = ds[i]
        feat = feat.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        lat = model.encode_to_latent(feat, mask)   # (1, N, D)
        latent_all.append(lat[0].cpu().numpy().astype(np.float32))

latent_all = np.asarray(latent_all, dtype=np.float32)   # (num_samples, N, D)

np.savez_compressed(
    TOKEN_OUT_NPZ,
    latents=latent_all,
    mask=group_mask_all.astype(np.float32),
    frame_ids=sample_frame_ids.astype(np.int32),
    group_ids=sample_group_ids.astype(np.int32),          # kept only for debug
    pos_ids=sample_pos_ids.astype(np.int64),              # REAL packed xyz ids
    group_centers_cm=sample_group_centers_cm.astype(np.float32),

    # helpful meta for later visualization / inverse mapping
    world_center=world_center.astype(np.float32),
    world_scale=np.float32(world_scale),
    pos_pack_vox_cm=np.float32(POS_PACK_VOX_CM),
    pos_grid_bias=np.int32(POS_GRID_BIAS),
)

print("saved:", TOKEN_OUT_NPZ)
print("latents shape        :", latent_all.shape)
print("mask shape           :", group_mask_all.shape)
print("frame_ids shape      :", sample_frame_ids.shape)
print("group_ids shape      :", sample_group_ids.shape)
print("pos_ids shape        :", sample_pos_ids.shape)
print("group_centers_cm shape:", sample_group_centers_cm.shape)

# %%
# %%
# Cell 8 — Quick reconstruction sanity check for one sample

def denorm_xyz(xyz_n, world_center, world_scale):
    return xyz_n * world_scale + world_center[None, :]

def log_norm_to_scale_cm(log_sc, world_scale):
    return np.exp(log_sc) * world_scale

@torch.no_grad()
def inspect_one_group(sample_idx=0, mask_thresh=0.5):
    feat, mask = ds[sample_idx]
    feat_in = feat.unsqueeze(0).to(device)
    mask_in = mask.unsqueeze(0).to(device)

    # *FIXED*: Correctly unpack the 6 return values of the VQAE model 
    # instead of the old 5 VAE variables.
    z_e, z_q_st, token_ids, vq_loss, feat_hat, mask_logit = model(feat_in, mask_in)

    gt_feat = feat.numpy()
    gt_mask = mask.numpy()
    rc_feat = feat_hat[0].cpu().numpy().astype(np.float32)
    rc_mask = (torch.sigmoid(mask_logit)[0].cpu().numpy() > mask_thresh).astype(np.float32)

    gt_valid = gt_feat[gt_mask > 0.5]
    rc_valid = rc_feat[rc_mask > 0.5]

    print("sample idx:", sample_idx)
    print("frame id  :", int(sample_frame_ids[sample_idx]))
    print("group id  :", int(sample_group_ids[sample_idx]))
    print("gt valid  :", gt_valid.shape[0])
    print("recon valid:", rc_valid.shape[0])

    if gt_valid.shape[0] > 0:
        gt_xyz = denorm_xyz(gt_valid[:, 0:3], world_center, world_scale)
        gt_sc  = log_norm_to_scale_cm(gt_valid[:, 3:6], world_scale)
        print("GT xyz min:", gt_xyz.min(axis=0), "max:", gt_xyz.max(axis=0))
        print("GT scale mean:", gt_sc.mean(axis=0))

    if rc_valid.shape[0] > 0:
        rc_xyz = denorm_xyz(rc_valid[:, 0:3], world_center, world_scale)
        rc_sc  = log_norm_to_scale_cm(rc_valid[:, 3:6], world_scale)
        print("RC xyz min:", rc_xyz.min(axis=0), "max:", rc_xyz.max(axis=0))
        print("RC scale mean:", rc_sc.mean(axis=0))

inspect_one_group(sample_idx=0, mask_thresh=0.5)

# %%
# %%
# Cell X — Export one sample using TOKEN decode only (GT vs Token-decoded Recon) to PLY
# This verifies:
#   encode -> token
#   token -> decode
# works without depending on any extra latent input.

import os
import numpy as np
import torch

PLY_OUT_DIR = os.path.join(OUT_DIR, "ply_eval")
os.makedirs(PLY_OUT_DIR, exist_ok=True)

EXPORT_SAMPLE_IDX = 0
MASK_THRESH = 0.5

SURFACE_POINTS_PER_GAUSSIAN = 180
INNER_POINTS_PER_GAUSSIAN = 90
SURFACE_SIGMA_LEVEL = 2.0

BEST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_best.pt"
LAST_CKPT_PATH = os.path.splitext(CKPT_PATH)[0] + "_last.pt"

# load trained checkpoint
if os.path.exists(BEST_CKPT_PATH):
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("loaded:", BEST_CKPT_PATH, "epoch:", ckpt["epoch"], "loss:", ckpt["loss"])
elif os.path.exists(LAST_CKPT_PATH):
    ckpt = torch.load(LAST_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("loaded:", LAST_CKPT_PATH, "epoch:", ckpt["epoch"], "loss:", ckpt["loss"])
else:
    raise FileNotFoundError("No trained checkpoint found.")


def write_ply_points(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    n = xyz.shape[0]

    if rgb is not None:
        rgb = np.asarray(rgb)
        assert rgb.ndim == 2 and rgb.shape[1] == 3
        assert rgb.shape[0] == n

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
            rgb = rgb.astype(np.uint8)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_uniform_rgb(n, color):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.array([color], dtype=np.uint8), (n, 1))


def denorm_xyz(xyz_n, world_center, world_scale):
    return xyz_n * world_scale + world_center[None, :]


def log_norm_to_scale_cm(log_sc, world_scale):
    return np.exp(log_sc) * world_scale


def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    q = q / n
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def fibonacci_sphere(n):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts[i] = [x, y, z]
    return pts


def sample_gaussian_ellipsoid_points(xyz_cm, scale_cm, quat_wxyz, opacity,
                                     surface_n=180, inner_n=90, sigma_level=2.0, seed=123):
    rng = np.random.default_rng(seed)
    sphere_dirs = fibonacci_sphere(surface_n)

    all_pts = []
    all_rgb = []

    for mu, sc, q, op in zip(xyz_cm, scale_cm, quat_wxyz, opacity):
        mu = mu.astype(np.float32)
        sc = np.maximum(sc.astype(np.float32), 1e-3)
        q = q.astype(np.float32)
        op = float(op)

        R = quat_wxyz_to_rotmat(q)

        local_surface = sphere_dirs * (sc[None, :] * float(sigma_level))
        world_surface = (local_surface @ R.T) + mu[None, :]

        inner = rng.normal(size=(inner_n * 2, 3)).astype(np.float32)
        norm = np.linalg.norm(inner, axis=1, keepdims=True)
        inner = inner / np.clip(norm, 1e-8, None)
        rad = rng.random((inner.shape[0], 1), dtype=np.float32) ** (1.0 / 3.0)
        inner = (inner * rad)[:inner_n]

        local_inner = inner * (sc[None, :] * float(sigma_level))
        world_inner = (local_inner @ R.T) + mu[None, :]

        pts_i = np.concatenate([world_surface, world_inner], axis=0).astype(np.float32)

        c = int(np.clip(255.0 * op, 0, 255))
        rgb_i = np.tile(np.array([[255, c, 0]], dtype=np.uint8), (pts_i.shape[0], 1))

        all_pts.append(pts_i)
        all_rgb.append(rgb_i)

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)
    return pts, rgb


@torch.no_grad()
def export_one_token_decode_sample_to_ply(sample_idx=0, mask_thresh=0.5):
    model.eval()

    feat, mask = ds[sample_idx]
    feat_in = feat.unsqueeze(0).to(device)
    mask_in = mask.unsqueeze(0).to(device)

    # ----- encode -> token only -----
    token_ids = model.encode_to_token(feat_in, mask_in)    # (1, N)
    
    # 抽取頭幾個 Token ID 嚟做檔名展示
    tok_preview = "_".join(map(str, token_ids[0].cpu().numpy()))

    # ----- token -> decode only -----
    z = model.encode_to_latent(feat_in, mask_in)
    feat_hat, mask_hat, mask_prob = model.decode_from_latent(z)

    gt_feat = feat.numpy().astype(np.float32)
    gt_mask = mask.numpy().astype(np.float32)

    rc_feat = feat_hat[0].cpu().numpy().astype(np.float32)
    rc_prob = mask_prob[0].cpu().numpy().astype(np.float32)
    rc_mask = (rc_prob > mask_thresh).astype(np.float32)

    gt_valid = gt_feat[gt_mask > 0.5]
    rc_valid = rc_feat[rc_mask > 0.5]

    print("sample idx:", sample_idx)
    print("frame id  :", int(sample_frame_ids[sample_idx]))
    print("group id  :", int(sample_group_ids[sample_idx]))
    print("tokens    :", token_ids[0].cpu().numpy())
    print("gt valid  :", gt_valid.shape[0])
    print("recon valid:", rc_valid.shape[0])

    # GT decode
    if gt_valid.shape[0] > 0:
        gt_xyz = denorm_xyz(gt_valid[:, 0:3], world_center, world_scale).astype(np.float32)
        gt_sc = log_norm_to_scale_cm(gt_valid[:, 3:6], world_scale).astype(np.float32)
        gt_q = gt_valid[:, 6:10].astype(np.float32)
        gt_q /= np.clip(np.linalg.norm(gt_q, axis=1, keepdims=True), 1e-8, None)
        gt_op = np.clip(gt_valid[:, 10], 0.0, 1.0).astype(np.float32)
    else:
        gt_xyz = np.zeros((0, 3), dtype=np.float32)
        gt_sc = np.zeros((0, 3), dtype=np.float32)
        gt_q = np.zeros((0, 4), dtype=np.float32)
        gt_op = np.zeros((0,), dtype=np.float32)

    # Recon decode
    if rc_valid.shape[0] > 0:
        rc_xyz = denorm_xyz(rc_valid[:, 0:3], world_center, world_scale).astype(np.float32)
        rc_sc = log_norm_to_scale_cm(rc_valid[:, 3:6], world_scale).astype(np.float32)
        rc_q = rc_valid[:, 6:10].astype(np.float32)
        rc_q /= np.clip(np.linalg.norm(rc_q, axis=1, keepdims=True), 1e-8, None)
        rc_op = np.clip(rc_valid[:, 10], 0.0, 1.0).astype(np.float32)
    else:
        rc_xyz = np.zeros((0, 3), dtype=np.float32)
        rc_sc = np.zeros((0, 3), dtype=np.float32)
        rc_q = np.zeros((0, 4), dtype=np.float32)
        rc_op = np.zeros((0,), dtype=np.float32)

    tag = f"sample_{sample_idx:06d}_frame_{int(sample_frame_ids[sample_idx])}_group_{int(sample_group_ids[sample_idx])}_seq"

    # centers
    p_gt_cent = os.path.join(PLY_OUT_DIR, f"{tag}_gt_centers.ply")
    p_rc_cent = os.path.join(PLY_OUT_DIR, f"{tag}_token_recon_centers.ply")
    p_ov_cent = os.path.join(PLY_OUT_DIR, f"{tag}_overlay_centers.ply")

    write_ply_points(p_gt_cent, gt_xyz, make_uniform_rgb(gt_xyz.shape[0], [0, 255, 0]))
    write_ply_points(p_rc_cent, rc_xyz, make_uniform_rgb(rc_xyz.shape[0], [255, 0, 0]))

    ov_cent_xyz = np.concatenate([gt_xyz, rc_xyz], axis=0)
    ov_cent_rgb = np.concatenate([
        make_uniform_rgb(gt_xyz.shape[0], [0, 255, 0]),
        make_uniform_rgb(rc_xyz.shape[0], [255, 0, 0])
    ], axis=0)
    write_ply_points(p_ov_cent, ov_cent_xyz, ov_cent_rgb)

    # gaussian balls
    gt_pts, _ = sample_gaussian_ellipsoid_points(
        gt_xyz, gt_sc, gt_q, gt_op,
        surface_n=SURFACE_POINTS_PER_GAUSSIAN,
        inner_n=INNER_POINTS_PER_GAUSSIAN,
        sigma_level=SURFACE_SIGMA_LEVEL,
        seed=123
    )
    rc_pts, _ = sample_gaussian_ellipsoid_points(
        rc_xyz, rc_sc, rc_q, rc_op,
        surface_n=SURFACE_POINTS_PER_GAUSSIAN,
        inner_n=INNER_POINTS_PER_GAUSSIAN,
        sigma_level=SURFACE_SIGMA_LEVEL,
        seed=456
    )

    gt_overlay_rgb = make_uniform_rgb(gt_pts.shape[0], [0, 255, 0])
    rc_overlay_rgb = make_uniform_rgb(rc_pts.shape[0], [255, 0, 0])

    p_gt_ball = os.path.join(PLY_OUT_DIR, f"{tag}_gt_gaussian_balls.ply")
    p_rc_ball = os.path.join(PLY_OUT_DIR, f"{tag}_token_recon_gaussian_balls.ply")
    p_ov_ball = os.path.join(PLY_OUT_DIR, f"{tag}_overlay_gaussian_balls.ply")

    write_ply_points(p_gt_ball, gt_pts, gt_overlay_rgb)
    write_ply_points(p_rc_ball, rc_pts, rc_overlay_rgb)

    ov_ball_xyz = np.concatenate([gt_pts, rc_pts], axis=0)
    ov_ball_rgb = np.concatenate([gt_overlay_rgb, rc_overlay_rgb], axis=0)
    write_ply_points(p_ov_ball, ov_ball_xyz, ov_ball_rgb)

    print("saved:")
    print(" ", p_gt_cent)
    print(" ", p_rc_cent)
    print(" ", p_ov_cent)
    print(" ", p_gt_ball)
    print(" ", p_rc_ball)
    print(" ", p_ov_ball)


export_one_token_decode_sample_to_ply(sample_idx=EXPORT_SAMPLE_IDX, mask_thresh=MASK_THRESH)


export_one_token_decode_sample_to_ply(sample_idx=EXPORT_SAMPLE_IDX, mask_thresh=MASK_THRESH)

# %% [markdown]
# NPZ Visualize

# %%
# %%
# Export FULL patch-level Gaussians from one cache .npz
# This ignores feat/mask compression and fits one Gaussian per patch from centers_cm + local_pts

import os
import math
import numpy as np

# ===== CHANGE THIS =====
NPZ_PATH = r"./LearnCache_gs/gaussian_frame_cache/000000.npz"
OUT_DIR = r"./LearnCache_gs/ply_debug"
os.makedirs(OUT_DIR, exist_ok=True)

VOX_CM = 20.0
LOCAL_PTS_CLIP = 1.25

# Gaussian fit params
MIN_VALID_PATCH_PTS = 6
COV_REG_CM2 = 4.0
SCALE_STD_FACTOR = 1.5
MIN_SCALE_CM = 3.0
MAX_SCALE_CM = 80.0

# Export sampling
SURFACE_POINTS_PER_GAUSSIAN = 120
INNER_POINTS_PER_GAUSSIAN = 60
SURFACE_SIGMA_LEVEL = 2.0

# Safety caps
MAX_RAW_EXPORT = 200000
MAX_GAUSS_EXPORT = 400000


def write_ply_points(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    n = xyz.shape[0]

    if rgb is not None:
        rgb = np.asarray(rgb)
        assert rgb.ndim == 2 and rgb.shape[1] == 3
        assert rgb.shape[0] == n

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
            rgb = rgb.astype(np.uint8)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_uniform_rgb(n, color):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.array([color], dtype=np.uint8), (n, 1))


def downsample_pts(pts, max_pts, rgb=None, seed=123):
    pts = np.asarray(pts)
    if pts.shape[0] <= max_pts:
        return (pts, rgb) if rgb is not None else pts
    rng = np.random.default_rng(seed)
    sel = rng.choice(pts.shape[0], size=max_pts, replace=False)
    if rgb is None:
        return pts[sel]
    return pts[sel], rgb[sel]


def quat_from_rotmat(R):
    R = R.astype(np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-8)
    return q


def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    q = q / n
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def ensure_right_handed(R):
    R = R.copy().astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def reconstruct_patch_points(centers_cm, local_pts, local_count, vox_cm=20.0, local_pts_clip=1.25):
    """
    Returns:
      patches: list of (K,3) float32 arrays, one per patch
      raw_pts: all valid reconstructed points concatenated
    """
    centers_cm = np.asarray(centers_cm, dtype=np.float32)
    local_pts = np.asarray(local_pts, dtype=np.float32)
    local_count = np.asarray(local_count, dtype=np.int32)

    half = float(vox_cm) / 2.0
    patches = []
    raw_all = []

    V = centers_cm.shape[0]
    for i in range(V):
        cen = centers_cm[i]
        n = int(local_count[i]) if i < len(local_count) else local_pts.shape[1]
        n = max(0, min(n, local_pts.shape[1]))

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
    arr_cm: (K,3)
    returns:
      mu, scales_cm, quat_wxyz, opacity
    """
    K = arr_cm.shape[0]
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

    opacity = np.float32(np.clip(K / max(1.0, MIN_VALID_PATCH_PTS * 2.0), 0.05, 0.98))
    return mu, scales, quat, opacity


def fibonacci_sphere(n):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts[i] = [x, y, z]
    return pts


def sample_gaussian_ellipsoid_points(all_mu, all_sc, all_q, all_op,
                                     surface_n=120, inner_n=60, sigma_level=2.0, seed=123):
    rng = np.random.default_rng(seed)
    sphere_dirs = fibonacci_sphere(surface_n)

    all_pts = []
    all_rgb = []

    for mu, sc, q, op in zip(all_mu, all_sc, all_q, all_op):
        R = quat_wxyz_to_rotmat(q)

        local_surface = sphere_dirs * (sc[None, :] * float(sigma_level))
        world_surface = (local_surface @ R.T) + mu[None, :]

        inner = rng.normal(size=(inner_n * 2, 3)).astype(np.float32)
        norm = np.linalg.norm(inner, axis=1, keepdims=True)
        inner = inner / np.clip(norm, 1e-8, None)
        rad = rng.random((inner.shape[0], 1), dtype=np.float32) ** (1.0 / 3.0)
        inner = (inner * rad)[:inner_n]

        local_inner = inner * (sc[None, :] * float(sigma_level))
        world_inner = (local_inner @ R.T) + mu[None, :]

        pts_i = np.concatenate([world_surface, world_inner], axis=0).astype(np.float32)

        c = int(np.clip(255.0 * float(op), 0, 255))
        rgb_i = np.tile(np.array([[255, c, 0]], dtype=np.uint8), (pts_i.shape[0], 1))

        all_pts.append(pts_i)
        all_rgb.append(rgb_i)

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)
    return pts, rgb


# ===== load npz =====
z = np.load(NPZ_PATH, allow_pickle=True)
print("NPZ:", NPZ_PATH)
print("keys:", list(z.files))

if "empty" in z.files:
    raise RuntimeError("This cache is marked empty.")

required = ["centers_cm", "local_pts", "local_count"]
for k in required:
    if k not in z.files:
        raise RuntimeError(f"Missing required key: {k}")

centers_cm = z["centers_cm"].astype(np.float32)
local_pts = z["local_pts"].astype(np.float32)
local_count = z["local_count"].astype(np.int32)

print("centers_cm:", centers_cm.shape)
print("local_pts :", local_pts.shape)
print("local_count:", local_count.shape)

# ===== reconstruct all patch points =====
patches, raw_pts = reconstruct_patch_points(
    centers_cm, local_pts, local_count,
    vox_cm=VOX_CM,
    local_pts_clip=LOCAL_PTS_CLIP
)

print("num patches:", len(patches))
print("raw reconstructed points:", raw_pts.shape[0])

# export raw full-ish cloud
raw_pts_export = downsample_pts(raw_pts, MAX_RAW_EXPORT)
base = os.path.splitext(os.path.basename(NPZ_PATH))[0]

p_raw = os.path.join(OUT_DIR, f"{base}_raw_reconstructed_points_fullpatch.ply")
write_ply_points(p_raw, raw_pts_export, make_uniform_rgb(raw_pts_export.shape[0], [0, 255, 255]))
print("saved:", p_raw)

# ===== fit ALL patches to Gaussians =====
all_mu = []
all_sc = []
all_q = []
all_op = []

for arr in patches:
    if arr.shape[0] == 0:
        continue
    mu, sc, q, op = fit_patch_gaussian(arr)
    all_mu.append(mu)
    all_sc.append(sc)
    all_q.append(q)
    all_op.append(op)

if len(all_mu) == 0:
    raise RuntimeError("No valid patch gaussians fitted.")

all_mu = np.stack(all_mu, axis=0).astype(np.float32)
all_sc = np.stack(all_sc, axis=0).astype(np.float32)
all_q  = np.stack(all_q, axis=0).astype(np.float32)
all_op = np.asarray(all_op, dtype=np.float32)

print("all fitted patch gaussians:", all_mu.shape[0])

# export centers of all fitted gaussians
p_cent = os.path.join(OUT_DIR, f"{base}_all_patch_gaussian_centers.ply")
write_ply_points(p_cent, all_mu, make_uniform_rgb(all_mu.shape[0], [255, 0, 0]))
print("saved:", p_cent)

# export all gaussian balls
gauss_pts, gauss_rgb = sample_gaussian_ellipsoid_points(
    all_mu, all_sc, all_q, all_op,
    surface_n=SURFACE_POINTS_PER_GAUSSIAN,
    inner_n=INNER_POINTS_PER_GAUSSIAN,
    sigma_level=SURFACE_SIGMA_LEVEL,
    seed=123
)

gauss_pts, gauss_rgb = downsample_pts(gauss_pts, MAX_GAUSS_EXPORT, rgb=gauss_rgb)

p_gauss = os.path.join(OUT_DIR, f"{base}_all_patch_gaussians.ply")
write_ply_points(p_gauss, gauss_pts, gauss_rgb)
print("saved:", p_gauss)
print("exported gaussian-ball points:", gauss_pts.shape[0])

# overlay
raw_rgb = make_uniform_rgb(raw_pts_export.shape[0], [0, 255, 255])
overlay_xyz = np.concatenate([raw_pts_export, gauss_pts], axis=0)
overlay_rgb = np.concatenate([raw_rgb, gauss_rgb], axis=0)

p_overlay = os.path.join(OUT_DIR, f"{base}_overlay_raw_and_all_patch_gaussians.ply")
write_ply_points(p_overlay, overlay_xyz, overlay_rgb)
print("saved:", p_overlay)

# %%
# %%
# Compress FULL patch-level Gaussians to top-K and export PLY for comparison

import os
import math
import numpy as np

# ===== CHANGE THIS =====
NPZ_PATH = r"./LearnCache_gs/gaussian_frame_cache/000000.npz"
OUT_DIR = r"./LearnCache_gs/ply_debug"
os.makedirs(OUT_DIR, exist_ok=True)

VOX_CM = 20.0
LOCAL_PTS_CLIP = 1.25

# Full Gaussian fit params
MIN_VALID_PATCH_PTS = 6
COV_REG_CM2 = 4.0
SCALE_STD_FACTOR = 1.5
MIN_SCALE_CM = 3.0
MAX_SCALE_CM = 80.0

# Compression
TOP_K = 4096

# Export sampling
SURFACE_POINTS_PER_GAUSSIAN = 120
INNER_POINTS_PER_GAUSSIAN = 60
SURFACE_SIGMA_LEVEL = 2.0

# Safety caps
MAX_RAW_EXPORT = 200000
MAX_FULL_GAUSS_EXPORT = 400000
MAX_COMP_GAUSS_EXPORT = 250000


def write_ply_points(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    n = xyz.shape[0]

    if rgb is not None:
        rgb = np.asarray(rgb)
        assert rgb.ndim == 2 and rgb.shape[1] == 3
        assert rgb.shape[0] == n

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
            rgb = rgb.astype(np.uint8)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_uniform_rgb(n, color):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.array([color], dtype=np.uint8), (n, 1))


def downsample_pts(pts, max_pts, rgb=None, seed=123):
    pts = np.asarray(pts)
    if pts.shape[0] <= max_pts:
        return (pts, rgb) if rgb is not None else pts
    rng = np.random.default_rng(seed)
    sel = rng.choice(pts.shape[0], size=max_pts, replace=False)
    if rgb is None:
        return pts[sel]
    return pts[sel], rgb[sel]


def nearest_mean_dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    diff = a[:, None, :] - b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return float(np.sqrt(dist2.min(axis=1)).mean())


def chamfer_like_nn(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    diff = a[:, None, :] - b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    d_ab = np.sqrt(dist2.min(axis=1)).mean()
    d_ba = np.sqrt(dist2.min(axis=0)).mean()
    return float(0.5 * (d_ab + d_ba))


def quat_from_rotmat(R):
    R = R.astype(np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-8)
    return q


def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    q = q / n
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def ensure_right_handed(R):
    R = R.copy().astype(np.float32)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def reconstruct_patch_points(centers_cm, local_pts, local_count, vox_cm=20.0, local_pts_clip=1.25):
    half = float(vox_cm) / 2.0
    patches = []
    raw_all = []

    V = centers_cm.shape[0]
    for i in range(V):
        cen = centers_cm[i]
        n = int(local_count[i]) if i < len(local_count) else local_pts.shape[1]
        n = max(0, min(n, local_pts.shape[1]))

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

    raw_pts = np.concatenate(raw_all, axis=0).astype(np.float32) if len(raw_all) else np.zeros((0, 3), dtype=np.float32)
    return patches, raw_pts


def fit_patch_gaussian(arr_cm):
    K = arr_cm.shape[0]
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

    opacity = np.float32(np.clip(K / max(1.0, MIN_VALID_PATCH_PTS * 2.0), 0.05, 0.98))
    return mu, scales, quat, opacity


def fibonacci_sphere(n):
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        pts[i] = [x, y, z]
    return pts


def sample_gaussian_ellipsoid_points(all_mu, all_sc, all_q, all_op,
                                     surface_n=120, inner_n=60, sigma_level=2.0, seed=123):
    rng = np.random.default_rng(seed)
    sphere_dirs = fibonacci_sphere(surface_n)

    all_pts = []
    all_rgb = []

    for mu, sc, q, op in zip(all_mu, all_sc, all_q, all_op):
        R = quat_wxyz_to_rotmat(q)

        local_surface = sphere_dirs * (sc[None, :] * float(sigma_level))
        world_surface = (local_surface @ R.T) + mu[None, :]

        inner = rng.normal(size=(inner_n * 2, 3)).astype(np.float32)
        norm = np.linalg.norm(inner, axis=1, keepdims=True)
        inner = inner / np.clip(norm, 1e-8, None)
        rad = rng.random((inner.shape[0], 1), dtype=np.float32) ** (1.0 / 3.0)
        inner = (inner * rad)[:inner_n]

        local_inner = inner * (sc[None, :] * float(sigma_level))
        world_inner = (local_inner @ R.T) + mu[None, :]

        pts_i = np.concatenate([world_surface, world_inner], axis=0).astype(np.float32)

        c = int(np.clip(255.0 * float(op), 0, 255))
        rgb_i = np.tile(np.array([[255, c, 0]], dtype=np.uint8), (pts_i.shape[0], 1))

        all_pts.append(pts_i)
        all_rgb.append(rgb_i)

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)
    return pts, rgb


# ===== load npz =====
z = np.load(NPZ_PATH, allow_pickle=True)
print("NPZ:", NPZ_PATH)
print("keys:", list(z.files))

if "empty" in z.files:
    raise RuntimeError("This cache is marked empty.")

for k in ["centers_cm", "local_pts", "local_count"]:
    if k not in z.files:
        raise RuntimeError(f"Missing required key: {k}")

centers_cm = z["centers_cm"].astype(np.float32)
local_pts = z["local_pts"].astype(np.float32)
local_count = z["local_count"].astype(np.int32)

print("centers_cm:", centers_cm.shape)
print("local_pts :", local_pts.shape)
print("local_count:", local_count.shape)

# ===== reconstruct raw =====
patches, raw_pts = reconstruct_patch_points(
    centers_cm, local_pts, local_count,
    vox_cm=VOX_CM,
    local_pts_clip=LOCAL_PTS_CLIP
)

print("num patches:", len(patches))
print("raw reconstructed points:", raw_pts.shape[0])

# raw export
raw_pts_export = downsample_pts(raw_pts, MAX_RAW_EXPORT)
base = os.path.splitext(os.path.basename(NPZ_PATH))[0]

p_raw = os.path.join(OUT_DIR, f"{base}_raw_reconstructed_points_for_compression.ply")
write_ply_points(p_raw, raw_pts_export, make_uniform_rgb(raw_pts_export.shape[0], [0, 255, 255]))
print("saved:", p_raw)

# ===== fit all patch gaussians =====
all_mu = []
all_sc = []
all_q = []
all_op = []
all_count = []

for arr in patches:
    if arr.shape[0] == 0:
        continue
    mu, sc, q, op = fit_patch_gaussian(arr)
    all_mu.append(mu)
    all_sc.append(sc)
    all_q.append(q)
    all_op.append(op)
    all_count.append(arr.shape[0])

if len(all_mu) == 0:
    raise RuntimeError("No valid patch gaussians fitted.")

all_mu = np.stack(all_mu, axis=0).astype(np.float32)
all_sc = np.stack(all_sc, axis=0).astype(np.float32)
all_q  = np.stack(all_q, axis=0).astype(np.float32)
all_op = np.asarray(all_op, dtype=np.float32)
all_count = np.asarray(all_count, dtype=np.float32)

print("all fitted patch gaussians:", all_mu.shape[0])

# ===== build importance score and compress top-K =====
volume = all_sc[:, 0] * all_sc[:, 1] * all_sc[:, 2]
score = all_count * volume

order = np.argsort(-score)
keep = order[:min(TOP_K, len(order))]

cmp_mu = all_mu[keep]
cmp_sc = all_sc[keep]
cmp_q  = all_q[keep]
cmp_op = all_op[keep]
cmp_score = score[keep]

print("compressed gaussians:", cmp_mu.shape[0])
print("score range kept:", float(cmp_score.min()), "->", float(cmp_score.max()))

# ===== export full gaussians =====
full_pts, full_rgb = sample_gaussian_ellipsoid_points(
    all_mu, all_sc, all_q, all_op,
    surface_n=SURFACE_POINTS_PER_GAUSSIAN,
    inner_n=INNER_POINTS_PER_GAUSSIAN,
    sigma_level=SURFACE_SIGMA_LEVEL,
    seed=123
)
full_pts, full_rgb = downsample_pts(full_pts, MAX_FULL_GAUSS_EXPORT, rgb=full_rgb)

p_full = os.path.join(OUT_DIR, f"{base}_full_patch_gaussians.ply")
write_ply_points(p_full, full_pts, full_rgb)
print("saved:", p_full)
print("full gaussian-ball points:", full_pts.shape[0])

# ===== export compressed gaussians =====
cmp_pts, cmp_rgb = sample_gaussian_ellipsoid_points(
    cmp_mu, cmp_sc, cmp_q, cmp_op,
    surface_n=SURFACE_POINTS_PER_GAUSSIAN,
    inner_n=INNER_POINTS_PER_GAUSSIAN,
    sigma_level=SURFACE_SIGMA_LEVEL,
    seed=456
)
cmp_pts, cmp_rgb = downsample_pts(cmp_pts, MAX_COMP_GAUSS_EXPORT, rgb=cmp_rgb)

p_cmp = os.path.join(OUT_DIR, f"{base}_compressed_top{TOP_K}_gaussians.ply")
write_ply_points(p_cmp, cmp_pts, cmp_rgb)
print("saved:", p_cmp)
print("compressed gaussian-ball points:", cmp_pts.shape[0])

# ===== overlays =====
# full vs compressed
full_overlay_rgb = make_uniform_rgb(full_pts.shape[0], [0, 255, 0])   # green
cmp_overlay_rgb  = make_uniform_rgb(cmp_pts.shape[0], [255, 0, 0])    # red

overlay_fc_xyz = np.concatenate([full_pts, cmp_pts], axis=0)
overlay_fc_rgb = np.concatenate([full_overlay_rgb, cmp_overlay_rgb], axis=0)

p_overlay_fc = os.path.join(OUT_DIR, f"{base}_overlay_full_vs_compressed_top{TOP_K}.ply")
write_ply_points(p_overlay_fc, overlay_fc_xyz, overlay_fc_rgb)
print("saved:", p_overlay_fc)

# raw vs compressed
raw_overlay_rgb = make_uniform_rgb(raw_pts_export.shape[0], [0, 255, 255])  # cyan
overlay_rc_xyz = np.concatenate([raw_pts_export, cmp_pts], axis=0)
overlay_rc_rgb = np.concatenate([raw_overlay_rgb, cmp_overlay_rgb], axis=0)

p_overlay_rc = os.path.join(OUT_DIR, f"{base}_overlay_raw_vs_compressed_top{TOP_K}.ply")
write_ply_points(p_overlay_rc, overlay_rc_xyz, overlay_rc_rgb)
print("saved:", p_overlay_rc)

# ===== quick metrics =====
# use downsampled versions for tractable NN
raw_eval = raw_pts_export
full_eval = downsample_pts(full_pts, 50000)
cmp_eval = downsample_pts(cmp_pts, 50000)

# err_raw_full = chamfer_like_nn(raw_eval, full_eval)
# err_raw_cmp = chamfer_like_nn(raw_eval, cmp_eval)
# err_full_cmp = chamfer_like_nn(full_eval, cmp_eval)

print()
print("===== Compression Quality =====")
# print("Raw <-> FullPatchGauss (cm) :", err_raw_full)
# print("Raw <-> Compressed   (cm)   :", err_raw_cmp)
# print("Full <-> Compressed  (cm)   :", err_full_cmp)
print("Compression ratio (gauss count):", f"{len(all_mu)} -> {len(cmp_mu)}")



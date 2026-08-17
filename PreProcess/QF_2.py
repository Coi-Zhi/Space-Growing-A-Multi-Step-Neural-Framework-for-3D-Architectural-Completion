

import os, math, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import OpenEXR, Imath
Project_name = "Kidasaki"
# ------------------ USER PARAMS ------------------
ROOT      = r"K:\MasterEssay\Attempt_source_03\Frames"
MANIFEST  = os.path.join(ROOT, f"frame_manifest_{Project_name}.csv")

OUT_DIR   = os.path.join(ROOT, f"LearnCache_seq_fast_{Project_name}")   # NEW output folder
CACHE_DIR = os.path.join(OUT_DIR, "frame_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# EXR reading
STRIDE    = 2
MAX_DIST  = 1000

# token voxel for TRAIN (match your VOX_CM)
VOX_CM      = 20.0
PTS_PER_VOX = 256

# voxel for OVERLAP computation (coarser= faster; 10cm/20cm通常够用)
OVLP_VOX_CM = 10.0

# pose-based decimation (remove 60fps redundancy)
POSE_MIN_TRANS_CM = 1.0   # e.g. 10cm move to keep a frame
POSE_MIN_ROT_DEG  = 0.0    # e.g. 5deg rotate to keep a frame

# chain overlap target range (like your 20~30%)
TARGET_LOW, TARGET_HIGH = 0.10, 0.20
TARGET_MID = 0.25

# when searching next keyframe in decimated list
SEARCH_AHEAD = 150    # how many decimated frames to look ahead
SEARCH_STEP  = 1

# sequence sampling (for "prefix -> next")
MAX_PREFIX_LEN = 16
N_SEQ_SAMPLES  = 60000

PT = Imath.PixelType(Imath.PixelType.FLOAT)

# ------------------ math utils ------------------
def quat_normalize(q):
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0,0,0,1], np.float64)
    return q / n

def quat_to_R_xyzw(qx, qy, qz, qw):
    # returns 3x3
    x,y,z,w = quat_normalize([qx,qy,qz,qw])
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], dtype=np.float32)
    return R

def rot_delta_deg(q1, q2):
    # angle between orientations (xyzw)
    q1 = quat_normalize(q1); q2 = quat_normalize(q2)
    # relative quaternion q = q2 * conj(q1)
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    # conj(q1) = (-x1,-y1,-z1,w1)
    xr =  w2*(-x1) + x2*( w1) + y2*(-z1) - z2*(-y1)
    yr =  w2*(-y1) - x2*(-z1) + y2*( w1) + z2*(-x1)
    zr =  w2*(-z1) + x2*(-y1) - y2*(-x1) + z2*( w1)
    wr =  w2*( w1) - x2*(-x1) - y2*(-y1) - z2*(-z1)
    wr = float(np.clip(abs(wr), 0.0, 1.0))
    ang = 2.0 * math.acos(wr)
    return ang * 180.0 / math.pi

def pack_keys_int64(v_idx_int32):
    # pack (x,y,z) int32 -> int64 for fast intersection
    v = v_idx_int32.astype(np.int64)
    BIAS = 1 << 20
    x = v[:,0] + BIAS
    y = v[:,1] + BIAS
    z = v[:,2] + BIAS
    # 21 bits each
    return (x << 42) | (y << 21) | z

def unique_vox_idx(pts_cm, vox_cm):
    v = np.floor(pts_cm / float(vox_cm)).astype(np.int32)
    # unique rows fast
    key_view = v.view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)
    uniq = np.unique(key_view)
    uniq_v = uniq.view(np.int32).reshape(-1,3)
    return uniq_v

def intersection_count_sorted(a, b):
    # a,b are sorted int64 unique arrays
    i=j=0; cnt=0
    na, nb = a.shape[0], b.shape[0]
    while i<na and j<nb:
        av, bv = a[i], b[j]
        if av == bv:
            cnt += 1; i += 1; j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return cnt

# ------------------ EXR read ------------------
def read_exr_xyz(exr_path, stride=2, max_dist_cm=1000.0):
    f = OpenEXR.InputFile(exr_path)
    dw = f.header()["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    def ch(name):
        return np.frombuffer(f.channel(name, PT), dtype=np.float32).reshape(H, W)
    x = ch("R")[::stride, ::stride]
    y = ch("G")[::stride, ::stride]
    z = ch("B")[::stride, ::stride]
    pts = np.stack([x,y,z], axis=-1).reshape(-1,3).astype(np.float32)

    m = np.isfinite(pts).all(axis=1)
    pts = pts[m]
    if pts.shape[0] == 0:
        return np.zeros((0,3), np.float32)

    # distance clip (camera space)
    m = (np.sum(pts*pts, axis=1) < (max_dist_cm*max_dist_cm))
    pts = pts[m]
    pts = pts[~np.any(np.abs(pts) >= 60000.0, axis=1)]
    return pts.astype(np.float32)

# ------------------ USER PARAMS (fix) ------------------
TARGET_LOW, TARGET_HIGH = 0.10, 0.20
TARGET_MID = 0.15  # ✅ mid 要落喺 [low, high] 入面，否则“最接近 mid”会偏走

# ✅ 来自你 overlap 搜索的结论：quat=xyzw, map=x_-y_z, rot=R
AXIS_MAP = "x_-y_z"   # only flip Y
ROT_MODE = "R"        # IMPORTANT: use pts @ R (NOT R.T)

def apply_axis_map_cam(pts):
    pts = pts.astype(np.float32).copy()
    if AXIS_MAP == "x_-y_z":
        pts[:, 1] *= -1.0
    elif AXIS_MAP == "x_-y_-z":
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0
    elif AXIS_MAP == "x_z_-y":
        pts = pts[:, [0, 2, 1]]
        pts[:, 2] *= -1.0
    elif AXIS_MAP == "x_-z_-y":
        pts = pts[:, [0, 2, 1]]
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0
    else:
        raise ValueError(f"Unknown AXIS_MAP: {AXIS_MAP}")
    return pts

# def transform_cam_to_world(pts_cam, row):
#     # 1. 數據準備
#     pts = pts_cam.astype(np.float32).copy()
    
#     # [分析結果] Quick_Finding.py 中沒有翻轉 Y，這裡建議先保持一致 (註釋掉)。
#     # 如果你發現導出的場景上下顛倒，再把下面這行取消註釋。
#     pts[:, 1] *= -1.0 

#     # 2. 計算旋轉矩陣 R 和 位移 t
#     # 必須恢復這些代碼，否則所有點都在原點！
#     R = quat_to_R_xyzw(row["qx"], row["qy"], row["qz"], row["qw"])
#     t = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)

#     # 3. [核心修正] 應用 Pose 變換
#     # 數學原理：Row_Vector_World = Row_Vector_Cam @ R.T + t
#     # 因為 pts 是 (N,3) 行向量，所以旋轉矩陣要轉置 (R.T) 乘在右邊
#     #pts_w = (pts @ R.T) + t[None, :]
#     pts_w = pts + t[None, :]
    
#     return pts_w

def transform_cam_to_world(pts_cam, row):
    pts = pts_cam.astype(np.float32).copy()
    
    # Step 1: axis remap (camera convention → your world convention)
    pts = apply_axis_map_cam(pts)  # call the function you already wrote!
    
    # Step 2: rotate + translate into world space
    R = quat_to_R_xyzw(row["qx"], row["qy"], row["qz"], row["qw"])
    t = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
    
    # Row-vector convention: P_world = P_cam @ R.T + t
    pts_w = (pts @ R.T) + t[None, :]
    
    return pts_w

# ------------------ frame token cache ------------------
def build_frame_cache(frame_id, row, exr_path, vox_cm, pts_per_vox, ovlp_vox_cm):
    """
    outputs:
      - centers_cm: (V,3) float32 voxel centers (world)
      - local_pts : (V,N,3) float16 in [-1,1] roughly (world->local normalized)
      - keys_pack : (V,) int64 packed voxel indices at VOX_CM
      - ovlp_keys : (K,) int64 packed voxel indices at OVLP_VOX_CM (for overlap)
    """
    pts_cam = read_exr_xyz(exr_path, stride=STRIDE, max_dist_cm=MAX_DIST)
    if pts_cam.shape[0] == 0:
        return None

    pts_w = transform_cam_to_world(pts_cam, row)  # world cm

    # overlap keys (coarse)
    ovlp_idx = unique_vox_idx(pts_w, ovlp_vox_cm)
    ovlp_keys = np.sort(pack_keys_int64(ovlp_idx))

    # training voxel grouping
    v_idx = np.floor(pts_w / float(vox_cm)).astype(np.int32)
    key_view = v_idx.view([("x", np.int32), ("y", np.int32), ("z", np.int32)]).reshape(-1)
    uniq, inv = np.unique(key_view, return_inverse=True)
    V = uniq.shape[0]
    uniq_idx = uniq.view(np.int32).reshape(-1,3)

    # sort points by voxel id for grouping
    order = np.argsort(inv)
    inv_s = inv[order]
    pts_s = pts_w[order]

    # boundaries
    cuts = np.nonzero(np.diff(inv_s))[0] + 1
    starts = np.concatenate([[0], cuts])
    ends   = np.concatenate([cuts, [inv_s.shape[0]]])

    half = float(vox_cm) / 2.0
    centers_cm = (uniq_idx.astype(np.float32) + 0.5) * float(vox_cm)

    local_pts = np.zeros((V, pts_per_vox, 3), np.float16)
    keys_pack = pack_keys_int64(uniq_idx)
    # sample per voxel
    for vi, (s,e) in enumerate(zip(starts, ends)):
        seg = pts_s[s:e]
        if seg.shape[0] == 0:
            continue
        if seg.shape[0] >= pts_per_vox:
            sel = np.random.choice(seg.shape[0], pts_per_vox, replace=False)
        else:
            sel = np.random.choice(seg.shape[0], pts_per_vox, replace=True)
        samp = seg[sel].astype(np.float32)
        cen = centers_cm[vi]
        loc = (samp - cen[None,:]) / half
        local_pts[vi] = loc.astype(np.float16)

    out = dict(
        centers_cm=centers_cm.astype(np.float32),
        local_pts=local_pts,
        keys_pack=keys_pack.astype(np.int64),
        ovlp_keys=ovlp_keys.astype(np.int64),
        frame_id=np.int32(frame_id),
        exr_path=str(exr_path),
    )
    return out

# ------------------ main: pose decimate -> cache -> chain -> seq index ------------------
df = pd.read_csv(MANIFEST).sort_values("seq_frame").reset_index(drop=True)
n_total = len(df)

# pose decimation (no EXR IO here)
keep = [0]
last = df.iloc[0]
last_q = [last["qx"], last["qy"], last["qz"], last["qw"]]
for i in range(1, n_total):
    r = df.iloc[i]
    dx = float(r["x"] - last["x"]); dy = float(r["y"] - last["y"]); dz = float(r["z"] - last["z"])
    dtrans = math.sqrt(dx*dx + dy*dy + dz*dz)
    drot = rot_delta_deg(last_q, [r["qx"], r["qy"], r["qz"], r["qw"]])
    if (dtrans >= POSE_MIN_TRANS_CM) or (drot >= POSE_MIN_ROT_DEG):
        keep.append(i)
        last = r
        last_q = [r["qx"], r["qy"], r["qz"], r["qw"]]

print(f"pose-decimate: {n_total} -> {len(keep)} frames")

# build caches (read EXR only for kept frames)
cache_paths = []
for ii in tqdm(range(len(keep)), desc="Caching keyframes"):
    src_i = keep[ii]
    row = df.iloc[src_i]
    exr_path = row["exr_path"]
    frame_id = int(row["seq_frame"]) if "seq_frame" in row else int(src_i)

    out_path = os.path.join(CACHE_DIR, f"{ii:06d}.npz")
    cache_paths.append(out_path)
    if os.path.exists(out_path):
        continue

    pack = build_frame_cache(ii, row, exr_path, VOX_CM, PTS_PER_VOX, OVLP_VOX_CM)
    if pack is None:
        # save empty marker
        np.savez_compressed(out_path, empty=np.int32(1))
        continue
    np.savez_compressed(out_path, **pack)

print("cache done:", CACHE_DIR)

# load ovlp keys for chain
ovlp_list = []
valid_ids = []
for i, p in enumerate(cache_paths):
    z = np.load(p, allow_pickle=True)
    if "empty" in z.files:
        continue
    ovlp_list.append(z["ovlp_keys"].astype(np.int64))
    valid_ids.append(i)

print("valid cached frames:", len(valid_ids))

def overlap_A_to_B(iA, iB):
    # overlap = |A ∩ B| / |B|
    A = ovlp_list[iA]
    B = ovlp_list[iB]
    if B.shape[0] == 0 or A.shape[0] == 0:
        return 0.0
    inter = intersection_count_sorted(A, B)
    return float(inter) / float(B.shape[0])

# build chain on decimated frames (indices in valid_ids space)
chain = [0]
cur = 0
while True:
    best = None
    best_ov = None
    # search forward
    found = False
    for step in range(1, SEARCH_AHEAD+1, SEARCH_STEP):
        cand = cur + step
        if cand >= len(valid_ids):
            break
        ov = overlap_A_to_B(cur, cand)
        if TARGET_LOW <= ov <= TARGET_HIGH:
            best = cand; best_ov = ov; found = True
            break
        # track closest to target mid
        if best is None or abs(ov - TARGET_MID) < abs(best_ov - TARGET_MID):
            best = cand; best_ov = ov

    if best is None or best <= cur:
        break
    chain.append(best)
    cur = best
    if cur >= len(valid_ids)-1:
        break

print("chain length:", len(chain), "overlap approx:", TARGET_LOW, "~", TARGET_HIGH)

# save chain indices (in CACHE index space)
os.makedirs(OUT_DIR, exist_ok=True)
chain_cache_ids = np.array([valid_ids[c] for c in chain], dtype=np.int32)
np.save(os.path.join(OUT_DIR, "chain_indices.npy"), chain_cache_ids)

# generate sequence training index: (start, prefix_len)
pairs = []
Lmax = min(MAX_PREFIX_LEN, len(chain_cache_ids)-1)
for _ in range(N_SEQ_SAMPLES):
    start = np.random.randint(0, len(chain_cache_ids)-1)
    maxL = min(Lmax, (len(chain_cache_ids)-1) - start)
    L = np.random.randint(1, maxL+1)
    pairs.append((start, L))
pairs = np.array(pairs, dtype=np.int32)
np.save(os.path.join(OUT_DIR, "seq_index.npy"), pairs)

meta = dict(
    ROOT=ROOT, MANIFEST=MANIFEST, OUT_DIR=OUT_DIR,
    VOX_CM=VOX_CM, PTS_PER_VOX=PTS_PER_VOX, OVLP_VOX_CM=OVLP_VOX_CM,
    decimate=dict(trans_cm=POSE_MIN_TRANS_CM, rot_deg=POSE_MIN_ROT_DEG),
    overlap=dict(low=TARGET_LOW, high=TARGET_HIGH, search_ahead=SEARCH_AHEAD),
    chain_len=int(chain_cache_ids.shape[0]),
    seq_samples=int(pairs.shape[0]),
    max_prefix_len=int(Lmax),
)
with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("✅ wrote:", OUT_DIR)
print(" - frame_cache/*.npz")
print(" - chain_indices.npy")
print(" - seq_index.npy")
print(" - meta.json")

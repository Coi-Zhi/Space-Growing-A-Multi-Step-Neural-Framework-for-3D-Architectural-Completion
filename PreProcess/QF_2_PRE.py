import os, re, glob
import pandas as pd
Project_name = "AZUMA"
root = r"K:\MasterEssay\Attempt_source_03\Frames"
frames_dir = os.path.join(root, Project_name)
pose_csv = os.path.join(root, f"camera_poses_{Project_name}.csv")
out_manifest = os.path.join(root, f"frame_manifest_{Project_name}.csv")

# 1) 讀 pose
df = pd.read_csv(pose_csv)
# 【修正點 1】：明確指定轉換為 int64
df["seq_frame"] = df["seq_frame"].astype("int64")
df = df.sort_values("seq_frame")

# 2) 掃 EXR，從檔名抽出 frame id
exr_files = sorted(glob.glob(os.path.join(frames_dir, "*.exr")))
pat = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)

exr_rows = []
bad = 0
for p in exr_files:
    m = pat.search(os.path.basename(p))
    if not m:
        bad += 1
        continue
    fid = int(m.group(1))
    exr_rows.append((fid, p))

exr_df = pd.DataFrame(exr_rows, columns=["exr_frame", "exr_path"])

print("pose rows:", len(df))
print("exr files:", len(exr_files), "parsed:", len(exr_df), "badname:", bad)

# 3) 智慧合併 
# 【修正點 2】：乘完 12 之後，也明確指定轉換為 int64
exr_df["target_pose_frame"] = (exr_df["exr_frame"] * 12).astype("int64")
exr_df = exr_df.sort_values("target_pose_frame")

# 自動為每一張 EXR 尋找時間上「最接近」的 Pose 軌跡
m = pd.merge_asof(exr_df, df, left_on="target_pose_frame", right_on="seq_frame", direction="nearest")

# 4) 加 0-based index
m["frame_idx"] = m.index

# 5) 只留需要的欄位
keep = [
    "frame_idx", "seq_frame", "time_sec", "exr_path",
    "x","y","z","qx","qy","qz","qw",
    "fov_deg","focal_mm","sensor_w_mm","sensor_h_mm",
    "camera_actor"
]
keep = [c for c in keep if c in m.columns]
m[keep].to_csv(out_manifest, index=False, encoding="utf-8")

print("✅ 成功！已完美配對並寫入:", out_manifest, "共:", len(m), "行資料")
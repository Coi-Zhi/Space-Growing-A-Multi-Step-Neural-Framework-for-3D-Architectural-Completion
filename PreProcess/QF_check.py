import os, json
import numpy as np

# ------------------ 參數設定 ------------------
Project_name = "AZUMA"
ROOT = r"K:\MasterEssay\Attempt_source_03\Frames"
OUT_DIR = os.path.join(ROOT, f"LearnCache_seq_fast_{Project_name}")
CACHE_DIR = os.path.join(OUT_DIR, "frame_cache")
PLY_OUT_DIR = os.path.join(OUT_DIR, "ply_check")

# 要檢查 Chain 裡面嘅頭幾多格？(設定 20-30 格已經好夠睇出對齊效果)
CHECK_FRAMES_COUNT = 30 

os.makedirs(PLY_OUT_DIR, exist_ok=True)

# ------------------ 工具函數 ------------------
def write_points_rgb_ply(points_xyz, colors_rgb, save_path):
    """將點雲與顏色寫入 PLY 檔案"""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points_xyz, colors_rgb):
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def get_distinct_color(idx):
    """為不同影格生成高對比度嘅顏色"""
    colors = [
        [255, 50, 50],   # 紅
        [50, 255, 50],   # 綠
        [50, 100, 255],  # 藍
        [255, 255, 50],  # 黃
        [255, 50, 255],  # 紫
        [50, 255, 255],  # 青
        [255, 150, 50],  # 橙
    ]
    return colors[idx % len(colors)]

# ------------------ 主程式 ------------------
def main():
    chain_path = os.path.join(OUT_DIR, "chain_indices.npy")
    if not os.path.exists(chain_path):
        print(f"❌ 搵唔到 Chain 檔案: {chain_path}")
        return

    # 讀取 chain
    chain_ids = np.load(chain_path).astype(np.int32)
    print(f"✅ 成功讀取 Chain，總長度: {len(chain_ids)}")
    
    check_len = min(CHECK_FRAMES_COUNT, len(chain_ids))
    
    all_pts = []
    all_colors = []

    for i in range(check_len):
        frame_idx = chain_ids[i]
        npz_path = os.path.join(CACHE_DIR, f"{frame_idx:06d}.npz")
        
        if not os.path.exists(npz_path):
            print(f"⚠️ 搵唔到快取檔案: {npz_path}")
            continue
            
        z = np.load(npz_path, allow_pickle=False)
        if "empty" in z.files:
            print(f"⚠️ 第 {i} 步 (frame {frame_idx}) 係空嘅。")
            continue
            
        # 讀取 Voxel 中心點 (呢個就係你個模型真正見到嘅空間結構)
        centers_cm = z["centers_cm"].astype(np.float32)
        
        # 1. 儲存單幀 PLY
        single_color = np.tile([200, 200, 200], (len(centers_cm), 1))
        single_ply_path = os.path.join(PLY_OUT_DIR, f"step_{i:03d}_frame_{frame_idx:06d}.ply")
        write_points_rgb_ply(centers_cm, single_color, single_ply_path)
        
        # 2. 準備累積場景的資料 (賦予當前影格特定顏色)
        frame_color = get_distinct_color(i)
        colors = np.tile(frame_color, (len(centers_cm), 1))
        
        all_pts.append(centers_cm)
        all_colors.append(colors)
        print(f"👉 處理咗第 {i+1}/{check_len} 格 (Frame {frame_idx}) - {len(centers_cm)} 個 Voxels")

    # 3. 儲存累積場景 PLY
    if len(all_pts) > 0:
        merged_pts = np.concatenate(all_pts, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0)
        
        merged_ply_path = os.path.join(PLY_OUT_DIR, "_Accumulated_Scene.ply")
        write_points_rgb_ply(merged_pts, merged_colors, merged_ply_path)
        print(f"🎉 成功！累積場景已儲存至: {merged_ply_path}")
        print(f"   總共包含 {len(merged_pts)} 個 Voxels。快啲打開 MeshLab 睇下啦！")
    else:
        print("❌ 冇任何有效點雲資料可以導出。")

if __name__ == "__main__":
    main()
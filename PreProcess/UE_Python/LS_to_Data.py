import unreal
import csv, os

# ========= 1) Level Sequence 路徑 =========
LEVEL_SEQUENCE_PATH = "/Script/LevelSequence.LevelSequence'/Game/Cinematics/Takes/2026-03-08/Scene_1_05.Scene_1_05'"

# ========= 2) 輸出 CSV =========
OUTPUT_CSV = r"K:\MasterEssay\Attempt_source_03\Frames\camera_poses_CASA.csv"

# ========= Load Sequence =========
seq = unreal.load_asset(LEVEL_SEQUENCE_PATH)
if not seq:
    raise RuntimeError("Failed to load Level Sequence")

# ========= FPS & Frame Range（UE5.5 正確 API） =========
fps_rate = unreal.MovieSceneSequenceExtensions.get_display_rate(seq)
fps = fps_rate.numerator / fps_rate.denominator

start_frame = unreal.MovieSceneSequenceExtensions.get_playback_start(seq)
end_frame   = unreal.MovieSceneSequenceExtensions.get_playback_end(seq)
frame_count = end_frame - start_frame

print("FPS:", fps, "| Frames:", frame_count, "| Range:", start_frame, "→", end_frame)

# ========= 取得 Editor World（正確方式） =========
editor_subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = editor_subsys.get_editor_world()

# ========= 建立 Sequence Player =========
player, seq_actor = unreal.LevelSequencePlayer.create_level_sequence_player(
    world,
    seq,
    unreal.MovieSceneSequencePlaybackSettings()
)

# ========= helper：跳到指定 frame =========
def set_to_frame(player, frame_int, start_frame, fps):
    # 優先：用 FRAME（最準）
    try:
        frame_num = unreal.FrameNumber(int(frame_int))
        frame_time = unreal.FrameTime(frame_num)
        params = unreal.MovieSceneSequencePlaybackParams(
            frame=frame_time,
            position_type=unreal.MovieScenePositionType.FRAME,
            update_method=unreal.UpdatePositionMethod.JUMP
        )
        player.set_playback_position(params)
        return
    except Exception as e:
        # fallback：用 TIME（仍然可靠，只係用秒）
        time_sec = (int(frame_int) - int(start_frame)) / float(fps)
        params = unreal.MovieSceneSequencePlaybackParams(
            time=float(time_sec),
            position_type=unreal.MovieScenePositionType.TIME,
            update_method=unreal.UpdatePositionMethod.JUMP
        )
        player.set_playback_position(params)

# ========= 輸出 =========
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "seq_frame", "time_sec",
        "x", "y", "z",
        "pitch", "yaw", "roll",
        "qx", "qy", "qz", "qw",
        "fov_deg", "focal_mm", "sensor_w_mm", "sensor_h_mm",
        "camera_actor"
    ])

    for frame in range(start_frame, end_frame):
        set_to_frame(player, frame, start_frame, fps)

        cam_comp = player.get_active_camera_component()
        if not cam_comp:
            # 無 camera cut（極少見）
            w.writerow([frame, (frame-start_frame)/fps] + [""]*14 + [""])
            continue

        cam_actor = cam_comp.get_owner()
        t = cam_actor.get_actor_transform()

        loc = t.translation
        q   = t.rotation
        r   = q.rotator()

        fov = float(cam_comp.field_of_view)
        focal = ""
        sw = ""
        sh = ""

        if isinstance(cam_comp, unreal.CineCameraComponent):
            focal = float(cam_comp.get_current_focal_length())
            filmback = cam_comp.get_filmback()
            sw = float(filmback.sensor_width)
            sh = float(filmback.sensor_height)

        time_sec = (frame - start_frame) / fps

        w.writerow([
            frame, time_sec,
            loc.x, loc.y, loc.z,
            r.pitch, r.yaw, r.roll,
            q.x, q.y, q.z, q.w,
            fov, focal, sw, sh,
            cam_actor.get_name()
        ])

print("✅ Camera pose exported to:", OUTPUT_CSV)
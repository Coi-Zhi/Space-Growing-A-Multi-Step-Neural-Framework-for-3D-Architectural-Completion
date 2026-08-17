import unreal

# 獲取當前在 Content Browser 中選取的資產
selected_assets = unreal.EditorUtilityLibrary.get_selected_assets()

for asset in selected_assets:
    if isinstance(asset, unreal.StaticMesh):
        # 獲取 BodySetup
        body_setup = asset.get_editor_property("body_setup")
        if body_setup:
            # 更改碰撞複雜度
            body_setup.set_editor_property("collision_trace_flag", unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
            print(f"已修改: {asset.get_name()}")
            
# 別忘了儲存修改的資產
unreal.EditorAssetLibrary.save_loaded_assets(selected_assets)
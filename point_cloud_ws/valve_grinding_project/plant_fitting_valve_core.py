#计算阀芯密封面的圆心位置和高度

import open3d as o3d
import numpy as np
import os

def process_valve_features(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty(): return

    # --- 1. 空间直通滤波 (裁剪底座) ---
    points = np.asarray(pcd.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # 保留底座以上的部分
    height_cutoff = z_min + (z_max - z_min) * 0.7
    keep_indices = np.where(points[:, 2] < height_cutoff)[0]
    pcd = pcd.select_by_index(keep_indices)
    
    points_clean = np.asarray(pcd.points)
    z_min_c, z_max_c = points_clean[:, 2].min(), points_clean[:, 2].max()
    z_range_c = z_max_c - z_min_c

    # --- 2. 提取特征面 ---
    top_mask = points_clean[:, 2] < (z_min_c + z_range_c * 0.5)
    ring_mask = (points_clean[:, 2] > (z_min_c + z_range_c * 0.5)) & \
                (points_clean[:, 2] < (z_min_c + z_range_c ))

    def get_plane_info(mask):
        indices = np.where(mask)[0]
        if len(indices) < 20: return None
        
        part_pcd = pcd.select_by_index(indices)
        plane_model, inliers = part_pcd.segment_plane(distance_threshold=0.1,
                                                     ransac_n=3,
                                                     num_iterations=500)
        inlier_cloud = part_pcd.select_by_index(inliers)
        center = np.mean(np.asarray(inlier_cloud.points), axis=0)
        return {
            'pcd': inlier_cloud,
            'center': center,
            'height': center[2]
        }

    top_info = get_plane_info(top_mask)
    ring_info = get_plane_info(ring_mask)

    if top_info is None or ring_info is None:
        print("未能提取到足够的特征面进行对比显示。")
        return

    # 确定哪个面距离相机最近 (假设 Z 越小越近)
    feature_list = [top_info, ring_info]
    nearest_feature = sorted(feature_list, key=lambda x: x['height'])[0]

    # --- 3. 使用新的 GUI API 显示 ---
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = app.create_window("阀芯特征测量展示", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
    win.add_child(widget3d)

    # 设置背景颜色为白色
    widget3d.scene.set_background([1, 1, 1, 1])

    # 定义材质
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # 添加几何体与标签
    # 顶部圆面显示为红色，下方环面显示为绿色（参考代码配色）
    top_info['pcd'].paint_uniform_color([1, 0, 0])
    ring_info['pcd'].paint_uniform_color([0, 0.8, 0])

    widget3d.scene.add_geometry("top_plane", top_info['pcd'], mat)
    widget3d.scene.add_geometry("ring_plane", ring_info['pcd'], mat)

    # 添加圆心标注球体
    sphere_mat = o3d.visualization.rendering.MaterialRecord()
    sphere_mat.shader = "defaultLit"
    
    for i, item in enumerate(feature_list):
        center = item['center']
        name = "top plane" if item == top_info else "valve core plane"
        
        # 仅为最近的面添加详细圆心标注，其他的仅显示高度
        if item == nearest_feature:
            label_text = (f"NEAREST: {name}\n"
                          f"Height: {item['height']:.2f}mm\n"
                          f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        else:
            label_text = f"{name}\nHeight: {item['height']:.2f}mm"
        
        widget3d.add_3d_label(center, label_text)

    # 添加参考坐标轴
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    widget3d.scene.add_geometry("mesh_frame", mesh_frame, mat)

    # 设置视角
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60, bbox, bbox.get_center())

    print("窗口已启动，显示高度差及圆心坐标。")
    app.run()

if __name__ == "__main__":
    input_file = r"D:\vs_ws\vision_detect\point_cloud_ws\valve_grinding_project\point_cloud_valve_core_data\valve_data_10.pcd" 
    process_valve_features(input_file)
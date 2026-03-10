#法兰倾斜度计算

import open3d as o3d
import numpy as np
import os

def fit_and_display_tilt_analysis(file_path):
    if not os.path.exists(file_path): return
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty(): return

    # --- 1. 平面拟合与倾角计算 ---
    # 提取点数最多的面（法兰面）
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=2000)
    [a, b, c, d] = plane_model
    flange_pcd = pcd.select_by_index(inliers)
    
    # 获取圆心位置
    pts = np.asarray(flange_pcd.points)
    center = np.mean(pts, axis=0)

    # 计算法向量 (确保朝向相机方向)
    n = np.array([a, b, c])
    if n[2] < 0: n = -n
    n = n / np.linalg.norm(n) # 单位化

    # 计算总倾角 (与相机基准轴 [0,0,1] 的夹角)
    cos_theta = n[2] # 向量点积: n · [0,0,1] = n_z
    total_tilt = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # 分解倾角分量 (更具工程参考价值)
    tilt_x = np.degrees(np.arctan2(n[1], n[2])) # 绕X轴偏转 (Roll)
    tilt_y = np.degrees(np.arctan2(-n[0], n[2])) # 绕Y轴偏转 (Pitch)

    # --- 2. 创建 3D 矢量指示器 ---
    # 创建红色箭头：代表法兰面实际法方向
    arrow_actual = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.8, cone_radius=1.5, cylinder_height=35, cone_height=7)
    arrow_actual.paint_uniform_color([1, 0, 0]) # 红色
    
    # 旋转箭头指向法线 n
    # 建立从 [0,0,1] 到 n 的旋转矩阵
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, n)
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * np.arccos(cos_theta))
        arrow_actual.rotate(rot_matrix, center=(0, 0, 0))
    arrow_actual.translate(center)

    # 创建蓝色虚影箭头：代表绝对垂直基准 (相机轴)
    arrow_ref = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.4, cone_radius=1.0, cylinder_height=45, cone_height=5)
    arrow_ref.paint_uniform_color([0, 0.5, 1]) # 蓝色
    arrow_ref.translate(center)

    # --- 3. 渲染展示 ---
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = app.create_window("相机基准面倾斜分析", 1280, 800)
    
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
    widget3d.scene.set_background([1, 1, 1, 1]) # 白色背景
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    
    flange_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # 灰色平面
    widget3d.scene.add_geometry("flange", flange_pcd, mat)
    widget3d.scene.add_geometry("vector_actual", arrow_actual, mat)
    widget3d.scene.add_geometry("vector_ref", arrow_ref, mat)

    # 在箭头上方实时标注角度信息
    label_pos = center + [0, 0, 50]
    label_text = (f"Total Tilt: {total_tilt:.4f}°\n"
                  f"Roll (X): {tilt_x:.3f}°\n"
                  f"Pitch (Y): {tilt_y:.3f}°")
    widget3d.add_3d_label(label_pos, label_text)

    win.add_child(widget3d)
    widget3d.setup_camera(60, widget3d.scene.bounding_box, center)
    app.run()

if __name__ == "__main__":
    path = r"D:\vs_ws\vision_detect\point_cloud_ws\valve_grinding_project\point_clound_data\2026-01-12-16-14-22_RTPointCloud.pcd"
    fit_and_display_tilt_analysis(path)
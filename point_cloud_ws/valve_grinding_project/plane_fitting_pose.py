#计算阀体密封面和法兰面的高度和法兰面的圆心位置

import open3d as o3d
import numpy as np
import os

def fit_and_report_measurements(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty(): return

    # --- 1. 算法提取部分 (保持已验证的参数) ---
    dist_threshold = 0.3
    absolute_min_points = 500
    cluster_eps = 10 
    
    found_planes_info = []  
    rest_pcd = pcd
    max_attempts = 15

    print("--- 正在提取平面并计算坐标 ---")

    for _ in range(max_attempts):
        if len(rest_pcd.points) < absolute_min_points: break
        plane_model, inliers = rest_pcd.segment_plane(
            distance_threshold=dist_threshold, ransac_n=3, num_iterations=3000)
        
        [a, b, c, d] = plane_model
        if abs(c) < 0.95: 
            rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
            continue
            
        candidate_cloud = rest_pcd.select_by_index(inliers)
        labels = np.array(candidate_cloud.cluster_dbscan(eps=cluster_eps, min_points=100))
        
        if len(labels) > 0 and labels.max() >= 0:
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            for label, p_count in zip(unique_labels, counts):
                if p_count > absolute_min_points:
                    cluster_indices = np.where(labels == label)[0]
                    cluster_pcd = candidate_cloud.select_by_index(cluster_indices)
                    
                    is_duplicate = False
                    for existing in found_planes_info:
                        if abs(d - existing['model'][3]) < 2.0:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        pts = np.asarray(cluster_pcd.points)
                        center = np.mean(pts, axis=0)
                        found_planes_info.append({
                            'model': plane_model, 
                            'pcd': cluster_pcd, 
                            'count': p_count,
                            'center': center,
                            'height': center[2]  # 确保有 height 属性用于后续排序和显示
                        })

        rest_pcd = rest_pcd.select_by_index(inliers, invert=True)

    # 初始按点数排序
    found_planes_info.sort(key=lambda x: x['count'], reverse=True)
    final_planes = found_planes_info[:2]

    if not final_planes:
        print("未检测到有效平面。")
        return

    # --- 2. 逻辑处理：找到距离相机最近的面 ---
    # 假设 Z 轴正方向指向远离相机的方向，则 Z 值最小的面即为最近的面
    nearest_plane_list = sorted(final_planes, key=lambda x: x['height'])
    nearest_plane = nearest_plane_list[0] if nearest_plane_list else None

    # --- 3. 使用新的可视化 API 显示标签 ---
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    # 创建窗口
    win = app.create_window("测量结果展示", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(win.renderer)
    win.add_child(widget3d)
    
    # 设置背景颜色为白色
    widget3d.scene.set_background([1, 1, 1, 1]) 
    
    colors = [[1, 0, 0], [0, 0.8, 0]] # 红、绿
    
    for i, item in enumerate(final_planes):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        
        # 添加点云
        pcd_name = f"plane_{i}"
        item['pcd'].paint_uniform_color(colors[i % 2])
        widget3d.scene.add_geometry(pcd_name, item['pcd'], mat)
        
        # --- 仅为最近的那个面添加圆心标注 ---
        center = item['center']
        if item is nearest_plane:
            # 最近的面：显示标题 + 高度 + 圆心坐标
            label_text = (f"NEAREST PLANE\n"
                          f"Height: {item['height']:.2f}mm\n"
                          f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        else:
            # 其他面：仅显示高度
            label_text = f"Height: {item['height']:.2f}mm"
        
        # 在平面中心点位置显示标签
        widget3d.add_3d_label(center, label_text)

    # 添加一个参考坐标轴（可选）
    # widget3d.scene.show_axes(True)
        
    # 设置视角
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60, bbox, bbox.get_center())
    
    print("窗口已启动，请查看可视化结果。")
    app.run()

if __name__ == "__main__":
    file_path = r"D:\vs_ws\vision_detect\point_cloud_ws\valve_grinding_project\point_clound_data\2026-01-12-16-14-22_RTPointCloud.pcd"
    fit_and_report_measurements(file_path)
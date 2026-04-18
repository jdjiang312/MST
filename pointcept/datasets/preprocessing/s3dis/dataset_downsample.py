#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np


def voxel_downsample(points, voxel_size):
    """
    使用体素网格降采样保留整体空间分布。
    points: (N, 4) 的 ndarray, 每行为 [x, y, z, intensity]
    voxel_size: float, 体素尺寸

    返回:
        downsampled_points: (M, 4) 降采样后的点云
    """
    if len(points) == 0:
        return points

    # （1）计算每个点所在的体素索引 (i, j, k)
    # floor(x / voxel_size), floor(y / voxel_size), floor(z / voxel_size)
    coords = points[:, :3]  # xyz
    voxel_indices = np.floor(coords / voxel_size).astype(np.int64)

    # （2）用字典将同一个体素内的点分组
    # key: (i, j, k)
    # value: list of point indices
    vox_dict = {}
    for idx, vox_idx in enumerate(voxel_indices):
        key = (vox_idx[0], vox_idx[1], vox_idx[2])
        if key not in vox_dict:
            vox_dict[key] = []
        vox_dict[key].append(idx)

    # （3）对每个体素内的点计算质心(或者根据需求选“任意一个原始点”等)
    downsampled_list = []
    for _, indices in vox_dict.items():
        # 当前体素内所有点
        pts_in_voxel = points[indices]
        # 质心（xyz）
        mean_xyz = np.mean(pts_in_voxel[:, :3], axis=0)
        # 强度也可做平均, 或保留第一个点强度等，这里演示取平均值
        mean_intensity = np.mean(pts_in_voxel[:, 3], axis=0)

        new_point = np.concatenate([mean_xyz, [mean_intensity]])
        downsampled_list.append(new_point)

    downsampled_points = np.array(downsampled_list)
    return downsampled_points


def detect_decimal_places(line):
    """
    简单探测单行中 xyz intensity 的小数位数（如果行的每列小数位不一致，这里只示范取“最大”小数位数）。
    line: 字符串，例如 "12.3456 78.9 0.123 255"

    返回: int, 检测到的最大小数位
    """
    parts = line.strip().split()
    decimal_places = 0
    for p in parts:
        if '.' in p:
            # 以最后一个'.'分割来统计小数长度
            frac_len = len(p.split('.')[-1])
            decimal_places = max(decimal_places, frac_len)
    return decimal_places


def format_point(pt, decimal_places):
    """
    将 [x, y, z, intensity] 用给定的小数位数格式化为字符串。
    """
    fmt = f"{{:.{decimal_places}f}}"
    return " ".join(fmt.format(v) for v in pt)


def downsample_txt_file(input_file, output_file, voxel_size=0.1, decimal_places=None):
    """
    对单个 txt 文件做体素降采样并写出到 output_file。

    参数：
        input_file (str)  : 输入 txt 文件路径，每行 [x, y, z, intensity]
        output_file (str) : 输出 txt 文件路径
        voxel_size (float): 体素大小
        decimal_places (int|None): 小数位数。如果 None，则自动从第一行中检测。
    """
    if not os.path.exists(input_file):
        print(f"[警告] 输入文件不存在: {input_file}")
        return

    with open(input_file, 'r') as f:
        lines = [ln for ln in f.readlines() if ln.strip()]  # 跳过空行

    if len(lines) == 0:
        print(f"[警告] 文件为空: {input_file}")
        return

    # 如果用户没有指定 decimal_places，则用第一行数据自动检测一下
    if decimal_places is None:
        decimal_places = detect_decimal_places(lines[0])

    # 解析 points (N, 4)
    points = []
    for ln_num, ln in enumerate(lines, start=1):
        parts = ln.strip().split()
        if len(parts) < 4:
            # 若行数据不完整, 跳过
            print(f"[警告] 行 {ln_num} 数据不完整，已跳过。")
            continue
        x, y, z, i = parts[:4]  # 只取前四个部分
        try:
            points.append([float(x), float(y), float(z), float(i)])
        except ValueError:
            # 若转换失败, 跳过
            print(f"[警告] 行 {ln_num} 数据格式错误，已跳过。")
            continue
    points = np.array(points, dtype=np.float32)  # (N, 4)

    # 做体素降采样
    downsampled_points = voxel_downsample(points, voxel_size)

    # 输出
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f_out:
        for pt in downsampled_points:
            f_out.write(format_point(pt, decimal_places) + "\n")


def process_annotations_folder(annotations_path, output_annotations_path, voxel_size, decimal_places, area_name, tunnel_name):
    """
    处理一个 Annotations 文件夹下的所有 .txt 文件。
    """
    if not os.path.exists(annotations_path):
        print(f"[警告] Annotations 文件夹不存在: {annotations_path}")
        return

    os.makedirs(output_annotations_path, exist_ok=True)

    txt_files = [f for f in os.listdir(annotations_path) if f.endswith(".txt")]
    total_files = len(txt_files)
    if total_files == 0:
        print(f"[信息] 在 {annotations_path} 中未找到任何 .txt 文件。")
        return

    print(f"开始处理 Area: '{area_name}' 下的 Tunnel: '{tunnel_name}'，共 {total_files} 个文件。")

    for idx, txt_name in enumerate(txt_files, start=1):
        txt_in_path = os.path.join(annotations_path, txt_name)
        txt_out_path = os.path.join(output_annotations_path, txt_name)

        print(f"  [{idx}/{total_files}] 处理文件: {txt_name}")
        # 对每个 txt 文件做降采样并输出
        downsample_txt_file(
            input_file=txt_in_path,
            output_file=txt_out_path,
            voxel_size=voxel_size,
            decimal_places=decimal_places
        )

    print(f"完成处理 Area: '{area_name}' 下的 Tunnel: '{tunnel_name}'。\n")


def main():
    # ----------------------#
    # 根据需要自行修改以下变量
    # ----------------------#
    input_root_dir = r"/media/jiang/Accept/##DATA/Tunnel_Data/Dataset/S3DIS_leakage/s3dis_leakage_original_txt"  # 输入根目录（包含多个 Area_i 文件夹）
    output_root_dir = r"/media/jiang/Accept/##DATA/Tunnel_Data/Dataset/S3DIS_leakage/s3dis_leakage_downsampled_6"  # 输出根目录
    voxel_size = 0.01  # 体素大小，可根据数据范围调整
    # 若你想手动指定统一小数位数，可自行改为整数，比如 decimals=3；若 None 则自动探测。
    decimals = None

    if not os.path.exists(input_root_dir):
        print(f"[错误] 输入根目录不存在: {input_root_dir}")
        return

    area_folders = [f for f in os.listdir(input_root_dir) if os.path.isdir(os.path.join(input_root_dir, f))]
    total_areas = len(area_folders)
    if total_areas == 0:
        print(f"[警告] 在输入根目录中未找到任何 Area_i 文件夹。")
        return

    print(f"开始处理所有 Area 文件夹，共 {total_areas} 个 Area。\n")

    # 遍历 input_root_dir 下所有 Area_i 文件夹
    for area_idx, area_name in enumerate(area_folders, start=1):
        area_path = os.path.join(input_root_dir, area_name)
        if not os.path.isdir(area_path):
            continue  # 跳过非文件夹

        print(f"[{area_idx}/{total_areas}] 正在处理 Area: '{area_name}'")

        # 遍历 Area_i 文件夹下的所有 tunnel_i 文件夹
        tunnel_folders = [f for f in os.listdir(area_path) if f.startswith("tunnel_") and os.path.isdir(os.path.join(area_path, f))]
        total_tunnels = len(tunnel_folders)
        if total_tunnels == 0:
            print(f"  [警告] Area '{area_name}' 中未找到任何 tunnel_i 文件夹。\n")
            continue

        for tunnel_idx, folder_name in enumerate(tunnel_folders, start=1):
            tunnel_path = os.path.join(area_path, folder_name)
            # Annotations 子文件夹
            annotations_path = os.path.join(tunnel_path, "Annotations")
            if not os.path.isdir(annotations_path):
                print(f"  [警告] Annotations 文件夹不存在: {annotations_path}")
                continue

            # 构造对应的输出 Annotations 路径：output_root_dir/Area_i/tunnel_i/Annotations/
            output_annotations_path = os.path.join(output_root_dir, area_name, folder_name, "Annotations")

            # 处理该 Annotations 文件夹下的所有 .txt 文件
            process_annotations_folder(
                annotations_path=annotations_path,
                output_annotations_path=output_annotations_path,
                voxel_size=voxel_size,
                decimal_places=decimals,
                area_name=area_name,
                tunnel_name=folder_name
            )

    print("全部处理完成! 所有文件已处理并输出到:", output_root_dir)


if __name__ == "__main__":
    main()

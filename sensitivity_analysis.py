import argparse
import sys
import os
import csv
import numpy as np
import logging

# 确保 src 目录在 python 路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.io_loader import PDBLoader
from src.surface import SurfaceGenerator
from src.topology import TopologyManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Sensitivity")

def run_sensitivity(pdb_path, chain_a, chain_b, output_dir="sensitivity_results"):
    """
    运行参数敏感性分析，记录总界面面积。
    结果保存为 CSV 文件，文件名包含 PDB ID。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    logger.info(f"[{pdb_id}] 正在加载结构...")

    # 1. 加载 PDB 数据
    try:
        loader = PDBLoader(pdb_path)
        coords_A, _ = loader.get_chain_data(chain_a)
        coords_B, _ = loader.get_chain_data(chain_b)
    except Exception as e:
        logger.error(f"[{pdb_id}] 跳过 - 加载失败或链 {chain_a}/{chain_b} 不存在: {e}")
        return

    # 初始化 SurfaceGenerator
    surf_gen = SurfaceGenerator(coords_A)

    # ==========================================
    # 实验 1: 改变 Cutoff (固定 Sigma = 1.5)
    # ==========================================
    logger.info(f"[{pdb_id}] 实验 1: 改变 Cutoff (固定 Sigma=1.5)")
    fixed_sigma = 1.5
    cutoff_range = np.arange(3.0, 8.5, 0.5) 
    
    mesh_A = surf_gen.generate_mesh(grid_resolution=1.0, sigma=fixed_sigma)
    results_cutoff = []
    
    if mesh_A:
        topo = TopologyManager(mesh_A, coords_B)
        for cutoff in cutoff_range:
            patches = topo.get_interface_patches(distance_cutoff=cutoff, min_patch_vertices=10)
            total_area = sum(p.area for p in patches) if patches else 0.0
            results_cutoff.append({"Cutoff": cutoff, "Total_Interface_Area": total_area})

    save_csv(results_cutoff, os.path.join(output_dir, f"{pdb_id}_sensitivity_cutoff.csv"))

    # ==========================================
    # 实验 2: 改变 Sigma (固定 Cutoff = 5.0)
    # ==========================================
    logger.info(f"[{pdb_id}] 实验 2: 改变 Sigma (固定 Cutoff=5.0)")
    fixed_cutoff = 5.0
    sigma_range = np.arange(0.5, 3.5, 0.5)
    
    results_sigma = []
    
    for sigma in sigma_range:
        mesh_A_sigma = surf_gen.generate_mesh(grid_resolution=1.0, sigma=sigma)
        if mesh_A_sigma:
            topo = TopologyManager(mesh_A_sigma, coords_B)
            patches = topo.get_interface_patches(distance_cutoff=fixed_cutoff, min_patch_vertices=10)
            total_area = sum(p.area for p in patches) if patches else 0.0
            results_sigma.append({"Sigma": sigma, "Total_Interface_Area": total_area})
        else:
            results_sigma.append({"Sigma": sigma, "Total_Interface_Area": 0.0})

    save_csv(results_sigma, os.path.join(output_dir, f"{pdb_id}_sensitivity_sigma.csv"))
    logger.info(f"[{pdb_id}] 完成。")

def save_csv(data_list, filepath):
    """辅助函数：保存列表到 CSV"""
    if not data_list:
        return
    keys = data_list[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TopoPPI 参数敏感性分析 (批量支持)")
    parser.add_argument("input_path", help="输入 PDB 文件路径 或 包含 PDB 的文件夹路径")
    parser.add_argument("-A", "--chain_a", required=True, help="链 A ID (所有文件共用)")
    parser.add_argument("-B", "--chain_b", required=True, help="链 B ID (所有文件共用)")
    parser.add_argument("-o", "--out_dir", default="sensitivity_results", help="输出文件夹")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"错误: 路径 {args.input_path} 不存在。")
        sys.exit(1)

    if os.path.isdir(args.input_path):
        # 文件夹模式
        pdb_files = [f for f in os.listdir(args.input_path) if f.lower().endswith('.pdb')]
        if not pdb_files:
            print(f"警告: 文件夹 {args.input_path} 中没有 .pdb 文件。")
            sys.exit(0)
            
        print(f"在文件夹中找到 {len(pdb_files)} 个 PDB 文件，开始批量处理...")
        for f in pdb_files:
            full_path = os.path.join(args.input_path, f)
            run_sensitivity(full_path, args.chain_a, args.chain_b, args.out_dir)
    else:
        # 单文件模式
        run_sensitivity(args.input_path, args.chain_a, args.chain_b, args.out_dir)

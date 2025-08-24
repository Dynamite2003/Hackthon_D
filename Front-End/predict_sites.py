"""
Li金属团簇位点预测主程序
提供命令行界面进行位点预测和优化
"""
import argparse
import numpy as np
import json
import os
import sys
from optimizer import SiteOptimizer

def parse_coordinates_from_string(coord_str):
    """
    从字符串解析坐标
    支持格式: "x1,y1,z1;x2,y2,z2;x3,y3,z3"
    """
    try:
        atoms = coord_str.strip().split(';')
        positions = []
        for atom in atoms:
            coords = [float(x.strip()) for x in atom.split(',')]
            if len(coords) != 3:
                raise ValueError(f"每个原子需要3个坐标值，获得了 {len(coords)} 个")
            positions.append(coords)
        return np.array(positions)
    except Exception as e:
        raise ValueError(f"坐标解析错误: {e}")

def parse_coordinates_from_file(file_path):
    """
    从文件读取坐标
    支持多种格式: CSV, XYZ, JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # CSV格式: x,y,z (每行一个原子)
        import pandas as pd
        df = pd.read_csv(file_path)
        if len(df.columns) < 3:
            raise ValueError("CSV文件需要至少3列坐标数据")
        return df.iloc[:, :3].values
        
    elif file_ext == '.xyz':
        # XYZ格式
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 跳过头两行（原子数和注释）
        positions = []
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 4:  # 元素符号 + 3个坐标
                coords = [float(parts[i]) for i in range(1, 4)]
                positions.append(coords)
        
        return np.array(positions)
        
    elif file_ext == '.json':
        # JSON格式
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'positions' in data:
            return np.array(data['positions'])
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("JSON文件格式不正确")
            
    else:
        # 尝试按纯文本格式解析（空格或逗号分隔）
        positions = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 尝试用逗号分隔，如果失败则用空格
                    if ',' in line:
                        coords = [float(x.strip()) for x in line.split(',')]
                    else:
                        coords = [float(x) for x in line.split()]
                    
                    if len(coords) >= 3:
                        positions.append(coords[:3])
        
        return np.array(positions)

def save_results_to_file(results, base_positions, output_path, format='json'):
    """
    保存结果到文件
    """
    output_data = {
        'base_structure': {
            'n_atoms': len(base_positions),
            'positions': base_positions.tolist()
        },
        'optimization_results': {
            'n_candidates_tested': results['n_candidates'],
            'best_sites': []
        }
    }
    
    for i, (site, energy) in enumerate(zip(results['best_sites'], results['best_energies'])):
        output_data['optimization_results']['best_sites'].append({
            'rank': i + 1,
            'position': site.tolist(),
            'predicted_energy_eV': float(energy),
            'structure_after_addition': np.vstack([base_positions, site]).tolist()
        })
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    print(f"结果已保存到: {output_path}")

def create_xyz_output(base_positions, best_sites, best_energies, output_path):
    """
    创建XYZ格式的输出文件，包含最佳位点
    """
    with open(output_path, 'w') as f:
        for i, (site, energy) in enumerate(zip(best_sites, best_energies)):
            # 创建包含新原子的完整结构
            full_structure = np.vstack([base_positions, site])
            n_atoms = len(full_structure)
            
            f.write(f"{n_atoms}\n")
            f.write(f"Li cluster with added atom #{i+1}, Energy: {energy:.6f} eV\n")
            
            # 写入原有原子
            for pos in base_positions:
                f.write(f"Li {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
            # 写入新原子
            f.write(f"Li {site[0]:.6f} {site[1]:.6f} {site[2]:.6f}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="Li金属团簇位点预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 直接输入坐标
  python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" --top-k 3

  # 从文件读取
  python predict_sites.py --file structure.xyz --strategy combined --top-k 5

  # 使用网格搜索
  python predict_sites.py --file structure.csv --strategy grid --grid-spacing 0.3

  # 保存详细结果
  python predict_sites.py --coords "0,0,0;2,0,0" --output results.json --save-xyz results.xyz
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--coords', type=str,
                           help='直接输入坐标，格式: "x1,y1,z1;x2,y2,z2;..."')
    input_group.add_argument('--file', type=str,
                           help='从文件读取坐标 (支持 .csv, .xyz, .json, .txt)')
    
    # 搜索策略选项
    parser.add_argument('--strategy', choices=['random', 'grid', 'spherical', 'neighbor', 'surface', 'surface_shell', 'combined', 'basin_hopping'],
                       default='combined', help='位点生成策略 (默认: combined)')
    parser.add_argument('--top-k', type=int, default=1,
                       help='返回前k个最佳位点 (默认: 1)')
    
    # 策略参数
    parser.add_argument('--n-random', type=int, default=100,
                       help='随机位点数量 (默认: 100)')
    parser.add_argument('--grid-spacing', type=float, default=0.5,
                       help='网格间距 (Å) (默认: 0.5)')
    parser.add_argument('--n-radial', type=int, default=5,
                       help='球面搜索径向层数 (默认: 5)')
    parser.add_argument('--n-angular', type=int, default=20,
                       help='球面搜索每层角度点数 (默认: 20)')
    
    # 物理约束
    parser.add_argument('--min-distance', type=float, default=1.0,
                       help='最小原子间距 (Å) (默认: 1.0)')
    parser.add_argument('--max-distance', type=float, default=4.0,
                       help='最大搜索距离 (Å) (默认: 4.0)')
    
    # 输出选项
    parser.add_argument('--output', type=str,
                       help='保存JSON格式结果文件路径')
    parser.add_argument('--save-xyz', type=str,
                       help='保存XYZ格式结构文件路径')
    parser.add_argument('--save-plot', type=str,
                       help='保存可视化图片路径')
    parser.add_argument('--no-plot', action='store_true',
                       help='不显示可视化图片')
    parser.add_argument('--molecular-view', action='store_true', default=True,
                       help='使用分子可视化视图显示真实原子团簇 (默认)')
    parser.add_argument('--traditional-view', action='store_true',
                       help='使用传统的点状可视化视图')
    parser.add_argument('--advanced-renderer', action='store_true', default=True,
                       help='使用高质量Plotly渲染器 (无化学键，完美球体) (默认)')
    parser.add_argument('--basic-renderer', action='store_true',
                       help='使用基础matplotlib渲染器 (带化学键)')
    parser.add_argument('--visualization-type', choices=['space_filling', 'ball_stick'], 
                       default='space_filling', help='可视化模型类型 (默认: space_filling)')
    
    # 高级选项
    parser.add_argument('--iterative', action='store_true',
                       help='使用迭代优化 (更精确但较慢)')
    parser.add_argument('--n-iterations', type=int, default=3,
                       help='迭代次数 (默认: 3)')
    
    args = parser.parse_args()
    
    # 解析输入坐标
    try:
        if args.coords:
            print("解析输入坐标...")
            base_positions = parse_coordinates_from_string(args.coords)
        else:
            print(f"从文件读取坐标: {args.file}")
            base_positions = parse_coordinates_from_file(args.file)
        
        print(f"成功读取 {len(base_positions)} 个Li原子的坐标")
        print("基础结构:")
        for i, pos in enumerate(base_positions):
            print(f"  原子 {i+1}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
    except Exception as e:
        print(f"错误: 无法解析坐标 - {e}")
        sys.exit(1)
    
    # 初始化优化器
    try:
        print("\n初始化模型和优化器...")
        optimizer = SiteOptimizer(
            min_distance=args.min_distance,
            max_distance=args.max_distance
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"错误: 模型加载失败 - {e}")
        sys.exit(1)
    
    # 执行位点优化
    try:
        print(f"\n开始位点搜索 (策略: {args.strategy})...")
        
        if args.iterative:
            print("使用迭代优化模式...")
            final_results, all_iterations = optimizer.optimize_iteratively(
                base_positions,
                n_iterations=args.n_iterations,
                candidates_per_iteration=args.n_random,
                top_k=args.top_k
            )
            results = final_results
        elif args.strategy == 'basin_hopping':
            results = optimizer.find_sites_with_basinhopping(
                base_positions,
                top_k=args.top_k
            )
        else:
            # 构建策略参数
            strategy_params = {}
            if args.strategy == 'random':
                strategy_params['n_candidates'] = args.n_random
            elif args.strategy == 'grid':
                strategy_params['grid_spacing'] = args.grid_spacing
            elif args.strategy == 'spherical':
                strategy_params['n_radial'] = args.n_radial
                strategy_params['n_angular'] = args.n_angular
            elif args.strategy == 'combined':
                strategy_params['n_random'] = args.n_random
                strategy_params['grid_spacing'] = args.grid_spacing
            
            results = optimizer.find_best_sites(
                base_positions,
                strategy=args.strategy,
                top_k=args.top_k,
                **strategy_params
            )
        
    except Exception as e:
        print(f"错误: 位点优化失败 - {e}")
        sys.exit(1)
    
    # 显示结果
    print(f"\n{'='*60}")
    print("位点优化完成!")
    print(f"{'='*60}")
    print(f"测试了 {results['n_candidates']} 个候选位点")
    print(f"找到前 {len(results['best_sites'])} 个最佳位点:\n")
    
    for i, (site, energy) in enumerate(zip(results['best_sites'], results['best_energies'])):
        # 安全转换坐标和能量
        if hasattr(site, 'cpu'):  # PyTorch tensor
            site = site.cpu().numpy()
        site = np.array(site).flatten()
        
        x = float(site[0])
        y = float(site[1])
        z = float(site[2])
        
        if hasattr(energy, 'cpu'):  # PyTorch tensor
            energy = energy.cpu().numpy()
        if hasattr(energy, 'item'):  # numpy scalar
            energy_val = energy.item()
        else:
            energy_val = float(energy)
            
        # 同样处理最佳能量
        best_energy = results['best_energies'][0]
        if hasattr(best_energy, 'cpu'):
            best_energy = best_energy.cpu().numpy()
        if hasattr(best_energy, 'item'):
            best_energy_val = best_energy.item()
        else:
            best_energy_val = float(best_energy)
        
        print(f"位点 #{i+1}:")
        print(f"  坐标: ({x:.4f}, {y:.4f}, {z:.4f}) Å")
        print(f"  预测能量: {energy_val:.6f} eV")
        print(f"  相对最佳能量: +{energy_val - best_energy_val:.6f} eV")
        print()
    
    # 保存结果
    if args.output:
        save_results_to_file(results, base_positions, args.output)
    
    if args.save_xyz:
        create_xyz_output(base_positions, results['best_sites'], 
                         results['best_energies'], args.save_xyz)
        print(f"XYZ结构文件已保存到: {args.save_xyz}")
    
    # 可视化
    if not args.no_plot:
        try:
            print("\n生成可视化...")
            # 确定使用哪种可视化方式
            use_molecular = not args.traditional_view  # 默认使用分子视图，除非指定传统视图
            use_advanced = not args.basic_renderer  # 默认使用高质量渲染器，除非指定基础渲染器
            
            optimizer.visualize_results(
                results, 
                save_path=args.save_plot,
                show_plot=(args.save_plot is None),  # 如果不保存则显示
                use_molecular_view=use_molecular,
                use_advanced_renderer=use_advanced,
                visualization_type=args.visualization_type
            )
            
            if use_molecular:
                if use_advanced:
                    if args.visualization_type == 'ball_stick':
                        print("✓ 使用高质量球棍模型可视化 (py3Dmol渲染，带化学键)")
                    else:
                        print("✓ 使用高质量空间填充模型可视化 (py3Dmol渲染，完美球体)")
                else:
                    print("✓ 使用基础分子可视化 (matplotlib渲染，带化学键)")
            else:
                print("✓ 使用传统可视化视图 (点状表示)")
                
        except Exception as e:
            print(f"警告: 可视化失败 - {e}")
    
    print("\n位点预测完成!")

if __name__ == "__main__":
    main()
"""
Li金属团簇表面识别模块
用于识别团簇表面原子和生成表面附近的候选位点
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, distance_matrix
# from sklearn.neighbors import NearestNeighbors  # 不需要sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    # 尝试相对导入（当作为模块使用时）
    from ..visualization_tools.molecular_visualizer import MolecularVisualizer
except ImportError:
    # 回退到绝对导入（当作为脚本直接运行时）
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    from visualization_tools.molecular_visualizer import MolecularVisualizer

class SurfaceDetector:
    def __init__(self, coordination_threshold=12, surface_shell_thickness=2.0):
        """
        初始化表面探测器
        
        参数:
        coordination_threshold: int, 配位数阈值，低于此值的原子被认为是表面原子
        surface_shell_thickness: float, 表面壳层厚度 (Angstrom)
        """
        self.coordination_threshold = coordination_threshold
        self.surface_shell_thickness = surface_shell_thickness
        
    def identify_surface_atoms_by_coordination(self, positions, cutoff_distance=3.5):
        """
        基于配位数识别表面原子
        
        参数:
        positions: np.array, 原子坐标 (n_atoms, 3)
        cutoff_distance: float, 配位数计算的截止距离 (Angstrom)
        
        返回:
        list: 表面原子的索引
        """
        n_atoms = len(positions)
        distances = cdist(positions, positions)
        
        surface_atom_indices = []
        coordination_numbers = []
        
        for i in range(n_atoms):
            # 计算配位数（距离小于cutoff_distance的邻居数量，排除自身）
            neighbors = np.sum((distances[i] < cutoff_distance) & (distances[i] > 0))
            coordination_numbers.append(neighbors)
            
            # 配位数低的原子更可能是表面原子
            if neighbors < self.coordination_threshold:
                surface_atom_indices.append(i)
        
        return surface_atom_indices, coordination_numbers
    
    def identify_surface_atoms_by_convex_hull(self, positions, tolerance=0.1):
        """
        基于凸包识别表面原子
        
        参数:
        positions: np.array, 原子坐标 (n_atoms, 3)
        tolerance: float, 距离凸包表面的容差 (Angstrom)
        
        返回:
        list: 表面原子的索引
        """
        try:
            # 计算凸包
            hull = ConvexHull(positions)
            
            # 凸包顶点就是表面原子
            hull_vertices = set(hull.vertices)
            
            # 额外检查：距离凸包表面很近的原子也可能是表面原子
            surface_atom_indices = list(hull_vertices)
            
            # 计算每个原子到凸包表面的距离
            for i, pos in enumerate(positions):
                if i not in hull_vertices:
                    min_dist_to_hull = float('inf')
                    
                    # 计算到每个凸包面的距离
                    for simplex in hull.simplices:
                        # 获取三角面的三个顶点
                        face_points = positions[simplex]
                        
                        # 计算点到平面的距离
                        v0, v1, v2 = face_points
                        normal = np.cross(v1 - v0, v2 - v0)
                        normal = normal / np.linalg.norm(normal)
                        
                        # 点到平面的距离
                        dist = abs(np.dot(pos - v0, normal))
                        min_dist_to_hull = min(min_dist_to_hull, dist)
                    
                    if min_dist_to_hull <= tolerance:
                        surface_atom_indices.append(i)
            
            return sorted(surface_atom_indices)
            
        except Exception as e:
            print(f"凸包计算失败: {e}")
            # 回退到配位数方法
            surface_atoms, _ = self.identify_surface_atoms_by_coordination(positions)
            return surface_atoms
    
    def identify_surface_atoms_hybrid(self, positions, cutoff_distance=3.5):
        """
        混合方法：结合配位数和凸包识别表面原子
        
        参数:
        positions: np.array, 原子坐标 (n_atoms, 3)
        cutoff_distance: float, 配位数计算的截止距离
        
        返回:
        list: 表面原子的索引
        dict: 详细信息
        """
        # 方法1：配位数
        coord_surface, coordination_numbers = self.identify_surface_atoms_by_coordination(
            positions, cutoff_distance)
        
        # 方法2：凸包
        hull_surface = self.identify_surface_atoms_by_convex_hull(positions)
        
        # 合并两种方法的结果
        all_surface = sorted(set(coord_surface) | set(hull_surface))
        
        info = {
            'coordination_surface': coord_surface,
            'hull_surface': hull_surface,
            'hybrid_surface': all_surface,
            'coordination_numbers': coordination_numbers
        }
        
        return all_surface, info
    
    def generate_surface_sites(self, positions, surface_atom_indices, 
                             n_sites_per_atom=8, radius_range=(1.5, 3.0)):
        """
        在表面原子周围生成候选位点
        
        参数:
        positions: np.array, 原子坐标 (n_atoms, 3)
        surface_atom_indices: list, 表面原子索引
        n_sites_per_atom: int, 每个表面原子周围的候选位点数
        radius_range: tuple, 候选位点距离表面原子的距离范围 (min, max)
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        candidate_sites = []
        
        for atom_idx in surface_atom_indices:
            atom_pos = positions[atom_idx]
            
            # 在每个表面原子周围生成候选位点
            for _ in range(n_sites_per_atom):
                # 随机生成方向
                direction = np.random.normal(0, 1, 3)
                direction = direction / np.linalg.norm(direction)
                
                # 随机生成距离
                distance = np.random.uniform(radius_range[0], radius_range[1])
                
                # 生成候选位点
                candidate_site = atom_pos + direction * distance
                
                # 检查候选位点是否距离其他原子太近
                distances_to_atoms = np.linalg.norm(positions - candidate_site, axis=1)
                
                # 确保新位点不与现有原子重叠
                if np.min(distances_to_atoms) >= 1.0:  # 最小距离1.0 Å
                    candidate_sites.append(candidate_site)
        
        return np.array(candidate_sites) if candidate_sites else np.empty((0, 3))
    
    def generate_surface_shell_sites(self, positions, shell_thickness=None, 
                                   n_points=200, method='random'):
        """
        在团簇表面壳层内生成候选位点
        
        参数:
        positions: np.array, 原子坐标
        shell_thickness: float, 壳层厚度，None使用默认值
        n_points: int, 生成的点数
        method: str, 生成方法 ('random', 'grid', 'fibonacci')
        
        返回:
        np.array: 候选位点坐标
        """
        if shell_thickness is None:
            shell_thickness = self.surface_shell_thickness
        
        # 计算团簇的几何中心和边界
        center = np.mean(positions, axis=0)
        distances_from_center = np.linalg.norm(positions - center, axis=1)
        
        # 计算团簇的"半径"
        max_radius = np.max(distances_from_center)
        min_radius = max_radius - shell_thickness
        
        candidate_sites = []
        
        if method == 'random':
            attempts = 0
            max_attempts = n_points * 10
            
            while len(candidate_sites) < n_points and attempts < max_attempts:
                # 在球壳内随机生成点
                direction = np.random.normal(0, 1, 3)
                direction = direction / np.linalg.norm(direction)
                
                radius = np.random.uniform(min_radius, max_radius)
                candidate = center + direction * radius
                
                # 检查点是否在合理位置
                distances_to_atoms = np.linalg.norm(positions - candidate, axis=1)
                
                # 确保点在表面附近但不与原子重叠
                if (np.min(distances_to_atoms) >= 1.0 and 
                    np.min(distances_to_atoms) <= shell_thickness + 1.0):
                    candidate_sites.append(candidate)
                
                attempts += 1
        
        elif method == 'fibonacci':
            # 使用斐波那契螺旋在球面上均匀分布点
            candidate_sites = self._fibonacci_sphere_points(
                center, max_radius, n_points, shell_thickness)
        
        return np.array(candidate_sites) if candidate_sites else np.empty((0, 3))
    
    def _fibonacci_sphere_points(self, center, radius, n_points, shell_thickness):
        """
        使用斐波那契螺旋在球面壳层上生成均匀分布的点
        """
        points = []
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(n_points):
            # 极角和方位角
            theta = 2 * np.pi * i / golden_ratio
            phi = np.arccos(1 - 2 * i / n_points)
            
            # 在壳层内随机选择径向距离
            r = np.random.uniform(radius - shell_thickness, radius)
            
            # 球坐标转笛卡尔坐标
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            point = center + np.array([x, y, z])
            points.append(point)
        
        return points
    
    def visualize_surface_analysis(self, positions, surface_info, save_path=None):
        """
        可视化表面分析结果
        
        参数:
        positions: np.array, 原子坐标
        surface_info: dict, 表面分析信息
        save_path: str, 保存路径
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 3D可视化
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 绘制所有原子
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='lightblue', s=100, alpha=0.6, label='Bulk atoms')
        
        # 高亮表面原子
        surface_atoms = surface_info['hybrid_surface']
        if surface_atoms:
            surface_pos = positions[surface_atoms]
            ax1.scatter(surface_pos[:, 0], surface_pos[:, 1], surface_pos[:, 2],
                       c='red', s=150, alpha=0.8, label='Surface atoms')
        
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title('Surface Atom Identification')
        ax1.legend()
        
        # 配位数分布
        ax2 = fig.add_subplot(222)
        coordination_nums = surface_info['coordination_numbers']
        ax2.hist(coordination_nums, bins=range(min(coordination_nums), 
                                             max(coordination_nums)+2), 
                alpha=0.7, edgecolor='black')
        ax2.axvline(self.coordination_threshold, color='red', linestyle='--',
                   label=f'Surface threshold ({self.coordination_threshold})')
        ax2.set_xlabel('Coordination Number')
        ax2.set_ylabel('Count')
        ax2.set_title('Coordination Number Distribution')
        ax2.legend()
        
        # 方法比较
        ax3 = fig.add_subplot(223)
        methods = ['Coordination', 'ConvexHull', 'Hybrid']
        counts = [len(surface_info['coordination_surface']),
                 len(surface_info['hull_surface']),
                 len(surface_info['hybrid_surface'])]
        
        bars = ax3.bar(methods, counts, alpha=0.7, 
                      color=['blue', 'green', 'orange'])
        ax3.set_ylabel('Number of Surface Atoms')
        ax3.set_title('Surface Detection Method Comparison')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 原子索引标注
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='lightblue', s=100, alpha=0.6)
        
        # 标注表面原子索引
        for idx in surface_atoms:
            pos = positions[idx]
            ax4.text(pos[0], pos[1], pos[2], f'{idx}', 
                    fontsize=10, color='red', weight='bold')
        
        ax4.set_xlabel('X (Å)')
        ax4.set_ylabel('Y (Å)')
        ax4.set_zlabel('Z (Å)')
        ax4.set_title('Surface Atom Indices')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"表面分析图已保存到: {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # 测试表面识别功能
    detector = SurfaceDetector()
    
    # 加载Li8测试结构
    li8_coords = np.array([
        [-1.7138, -2.5707, 2.1413],
        [-3.4276, -0.8569, 0.4275],
        [0.0000, -0.8569, 0.4275],
        [0.0000, -4.2826, 0.4275],
        [1.7138, 0.8569, -1.2844],
        [3.4276, 2.5707, 0.4275],
        [0.0000, 2.5707, 0.4275],
        [0.0000, 2.5707, -2.9982]
    ])
    
    print("=== Li8团簇表面分析 ===")
    print(f"团簇包含 {len(li8_coords)} 个原子")
    
    # 识别表面原子
    surface_atoms, info = detector.identify_surface_atoms_hybrid(li8_coords)
    
    print(f"\n表面原子识别结果:")
    print(f"配位数方法识别的表面原子: {info['coordination_surface']}")
    print(f"凸包方法识别的表面原子: {info['hull_surface']}")
    print(f"混合方法识别的表面原子: {info['hybrid_surface']}")
    
    print(f"\n配位数信息:")
    for i, coord_num in enumerate(info['coordination_numbers']):
        atom_type = "表面" if i in surface_atoms else "内部"
        print(f"原子{i}: 配位数={coord_num} ({atom_type})")
    
    # 生成表面候选位点
    print(f"\n=== 生成表面候选位点 ===")
    surface_sites = detector.generate_surface_sites(
        li8_coords, surface_atoms, n_sites_per_atom=6)
    
    print(f"基于表面原子生成了 {len(surface_sites)} 个候选位点")
    
    # 生成壳层候选位点
    shell_sites = detector.generate_surface_shell_sites(
        li8_coords, shell_thickness=2.0, n_points=50)
    
    print(f"基于表面壳层生成了 {len(shell_sites)} 个候选位点")
    
    # 可视化结果
    detector.visualize_surface_analysis(li8_coords, info, 
                                      save_path="li8_surface_analysis.png")
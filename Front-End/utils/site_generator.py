"""
Li金属团簇位点生成器
实现多种策略生成候选原子位置
"""
import numpy as np
from scipy.spatial.distance import cdist
import itertools
try:
    # 尝试相对导入（当作为模块使用时）
    from .surface_detector import SurfaceDetector
except ImportError:
    # 回退到绝对导入（当作为脚本直接运行时）
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    from utils.surface_detector import SurfaceDetector

class SiteGenerator:
    def __init__(self, min_distance=1.0, max_distance=4.0):
        """
        初始化位点生成器
        
        参数:
        min_distance: float, 新原子与现有原子的最小距离 (Angstrom)
        max_distance: float, 新原子与现有原子的最大距离 (Angstrom)
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.surface_detector = SurfaceDetector()
    
    def generate_random_sites(self, base_positions, n_sites=100, seed=None):
        """
        随机生成候选位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        n_sites: int, 生成的候选位点数量
        seed: int, 随机种子
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 计算基础结构的边界
        min_coords = np.min(base_positions, axis=0)
        max_coords = np.max(base_positions, axis=0)
        
        # 扩展搜索空间
        expansion = self.max_distance
        min_coords -= expansion
        max_coords += expansion
        
        valid_sites = []
        attempts = 0
        max_attempts = n_sites * 100  # 防止无限循环
        
        while len(valid_sites) < n_sites and attempts < max_attempts:
            # 随机生成位点
            candidate = np.random.uniform(min_coords, max_coords)
            
            # 检查距离约束
            distances = np.linalg.norm(base_positions - candidate, axis=1)
            
            if (np.min(distances) >= self.min_distance and 
                np.min(distances) <= self.max_distance):
                valid_sites.append(candidate)
            
            attempts += 1
        
        if len(valid_sites) < n_sites:
            print(f"警告: 只生成了 {len(valid_sites)} 个有效位点，目标是 {n_sites} 个")
        
        return np.array(valid_sites)
    
    def generate_grid_sites(self, base_positions, grid_spacing=0.5):
        """
        基于网格生成候选位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        grid_spacing: float, 网格间距 (Angstrom)
        
        返回:
        np.array: 候选位点坐标 (n_valid_sites, 3)
        """
        # 计算搜索范围
        min_coords = np.min(base_positions, axis=0) - self.max_distance
        max_coords = np.max(base_positions, axis=0) + self.max_distance
        
        # 生成网格点
        x_points = np.arange(min_coords[0], max_coords[0] + grid_spacing, grid_spacing)
        y_points = np.arange(min_coords[1], max_coords[1] + grid_spacing, grid_spacing)
        z_points = np.arange(min_coords[2], max_coords[2] + grid_spacing, grid_spacing)
        
        # 生成所有网格点的组合
        grid_points = np.array(list(itertools.product(x_points, y_points, z_points)))
        
        # 筛选有效位点
        valid_sites = []
        
        for point in grid_points:
            distances = np.linalg.norm(base_positions - point, axis=1)
            
            if (np.min(distances) >= self.min_distance and 
                np.min(distances) <= self.max_distance):
                valid_sites.append(point)
        
        return np.array(valid_sites) if valid_sites else np.empty((0, 3))
    
    def generate_spherical_sites(self, base_positions, target_atom_idx=None, 
                                 n_radial=5, n_angular=20):
        """
        基于球面坐标系生成位点（围绕特定原子或团簇中心）
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        target_atom_idx: int or None, 目标原子索引，None则使用几何中心
        n_radial: int, 径向层数
        n_angular: int, 每层的角度采样点数
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        if target_atom_idx is not None:
            center = base_positions[target_atom_idx]
        else:
            center = np.mean(base_positions, axis=0)
        
        valid_sites = []
        
        # 径向距离分布
        radial_distances = np.linspace(self.min_distance, self.max_distance, n_radial)
        
        for r in radial_distances:
            # 生成球面上的均匀分布点
            phi = np.random.uniform(0, 2*np.pi, n_angular)  # 方位角
            theta = np.arccos(np.random.uniform(-1, 1, n_angular))  # 极角
            
            for i in range(n_angular):
                # 球面坐标到笛卡尔坐标转换
                x = r * np.sin(theta[i]) * np.cos(phi[i])
                y = r * np.sin(theta[i]) * np.sin(phi[i])
                z = r * np.cos(theta[i])
                
                candidate = center + np.array([x, y, z])
                
                # 检查与所有现有原子的距离
                distances = np.linalg.norm(base_positions - candidate, axis=1)
                
                if np.min(distances) >= self.min_distance:
                    valid_sites.append(candidate)
        
        return np.array(valid_sites) if valid_sites else np.empty((0, 3))
    
    def generate_neighbor_sites(self, base_positions, n_neighbors=3, offset_distance=None):
        """
        基于现有原子邻居关系生成位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        n_neighbors: int, 考虑的最近邻数量
        offset_distance: float, 偏移距离，None则使用平均键长
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        if len(base_positions) < 2:
            return np.empty((0, 3))
        
        # 计算所有原子间距离
        distances = cdist(base_positions, base_positions)
        
        # 计算平均键长
        if offset_distance is None:
            # 排除自身距离(0)，取最小非零距离作为键长估计
            non_zero_distances = distances[distances > 0]
            if len(non_zero_distances) > 0:
                offset_distance = np.mean(non_zero_distances[:min(len(non_zero_distances), 
                                                                 len(base_positions))])
            else:
                offset_distance = self.min_distance
        
        valid_sites = []
        
        for i, atom_pos in enumerate(base_positions):
            # 找到最近的n_neighbors个邻居
            atom_distances = distances[i]
            neighbor_indices = np.argsort(atom_distances)[1:n_neighbors+1]  # 排除自身
            
            for j in neighbor_indices:
                if j < len(base_positions):
                    neighbor_pos = base_positions[j]
                    
                    # 计算键向量
                    bond_vector = neighbor_pos - atom_pos
                    bond_length = np.linalg.norm(bond_vector)
                    
                    if bond_length > 0:
                        # 归一化键向量
                        bond_unit = bond_vector / bond_length
                        
                        # 在键的延长线上生成位点
                        candidate1 = atom_pos - bond_unit * offset_distance
                        candidate2 = neighbor_pos + bond_unit * offset_distance
                        
                        for candidate in [candidate1, candidate2]:
                            # 检查距离约束
                            distances_to_all = np.linalg.norm(base_positions - candidate, axis=1)
                            
                            if (np.min(distances_to_all) >= self.min_distance and
                                np.min(distances_to_all) <= self.max_distance):
                                valid_sites.append(candidate)
        
        # 去除重复的位点
        if valid_sites:
            valid_sites = np.array(valid_sites)
            # 简单的去重：如果两个点距离小于0.1Å则认为是重复的
            unique_sites = []
            for site in valid_sites:
                is_duplicate = False
                for unique_site in unique_sites:
                    if np.linalg.norm(site - unique_site) < 0.1:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_sites.append(site)
            return np.array(unique_sites) if unique_sites else np.empty((0, 3))
        
        return np.empty((0, 3))
    
    def generate_combined_sites(self, base_positions, n_random=50, grid_spacing=0.8):
        """
        结合多种策略生成候选位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        n_random: int, 随机位点数量
        grid_spacing: float, 网格间距
        
        返回:
        np.array: 候选位点坐标 (n_total_sites, 3)
        """
        all_sites = []
        
        # 随机位点
        random_sites = self.generate_random_sites(base_positions, n_random)
        if len(random_sites) > 0:
            all_sites.append(random_sites)
        
        # 球面位点（围绕几何中心）
        spherical_sites = self.generate_spherical_sites(base_positions, 
                                                       target_atom_idx=None,
                                                       n_radial=3, n_angular=12)
        if len(spherical_sites) > 0:
            all_sites.append(spherical_sites)
        
        # 邻居位点
        neighbor_sites = self.generate_neighbor_sites(base_positions)
        if len(neighbor_sites) > 0:
            all_sites.append(neighbor_sites)
        
        if all_sites:
            # 合并所有位点
            combined_sites = np.vstack(all_sites)
            
            # 简单去重 - 放宽阈值避免过度去重
            unique_sites = []
            for site in combined_sites:
                is_duplicate = False
                for unique_site in unique_sites:
                    if np.linalg.norm(site - unique_site) < 0.1:  # 从0.2降到0.1
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_sites.append(site)
            
            return np.array(unique_sites) if unique_sites else np.empty((0, 3))
        
        return np.empty((0, 3))
    
    def generate_surface_sites(self, base_positions, n_sites_per_surface=8, 
                             surface_method='hybrid', radius_range=(1.5, 3.0)):
        """
        基于表面原子生成候选位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        n_sites_per_surface: int, 每个表面原子周围的候选位点数
        surface_method: str, 表面识别方法 ('coordination', 'hull', 'hybrid')
        radius_range: tuple, 候选位点距离表面原子的距离范围
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        # 识别表面原子
        if surface_method == 'coordination':
            surface_atoms, _ = self.surface_detector.identify_surface_atoms_by_coordination(base_positions)
        elif surface_method == 'hull':
            surface_atoms = self.surface_detector.identify_surface_atoms_by_convex_hull(base_positions)
        else:  # hybrid
            surface_atoms, _ = self.surface_detector.identify_surface_atoms_hybrid(base_positions)
        
        if not surface_atoms:
            return np.empty((0, 3))
        
        print(f"识别到 {len(surface_atoms)} 个表面原子: {surface_atoms}")
        
        # 在表面原子周围生成候选位点
        candidate_sites = self.surface_detector.generate_surface_sites(
            base_positions, surface_atoms, 
            n_sites_per_atom=n_sites_per_surface,
            radius_range=radius_range
        )
        
        # 应用距离约束筛选
        valid_sites = []
        for site in candidate_sites:
            distances = np.linalg.norm(base_positions - site, axis=1)
            if (np.min(distances) >= self.min_distance and 
                np.min(distances) <= self.max_distance):
                valid_sites.append(site)
        
        return np.array(valid_sites) if valid_sites else np.empty((0, 3))
    
    def generate_surface_shell_sites(self, base_positions, shell_thickness=2.0,
                                   n_points=200, method='random'):
        """
        在团簇表面壳层内生成候选位点
        
        参数:
        base_positions: np.array, 基础结构坐标 (n_atoms, 3)
        shell_thickness: float, 表面壳层厚度 (Angstrom)
        n_points: int, 生成的候选位点数
        method: str, 生成方法 ('random', 'fibonacci')
        
        返回:
        np.array: 候选位点坐标 (n_sites, 3)
        """
        # 使用表面探测器生成壳层位点
        candidate_sites = self.surface_detector.generate_surface_shell_sites(
            base_positions, shell_thickness, n_points, method
        )
        
        # 应用距离约束筛选
        valid_sites = []
        for site in candidate_sites:
            distances = np.linalg.norm(base_positions - site, axis=1)
            if (np.min(distances) >= self.min_distance and 
                np.min(distances) <= self.max_distance):
                valid_sites.append(site)
        
        return np.array(valid_sites) if valid_sites else np.empty((0, 3))

if __name__ == "__main__":
    # 测试位点生成器
    generator = SiteGenerator(min_distance=1.0, max_distance=3.0)
    
    # 测试结构 - 三角形Li3团簇
    test_positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.732, 0.0]  # 等边三角形
    ])
    
    print("基础结构:")
    print(test_positions)
    
    # 测试不同的位点生成策略
    print(f"\n随机位点生成:")
    random_sites = generator.generate_random_sites(test_positions, n_sites=10, seed=42)
    print(f"生成了 {len(random_sites)} 个随机位点")
    
    print(f"\n球面位点生成:")
    spherical_sites = generator.generate_spherical_sites(test_positions)
    print(f"生成了 {len(spherical_sites)} 个球面位点")
    
    print(f"\n邻居位点生成:")
    neighbor_sites = generator.generate_neighbor_sites(test_positions)
    print(f"生成了 {len(neighbor_sites)} 个邻居位点")
    
    print(f"\n组合位点生成:")
    combined_sites = generator.generate_combined_sites(test_positions)
    print(f"生成了 {len(combined_sites)} 个组合位点")
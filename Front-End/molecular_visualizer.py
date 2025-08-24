"""
分子可视化模块 - 真实原子团簇建模
用球体和键连显示真实的原子团簇结构
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import cdist
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch

class MolecularVisualizer:
    def __init__(self):
        """
        初始化分子可视化器
        """
        # 元素属性定义
        self.element_properties = {
            'Li': {
                'color': '#CC80FF',  # 淡紫色，Li的典型颜色
                'radius': 1.52,      # Li原子的范德华半径 (Angstrom)
                'atomic_number': 3,
                'name': 'Lithium'
            },
            'predicted': {
                'color': '#FF4444',  # 红色表示预测位点
                'radius': 1.20,      # 稍小一点表示新原子位点
                'alpha': 0.8
            },
            'best': {
                'color': '#FFD700',  # 金色表示最佳位点
                'radius': 1.35,
                'alpha': 1.0
            }
        }
        
        # 键连参数
        self.bond_cutoff = 3.5  # Li-Li键的最大距离 (Angstrom)
        self.bond_color = '#666666'
        self.bond_width = 2.0
    
    def calculate_bonds(self, positions, cutoff=None):
        """
        计算原子间的键连
        
        参数:
        positions: np.array, 原子坐标
        cutoff: float, 键连的最大距离
        
        返回:
        list: 键连对的列表 [(i, j), ...]
        """
        if cutoff is None:
            cutoff = self.bond_cutoff
        
        n_atoms = len(positions)
        distances = cdist(positions, positions)
        
        bonds = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i, j] <= cutoff:
                    bonds.append((i, j))
        
        return bonds
    
    def draw_sphere(self, ax, center, radius, color, alpha=1.0):
        """
        绘制球体表示原子
        
        参数:
        ax: 3D轴对象
        center: tuple, 球心坐标 (x, y, z)
        radius: float, 球体半径
        color: str, 颜色
        alpha: float, 透明度
        """
        # 创建球面坐标
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)
    
    def draw_bond(self, ax, pos1, pos2, color=None, width=None):
        """
        绘制原子间的键连
        
        参数:
        ax: 3D轴对象  
        pos1, pos2: tuple, 两个原子的坐标
        color: str, 键的颜色
        width: float, 键的宽度
        """
        if color is None:
            color = self.bond_color
        if width is None:
            width = self.bond_width
        
        ax.plot([pos1[0], pos2[0]], 
               [pos1[1], pos2[1]], 
               [pos1[2], pos2[2]], 
               color=color, linewidth=width, alpha=0.8)
    
    def visualize_cluster(self, base_positions, predicted_sites=None, 
                         best_site_idx=None, title="Li团簇结构", 
                         save_path=None, show_bonds=True, show_labels=True):
        """
        可视化Li团簇和预测位点
        
        参数:
        base_positions: np.array, 基础Li原子坐标
        predicted_sites: np.array, 预测的候选位点
        best_site_idx: int, 最佳位点索引
        title: str, 图标题
        save_path: str, 保存路径
        show_bonds: bool, 是否显示键连
        show_labels: bool, 是否显示原子标签
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取Li原子属性
        li_props = self.element_properties['Li']
        
        # 1. 绘制基础Li原子（球体）
        print(f"绘制 {len(base_positions)} 个Li原子...")
        for i, pos in enumerate(base_positions):
            self.draw_sphere(ax, pos, li_props['radius'], li_props['color'])
            
            # 添加原子标签
            if show_labels:
                ax.text(pos[0], pos[1], pos[2] + li_props['radius'] + 0.3, 
                       f'Li{i+1}', fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 2. 绘制原子间键连
        if show_bonds and len(base_positions) > 1:
            bonds = self.calculate_bonds(base_positions)
            print(f"绘制 {len(bonds)} 条键连...")
            for i, j in bonds:
                self.draw_bond(ax, base_positions[i], base_positions[j])
        
        # 3. 绘制预测位点
        if predicted_sites is not None and len(predicted_sites) > 0:
            pred_props = self.element_properties['predicted']
            print(f"绘制 {len(predicted_sites)} 个预测位点...")
            
            for i, site in enumerate(predicted_sites):
                # 判断是否是最佳位点
                if i == best_site_idx:
                    best_props = self.element_properties['best']
                    self.draw_sphere(ax, site, best_props['radius'], 
                                   best_props['color'], best_props['alpha'])
                    
                    # 最佳位点特殊标记
                    if show_labels:
                        ax.text(site[0], site[1], site[2] + best_props['radius'] + 0.5, 
                               '★ BEST', fontsize=12, ha='center', va='bottom',
                               color='red', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
                else:
                    self.draw_sphere(ax, site, pred_props['radius'], 
                                   pred_props['color'], pred_props['alpha'])
                    
                    # 普通预测位点标签
                    if show_labels:
                        ax.text(site[0], site[1], site[2] + pred_props['radius'] + 0.3, 
                               f'P{i+1}', fontsize=9, ha='center', va='bottom',
                               color='red', alpha=0.8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.6))
                
                # 绘制预测位点到最近Li原子的虚线连接
                distances = np.linalg.norm(base_positions - site, axis=1)
                closest_li_idx = np.argmin(distances)
                closest_li_pos = base_positions[closest_li_idx]
                
                # 虚线连接
                ax.plot([site[0], closest_li_pos[0]], 
                       [site[1], closest_li_pos[1]], 
                       [site[2], closest_li_pos[2]], 
                       '--', color='red', alpha=0.5, linewidth=1.5)
        
        # 4. 设置坐标轴和外观
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        # 计算合适的显示范围
        all_positions = [base_positions]
        if predicted_sites is not None and len(predicted_sites) > 0:
            all_positions.append(predicted_sites)
        
        all_coords = np.vstack(all_positions)
        margin = 3.0  # 额外边距
        
        x_range = [np.min(all_coords[:, 0]) - margin, np.max(all_coords[:, 0]) + margin]
        y_range = [np.min(all_coords[:, 1]) - margin, np.max(all_coords[:, 1]) + margin]
        z_range = [np.min(all_coords[:, 2]) - margin, np.max(all_coords[:, 2]) + margin]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # 5. 添加图例
        legend_elements = []
        
        # Li原子图例
        li_patch = plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=li_props['color'], 
                             markersize=15, label='Li 原子')
        legend_elements.append(li_patch)
        
        if predicted_sites is not None and len(predicted_sites) > 0:
            # 预测位点图例
            pred_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=pred_props['color'], 
                                   markersize=12, alpha=pred_props['alpha'],
                                   label='预测位点')
            legend_elements.append(pred_patch)
            
            if best_site_idx is not None:
                # 最佳位点图例
                best_props = self.element_properties['best']
                best_patch = plt.Line2D([0], [0], marker='*', color='w', 
                                       markerfacecolor=best_props['color'], 
                                       markersize=18, label='最佳位点')
                legend_elements.append(best_patch)
        
        if show_bonds:
            # 键连图例
            bond_patch = plt.Line2D([0], [0], color=self.bond_color, 
                                   linewidth=self.bond_width, label='Li-Li 键')
            legend_elements.append(bond_patch)
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # 6. 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 7. 添加信息框
        info_text = f"Li团簇: {len(base_positions)} 个原子"
        if predicted_sites is not None and len(predicted_sites) > 0:
            info_text += f"\n预测位点: {len(predicted_sites)} 个"
            if best_site_idx is not None:
                info_text += f"\n最佳位点: 第{best_site_idx+1}个"
        
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                 verticalalignment='bottom', fontsize=10)
        
        # 8. 保存和显示
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"分子可视化图已保存到: {save_path}")
        
        plt.show()
        return fig, ax
    
    def visualize_comparison(self, base_positions, results_dict, 
                           save_path=None, title="策略比较 - 原子团簇可视化"):
        """
        比较多种策略的结果
        
        参数:
        base_positions: np.array, 基础原子坐标
        results_dict: dict, 不同策略的结果 {'strategy_name': result, ...}
        save_path: str, 保存路径
        title: str, 总标题
        """
        n_strategies = len(results_dict)
        if n_strategies == 0:
            return
        
        # 计算子图布局
        cols = min(2, n_strategies)
        rows = (n_strategies + cols - 1) // cols
        
        fig = plt.figure(figsize=(8*cols, 6*rows))
        
        for i, (strategy_name, result) in enumerate(results_dict.items()):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            
            # 获取结果数据
            predicted_sites = result.get('best_sites', [])
            predicted_energies = result.get('best_energies', [])
            
            # 绘制基础Li原子
            li_props = self.element_properties['Li']
            for j, pos in enumerate(base_positions):
                self.draw_sphere(ax, pos, li_props['radius'], li_props['color'])
            
            # 绘制键连
            bonds = self.calculate_bonds(base_positions)
            for i_bond, j_bond in bonds:
                self.draw_bond(ax, base_positions[i_bond], base_positions[j_bond])
            
            # 绘制预测位点
            if len(predicted_sites) > 0:
                pred_props = self.element_properties['predicted']
                best_props = self.element_properties['best']
                
                for k, site in enumerate(predicted_sites):
                    if k == 0:  # 最佳位点
                        self.draw_sphere(ax, site, best_props['radius'], 
                                       best_props['color'], best_props['alpha'])
                    else:
                        self.draw_sphere(ax, site, pred_props['radius'], 
                                       pred_props['color'], pred_props['alpha'])
            
            # 设置子图
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            
            # 子图标题包含性能信息
            n_candidates = result.get('n_candidates', 0)
            best_energy = predicted_energies[0] if len(predicted_energies) > 0 else 0
            ax.set_title(f'{strategy_name}\n候选数: {n_candidates}, 最佳: {best_energy:.4f} eV', 
                        fontsize=11)
            
            # 设置相同的显示范围
            all_coords = [base_positions]
            if len(predicted_sites) > 0:
                all_coords.append(np.array(predicted_sites))
            
            all_positions = np.vstack(all_coords)
            margin = 2.0
            
            ax.set_xlim([np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin])
            ax.set_ylim([np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin])
            ax.set_zlim([np.min(all_positions[:, 2]) - margin, np.max(all_positions[:, 2]) + margin])
            
            ax.view_init(elev=20, azim=45)
        
        fig.suptitle(title, fontsize=16, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"策略比较图已保存到: {save_path}")
        
        plt.show()
        return fig

if __name__ == "__main__":
    # 测试分子可视化器
    visualizer = MolecularVisualizer()
    
    # Li8测试结构
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
    
    # 模拟一些预测位点
    predicted_sites = np.array([
        [1.0, 1.0, 1.0],
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    print("测试分子可视化器...")
    visualizer.visualize_cluster(
        li8_coords, 
        predicted_sites, 
        best_site_idx=0,
        title="Li8团簇 + 预测位点 - 真实原子建模",
        save_path="test_molecular_visualization.png"
    )
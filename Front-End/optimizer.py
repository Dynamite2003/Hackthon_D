"""
Li金属团簇位点优化器
整合位点生成和能量预测，找到最佳位点
"""
import numpy as np
from scipy.optimize import basinhopping
from inference import LiClusterInference
from site_generator import SiteGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from molecular_visualizer import MolecularVisualizer
from advanced_molecular_visualizer import AdvancedMolecularVisualizer
from ball_stick_visualizer import BallStickVisualizer

class SiteOptimizer:
    def __init__(self, model_path='best_schnet_model.pt', data_root='./li_dataset_processed',
                 min_distance=1.0, max_distance=4.0):
        """
        初始化位点优化器
        """
        self.inference = LiClusterInference(model_path, data_root)
        self.site_generator = SiteGenerator(min_distance, max_distance)
        self.molecular_visualizer = MolecularVisualizer()
        self.advanced_visualizer = AdvancedMolecularVisualizer()
        self.ball_stick_visualizer = BallStickVisualizer()

    def find_sites_with_basinhopping(self, base_positions, top_k=5, niter=20):
        """
        使用盆地跳跃算法寻找全局最优位点。
        """
        print(f"使用盆地跳跃策略进行全局优化 (迭代次数: {niter}, 目标数量: {top_k})...")

        def objective_func(x):
            energy = self.inference.predict_with_new_atom(base_positions, [x.tolist()])
            return energy[0]

        results = []
        print(f"将进行 {top_k} 轮独立的盆地跳跃搜索以寻找多个最优解...")
        for i in range(top_k):
            center = np.mean(base_positions, axis=0)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            initial_guess = center + direction * (np.random.rand() * 3.0 + 2.0)

            print(f"  [第 {i+1}/{top_k} 轮] 从初始点 {initial_guess} 开始搜索...")
            minimizer_kwargs = {"method": "L-BFGS-B"}
            bh_result = basinhopping(
                objective_func,
                initial_guess,
                niter=niter,
                minimizer_kwargs=minimizer_kwargs,
                stepsize=0.5
            )
            results.append(bh_result)
            print(f"  [第 {i+1}/{top_k} 轮] 搜索完成, 找到能量最低点: {bh_result.fun:.6f} eV")

        sorted_results = sorted(results, key=lambda r: r.fun)
        
        best_sites = np.array([res.x for res in sorted_results[:top_k]])
        best_energies = np.array([res.fun for res in sorted_results[:top_k]])
        
        total_evals = sum(res.nfev for res in results)
        print(f"盆地跳跃完成, 总计评估了 {total_evals} 个构型。")

        return {
            'best_sites': best_sites,
            'best_energies': best_energies,
            'all_sites': best_sites,
            'all_energies': best_energies,
            'n_candidates': total_evals,
            'base_positions': base_positions
        }

    def find_best_sites(self, base_positions, strategy='combined', n_candidates=None, 
                       top_k=5, **kwargs):
        """
        寻找最佳添加位点
        """
        print(f"使用策略 '{strategy}' 生成候选位点...")
        
        if strategy == 'random':
            candidate_sites = self.site_generator.generate_random_sites(base_positions, n_sites=n_candidates or 100, **kwargs)
        elif strategy == 'grid':
            candidate_sites = self.site_generator.generate_grid_sites(base_positions, grid_spacing=kwargs.get('grid_spacing', 0.5))
        elif strategy == 'spherical':
            candidate_sites = self.site_generator.generate_spherical_sites(base_positions, target_atom_idx=kwargs.get('target_atom_idx', None), n_radial=kwargs.get('n_radial', 5), n_angular=kwargs.get('n_angular', 20))
        elif strategy == 'neighbor':
            candidate_sites = self.site_generator.generate_neighbor_sites(base_positions, n_neighbors=kwargs.get('n_neighbors', 3), offset_distance=kwargs.get('offset_distance', None))
        elif strategy == 'surface':
            candidate_sites = self.site_generator.generate_surface_sites(base_positions, n_sites_per_surface=kwargs.get('n_sites_per_surface', 8), surface_method=kwargs.get('surface_method', 'hybrid'), radius_range=kwargs.get('radius_range', (1.5, 3.0)))
        elif strategy == 'surface_shell':
            candidate_sites = self.site_generator.generate_surface_shell_sites(base_positions, shell_thickness=kwargs.get('shell_thickness', 2.0), n_points=kwargs.get('n_points', 200), method=kwargs.get('method', 'random'))
        elif strategy == 'combined':
            candidate_sites = self.site_generator.generate_combined_sites(base_positions, n_random=kwargs.get('n_random', 50), grid_spacing=kwargs.get('grid_spacing', 0.8))
        else:
            raise ValueError(f"未知的策略: {strategy}")
        
        if len(candidate_sites) == 0:
            return {'best_sites': [], 'best_energies': [], 'all_sites': [], 'all_energies': [], 'n_candidates': 0}
        
        print(f"生成了 {len(candidate_sites)} 个候选位点")
        print("进行批量能量预测...")
        
        predicted_energies = self.inference.predict_with_new_atom(base_positions, candidate_sites.tolist())
        energies_array = np.array(predicted_energies)
        sorted_indices = np.argsort(energies_array)
        
        top_k = min(top_k, len(candidate_sites))
        best_indices = sorted_indices[:top_k]
        best_sites = candidate_sites[best_indices]
        best_energies = energies_array[best_indices]
        
        print(f"找到前 {top_k} 个最佳位点")
        print(f"最低能量: {best_energies[0].item():.6f} eV")
        print(f"最高能量: {best_energies[-1].item():.6f} eV")
        
        return {
            'best_sites': best_sites,
            'best_energies': best_energies,
            'all_sites': candidate_sites,
            'all_energies': energies_array,
            'n_candidates': len(candidate_sites),
            'base_positions': base_positions
        }
    
    def visualize_results(self, results, save_path=None, show_plot=True, 
                         use_molecular_view=True, use_advanced_renderer=True, 
                         visualization_type='space_filling'):
        """
        可视化优化结果
        """
        if use_molecular_view and len(results['best_sites']) > 0:
            title = f"Li团簇位点预测结果 (候选: {results['n_candidates']})"
            if use_advanced_renderer:
                # 根据可视化类型选择渲染器
                if visualization_type == 'ball_stick':
                    self.ball_stick_visualizer.visualize_cluster(
                        base_positions=results['base_positions'],
                        predicted_sites=results['best_sites'],
                        predicted_energies=results['best_energies'],
                        best_site_idx=0,
                        title=title,
                        save_path=save_path
                    )
                else:  # space_filling (默认)
                    self.advanced_visualizer.visualize_cluster(
                        base_positions=results['base_positions'],
                        predicted_sites=results['best_sites'],
                        predicted_energies=results['best_energies'],
                        best_site_idx=0,
                        title=title,
                        save_path=save_path
                    )
            else:
                self.molecular_visualizer.visualize_cluster(
                    base_positions=results['base_positions'],
                    predicted_sites=results['best_sites'],
                    best_site_idx=0,
                    title=title,
                    save_path=save_path,
                    show_bonds=True,
                    show_labels=True
                )
        else:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            base_pos = results['base_positions']
            best_sites = results['best_sites']
            ax1.scatter(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], c='blue', s=100, alpha=0.8, label='Base Li atoms')
            if len(best_sites) > 0:
                ax1.scatter(best_sites[:, 0], best_sites[:, 1], best_sites[:, 2], c='red', s=100, alpha=0.8, label='Best sites')
                for i, site in enumerate(best_sites[:3]):
                    ax1.text(site[0], site[1], site[2], f'#{i+1}', fontsize=10)
            ax1.set_xlabel('X (Å)')
            ax1.set_ylabel('Y (Å)')
            ax1.set_zlabel('Z (Å)')
            ax1.set_title('Best Sites Visualization')
            ax1.legend()
            ax2 = fig.add_subplot(132)
            if len(results['all_energies']) > 0:
                ax2.hist(results['all_energies'], bins=min(30, len(results['all_energies'])//2), alpha=0.7, edgecolor='black')
                ax2.axvline(results['best_energies'][0], color='red', linestyle='--', label=f'Best: {results["best_energies"][0]:.4f} eV')
                ax2.set_xlabel('Energy (eV)')
                ax2.set_ylabel('Count')
                ax2.set_title('Energy Distribution')
                ax2.legend()
            ax3 = fig.add_subplot(133)
            if len(best_sites) > 0:
                indices = range(1, len(best_sites) + 1)
                ax3.bar(indices, results['best_energies'], alpha=0.7)
                ax3.set_xlabel('Site Rank')
                ax3.set_ylabel('Energy (eV)')
                ax3.set_title('Top Sites Energy Ranking')
                ax3.set_xticks(indices)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图片已保存到: {save_path}")
            if show_plot:
                plt.show()
            else:
                plt.close()

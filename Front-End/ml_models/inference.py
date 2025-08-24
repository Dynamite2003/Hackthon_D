"""
Li金属团簇推理模块
用于加载训练好的模型并进行能量预测
"""
import torch
import numpy as np
from torch_geometric.data import Data, Batch
try:
    # 尝试相对导入（当作为模块使用时）
    from .model import SchNetModel
    from ..data_processing.li_dataset import LiClusterDataset
except ImportError:
    # 回退到绝对导入（当作为脚本直接运行时）
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    from ml_models.model import SchNetModel
    from data_processing.li_dataset import LiClusterDataset
import os

class LiClusterInference:
    def __init__(self, model_path=None, data_root=None):
        # 动态确定模型和数据路径
        if model_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(current_dir, 'ml_models', 'best_schnet_model.pt')
        if data_root is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_root = os.path.join(current_dir, 'data_processing', 'li_dataset_processed')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SchNetModel().to(self.device)
        
        # 加载训练好的模型
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"成功加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 加载数据集统计信息用于反归一化
        stats_path = os.path.join(data_root, 'processed', 'stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=self.device, weights_only=False)
            self.mean = stats['mean'].item()
            self.std = stats['std'].item()
            print(f"加载统计信息 - 均值: {self.mean:.6f}, 标准差: {self.std:.6f}")
        else:
            raise FileNotFoundError(f"统计文件未找到: {stats_path}")
    
    def prepare_data(self, positions, num_atoms=None):
        """
        准备单个结构的数据
        
        参数:
        positions: np.array 或 list, 形状为 (n_atoms, 3) 的原子坐标
        num_atoms: int, 原子数量（可选，从positions推断）
        
        返回:
        torch_geometric.data.Data 对象
        """
        if isinstance(positions, list):
            positions = np.array(positions)
        
        if num_atoms is None:
            num_atoms = len(positions)
        
        # 创建原子序数张量（全为Li，原子序数3）
        atomic_numbers = torch.tensor([3] * num_atoms, dtype=torch.long)
        
        # 转换坐标为张量
        positions_tensor = torch.tensor(positions, dtype=torch.float)
        
        # 创建PyG数据对象
        data = Data(z=atomic_numbers, pos=positions_tensor)
        
        return data
    
    def predict_single(self, positions, num_atoms=None):
        """
        预测单个结构的结合能
        
        参数:
        positions: np.array 或 list, 形状为 (n_atoms, 3) 的原子坐标
        num_atoms: int, 原子数量（可选）
        
        返回:
        float: 预测的结合能 (eV)
        """
        data = self.prepare_data(positions, num_atoms)
        data = data.to(self.device)
        
        with torch.no_grad():
            # 模型预测（归一化的能量）
            pred_normalized = self.model(data)
            
            # 反归一化得到真实能量
            pred_real = pred_normalized.item() * self.std + self.mean
            
        return pred_real
    
    def predict_batch(self, structures_list):
        """
        批量预测多个结构的结合能
        
        参数:
        structures_list: list of positions, 每个元素是 (n_atoms, 3) 的坐标数组
        
        返回:
        list: 预测的结合能列表 (eV)
        """
        data_list = []
        for positions in structures_list:
            data = self.prepare_data(positions)
            data_list.append(data)
        
        # 创建批次
        batch = Batch.from_data_list(data_list).to(self.device)
        
        with torch.no_grad():
            # 批量预测
            pred_normalized = self.model(batch)
            
            # 反归一化
            pred_real = pred_normalized.cpu().numpy() * self.std + self.mean
            
        # 确保返回一维列表
        return pred_real.flatten().tolist()
    
    def predict_with_new_atom(self, base_positions, new_atom_positions):
        """
        预测添加新原子后的结合能
        
        参数:
        base_positions: np.array, 基础结构的原子坐标 (n_atoms, 3)
        new_atom_positions: np.array or list, 新原子的候选位置 [(x,y,z), ...] 或单个 [x,y,z]
        
        返回:
        如果new_atom_positions是单个位置: float
        如果new_atom_positions是多个位置: list of floats
        """
        # 统一转换为numpy数组
        if isinstance(new_atom_positions, list):
            new_atom_positions = np.array(new_atom_positions)
        
        # 判断是单个位置还是多个位置
        if new_atom_positions.ndim == 1:
            # 单个位置 [x, y, z]
            new_structure = np.vstack([base_positions, new_atom_positions.reshape(1, 3)])
            return self.predict_single(new_structure)
        elif new_atom_positions.ndim == 2 and new_atom_positions.shape[1] == 3:
            # 多个位置 [[x1,y1,z1], [x2,y2,z2], ...]
            structures = []
            for new_pos in new_atom_positions:
                new_structure = np.vstack([base_positions, new_pos.reshape(1, 3)])
                structures.append(new_structure)
            return self.predict_batch(structures)
        else:
            raise ValueError(f"Invalid shape for new_atom_positions: {new_atom_positions.shape}")

if __name__ == "__main__":
    # 测试推理模块
    inference = LiClusterInference()
    
    # 测试单个结构预测
    test_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.75, 1.3, 0.0]
    ])
    
    energy = inference.predict_single(test_positions)
    print(f"测试结构的预测能量: {energy:.6f} eV")
    
    # 测试添加新原子
    new_atom_pos = [0.75, 0.43, 1.2]
    new_energy = inference.predict_with_new_atom(test_positions, new_atom_pos)
    print(f"添加新原子后的预测能量: {new_energy:.6f} eV")
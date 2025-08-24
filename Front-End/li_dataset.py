import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from tqdm import tqdm
import os

class LiClusterDataset(InMemoryDataset):
    def __init__(self, root, data_path=None, transform=None, pre_transform=None):
        # 设置默认数据路径
        if data_path is None:
            data_path = "/data/Hackthon/data/TheDataOfClusters_4_40 copy.data"
        self.data_path = data_path
        super().__init__(root, transform, pre_transform)
        
        # 加载处理后的数据
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        # 加载均值和标准差
        stats_path = os.path.join(self.processed_dir, 'stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, weights_only=False)
            self.mean = stats['mean']
            self.std = stats['std']

    @property
    def raw_file_names(self):
        return [os.path.basename(self.data_path)]

    @property
    def processed_file_names(self):
        return ['data.pt', 'stats.pt']

    def download(self):
        pass

    def process(self):
        print("首次处理数据，将文本文件转换为PyG格式并保存...")
        
        data_list = []
        all_energies = []

        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing raw data"):
                parts = [float(p) for p in line.strip().split()]
                
                num_atoms = int(parts[0])
                atomic_numbers = torch.tensor([3] * num_atoms, dtype=torch.long)
                positions = torch.tensor(parts[1:-1], dtype=torch.float).view(num_atoms, 3)
                energy = torch.tensor([parts[-1]], dtype=torch.float)
                
                all_energies.append(energy.item())
                data = Data(z=atomic_numbers, pos=positions, y=energy)
                data_list.append(data)

        # 计算能量的均值和标准差
        energies_tensor = torch.tensor(all_energies)
        mean = energies_tensor.mean()
        std = energies_tensor.std()
        
        # 对能量进行归一化
        for data in data_list:
            data.y = (data.y - mean) / std

        # 使用InMemoryDataset的collate方法
        data, slices = self.collate(data_list)
        
        # 保存数据和统计信息
        torch.save((data, slices), self.processed_paths[0])
        torch.save({'mean': mean, 'std': std}, self.processed_paths[1])
        
        print(f"数据处理完成，保存了 {len(data_list)} 个样本")
        
        # 设置实例属性
        self.mean = mean
        self.std = std

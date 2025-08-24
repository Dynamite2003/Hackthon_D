# model.py
import torch
from torch_geometric.nn import SchNet

class SchNetModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6, cutoff=10.0):
        super(SchNetModel, self).__init__()
        
        # SchNet模型
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions, # SchNet的核心交互层数
            num_gaussians=50,
            cutoff=cutoff, # 原子间相互作用的截断半径 (Angstrom)
            readout='add' # 将所有原子贡献加起来得到总能量
        )

    def forward(self, data):
        # 从data对象中解包所需的信息
        z, pos, batch = data.z, data.pos, data.batch
        
        # SchNet的输入是原子序数(z)和坐标(pos)
        energy = self.schnet(z, pos, batch=batch)
        return energy
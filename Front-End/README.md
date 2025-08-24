# Li金属团簇位点预测系统

这是一个基于训练好的SchNet模型的Li金属团簇位点预测系统，能够自动寻找添加新原子的最佳位置。

## 功能特性

- **多种位点生成策略**: 支持随机、网格、球面、邻居和组合搜索策略
- **批量推理**: 高效的批量能量预测
- **灵活的输入格式**: 支持命令行直接输入、CSV、XYZ、JSON等多种文件格式
- **结果可视化**: 自动生成3D可视化图表和能量分布图
- **多种输出格式**: JSON结构化结果、XYZ分子结构文件、CSV数据表

## 文件结构

```
code/
├── inference.py          # 推理模块 - 加载模型和预测功能
├── site_generator.py     # 位点生成器 - 多种位点生成策略
├── optimizer.py          # 优化器 - 整合位点生成和能量预测
├── predict_sites.py      # 主程序 - 命令行界面
├── test_system.py        # 测试脚本 - 系统功能验证
├── model.py             # SchNet模型定义
├── li_dataset.py        # 数据集处理
├── best_schnet_model.pt # 训练好的模型文件
└── li_dataset_processed/ # 处理后的数据集
```

## 安装依赖

```bash
pip install torch torch-geometric torch-cluster torch-scatter torch-sparse matplotlib pandas scipy tqdm
```

## 使用方法

### 1. 命令行直接输入坐标

```bash
# 基本用法 - 预测Li3三角形团簇的最佳新原子位置
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" --strategy combined --top-k 5

# 使用随机策略，生成100个候选位点
python predict_sites.py --coords "0,0,0;2.5,0,0" --strategy random --n-random 100 --top-k 3

# 使用网格搜索，间距0.3Å
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" --strategy grid --grid-spacing 0.3 --top-k 5
```

### 2. 从文件读取坐标

```bash
# 从XYZ文件读取
python predict_sites.py --file structure.xyz --strategy combined --top-k 5

# 从CSV文件读取
python predict_sites.py --file positions.csv --strategy combined --top-k 3

# 从JSON文件读取
python predict_sites.py --file cluster.json --strategy combined --top-k 5
```

### 3. 保存结果

```bash
# 保存JSON格式结果和XYZ结构文件
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" \\
    --output results.json --save-xyz structures.xyz --top-k 3

# 保存可视化图片
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" \\
    --save-plot visualization.png --no-plot --top-k 5
```

### 4. 使用迭代优化

```bash
# 使用迭代优化获得更精确的结果（较慢但更准确）
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" \\
    --iterative --n-iterations 5 --top-k 3
```

### 5. 调整物理约束

```bash
# 自定义原子间距约束
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0" \\
    --min-distance 1.2 --max-distance 3.5 --top-k 5
```

## 输入格式说明

### 命令行坐标格式
使用分号分隔不同原子，逗号分隔xyz坐标：
```
"x1,y1,z1;x2,y2,z2;x3,y3,z3"
```

### XYZ文件格式
```
3
Li3 triangle cluster
Li 0.000000 0.000000 0.000000
Li 2.000000 0.000000 0.000000  
Li 1.000000 1.732000 0.000000
```

### CSV文件格式
```csv
x,y,z
0.0,0.0,0.0
2.0,0.0,0.0
1.0,1.732,0.0
```

### JSON文件格式
```json
{
  "positions": [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0], 
    [1.0, 1.732, 0.0]
  ]
}
```

## 位点生成策略

1. **random**: 随机生成候选位点，适合快速探索
2. **grid**: 基于网格的系统性搜索，适合精确定位
3. **spherical**: 基于球面坐标的搜索，适合对称性结构
4. **neighbor**: 基于现有原子邻居关系的搜索，适合化学直觉
5. **combined**: 结合多种策略的综合搜索，推荐使用

## 参数说明

### 必需参数
- `--coords` 或 `--file`: 输入坐标（二选一）

### 搜索策略参数  
- `--strategy`: 位点生成策略 (默认: combined)
- `--top-k`: 返回前k个最佳位点 (默认: 5)
- `--n-random`: 随机位点数量 (默认: 100)
- `--grid-spacing`: 网格间距 (默认: 0.5 Å)
- `--n-radial`: 球面搜索径向层数 (默认: 5)
- `--n-angular`: 球面搜索角度点数 (默认: 20)

### 物理约束参数
- `--min-distance`: 最小原子间距 (默认: 1.0 Å)
- `--max-distance`: 最大搜索距离 (默认: 4.0 Å)

### 输出选项
- `--output`: JSON结果文件路径
- `--save-xyz`: XYZ结构文件路径  
- `--save-plot`: 可视化图片路径
- `--no-plot`: 不显示可视化图片

### 高级选项
- `--iterative`: 使用迭代优化
- `--n-iterations`: 迭代次数 (默认: 3)

## 输出结果说明

程序会输出以下信息：

1. **基础结构**: 显示输入的原子坐标
2. **搜索统计**: 生成的候选位点数量和测试结果
3. **最佳位点**: 按能量排序的前k个位点，包括：
   - 位点坐标 (Å)
   - 预测结合能 (eV)
   - 相对最佳能量差值 (eV)

### JSON结果文件格式
```json
{
  "base_structure": {
    "n_atoms": 3,
    "positions": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.732, 0.0]]
  },
  "optimization_results": {
    "n_candidates_tested": 134,
    "best_sites": [
      {
        "rank": 1,
        "position": [1.0, 0.577, 1.633],
        "predicted_energy_eV": -0.45234,
        "structure_after_addition": [...]
      }
    ]
  }
}
```

## 系统测试

运行完整的系统测试来验证功能：

```bash
python test_system.py
```

测试包括：
- 推理模块测试
- 位点生成器测试  
- 优化器测试
- 集成测试
- 性能测试
- 文件I/O测试

## 使用示例

### 示例1: Li2二原子添加第三个原子
```bash
python predict_sites.py --coords "0,0,0;2.5,0,0" --strategy combined --top-k 3
```

### 示例2: Li4四面体添加第五个原子  
```bash
python predict_sites.py --coords "0,0,0;2,0,0;1,1.732,0;1,0.577,1.633" \\
    --strategy combined --top-k 5 --output li5_results.json
```

### 示例3: 从文件读取并保存完整结果
```bash
python predict_sites.py --file my_cluster.xyz \\
    --strategy combined --top-k 5 \\
    --output optimization_results.json \\
    --save-xyz best_structures.xyz \\
    --save-plot analysis.png
```

## 技术细节

- **模型**: 基于PyTorch Geometric的SchNet架构
- **训练数据**: Li团簇结合能数据（4-40原子）
- **预测精度**: 模型在测试集上的MAE约为0.01 eV
- **计算效率**: 批量推理，单次预测约0.001秒
- **内存需求**: 约500MB（包括模型和数据）

## 注意事项

1. **坐标单位**: 所有坐标使用埃(Å)单位
2. **能量单位**: 预测能量使用电子伏特(eV)单位  
3. **原子类型**: 当前版本仅支持Li原子
4. **距离约束**: 建议使用1.0-4.0Å的距离约束范围
5. **计算资源**: 更多候选位点需要更长计算时间

## 故障排除

### 常见错误

1. **ModuleNotFoundError**: 检查是否安装了所有依赖包
2. **FileNotFoundError**: 确认模型文件`best_schnet_model.pt`存在
3. **坐标解析错误**: 检查输入格式是否正确
4. **内存不足**: 减少候选位点数量或使用更小的批次

### 性能优化

1. 使用`--no-plot`跳过可视化以节省时间
2. 调整`--n-random`参数平衡精度和速度
3. 使用`--grid-spacing`较大值进行快速搜索
4. 利用`--iterative`模式获得更精确结果

## 更新日志

- v1.0: 初始版本，支持基本位点预测功能
- v1.1: 添加多种位点生成策略
- v1.2: 完善命令行界面和文件I/O支持
- v1.3: 加入迭代优化和可视化功能

## 联系方式

如有问题或建议，请联系开发团队。
#!/usr/bin/env python3
"""
ML模型推理运行脚本
解决相对导入问题
"""
import sys
import os

# 添加当前路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 现在可以正常导入
from ml_models.inference import LiClusterInference
from ml_models.model import SchNetModel

if __name__ == "__main__":
    print("Testing inference module...")
    try:
        model_path = os.path.join(current_dir, 'ml_models', 'best_schnet_model.pt')
        data_root = os.path.join(current_dir, 'data_processing', 'li_dataset_processed')
        
        print(f"Model path: {model_path}")
        print(f"Data root: {data_root}")
        
        if os.path.exists(model_path):
            print("✓ Model file found")
        else:
            print("✗ Model file not found")
            
        if os.path.exists(data_root):
            print("✓ Data directory found")
        else:
            print("✗ Data directory not found")
            
        # 尝试加载推理引擎
        # inference = LiClusterInference(model_path, data_root)
        # print("✓ Inference engine loaded successfully")
        
    except Exception as e:
        print(f"Error: {e}")
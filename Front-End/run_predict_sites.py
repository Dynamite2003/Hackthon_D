#!/usr/bin/env python3
"""
位点预测运行脚本
解决相对导入问题
"""
import sys
import os

# 添加当前路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 现在可以正常导入
from utils.predict_sites import main

if __name__ == "__main__":
    # 调用位点预测主函数
    main()
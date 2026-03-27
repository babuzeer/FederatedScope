#!/usr/bin/env python
"""
启动脚本 - 自动设置 Python 路径
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 运行主程序
from federatedscope.main import main

if __name__ == '__main__':
    main()

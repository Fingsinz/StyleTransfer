"""
@file: utils.py
@brief: 包含实用函数
@author: -
@date: 2025-02-13
"""

import torch
from pathlib import Path
from typing import Union


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    保存模型

    参数:
        model (torch.nn.Module): 要保存的模型。
        target_dir (str): 保存模型的目标目录。
        model_name (str): 保存的模型文件的名称。

    异常:
        断言错误: 如果 model_name 不以.pt或. pth结尾。
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    


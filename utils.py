# =============================================================================
# 文件名称：utils.py
# 模块功能：
#   提供一些辅助工具函数，此文件中主要实现了文件查找功能，便于搜索指定目录下的文件。
#
# 核心函数：
#   find_files_with_suffix(dir_path, suffix)
#     - 在指定目录下查找所有以给定后缀结尾的文件
#
# 使用说明：
#   调用 find_files_with_suffix 函数获取符合条件的文件列表，便于在主流程中批量处理文件。
# =============================================================================

import os
import glob

# def find_files_with_suffix(dir_path, suffix):
#     """查找目录下所有指定后缀的文件
#     参数：
#         dir_path - 要查找的目录路径
#         suffix - 文件后缀，例如 ".mp3"
#     返回：
#         匹配到的文件路径列表
#     """
#     pattern = os.path.join(dir_path, f"*{suffix}")
#     return glob.glob(pattern)

def find_files_with_suffix(dir_path, suffix):
    """
    Recursively find all files under dir_path that end with the given suffix.

    Args:
        dir_path (str): Root directory to start searching from
        suffix (str): File suffix to match (e.g., '.npz')

    Returns:
        list: List of full file paths matching the suffix
    """
    matched_files = []
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files
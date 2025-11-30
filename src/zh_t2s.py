# =============================================================================
# 文件名称：zh_t2s.py
# 模块功能：
#   该模块利用 OpenCC 实现繁体中文到简体中文的转换，保证转录文本的一致性。
#
# 核心功能：
#   初始化 OpenCC 对象，并提供 zh_t2s() 函数进行文本转换。
#
# 使用说明：
#   在需要进行繁简转换处调用 zh_t2s(traditional_text) 即可得到转换后的简体中文文本。
# =============================================================================

import opencc

# 创建 OpenCC 对象，配置为繁体转简体模式（t2s）
cc = opencc.OpenCC('t2s')

def zh_t2s(traditional_text):
    """繁体转简体文本转换函数
    参数：
        traditional_text - 输入的繁体中文字符串
    返回：
        simplified_text - 转换后的简体中文字符串
    """
    simplified_text = cc.convert(traditional_text)
    return simplified_text

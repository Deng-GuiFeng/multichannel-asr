# =============================================================================
# 文件名称：match_speaker.py
# 模块功能：
#   实现说话人匹配逻辑，通过对比转录数据中的时间戳与 VAD 得到的
#   时间段数据，决定转录文本对应的说话人，同时对短噪声内容进行过滤。
#
# 核心函数：
#   parse_segment_info(filename)         - 从文件名解析出时间信息（起止毫秒）
#   load_transcripts(paths)                - 加载并合并所有分割转录文件（Excel 格式）
#   load_timestamps(path)                  - 加载合并后的时间戳 Excel 文件
#   calculate_overlap(...)                 - 计算两个时间段之间的重叠时长
#   find_nearest_vad(...)                  - 查找与转录时间最近的 VAD 片段
#   match_speakers(transcript_df, vad_df)  - 核心匹配逻辑，将转录数据与 VAD 数据进行对应
#   match_pipeline(...)                    - 完整的说话人匹配处理流程
#
# 使用说明：
#   该模块适用于对转录后的音频文本根据时间信息进行自动说话人标注。
# =============================================================================

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple
from config import *

def parse_segment_info(filename: str) -> Tuple[int, int]:
    """解析分割文件名中的时间信息
    参数：
        filename - 分割文件的文件名，要求文件名中包含 _开始时间_结束时间 格式
    返回：
        (起始毫秒数, 结束毫秒数)
    """
    base_name = Path(filename).stem
    match = re.search(r"_(\d+)_(\d+)$", base_name)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), int(match.group(2))

def load_transcripts(paths: List[str]) -> pd.DataFrame:
    """加载并合并所有分割转录文件
    参数：
        paths - 包含所有 Excel 转录文件的路径列表
    返回：
        合并后的 DataFrame，按时间戳排序
    """
    all_dfs = []
    
    for path in paths:
        # 解析片段起始时间和结束时间（用于时间对齐）
        seg_begin, seg_end = parse_segment_info(path)
        
        # 加载 Excel 数据，并调整开始、结束时间为绝对时间（单位：毫秒）
        df = pd.read_excel(path)
        df["begin"] = df["begin"] + seg_begin  # 调整开始时间
        df["end"] = df["end"] + seg_begin        # 调整结束时间
        df = df[df["end"] <= seg_end]  # 过滤掉超出分割片段的部分
        
        all_dfs.append(df)
    
    # 合并所有 DataFrame 并按开始时间排序
    merged = pd.concat(all_dfs).sort_values(by="begin")
    return merged.reset_index(drop=True)

def load_timestamps(path: str) -> pd.DataFrame:
    """加载合并后的时间戳信息
    参数：
        path - 包含时间戳数据的 Excel 文件路径
    返回：
        按开始时间排序的时间戳 DataFrame
    """
    df = pd.read_excel(path)
    return df.sort_values(by="begin").reset_index(drop=True)

def calculate_overlap(t_start, t_end, v_start, v_end) -> int:
    """计算两个时间段的重叠时长（单位：毫秒）
    参数：
        t_start, t_end - 转录文本时间段的开始和结束时间
        v_start, v_end - VAD 片段时间段的开始和结束时间
    返回：
        重叠时长（若无重叠，则返回0）
    """
    overlap_start = max(t_start, v_start)
    overlap_end = min(t_end, v_end)
    return max(0, overlap_end - overlap_start)

def find_nearest_vad(t_start, t_end, vad_starts, vad_ends):
    """查找与当前转录时间段最接近的 VAD 片段
    参数：
        t_start, t_end - 转录文本时间段的起止时间
        vad_starts, vad_ends - VAD 数据中所有片段的起始和结束时间（数组形式）
    返回：
        与转录时间段时间距离最小的 VAD 片段索引（无匹配则返回 None）
    """
    min_distance = float('inf')
    nearest_idx = -1
    
    for i, (v_start, v_end) in enumerate(zip(vad_starts, vad_ends)):
        # 计算转录片段与当前 VAD 片段在四种情况（转录在前/后）下的时间距离
        distances = []
        
        # 当转录时间段在 VAD 片段之后时
        if t_start >= v_end:
            distances.append(t_start - v_end)  # 转录开始与 VAD 结束之间的距离
            distances.append(t_end - v_end)    # 转录结束与 VAD 结束之间的距离
        
        # 当转录时间段在 VAD 片段之前时
        elif t_end <= v_start:
            distances.append(v_start - t_end)  # VAD 开始与转录结束之间的距离
            distances.append(v_start - t_start)  # VAD 开始与转录开始之间的距离
        
        # 选择当前 VAD 片段与转录段的最小距离作为参考
        if distances:
            min_local = min(distances)
            if min_local < min_distance:
                min_distance = min_local
                nearest_idx = i
    
    return nearest_idx if nearest_idx != -1 else None

def match_speakers(transcript_df: pd.DataFrame, vad_df: pd.DataFrame) -> pd.DataFrame:
    """核心说话人匹配逻辑
    参数：
        transcript_df - 包含转录文本及对应时间戳的 DataFrame
        vad_df - 包含 VAD 检测后各语音片段信息的 DataFrame（含开始、结束时间、说话人及静音时长）
    返回：
        匹配后的 DataFrame，每一行包含匹配成功的说话人、调整后的开始结束时间以及转录内容
    """
    results = []
    
    # 预处理 VAD 数据，转换为 numpy 数组以加速计算
    vad_starts = vad_df["begin"].values
    vad_ends = vad_df["end"].values
    vad_speakers = vad_df["speaker"].values
    vad_silence = vad_df["total_silence_duration"].values
    
    # 遍历转录数据中的每一行记录
    for _, row in transcript_df.iterrows():
        t_start = row["begin"]
        t_end = row["end"]
        content = row["content"]
        
        # 查找当前转录段与各个 VAD 片段的重叠时长
        overlaps = []
        for i, (v_start, v_end) in enumerate(zip(vad_starts, vad_ends)):
            ol = calculate_overlap(t_start, t_end, v_start, v_end)
            if ol > 0:
                overlaps.append((i, ol))
        
        # 根据是否存在重叠决定说话人匹配逻辑
        speaker = None
        
        if len(overlaps) == 0:
            # 若转录段与任何 VAD 片段都无重叠，则对短噪声内容进行过滤
            if len(content) <= MIN_CONTENT_LENGTH and content in NOISE_WORDS:
                continue  # 忽略此段噪声
            else:
                # 查找最近的 VAD 片段作为候选
                nearest_idx = find_nearest_vad(t_start, t_end, vad_starts, vad_ends)
                if nearest_idx is not None:
                    speaker = vad_speakers[nearest_idx]
                    silence_duration = vad_silence[nearest_idx]
        else:
            # 若存在重叠，则选择重叠最长的 VAD 片段作为匹配依据
            max_idx = max(overlaps, key=lambda x: x[1])[0]
            speaker = vad_speakers[max_idx]
            silence_duration = vad_silence[max_idx]
        
        # 若匹配到说话人，则将匹配结果记录下来
        if speaker:
            results.append({
                "speaker": speaker,
                "begin": t_start - silence_duration,
                "end": t_end - silence_duration,
                "content": content
            })
    
    return pd.DataFrame(results)

def match_pipeline(
    raw_transcript_path: str,
    speaker_timestamps_path: str,
    save_transcript_path: str
) -> None:
    """完整的说话人匹配处理流程
    参数：
        raw_transcript_path - 包含原始转录文本和时间戳的 Excel 文件路径
        speaker_timestamps_path - 包含左右声道合并后时间戳和说话人信息的 Excel 文件路径
        save_transcript_path - 匹配后输出转录文件的保存路径
    功能：
        1. 加载原始转录文件和时间戳文件；
        2. 通过 match_speakers 函数进行说话人匹配；
        3. 将匹配结果保存到指定的 Excel 文件中
    """
    # 1. 加载原始转录数据
    transcript_df = pd.read_excel(raw_transcript_path)
    
    # 2. 加载合并后的时间戳数据
    timestamps_df = load_timestamps(speaker_timestamps_path)
    
    # 3. 执行说话人匹配
    matched_df = match_speakers(transcript_df, timestamps_df)
    
    # 4. 保存匹配结果至 Excel 文件
    matched_df.to_excel(save_transcript_path, index=False)

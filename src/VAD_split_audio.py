# =============================================================================
# 文件名称：VAD_split_audio.py
# 模块功能：
#   本模块实现对输入 MP3 文件的双声道分离及语音活动检测（VAD）。
#   利用 WebRTC VAD 对左右声道分别进行处理，合并相邻语音片段，
#   并输出包含缓冲时间的左右声道语音片段数据。
#
# 核心函数：
#   load_and_split_channels(mp3_path)  - 加载 MP3 文件并分离左右声道
#   detect_voice_activity(audio, params) - 基于 WebRTC VAD 对单个声道检测语音活动
#   merge_and_filter_segments(segments, params) - 合并相邻语音片段并过滤短片段
#   process_audio(...) - 主流程，调用以上函数并保存处理结果
#
# 使用说明：
#   直接调用 process_audio 函数，传入 MP3 文件路径、输出目录及各项参数，
#   函数将返回左右声道的合并后的语音段及原始 AudioSegment 对象。
# =============================================================================

import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from webrtcvad import Vad
from typing import List, Tuple, Dict
from config import *

def load_and_split_channels(mp3_path: str) -> Tuple[AudioSegment, AudioSegment]:
    """加载并分离双声道音频
    参数：
        mp3_path - 输入 MP3 文件的路径
    返回：
        (左声道 AudioSegment 对象, 右声道 AudioSegment 对象)
    说明：
        利用 pydub 库加载音频，并调用 split_to_mono() 分离左右声道
    """
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio = audio.set_frame_rate(SAMPLE_RATE)
    return audio.split_to_mono()[0], audio.split_to_mono()[1]

def detect_voice_activity(audio: AudioSegment, params: Dict) -> List[Tuple[int, int]]:
    """语音活动检测核心逻辑
    参数：
        audio - 单个声道的 AudioSegment 对象
        params - 包含 VAD 参数的字典（例如：'aggressiveness' 和 'frame_duration_ms'）
    返回：
        语音活动段列表，每个元素为 (start_ms, end_ms)
    说明：
        将音频转换为样本序列，根据帧长度逐帧检测语音活动，记录语音段的起止时间
    """
    vad = Vad(params['aggressiveness'])
    samples = np.array(audio.get_array_of_samples())
    frame_length = int(SAMPLE_RATE * params['frame_duration_ms'] / 1000)
    
    active_segments = []
    current_start = None
    
    for i in range(0, len(samples), frame_length):
        frame = samples[i:i+frame_length]
        if len(frame) < frame_length:
            continue
        
        timestamp_ms = i * 1000 // SAMPLE_RATE
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        
        if is_speech and current_start is None:
            current_start = timestamp_ms
        elif not is_speech and current_start is not None:
            active_segments.append((current_start, timestamp_ms))
            current_start = None
    
    # 若语音段未结束，处理最后剩余部分
    if current_start is not None:
        active_segments.append((current_start, len(samples)*1000//SAMPLE_RATE))
    
    return active_segments

def merge_and_filter_segments(segments: List[Tuple[int, int]], params: Dict) -> List[Tuple[int, int]]:
    """合并相邻片段并过滤短片段
    参数：
        segments - 原始检测得到的语音段列表
        params - 合并与过滤参数字典（包含 'merge_gap_ms' 和 'min_length_ms'）
    返回：
        合并并过滤后的语音段列表，每个元素为 (start_ms, end_ms)
    说明：
        先合并相邻时间间隔小于 merge_gap_ms 的片段，再过滤掉总时长小于 min_length_ms 的片段
    """
    if not segments:
        return []
    
    # 合并相邻语音段
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= params['merge_gap_ms']:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    
    # 过滤掉太短的片段
    return [(s, e) for s, e in merged if (e - s) >= params['min_length_ms']]

def process_audio(
    mp3_path: str,
    output_dir: str,
    vad_aggressiveness: int,
    frame_duration_ms: int,
    merge_gap_ms: int,
    min_segment_length_ms: int,
):
    """主处理流程：VAD 检测并保存语音片段信息
    参数：
        mp3_path - 输入 MP3 文件路径
        output_dir - 输出结果（如分割音频和原始数据）保存目录
        vad_aggressiveness - VAD 的敏感度设置
        frame_duration_ms - 分帧时长（毫秒）
        merge_gap_ms - 合并片段的最大间隔（毫秒）
        min_segment_length_ms - 保留语音段的最小时长（毫秒）
    返回：
        (左声道合并后语音片段列表, 右声道合并后语音片段列表, 左声道 AudioSegment, 右声道 AudioSegment)
    功能说明：
        1. 分离左右声道
        2. 对每个声道执行语音活动检测
        3. 合并相邻语音段并保存检测结果至 Excel 文件（raw_segments.xlsx）
        4. 导出左右声道原始音频为 WAV 格式，供后续处理使用
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义 VAD 及合并所需的参数字典
    vad_params = {
        'aggressiveness': vad_aggressiveness,
        'frame_duration_ms': frame_duration_ms
    }
    merge_params = {
        'merge_gap_ms': merge_gap_ms,
        'min_length_ms': min_segment_length_ms
    }

    # 声道分离及 VAD 检测
    left, right = load_and_split_channels(mp3_path)
    left_segments = detect_voice_activity(left, vad_params)
    right_segments = detect_voice_activity(right, vad_params)
    
    # 合并左右声道检测到的语音段
    left_merged = merge_and_filter_segments(left_segments, merge_params)
    right_merged = merge_and_filter_segments(right_segments, merge_params)
    
    # 构建原始分段结果数据，用于后续说话人匹配
    df_data = []
    for speaker, segments in zip(CHANNEL_NAMES, [left_merged, right_merged]):
        for s, e in segments:
            df_data.append({
                "speaker": {"求助者": 1, "接线员": 0}[speaker],
                "begin": max(0, s - BUFFER_TIME_MS), # 添加缓冲时间后作为开始时间
                "end": e + BUFFER_TIME_MS,          # 添加缓冲时间后作为结束时间
                "content": ""
            })
    
    df = pd.DataFrame(df_data).sort_values("begin").reset_index(drop=True)
    df["speaker"] = df["speaker"].map(VAD_SPEAKER_MAPPING)
    df.to_excel(os.path.join(output_dir, "raw_segments.xlsx"), index=False)
    
    # 保存左右声道的原始音频片段
    left.export(os.path.join(output_dir, "left.wav"), format="wav")
    right.export(os.path.join(output_dir, "right.wav"), format="wav")

    return left_merged, right_merged, left, right

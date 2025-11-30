# =============================================================================
# 文件名称：combine_channels.py
# 模块功能： 
#   该模块用于将左右声道经过 VAD 分割后的音频片段按时间排序合并，
#   在合并过程中根据说话人身份不同插入不同长度的静音间隔，
#   并记录每个合并片段的原始时间戳、说话人标签以及插入的静音时长。
#
# 核心函数：
#   combine_channels(left_segments, right_segments, output_dir, left_audio, right_audio)
#     - 输入左右声道的语音片段（起始与结束时间）
#     - 输入左右声道音频对象
#     - 指定输出目录，合成后的音频保存成 WAV 文件
#     - 返回合成后的每个片段的时间戳、说话人信息和总静音时长，以及合成音频文件的路径
#
# 使用场景：
#   在进行语音转录前，将左右声道经过各自独立 VAD 分割的片段合并，以便生成统一的音频流
#   并记录好对应的时间信息，为后续说话人匹配提供依据。
# =============================================================================

import os
from pydub import AudioSegment
from typing import List, Tuple
from config import *

def combine_channels(
    left_segments: List[Tuple[int, int]],
    right_segments: List[Tuple[int, int]],
    output_dir: str,
    left_audio: AudioSegment,
    right_audio: AudioSegment,
) -> List[Tuple[int, int, str]]:
    """
    组合左右声道片段，生成单声道音频，并记录每个片段的原始时间戳和说话人信息。

    Args:
        left_segments: 左声道语音片段列表，每个元素为 (start_ms, end_ms)
        right_segments: 右声道语音片段列表，每个元素为 (start_ms, end_ms)
        output_dir: 合成音频输出路径
        left_audio: 左声道音频对象
        right_audio: 右声道音频对象

    Returns:
        List[Tuple[int, int, str]]: 合成音频中每个片段的原始时间戳、说话人和该片段前的静音时长
    """
    # 创建输出目录（如果不存在则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 初始化一个空的音频对象用于存放最终的合并音频
    combined = AudioSegment.empty()
    # 用于记录每个片段的起始和结束时间、说话人及插入的静音时长
    timestamps = []
    # 当前合并后音频的时间指针
    current_pos = 0

    # 定义两种静音间隔：同一说话人间用短间隔，不同说话人间用长间隔
    short_silence = AudioSegment.silent(duration=SHORT_GAP_DURATION)
    long_silence = AudioSegment.silent(duration=LONG_GAP_DURATION)

    # 将左右声道的语音片段按起始时间排序，并标记所属声道
    segments = sorted(
        [(s, e, "left") for s, e in left_segments]
        + [(s, e, "right") for s, e in right_segments],
        key=lambda x: x[0],
    )

    # 合并连续同一说话人（即同一声道）的片段，减少不必要的静音插入
    merged_segments = []
    i = 0
    while i < len(segments):
        start, end, channel = segments[i]
        j = i + 1
        while j < len(segments):
            next_start, next_end, next_channel = segments[j]
            if channel == next_channel:
                end = next_end
                j += 1
            else:
                break
        merged_segments.append((start, end, channel))
        i = j

    # 依次处理每个合并后的片段，构造最终音频
    for i, (start, end, channel) in enumerate(merged_segments):
        # 根据声道信息获取对应的音频片段
        if channel == "left":
            segment = left_audio[start:end]
            speaker = "求助者"
        else:
            segment = right_audio[start:end]
            speaker = "接线员"

        # 将获取到的音频片段添加到合成音频中
        combined += segment

        # 计算该片段之前的静音时长（用于后续对齐时间戳）
        total_silence_duration = current_pos - start
        # 记录合并片段在最终音频中的开始和结束时间、说话人以及前方静音时长
        timestamps.append((current_pos, current_pos + len(segment), speaker, total_silence_duration))

        # 更新当前合成音频的总时长
        current_pos += len(segment)

        # 根据下一个片段的说话人决定插入短静音还是长静音
        if i < len(merged_segments) - 1:
            next_start, next_end, next_channel = merged_segments[i + 1]
            if channel == next_channel:
                combined += short_silence
                current_pos += SHORT_GAP_DURATION
            else:
                combined += long_silence
                current_pos += LONG_GAP_DURATION

    # 导出合并后的音频为 WAV 文件
    combined_audio_path = os.path.join(output_dir, "combined.wav")
    combined.export(combined_audio_path, format="wav")

    # 返回记录的时间戳信息与合成音频文件路径
    return timestamps, combined_audio_path

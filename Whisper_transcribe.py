# =============================================================================
# 文件名称：Whisper_transcribe.py
# 模块功能：
#   利用 Whisper 模型对预处理后的音频进行转录。包括生成转录文本，
#   解析文本中的时间戳与语音内容，并导出为结构化 Excel 文件。
#
# 核心函数：
#   generate_transcript(model, processor, waveform, device, gen_kwargs)
#       - 利用 Whisper 模型生成转录结果（含时间戳）
#   parse_transcript(pred_text, speaker)
#       - 解析转录文本，提取每段的开始时间、结束时间及内容
#   save_to_excel(transcript, output_file)
#       - 保存转录数据到 Excel 文件
#   transcribe_audio(model, processor, device, audio_path, output_path, gen_kwargs)
#       - 综合调用上述函数完成单个音频文件的转录流程
#
# 使用说明：
#   调用 transcribe_audio 函数对指定的音频文件进行转录，并保存结果到指定路径。
# =============================================================================

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pydub import AudioSegment
import numpy as np
import pandas as pd
import os
import re
from zh_t2s import zh_t2s

def generate_transcript(model, processor, waveform, device, gen_kwargs):
    """
    利用 Whisper 模型生成转录结果（含时间戳）
    
    参数：
        model - 已加载的 Whisper 模型
        processor - 对应的处理器
        waveform (Tensor) - 输入音频的波形张量
        device (str) - 运算使用的设备
        gen_kwargs (dict) - 模型生成参数，如束搜索数、温度等
    
    返回：
        pred_text (List): 包含转录文本和对应时间戳的列表
    """
    inputs = processor(
        waveform, 
        sampling_rate=16000,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )

    # 根据设备情况，将输入转换为适合模型的数据类型
    inputs = {key: value.to(device).to(model.dtype) if value.dtype == torch.float32 else value.to(device) for key, value in inputs.items()}

    # 生成预测的 id 序列
    pred_ids = model.generate(**inputs, **gen_kwargs)
    
    # 利用处理器进行解码，生成文本（同时获取时间戳信息）
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=True)
    
    return pred_text

def parse_transcript(pred_text, speaker):
    """
    解析转录文本，提取时间戳和内容，并处理时间戳可能缺失的情况。
    
    参数：
        pred_text - Whisper 转录返回的原始文本字符串，其中包含特殊格式的时间戳标记
        speaker - 语音对应的说话人标签（例如 "原始音频"）
    
    返回：
        transcript_data - 包含每段转录的字典列表，每个字典包括:
                          "speaker" - 说话人标签
                          "begin"   - 开始时间（毫秒）
                          "end"     - 结束时间（毫秒）
                          "content" - 转录文本
    """
    transcript_data = []
    
    # 利用正则表达式分割字符串，得到包含时间戳与内容的交替列表
    split_parts = re.split(r'(<\|[\d.]+\|>)', pred_text)
    elements = []
    for part in split_parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('<|') and part.endswith('|>'):
            elements.append(('timestamp', part))
        else:
            elements.append(('content', part))
    
    current_start = None
    i = 0
    while i < len(elements):
        elem_type, elem_val = elements[i]
        
        # 处理时间戳元素，提取数值更新当前起始时间
        if elem_type == 'timestamp':
            try:
                ts_str = elem_val[2:-2]  # 去除标记符号 <| 和 |>
                current_start = float(ts_str)
            except ValueError:
                current_start = None
            i += 1
        
        # 处理内容元素，查找当前段的结束时间戳并记录内容
        elif elem_type == 'content':
            content = elem_val.strip()
            if not content:
                i += 1
                continue
            
            # 查找该内容对应的结束时间（下一个时间戳）
            end_time = None
            j = i + 1
            while j < len(elements):
                if elements[j][0] == 'timestamp':
                    try:
                        ts_str = elements[j][1][2:-2]
                        end_time = float(ts_str)
                    except ValueError:
                        pass
                    break
                j += 1
            
            # 检查时间戳是否有效，若有效则记录本段转录数据
            if current_start is not None and end_time is not None:
                transcript_data.append({
                    "speaker": speaker,
                    "begin": int(current_start * 1000),
                    "end": int(end_time * 1000),
                    "content": content
                })
            else:
                if current_start is None:
                    print(f"内容缺失开始时间戳: {content}")
                    raise ValueError(f"内容缺失开始时间戳: {content}")
                if end_time is None:
                    print(f"内容缺失结束时间戳: {content}")
                    raise ValueError(f"内容缺失结束时间戳: {content}")
            
            i = j if j < len(elements) else i + 1
    
    return transcript_data

def save_to_excel(transcript, output_file):
    """
    保存转录数据到 Excel 文件
    
    参数：
        transcript - 包含转录字典数据的列表
        output_file - 输出 Excel 文件的路径
    """
    df = pd.DataFrame(transcript)
    df.to_excel(output_file, index=False)

def transcribe_audio(model, processor, device, audio_path, output_path, gen_kwargs):
    """
    完整的语音转录流程
    参数：
        model - 已加载的 Whisper 模型
        processor - 模型对应的处理器
        device - 运算设备（如 'cuda:0' 或 'cpu'）
        audio_path - 待转录的音频文件路径
        output_path - 转录结果保存路径（Excel 格式）
        gen_kwargs - Whisper 模型生成参数字典
    流程说明：
        1. 加载并预处理音频（确保 16kHz 采样率、单声道）
        2. 转换为浮点数波形数据并生成批处理格式
        3. 调用 generate_transcript 生成转录文本
        4. 进行繁体转简体转换，确保文本统一
        5. 调用 parse_transcript 解析文本为结构化数据
        6. 调用 save_to_excel 保存结果
    """
    # 加载音频文件，格式为 WAV（经过预处理的合并音频）
    audio = AudioSegment.from_file(audio_path, format="wav") # 修改为 wav 格式
    audio = audio.set_frame_rate(16000).set_channels(1)  # 保证采样率 16kHz 且为单声道
    waveform_raw = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    waveform_raw = np.expand_dims(waveform_raw, axis=0)

    # 利用 Whisper 模型生成转录文本（含时间戳）
    raw_pred_text = generate_transcript(model, processor, waveform_raw, device, gen_kwargs)[0]

    # 将转录结果由繁体转为简体
    raw_pred_text = zh_t2s(raw_pred_text)

    # 解析转录文本，获得结构化转录数据
    raw_transcript = parse_transcript(raw_pred_text, "原始音频")

    # 保存转录数据到 Excel 文件
    save_to_excel(raw_transcript, output_path)

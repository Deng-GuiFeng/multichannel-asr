# =============================================================================
# 文件名称：pipeline.py
# 模块功能：
#   该模块为项目的主流程控制模块，整合了音频预处理、左右声道合并、
#   语音分段转录、说话人匹配以及多进程调度等功能。它协调调用各个模块，
#   并实现对多个 MP3 文件的并行处理。
#
# 核心功能：
#   - load_model_and_processor_global：加载 Whisper 模型及处理器至全局变量
#   - init_worker：多进程工作进程的初始化，加载必要的全局变量
#   - generate_segments：根据合并后音频及时间戳生成转录分段
#   - transcribe_pipline：对单个 MP3 文件进行整体处理，包括 VAD 分割、合并、分段转录、说话人匹配
#   - run_pipline：包装 transcribe_pipline，捕获异常，防止多进程进程崩溃
#   - 主进程：利用 multiprocessing.Pool 实现对待处理文件的并行调度和实时进度更新
#
# 使用说明：
#   直接运行该脚本，即可自动处理指定目录下的 MP3 文件，并将转录结果输出到目标目录。
# =============================================================================

from VAD_split_audio import process_audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from Whisper_transcribe import transcribe_audio
from match_speaker import match_pipeline
from utils import *
from combine_channels import combine_channels
import os
from multiprocessing import Pool, Manager, set_start_method
from tqdm import tqdm
import bisect
import pandas as pd
from pydub import AudioSegment
from config import *

# --- 全局变量定义 ---
model = None
processor = None
device = None
src_dir = None
tgt_dir = None
match_delta = None
progress_counter = None  # 用于多进程间共享进度计数器
total_files = None       # 总待处理文件数（共享变量）
progress_lock = None     # 进度更新时的互斥锁

def load_model_and_processor_global(model_id, device_str):
    """加载 Whisper 模型和处理器到全局变量
    参数：
        model_id - 模型路径或标识符
        device_str - 使用的设备（例如 'cuda:0' 或 'cpu'）
    功能：
        根据指定的设备加载模型，设置数据类型以适应计算要求
    """
    global model, processor, device
    device = device_str
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
        low_cpu_mem_usage=False,
        use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

def init_worker(model_id, device_str, src_dir_str, tgt_dir_str, match_delta_val, counter, total, lock):
    """多进程工作进程初始化函数
    参数：
        model_id - 模型标识符
        device_str - 设备设置
        src_dir_str - 源文件目录路径
        tgt_dir_str - 目标输出目录路径
        match_delta_val - 说话人匹配相关参数（备用）
        counter - 多进程共享进度计数器
        total - 待处理文件总数（共享）
        lock - 进度更新时的互斥锁对象
    功能：
        初始化全局变量，加载模型（避免 CUDA 重复加载）
    """
    global model, processor, device, src_dir, tgt_dir, match_delta, progress_counter, total_files, progress_lock
    device = device_str
    src_dir = src_dir_str
    tgt_dir = tgt_dir_str
    match_delta = match_delta_val
    progress_counter = counter
    total_files = total
    progress_lock = lock
    # 若模型未加载，则加载之（主要针对 CUDA 环境下避免重复加载）
    if model is None:
        load_model_and_processor_global(model_id, device)

def generate_segments(timestamps, total_duration, segment_duration_sec=SEGMENT_DURATION_SEC, min_segment_sec=MIN_SEGMENT_SEC):
    """根据时间戳和总音频时长生成转录分段
    参数：
        timestamps - 合并音频时记录的时间戳列表
        total_duration - 合并后音频的总时长（单位：毫秒）
        segment_duration_sec - 目标分段时长（单位：秒）
        min_segment_sec - 分段的最小允许时长（单位：秒）
    返回：
        分段列表，每个分段为 (start_ms, end_ms)
    说明：
        利用 bisect 对齐已存在的切割点，确保各段均包含 VAD 边界信息
    """
    segment_duration_ms = segment_duration_sec * 1000
    min_segment_ms = min_segment_sec * 1000
    
    cut_points = set()
    cut_points.add(0)
    cut_points.add(total_duration)

    for start, end, _, _ in timestamps:
        cut_points.add(start)
        cut_points.add(end)
    cut_points = sorted(list(cut_points))
    
    chunks = []
    current_start = 0
    
    while current_start < total_duration:
        target_end = current_start + segment_duration_ms
        idx = bisect.bisect_right(cut_points, target_end) - 1
        
        # 处理切割点边界情况
        if idx < 0 or cut_points[idx] <= current_start:
            next_idx = bisect.bisect_right(cut_points, current_start)
            end = cut_points[next_idx] if next_idx < len(cut_points) else total_duration
        else:
            end = cut_points[idx]
        
        # 若分段时长不足最小要求，则与前一段合并
        if end - current_start < min_segment_ms and len(chunks) > 0:
            prev_start, prev_end = chunks[-1]
            if current_start - prev_end < min_segment_ms:
                chunks[-1] = (prev_start, end)
                current_start = end
                continue
        
        if end > current_start:
            chunks.append((current_start, end))
            current_start = end
        else:
            break
    
    return chunks

def transcribe_pipline(mp3_path):
    """对单个 MP3 文件执行完整的转录流程
    参数：
        mp3_path - 单个 MP3 音频文件路径
    流程：
        1. 对音频进行 VAD 分割，生成左右声道的语音片段
        2. 合并左右声道，并记录各片段时间戳及说话人信息
        3. 对合并后的音频按照生成的分段进行逐段 Whisper 转录
        4. 调整转录结果的时间戳并合并成最终的转录文件
        5. 进行说话人匹配，并保存最终的匹配结果
    """
    mp3_id = os.path.basename(mp3_path).split(".")[0]
    mp3_out_dir = os.path.join(tgt_dir, mp3_id)
    os.makedirs(mp3_out_dir, exist_ok=True)

    # 1. VAD 分割音频，分别获得左右声道语音片段及原始音频对象
    VAD_dir = os.path.join(mp3_out_dir, "VAD")
    left_segments, right_segments, left_audio, right_audio = process_audio(
        mp3_path,
        VAD_dir,
        vad_aggressiveness = VAD_AGGRESSIVENESS,
        frame_duration_ms = FRAME_DURATION_MS,
        merge_gap_ms = MERGE_GAP_MS,
        min_segment_length_ms = MIN_SEGMENT_LENGTH_MS,
    )

    # 2. 合并左右声道音频，并记录各片段的原始时间戳与说话人信息
    combined_dir = os.path.join(mp3_out_dir, "combined")
    timestamps, combined_audio_path = combine_channels(
        left_segments,
        right_segments,
        combined_dir,
        left_audio,
        right_audio,
    )
    timestamps_df = pd.DataFrame(timestamps, columns=["begin", "end", "speaker", "total_silence_duration"])
    timestamps_df.to_excel(os.path.join(mp3_out_dir, "timestamps.xlsx"), index=False)

    # 3. 根据合并后音频和时间戳信息生成转录分段
    audio = AudioSegment.from_wav(combined_audio_path)
    total_duration = len(audio)
    segments = generate_segments(timestamps, total_duration)

    # 4. 对每个分段调用 Whisper 进行转录
    transcripts = []
    for i, (start, end) in enumerate(segments):
        segment_audio = audio[start:end]
        segment_audio_path = os.path.join(combined_dir, f"segment_{i}.wav")
        segment_audio.export(segment_audio_path, format="wav")

        transcript_path = os.path.join(combined_dir, f"transcript_{i}.xlsx")
        transcribe_audio(
            model,
            processor,
            device,
            segment_audio_path,
            transcript_path,
            GEN_KWARGS,
        )

        # 调整转录的时间戳（加上分段的起始时间偏移）
        transcript_df = pd.read_excel(transcript_path)
        transcript_df["begin"] += start
        transcript_df["end"] += start
        transcripts.append(transcript_df)

    # 5. 合并所有分段的转录结果，输出为最终转录文件
    final_transcript_df = pd.concat(transcripts, ignore_index=True)
    final_transcript_path = os.path.join(mp3_out_dir, "whisper_transcript.xlsx")
    final_transcript_df.to_excel(final_transcript_path, index=False)

    # 6. 根据声道合并时记录的时间戳数据执行说话人匹配
    match_pipeline(
        final_transcript_path,
        os.path.join(mp3_out_dir, "timestamps.xlsx"),
        os.path.join(mp3_out_dir, "transcript.xlsx"),
    )

def run_pipline(*args):
    """包装 transcribe_pipline，捕获异常防止单文件错误中断整体流程
    参数：
        args - 传递给 transcribe_pipline 的参数（此处为单个 MP3 文件路径）
    """
    try:
        transcribe_pipline(*args)
    except Exception as e:
        print(f"Error: {args[0]}: {e}")

if __name__ == "__main__":
    # 在某些平台上，在导入 torch 前必须设置多进程启动方式
    set_start_method("spawn", force=True)  # 对于 CUDA 多进程处理至关重要

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="心理热线语音识别与说话人匹配 pipeline")
    parser.add_argument("--model_id", type=str, required=True, help="Whisper 模型的路径或标识符")
    parser.add_argument("--device", type=str, required=True, help="运算设备，例如 'cuda:0' 或 'cpu'")
    parser.add_argument("--src_dir", type=str, required=True, help="待处理 MP3 文件所在目录")
    parser.add_argument("--tgt_dir", type=str, required=True, help="输出转录结果保存的目录")
    args = parser.parse_args()

    model_id = args.model_id
    device = args.device
    src_dir = args.src_dir
    tgt_dir = args.tgt_dir

    os.makedirs(tgt_dir, exist_ok=True)

    # 获取所有未完成处理的 MP3 文件列表（判断条件基于目标目录中的子目录名称）
    unfinished_ids = [mp3_file for mp3_file in find_files_with_suffix(src_dir, '.mp3') if os.path.basename(mp3_file).split(".")[0] not in [d for d in os.listdir(tgt_dir) if os.path.isdir(os.path.join(tgt_dir, d))]]
    print(f"Found {len(unfinished_ids)} files to process.")

    # 使用 Manager 管理共享变量（进度计数器、总文件数及互斥锁）
    with Manager() as manager:
        progress_counter = manager.Value('i', 0)  # 整数计数器
        total_files = manager.Value('i', len(unfinished_ids))
        progress_lock = manager.Lock() # 创建共享锁

        # 使用进程池进行多进程并行处理
        with Pool(processes=4, initializer=init_worker,
                  initargs=(model_id, device, src_dir, tgt_dir, match_delta, progress_counter, total_files, progress_lock)) as pool:
            # 使用 imap_unordered 以实现实时进度更新
            for _ in tqdm(pool.imap_unordered(run_pipline, unfinished_ids), total=len(unfinished_ids)):
                pass  # 进度条更新由 tqdm 自动处理

    print("All files processed.")

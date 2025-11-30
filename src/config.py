# =============================================================================
# 文件名称：config.py
# 模块功能：
#   本文件用于设置项目中所有模块的公共配置参数，包括 VAD 分割、
#   Whisper 转录、音频处理、分段转录以及说话人匹配等各项参数。
#
# 使用说明：
#   根据实际需求调整每个参数的值，参数命名均较为直观说明了其用途。
# =============================================================================

# ========== 可调参数 ==========

# VAD 参数设置
VAD_AGGRESSIVENESS = 3       # VAD 敏感度，取值 0~3，值越大过滤非语音越严格
FRAME_DURATION_MS = 30       # 分帧时长（单位：毫秒），常用值：10/20/30ms
MERGE_GAP_MS = 100           # 合并相邻语音段的最大间隔（单位：毫秒）
MIN_SEGMENT_LENGTH_MS = 300  # 语音片段的最小保留时长（单位：毫秒）

# Whisper 参数设置
GEN_KWARGS = {
    "num_beams": 1,  # 解码时束搜索的宽度
    "condition_on_prev_tokens": False,  # 是否依赖前序已生成的 token
    "compression_ratio_threshold": 1.20,  # 压缩率阈值，防止过长重复输出
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # 温度参数，影响采样多样性
    "logprob_threshold": -1.0,  # 概率阈值
    "no_speech_threshold": 0.6,  # 无语音检测阈值
    "return_timestamps": True,  # 返回每段转录的时间戳
    "language": "zh",  # 指定转录语言为中文
}

# 音频处理相关参数
SAMPLE_RATE = 16000          # 设置采样率为 16kHz（WebRTC VAD 要求）
CHANNEL_NAMES = ["求助者", "接线员"]  # 左/右声道对应的说话人标签

# 缓冲时间设置
BUFFER_TIME_MS = 50  # 为每个语音段添加的缓冲时长（单位：毫秒）

# 合并声道时的静音插入参数
SHORT_GAP_DURATION = 100  # 同一说话人语音段间插入的静音时长（单位：毫秒）
LONG_GAP_DURATION = 1000  # 不同说话人语音段间插入的静音时长（单位：毫秒）

# 分段转录参数设置
SEGMENT_DURATION_SEC = 300 # 转录时每个片段的时长（单位：秒）
MIN_SEGMENT_SEC = 10       # 分段转录时保留的最小片段时长（单位：秒）

# 说话人匹配参数设置
VAD_SPEAKER_MAPPING =  {1: "求助者", 0: "接线员"}  # VAD 模块中声道编号映射到说话人标签
MIN_CONTENT_LENGTH = 2  # 保留转录文本的最小长度
NOISE_WORDS = {"嗯", "啊", "哦", "呃", "嗯嗯", "啊啊", "哦哦", "呃呃"}  # 被视为噪声的语气词集合

CHUNK = 2048  # audio frame length
FS = 48000  # sample rate
STEP = 350.0  # 每个频率的跨度
NUM_OF_FREQ = 8  # 频率数量
DELAY_TIME = 1  # 麦克风的延迟时间
STD_THRESHOLD = 0.022  # 相位标准差阈值
N_CHANNELS = 7  # 声道数
F0 = 17000.0

RECEIVE_CHANNELS = 8  # 可以改进，等测试，直接用7channel

THRESHOLD = 0.008  # 运动阈值

PADDING_LEN = 1400
from audio_frame_process import WakeOrRecognition, get_waken_gesture_data
import queue
import numpy as np
import wave
from config import *
import threading
import log
from tensorflow.keras import models

from models import reco_model_cons, au_model_cons
from record_tool import PlayRecorder

audio_queue = queue.Queue(maxsize=10)


def get_dtype_from_width(width, unsigned=True):
    if width == 1:
        if unsigned:
            return np.uint8
        else:
            return np.int8
    elif width == 2:
        return np.int16
    elif width == 3:
        raise ValueError("unsupported type: int24")
    elif width == 4:
        return np.float32
    else:
        raise ValueError("Invalid width: %d" % width)


def load_audio_data(filename, type='pcm'):
    if type == 'pcm':
        rawdata = np.memmap(filename, dtype=np.float32, mode='r')
        return rawdata, 48e3
    elif type == 'wav':
        wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
        num_frame = wav.getnframes()  # 获取帧数
        num_channel = wav.getnchannels()  # 获取声道数
        framerate = wav.getframerate()  # 获取帧速率
        num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
        str_data = wav.readframes(num_frame)  # 读取全部的帧
        wav.close()  # 关闭流
        wave_data = np.frombuffer(str_data, dtype=get_dtype_from_width(num_sample_width))  # 将声音文件数据转换为数组矩阵形式
        wave_data = wave_data.reshape((-1, num_channel))  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        return wave_data, framerate


def test():
    def put_data():
        origin_data, fs = load_audio_data(r'D:\projects\pyprojects\gestrecodemo\realtimesys\test.wav', 'wav')
        data = origin_data.reshape((-1, N_CHANNELS + 1))
        data = data.T  # shape = (num_of_channels, all_frames)
        data = data[:, int(fs * 1):]
        for i in range(0, data.shape[1], CHUNK):
            frame: np.ndarray = data[:, i:i + CHUNK]
            frame_1 = frame.T
            frame_1 = frame_1.flatten()
            frame_byte = b''.join(frame_1)
            try:
                audio_queue.put(frame_byte, block=True)
            except queue.Full:
                log.logger.warning('Audio queue is full because of processing audio frame too slowly')

    put_thread = threading.Thread(target=put_data)
    put_thread.start()

    # 识别模型加载
    reco_model_file = r'models/fusion.h5'
    phase_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
    magn_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
    n_classes = 10
    reco_model = reco_model_cons(n_classes, phase_input_shape, magn_input_shape)
    reco_model.load_weights(reco_model_file)
    # 认证模型加载
    au_model_file = r'models/t1.h5'
    au_model = au_model_cons()
    au_model.load_weights(au_model_file)
    # 加载认证用数据
    waken_gesture_data = get_waken_gesture_data(r'waken_gesture_data/push')

    wr = WakeOrRecognition(audio_queue, reco_model, au_model, waken_gesture_data)
    wr.run()
    put_thread.join()

def cos_wave(A, f, fs, t, phi=0):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    # 默认类型为float32
    y = A * np.cos(2 * np.pi * f * n * Ts + phi * (np.pi / 180)).astype(np.float32)
    return y

def main():
    t = 300
    A = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, 17350, 48e3, t)
    for i in range(1, 8):
        y = y + A[i] * cos_wave(1, 17350 + i * 700, 48e3, t)
    signal = alpha * y

    play_recorder = PlayRecorder(12, 4, 8, audio_queue)
    play_recorder.play_and_record(signal)
    # wr = WakeOrRecognition(audio_queue)
    # wr.run()
    # 识别模型加载
    reco_model_file = r'models/fusion.h5'
    phase_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
    magn_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
    n_classes = 10
    reco_model = reco_model_cons(n_classes, phase_input_shape, magn_input_shape)
    reco_model.load_weights(reco_model_file)
    # 认证模型加载
    au_model_file = r'models/t1.h5'
    au_model = au_model_cons()
    au_model.load_weights(au_model_file)
    # 加载认证用数据
    waken_gesture_data = get_waken_gesture_data(r'waken_gesture_data/push')

    wr = WakeOrRecognition(audio_queue, reco_model, au_model, waken_gesture_data)
    wr.run()


if __name__ == '__main__':
    # 没用归一化, 没用cuda加速，padding能优化，多取前后两个CHUNK能优化
    main()
    # test()

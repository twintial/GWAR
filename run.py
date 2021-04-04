from autio_frame_process import WakeOrRecognition
import queue
import numpy as np
import wave
from config import *
import threading
import log

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


def main():
    signal = ''
    # play_recorder = PlayRecorder()
    # play_recorder.play_and_record(signal)
    # wr = WakeOrRecognition(audio_queue)
    # wr.run()


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

    from tensorflow.keras import models
    model_file = r'D:\projects\pyprojects\gestrecodemo\nn\models\mic_speaker_phase_234_5.h5'
    model: models.Sequential = models.load_model(model_file)
    wr = WakeOrRecognition(audio_queue, model, None, None)
    wr.run()
    put_thread.join()


if __name__ == '__main__':
    # 没用归一化
    # main()
    test()

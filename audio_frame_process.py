import threading
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Lock
import os

from config import *
import log
import queue
import numpy as np
import socket

from audio_util import butter_bandpass_filter, get_cos_IQ_raw_offset, butter_lowpass_filter, get_phase, get_cos_IQ_raw, \
    move_average_overlap_filter, padding_or_clip, get_magnitude, convert_wavfile_to_phase_and_magnitude


def get_waken_gesture_data(npz_path):
    waken_gesture_data = []
    file_names = os.listdir(npz_path)
    for file_name in file_names:
        abs_path = os.path.join(npz_path, file_name)
        data = np.load(abs_path)
        phase_diff: np.ndarray = data['phase_diff']
        waken_gesture_data.append(phase_diff[..., np.newaxis])
    return waken_gesture_data


def get_pair(wake_gesture_data, input_gesture):
    pairs = []
    for wake_gesture in wake_gesture_data:
        pairs.append([wake_gesture, input_gesture])
    return np.array(pairs)


# test
def socket_client(phase_diff):
    address = ('127.0.0.push', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    tcp_socket.setblocking(False)
    buffer = b''.join(phase_diff)
    print(len(buffer))
    tcp_socket.send(buffer)


def socket_send(ip, port, buffer):
    address = (ip, port)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    tcp_socket.setblocking(False)
    tcp_socket.send(buffer)


class WakeOrRecognition:
    def __init__(self, audio_queue: queue.Queue, reco_model, wake_model, wake_gesture_data_path):
        self.lock = Lock()
        self.recording_gesture = 0
        self.new_gesture_raw_data = []

        self.ready_save_data = []

        put_thread = threading.Thread(target=self._set_socket_conn)
        put_thread.start()

        self._audio_queue = audio_queue
        self.reco_model = reco_model
        self.wake_model = wake_model
        self.wake_gesture_data = wake_gesture_data_path

        self._processed_frames = None

        self._waken = False

        self._motion_start = False
        self._motion_end = False

        # 运动检测参数
        self.continuous_threshold = 4
        self.motion_start_index = -1
        self.motion_start_index_constant = -1  # 为了截取运动片段
        self.motion_stop_index = -1
        self.lower_than_threshold_count = 0  # 超过3次即运动停止
        self.higher_than_threshold_count = 0  # 超过3次即运动开始
        self.pre_frame = 4

        self._gesture_frame_len = 0

        self._offset = 0

        self._max_frame_count_in_memory = FS * 2  # 2s

    def _revise_recording_gesture(self, v):
        self.lock.acquire()
        self.recording_gesture = v
        self.lock.release()

    def _set_socket_conn(self):
        address = ('127.0.0.1', 31503)
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.bind(address)
        tcp_socket.listen(1)
        while True:
            connection, ad = tcp_socket.accept()
            print(f'got connected from {ad}')
            buffer = connection.recv(10)
            code = int(str(buffer, "utf-8"))
            print(code)
            self._revise_recording_gesture(code)
            if code == 0:
                pass  # nothing
            elif code == 1:
                # record
                self.new_gesture_raw_data.clear()
                self.ready_save_data.clear()
            elif code == 2:
                # start
                self.new_gesture_raw_data.clear()
            elif code == 3:
                # stop and convert
                self.ready_save_data.append(np.hstack(self.new_gesture_raw_data))
            elif code == 4:
                # finish and save
                os.makedirs('waken_gesture_data/test')
                for index, data in enumerate(self.ready_save_data):
                    phase_diff, magn_diff = convert_wavfile_to_phase_and_magnitude(data[:N_CHANNELS])
                    save_file = f'waken_gesture_data/test/{index}.npz'
                    np.savez_compressed(save_file,
                                        phase_diff=phase_diff,
                                        magn_diff=magn_diff)


    def _get_next_frame(self):
        try:
            return self._audio_queue.get(block=True, timeout=10)
        except queue.Empty:
            log.logger.error('No audio frames come in over 10s')
            exit(1)

    def _frame_data_process(self, frame_data):
        frame_data = np.frombuffer(frame_data, dtype=np.int16)
        frame_data = frame_data.reshape(-1, RECEIVE_CHANNELS).T  # shape=(channels, frame_count)
        # 能不能改进
        self._processed_frames = frame_data if self._processed_frames is None \
            else np.hstack((self._processed_frames, frame_data))

    def _recording_frame_process(self, frame_data):
        frame_data = np.frombuffer(frame_data, dtype=np.int16)
        frame_data = frame_data.reshape(-1, RECEIVE_CHANNELS).T  # shape=(channels, frame_count)
        self.new_gesture_raw_data.append(frame_data)

    def _frame2phase(self, frame_segment):
        # 只转换第一个频率的phase，是不是可以用I/Q？
        fc = F0
        filtered_frame = butter_bandpass_filter(frame_segment, fc - 150, fc + 150)
        I_raw, Q_raw = get_cos_IQ_raw_offset(filtered_frame, fc, self._offset)
        # 保证phase的连贯
        self._offset += CHUNK
        I = butter_lowpass_filter(I_raw, 200)
        Q = butter_lowpass_filter(Q_raw, 200)
        I = I[:, CHUNK:CHUNK * 2]
        Q = Q[:, CHUNK:CHUNK * 2]
        unwrapped_phase = get_phase(I, Q)
        assert unwrapped_phase.shape[0] == 1
        return unwrapped_phase.reshape(-1)

    def _motion_detection(self, frame_segment):
        # 除了运动停止以外都return False
        phase = self._frame2phase(frame_segment)
        std = np.std(phase)
        if self._motion_start:
            self._gesture_frame_len += CHUNK
            if std < STD_THRESHOLD:
                self.lower_than_threshold_count += 1
                if self.lower_than_threshold_count >= self.continuous_threshold:
                    # 运动结束，在前4CHUNK阈值已经低于，另外减去pre_frame*CHUNK的提前量(pre_frame大于lower_than_threshold_count则多出来的部分无法取到)
                    self._gesture_frame_len -= (self.continuous_threshold - self.pre_frame) * CHUNK
                    self._motion_start = False
                    self.lower_than_threshold_count = 0
                    # 运动停止
                    return True
            else:
                self.lower_than_threshold_count = 0
        else:
            if std > STD_THRESHOLD:
                self.higher_than_threshold_count += 1
                if self.higher_than_threshold_count >= self.continuous_threshold:
                    # 运动开始，在前4CHUNK阈值已经超过，另外加上pre_frame*CHUNK的提前量
                    self._gesture_frame_len = (self.continuous_threshold + self.pre_frame) * CHUNK
                    self._motion_start = True
                    self.higher_than_threshold_count = 0
            else:
                self.higher_than_threshold_count = 0
        return False

    # wake和gesture共用
    def _gesture_action_multithread(self, gesture_frames, action: callable):
        unwrapped_phase_diff_list = [None] * NUM_OF_FREQ
        magnitude_diff_list = [None] * NUM_OF_FREQ

        def get_phase_and_diff(i):
            fc = F0 + i * STEP
            data_filter = butter_bandpass_filter(gesture_frames, fc - 150, fc + 150)
            I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, FS)
            # 滤波+下采样
            I = move_average_overlap_filter(I_raw[:, I_Q_skip:-I_Q_skip])
            Q = move_average_overlap_filter(Q_raw[:, I_Q_skip:-I_Q_skip])
            # denoise,暂时不用

            unwrapped_phase = get_phase(I, Q)  # 这里的展开目前没什么效果
            unwrapped_phase = unwrapped_phase.astype(dtype=np.float32)
            # import matplotlib.pyplot as plt
            # if i == 0:
            #     plt.figure()
            #     for j in range(7):
            #         plt.subplot(4, 2, j + push)
            #         plt.plot(unwrapped_phase[j])
            #     plt.show()
            unwrapped_phase_diff = np.diff(unwrapped_phase)
            # pad是不是可以放到后面做
            unwrapped_phase_diff_padded = padding_or_clip(unwrapped_phase_diff, PADDING_LEN)
            unwrapped_phase_diff_list[i] = unwrapped_phase_diff_padded

            magnitude = get_magnitude(I, Q)
            magnitude_diff = np.diff(magnitude)
            magnitude_diff_padded = padding_or_clip(magnitude_diff, PADDING_LEN)
            magnitude_diff_list[i] = magnitude_diff_padded

        with ThreadPoolExecutor(max_workers=8) as pool:
            pool.map(get_phase_and_diff, [i for i in range(NUM_OF_FREQ)])
        # for i in range(NUM_OF_FREQ):
        #     get_phase_and_diff(i)
        unwrapped_phase_diff_list = np.array(unwrapped_phase_diff_list).reshape(data_shape)
        magnitude_diff_list = np.array(magnitude_diff_list).reshape(data_shape)
        action(unwrapped_phase_diff_list, magnitude_diff_list)

    def gesture_recognition(self, phase_diff_data, magn_diff_data):
        phase_diff_data = phase_diff_data[np.newaxis, ..., np.newaxis]
        magn_diff_data = magn_diff_data[np.newaxis, ..., np.newaxis]
        y_predict = self.reco_model.predict([phase_diff_data, magn_diff_data])
        label = [
            '抓住-松开', '顺时针画圈', '逆时针画圈', '前推-后拉', '后拉-前推',
            '单击', '双击', '左-右滑动', '右-左滑动', '下-上滑动']
        label_num = int(np.argmax(y_predict[0]))
        print(label_num)
        print(label[label_num])
        # socket
        socket_send('127.0.0.1', 31500, b''.join(phase_diff_data))
        socket_send('127.0.0.1', 31502, label_num.to_bytes(8, 'little'))

    def gesture_wake(self, phase_diff_data, magn_diff_data):
        input_gesture = phase_diff_data[..., np.newaxis]
        input_pairs = get_pair(self.wake_gesture_data, input_gesture)
        y_predict = self.wake_model.predict([input_pairs[:, 0], input_pairs[:, 1]])

        dist = np.mean(y_predict)
        print(f'相似度：{dist}')
        if dist < 0.5:
            print("\033[push;31m wake gesture\033[0m")
            # 什么时候变成Flase？
            self._waken = True
            # socket
            socket_send('127.0.0.1', 31501, b'1')
        else:
            print("not wake gesture")
        # socket
        socket_send('127.0.0.1', 31500, b''.join(phase_diff_data))

    def run(self):
        # 直接在主线程中运行
        while True:
            next_frame = self._get_next_frame()  # 会阻塞
            # lock
            self.lock.acquire()
            if self.recording_gesture == 0 or self.recording_gesture == 4:
                self._frame_data_process(next_frame)
                if self._processed_frames.shape[1] > self._max_frame_count_in_memory:
                    self._processed_frames = self._processed_frames[:, CHUNK:]
                if self._processed_frames.shape[1] > 3 * CHUNK:
                    # 前后都多拿一个CHUNK
                    frame_segments = self._processed_frames[:, -3 * CHUNK:]
                    self._motion_end = self._motion_detection(frame_segments[0].reshape(1, -1))
                    if self._motion_end:
                        gesture_frames = self._processed_frames[:N_CHANNELS, -self._gesture_frame_len:]
                        # 测试socket
                        # self._gesture_action_multithread(gesture_frames, socket_client)
                        if self._waken:
                            # 已经唤醒
                            self._gesture_action_multithread(gesture_frames, self.gesture_recognition)
                        else:
                            # 没有唤醒
                            self._gesture_action_multithread(gesture_frames, self.gesture_wake)
            elif self.recording_gesture == 1 or self.recording_gesture == 3:
                pass
                # print(1)
            elif self.recording_gesture == 2:
                self._recording_frame_process(next_frame)
            self.lock.release()

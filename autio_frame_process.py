from concurrent.futures.thread import ThreadPoolExecutor

from config import *
import log
import queue
import numpy as np
import socket

from audio_util import butter_bandpass_filter, get_cos_IQ_raw_offset, butter_lowpass_filter, get_phase, get_cos_IQ_raw, \
    move_average_overlap_filter, padding_or_clip


def get_pair(wake_gesture_data, input_gesture):
    pass

# test
def socket_client(buffer):
    address = ('127.0.0.1', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    tcp_socket.send(buffer)


class WakeOrRecognition:
    def __init__(self, audio_queue: queue.Queue, reco_model, wake_model, wake_gesture_data):
        self._audio_queue = audio_queue
        self.reco_model = reco_model
        self.wake_model = wake_model
        self.wake_gesture_data = wake_gesture_data

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
        unwrapped_phase_list = [None] * NUM_OF_FREQ

        def get_phase_and_diff(i):
            fc = F0 + i * STEP
            data_filter = butter_bandpass_filter(gesture_frames, fc - 150, fc + 150)
            I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, FS)
            # 滤波+下采样
            I = move_average_overlap_filter(I_raw[:, I_Q_skip:-I_Q_skip])
            Q = move_average_overlap_filter(Q_raw[:, I_Q_skip:-I_Q_skip])
            # denoise,暂时不用

            unwrapped_phase = get_phase(I, Q)  # 这里的展开目前没什么效果
            # import matplotlib.pyplot as plt
            # if i == 0:
            #     plt.figure()
            #     for j in range(7):
            #         plt.subplot(4, 2, j + 1)
            #         plt.plot(unwrapped_phase[j])
            #     plt.show()
            unwrapped_phase_diff = np.diff(unwrapped_phase)
            # pad是不是可以放到后面做
            unwrapped_phase_diff_padded = padding_or_clip(unwrapped_phase_diff, PADDING_LEN)
            unwrapped_phase_list[i] = unwrapped_phase_diff_padded

        with ThreadPoolExecutor(max_workers=8) as pool:
            pool.map(get_phase_and_diff, [i for i in range(NUM_OF_FREQ)])
        merged_unwrapped_phase = np.array(unwrapped_phase_list).reshape(data_shape)
        action(merged_unwrapped_phase)

    def gesture_recognition(self, phase_data):
        y_predict = self.reco_model.predict(phase_data.reshape((1, phase_data.shape[0], phase_data.shape[1], 1)))
        label = ['握紧', '张开', '左滑', '右滑', '上滑', '下滑', '前推', '后推', '顺时针转圈', '逆时针转圈']
        print(np.argmax(y_predict[0]))
        print(label[np.argmax(y_predict[0])])

    def gesture_wake(self, phase_data):
        input_gesture = phase_data.reshape((phase_data.shape[0], phase_data.shape[1], 1))
        input_pairs = get_pair(self.wake_gesture_data, input_gesture)
        y_predict = self.wake_model.predict([input_pairs[:, 0], input_pairs[:, 1]])

        dist = np.mean(y_predict)
        print(f'相似度：{dist}')
        if dist < 0.5:
            print("\033[1;31m wake gesture\033[0m")
            # 什么时候变成Flase？
            self._waken = True
        else:
            print("not wake gesture")

    def run(self):
        # 直接在主线程中运行
        while True:
            next_frame = self._get_next_frame()  # 会阻塞
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
                    self._gesture_action_multithread(gesture_frames, self.gesture_recognition)
                    # if self._waken:
                    #     # 已经唤醒
                    #     self._gesture_action_multithread(gesture_frames, self.gesture_recognition)
                    # else:
                    #     # 没有唤醒
                    #     self._gesture_action_multithread(gesture_frames, self.gesture_wake)

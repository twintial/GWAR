from config import *
import log
import queue
import numpy as np

from audio_util import butter_bandpass_filter, get_cos_IQ_raw_offset, butter_lowpass_filter, get_phase


class WakeOrRecognition:
    def __init__(self, audio_queue: queue.Queue):
        self._audio_queue = audio_queue
        self._processed_frames = None

        self._waken = False

        self._motion_start = False
        self._motion_end = False

        self._offset = 0

        self._max_frame_count_in_memory = FS * 3  # 3s

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
        phase = self._frame2phase(frame_segment)
        std = np.std(phase)
        if self._motion_start:
            if std < THRESHOLD:
                pass
        else:
            if std > THRESHOLD:
                pass

    def run(self):
        # 直接在主线程中运行
        while True:
            next_frame = self._get_next_frame()
            self._frame_data_process(next_frame)
            if self._processed_frames.shape[1] > self._max_frame_count_in_memory:
                self._processed_frames = self._processed_frames[:, CHUNK:]
            if self._processed_frames.shape[1] > 3 * CHUNK:
                # 前后都多拿一个CHUNK
                frame_segments = self._processed_frames[:, -3 * CHUNK:]
                self._motion_end = self._motion_detection(frame_segments[0].reshape(1, -1))
                if self._motion_end:
                    if self._waken:
                        pass
                    else:
                        pass

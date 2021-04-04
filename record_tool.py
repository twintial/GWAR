import wave
import queue
import pyaudio
import log
from config import *


class PlayRecorder:
    def __init__(self, input_id, output_id, channels, audio_queue: queue.Queue):
        self._audio_queue = audio_queue
        self.wav_file = None

        self.signal = None
        self.cursor = 0

        self.t = 0

        self.chunk = CHUNK
        self.fs = FS
        self.format = pyaudio.paInt16  # int24在数据处理上不方便,改为int16,注意和input_callback中的np.int对于
        self._recording = False
        self._playing = False

        self.input_device_index = None
        self.output_device_index = None
        self.channels = None

        self._p = pyaudio.PyAudio()
        self._record_stream = None
        self._play_stream = None

        self._set_param(input_id, output_id, channels)

    def _set_param(self, input_id, output_id, channels):
        self.input_device_index = input_id
        self.output_device_index = output_id
        self.channels = channels

    def input_callback(self, in_data, frame_count, time_info, status_flags):
        # in_data是byte类型，放到一个queue中去
        try:
            self._audio_queue.put(in_data, block=True, timeout=1)
        except queue.Full:
            log.logger.warning('Audio queue is full because of processing audio frame too slowly')
        self.t = self.t + frame_count / self.fs
        print(self.t)
        return in_data, pyaudio.paContinue

    def output_callback(self, in_data, frame_count, time_info, status_flags):
        out_data = self.signal[self.cursor:self.cursor + frame_count]
        self.cursor = self.cursor + frame_count
        if self.cursor + frame_count > len(self.signal):
            self.cursor = 0
        return out_data, pyaudio.paContinue

    def wavfile_output_callback(self, in_data, frame_count, time_info, status_flags):
        out_data = self.wav_file.readframes(frame_count)
        return out_data, pyaudio.paContinue

    def play_signal(self, signal):
        # signal可能是一个wav文件名称或者是一串信号
        self._playing = True
        self.signal = signal
        if type(signal) == str:
            self.wav_file = wave.open(signal, "rb")
            self._play_stream = self._p.open(format=self._p.get_format_from_width(self.wav_file.getsampwidth()),
                                             channels=self.wav_file.getnchannels(),
                                             rate=self.wav_file.getframerate(),
                                             output=True,
                                             output_device_index=self.output_device_index,
                                             frames_per_buffer=self.chunk,
                                             stream_callback=self.wavfile_output_callback
                                             )
        else:
            self._play_stream = self._p.open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=self.fs,
                                             output=True,
                                             output_device_index=self.output_device_index,
                                             frames_per_buffer=self.chunk,
                                             stream_callback=self.output_callback)
        self._play_stream.start_stream()

    def record(self):
        self._recording = True
        self._record_stream = self._p.open(format=self.format,
                                           channels=self.channels,
                                           rate=self.fs,
                                           input=True,
                                           input_device_index=self.input_device_index,
                                           frames_per_buffer=self.chunk,
                                           stream_callback=self.input_callback)
        self._record_stream.start_stream()

    def play_and_record(self, signal):
        if signal is not None:
            self.play_signal(signal)
        self.record()

    def stop(self):
        if self._recording:
            self._record_stream.stop_stream()
            self._record_stream.close()
            self._recording = False

        if self._playing:
            self._play_stream.stop_stream()
            self._play_stream.close()
            self._playing = False

        if type(self.signal) == str:
            self.wav_file.close()

        self.t = 0

    def terminate(self):
        self._p.terminate()
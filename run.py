from autio_frame_process import WakeOrRecognition
import queue

from record_tool import PlayRecorder

audio_queue = queue.Queue(maxsize=10)


def main():
    signal = ''
    # play_recorder = PlayRecorder()
    # play_recorder.play_and_record(signal)
    # wr = WakeOrRecognition(audio_queue)
    # wr.run()


def test():
    pass


if __name__ == '__main__':
    # 没用归一化
    # main()
    test()
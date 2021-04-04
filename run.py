from autio_frame_process import WakeOrRecognition
import queue

audio_queue = queue.Queue()


def main():
    wr = WakeOrRecognition(audio_queue)
    wr.run()


if __name__ == '__main__':
    main()

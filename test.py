from queue import Queue


class A:
    def __init__(self, queue):
        self.queue: Queue = queue

    def get(self):
        return self.queue.get()

    def put(self, item):
        self.queue.put(item)


class B:
    def __init__(self, queue):
        self.queue: Queue = queue

    def get(self):
        return self.queue.get()

    def put(self, item):
        self.queue.put(item)


if __name__ == '__main__':
    # q = Queue()
    # sjj = A(q)
    # b = B(q)
    # sjj.put(push)
    # print(b.get())
    # while not q.empty():
    #     print(q.get())
    import numpy as np
    # sjj = np.array([[1,2,3],[3,4,6]])
    # print(b''.join(sjj))
    # x = np.frombuffer(sjj, dtype=np.int32)
    # print(x.reshape(sjj.shape))

    a = []
    a.append([[1,2],[3,4]])
    a.append([[5,6],[7,8]])
    a.append([[5,6],[7,8]])
    a.append([[5,6],[7,8]])
    print(np.hstack(a))

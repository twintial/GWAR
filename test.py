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
    # a = A(q)
    # b = B(q)
    # a.put(push)
    # print(b.get())
    # while not q.empty():
    #     print(q.get())
    import numpy as np
    a = np.array([[1,2,3],[3,4,6]])
    print(b''.join(a))
    x = np.frombuffer(a, dtype=np.int32)
    print(x.reshape(a.shape))

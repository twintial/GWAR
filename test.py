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
    q = Queue()
    a = A(q)
    b = B(q)
    a.put(1)
    print(b.get())
    while not q.empty():
        print(q.get())

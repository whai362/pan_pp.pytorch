import heapq


class TopK(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def reset(self):
        self.data = []

    def size(self):
        return len(self.data)

    def push(self, x):
        if len(self.data) < self.k:
            heapq.heappush(self.data, x)
        else:
            heapq.heappushpop(self.data, x)

    def extract(self, sort=False):
        if sort:
            self.data.sort(reverse=True)
        return self.data

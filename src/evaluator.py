class Evaluator:
    def __init__(self):
        self.clicks = 0
        self.steps = 0

    def log(self, r):
        self.steps += 1
        self.clicks += r

    def ctr(self):
        return self.clicks / self.steps

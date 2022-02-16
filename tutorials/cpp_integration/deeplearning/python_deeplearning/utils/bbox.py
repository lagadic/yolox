

class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, score, cls):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.score = score
        self.cls = int(cls)

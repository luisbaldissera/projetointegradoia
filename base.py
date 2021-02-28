import math

class ClassificationResults(object):
    def __init__(self, x, y, accurracy):
        self._x = x
        self._y = y
        self._accuracy = accurracy
    def accurracy(self):
        return self._accuracy
    def x(self):
        return self._x
    def y(self):
        return self._y
        
def euclidean(a, b):
    diff = [ (i-j) * (i-j) for i,j in zip(a,b) ]
    return math.sqrt(math.fsum(diff))


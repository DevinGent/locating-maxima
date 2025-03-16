import numpy as np
import copy

class Coordinates:
    def __init__(self, xs: np.array, ys:np.array):
        self.x=xs
        self.y=ys



array1=np.array((1,2,3,4),dtype=float)
print(np.insert(array1,4,10))

print(array1.dtype)
print(np.insert(array1,3,7.162))
print(np.insert(array1,3,7.162))
import numpy as np
import copy

class Coordinates:
    def __init__(self, xs: np.array, ys:np.array):
        self.x=xs
        self.y=ys



array1=np.array((1,2,3,4))
print(array1, type(array1))
array2=np.array([4,5,6,7])

coord_1=Coordinates(array1,array2)
coord_2=copy.copy(coord_1)

print(coord_1.x)
print(coord_2.x)

coord_1.x=np.insert(coord_1.x,3,87)
print(coord_1.x)
print(coord_2.x)

array3=np.array([1,2,3,4,5])
print(np.insert(array3,-1,30)[-1])
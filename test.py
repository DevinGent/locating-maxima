import numpy as np
import copy
import pandas as pd


class Coordinates:
    def __init__(self, xs: np.array, ys:np.array):
        self.x=xs
        self.y=ys



number_of_graphs=7
print(number_of_graphs)
full_graphs=number_of_graphs//8
print(full_graphs)
number_of_graphs=number_of_graphs%8
print(number_of_graphs)
print('DONE')
for i in range(0):
    print("i is {}".format(i))

array1=np.array((6,2,7,4),dtype=float)

array2=np.array([1,9,8,13])

zipped=zip(array1,array2)

list_zip=list(zipped)
print("A1:",array1)
print("A2:",array2)
print('Zipped',list_zip)
sorted_zip=sorted(list_zip)
print('Sorted Zipped',sorted_zip)
print('Sorted Zipped',list(sorted_zip))
print(*sorted_zip)
print(list(zip(*sorted_zip)))
x,y=zip(*sorted_zip)
print(x)
print(y)

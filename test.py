import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt


class Coordinates:
    def __init__(self, xs: np.array, ys:np.array):
        self.x=xs
        self.y=ys
        self.square_it()
    
    def square_it(self):
        self.x=np.array([x*x for x in self.x])


coord=Coordinates(np.array([3,7,8]),np.array([9,-2,4,7]))

print(coord.x)
print(coord.y)
print(len(coord.x))
print(np.insert(coord.x,-1,800))
starting=[(1,3),(-8,4),(.2,1)]
print('starting',starting)
print(sorted(starting))
print(list(zip(*sorted(starting))))
print(np.unique(coord.x))
print([i for i in range(len(coord.x))])
string_math="3+5*7"
print(string_math)
print(eval(string_math))

def do_x_to_y(x, y):
    return eval(x)

print(do_x_to_y('y+3',5))

results_df=pd.DataFrame(columns=['Known Points','RoI','Least Possible Maximum','Greatest Possible Maximum'])
results_df.set_index('Known Points',inplace=True)   
print(results_df.info())

print(results_df.head())

print(float('3.4'))
x=9
if x>=5 and x<=10:
    print("Success.")
else:
    print("Failure.")


df=pd.DataFrame(zip([1,2],[10,5],[5,15],[25,25]), columns=['Known Points','RoI','Least Possible Maximum','Greatest Possible Maximum'])
print(df.info())
print(df.head())
df.loc[-1]=[8,3,7,10]
print(df.head())

print(*(3,[1,2,5]))

l=[1,2,3,4,5]
print(l[-5])

plt.margins()
plt.close()
plt.figure(figsize=(7,5), num="Other Points Known")
plt.scatter([1,2,3],[10,-2,7])
plt.show()
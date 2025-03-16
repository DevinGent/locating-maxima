"""Based on a project from my undergraduate education, this script will work through a method for determining the 
maxima and minima of a Lipschitz continuous function over a given interval.  For information on Lipschitz continuity see:
https://en.wikipedia.org/wiki/Lipschitz_continuity


"""

import matplotlib.pyplot as plt
from fractions import Fraction
import numpy as np
import pandas as pd
import math
import random
from bisect import bisect_left



class Graph:
    """
    The plotting data for an axis along with a function to plot to a specified axis.
    """
    def __init__(self, xs: np.array, ys:np.array, interval: tuple):
        self.xs=xs
        self.ys=ys
        self.interval=interval

    def draw(self, axis):
            """Draws the graph onto the given axis."""
            axis.plot(self.xs,self.ys)


class Coordinates:
    """
    Maintains four arrays and methods to update them. One for the known x values, one for the known y values, 
    one for the x values where the function could reach a maximum on each interval, and one for the corresponding maximum y values."""
    def __init__(self, xs: np.array, ys:np.array, interval_xs: np.array, interval_ys:np.array):
        self.known_x=xs
        self.known_y=ys
        self.interval_x=interval_xs
        self.interval_y=interval_ys

    def update_arrays(self, x, y, index, interval,lipschitz_constant):
        """Takes an x and y value to insert in the known value arrays as well as an index in which to insert them, then updates
        all the arrays."""

        if index==0:
            if index==len(self.known_x):
                self.interval_x=np.array(interval,dtype=float)
                self.interval_y=np.array([y-lipschitz_constant*(interval[0]-x),
                                           y+lipschitz_constant*(interval[1]-x)],dtype=float)
            else:
                # The dtype of the array is wrong.  I should make sure all the arrays are initiated to be float arrays.
                print('New int:',(self.known_y[0]-y+lipschitz_constant*(self.known_x[0]+x))/(2*lipschitz_constant))
                print(np.insert(self.interval_x,1,(self.known_y[0]-y+lipschitz_constant*(self.known_x[0]+x))/(2*lipschitz_constant)).dtype)
                print(np.insert(self.interval_x,1,(self.known_y[0]-y+lipschitz_constant*(self.known_x[0]+x))/(2*lipschitz_constant)))
                print(np.insert(self.interval_x,1,8))
                self.interval_x=np.insert(self.interval_x,1,(self.known_y[0]-y+lipschitz_constant*(self.known_x[0]+x))/(2*lipschitz_constant))
                self.interval_y=np.insert(self.interval_y,1,(self.known_y[0]+y+lipschitz_constant*(self.known_x[0]-x))/2)
                self.interval_y[0]=y-lipschitz_constant*(interval[0]-x)

        elif index==len(self.known_x):
            self.interval_x=np.insert(self.interval_x,-1,(y-self.known_y[-1]+lipschitz_constant*(self.known_x[-1]+x))/(2*lipschitz_constant))
            self.interval_y=np.insert(self.interval_y,-1,(self.known_y[-1]+y+lipschitz_constant*(x-self.known_x[-1]))/2)
            self.interval_y[-1]=y+lipschitz_constant*(interval[1]-x)
        
        else:
            self.interval_x=np.insert(self.interval_x,index,(y-self.known_y[index-1]+lipschitz_constant*(self.known_x[index-1]+x))/(2*lipschitz_constant))
            self.interval_x[index+1]=(self.known_y[index]-y+lipschitz_constant*(self.known_x[index]+x))/(2*lipschitz_constant)
            self.interval_y=np.insert(self.interval_x,index,(self.known_y[index-1]+y+lipschitz_constant*(x-self.known_x[-1]))/2)
            self.interval_y[index+1]=(self.known_y[index]+y+lipschitz_constant*(self.known_x[index]-x))/2

        self.known_x=np.insert(self.known_x,index,x)
        self.known_y=np.insert(self.known_y,index,y)


def sample_function(x):
    """Given an x value (input) will output the correct value of the function (f(x))."""
    return x*x*x-x*x

def random_function(x,lipschitz_constant,previous_coordinates=None,next_coordinates=None):
    """Given an x value, the Lipschitz constant, and (optionally) a pair of tuples each of the form (x_0,y_0),
    produces a random output for y such that (x,y), when added to the known coordinates, will still adhere to the Lipschitz constraint.
    The variables previous_coordinates and next_coordinates should represent the nearest known coordinates to the left and right
    of x respectively."""
    y_low=-50
    y_high=50
    if previous_coordinates is not None:
        if next_coordinates is not None:
            y_high=min(
                previous_coordinates[1]+lipschitz_constant*(x-previous_coordinates[0]),
                next_coordinates[1]-lipschitz_constant*(x-next_coordinates[0]))
            y_low=max(
                previous_coordinates[1]-lipschitz_constant*(x-previous_coordinates[0]),
                next_coordinates[1]+lipschitz_constant*(x-next_coordinates[0])
            )
        else:
            y_high=previous_coordinates[1]+lipschitz_constant*(x-previous_coordinates[0])
            y_low=previous_coordinates[1]-lipschitz_constant*(x-previous_coordinates[0])
    elif next_coordinates is not None:
        y_high=next_coordinates[1]-lipschitz_constant*(x-next_coordinates[0])
        y_low=next_coordinates[1]+lipschitz_constant*(x-next_coordinates[0])
    print(y_low,y_high)
    return random.uniform(y_low,y_high)



def radius_of_information(greatest_max,least_max):
    """Returns the radius of information given the greatest possible maximum and the smallest possible maximum on the interval."""
    return (greatest_max-least_max)/2

def get_results(known_y, interval_y):
    """Given two arrays consisting of the known y values and the greatest y values on each interval, produces a row for the results
    dataframe and a check to see if the program should be terminated early."""
    least_max=max(known_y)
    greatest_max=max(interval_y)
    greatest_max=max(least_max,greatest_max)
    roi=radius_of_information(greatest_max,least_max)
    finished=False
    if roi==0:
        finished=True
    return ([roi,least_max,greatest_max],finished)


def is_fraction(string):
    """Tests to see if a user input is a valid number by checking if it is a fraction or float."""
    try:
        Fraction(string)
        return True
    except ValueError:
        return False

def next_x(interval,lipschitz_constant,known_x,known_y,interval_y):
    """Given information on the interval, the Lipschitz constant, the known coordinates, and the greatest possible y values between
    any pair of points, produces a pair of the form (new x coordinate, index in which to insert it in known_x)"""
    if len(known_x)==0:
        return((interval[0]+interval[1])/2,0)
    interval_index=np.argmax(interval_y)
    print("New TURN!")
    print('Interval y', interval_y)
    print("Int index:",interval_index)
    if interval_index==0:
        next_x_value = (2*interval[0]+known_x[0])/3
    elif interval_index==len(interval_y)-1:
        next_x_value = (2*interval[1]+known_x[-1])/3
        print('Inner x is',next_x_value)
    else:
        next_x_value = (known_y[interval_index]-known_y[interval_index-1]+
                        lipschitz_constant*(known_x[interval_index]+known_x[interval_index-1]))/(2*lipschitz_constant)

    return (next_x_value,interval_index)

def next_y(x,x_index, lipschitz_constant,known_x,known_y, function_type):
    if function_type=='sample':
        return sample_function(x) 
    elif function_type=='optimal':
        if len(known_y)==0:
            return random.uniform(-50,50)
        else:
            return known_y[0]
    elif function_type=='random':
        if len(known_y)==0:
            return random.uniform(-50,50)
        if x_index==0:
            return random_function(x,lipschitz_constant,next_coordinates=(known_x[0],known_y[0]))
        elif x_index==len(known_x)-1:
            return random_function(x,lipschitz_constant,previous_coordinates=(known_x[-1],known_y[-1]))
        else:
            return random_function(x,
                                   lipschitz_constant,
                                   previous_coordinates=(known_x[x_index-1],known_y[x_index-1]),
                                   next_coordinates=(known_x[x_index],known_y[x_index]))


def adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type):
    """Selects x values turn by turn one at a time instead of all at once."""

    coordinates=Coordinates(np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float))

    y_optimal=random.uniform(-20,20)

    # Eventually we may add the capability to select some x
    #x_optimal=True

    for turn in range(number_of_x):
        x_to_insert=next_x(interval,lipschitz_constant,coordinates.known_x,coordinates.known_y,coordinates.interval_y)
        y_to_insert=next_y(*x_to_insert,lipschitz_constant,coordinates.known_x,coordinates.known_y,function_type)
        print()
        print("TURN", turn)
        print("x to insert:",x_to_insert)
        print("y to insert:",y_to_insert)
        print('Known x:', coordinates.known_x)
        print('Known y:', coordinates.known_y)
        print('Interval x:',coordinates.interval_x)
        print('Interval y:',coordinates.interval_y)
        print('y len',len(coordinates.known_y))

        coordinates.update_arrays(x_to_insert[0],y_to_insert,x_to_insert[1],interval,lipschitz_constant)
    print(coordinates.known_x)
    print(coordinates.known_y)


def non_adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type,results_df):
    print("This part of the program is still being worked on.")

def main():
    """The main code."""


    print(random_function(3,5,(1,2),(8,7)))

    array1=np.array([9,4,7,10.3625])
    print(array1)
    print(type(array1))

    print(np.insert(array1,2,14.44))

    # Enter the interval and Lipschitz constant here.  All calculations will include the endpoints of the interval.
    interval=(0,1)
    lipschitz_constant=1

    # Set the number of x values that will be chosen.
    number_of_x=5

    # Set this to True for an adaptive strategy (x values are chosen one at a time) and false for 
    # a non-adaptive strategy (all x values are chosen at once.)
    adaptive=True

    # This variable determines how the computer chooses y values for the selected x values. The options are:
    # 'random' - y values are randomly chosen which satisfy the lipschitz condition
    # 'sample' - y values are decided from a pre-decided function (which can be edited in the sample_function above).
    # 'optimal' - y values are always those which maximize the radius of information (and weaken our prediction of maxima and minima) 
 
    function_type='sample'



    adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type)

    df=pd.DataFrame(columns=['A','B'])
    df.loc[1]=[5,'Yes!']
    print(df)



    results_df=pd.DataFrame(columns=['RoI','Least Maximum','Greatest Maximum'])
    results_df.index +=1
    results_df.index.name='x Chosen'

    if adaptive==True:
        adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type,results_df)
    else:
        non_adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type,results_df)



    # Add a flag here for incorrect variable type.
    known_x=np.array([1,2,3,4])
    known_y=np.array([3,6,9,3])
    print(known_x)
    print(known_y)
    plt.plot(known_x,known_y)
    plt.show()
    graph=Graph(known_x,known_y,interval)
    print(graph.xs)
    known_x=np.insert(known_x,2,100)
    print(known_x)
    print(graph.xs)



    known_points=[(1,6),(2,8),(3,-1),(4,7),(5,5)]
    print(known_points)
    print(known_points[:])
    fig, axs = plt.subplots(1,2,figsize=(6,6))
    axs=axs.flatten()
    axs[0].plot([i[0] for i in known_points], [i[1] for i in known_points])

    axs[1].plot(known_points[:])
    plt.show()

    print(type(None))
    print(type((3,4.2)))
    if type((3.7,4.8))==tuple:
        print("YYYYYYES!")
    print()


    underlying_function= pd.DataFrame()
    underlying_function['x']=np.linspace(interval[0],interval[1],10000)
    underlying_function['y']=underlying_function['x'].map(sample_function)



    plt.plot(underlying_function['x'],underlying_function['y'])
    plt.axvline(x_min,color='red',linestyle='dashed')
    plt.axvline(x_max,color='red',linestyle='dashed')
    plt.show()

    # Graphing results:
    """
        if len(rows_to_plot)==2:
        fig, axs = plt.subplots(1,2,figsize=(6,6))
    elif len(rows_to_plot)==3:
        fig, axs = plt.subplots(2,2,figsize=(12,6))
        # We only want three axes so we delete the lower right one.
        fig.delaxes(axs[1,1])
    elif len(rows_to_plot)==4:
        fig, axs = plt.subplots(2,2,figsize=(12,6))
    # In any case we now want to flatten the axes so we can call plot_pie()
    axs=axs.flatten()
    for ax in axs:
        graphs[0].draw(ax) # graphs is a list of Graph class objects.
    fig.suptitle('{} Turns'.format{number_of_turns})
    plt.tight_layout()
    plt.savefig(file_path+'finding_maxima_and_minima') # optional
    plt.show()
    """



if __name__ == '__main__':
    main()
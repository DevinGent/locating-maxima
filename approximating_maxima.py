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

def sample_function(x):
    """Given an x value (input) will output the correct value of the function (f(x))."""
    return x*x+math.sin(x)

def random_function(x,lipschitz_constant,previous_coordinates='None',next_coordinates='None'):
    if type(previous_coordinates)==set:
        print('Woo!')
        random.uniform(y_low,y_high)

    

def is_fraction(string):
    """Tests to see if a user input is a valid number by checking if it is a fraction or float."""
    try:
        Fraction(string)
        return True
    except ValueError:
        return False

def radius_of_information(max,min):
    return (max-min)/2

def adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type):
    pass

def non_adaptive_strategy():
    pass

def main():
    """The main code."""


    # Enter the interval and Lipschitz constant here.  All calculations will include the endpoints of the interval.
    interval=(2,65)
    lipschitz_constant=3

    # Set the number of x values that will be chosen.
    number_of_x=5

    # Set this to True for an adaptive strategy (x values are chosen one at a time) and false for 
    # a non-adaptive strategy (all x values are chosen at once.)
    adaptive=True

    # This variable determines how the computer chooses y values for the selected x values. The options are:
    # 'random' - y values are randomly chosen which satisfy the lipschitz condition
    # 'sample' - y values are decided from a pre-decided function (which can be edited in the sample_function above).
    # 'optimal' - y values are always those which maximize the radius of information (and weaken our prediction of maxima and minima) 
 
    function_type='random'

    if adaptive==True:
        adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type)
    else:
        non_adaptive_strategy(interval,lipschitz_constant,number_of_x,function_type)



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
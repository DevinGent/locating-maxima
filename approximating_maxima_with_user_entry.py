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


def sample_function(x):
    """Given an x value (input) will output the correct value of teh function (f(x))."""
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

def main():
    """The main code."""

    """
    while True:
        x_min=input("Enter the left boundary of the interval: ")
        if is_fraction(x_min)==True:
            x_min=Fraction(x_min)
            x_min=float(x_min)
            break
        else:
            print("Please enter a number.")

    while True:
        x_max=input("Enter the right boundary of the interval: ")
        if is_fraction(x_max)==True:
            x_max=Fraction(x_max)
            x_max=float(x_max)
            if x_max>x_min:
                break
            else:
                print("The right end of the boundary must be greater than", x_min)
        else:
            print("Please enter a number.")

    while True:
        lipschitz_constant=input("Enter the Lipschitz constant: ")
        if is_fraction(lipschitz_constant)==True:
            lipschitz_constant=Fraction(lipschitz_constant)
            lipschitz_constant=float(lipschitz_constant)
            break
        else:
            print("Please enter a number.")
    """


    # Just for testing.  Remove this later.
    x_min=2
    x_max=65
    lipschitz_constant=3

    print(type(None))
    print(type((3,4.2)))
    if type((3.7,4.8))==tuple:
        print("YYYYYYES!")
    print()


    underlying_function= pd.DataFrame()
    underlying_function['x']=np.linspace(x_min,x_max,10000)
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
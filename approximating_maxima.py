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
import copy



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
            if len(self.known_x)==0:
                self.interval_x=np.array(interval,dtype=float)
                self.interval_y=np.array([y-lipschitz_constant*(interval[0]-x),
                                           y+lipschitz_constant*(interval[1]-x)],dtype=float)
            else:
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
            self.interval_y=np.insert(self.interval_y,index,(self.known_y[index-1]+y+lipschitz_constant*(x-self.known_x[index-1]))/2)
            self.interval_y[index+1]=(self.known_y[index]+y+lipschitz_constant*(self.known_x[index]-x))/2

        self.known_x=np.insert(self.known_x,index,x)
        self.known_y=np.insert(self.known_y,index,y)

    def draw(self, axis):
        """Draws a graph of the currently known points onto the target axis. The figure must still be shown to display."""
        axis.axvline(self.interval_x[0],color='red',linestyle='dashed')
        axis.axvline(self.interval_x[-1],color='red',linestyle='dashed')
        axis.scatter(self.known_x,self.known_y)
        # We now combine the interval and known points to create a line graph.
        zipped=zip(np.concatenate([self.interval_x,self.known_x]),np.concatenate([self.interval_y,self.known_y]))
        zipped=sorted(list(zipped))
        x,y= zip(*zipped)
        axis.plot(x,y, linestyle='dashed')
        points_known=len(self.known_x)
        if points_known==1:
            axis.set_title("{} Point Known".format(points_known))
        else:
            axis.set_title("{} Points Known".format(points_known))





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
    known_points=len(known_y)
    least_max=max(known_y)
    greatest_max=max(interval_y)
    greatest_max=max(least_max,greatest_max)
    roi=radius_of_information(greatest_max,least_max)
    finished=False
    if roi==0:
        finished=True
    return ([known_points,roi,least_max,greatest_max],finished)


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

    # Keeps track of the known coordinates and intersections.
    coordinates=Coordinates(np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float))
    # Holds rows which will be converted into a dataframe.
    result_rows=[]
    # Holds previous coordinates so that they can be drawn at the end.
    graphs=[]
    # Allows the process to terminate prematurely if the maximum is located.
    stop=False
    # Sets a fixed value for the y value which is optimal.
    y_optimal=random.uniform(-20,20)

    # Eventually we may add the capability to select custom x instead of just optimal x.
    # x_optimal=True

    # number_of_x holds the total number of points to choose.  One by one an x value is chosen, the corresponding y value is found
    # and the coordinates are updated. 
    for turn in range(number_of_x):
        # Determining the next x value to add.
        x_to_insert=next_x(interval,lipschitz_constant,coordinates.known_x,coordinates.known_y,coordinates.interval_y)

        # Determining what the paired y value should be.
        y_to_insert=next_y(*x_to_insert,lipschitz_constant,coordinates.known_x,coordinates.known_y,function_type)

        # Updating the arrays of the coordinate class object.
        coordinates.update_arrays(x_to_insert[0],y_to_insert,x_to_insert[1],interval,lipschitz_constant)

        # Adding a copy of the current coordinate class object to the list of graphs.
        graphs.append(copy.copy(coordinates))

        # Obtaining the next row of the dataframe and checking if the loop can be terminated early.
        (new_row,stop)=get_results(coordinates.known_y,coordinates.interval_y)

        # Adding the new row to the list of rows.
        result_rows.append(new_row)

        # Checking if the process should be ended early.
        if stop==True:
            break
    
    # Constructing a dataframe from the rows of results and indexing by the number of known points.
    results_df=pd.DataFrame(result_rows,columns=['Known Points','RoI','Least Possible Maximum','Greatest Possible Maximum'])
    results_df.set_index('Known Points',inplace=True)    

    # Displaying the dataframe.
    print(results_df.head(number_of_x))

    print(coordinates.known_x)
    print(coordinates.known_y)

    # Determining the number of graphs to plot.  Up to 8 can be plotted in a single figure.
    number_of_graphs=len(graphs)
    # Determining how many full figures will be plotted.
    full_graphs=number_of_graphs//8
    # Determining how many graphs should appear in the partial figure (if any)
    number_of_graphs=number_of_graphs%8

    # For each full window/figure a figure and axes are created and then displayed.
    for window in range(full_graphs):
        # Each full window should include two rows of graphs with four graphs in each row.
        fig, axs = plt.subplots(2,4,figsize=(12,6))
        # Flatten the axes so they can be iterated through numerically more easily.
        axs=axs.flatten()
        # Drawing each of the graphs.
        for i in range(len(axs)):
            graphs[i+8*window].draw(axs[i])
        # All the graphs should have the same x and y limits. We take these limits from the first graph of the first window.
        if window==0:
            xlimits=axs[0].get_xlim()
            ylimits=axs[0].get_ylim()
        # Setting the x and y limits of all the graphs on the current figure.
        plt.setp(axs, ylim=ylimits,xlim=xlimits)
        plt.tight_layout()
        # Displaying without blocking so the user can flip between multiple windows and compare.
        plt.show(block=False)

    # The layout of the last figure is determined by how many graphs remain.
    if number_of_graphs<3:
        fig, axs = plt.subplots(1,number_of_graphs,figsize=(12,6))
    elif number_of_graphs<5:
        fig, axs = plt.subplots(2,2,figsize=(12,6))
    elif number_of_graphs<7:
        fig, axs = plt.subplots(2,3,figsize=(12,6))
    elif number_of_graphs<9:
        fig, axs = plt.subplots(2,4,figsize=(12,6))
    else:
        raise ValueError('The number of graphs should not exceed 8 on any one figure.')
    axs=axs.flatten()

    # Each graph is drawn.
    for i in range(number_of_graphs):
        graphs[i+8*full_graphs].draw(axs[i])
    # The excess axes are removed.
    for i in range(number_of_graphs,len(axs)):
        fig.delaxes(axs[i])    

    # If there were no complete figures the x and y limits are chosen by the first graph of this incomplete figure.
    if full_graphs==0:
        xlimits=axs[0].get_xlim()
        ylimits=axs[0].get_ylim()
    plt.setp(axs, ylim=ylimits,xlim=xlimits)
    plt.tight_layout()
    plt.show(block=False)

    # A final figure is included showing, in greater size, the final configuration achieved.
    plt.figure(figsize=(10,5), num="{} Points Known".format(len(graphs[-1].known_x)))
    ax=plt.gca()
    graphs[-1].draw(ax)
    plt.show()


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
    number_of_x=6

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









if __name__ == '__main__':
    main()
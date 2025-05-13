"""Based on a project from my undergraduate education, this script will work through a method for determining the 
maxima and minima of a Lipschitz continuous function over a given interval.  For information on Lipschitz continuity see:
https://en.wikipedia.org/wiki/Lipschitz_continuity and the included README.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random




class ApproximateMaxima:
    """
    Assuming f is a Lipschitz continuous function on the given interval and with the given Lipschitz constraint, 
    determines the optimal choices of x values to approximate the maximum value of f. 
    
    A set of initial points can be given 
    as a list of (x,y) pairs for the optional argument starting_points. 
    
    The argument sample_function can be set, as a string, 
    to set the function f explicitly for testing purposes. sample_function must be a Lipschitz continuous function
    on the stated interval given as a string containing a function of x to be evaluated in Python's mathematics interpreter
    (see the README for further details).
    """

    def __init__(self, interval: tuple, lipschitz_constraint: float,starting_points=None,sample_function=None):
        
        # Setting variables using the given arguments.
        # The interval on which the function is Lipschitz continuous
        self.interval=interval
        # A Lipschitz constraint
        self.lipschitz_constraint=lipschitz_constraint
        # The sample function for testing.
        self.sample_function=sample_function

        # Additional variables which will be filled later are added.
        # The most recently added x value.
        self.latest_x=None
        # The most recently added y value.
        self.latest_y=None
        # The largest known y value.
        self.max_y=None
        # The largest possible y value the function could achieve given the Lipschitz constraint.
        self.max_possible_y=None
        # A list of graphs that can be displayed.
        self.graphs=[]
        # A dataframe displaying the number of points known and information on what the maximum value of the function could be.
        self.results_df=pd.DataFrame(columns=['Radius of Information','Least Possible Maximum','Greatest Possible Maximum','Approximated Maximum'])
        # We index the dataframe by the number of known points at each step.
        self.results_df.index.name='Known Points'

        # We set variables for the known x values (in ascending order), the known y values (matching the order of xs),
        # the x values where a maximum could occur between known points (_interval_x), and the corresponding y values (_interval_y)
        if starting_points==None:
            # If no starting points are given we set all the arrays to empty and set them to accept float entries.
            self.known_x=np.array([],dtype=float)
            self.known_y=np.array([],dtype=float)
            self._interval_x=np.array([],dtype=float)
            self._interval_y=np.array([],dtype=float)
        else:
            # If the optional argument starting_points was given, we sort and unpack the list of pairs into two individual 
            # lists representing the initial x values and initial ys.
            initial_x, initial_y=list(zip(*sorted(starting_points)))
            # We set these as the known x and y.
            self.known_x=np.array(initial_x,dtype=float)
            self.known_y=np.array(initial_y,dtype=float)
            # Check for errors and illegal x and y pairs.
            self._legal_arrays()

            # We set the maximum y value
            self.max_y=max(self.known_y)

            # Now define self._interval_x and self._interval_y using the following method based on the current known values.
            interval_xy=self._get_interval_xy()
            self._interval_x=np.array(interval_xy[0],dtype=float)
            self._interval_y=np.array(interval_xy[1],dtype=float)

            # Set the maximum possible y to be the largest y occurring on either the intervals or known points.
            self.max_possible_y=max(self._interval_y)
            self.max_possible_y=max(self.max_possible_y,self.max_y)

            # Add a graph of the starting configuration.
            self.add_graph()

            # Finally set the first row of the results dataframe based on the current points.
            (index, results)=self._get_results()
            self.results_df.loc[index]=results

        
    def _legal_arrays(self):
        """Checks if the current set of x and y values is possible given the interval and Lipschitz constraint.
        This method raises an error if it encounters an illegal x or y value."""

        # Check if the values (presorted) are inbounds.
        if self.known_x[0]<self.interval[0]:
            raise ValueError("At least one point, {}, does not lie on the interval [{},{}].".format((self.known_x[0],self.known_y[0]),*self.interval))
        if self.known_x[-1]>self.interval[1]:
            raise ValueError("At least one point, {}, does not lie on the interval [{},{}].".format((self.known_x[-1],self.known_y[-1]),*self.interval))
        
        # Check if there are any repeat occurrences of a given x value.
        if len(np.unique(self.known_x))!=len(self.known_x):
            raise ValueError("Each x value should only appear once.")
        
        # See if all values satisfy the Lipschitz constraint.
        for i in range(1,len(self.known_x)):
            # Starting at the second known point we determine what the greatest and least y value could be
            # for the given x value based on the preceding point and the Lipschitz constraint.
            y_low=self.known_y[i-1]-self.lipschitz_constraint*(self.known_x[i]-self.known_x[i-1])
            y_high=self.known_y[i-1]+self.lipschitz_constraint*(self.known_x[i]-self.known_x[i-1])
            # If the assigned y is below or above what is possible an error is raised.
            if self.known_y[i]< y_low or self.known_y[i]>y_high:
                raise ValueError("The pairs of points {} and {} violate the Lipschitz continuity with constraint {}.".format(
                    (self.known_x[i-1],self.known_y[i-1]),(self.known_x[i],self.known_y[i]),self.lipschitz_constraint))
        


    def _get_interval_xy(self):
        """Takes the current known x and y and constructs the points where a maximum could occur on the intervals between them."""

        # We develop the interval x and ys from left to right. First we consider the left endpoint.
        interval_x=[self.interval[0]]
        # We use point-slope form to acquire y.
        interval_y=[self.known_y[0]-self.lipschitz_constraint*(self.interval[0]-self.known_x[0])]

        # Next we add the points between each pair of known x and y
        for i in range(len(self.known_x)-1):
            interval_x.append(
                (self.known_y[i+1]-self.known_y[i]+self.lipschitz_constraint*(self.known_x[i+1]+self.known_x[i]))
                /(2*self.lipschitz_constraint)
                )
            interval_y.append(
                (self.known_y[i+1]+self.known_y[i]+self.lipschitz_constraint*(self.known_x[i+1]-self.known_x[i]))
                /(2)
                )
        # Finally we add the right endpoint.
        interval_x.append(self.interval[1])
        interval_y.append(self.known_y[-1]+self.lipschitz_constraint*(self.interval[1]-self.known_x[-1]))
        return(interval_x,interval_y)


    def _get_results(self):
        """Produces a row for the results dataframe and returns a pair of the form (row_index, row)."""

        known_points=len(self.known_y)
        least_max=self.max_y
        greatest_max=self.max_possible_y
        roi=radius_of_information(greatest_max,least_max)
        approx=(least_max+greatest_max)/2
        return((known_points,[roi,least_max,greatest_max,approx]))


    def add_graph(self):
        """Adds a graph of the current configuration to the list of graphs."""

        self.graphs.append(Graph(self.known_x,self.known_y,self._interval_x,self._interval_y,self.max_y,self.max_possible_y,self.latest_x,self.latest_y))


    def _get_manual_x(self):
        while True:
            user_input=input("Enter an x value (between {} and {}) as a decimal".format(self.interval[0],self.interval[1]))
            try:
                user_input=float(user_input)
                if user_input > self.interval[1] or user_input < self.interval[0]:
                    print("Your number must be between {} and {}".format(self.interval[0],self.interval[1]))
                elif user_input in self.known_x:
                    print("Each x value can only appear once and x={} is already known.".format(user_input))
                else:
                    return (user_input,np.searchsorted(self.known_x,user_input))
            except ValueError:
                print("Please enter as a decimal number.")

    def get_optimal_x(self):
        """Produces a pair of the form (new x coordinate, index in which to insert it in known_x)"""

        # If no points are currently known we select x to be the midway point on the interval
        # and set the index to insert to 0.
        if len(self.known_x)==0:
            return((self.interval[0]+self.interval[1])/2,0)
        
        # Otherwise we determine the index on _interval_y in which the maximum possible value occurs.
        interval_index=np.argmax(self._interval_y)
        # If the maximum occurs on the left end.
        if interval_index==0:
            next_x_value = (self.lipschitz_constraint*(2*self.interval[0]+self.known_x[0])+
                            self.known_y[0]-self.max_y)/(3*self.lipschitz_constraint)
        # If the maximum occurs on the right end.
        elif interval_index==len(self._interval_y)-1:
            next_x_value = (self.lipschitz_constraint*(2*self.interval[1]+self.known_x[-1])+
                            self.max_y-self.known_y[-1])/(3*self.lipschitz_constraint)
        # If the maximum occurs between two known points.
        else:
            next_x_value = (self.known_y[interval_index]-self.known_y[interval_index-1]+
                            self.lipschitz_constraint*(self.known_x[interval_index]+self.known_x[interval_index-1]))/(2*self.lipschitz_constraint)
        # Return the result.
        return (next_x_value,interval_index)
    
    def _legal_y(self, x, x_index):
        """Returns a pair (y_low,y_high) containing the maximum and minimum allowable y values for the given x value."""

        # If there are no known points we set the minimum to -infinity and the maximum to infinity.
        if len(self.known_y)==0:
                return (-np.inf,np.inf)
        # Otherwise we pick values that satisfies the Lipschitz constraint.
        # Picking on the left.
        if x_index==0:
            y_low=self.known_y[0]+self.lipschitz_constraint*(x-self.known_x[0])
            y_high=self.known_y[0]-self.lipschitz_constraint*(x-self.known_x[0])
        # Picking on the right.
        elif x_index==len(self.known_x):
            y_low=self.known_y[-1]-self.lipschitz_constraint*(x-self.known_x[-1])
            y_high=self.known_y[-1]+self.lipschitz_constraint*(x-self.known_x[-1])
        # Picking in the middle.
        else:
            y_low=max(
                self.known_y[x_index-1]-self.lipschitz_constraint*(x-self.known_x[x_index-1]),
                self.known_y[x_index]+self.lipschitz_constraint*(x-self.known_x[x_index])
            )
            y_high=min(
                self.known_y[x_index-1]+self.lipschitz_constraint*(x-self.known_x[x_index-1]),
                self.known_y[x_index]-self.lipschitz_constraint*(x-self.known_x[x_index])
            )
        return(y_low,y_high)

    def get_y(self, x, x_index, function_type):
        """Obtains a y value for the selected x value using the given function type."""

        legal_y=self._legal_y(x,x_index)

        if function_type=='sample':   
            # If no sample function has been defined we raise an exception.
            if self.sample_function==None:
                raise Exception("A sample function must be given before it can be used.")
            # Otherwise we evaluate the sample expression at the chosen x.
            else:
                new_y=eval(self.sample_function)
                if new_y <legal_y[0] or new_y>legal_y[1]:
                    raise Exception("The given function, {}, gives an illegal y value of {} when evaluated at x={} in violation of the Lipschitz constraint {}.".format(
                        self.sample_function,new_y,x,self.lipschitz_constraint
                    ))
                else:
                    return new_y
            
        elif function_type=='optimal':
            # If no points are currently known we choose at random in the range between -50 and 50.
            if len(self.known_y)==0:
                return random.uniform(-50,50)

            # Otherwise we pick the smaller value between the largest known y and the greatest possible y between the adjacent points.
            return min(self.max_y,legal_y[1])

                
        elif function_type=='random':
            # If no points are currently known we choose at random in the range between -50 and 50.
            if len(self.known_y)==0:
                return random.uniform(-50,50)
            else:
                return random.uniform(*legal_y)
        
        elif function_type=='manual':
            while True:
                user_input=input("Enter a y value (between {} and {}), as a decimal, for x={}".format(legal_y[0],legal_y[1],x))
                try:
                    user_input=float(user_input)
                    if user_input >= legal_y[0] and user_input <= legal_y[1]:
                        return user_input
                    else:
                        print("Your number must be between {} and {}".format(legal_y[0],legal_y[1]))
                except ValueError:
                    print("Please enter as a decimal number.")

        # Raise an error if the function type is not supported.
        else:
            raise ValueError("The function type must be one of the strings 'manual', 'optimal', 'random', or 'sample")


    def update_arrays(self, x, index, y):
        """Takes an x and y value to insert in the known value arrays as well as an index in which to insert them, then updates
        all the arrays."""

        # If the new x is to be inserted to the left of all currently known x
        if index==0:
            # If there are no currently known points the interval highs occur at the ends of the interval.
            if len(self.known_x)==0:
                self._interval_x=np.array(self.interval,dtype=float)
                self._interval_y=np.array([y-self.lipschitz_constraint*(self.interval[0]-x),
                                           y+self.lipschitz_constraint*(self.interval[1]-x)],dtype=float)
        
            # If there are known points and the new x is inserted to the left of them all.
            else:
                # Insert a new value into the _interval_x array to become the new index 1 (leaving the left endpoint at index 0 unchanged)
                self._interval_x=np.insert(self._interval_x,1,(self.known_y[0]-y+self.lipschitz_constraint*(self.known_x[0]+x))/(2*self.lipschitz_constraint))
                # Inserting the corresponding y value
                self._interval_y=np.insert(self._interval_y,1,(self.known_y[0]+y+self.lipschitz_constraint*(self.known_x[0]-x))/2)
                # Adjusting the y value of the left endpoint with respect to the new point included.
                self._interval_y[0]=y-self.lipschitz_constraint*(self.interval[0]-x)
        # If the new x is to be inserted to the right of all known x
        elif index==len(self.known_x):
            self._interval_x=np.insert(self._interval_x,-1,(y-self.known_y[-1]+self.lipschitz_constraint*(self.known_x[-1]+x))/(2*self.lipschitz_constraint))
            self._interval_y=np.insert(self._interval_y,-1,(self.known_y[-1]+y+self.lipschitz_constraint*(x-self.known_x[-1]))/2)
            self._interval_y[-1]=y+self.lipschitz_constraint*(self.interval[1]-x)
        # If the new x is to be inserted between two known points.
        else:
            self._interval_x=np.insert(self._interval_x,index,(y-self.known_y[index-1]+self.lipschitz_constraint*(self.known_x[index-1]+x))/(2*self.lipschitz_constraint))
            self._interval_x[index+1]=(self.known_y[index]-y+self.lipschitz_constraint*(self.known_x[index]+x))/(2*self.lipschitz_constraint)
            self._interval_y=np.insert(self._interval_y,index,(self.known_y[index-1]+y+self.lipschitz_constraint*(x-self.known_x[index-1]))/2)
            self._interval_y[index+1]=(self.known_y[index]+y+self.lipschitz_constraint*(self.known_x[index]-x))/2

        # Adding the given x and y to the arrays of known values.
        self.known_x=np.insert(self.known_x,index,x)
        self.known_y=np.insert(self.known_y,index,y)
        # Updating the instance variables.
        self.latest_x=x
        self.latest_y=y
        self.max_y=max(self.known_y)
        self.max_possible_y=max(self._interval_y)
        self.max_possible_y=max(self.max_possible_y,self.max_y)

    def add_n_points(self, n: int, function_type, adaptive=True, optimal_x=True):
        """Work through the process of adding n more points."""
        
        if n<1:
            raise ValueError("You must add at least one point.")
        # If non-adaptive
        if adaptive==False and optimal_x==False:
            self._non_adaptive(n,function_type)
        # If adaptive (or manually choosing x).
        else:
            for turn in range(n):
                self._add_one_point(function_type,optimal_x)

                # Checking if the process should be ended early.
                if self.max_y==self.max_possible_y:
                    break

    def _add_one_point(self, function_type, optimal_x=True):
        """Adds a single point to the current configuration."""

        # Determining the next x value to add.
        if optimal_x==True:
            x_to_insert=self.get_optimal_x()
        else:
            x_to_insert=self._get_manual_x()

        # Determining what the paired y value should be.
        y_to_insert=self.get_y(*x_to_insert, function_type)

        # Updating the arrays.
        self.update_arrays(*x_to_insert,y_to_insert)

        # Adding a graphable copy of the current arrays to the stored list.
        self.add_graph()

        # Obtaining the next row of the dataframe.
        (index,new_row)=self._get_results()
        self.results_df.loc[index]=new_row

    def _fit_n_on_interval(self, n:int, interval_index: int):
        """Determines what the optimal placement of n new x values should be on the given interval and returns a pair
        (xs, max_y) with a list of these x values and what the maximum value is between them."""

        # If we need to fit points between the left boundary and the first known x.
        if interval_index==0:
            right_end=self.known_x[0]-(self.max_y-self.known_y[0])/self.lipschitz_constraint
            spacing=(right_end-self.interval[0])/(n+.5)
            starting_x=self.interval[0]+spacing/2
        # If we need to fit points between the left boundary and the last known x.
        elif interval_index==len(self._interval_y)-1:
            left_end=self.known_x[-1]+(self.max_y-self.known_y[-1])/self.lipschitz_constraint
            spacing=(self.interval[1]-left_end)/(n+.5)
            starting_x=left_end+spacing
        # If we need to fit points between two known points.    
        else:
            left_end=self.known_x[interval_index-1]+(self.max_y-self.known_y[interval_index-1])/self.lipschitz_constraint
            right_end=self.known_x[interval_index]-(self.max_y-self.known_y[interval_index])/self.lipschitz_constraint
            spacing=(right_end-left_end)/(n+1)
            starting_x=left_end+spacing

        max_y=(2*self.max_y+self.lipschitz_constraint*spacing)/2
        xs=[starting_x+i*spacing for i in range(n)]
        return(xs,max_y)

        


    def _non_adaptive(self, n: int, function_type):
        """Adds n points where the x values are chosen optimally simultaneously (rather than one by by adaptively) based
        on the current configuration of the approximator."""

        # Creating a list of new x values (and their indices) to add.
        xs_to_add=[]

        # If there is only one point to be added we do it simply.
        if n==1:
            self._add_one_point(function_type)
            return
        # If no points are known the method is also easier.
        elif len(self.known_x)==0:
            for i in range(1,n+1):
                xs_to_add.append(((2*(self.interval[0]*n+self.interval[1]*i-self.interval[0]*i)+self.interval[0]-self.interval[1])/(2*n),i-1))
        # Otherwise we add new x values to intervals to decrease the radius of information assuming that nature plays optimally.
        else:
            # We create a dataframe to search through and update as points are added.
            # The column Interval_max records the greatest y value on the interval, xs added the number of xs to be fit inside,
            # and xs a list of xs to be added on that interval.
            temp_df=pd.DataFrame({'Interval_max':self._interval_y}, index='Interval')
            temp_df['xs added']=0
            temp_df['xs']=None
            # We now do the following process n times.
            for i in range(n):
                # Find the interval where the max possible y could be.
                max_index=temp_df['Interval_max'].idxmax()
                # See how many xs are supposed to be added to that interval and try to add one more.
                number_to_add=temp_df.at[max_index,'xs added']+1
                # See how to select those xs and what the new maximum y would be.
                (xs,int_max)=self._fit_n_on_interval(number_to_add,max_index)
                # Replace the entry in the dataframe with the new information.
                temp_df.loc[max_index]=[int_max,number_to_add,xs]
            # After exiting the for loop we now add all the necessary xs to our list.
            points_added=0
            for i in temp_df.index:
                if temp_df.at[i,'xs added']>0:
                    for x in temp_df.at[i,'xs']:
                        # We add pairs of the form (x_to_insert,index_to_insert) to our list.
                        xs_to_add.append((x,i+points_added))
                        points_added+=1


        for x_to_insert in xs_to_add:
            # Determining what the paired y value should be.
            y_to_insert=self.get_y(*x_to_insert, function_type)

            # Updating the arrays.
            self.update_arrays(*x_to_insert,y_to_insert)
        
        # We have now added n points to the approximator.
        # Since points were supposed to be added all at once we set the latest x and y to None.
        self.latest_x=None
        self.latest_y=None
        # Adding a graphable copy of the current arrays to the stored list.
        self.add_graph()

        # Obtaining the next row of the dataframe.
        (index,new_row)=self._get_results()
        # Adding the new row.
        self.results_df.loc[index]=new_row

    def display_graphs(self, n=None, first=False, display_region=False):
        """Displays a collection of n graphs. By default this method produces all the currently available graphs. If last is set to False
        (by default) then the first n graphs are given, and if set to True then the most recent n graphs are displayed."""


        # Determining the number of graphs to plot.  Up to 8 can be plotted in a single figure.
        total_graphs=len(self.graphs)
        
        if n==None:
            n=total_graphs
        elif type(n)!=int:
            raise TypeError("n must be an integer.")
        elif n<1:
            raise ValueError('n must be greater than 0.')
        elif n>total_graphs:
            raise Warning('There are only {} graphs so only {} will be displayed.'.format(total_graphs,total_graphs))

        # Determining the indices of the first and last graph to display.
        if first==True:
            first_index=0
            last_index=n-1
        else:
            first_index=total_graphs-n
            last_index=total_graphs-1
        
        # Obtaining the x and y limits of each graph.
        y_min=min(self.graphs[last_index].known_y)
        y_max=self.graphs[first_index].max_possible_y
        x_margins=plt.margins()[0]*(self.interval[1]-self.interval[0])
        y_margins=plt.margins()[1]*(y_max-y_min)
        x_limits=(self.interval[0]-x_margins,self.interval[1]+x_margins)
        y_limits=(y_min-y_margins,y_max+y_margins)
        plt.close()
        

        # Determining how many full figures will be plotted.
        full_graphs=n//8
        # Determining how many graphs should appear in the partial figure (if any)
        n=n%8

        # For each full window/figure a figure and axes are created and then displayed.
        for window in range(full_graphs):
            # Each full window should include two rows of graphs with four graphs in each row.
            fig, axs = plt.subplots(2,4,figsize=(12,6))
            # Flatten the axes so they can be iterated through numerically more easily.
            axs=axs.flatten()

            # Drawing each of the graphs. 
            for i in range(len(axs)):
                self.graphs[i+8*window+first_index].draw_to_axis(axs[i], display_region)

            # Setting the x and y limits of all the graphs on the current figure.
            plt.setp(axs, ylim=y_limits,xlim=x_limits)
            plt.tight_layout()
            # Displaying without blocking so the user can flip between multiple windows and compare.
            plt.show(block=False)

        # The layout of the last figure is determined by how many graphs remain.
        if n<=1:
            # If there is only one graph to display it will be displayed at the end and we can ignore it here.
            pass
        elif n<3:
            fig, axs = plt.subplots(1,n,figsize=(12,6))
        elif n<5:
            fig, axs = plt.subplots(2,2,figsize=(12,6))
        elif n<7:
            fig, axs = plt.subplots(2,3,figsize=(12,6))
        elif n<9:
            fig, axs = plt.subplots(2,4,figsize=(12,6))
        else:
            raise ValueError('The number of graphs should not exceed 8 on any one figure.')

        # If there is only one graph we only display the final result.
        if n<=1:
            pass
        else:
            # If there are multiple graphs we can flatten the axes.
            axs=axs.flatten()
            # Each graph is drawn.
            for i in range(n):
                self.graphs[i+8*full_graphs+first_index].draw_to_axis(axs[i],display_region)

            # The excess axes are removed.
            for i in range(n,len(axs)):
                fig.delaxes(axs[i])    

            plt.setp(axs, ylim=y_limits,xlim=x_limits)
            plt.tight_layout()
            plt.show(block=False)

        # A final figure is included showing, in greater size, the final configuration achieved.
        points_known=len(self.graphs[last_index].known_x)
        if points_known==1:
            plt.figure(figsize=(10,5), num="1 Point Known")
        else:
            plt.figure(figsize=(10,5), num="{} Points Known".format(points_known))
        ax=plt.gca()
        self.graphs[last_index].draw_to_axis(ax,display_region)
        plt.show()

    def get_known_pairs(self):
        """Returns a list of known x and y values in the form of pairs (x,y)."""
        
        zipped=zip(self.known_x,self.known_y)
        return list(zipped)
    
    def revert_to_state(self, n):
        """Reverts the approximator to the state when n points were known."""

        # Create a list of the graphs to preserve.
        graphs_to_keep=[]
        for graph in self.graphs:
            # Add each graph to the list as long as the number of known points in the graph was less than n.
            if len(graph.known_x)<=n:
                graphs_to_keep.append(graph)
            # End the for loop the first time a graph has more than n points.
            else:
                break
        # We will check to see if the final graph added to the above list had the correct number of points.
        last_graph=graphs_to_keep[-1]
        # If it does not have the correct number we raise an error.
        if len(last_graph.known_x)!=n:
            raise ValueError("The current approximator has no record of having had a state with exactly {} points known.".format(n))
        # Otherwise we replace the current state of the approximator to match that of the last_graph.
        else:
            # We match the instance variables of the approximator to those of the last_graph.
            for key in last_graph.__dict__.keys():
                self.__dict__[key]=last_graph.__dict__[key]
            # Updating the list of graphs.
            self.graphs=graphs_to_keep
            # Cutting the results_df down to only include entries when there were at most n points known.
            self.results_df=self.results_df[self.results_df.index<=n]




         
            
class Graph:
    """
    Maintains four arrays. One for the known x values, one for the known y values, 
    one for the x values where the function could reach a maximum on each interval, and one for the corresponding maximum y values."""

    def __init__(self, xs: np.array, ys:np.array, interval_x: np.array, interval_y: np.array,max_y,max_possible_y,latest_x=None,latest_y=None):
        self.known_x=xs
        self.known_y=ys
        self.interval_x=interval_x
        self.interval_y=interval_y
        self.max_y=max_y
        self.max_possible_y=max_possible_y
        self.latest_x=latest_x
        self.latest_y=latest_y

    def draw_to_axis(self, axis, display_region=False):
        """Draws the graph onto the target axis. The figure must still be shown to display."""
        axis.axvline(self.interval_x[0],color='red',linestyle='dashed')
        axis.axvline(self.interval_x[-1],color='red',linestyle='dashed')
        axis.scatter(self.known_x,self.known_y, alpha=.95)
        if self.latest_x!=None:
            axis.scatter(self.latest_x,self.latest_y, zorder=3,color='tab:blue', marker='D')
        # We now combine the interval and known points to create a line graph.
        zipped=zip(np.concatenate([self.interval_x,self.known_x]),np.concatenate([self.interval_y,self.known_y]))
        zipped=sorted(list(zipped))
        x,y= zip(*zipped)
        axis.plot(x,y, linestyle='dashed', alpha=.8)
        points_known=len(self.known_x)
        if display_region==True:
            axis.axhspan(self.max_y,self.max_possible_y, alpha=.2, color='darkgrey')

        if points_known==1:
            axis.set_title("{} Point Known".format(points_known))
        else:
            axis.set_title("{} Points Known".format(points_known))
        

def radius_of_information(greatest_max=None,least_max=None):
    """Returns the radius of information given the greatest possible maximum and the smallest possible maximum on the interval."""
        
    return (greatest_max-least_max)/2



















if __name__ == '__main__':
    """The class ApproximateMaxima provides a means to approximate the maximum value of an unknown function f, 
    defined on an interval [a,b], that satisfies a Lipschitz constraint M. i.e. if f is differentiable on [a,b] then 
    f'(x)< M for all x in the interval [a,b]."""
    # Enter the interval and Lipschitz constant here.  All calculations will include the endpoints of the interval.
    interval=(0,1)
    lipschitz_constant=1

    # Set the number of x values that will be chosen.
    number_of_x=3

    # Set this to True for an adaptive strategy (x values are chosen one at a time) and false for 
    # a non-adaptive strategy (all x values are chosen at once.)
    adaptive=True

    # This variable determines how the computer chooses y values for the selected x values. The options are:
    # 'random' - y values are randomly chosen which satisfy the lipschitz condition
    # 'sample' - y values are decided from a pre-decided function (which can be edited in the sample_function above).
    # 'optimal' - y values are always those which maximize the radius of information (and weaken our prediction of maxima and minima) 
    function_type='sample'

    # Determine whether to shade the vertical area where the maximum value of the function could occur on each graph.
    display_region=False

    # List any known points (this should be a list of pairs of the form (x,y))
    known=[(1,3),(3,5)]

    approximation=ApproximateMaxima(interval,lipschitz_constant,sample_function='x*x*x-x*x')
    #approximate_maximum(interval, lipschitz_constant, number_of_x, function_type, adaptive,display_region=True)
    approximation.add_n_points(5, 'sample')
    print(approximation.known_x)

    approximation.display_graphs(3)





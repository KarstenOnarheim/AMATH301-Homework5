import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

##################### Coding Problem 1 ##########################
## Part a - Load in the data
# The data is called 'CO2_data.csv'

M = np.genfromtxt('CO2_data.csv', delimiter=',')
t = M[:, 0]
CO2 = M[:, 1]

A1 = t
A2 = CO2
## (b)
# Define the error function that calculates the sum of squared error.
# I've started this for you, but you need to fill it in and uncomment. 

def sumSquaredError(a, b, r):
    # Define the model y
    y = lambda t: a + b*(np.e**(r*t))

    # Compute the error using sum-of-squared error
    error = (sum((y(t) - CO2)**2) / t.size)**.5
    return error

# Check the error function by defining A3
A3 = sumSquaredError(300, 30, .3)

## (c)
# We need an adapter function to make this work with scipy.optimize.fmin
# Uncomment the line below to use the adapter function
# adapter = lambda p: sumSquaredError(p[0], p[1], p[2])

# Once adapter is defined, use fmin
# We use the following guess
guess = np.array([300, 30, 0.03])

## (d)
# Once we have found the optimal parameters, 
# find the error for those optimal parameters

## (e)
# Now we do the same thing except with max error. 
# Your function looks similar, except use the max error

## (f)
# This error function has more inputs, but it's the same idea.
# Make sure to use sum of squared error!

# And we need to make a new adapter function
# Again, this will have more inputs but will look pretty similar. 

## (h)
# Once we have found the optimal parameters, find the associated error.


######################### Coding problem 2 ###################
## Part (a)
M = np.genfromtxt('salmon_data.csv', delimiter=',')

year = M[:,0] #Assign the 'year' array to the first column of the data
salmon = M[:,1] #Assign the 'salmon' array to the first column of the data

## (b) - Degree-1 polynomial

## (c) - Degree-3 polynomial

## (d) - Degree-5 polynomial

## (e) - compare to exact number of salmon
exact =  752638 # The exact number of salmon


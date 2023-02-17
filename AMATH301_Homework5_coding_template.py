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
    #error = (sum((y(t) - CO2)**2) / t.size)**.5
    error = sum((y(t) - CO2)**2)
    return error

# Check the error function by defining A3
A3 = sumSquaredError(300, 30, .03)

## (c)
# We need an adapter function to make this work with scipy.optimize.fmin
adapter = lambda p: sumSquaredError(p[0], p[1], p[2])
guess = np.array([300, 30, 0.03])

A4 = scipy.optimize.fmin(adapter, guess, maxiter=2000)
## (d)
A5 = adapter(A4)

## (e)
# Now we do the same thing except with max error. 
def maxError(a, b, r):
    # Define the model y
    y = lambda t: a + b*(np.e**(r*t))

    # Compute the error using sum-of-squared error
    error = np.amax(np.abs(y(t) - CO2))
    return error
adapterME = lambda p: maxError(p[0], p[1], p[2])

A6 = maxError(300, 30, .03)
A7 = scipy.optimize.fmin(adapterME, guess, maxiter=2000)
## (f)
# This error function has more inputs, but it's the same idea.
# Make sure to use sum of squared error!

def sumSquaredError(a, b, r, c, d, e):
    # Define the model y
    y = lambda t: a + b*(np.e**(r*t)) + c*np.sin(d*(t - e))

    # Compute the error using sum-of-squared error
    #error = (sum((y(t) - CO2)**2) / t.size)**.5
    error = sum((y(t) - CO2)**2)
    return error

adapterSSQ = lambda p: sumSquaredError(p[0], p[1], p[2], p[3], p[4], p[5])
A8 = adapterSSQ([300, 30, .03, -5, 4, 0])

guess1 = np.array([A4[0], A4[1], A4[2], -5, 4, 0])

A9 = scipy.optimize.fmin(adapterSSQ, guess1, maxiter=2000)
A10 = adapterSSQ(A9)


######################### Coding problem 2 ###################
## Part (a)
M = np.genfromtxt('salmon_data.csv', delimiter=',')

year = M[:,0] #Assign the 'year' array to the first column of the data
salmon = M[:,1] #Assign the 'salmon' array to the first column of the data

## (b) - Degree-1 polynomial
A11 = np.polyfit(year, salmon, 1)

## (c) - Degree-3 polynomial
A12 = np.polyfit(year, salmon, 3)

## (d) - Degree-5 polynomial
A13 = np.polyfit(year, salmon, 5)

## (e) - compare to exact number of salmon
exact =  752638 # The exact number of salmon

er1 = np.abs(np.polyval(A11, 2022) - exact) / exact
er2 = np.abs(np.polyval(A12, 2022) - exact) / exact
er3 = np.abs(np.polyval(A13, 2022) - exact) / exact

A14 = np.array([er1, er2, er3])


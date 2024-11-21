# %%
#I AM INCLUDING CODE FROM OUTSIDE THE QUESTIONS THEMSELVES SO THAT THE CELLS ARE EXECUTABLE AND CAN BE EASILY CHECKED
#curse of dimensionality
#generating samples of a function and creating an  interpolator which will approximate it as accurately as possible
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# %%
#Q2
#rerunning on a finer grid
#same parameters as before
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

a = 0.1
b = -0.13
c = 9
#keeping same time range, finer grid
t_fine = np.linspace(0, 1, 300)  

#generating analyzed values
#saving original data points and interpolated points to match shapes when plotting
f_values = f(t_fine, a, b, c)
interpolator = interp1d(t_fine, f_values, kind='cubic')
linear_interpolator = interp1d(t_fine, f_values, kind='linear')
f_linear_interpolated = linear_interpolator(t_fine)
f_interpolated = interpolator(t_fine)
y_original = f(0.5, a, b, c)
print(f"Original value at x=0.5: {y_original}")
y_interp = interpolator(0.5)
print(f"Interpolated value at x=0.5 using cubic interpolation: {y_interp}")
y_interp_linear = linear_interpolator(0.5)
print(f"Interpolated value at x=0.5 using linear interpolation: {y_interp_linear}")
#greater amount of data provides us with a closer match for both types of interpolation, as would be expected
#%%
#Q3
#plot
plt.figure(figsize=(15, 9))
plt.plot(t_fine, f_values, 'o', label='Original Points', markersize=1)
plt.plot(t_fine, f_interpolated, '-', label='Cubic Interpolation', linewidth=1)
plt.plot(t_fine, f_linear_interpolated, '--', label='Linear Interpolation', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic and Linear Interpolation with interp1d with a finer grid')
plt.legend()
plt.grid(True)
plt.show()
#finer grid creates even more accurate interpolation for our data
#interpolator almost exactly follows the plot of the original function


# %%
#Q6

#generating function within which a and t are no longer fixed
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Example usage
# Define parameters
b = -0.13
c = 9
#time variable
t = np.linspace(0, 1, 100)
t_fine = np.linspace(0, 1, 300)

# Calculate f(t) for these parameters
#a_values hol
a_values = np.linspace(0, 1, 10) #holder for the 10 different a values
plt.figure(figsize=(20, 12))
results = []
for a in a_values:
    f_values = f(t, a, b, c) # generates f values for each value of coefficient a
    interpolator = interp1d(t, f_values, kind='cubic')
    f_interpolated = interpolator(t)
    for tx, fx in zip (t, f_values): #zip pairs t and f values effectively generating pairs for generating the individual functions
        results.append({'a': a, 't': tx, 'f(t)': fx})
        plt.plot(t, f_values, '-', label=f'a={a:.2f}', color='blue') # since here first all values for a=0 plotted then a=0.1 etc.
df = pd.DataFrame(results)
print(df)
#progressively more curvy as a increases, this is to be expected as without a much of the original function is zeroed out
# %%
prediction_ratio = (f_values - f_interpolated)/(f_values)
plt.plot(t, prediction_ratio)
#some minimal deviations with the cubic interpolator still at the same scale as in previous questions 
# %%
#plotting for a particular value
#doing this over the finer grid
a_value = 0.125
f_values = f(t_fine, a_value, b, c)
f_value_specific = f(t_fine, a_value, b, c)


interpolator_specific = interp1d(t_fine, f_value_specific, kind='cubic')
f_value_fine = interpolator_specific(t_fine)

plt.figure(figsize=(10, 6))
plt.plot(t_fine, f_values, 'o', label=f'Original Function (a={a_value})')
plt.plot(t_fine, f_value_fine, '-', label=f'Cubic Interpolation (a={a_value})')
plt.xlabel('Time t')
plt.ylabel('f(t)')
plt.title('Interpolation Over Parameter a')
plt.legend()
plt.grid(True)
plt.show()
# %%
#Q11
#i understood the question in a way where we are supposed to print all datapoints with time varying
#lack of dimensions lead to time being on a sliding scale
from scipy.interpolate import griddata
from ipywidgets import interact
from pyDOE import lhs
#points stand for sampled values of a and b
#values stand for resulting values of our f function

#defining parameters
#uniform, stepwise distribution here
t = np.linspace(0, 1, 100)
a_range = [0,1]
b_range = [-0.5,0.5]
c = 9

factors = 2  
num_samples = 100
lhs_samples = lhs(factors, samples=num_samples)

a_samples = a_range[0] + (a_range[1] - a_range[0]) * lhs_samples[:, 0]
b_samples = b_range[0] + (b_range[1] - b_range[0]) * lhs_samples[:, 1]
a_samples = a_range[0] + (a_range[1] - a_range[0]) * lhs_samples[:, 0]
b_samples = b_range[0] + (b_range[1] - b_range[0]) * lhs_samples[:, 1]

A, B = np.meshgrid(np.linspace(0, 1, 10), np.linspace(-0.5, 0.5, 10))
f_values = f(t, a_samples, a_samples, c)
interpolator = interp1d(t, f_values, kind='cubic')
f_interpolated = interpolator(t)

points = np.array(list(zip(a_samples, b_samples)))
values = np.array([f(0.5, a, b, c) for a, b in points])

f_interpolated = griddata(points, values, (A, B), method='cubic')

# Plot the interpolated parameter space
def plot(t):
    f_values = f(t, A, B, c)
    plt.figure(figsize=(15, 9))
    plt.contourf(A, B, f_values, cmap='viridis')
    plt.colorbar(label='f(a, b) at given t')
    plt.xlabel('Parameter a')
    plt.ylabel('Parameter b')
    plt.title('Interpolation for three free variables t,a,b')
    plt.grid(True)
    plt.show()
interact(plot, t=(0, 1, 0.05))
# %%
#Q13
#i got bogged down so Q13 not solved
#i would assume interpolator call given the high dimensions of our request would take way longer than the original function
#will redo after class
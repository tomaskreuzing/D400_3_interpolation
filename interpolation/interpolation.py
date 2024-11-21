# %%
#curse of dimensionality
#generating samples of a function and creating an  interpolator which will approximate it as accurately as possible
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


# %%
#generating the original function and plotting it with set coefficients
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Example usage
# Define parameters
a = 0.1
b = -0.13
c = 9


# Define time range
t = np.linspace(0, 1, 100)

# Calculate f(t) for these parameters
f_values = f(t, a, b, c)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(t, f_values, label=r'$f(t; a, b, c)$', color='blue')
plt.xlabel('Time $t$')
plt.ylabel(r'$f(t)$')
plt.title(r'Time Series $f(t; a, b, c)$')
plt.legend()
plt.grid(True)
plt.show()

# %%
#plotting the original function in a widget and playing with the coefficients
from ipywidgets import interactive
import ipywidgets as widgets


def plot_with_parameters(a, b, c):
    # Define time range
    t = np.linspace(0, 1, 100)

    # Calculate f(t) for these parameters
    f_values = f(t, a, b, c)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(t, f_values, label=r'$f(t; a, b, c)$', color='blue')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$f(t)$')
    plt.title(r'Time Series $f(t; a, b, c)$')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widget
interactive_plot = interactive(
    plot_with_parameters,
    a=widgets.FloatSlider(min=0, max=1, step=0.01, value=0.1),
    b=widgets.FloatSlider(min=-0.5, max=0.5, step=0.01, value=-0.13),
    c=widgets.FloatSlider(min=5, max=10, step=0.1, value=9)
)

display(interactive_plot)

# %%
#Q1
#creating an interp1d interpolator for our generated data
# trial interpolator
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Define parameters
a = 0.1
b = -0.13
c = 9

#keeping same time range
t = np.linspace(0, 1, 100)  

#generating analyzed values
#saving original data points and interpolated points to match shapes when plotting
f_values = f(t, a, b, c)
interpolator = interp1d(t, f_values, kind='cubic')
linear_interpolator = interp1d(t, f_values, kind='linear')
f_linear_interpolated = linear_interpolator(t)
f_interpolated = interpolator(t)
y_original = f(0.5, a, b, c)
print(f"Original value at x=0.5: {y_original}")
y_interp = interpolator(0.5)
print(f"Interpolated value at x=0.5 using cubic interpolation: {y_interp}")
y_interp_linear = linear_interpolator(0.5)
print(f"Interpolated value at x=0.5 using linear interpolation: {y_interp_linear}")
#there is a small difference between the two interpolated values, at the order of the thousandth digit
#there is a small difference between the original and interpolated values but really negligible for our purposes
#however we do know that the interpolation does differ as expected and no mistake has been made in the provided functions
#%%
#plot
plt.figure(figsize=(10, 6))
plt.plot(t, f_values, 'o', label='Original Points', markersize=1)
plt.plot(t, f_interpolated, '-', label='Cubic Interpolation', linewidth=1)
plt.plot(t, f_linear_interpolated, '--', label='Linear Interpolation', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic and Linear Interpolation with interp1d')
plt.legend()
plt.grid(True)
plt.show()
#result very closely approximates our datapoints already

# %%
#Q2
#rerunning on a finer grid
#same parameters as before

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
#Q4
prediction_ratio = (f_values - f_interpolated)/(f_values)
plt.plot(t_fine, prediction_ratio)
#deviations of the cubic interpolator but at extremely low values
prediction_ratio_linear = (f_values - f_linear_interpolated)/(f_values)
plt.plot(t_fine, prediction_ratio_linear)
#no deviations for the linear approximation, exactly equal to true values apparently
# %%
#Q5
#generating function within which a and t are no longer fixed
# Define parameters
b = -0.13
c = 9
#time variable
t = np.linspace(0, 1, 100)

# Calculate f(t) for these parameters
a_values = np.linspace(0, 1, 10) #holder for the 10 different a values
plt.figure(figsize=(20, 12))
results = []
for a in a_values:
    f_values = f(t, a, b, c) # generates f values for each value of coefficient a
    for tx, fx in zip (t, f_values): #zip pairs t and f values effectively generating pairs for generating the individual functions
        results.append({'a': a, 't': tx, 'f(t)': fx})
        plt.plot(t, f_values, '-', label=f'a={a:.2f}', linewidth = 0.5, color='blue') # since here first all values for a=0 plotted then a=0.1 etc.
        
#values follow the expected distribution of our plot as tested through ipywidgets
#when a=0 data become much less oscillatory
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
f_values = f(t, a_value, b, c)
f_value_specific = f(t, a_value, b, c)


interpolator_specific = interp1d(t, f_value_specific, kind='cubic')
f_value_specific = interpolator_specific(t)

plt.figure(figsize=(10, 6))
plt.plot(t, f_values, 'o', label=f'Original Function (a={a_value})')
plt.plot(t, f_value_specific, '-', label=f'Cubic Interpolation (a={a_value})')
plt.xlabel('Time t')
plt.ylabel('f(t)')
plt.title('Interpolation Over Parameter a')
plt.legend()
plt.grid(True)
plt.show()
# %%
prediction_ratio = (f_values - f_value_specific)/(f_values)
plt.plot(t, prediction_ratio)
#there is a substantial deviation right before reaching one, possibly because of sparsity of data in that section
#%%
#Q7
#plotting 0.1 stepwise graph for interpolated vs original values
# %%
def plot_with_parameters(a, b, c):
    # Define time range
    t = np.linspace(0, 1, 100)

    # Calculate f(t) for these parameters
    f_values = f(t, a, b, c)
    interpolator = interp1d(t, f_values, kind='cubic')
    f_interpolated = interpolator(t)
    prediction_ratio = (f_values - f_interpolated)/(f_values)
    # Plot the function
    plt.figure(figsize=(10, 6))
    #plt.plot(t, f_values, label=r'$f(t; a, b, c)$', color='blue')
    plt.plot(t, prediction_ratio, label=r'$f(t; a, b, c)$', color='blue')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$f(t)$')
    plt.title(r'Time Series $f(t; a, b, c)$')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widget
interactive_plot = interactive(
    plot_with_parameters,
    a=widgets.FloatSlider(min=0, max=1, step=0.1, value=0.1),
    b=widgets.FloatSlider(min=-0.5, max=0.5, step=0.1, value=-0.13),
    c=widgets.FloatSlider(min=5, max=10, step=0.1, value=9)
)
#i need interpolated values for different values of a (lets use 0.1 steps)
#i need them to match with values in the floatslider
display(interactive_plot)
# %%
#Q8
#errors change marginally with occasional outliers in places where concavity changes and arguably with lesser datapoints
#another way to understand this question would be to compare the values for the sliding scale function and the values with set a from Q6
#in that case extreme outliers present, but this did not make too much sense to me to plot
# %%
#Q9 and Q10
#find out more about latin hypercube sampling
from pyDOE import lhs
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

a_uniform = np.random.uniform(0,1, size=100)
b_uniform = np.random.uniform(-0.5,0.5, size=100)


#not necessary

plt.figure(figsize=(10, 6))
plt.scatter(a_samples, b_samples, c='b', label = 'Latin hypercube sampling')
plt.scatter(a_uniform, b_uniform, c='r', label = 'Uniform sampling')
plt.xlabel('Parameter a')
plt.ylabel('Parameter b')
plt.title('Latin hypercube x uniform sampling comparison')
plt.grid(True)
plt.legend()
plt.show()
#nicely distributed sample of a and b values according to their expected ranges
#latin hypercube arguably less clustered
# %%
#Q11
from scipy.interpolate import griddata
from ipywidgets import interact
#points stand for sampled vlaue sof a and b
#values stand for resulting values of our f function
A, B = np.meshgrid(np.linspace(0, 1, 10), np.linspace(-0.5, 0.5, 10))
f_values = f(t, a_samples, a_samples, c)
interpolator = interp1d(t, f_values, kind='cubic')
f_interpolated = interpolator(t)

points = np.array(list(zip(a_samples, b_samples)))
values = np.array([f(0.5, a, b, c) for a, b in points])  # Evaluate function at t=0.5

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
interact(plot, t=(0, 1, 0.1))


# %%
#Q12

# %%
#duplicate in case i fuck up
#Q11
A, B = np.meshgrid(np.linspace(0, 1, 10), np.linspace(-0.5, 0.5, 10))
points = np.array(list(zip(a_samples, b_samples)))
values = np.array([f(0.5, a, b, c) for a, b in points])  # Evaluate function at t=0.5

f_interpolated = griddata(points, values, (A, B), method='cubic')

# Plot the interpolated parameter space
plt.figure(figsize=(15, 9))
plt.contourf(A, B, f_interpolated, cmap='viridis')
plt.colorbar(label='f(a, b) at t=0.5')
plt.xlabel('Parameter a')
plt.ylabel('Parameter b')
plt.title('Interpolation for three free variables t,a,b')
plt.grid(True)
plt.show()

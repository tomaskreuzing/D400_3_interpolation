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
f_interpolated = interpolator(t)

#plot
plt.figure(figsize=(10, 6))
plt.plot(t, f_values, 'o', label='Original Points', markersize=1)
plt.plot(t, f_interpolated, '-', label='Cubic Interpolation', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Interpolation with interp1d')
plt.legend()
plt.grid(True)
plt.show()
#result very closely approximates our datapoints already

# %%
#Q2
#rerunning on a finer grid
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Define parameters
a = 0.1
b = -0.13
c = 9

#keeping same time range, finer grid
t = np.linspace(0, 1, 300)  

#generating analyzed values
#saving original data points and interpolated points to match shapes when plotting
f_values = f(t, a, b, c)
interpolator = interp1d(t, f_values, kind='cubic')
f_interpolated = interpolator(t)

#Q3
#plot
plt.figure(figsize=(20, 12))
plt.plot(t, f_values, '--', label='Original Points', markersize=1)
plt.plot(t, f_interpolated, '-', label='Cubic Interpolation', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Interpolation with interp1d')
plt.legend()
plt.grid(True)
plt.show()
#finer grid creates even more accurate interpolation for our data
#interpolator almost exactly follows the plot of the original function

# %%
#Q4
prediction_ratio = f_values - f_interpolated
print(prediction_ratio.var())
#there is almost no difference between the interpolated and original values
#we do not have zeros everywhere due to python formatting, but the valus we get might as well be zero (10*-16)
# %%
#Q5
#generating function within which a and t are no longer fixed
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Example usage
# Define parameters
b = -0.13
c = 9
#time variable
t = np.linspace(0, 1, 100)

# Calculate f(t) for these parameters
#a_values hol
a_values = np.linspace(0, 1, 10) #holder for the 10 different a values
plt.figure(figsize=(20, 12))
results = []
for a in a_values:
    f_values = f(t, a, b, c) # generates f values for each value of coefficient a
    for tx, fx in zip (t, f_values): #zip pairs t and f values effectively generating pairs for generating the individual functions
        results.append({'a': a, 't': tx, 'f(t)': fx})
        plt.plot(t, f_values, 'o', label=f'a={a:.2f}', linewidth = 1, color='blue') # since here first all values for a=0 plotted then a=0.1 etc.
        
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

#%%
#continue here when needed
# + plot the result for a=0.125, like in question 3.
# Define parameters
a = 0.125
b = -0.13
c = 9
#time variable
t = np.linspace(0, 1, 100)
f_values = f(t, a, b, c) 
interpolator = interp1d(t, f_values, kind='cubic')
f_interpolated = interpolator(t)
plt.plot(t, f_values, 'o', label=f'a={a:.2f}', color='blue')
plt.plot(t, f_interpolated, '--', label=f'a={a:.2f}', color='red')
#still seems very accurate
# + plot ratio like in question 4
prediction_ratio = f_values - f_interpolated
print(prediction_ratio)
#still basically zero, how?
#plt.bar(t, prediction_ratio)

# %%
df.plot(x="t", y="f(t)", kind='scatter', figsize=(20, 12))
# %%
interpolator = interp1d(t, df, kind='cubic')
f_interpolated = interpolator(t)
plt.plot(t, f_interpolated, '--', linewidth = 1, color='red')


# %%
#Q7
a_values = np.linspace(0, 1, 10) #holder for the 10 different a values
plt.figure(figsize=(20, 12))
results = []
for a in a_values:
    f_values = f(t, a, b, c) # generates f values for each value of coefficient a
    for tx, fx in zip (t, f_values): #zip pairs t and f values effectively generating pairs for generating the individual functions
        results.append({'a': a, 't': tx, 'f(t)': fx})
        plt.plot(t, f_values, 'o', label=f'a={a:.2f}', color='blue') # since here first all values for a=0 plotted then a=0.1 etc.
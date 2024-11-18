# %%
#curse of dimensionality
#generating samples of a function and creating an  interpolator which will approximate it as accurately as possible

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
t = np.linspace(0, 1, 100)  # time from 0 to 1 with 100 points

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
#Q3
#going for interpolation
#repeating function definition
#plotting over 100 datapoints will not show much of linearity/difference from the original data
def f(t, a, b, c):
    return np.sqrt(a) * np.exp(-b * t) * np.sin(c * t) + 0.5 * np.cos(2 * t)

# Example usage
# Define parameters
a = 0.1
b = -0.13
c = 9

# Define time range
#keeping same time range
t = np.linspace(0, 1, 1000)  # time from 0 to 1 with 100 points

# Calculate f(t) for these parameters
# saving original data points and interpolated points to match shapes when plotting
f_values = f(t, a, b, c)
interpolator = interp1d(t, f_values, kind='cubic')
f_interpolated = interpolator(t)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot original data points
plt.plot(t, f_values, 'o', label='Original Data', color='blue')
# Plot interpolated curve
plt.plot(t, f_interpolated, '-', label='Interpolated Curve (Cubic)', color='red')



# Add labels, legend, and title
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Original vs Interpolated Function')
plt.legend()
plt.grid(True)
#really weird behaviour of the function here
# %%
print(f_interpolated - )
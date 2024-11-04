import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data
data = pd.read_csv(r'C:\Users\Sondre\Downloads\N_07_extent_v3.0.csv')

# Set the index to 'extent' and clean the column names
data.set_index(' extent', inplace=True)

# Extract 'year', 'extent' and 'sea ice area' for modeling
X = data['year'].values.reshape(-1, 1)  # Reshape for sklearn

# Sea Ice Extent (convert index to float)
y_extent = data.index.astype(float)

# Sea Ice Area (ensure no leading/trailing spaces in column names)
y_area = data['   area'].astype(float)

# Create polynomial features for polynomial regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Create two polynomial regression models (one for extent, one for area)
model_extent_poly = LinearRegression()
model_area_poly = LinearRegression()

# Fit both polynomial models to the data
model_extent_poly.fit(X_poly, y_extent)
model_area_poly.fit(X_poly, y_area)

# Predict the trend for future years (e.g., next 30 years), starting 1 year earlier
future_years = np.arange(data['year'].max(), data['year'].max() + 30).reshape(-1, 1)
future_years_poly = poly.transform(future_years)  # Apply polynomial transformation

# Initial decline factors
decline_factor_extent = 1
decline_factor_area = 1

# Make predictions for future years
future_extent_pred = model_extent_poly.predict(future_years_poly)
future_area_pred = model_area_poly.predict(future_years_poly)

# Function to update the graph
def update(val):
    global decline_factor_extent, decline_factor_area

    # Get the current slider values
    decline_factor_extent = slider_extent.val
    decline_factor_area = slider_area.val

    # Manually adjust the future predictions for extent and area
    future_extent_pred_adjusted = future_extent_pred * (1 - (np.arange(len(future_extent_pred)) / len(future_extent_pred)) * decline_factor_extent)
    future_area_pred_adjusted = future_area_pred * (1 - (np.arange(len(future_area_pred)) / len(future_area_pred)) * decline_factor_area)

    # Clear the current plot
    ax.cla()

    # Plot the original sea ice extent data
    ax.plot(data['year'], y_extent, label='Observed Extent', color='blue')

    # Plot the original sea ice area data
    ax.plot(data['year'], y_area, label='Observed Area', color='red')

    # Plot the polynomial trendline for sea ice extent
    ax.plot(data['year'], model_extent_poly.predict(X_poly), color='cyan', label='Polynomial Trendline (Extent)', linestyle='--')

    # Plot the polynomial trendline for sea ice area
    ax.plot(data['year'], model_area_poly.predict(X_poly), color='magenta', label='Polynomial Trendline (Area)', linestyle='--')

    # Plot the adjusted future predictions for sea ice extent
    ax.plot(future_years, future_extent_pred_adjusted, color='cyan', linestyle='-')

    # Plot the adjusted future predictions for sea ice area
    ax.plot(future_years, future_area_pred_adjusted, color='magenta', linestyle='-')

    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Sea Ice Extent / Area (Millions of sq. km)')
    ax.set_title('Average Sea Ice Extent and Area Over Time with Future Predictions')

    # Limit the y-axis range for clarity
    ax.set_ylim(0, 12)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    # Redraw the updated figure
    plt.draw()

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # Adjust the bottom to make space for sliders

# Slider for decline factor for sea ice extent
ax_decline_extent = plt.axes([0.1, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
slider_extent = Slider(ax_decline_extent, 'Change in CO2 Emissions on Extent', 0.5, 1.1, valinit=decline_factor_extent)

# Slider for decline factor for sea ice area
ax_decline_area = plt.axes([0.1, 0.15, 0.65, 0.03])  # [left, bottom, width, height]
slider_area = Slider(ax_decline_area, 'Change in CO2 Emissions on Area', 0.5, 1.1, valinit=decline_factor_area)

# Connect the sliders to the update function
slider_extent.on_changed(update)
slider_area.on_changed(update)

# Initial plotting of the graph
update(0)  # Initial call to plot the data

# Show the plot with sliders
plt.show()


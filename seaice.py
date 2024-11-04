import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def load_data(file_path):
    """Load and prepare the data from the CSV file."""
    data = pd.read_csv(file_path)
    data.set_index(' extent', inplace=True)  # Set the index
    return data

def prepare_data(data):
    """Extract features and target variables for modeling."""
    X = data['year'].values.reshape(-1, 1)  # Reshape for sklearn
    y_extent = data.index.astype(float)  # Sea Ice Extent
    y_area = data['   area'].astype(float)  # Sea Ice Area
    return X, y_extent, y_area

def fit_polynomial_models(X, y_extent, y_area, degree=3):
    """Fit polynomial regression models for extent and area."""
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model_extent_poly = LinearRegression()
    model_area_poly = LinearRegression()

    model_extent_poly.fit(X_poly, y_extent)
    model_area_poly.fit(X_poly, y_area)

    return model_extent_poly, model_area_poly, poly

def predict_future(model_extent_poly, model_area_poly, future_years, poly):
    """Predict future extent and area based on the fitted models."""
    future_years_poly = poly.transform(future_years)  # Apply polynomial transformation
    future_extent_pred = model_extent_poly.predict(future_years_poly)
    future_area_pred = model_area_poly.predict(future_years_poly)
    return future_extent_pred, future_area_pred

def update(val, ax, data, future_extent_pred, future_area_pred, slider_extent, slider_area, model_extent_poly, model_area_poly, poly, future_years):
    """Update the plot based on slider values."""
    decline_factor_extent = slider_extent.val
    decline_factor_area = slider_area.val

    # Adjust predictions
    future_extent_pred_adjusted = future_extent_pred * (1 - (np.arange(len(future_extent_pred)) / len(future_extent_pred)) * decline_factor_extent)
    future_area_pred_adjusted = future_area_pred * (1 - (np.arange(len(future_area_pred)) / len(future_area_pred)) * decline_factor_area)

    # Clear and plot data
    ax.cla()
    ax.plot(data['year'], data.index.astype(float), label='Observed Extent', color='blue')
    ax.plot(data['year'], data['   area'].astype(float), label='Observed Area', color='red')
    ax.plot(data['year'], model_extent_poly.predict(poly.transform(data['year'].values.reshape(-1, 1))), color='cyan', label='Polynomial Trendline (Extent)', linestyle='--')
    ax.plot(data['year'], model_area_poly.predict(poly.transform(data['year'].values.reshape(-1, 1))), color='magenta', label='Polynomial Trendline (Area)', linestyle='--')
    ax.plot(future_years, future_extent_pred_adjusted, color='cyan', linestyle='-')
    ax.plot(future_years, future_area_pred_adjusted, color='magenta', linestyle='-')

    # Labels and formatting
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Sea Ice Extent / Area (Millions of sq. km)')
    ax.set_title('Average Sea Ice Extent and Area Over Time with Future Predictions')
    ax.set_ylim(0, 12)
    ax.legend()
    ax.grid(True)
    plt.draw()

# Main function to run the analysis
def main():
    file_path = r'C:\Users\Sondre\Downloads\N_07_extent_v3.0.csv'
    data = load_data(file_path)
    X, y_extent, y_area = prepare_data(data)

    model_extent_poly, model_area_poly, poly = fit_polynomial_models(X, y_extent, y_area)

    # Predict future years
    future_years = np.arange(data['year'].max(), data['year'].max() + 30).reshape(-1, 1)
    future_extent_pred, future_area_pred = predict_future(model_extent_poly, model_area_poly, future_years, poly)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Make space for sliders

    # Sliders for decline factors
    ax_decline_extent = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider_extent = Slider(ax_decline_extent, 'Change in CO2 Emissions on Extent', 0.5, 1.1, valinit=1.0)

    ax_decline_area = plt.axes([0.1, 0.15, 0.65, 0.03])
    slider_area = Slider(ax_decline_area, 'Change in CO2 Emissions on Area', 0.5, 1.1, valinit=1.0)

    # Connect the sliders to the update function
    slider_extent.on_changed(lambda val: update(val, ax, data, future_extent_pred, future_area_pred, slider_extent, slider_area, model_extent_poly, model_area_poly, poly, future_years))
    slider_area.on_changed(lambda val: update(val, ax, data, future_extent_pred, future_area_pred, slider_extent, slider_area, model_extent_poly, model_area_poly, poly, future_years))

    # Initial plotting of the graph
    update(0, ax, data, future_extent_pred, future_area_pred, slider_extent, slider_area, model_extent_poly, model_area_poly, poly, future_years)

    # Show the plot with sliders
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()






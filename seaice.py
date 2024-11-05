import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """Load and prepare the data from the CSV file."""
    try:
        data = pd.read_csv(file_path)
        data.set_index(' extent', inplace=True)  # Set the index to 'extent'
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

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

    return model_extent_poly, model_area_poly, poly, X_poly

def predict_future(model_extent_poly, model_area_poly, future_years, poly):
    """Predict future extent and area based on the fitted models."""
    future_years_poly = poly.transform(future_years)  # Apply polynomial transformation
    future_extent_pred = model_extent_poly.predict(future_years_poly)
    future_area_pred = model_area_poly.predict(future_years_poly)
    return future_extent_pred, future_area_pred

def evaluate_model(y_true, y_pred):
    """Evaluate the model performance using RMSE and R^2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Model Evaluation: RMSE = {rmse:.2f}, R^2 = {r2:.2f}")

def update(val, ax, data, future_extent_pred, future_area_pred, future_years, slider_extent, slider_area, X_poly, model_extent_poly, model_area_poly):
    """Update the plot based on slider values."""
    decline_factor_extent = slider_extent.val
    decline_factor_area = slider_area.val

    # Adjust predictions
    future_extent_pred_adjusted = future_extent_pred * (1 - (np.arange(len(future_extent_pred)) / len(future_extent_pred)) * decline_factor_extent)
    future_area_pred_adjusted = future_area_pred * (1 - (np.arange(len(future_area_pred)) / len(future_area_pred)) * decline_factor_area)

    # Clear and plot data
    ax.cla()
    ax.plot(data['year'], data.index, label='Observed Extent', color='blue')
    ax.plot(data['year'], data['   area'], label='Observed Area', color='red')
    ax.plot(data['year'], model_extent_poly.predict(X_poly), color='cyan', label='Polynomial Trendline (Extent)', linestyle='--')
    ax.plot(data['year'], model_area_poly.predict(X_poly), color='magenta', label='Polynomial Trendline (Area)', linestyle='--')
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

    if data is not None:
        X, y_extent, y_area = prepare_data(data)

        # Fit polynomial models
        model_extent_poly, model_area_poly, poly, X_poly = fit_polynomial_models(X, y_extent, y_area)

        # Evaluate the model performance
        evaluate_model(y_extent, model_extent_poly.predict(X_poly))

        # Predict future years
        future_years = np.arange(data['year'].max(), data['year'].max() + 30).reshape(-1, 1)
        future_extent_pred, future_area_pred = predict_future(model_extent_poly, model_area_poly, future_years, poly)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25, top=0.9)  # Make space for sliders and text

        # Sliders for decline factors
        ax_decline_extent = plt.axes([0.1, 0.1, 0.65, 0.03])
        slider_extent = Slider(ax_decline_extent, 'Change in CO2 Emissions on Extent', 0.5, 1.1, valinit=1.0)

        ax_decline_area = plt.axes([0.1, 0.15, 0.65, 0.03])
        slider_area = Slider(ax_decline_area, 'Change in CO2 Emissions on Area', 0.5, 1.1, valinit=1.0)

        # Connect the sliders to the update function
        slider_extent.on_changed(lambda val: update(val, ax, data, future_extent_pred, future_area_pred, future_years, slider_extent, slider_area, X_poly, model_extent_poly, model_area_poly))
        slider_area.on_changed(lambda val: update(val, ax, data, future_extent_pred, future_area_pred, future_years, slider_extent, slider_area, X_poly, model_extent_poly, model_area_poly))

        # Initial plotting of the graph
        update(0, ax, data, future_extent_pred, future_area_pred, future_years, slider_extent, slider_area, X_poly, model_extent_poly, model_area_poly)

        # Create a separate axes for the text box below the plot
        text_ax = fig.add_axes([0.1, -0.02, 0.8, 0.1])  # [left, bottom, width, height]
        text_ax.axis('off')  # Hide the axes

        textstr = (
            "This analysis utilizes real data gathered from the Climate Data Guide (https://climatedataguide.ucar.edu/formats/csv) to plot historical sea ice extent and area. "
            "A polynomial regression model is then created to estimate future trends in sea ice melt. According to projections from the same source, "
            "if current trends continue without intervention, the sea ice area could fall below 1 million square kilometers by 2050. "
            "This model accounts for worst-case scenarios, with the slider allowing predictions that indicate the potential for sea ice area to drop beneath this critical threshold in 2050. \n\n"
            "Disclaimer: This is a school project and should not be considered a scientific estimate of sea ice melt."
        )

        # Add the text to the text_ax
        text_ax.text(0.0, 0.5, textstr, fontsize=8, va='center', wrap=True)

        # Show the plot with sliders
        plt.show()


# Run the main function
if __name__ == "__main__":
    main()











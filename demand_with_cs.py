import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data and clean column names
data = pd.read_csv('Survey Data/v2survey_11-06-24.csv')
data.columns = data.columns.str.strip()

# Define price points and corresponding average demand column names
price_points = {
    'free': 0,
    'half': 1.13,
    'normal': 2.25,
    'extra': 3.37,
    'double': 5.50
}

# Extract average demands and corresponding prices
prices, avg_demand = [], []
for price, value in price_points.items():
    avg_col = f'avg_demand_{price}'
    if avg_col in data.columns:
        avg_value = data[avg_col].mean()
        if not np.isnan(avg_value):
            prices.append(value)
            avg_demand.append(avg_value)

# Create DataFrame and remove N/A values
df = pd.DataFrame({'avg_demand': avg_demand, 'prices': prices}).dropna()

# Set up linear regression model
x = df['avg_demand'].values.reshape(-1,1)
y = df['prices'].values.reshape(-1,1)
model = LinearRegression()
model.fit(x, y)

# Calculate the x-intercept (quantity where price is zero)
a = model.intercept_[0]  # Intercept (willingness to pay at zero demand)
b = model.coef_[0][0]    # Slope
max_quantity = -a / b     # Quantity at zero price

# Consumer surplus calculation function
def calculate_consumer_surplus(intercept, max_quantity):
    return 0.5 * max_quantity * intercept

# Calculate consumer surplus
consumer_surplus = calculate_consumer_surplus(a, max_quantity)
print(consumer_surplus)

# Generate demand curve predictions
extended_demand = np.linspace(0, max(x), 100).reshape(-1,1)
predicted_prices = model.predict(extended_demand)

# Demand Curve Visualization with Consumer Surplus annotation
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label='Avg. Demand at Price Point')
plt.plot(extended_demand, predicted_prices, 'black', label='Demand Curve')

# Annotate consumer surplus on the plot
plt.fill_between(extended_demand.flatten(), 0, predicted_prices.flatten(), where=(extended_demand.flatten() <= max_quantity), color='lightgreen', alpha=0.5, label=f'Consumer Surplus = ${consumer_surplus:.2f}')

plt.title('Demand Curve for Berkeley AC Transit Buses')
plt.xlabel('Quantity Demanded (Rides per Week)')
plt.ylabel('Price ($ per Ride)')
plt.ylim(bottom=-0.5)
plt.legend()
plt.grid(True)
plt.show()

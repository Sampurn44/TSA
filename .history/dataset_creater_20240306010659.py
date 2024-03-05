import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate time series data for sales with different scenarios
def generate_sales_data(length, trend_type, seasonality_coef, noise_std):
    time = np.arange(length)
    noise = np.random.normal(0, noise_std, length)
    
    if trend_type == 'linear_up':
        trend = 0.5 * time
    elif trend_type == 'exp_down':
        trend = 100 * np.exp(-0.05 * time)
    elif trend_type == 'exp_up':
        trend = 10 * np.exp(0.02 * time)
    else:  # constant
        trend = np.zeros(length)
    
    seasonality = seasonality_coef * np.sin(2 * np.pi * time / 12)
    
    # Generate sales data and add the absolute value of the largest negative number
    sales_data = trend + seasonality + noise
    sales_data += np.abs(np.min(sales_data))
    
    return sales_data

# Number of years
years = 10

# Number of months
months = years * 12

# Create DataFrame to store time series data
df_sales = pd.DataFrame(index=pd.date_range(start='1/1/2010', periods=months, freq='ME'))

# Generate sales data for different scenarios in each category with different seasonality and noise
df_sales['Tshirt'] = generate_sales_data(months, 'linear_up', 50, 30)
df_sales['Shirt'] = generate_sales_data(months, 'exp_down', 30, 20)
df_sales['Jeans'] = generate_sales_data(months, 'exp_up', 20, 15)
df_sales['Pant'] = generate_sales_data(months, 'constant', 0, 10)

# Adjust seasonality and noise for each category
df_sales['Tshirt'] += 10 * np.sin(2 * np.pi * df_sales.index.month / 12) + 10 * np.random.normal(0, 5, months)
df_sales['Shirt'] += 5 * np.sin(2 * np.pi * df_sales.index.month / 6) + 8 * np.random.normal(0, 4, months)
df_sales['Jeans'] += 8 * np.sin(2 * np.pi * df_sales.index.month / 4) + 7 * np.random.normal(0, 3, months)
df_sales['Pant'] += 3 * np.sin(2 * np.pi * df_sales.index.month / 12) + 5 * np.random.normal(0, 2, months)

# Ensure there are no negative values
df_sales = np.maximum(df_sales, 0)

# Export DataFrame to CSV
df_sales.to_csv('time_series_dataset.csv', index=True)

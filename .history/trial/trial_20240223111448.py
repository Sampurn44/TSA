import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df = pd.read_csv('time_series_dataset.csv')
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Tshirt'], label='Tshirt')
plt.plot(df.index, df['Shirt'], label='Shirt')
plt.plot(df.index, df['Jeans'], label='Jeans')
plt.plot(df.index, df['Pant'], label='Pant')
plt.title('Monthly Sales Over 10 Years with Different Scenarios')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Tshirt'], label='Tshirt')
plt.title('Monthly Sales Over 10 Years with Different Scenarios')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
tshirt_data = df['Tshirt']
result = seasonal_decompose(tshirt_data, model='additive', period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tshirt_data, label='Original')
plt.legend()
plt.title('Original Time Series')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.title('Trend Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.title('Seasonal Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend()
plt.title('Residual Component')
plt.tight_layout()
plt.show()
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_1, test0_1 = tshirt_data[:train_size], tshirt_data[train_size:]
model = ExponentialSmoothing(train0_1, seasonal='add', seasonal_periods=12, trend='add', damped_trend=True)
fit_model = model.fit()
forecast_values = fit_model.forecast(steps=len(test0_1))
plt.figure(figsize=(12, 6))
plt.plot(train0_1.index, train0_1, label='Train')
plt.plot(test0_1.index, test0_1, label='Test')
plt.plot(test0_1.index, forecast_values, label='Forecast', linestyle='--', color='green')
plt.title('Holt-Winters Exponential Smoothing Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_1 = mean_absolute_error(test0_1, forecast_values)
print(f"Mean Absolute Error (MAE): {mae0_1}")
# Calculate Mean Squared Error (MSE)
mse0_1 = mean_squared_error(test0_1, forecast_values)
print(f"Mean Squared Error (MSE): {mse0_1}")
rmse0_1 = np.sqrt(mse0_1)
print(f"Root Mean Squared Error (RMSE): {rmse0_1}")
# Calculate Simple Moving Average (SMA)
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()

train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_2, test0_2 = tshirt_data[:train_size], tshirt_data[train_size:]
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()
forecast_values_sma_0 = sma[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Train',color='black')
plt.plot(test0_2.index, test0_2, label='Test',color='green')
#plt.plot(tshirt_data, label='Original')
plt.plot(tshirt_data, label='Original')
plt.plot(test0_2.index, test0_2, label='Test')
plt.plot(test0_2.index, forecast_values_sma_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Moving Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_2 = mean_absolute_error(test0_2, forecast_values_sma_0)
print(f"Mean Absolute Error (MAE): {mae0_2}")

# Calculate Mean Squared Error (MSE)
mse0_2 = mean_squared_error(test0_2, forecast_values_sma_0)
print(f"Mean Squared Error (MSE): {mse0_2}")

rmse0_2 = np.sqrt(mse0_2)
print(f"Root Mean Squared Error (RMSE): {rmse0_2}")
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_3, test0_3= tshirt_data[:train_size], tshirt_data[train_size:]
# Calculate Simple Average
simple_average = train0_3.mean()

# Repeat the simple average for the length of the test set
forecast_values_simple_average_0 = pd.Series([simple_average] * len(test0_3), index=test0_3.index)

# Plot the original time series, test set, and Simple Average forecast
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Original')
plt.plot(test0_3.index, test0_3, label='Test')
plt.plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Simple Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_3 = mean_absolute_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Absolute Error (MAE): {mae0_3}")

# Calculate Mean Squared Error (MSE)
mse0_3 = mean_squared_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Squared Error (MSE): {mse0_3}")

rmse0_3 = np.sqrt(mse0_3)
print(f"Root Mean Squared Error (RMSE): {rmse0_3}")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot for Exponential
axes[0, 0].plot(test0_1.index, test0_1, label='Exponential')
axes[0, 0].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Exponential Forecasting\nRMSE: {:.2f}'.format(rmse0_1))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Moving
axes[0, 1].plot(test0_2.index, test0_2, label='Moving Average')
axes[0, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Moving Average Forecast\nRMSE: {:.2f}'.format(rmse0_2))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Simple
axes[1, 0].plot(test0_3.index, test0_3, label='Simple Average')
axes[1, 0].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Simple Average Forecast\nRMSE: {:.2f}'.format(rmse0_3))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test0_1.index, test0_1, label='Tshirt Test')
axes[1, 1].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[1, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[1, 1].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 1].set_title('TShirt Forecasting')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Tshirt'], label='Tshirt')
plt.title('Monthly Sales Over 10 Years with Different Scenarios')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
tshirt_data = df['Tshirt']
result = seasonal_decompose(tshirt_data, model='additive', period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tshirt_data, label='Original')
plt.legend()
plt.title('Original Time Series')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.title('Trend Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.title('Seasonal Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend()
plt.title('Residual Component')
plt.tight_layout()
plt.show()
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_1, test0_1 = tshirt_data[:train_size], tshirt_data[train_size:]
model = ExponentialSmoothing(train0_1, seasonal='add', seasonal_periods=12, trend='add', damped_trend=True)
fit_model = model.fit()
forecast_values = fit_model.forecast(steps=len(test0_1))
plt.figure(figsize=(12, 6))
plt.plot(train0_1.index, train0_1, label='Train')
plt.plot(test0_1.index, test0_1, label='Test')
plt.plot(test0_1.index, forecast_values, label='Forecast', linestyle='--', color='green')
plt.title('Holt-Winters Exponential Smoothing Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_1 = mean_absolute_error(test0_1, forecast_values)
print(f"Mean Absolute Error (MAE): {mae0_1}")
# Calculate Mean Squared Error (MSE)
mse0_1 = mean_squared_error(test0_1, forecast_values)
print(f"Mean Squared Error (MSE): {mse0_1}")
rmse0_1 = np.sqrt(mse0_1)
print(f"Root Mean Squared Error (RMSE): {rmse0_1}")
# Calculate Simple Moving Average (SMA)
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()

train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_2, test0_2 = tshirt_data[:train_size], tshirt_data[train_size:]
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()
forecast_values_sma_0 = sma[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Train',color='black')
plt.plot(test0_2.index, test0_2, label='Test',color='green')
#plt.plot(tshirt_data, label='Original')
plt.plot(tshirt_data, label='Original')
plt.plot(test0_2.index, test0_2, label='Test')
plt.plot(test0_2.index, forecast_values_sma_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Moving Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_2 = mean_absolute_error(test0_2, forecast_values_sma_0)
print(f"Mean Absolute Error (MAE): {mae0_2}")

# Calculate Mean Squared Error (MSE)
mse0_2 = mean_squared_error(test0_2, forecast_values_sma_0)
print(f"Mean Squared Error (MSE): {mse0_2}")

rmse0_2 = np.sqrt(mse0_2)
print(f"Root Mean Squared Error (RMSE): {rmse0_2}")
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_3, test0_3= tshirt_data[:train_size], tshirt_data[train_size:]
# Calculate Simple Average
simple_average = train0_3.mean()

# Repeat the simple average for the length of the test set
forecast_values_simple_average_0 = pd.Series([simple_average] * len(test0_3), index=test0_3.index)

# Plot the original time series, test set, and Simple Average forecast
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Original')
plt.plot(test0_3.index, test0_3, label='Test')
plt.plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Simple Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_3 = mean_absolute_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Absolute Error (MAE): {mae0_3}")

# Calculate Mean Squared Error (MSE)
mse0_3 = mean_squared_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Squared Error (MSE): {mse0_3}")

rmse0_3 = np.sqrt(mse0_3)
print(f"Root Mean Squared Error (RMSE): {rmse0_3}")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot for Exponential
axes[0, 0].plot(test0_1.index, test0_1, label='Exponential')
axes[0, 0].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Exponential Forecasting\nRMSE: {:.2f}'.format(rmse0_1))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Moving
axes[0, 1].plot(test0_2.index, test0_2, label='Moving Average')
axes[0, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Moving Average Forecast\nRMSE: {:.2f}'.format(rmse0_2))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Simple
axes[1, 0].plot(test0_3.index, test0_3, label='Simple Average')
axes[1, 0].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Simple Average Forecast\nRMSE: {:.2f}'.format(rmse0_3))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test0_1.index, test0_1, label='Tshirt Test')
axes[1, 1].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[1, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[1, 1].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 1].set_title('TShirt Forecasting')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Tshirt'], label='Tshirt')
plt.title('Monthly Sales Over 10 Years with Different Scenarios')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
tshirt_data = df['Tshirt']
result = seasonal_decompose(tshirt_data, model='additive', period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tshirt_data, label='Original')
plt.legend()
plt.title('Original Time Series')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.title('Trend Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.title('Seasonal Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend()
plt.title('Residual Component')
plt.tight_layout()
plt.show()
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_1, test0_1 = tshirt_data[:train_size], tshirt_data[train_size:]
model = ExponentialSmoothing(train0_1, seasonal='add', seasonal_periods=12, trend='add', damped_trend=True)
fit_model = model.fit()
forecast_values = fit_model.forecast(steps=len(test0_1))
plt.figure(figsize=(12, 6))
plt.plot(train0_1.index, train0_1, label='Train')
plt.plot(test0_1.index, test0_1, label='Test')
plt.plot(test0_1.index, forecast_values, label='Forecast', linestyle='--', color='green')
plt.title('Holt-Winters Exponential Smoothing Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_1 = mean_absolute_error(test0_1, forecast_values)
print(f"Mean Absolute Error (MAE): {mae0_1}")
# Calculate Mean Squared Error (MSE)
mse0_1 = mean_squared_error(test0_1, forecast_values)
print(f"Mean Squared Error (MSE): {mse0_1}")
rmse0_1 = np.sqrt(mse0_1)
print(f"Root Mean Squared Error (RMSE): {rmse0_1}")
# Calculate Simple Moving Average (SMA)
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()

train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_2, test0_2 = tshirt_data[:train_size], tshirt_data[train_size:]
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()
forecast_values_sma_0 = sma[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Train',color='black')
plt.plot(test0_2.index, test0_2, label='Test',color='green')
#plt.plot(tshirt_data, label='Original')
plt.plot(tshirt_data, label='Original')
plt.plot(test0_2.index, test0_2, label='Test')
plt.plot(test0_2.index, forecast_values_sma_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Moving Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_2 = mean_absolute_error(test0_2, forecast_values_sma_0)
print(f"Mean Absolute Error (MAE): {mae0_2}")

# Calculate Mean Squared Error (MSE)
mse0_2 = mean_squared_error(test0_2, forecast_values_sma_0)
print(f"Mean Squared Error (MSE): {mse0_2}")

rmse0_2 = np.sqrt(mse0_2)
print(f"Root Mean Squared Error (RMSE): {rmse0_2}")
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_3, test0_3= tshirt_data[:train_size], tshirt_data[train_size:]
# Calculate Simple Average
simple_average = train0_3.mean()

# Repeat the simple average for the length of the test set
forecast_values_simple_average_0 = pd.Series([simple_average] * len(test0_3), index=test0_3.index)

# Plot the original time series, test set, and Simple Average forecast
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Original')
plt.plot(test0_3.index, test0_3, label='Test')
plt.plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Simple Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_3 = mean_absolute_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Absolute Error (MAE): {mae0_3}")

# Calculate Mean Squared Error (MSE)
mse0_3 = mean_squared_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Squared Error (MSE): {mse0_3}")

rmse0_3 = np.sqrt(mse0_3)
print(f"Root Mean Squared Error (RMSE): {rmse0_3}")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot for Exponential
axes[0, 0].plot(test0_1.index, test0_1, label='Exponential')
axes[0, 0].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Exponential Forecasting\nRMSE: {:.2f}'.format(rmse0_1))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Moving
axes[0, 1].plot(test0_2.index, test0_2, label='Moving Average')
axes[0, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Moving Average Forecast\nRMSE: {:.2f}'.format(rmse0_2))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Simple
axes[1, 0].plot(test0_3.index, test0_3, label='Simple Average')
axes[1, 0].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Simple Average Forecast\nRMSE: {:.2f}'.format(rmse0_3))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test0_1.index, test0_1, label='Tshirt Test')
axes[1, 1].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[1, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[1, 1].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 1].set_title('TShirt Forecasting')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Tshirt'], label='Tshirt')
plt.title('Monthly Sales Over 10 Years with Different Scenarios')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
tshirt_data = df['Tshirt']
result = seasonal_decompose(tshirt_data, model='additive', period=12)
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tshirt_data, label='Original')
plt.legend()
plt.title('Original Time Series')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.title('Trend Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.title('Seasonal Component')
plt.tight_layout()
plt.show()
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend()
plt.title('Residual Component')
plt.tight_layout()
plt.show()
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_1, test0_1 = tshirt_data[:train_size], tshirt_data[train_size:]
model = ExponentialSmoothing(train0_1, seasonal='add', seasonal_periods=12, trend='add', damped_trend=True)
fit_model = model.fit()
forecast_values = fit_model.forecast(steps=len(test0_1))
plt.figure(figsize=(12, 6))
plt.plot(train0_1.index, train0_1, label='Train')
plt.plot(test0_1.index, test0_1, label='Test')
plt.plot(test0_1.index, forecast_values, label='Forecast', linestyle='--', color='green')
plt.title('Holt-Winters Exponential Smoothing Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_1 = mean_absolute_error(test0_1, forecast_values)
print(f"Mean Absolute Error (MAE): {mae0_1}")
# Calculate Mean Squared Error (MSE)
mse0_1 = mean_squared_error(test0_1, forecast_values)
print(f"Mean Squared Error (MSE): {mse0_1}")
rmse0_1 = np.sqrt(mse0_1)
print(f"Root Mean Squared Error (RMSE): {rmse0_1}")
# Calculate Simple Moving Average (SMA)
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()

train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_2, test0_2 = tshirt_data[:train_size], tshirt_data[train_size:]
window_size = 3  # Adjust the window size as needed
sma = tshirt_data.rolling(window=window_size).mean()
forecast_values_sma_0 = sma[train_size:]
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Train',color='black')
plt.plot(test0_2.index, test0_2, label='Test',color='green')
#plt.plot(tshirt_data, label='Original')
plt.plot(tshirt_data, label='Original')
plt.plot(test0_2.index, test0_2, label='Test')
plt.plot(test0_2.index, forecast_values_sma_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Moving Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_2 = mean_absolute_error(test0_2, forecast_values_sma_0)
print(f"Mean Absolute Error (MAE): {mae0_2}")

# Calculate Mean Squared Error (MSE)
mse0_2 = mean_squared_error(test0_2, forecast_values_sma_0)
print(f"Mean Squared Error (MSE): {mse0_2}")

rmse0_2 = np.sqrt(mse0_2)
print(f"Root Mean Squared Error (RMSE): {rmse0_2}")
train_size = int(len(tshirt_data) * 0.8)  # Use 80% of the data for training
train0_3, test0_3= tshirt_data[:train_size], tshirt_data[train_size:]
# Calculate Simple Average
simple_average = train0_3.mean()

# Repeat the simple average for the length of the test set
forecast_values_simple_average_0 = pd.Series([simple_average] * len(test0_3), index=test0_3.index)

# Plot the original time series, test set, and Simple Average forecast
plt.figure(figsize=(12, 6))
plt.plot(tshirt_data, label='Original')
plt.plot(test0_3.index, test0_3, label='Test')
plt.plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='orange')
plt.title('Simple Average Forecasting')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
# Calculate Mean Absolute Error (MAE)
mae0_3 = mean_absolute_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Absolute Error (MAE): {mae0_3}")

# Calculate Mean Squared Error (MSE)
mse0_3 = mean_squared_error(test0_3, forecast_values_simple_average_0)
print(f"Mean Squared Error (MSE): {mse0_3}")

rmse0_3 = np.sqrt(mse0_3)
print(f"Root Mean Squared Error (RMSE): {rmse0_3}")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Plot for Exponential
axes[0, 0].plot(test0_1.index, test0_1, label='Exponential')
axes[0, 0].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Exponential Forecasting\nRMSE: {:.2f}'.format(rmse0_1))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Moving
axes[0, 1].plot(test0_2.index, test0_2, label='Moving Average')
axes[0, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Moving Average Forecast\nRMSE: {:.2f}'.format(rmse0_2))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Simple
axes[1, 0].plot(test0_3.index, test0_3, label='Simple Average')
axes[1, 0].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Simple Average Forecast\nRMSE: {:.2f}'.format(rmse0_3))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test0_1.index, test0_1, label='Tshirt Test')
axes[1, 1].plot(test0_1.index, forecast_values, label='Exponential Forecast', linestyle='--', color='pink')
axes[1, 1].plot(test0_2.index, forecast_values_sma_0, label='Moving Average Forecast', linestyle='--', color='black')
axes[1, 1].plot(test0_3.index, forecast_values_simple_average_0, label='Simple Average Forecast', linestyle='--', color='purple')
axes[1, 1].set_title('TShirt Forecasting')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

from math import sqrt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))


# Plot for Tshirt
axes[0, 0].plot(test0_1.index, test0_1, label='Tshirt')
axes[0, 0].plot(test0_1.index, forecast_values, label='Tshirt Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Tshirt Exponential Forecasting\nRMSE: {:.2f}'.format(rmse0_1))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Shirt
axes[0, 1].plot(test1_1.index, test1_1, label='Shirt Test')
axes[0, 1].plot(test1_1.index, forecast_values1, label='Shirt Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Shirt Exponential Forecasting\nRMSE: {:.2f}'.format(rmse1_1))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Jeans
axes[1, 0].plot(test2_1.index, test2_1, label='Jeans Test')
axes[1, 0].plot(test2_1.index, forecast_values2, label='Jeans Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Jeans Exponential Forecasting\nRMSE: {:.2f}'.format(rmse2_1))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test3_1.index, test3_1, label='Pant Test')
axes[1, 1].plot(test3_1.index, forecast_values3, label='Pant Forecast', linestyle='--', color='indigo')
axes[1, 1].set_title('Pant Exponential Forecasting\nRMSE: {:.2f}'.format(rmse3_1))
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

from math import sqrt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))


# Plot for Tshirt
axes[0, 0].plot(test0_1.index, test0_1, label='Tshirt')
axes[0, 0].plot(test0_1.index, forecast_values_sma_0, label='Tshirt Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Tshirt Moving Average Forecasting\nRMSE: {:.2f}'.format(rmse0_2))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Shirt
axes[0, 1].plot(test1_1.index, test1_1, label='Shirt Test')
axes[0, 1].plot(test1_1.index, forecast_values_sma_1, label='Shirt Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Shirt Moving Average Forecasting\nRMSE: {:.2f}'.format(rmse1_2))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Jeans
axes[1, 0].plot(test2_1.index, test2_1, label='Jeans Test')
axes[1, 0].plot(test2_1.index, forecast_values_sma_2, label='Jeans Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Jeans Moving Average Forecasting\nRMSE: {:.2f}'.format(rmse2_2))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test3_1.index, test3_1, label='Pant Test')
axes[1, 1].plot(test3_1.index, forecast_values_sma_3, label='Pant Forecast', linestyle='--', color='indigo')
axes[1, 1].set_title('Pant Moving Average Forecasting\nRMSE: {:.2f}'.format(rmse3_2))
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

from math import sqrt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))


# Plot for Tshirt
axes[0, 0].plot(test0_3.index, test0_3, label='Tshirt')
axes[0, 0].plot(test0_3.index, forecast_values_simple_average_0, label='Tshirt Forecast', linestyle='--', color='pink')
axes[0, 0].set_title('Tshirt Simple Average Forecasting\nRMSE: {:.2f}'.format(rmse0_3))
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].legend()

# Plot for Shirt
axes[0, 1].plot(test1_3.index, test1_3, label='Shirt Test')
axes[0, 1].plot(test1_3.index, forecast_values_simple_average_1, label='Shirt Forecast', linestyle='--', color='black')
axes[0, 1].set_title('Shirt Simple Average Forecasting\nRMSE: {:.2f}'.format(rmse1_3))
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Sales')
axes[0, 1].legend()

# Plot for Jeans
axes[1, 0].plot(test2_3.index, test2_3, label='Jeans Test')
axes[1, 0].plot(test2_3.index, forecast_values_simple_average_2, label='Jeans Forecast', linestyle='--', color='purple')
axes[1, 0].set_title('Jeans Simple Average Forecasting\nRMSE: {:.2f}'.format(rmse2_3))
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Sales')
axes[1, 0].legend()

# Plot for Pant
axes[1, 1].plot(test3_3.index, test3_3, label='Pant Test')
axes[1, 1].plot(test3_3.index, forecast_values_simple_average_3, label='Pant Forecast', linestyle='--', color='indigo')
axes[1, 1].set_title('Pant Simple Average Forecasting\nRMSE: {:.2f}'.format(rmse3_3))
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

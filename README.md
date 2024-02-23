# Time Series Analysis Project

## Overview

This project involves time series analysis for sales data of different clothing categories (Tshirt, Shirt, Jeans, Pant). The analysis includes data preprocessing, dataset generation, and forecasting using various methods like Exponential Smoothing, Moving Average, and Simple Average.

## Getting Started

### Prerequisites

- Python (version 3.x)
- Pandas
- Statsmodels
- Matplotlib
- Scikit-learn

### Theory
Let’s assume we are dealing with the additive model, that is, consisting of a linear trend and seasonal cycle with the same frequency (width) and amplitude (height). For the multiplicative model, you just need to replace the additions with multiplications and subtractions with divisions.

#### Trend component
Trend is calculated using a centered moving average of the time series. The moving average is calculated using a window length corresponding to the frequency of the time series. For example, we would use a window of length 12 for monthly data.

Smoothing the series using such a moving average comes together with some disadvantages. First, we are “losing” the first and last few observations of the series. Second, the MA tends to over-smooth the series, which makes it less reactive to sudden changes in the trend or jumps.

#### Seasonal component
To calculate the seasonal component, we first need to detrend the time series. We do it by subtracting the trend component from the original time series (remember, we divide for the multiplicative variant).

Having done that, we calculate the average values of the detrended series for each seasonal period. In the case of months, we would calculate the average detrended value for each month.

The seasonal component is simply built from the seasonal averages repeated for the length of the entire series Again, this is one of the arguments against using the simple seasonal decomposition — the seasonal component is not allowed to change over time, which can be a very strict and often unrealistic assumption for longer time series.

On a side note, in the additive decomposition the detrended series is centered at zero, as adding zero makes no change to the trend. The same logic is applied in the multiplicative approach, with the difference that it is centered around one. That is because multiplying the trend by one also has no effect on it.

#### Residuals
The last component is simply what is left after removing (by subtracting or dividing) the trend and seasonal components from the original time series.

That would be all for the theory, let’s code!

### Installation

To install the required dependencies, run:

```bash
pip install pandas statsmodels matplotlib scikit-learn
```

### Project Structure
**1. dataset_generator.py**: Python script to generate a synthetic time series dataset with multiple categories, trends, and seasonality.

**2. tsa.ipynb**: Python script for forecasting analysis using Exponential Smoothing, Moving Average, and Simple Average.

**3. time_series_dataset.csv**: Generated synthetic time series dataset.

**4. manual.ipynb**: Python script for manual analysis of time series components trend, seasonality, and Residue/Noise using additivie approach

## Usage
1. Generate Dataset:
   To generate a synthetic time series dataset, run:
  ```bash
  python dataset_generator.py
  ```
  This script generates a synthetic time series dataset and saves it as time_series_dataset.csv.
  #### But we already have the dataset we need not to run this code ####

2. Forecasting Analysis:
  To perform forecasting analysis using Exponential Smoothing, Moving Average, and Simple Average, run:  
  ```bash
  python manual.ipynb
  python tsa.ipynb
  ```
  This script displays results with plots for each category and also the manual analysis for the pant sales.

## Results
The manual.ipynb file outputs the trend,sesaonality and noise for the pant category done manually without using the seasonal_decompose library
The forecasting analysis results, including plots and RMSE values, can be found in the output of the tsa.ipynb script.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

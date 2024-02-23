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

### Installation

To install the required dependencies, run:

```bash
pip install pandas statsmodels matplotlib scikit-learn
```

### Project Structure
**1. dataset_generator.py**: Python script to generate a synthetic time series dataset with multiple categories, trends, and seasonality.

** tsa.ipynb**: Python script for forecasting analysis using Exponential Smoothing, Moving Average, and Simple Average.

**3. time_series_dataset.csv**: Generated synthetic time series dataset.
## Usage
1. Generate Dataset:
   To generate a synthetic time series dataset, run:
  ```bash
  python dataset_generator.py
  ```
  This script generates a synthetic time series dataset and saves it as time_series_dataset.csv.

2. Forecasting Analysis:
  To perform forecasting analysis using Exponential Smoothing, Moving Average, and Simple Average, run:  
  ```bash
  python forecasting_analysis.py
  ```
  This script displays results with plots for each category.

## Results
The forecasting analysis results, including plots and RMSE values, can be found in the output of the forecasting_analysis.py script.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

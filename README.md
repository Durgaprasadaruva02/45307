# FTSE 100 Index Volatility Modeling and Forecasting

This project performs a comprehensive time series analysis on the FTSE 100 Index daily data, focusing on calculating logarithmic returns and subsequently modeling both the mean (using ARIMA) and volatility (using GARCH and LSTM) components.

## Data Source

The historical stock market data for the FTSE 100 Index is sourced from **Yahoo Finance**, a publicly available data provider, using the `yfinance` Python library.

| Parameter | Value |
| :--- | :--- |
| Ticker Symbol | `^FTSE` (FTSE 100 Index) |
| Start Date | 2019-01-01 |
| End Date | 2025-01-01 (exclusive) |
| Total Entries | 1513 daily data points (after cleaning and return calculation) |

## Project Steps and Methodology

The analysis is structured into four main steps: Data Acquisition and Cleaning, Log Returns Calculation, Visualization, and Model Fitting/Evaluation.

### 1. Data Acquisition and Preprocessing

- Data for the `^FTSE` ticker was downloaded using `yfinance`.
- **Column Flattening:** MultiIndex columns (e.g., `('Close', '^FTSE')`) were flattened to single-level names (e.g., `'Close'`).
- **Column Renaming:** The `'Close'` column was renamed to `'Adj Close'` for consistency in financial time series analysis.
- **Missing Values:** Missing values were handled using **forward-fill (`ffill`)** to propagate the last valid observation forward, followed by a final `dropna` to remove any remaining boundary NaNs.
- **Index Check:** The index was verified and sorted to ensure uniqueness and chronological order.

### 2. Log Returns Calculation

Daily logarithmic returns (`Log_Returns`) were calculated using the formula:
$$
R_t = \ln \left( \frac{\text{Adj Close}_t}{\text{Adj Close}_{t-1}} \right)
$$
The first row containing an `NaN` return value was dropped. The final dataset shape is `(1513, 6)`.

### 3. Data Visualization

- **Adjusted Close Price Plot:** Shows the overall price trend of the FTSE 100 Index.
- **Daily Log Returns Plot:** Highlights the characteristic clustering of volatility in the return series.
- **Log Returns Histogram:** Confirms the leptokurtic (fat-tailed) nature of financial returns.

### 4. Time Series Modeling and Evaluation (ARIMA, GARCH, LSTM)

The processed log returns series was split into an **80% Training Set** (1210 samples) and a **20% Testing Set** (303 samples).

#### 4.1. ARIMA Model (Mean Prediction)

- **Stationarity Check (ADF Test):** The Augmented Dickey-Fuller test indicated that the log returns series is **stationary** (p-value: 7.38e-12).
- **Model Selection (`auto_arima`):** The optimal ARIMA order was found to be **ARIMA(0, 0, 0)**, suggesting the log returns series behaves like white noise with respect to the mean.
- **Performance (Predicting Returns):**
    - **RMSE:** 0.005942
    - **MAPE:** 101.83% (High MAPE is typical for return prediction.)

#### 4.2. GARCH Model (Volatility Forecasting)

GARCH models were used to forecast volatility (using absolute log returns as the target). Models utilized a **Zero Mean** and **Student's t** distribution.

- **Hyperparameter Tuning:** Multiple GARCH($p, q$) experiments were conducted.
- **Best GARCH Model (Based on MAPE):** **GARCH(1,2)**
    - **RMSE:** 0.004800
    - **MAPE (Mean Absolute Percentage Error):** 581.07%

#### 4.3. LSTM Model (Volatility Forecasting)

An LSTM network was employed, with absolute log returns scaled to the range `(-1, 1)` as the target.

- **Sequence Creation:** A look-back window (`n_steps`) was used for sequence input.
- **Hyperparameter Tuning:** 6 experiments were run, varying the sequence length, units, dropout, and batch size.
- **Best LSTM Model (Based on MAPE):** **n\_steps=60, units=50, dropout=0.1, batch\_size=32** (Experiment 5)
    - **RMSE:** 0.003904
    - **MAPE (Mean Absolute Percentage Error):** **237.61%**

## Key Findings and Model Comparison

| Model | Forecast Target | Optimal Order/Config | RMSE (Volatility) | MAPE (Volatility) |
| :--- | :--- | :--- | :--- | :--- |
| **ARIMA** | Returns ($\mu_t$) | (0, 0, 0) | N/A | 101.83% (on $\mu_t$) |
| **GARCH** | Volatility ($\sigma_t$) | (1, 2) | 0.004800 | 581.07% |
| **LSTM** | Volatility ($\sigma_t$) | 60 steps, 50 units, 0.1 dropout | **0.003904** | **237.61%** |

The **LSTM model achieved the lowest MAPE and RMSE** for volatility forecasting (absolute log returns), demonstrating superior performance in capturing the non-linear dynamics of volatility clustering compared to the GARCH model in this analysis.

## Prerequisites

The following Python libraries are required to run this project. They can be installed using `pip`:

```bash
pip install yfinance pandas numpy matplotlib seaborn pmdarima arch tensorflow keras scikit-learn
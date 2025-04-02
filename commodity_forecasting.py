import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date')

# List available commodities
commodity_columns = [col for col in df.columns if '_closing_price' in col]
commodities = [col.split('_')[0] for col in commodity_columns]
print(f"Available commodities: {', '.join(commodities)}")

# For demonstration, we'll focus on gold (GC=F) as it typically has good data quality
target_commodity = 'GC=F_closing_price'
print(f"\nPerforming time series analysis on {target_commodity.split('_')[0]}")

# Get data for the selected commodity
commodity_data = df[['Date', target_commodity]].copy()
commodity_data = commodity_data.dropna()
commodity_data.set_index('Date', inplace=True)

# If we have enough data, focus on the last 5 years
if len(commodity_data) > 5*252:  # Approx 252 trading days per year
    start_date = commodity_data.index[-5*252]
    commodity_data = commodity_data[commodity_data.index >= start_date]

print(f"Analysis period: {commodity_data.index.min().strftime('%d %b %Y')} to {commodity_data.index.max().strftime('%d %b %Y')}")
print(f"Number of data points: {len(commodity_data)}")

# 1. Visualize the time series
plt.figure(figsize=(15, 7))
plt.plot(commodity_data)
plt.title(f'{target_commodity.split("_")[0]} Price Time Series', fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylabel('Price (USD)', fontsize=14)
plt.savefig('timeseries_plot.png')

# 2. Decompose the time series to trend, seasonal, and residual components
print("\nDecomposing time series into trend, seasonal, and residual components...")
# Resample to monthly data for clearer seasonality patterns
monthly_data = commodity_data.resample('M').mean()

# Perform decomposition
decomposition = seasonal_decompose(monthly_data, model='additive', period=12)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed', fontsize=14)
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend', fontsize=14)
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal', fontsize=14)
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual', fontsize=14)
plt.tight_layout()
plt.savefig('time_series_decomposition.png')
print("Saved time series decomposition as 'time_series_decomposition.png'")

# 3. Autocorrelation and Partial Autocorrelation Analysis
print("\nAnalyzing autocorrelation patterns...")
# Calculate maximum allowable lags (up to 50% of sample size)
max_lags = len(monthly_data) // 2 - 1

# Use the calculated max_lags value (or a sensible default if sample is very small)
lags_to_use = min(max(max_lags, 12), 36)  # At least 12, at most 36 lags
print(f"Using {lags_to_use} lags for ACF/PACF analysis (data size: {len(monthly_data)} months)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
plot_acf(monthly_data, ax=ax1, lags=lags_to_use)
ax1.set_title('Autocorrelation Function', fontsize=14)
plot_pacf(monthly_data, ax=ax2, lags=lags_to_use)
ax2.set_title('Partial Autocorrelation Function', fontsize=14)
plt.tight_layout()
plt.savefig('autocorrelation_analysis.png')
print("Saved autocorrelation analysis as 'autocorrelation_analysis.png'")

# 4. ARIMA Modeling and Forecasting
print("\nBuilding ARIMA forecast model...")
# Split data into train and test
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data[0:train_size], monthly_data[train_size:]
print(f"Training data: {train.index.min().strftime('%d %b %Y')} to {train.index.max().strftime('%d %b %Y')}")
print(f"Testing data: {test.index.min().strftime('%d %b %Y')} to {test.index.max().strftime('%d %b %Y')}")

# Fit ARIMA model - using auto_arima would be better but keeping it simple
# Parameters (p,d,q) = (1,1,1) is just a starting point
print("Fitting ARIMA model...")
try:
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()
    print("\nARIMA Model Summary:")
    print(model_fit.summary())
    
    # Forecast
    forecast_steps = len(test)
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = test.index
    
    # Calculate forecast error metrics
    # Make sure test and forecast have the same shape for calculations
    test_values = test.values.flatten()
    forecast_values = forecast
    
    # Check if lengths match
    if len(test_values) != len(forecast_values):
        print(f"Warning: Test data length ({len(test_values)}) doesn't match forecast length ({len(forecast_values)})")
        # Trim to the shorter length
        min_length = min(len(test_values), len(forecast_values))
        test_values = test_values[:min_length]
        forecast_values = forecast_values[:min_length]
    
    # Calculate metrics
    rmse = sqrt(mean_squared_error(test_values, forecast_values))
    # Safely calculate MAPE avoiding division by zero
    mape = np.mean(np.abs((test_values - forecast_values) / np.where(test_values != 0, test_values, 1))) * 100
    print(f"\nForecast Error Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot forecast vs actual
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title(f'ARIMA Forecast for {target_commodity.split("_")[0]}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_forecast.png')
    print("Saved ARIMA forecast as 'arima_forecast.png'")
    
except Exception as e:
    print(f"Error in ARIMA modeling: {str(e)}")
    print("Trying a simpler model configuration...")
    
    # Try a simpler model
    try:
        model = ARIMA(train, order=(1,1,0))  # Simpler model without MA term
        model_fit = model.fit()
        
        # Continue with forecasting using the simpler model
        forecast = model_fit.forecast(steps=len(test))
        
        # Add a note to the plot title
        simple_title = f'Simple ARIMA(1,1,0) Forecast for {target_commodity.split("_")[0]}'
        # ... rest of plotting code here
        
    except Exception as e2:
        print(f"Even simple model failed: {str(e2)}")
        print("Proceeding with analysis without ARIMA forecasting.")
        # Set a flag to skip forecasting steps that depend on this model
        forecast_failed = True

# Initialize forecast_failed flag
forecast_failed = False

# 5. Future Forecast
print("\nGenerating future forecast...")

# Only proceed if the previous forecasting didn't fail
if not forecast_failed:
    try:
        # Fit model on all data
        full_model = ARIMA(monthly_data, order=(1,1,1))
        full_model_fit = full_model.fit()
        
        # Forecast 12 months into the future
        future_steps = 12
        future_forecast = full_model_fit.forecast(steps=future_steps)
        
        # Create future date index
        last_date = monthly_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='M')
        
        # Plot historical data and future forecast
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_data.index, monthly_data, label='Historical Data')
        plt.plot(future_dates, future_forecast, label='Future Forecast', color='red')
        
        # Add confidence intervals (just an approximation)
        forecast_stderr = np.sqrt(full_model_fit.params['sigma2'])
        conf_int_lower = future_forecast - 1.96 * forecast_stderr
        conf_int_upper = future_forecast + 1.96 * forecast_stderr
        plt.fill_between(future_dates, conf_int_lower, conf_int_upper, color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(f'12-Month Future Forecast for {target_commodity.split("_")[0]}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('future_forecast.png')
        print("Saved future forecast as 'future_forecast.png'")
    except Exception as e:
        print(f"Error generating future forecast: {str(e)}")
        forecast_failed = True
else:
    print("Skipping future forecast generation due to previous errors.")

# 6. Rolling Window Analysis - this doesn't depend on forecasting
print("\nPerforming rolling window analysis...")
try:
    window_size = 30  # 30-day rolling window
    rolling_mean = commodity_data.rolling(window=window_size).mean()
    rolling_std = commodity_data.rolling(window=window_size).std()
    
    plt.figure(figsize=(15, 7))
    plt.plot(commodity_data, label='Daily Price', alpha=0.5)
    plt.plot(rolling_mean, label=f'{window_size}-Day Moving Average', linewidth=2)
    plt.plot(rolling_std, label=f'{window_size}-Day Standard Deviation', linewidth=2)
    plt.title(f'Rolling Window Analysis - {target_commodity.split("_")[0]}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rolling_window_analysis.png')
    print("Saved rolling window analysis as 'rolling_window_analysis.png'")
except Exception as e:
    print(f"Error in rolling window analysis: {str(e)}")

# 7. Technical Indicators - these don't depend on forecasting
print("\nCalculating technical indicators...")
try:
    # Calculate some common technical indicators
    # 1. RSI (Relative Strength Index)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    # 2. MACD (Moving Average Convergence Divergence)
    def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    # Calculate indicators
    commodity_data['RSI'] = calculate_rsi(commodity_data[target_commodity])
    macd_line, signal_line, macd_histogram = calculate_macd(commodity_data[target_commodity])
    commodity_data['MACD'] = macd_line
    commodity_data['MACD_Signal'] = signal_line
    commodity_data['MACD_Histogram'] = macd_histogram
    
    # Plot RSI
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(commodity_data[target_commodity], label='Price')
    plt.title(f'{target_commodity.split("_")[0]} Price', fontsize=14)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(commodity_data['RSI'], color='purple')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI (Relative Strength Index)', fontsize=14)
    plt.ylabel('RSI Value')
    
    plt.subplot(3, 1, 3)
    plt.plot(commodity_data['MACD'], label='MACD Line')
    plt.plot(commodity_data['MACD_Signal'], label='Signal Line')
    plt.bar(commodity_data.index, commodity_data['MACD_Histogram'], label='Histogram', alpha=0.3)
    plt.title('MACD (Moving Average Convergence Divergence)', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('technical_indicators.png')
    print("Saved technical indicators as 'technical_indicators.png'")
except Exception as e:
    print(f"Error calculating technical indicators: {str(e)}")

print("\nForecasting analysis complete! Check the generated visualizations for insights.")

# Show all figures
plt.show() 
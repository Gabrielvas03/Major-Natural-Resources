import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')

print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date')

# For gold forecasting
target_commodity = 'GC=F_closing_price'
print(f"\nPerforming Prophet forecasting on {target_commodity.split('_')[0]}")

# Extract gold data and handle missing values
gold_data = df[['Date', target_commodity]].copy()
gold_data = gold_data.dropna()

# Prophet requires columns named 'ds' and 'y'
gold_data.columns = ['ds', 'y']

# If we have enough data, focus on the last 10 years for better seasonal patterns
if len(gold_data) > 10*252:  # Approx 252 trading days per year
    start_date = gold_data['ds'].max() - pd.DateOffset(years=10)
    gold_data = gold_data[gold_data['ds'] >= start_date]

print(f"Analysis period: {gold_data['ds'].min().strftime('%d %b %Y')} to {gold_data['ds'].max().strftime('%d %b %Y')}")
print(f"Number of data points: {len(gold_data)}")

# 1. Split data into train/test sets
train_size = int(len(gold_data) * 0.8)
train = gold_data[:train_size].copy()
test = gold_data[train_size:].copy()
print(f"Training data: {train['ds'].min().strftime('%d %b %Y')} to {train['ds'].max().strftime('%d %b %Y')}")
print(f"Testing data: {test['ds'].min().strftime('%d %b %Y')} to {test['ds'].max().strftime('%d %b %Y')}")

# 2. Build and train the Prophet model
print("\nTraining Prophet model...")
prophet_model = Prophet(
    # Model configuration
    changepoint_prior_scale=0.05,  # Flexibility in trend changes (0.05 is default)
    seasonality_prior_scale=10,    # Strength of seasonality (10 for stronger seasonal patterns)
    seasonality_mode='multiplicative',  # Good for financial data, especially with upward trends
    yearly_seasonality=True,       # Enable yearly seasonality component
    weekly_seasonality=True,       # Enable weekly patterns
    daily_seasonality=False,       # Disable daily patterns (usually noisy in financial data)
    changepoint_range=0.9,         # Allow trend changes up to 90% of the time series
)

# Add quarterly seasonality
prophet_model.add_seasonality(
    name='quarterly',
    period=91.25,  # Average days in a quarter
    fourier_order=5  # Higher order for more flexibility
)

# Fit the model
prophet_model.fit(train)

# 3. Evaluate on test set
print("Evaluating model on test data...")
# The best approach is to create a dataframe with the exact test dates
test_dates = test['ds'].tolist()
future = pd.DataFrame({'ds': test_dates})

# Make predictions on exactly these test dates
test_forecast = prophet_model.predict(future)

# Verify sizes match
print(f"Test set size: {len(test)}, Test predictions size: {len(test_forecast)}")

# Ensure the order matches exactly
test_forecast = test_forecast.sort_values('ds')
test_sorted = test.sort_values('ds').reset_index(drop=True)

# Calculate error metrics
y_true = test_sorted['y'].values
y_pred = test_forecast['yhat'].values

print(f"Test data shape: {y_true.shape}, Predictions shape: {y_pred.shape}")
rmse = sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print(f"Test Set RMSE: {rmse:.2f}")
print(f"Test Set MAPE: {mape:.2f}%")

# 4. Plot test predictions vs actual
plt.figure(figsize=(15, 7))
plt.plot(test_sorted['ds'], test_sorted['y'], label='Actual Gold Price', color='blue')
plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Prophet Forecast', color='red')
plt.fill_between(
    test_forecast['ds'],
    test_forecast['yhat_lower'],
    test_forecast['yhat_upper'],
    color='red', alpha=0.2, label='95% Confidence Interval'
)
plt.title('Prophet Gold Price Forecast vs Actual (Test Set)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Gold Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prophet_test_forecast.png')
print("Saved test forecast comparison as 'prophet_test_forecast.png'")

# 5. Create future forecast
print("\nGenerating future forecast with Prophet...")

# Retrain on full dataset
full_prophet_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_range=0.9,
)

full_prophet_model.add_seasonality(
    name='quarterly',
    period=91.25,
    fourier_order=5
)

full_prophet_model.fit(gold_data)

# Create future dataframe for 24 months ahead (with daily frequency)
# Get the last date in our data
last_date = gold_data['ds'].max()
months_to_forecast = 24

# Create a properly spaced future dataframe
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),  # Start from the day after the last date
    periods=months_to_forecast * 30,         # Approx 30 days per month
    freq='D'                                 # Daily frequency
)
future = pd.DataFrame({'ds': future_dates})

# Make predictions
forecast = full_prophet_model.predict(future)

# 6. Plot full forecast with components
print("Creating visualization of forecast components...")
# Full forecast plot
fig = plot_plotly(full_prophet_model, forecast)
fig.write_html("prophet_full_forecast.html")

# Components plot (trend, seasonality)
fig_comp = plot_components_plotly(full_prophet_model, forecast)
fig_comp.write_html("prophet_components.html")

print("Saved interactive Prophet plots as HTML files")

# 7. Plot future forecast with matplotlib (for static image)
plt.figure(figsize=(15, 7))

# Get most recent date in data
last_date = gold_data['ds'].max()

# Filter to show only the past 3 years + forecast
three_years_ago = last_date - pd.DateOffset(years=3)
recent_data = gold_data[gold_data['ds'] >= three_years_ago]

# Plot actual data
plt.plot(recent_data['ds'], recent_data['y'], label='Historical Data', color='blue')

# Plot future forecast (forecast beyond the last actual date)
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(
    forecast['ds'],
    forecast['yhat_lower'],
    forecast['yhat_upper'],
    color='red', alpha=0.2, label='95% Confidence Interval'
)

plt.title('Prophet 24-Month Gold Price Forecast', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Gold Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prophet_future_forecast.png')
print("Saved future forecast as 'prophet_future_forecast.png'")

# 8. Export forecast data to CSV
future_forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
future_forecast_df.columns = ['Date', 'Forecasted Price', 'Lower Bound (95%)', 'Upper Bound (95%)']
future_forecast_df.set_index('Date', inplace=True)

# Round monthly dates (for easier reading)
monthly_forecast = future_forecast_df.resample('M').mean()

# Save to CSV
monthly_forecast.to_csv('prophet_gold_forecast.csv')

print("\nForecasted Gold Prices (monthly):")
pd.set_option('display.float_format', '${:.2f}'.format)
print(monthly_forecast.head(12))  # Show first year forecast
print("Saved monthly forecast to 'prophet_gold_forecast.csv'")

# 9. Analyze seasonal components
plt.figure(figsize=(15, 10))

# Create subplot for yearly seasonality
plt.subplot(2, 1, 1)
# Check if yearly seasonality column exists
if 'yearly' in forecast.columns:
    yearly_seasonality = forecast.groupby(forecast['ds'].dt.month)['yearly'].mean()
    plt.plot(yearly_seasonality.index, yearly_seasonality.values, marker='o', color='darkblue')
    plt.title('Yearly Seasonality Component in Gold Prices', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Effect on Price', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Yearly seasonality component not available', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)

# Create subplot for weekly seasonality
plt.subplot(2, 1, 2)
# Check if weekly seasonality column exists
if 'weekly' in forecast.columns:
    weekly_seasonality = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()
    plt.plot(weekly_seasonality.index, weekly_seasonality.values, marker='o', color='darkgreen')
    plt.title('Weekly Seasonality Component in Gold Prices', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Effect on Price', fontsize=12)
    plt.xticks(range(0, 7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Weekly seasonality component not available', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)

plt.tight_layout()
plt.savefig('prophet_seasonality_components.png')
print("Saved seasonality components as 'prophet_seasonality_components.png'")

print("\nProphet forecasting analysis complete! Check the generated visualizations and CSV files for detailed forecasts.")

# Optional: Show all figures if running in interactive mode
plt.show() 
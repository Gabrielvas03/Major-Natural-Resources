import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date')

# For gold forecasting, we'll focus on GC=F as it typically has good data quality
target_commodity = 'GC=F_closing_price'
print(f"\nPerforming advanced time series analysis on {target_commodity.split('_')[0]}")

# Extract gold data and handle missing values
gold_data = df[['Date', target_commodity]].copy()
gold_data = gold_data.dropna()
gold_data.set_index('Date', inplace=True)

# If we have enough data, focus on the last 10 years for better seasonal patterns
if len(gold_data) > 10*252:  # Approx 252 trading days per year
    start_date = gold_data.index[-10*252]
    gold_data = gold_data[gold_data.index >= start_date]

print(f"Analysis period: {gold_data.index.min().strftime('%d %b %Y')} to {gold_data.index.max().strftime('%d %b %Y')}")
print(f"Number of data points: {len(gold_data)}")

# 1. Resample to monthly data for clearer seasonality patterns
monthly_data = gold_data.resample('M').mean()
print(f"Using {len(monthly_data)} months of data for analysis")

# 2. Seasonal Decomposition for visualization
print("\nDecomposing gold price time series...")
decomposition = seasonal_decompose(monthly_data, model='additive', period=12)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed Gold Price', fontsize=14)
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend Component', fontsize=14)
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal Component', fontsize=14)
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual Component', fontsize=14)
plt.tight_layout()
plt.savefig('gold_decomposition.png')
print("Saved gold price decomposition as 'gold_decomposition.png'")

# 3. Split data into train/test sets
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data[0:train_size], monthly_data[train_size:]
print(f"Training data: {train.index.min().strftime('%d %b %Y')} to {train.index.max().strftime('%d %b %Y')}")
print(f"Testing data: {test.index.min().strftime('%d %b %Y')} to {test.index.max().strftime('%d %b %Y')}")

# 4. Build and evaluate several models
models = [
    # Standard ARIMA model
    ('ARIMA(1,1,1)', SARIMAX(train, order=(1,1,1), seasonal_order=(0,0,0,0))),
    
    # SARIMA models with different seasonal components
    ('SARIMA(1,1,1)x(1,0,1,12)', SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,12))),
    ('SARIMA(1,1,1)x(0,1,1,12)', SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,12))),
    ('SARIMA(2,1,2)x(1,1,1,12)', SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,12)))
]

# Dictionary to store results
model_results = {}

print("\nFitting and evaluating forecasting models...")
for name, model in models:
    try:
        # Fit the model with controlled iterations
        print(f"Fitting {name}...")
        fitted_model = model.fit(disp=False, maxiter=50)
        
        # Check if the model converged
        if not fitted_model.mle_retvals.get('converged', True):
            print(f"  Warning: {name} did not converge. Results may be unreliable.")
        
        # Forecast on test data
        forecast = fitted_model.forecast(steps=len(test))
        
        # Ensure forecast values are not NaN
        if np.isnan(forecast).any():
            print(f"  Error: {name} produced NaN forecasts")
            continue
            
        # Calculate error metrics safely
        test_values = test.values.flatten()
        try:
            rmse = sqrt(mean_squared_error(test_values, forecast))
            mape = mean_absolute_percentage_error(test_values, forecast) * 100
            
            # Store results
            model_results[name] = {
                'model': fitted_model,
                'forecast': forecast,
                'rmse': rmse,
                'mape': mape
            }
            
            print(f"  RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"  Error calculating metrics for {name}: {str(e)}")
            
    except Exception as e:
        print(f"  Error fitting {name}: {str(e)}")

# If no models were successfully fit, try a simpler model
if not model_results:
    print("\nNo models were successfully fit. Trying a simpler model...")
    try:
        simple_model = SARIMAX(train, order=(1,1,0), seasonal_order=(0,0,0,0))
        simple_fitted = simple_model.fit(disp=False)
        simple_forecast = simple_fitted.forecast(steps=len(test))
        
        rmse = sqrt(mean_squared_error(test.values.flatten(), simple_forecast))
        mape = mean_absolute_percentage_error(test.values.flatten(), simple_forecast) * 100
        
        model_results["ARIMA(1,1,0)"] = {
            'model': simple_fitted,
            'forecast': simple_forecast,
            'rmse': rmse,
            'mape': mape
        }
        print(f"  Simple ARIMA(1,1,0) - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    except Exception as e:
        print(f"  Even simple model failed: {str(e)}")

# 5. Identify best model
if model_results:
    # Sort models by RMSE (lower is better)
    best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['rmse'])
    best_model = model_results[best_model_name]
    print(f"\nBest model: {best_model_name} with RMSE: {best_model['rmse']:.2f}, MAPE: {best_model['mape']:.2f}%")
    
    # Plot best model's forecast vs actual
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(test.index, best_model['forecast'], label=f'{best_model_name} Forecast', color='red')
    plt.title(f'Gold Price Forecast Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Gold Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gold_forecast_comparison.png')
    print("Saved forecast comparison as 'gold_forecast_comparison.png'")
    
    # 6. Use best model to forecast future prices
    print("\nForecasting future gold prices...")
    
    # Refit model on all available data
    if 'SARIMA' in best_model_name:
        # Extract order and seasonal_order from the model name
        order_parts = best_model_name.split('(')[1].split(')')[0].split(',')
        seasonal_parts = best_model_name.split('x(')[1].split(')')[0].split(',')
        
        p, d, q = map(int, order_parts)
        P, D, Q, s = map(int, seasonal_parts)
        
        full_model = SARIMAX(monthly_data, 
                            order=(p, d, q), 
                            seasonal_order=(P, D, Q, s))
    else:
        # Default to ARIMA(1,1,1) if parsing fails
        full_model = SARIMAX(monthly_data, order=(1,1,1), seasonal_order=(0,0,0,0))
    
    full_model_fit = full_model.fit(disp=False)
    
    # Forecast next 24 months
    future_steps = 24
    future_forecast = full_model_fit.forecast(steps=future_steps)
    
    # Create future date index
    last_date = monthly_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                periods=future_steps, freq='M')
    
    # Get prediction intervals
    pred_intervals = full_model_fit.get_forecast(steps=future_steps).conf_int(alpha=0.05)
    lower_bound = pred_intervals.iloc[:,0]
    upper_bound = pred_intervals.iloc[:,1]
    
    # Plot historical data and future forecast with confidence intervals
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_data.index, monthly_data, label='Historical Data')
    plt.plot(future_dates, future_forecast, label='Future Forecast', color='red')
    plt.fill_between(future_dates, lower_bound, upper_bound, 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f'24-Month Gold Price Forecast ({best_model_name})', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Gold Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gold_future_forecast.png')
    print("Saved future forecast as 'gold_future_forecast.png'")
    
    # Create a table of forecasted values
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': future_forecast,
        'Lower Bound (95%)': lower_bound,
        'Upper Bound (95%)': upper_bound
    })
    forecast_df.set_index('Date', inplace=True)
    
    print("\nForecasted Gold Prices (monthly):")
    pd.set_option('display.float_format', '${:.2f}'.format)
    print(forecast_df.head(12))  # Show first year
    
    # Save forecast to CSV
    forecast_df.to_csv('gold_price_forecast.csv')
    print("Saved complete forecast to 'gold_price_forecast.csv'")
    
    # 7. Create visual summary of model performance
    plt.figure(figsize=(10, 6))
    model_names = list(model_results.keys())
    rmse_values = [model_results[model]['rmse'] for model in model_names]
    
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    bars[model_names.index(best_model_name)].set_color('green')
    
    plt.title('Model Performance Comparison (RMSE)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('RMSE (lower is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Saved model comparison as 'model_comparison.png'")
    
    # 8. Analyze forecast seasonality
    if len(future_forecast) >= 12:
        forecast_series = pd.Series(future_forecast, index=future_dates)
        forecast_monthly_avg = forecast_series.groupby(forecast_series.index.month).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_monthly_avg.index, forecast_monthly_avg.values, marker='o', color='purple')
        plt.title('Forecasted Seasonal Pattern in Gold Prices', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Forecasted Price', fontsize=14)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('forecast_seasonality.png')
        print("Saved seasonal forecast pattern as 'forecast_seasonality.png'")

else:
    print("No successful models to generate forecasts")

print("\nAdvanced forecasting analysis complete! Check the generated visualizations and CSV file for detailed forecasts.")

plt.show() 
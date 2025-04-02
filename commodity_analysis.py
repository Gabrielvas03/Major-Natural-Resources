import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

# Set style for the plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# 1. Market Trend Analysis
# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

# Sort the data chronologically
df = df.sort_values('Date')

# Print dataset info
print(f"Dataset spans from {df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}")
print(f"Number of records: {len(df)}")
print(f"Available commodities: {', '.join([col.split('_')[0] for col in df.columns if '_closing_price' in col])}")

# Select major commodities for analysis
# CL=F: Crude Oil
# GC=F: Gold
# SI=F: Silver
# NG=F: Natural Gas
commodities = ['CL=F', 'GC=F', 'SI=F', 'NG=F']
columns = [f"{commodity}_closing_price" for commodity in commodities]

# Filter for only the most recent 5 years of data to make the chart more readable
five_years_ago = df['Date'].max() - pd.DateOffset(years=5)
recent_data = df[df['Date'] >= five_years_ago].copy()

# Plot historical closing prices of major commodities
plt.figure(figsize=(14, 8))

for commodity, column in zip(commodities, columns):
    plt.plot(recent_data['Date'], recent_data[column], label=commodity)

plt.title('Historical Closing Prices of Major Commodities (Last 5 Years)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('commodity_prices.png')
print("Saved historical prices chart as 'commodity_prices.png'")

# Optional: Plot with log scale for better visibility of relative changes
plt.figure(figsize=(14, 8))

for commodity, column in zip(commodities, columns):
    plt.semilogy(recent_data['Date'], recent_data[column], label=commodity)

plt.title('Historical Closing Prices of Major Commodities (Log Scale)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price (USD, Log Scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('commodity_prices_log_scale.png')
print("Saved log-scale historical prices chart as 'commodity_prices_log_scale.png'")

# 2. Daily Returns Calculation
print("\nCalculating daily returns...")

# Calculate daily percentage change (returns) for each commodity
for commodity in commodities:
    column = f"{commodity}_closing_price"
    return_col = f"{commodity}_daily_return"
    recent_data[return_col] = recent_data[column].pct_change() * 100

# Drop NaN values (first row has no return)
recent_data_returns = recent_data.dropna(subset=[f"{commodities[0]}_daily_return"])

# Plot daily returns for two commodities (Crude Oil and Gold)
plt.figure(figsize=(14, 8))

plt.plot(recent_data_returns['Date'], recent_data_returns[f"{commodities[0]}_daily_return"], 
         label=f"{commodities[0]} Daily Returns", alpha=0.7)
plt.plot(recent_data_returns['Date'], recent_data_returns[f"{commodities[1]}_daily_return"], 
         label=f"{commodities[1]} Daily Returns", alpha=0.7)

plt.title(f'Daily Returns: {commodities[0]} vs {commodities[1]}', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Return (%)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

# Add a horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('daily_returns.png')
print("Saved daily returns chart as 'daily_returns.png'")

# Create correlation heatmap of daily returns
plt.figure(figsize=(10, 8))

# Extract just the return columns for correlation analysis
return_columns = [f"{commodity}_daily_return" for commodity in commodities]
correlation_matrix = recent_data_returns[return_columns].corr()

# Rename columns and index for better readability
correlation_matrix.columns = commodities
correlation_matrix.index = commodities

# Generate heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation of Daily Returns Between Commodities', fontsize=16)
plt.tight_layout()
plt.savefig('return_correlation_heatmap.png')
print("Saved correlation heatmap as 'return_correlation_heatmap.png'")

# Calculate summary statistics for daily returns
return_stats = recent_data_returns[return_columns].describe().T
return_stats.index = commodities
print("\nSummary Statistics for Daily Returns:")
print(return_stats)

# Show all figures
plt.show() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno

# Set style for the plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load and prepare the dataset
print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

# Sort the data chronologically
df = df.sort_values('Date')

# 1. Basic Dataset Information
print("\n==== DATASET OVERVIEW ====")
print(f"Dataset spans from {df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}")
print(f"Total time period: {(df['Date'].max() - df['Date'].min()).days} days")
print(f"Number of records: {len(df)}")

# List all commodities
commodity_columns = [col for col in df.columns if '_closing_price' in col]
commodities = [col.split('_')[0] for col in commodity_columns]
print(f"\nAvailable commodities: {', '.join(commodities)}")

# 2. Data Completeness Analysis
print("\n==== DATA COMPLETENESS ====")
# Calculate the percentage of non-null values for each commodity
completeness = df[commodity_columns].notna().mean() * 100
completeness = completeness.sort_values(ascending=False)

print("Data completeness by commodity (% of days with values):")
for col, pct in completeness.items():
    print(f"  {col.split('_')[0]}: {pct:.2f}%")

# Visualize missing data patterns
plt.figure(figsize=(12, 8))
msno.matrix(df[['Date'] + commodity_columns], figsize=(12, 8), color=(0.27, 0.52, 0.71))
plt.title('Missing Data Visualization', fontsize=16)
plt.tight_layout()
plt.savefig('missing_data_pattern.png')
print("\nSaved missing data visualization as 'missing_data_pattern.png'")

# 3. Time Series Completeness
# When did each commodity start having consistent data?
first_appearance = {}
for col in commodity_columns:
    # Find the first non-null value
    first_date = df[df[col].notna()]['Date'].min()
    # Find the date when values become consistent (less than 5% missing in the future)
    consistent_data = df[df['Date'] >= first_date].copy()
    rolling_completeness = consistent_data[col].notna().rolling(window=30).mean()
    try:
        consistent_start = consistent_data.loc[rolling_completeness >= 0.95, 'Date'].min()
        first_appearance[col.split('_')[0]] = {
            'first_value': first_date,
            'consistent_data': consistent_start
        }
    except:
        first_appearance[col.split('_')[0]] = {
            'first_value': first_date,
            'consistent_data': "Never consistently present"
        }

print("\nData availability timeline:")
for commodity, dates in first_appearance.items():
    first = dates['first_value'].strftime('%d %b %Y') if isinstance(dates['first_value'], pd.Timestamp) else "N/A"
    consistent = dates['consistent_data'].strftime('%d %b %Y') if isinstance(dates['consistent_data'], pd.Timestamp) else dates['consistent_data']
    print(f"  {commodity}: First appearance: {first}, Consistent data from: {consistent}")

# 4. Statistical Summary of Closing Prices
print("\n==== STATISTICAL SUMMARY ====")
# Calculate basic statistics for each commodity
stats = df[commodity_columns].describe().T
stats['missing_pct'] = (1 - df[commodity_columns].count() / len(df)) * 100
stats.index = [col.split('_')[0] for col in stats.index]
print("\nStatistical summary of closing prices:")
print(stats)

# 5. Volatility Analysis
print("\n==== VOLATILITY ANALYSIS ====")
# Calculate the coefficient of variation for each commodity
# Use the last 5 years of data to focus on more recent trends
five_years_ago = df['Date'].max() - pd.DateOffset(years=5)
recent_data = df[df['Date'] >= five_years_ago].copy()

volatility = {}
for col in commodity_columns:
    commodity = col.split('_')[0]
    # Calculate daily returns
    returns = recent_data[col].pct_change().dropna()
    
    if not returns.empty:
        # Coefficient of variation (standardized measure of dispersion)
        cv = returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf')
        # Average absolute daily change
        avg_abs_change = returns.abs().mean()
        # Max drawdown (maximum % loss from a peak)
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        volatility[commodity] = {
            'cv': cv,
            'avg_daily_change': avg_abs_change * 100,  # as percentage
            'max_drawdown': max_drawdown * 100  # as percentage
        }

# Sort commodities by volatility
sorted_volatility = sorted(volatility.items(), key=lambda x: x[1]['cv'], reverse=True)

print("Volatility measures (last 5 years, most volatile first):")
for commodity, metrics in sorted_volatility:
    print(f"  {commodity}: CV: {metrics['cv']:.2f}, Avg Daily Change: {metrics['avg_daily_change']:.2f}%, "
          f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

# 6. Seasonality Analysis
print("\n==== SEASONALITY ANALYSIS ====")
# Extract month from the date
df['Month'] = df['Date'].dt.month

# For each commodity, calculate average price by month
seasonal_patterns = {}
for col in commodity_columns:
    commodity = col.split('_')[0]
    
    # Skip commodities with too many missing values
    if df[col].isna().mean() > 0.5:
        continue
    
    # Group by month and calculate mean price
    monthly_avg = df.groupby('Month')[col].mean()
    if not monthly_avg.empty and not monthly_avg.isna().all():
        # Normalize to show relative seasonal changes
        normalized = monthly_avg / monthly_avg.mean()
        seasonal_patterns[commodity] = normalized

# Visualize seasonal patterns
if seasonal_patterns:
    plt.figure(figsize=(14, 8))
    for commodity, pattern in seasonal_patterns.items():
        plt.plot(pattern.index, pattern.values, marker='o', label=commodity)
    
    plt.title('Seasonal Price Patterns by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Relative Price (Normalized)', fontsize=14)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('seasonal_patterns.png')
    print("Saved seasonal analysis as 'seasonal_patterns.png'")

# 7. Correlation Over Time Analysis
print("\n==== CORRELATION OVER TIME ====")
# Analyze how correlations have changed over time
# Use 1-year rolling windows

# Function to calculate correlation matrix for a time period
def get_correlation_for_period(start_date, end_date):
    period_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return period_data[commodity_columns].corr()

# Calculate correlations for different time periods
time_periods = [
    ("Last Year", df['Date'].max() - pd.DateOffset(years=1), df['Date'].max()),
    ("Last 5 Years", df['Date'].max() - pd.DateOffset(years=5), df['Date'].max()),
    ("Last 10 Years", df['Date'].max() - pd.DateOffset(years=10), df['Date'].max()),
    ("All Time", df['Date'].min(), df['Date'].max())
]

print("Correlation analysis across different time periods:")
for period_name, start_date, end_date in time_periods:
    print(f"\n{period_name} correlation matrix:")
    try:
        corr_matrix = get_correlation_for_period(start_date, end_date)
        # Only show the upper triangle of the correlation matrix to avoid redundancy
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    mask=mask, vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix: {period_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'correlation_{period_name.replace(" ", "_").lower()}.png')
        print(f"Saved correlation matrix as 'correlation_{period_name.replace(' ', '_').lower()}.png'")
    except Exception as e:
        print(f"Could not calculate correlation for {period_name}: {str(e)}")

print("\nEDA complete! Check the generated visualizations for deeper insights.")

# Show all figures
plt.show() 
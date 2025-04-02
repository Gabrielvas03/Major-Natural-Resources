import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

print("Loading the dataset...")
df = pd.read_csv('Futures_Resources_Data.csv')

# Parse the Date column
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.sort_values('Date')

# Select major commodities for portfolio analysis
# Choose commodities with good data coverage
commodities = ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'PL=F']  # Oil, Gold, Silver, Copper, Platinum
print(f"Selected commodities for portfolio analysis: {', '.join(commodities)}")

# Focus on last 5 years of data for more relevant results
five_years_ago = df['Date'].max() - pd.DateOffset(years=5)
recent_data = df[df['Date'] >= five_years_ago].copy()

# Extract price data for selected commodities
price_columns = [f"{commodity}_closing_price" for commodity in commodities]
prices = recent_data[['Date'] + price_columns].copy()

# Check for missing values and handle them
missing_pct = prices[price_columns].isna().mean() * 100
print("\nMissing data percentage for each commodity:")
for col, pct in zip(commodities, missing_pct):
    print(f"  {col}: {pct:.2f}%")

# Forward fill missing values (use previous day's price)
prices.set_index('Date', inplace=True)
prices = prices.fillna(method='ffill')

# Calculate daily returns
daily_returns = prices.pct_change().dropna()

# 1. Return and Risk Analysis
print("\n==== RETURN AND RISK ANALYSIS ====")
# Calculate annualized returns and volatility
# Assuming 252 trading days in a year
annualized_returns = daily_returns.mean() * 252 * 100  # convert to percentage
annualized_volatility = daily_returns.std() * np.sqrt(252) * 100  # convert to percentage

# Create risk-return summary
risk_return = pd.DataFrame({
    'Annualized Return (%)': annualized_returns,
    'Annualized Volatility (%)': annualized_volatility,
    'Sharpe Ratio': annualized_returns / annualized_volatility  # Assuming risk-free rate of 0 for simplicity
})
risk_return.index = commodities

print(risk_return)

# Plot risk-return chart
plt.figure(figsize=(10, 8))
plt.scatter(risk_return['Annualized Volatility (%)'], risk_return['Annualized Return (%)'], s=100)

# Label each point with the commodity name
for i, commodity in enumerate(commodities):
    plt.annotate(commodity, 
                 (risk_return['Annualized Volatility (%)'][i], risk_return['Annualized Return (%)'][i]),
                 textcoords="offset points", 
                 xytext=(10, 5), 
                 ha='center',
                 fontsize=12)

plt.title('Risk-Return Profile of Commodities', fontsize=16)
plt.xlabel('Annualized Volatility (%)', fontsize=14)
plt.ylabel('Annualized Return (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('risk_return_profile.png')
print("Saved risk-return profile as 'risk_return_profile.png'")

# 2. Portfolio Optimization - Efficient Frontier
print("\n==== PORTFOLIO OPTIMIZATION ====")

# Function to calculate portfolio returns and volatility
def portfolio_performance(weights, returns):
    # Returns in decimal
    returns_array = returns.mean() * 252
    # Volatility in decimal
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    # Return annualized returns and volatility in percentage
    return returns_array @ weights * 100, volatility * 100

# Function to minimize negative Sharpe Ratio
def negative_sharpe_ratio(weights, returns):
    p_returns, p_volatility = portfolio_performance(weights, returns)
    return -p_returns / p_volatility

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Bounds: weights are between 0 and 1 (no short selling)
bounds = tuple((0, 1) for _ in range(len(commodities)))
# Initial guess: equal weights
initial_weights = np.array([1/len(commodities)] * len(commodities))

# Find the optimal portfolio (maximum Sharpe ratio)
print("Calculating optimal portfolio allocation...")
optimal_result = minimize(negative_sharpe_ratio, initial_weights, args=(daily_returns,), 
                         method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimal_result['x']
optimal_returns, optimal_volatility = portfolio_performance(optimal_weights, daily_returns)
optimal_sharpe = optimal_returns / optimal_volatility

print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
for commodity, weight in zip(commodities, optimal_weights):
    print(f"  {commodity}: {weight*100:.2f}%")
print(f"Expected Annual Return: {optimal_returns:.2f}%")
print(f"Expected Annual Volatility: {optimal_volatility:.2f}%")
print(f"Sharpe Ratio: {optimal_sharpe:.3f}")

# Generate random portfolios for the efficient frontier
print("\nGenerating efficient frontier...")
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
all_weights = np.zeros((num_portfolios, len(commodities)))

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(commodities))
    weights = weights / np.sum(weights)
    all_weights[i,:] = weights
    
    # Calculate portfolio returns and volatility
    portfolio_return, portfolio_volatility = portfolio_performance(weights, daily_returns)
    
    # Store results
    results[0,i] = portfolio_volatility  # Volatility
    results[1,i] = portfolio_return      # Return
    results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

# Find the minimum volatility portfolio
min_vol_idx = np.argmin(results[0])
min_vol_weights = all_weights[min_vol_idx,:]
min_vol_return, min_vol_volatility = portfolio_performance(min_vol_weights, daily_returns)
min_vol_sharpe = min_vol_return / min_vol_volatility

print("\nMinimum Volatility Portfolio:")
for commodity, weight in zip(commodities, min_vol_weights):
    print(f"  {commodity}: {weight*100:.2f}%")
print(f"Expected Annual Return: {min_vol_return:.2f}%")
print(f"Expected Annual Volatility: {min_vol_volatility:.2f}%")
print(f"Sharpe Ratio: {min_vol_sharpe:.3f}")

# Find the maximum return portfolio
max_return_idx = np.argmax(results[1])
max_return_weights = all_weights[max_return_idx,:]
max_return, max_return_volatility = portfolio_performance(max_return_weights, daily_returns)
max_return_sharpe = max_return / max_return_volatility

print("\nMaximum Return Portfolio:")
for commodity, weight in zip(commodities, max_return_weights):
    print(f"  {commodity}: {weight*100:.2f}%")
print(f"Expected Annual Return: {max_return:.2f}%")
print(f"Expected Annual Volatility: {max_return_volatility:.2f}%")
print(f"Sharpe Ratio: {max_return_sharpe:.3f}")

# Plot the efficient frontier
plt.figure(figsize=(12, 8))

# Plot random portfolios
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', 
            alpha=0.3, s=10, label='Potential Portfolios')

# Plot individual commodities
for i, commodity in enumerate(commodities):
    single_asset_volatility = daily_returns.iloc[:, i].std() * np.sqrt(252) * 100
    single_asset_return = daily_returns.iloc[:, i].mean() * 252 * 100
    plt.scatter(single_asset_volatility, single_asset_return, s=100, label=commodity)
    plt.annotate(commodity, (single_asset_volatility, single_asset_return),
                xytext=(10, 5), textcoords='offset points')

# Plot optimal portfolio
plt.scatter(optimal_volatility, optimal_returns, s=200, c='red', marker='*', label='Maximum Sharpe Ratio Portfolio')
# Plot min volatility portfolio
plt.scatter(min_vol_volatility, min_vol_return, s=150, c='gold', marker='D', label='Minimum Volatility Portfolio')
# Plot max return portfolio
plt.scatter(max_return_volatility, max_return, s=150, c='green', marker='X', label='Maximum Return Portfolio')

plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Expected Volatility (%)', fontsize=14)
plt.ylabel('Expected Return (%)', fontsize=14)
plt.title('Efficient Frontier with Commodity Portfolios', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('efficient_frontier.png')
print("Saved efficient frontier as 'efficient_frontier.png'")

# 3. Optimal Portfolio Weightings Visualization
print("\nVisualizing portfolio weightings...")
# Create a DataFrame with the weights of each portfolio type
portfolio_weights = pd.DataFrame({
    'Max Sharpe': optimal_weights,
    'Min Volatility': min_vol_weights,
    'Max Return': max_return_weights
}, index=commodities)

# Plot the weights
plt.figure(figsize=(14, 8))
portfolio_weights.plot(kind='bar', figsize=(14, 8))
plt.title('Portfolio Allocation by Strategy', fontsize=16)
plt.xlabel('Commodity', fontsize=14)
plt.ylabel('Weight in Portfolio', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('portfolio_allocations.png')
print("Saved portfolio allocations as 'portfolio_allocations.png'")

# 4. Correlation Heatmap for Portfolio Components
plt.figure(figsize=(10, 8))
correlation_matrix = daily_returns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Commodity Returns', fontsize=16)
plt.tight_layout()
plt.savefig('portfolio_correlation.png')
print("Saved correlation matrix as 'portfolio_correlation.png'")

# 5. Portfolio Rebalancing Analysis
print("\n==== PORTFOLIO REBALANCING ANALYSIS ====")
# Simulate portfolio performance with monthly rebalancing vs buy-and-hold
# Divide the data into quarters for analysis
days_per_month = 21  # approx trading days per month
num_months = len(daily_returns) // days_per_month

if num_months > 0:
    # Initialize portfolio values
    initial_investment = 10000  # $10,000 investment
    
    # Buy-and-hold strategy: allocate once using optimal weights and never rebalance
    buy_hold_portfolio = pd.DataFrame(index=daily_returns.index)
    buy_hold_portfolio['value'] = initial_investment
    
    # Monthly rebalancing strategy: reallocate to optimal weights each month
    rebalanced_portfolio = pd.DataFrame(index=daily_returns.index)
    rebalanced_portfolio['value'] = initial_investment
    
    # Simulate portfolio performance
    for i in range(num_months):
        start_idx = i * days_per_month
        end_idx = min((i + 1) * days_per_month, len(daily_returns))
        
        month_returns = daily_returns.iloc[start_idx:end_idx]
        
        if i == 0:
            # Initial allocation for both portfolios is the same
            buy_hold_cumulative_return = (month_returns @ optimal_weights + 1).cumprod()
            rebalance_cumulative_return = buy_hold_cumulative_return
        else:
            # Buy and hold: continue with existing allocations
            month_buy_hold = (month_returns @ current_weights + 1).cumprod()
            buy_hold_cumulative_return = buy_hold_cumulative_return.iloc[-1] * month_buy_hold
            
            # Rebalanced: reset to optimal weights at the start of each month
            month_rebalance = (month_returns @ optimal_weights + 1).cumprod()
            rebalance_cumulative_return = rebalance_cumulative_return.iloc[-1] * month_rebalance
            
        # Update buy and hold weights based on asset performance
        current_weights = optimal_weights * (1 + month_returns.mean())
        current_weights = current_weights / np.sum(current_weights)
        
        # Store portfolio values
        buy_hold_portfolio.iloc[start_idx:end_idx, 0] = initial_investment * buy_hold_cumulative_return
        rebalanced_portfolio.iloc[start_idx:end_idx, 0] = initial_investment * rebalance_cumulative_return
    
    # Plot the portfolio values over time
    plt.figure(figsize=(14, 8))
    plt.plot(buy_hold_portfolio.index, buy_hold_portfolio['value'], label='Buy and Hold')
    plt.plot(rebalanced_portfolio.index, rebalanced_portfolio['value'], label='Monthly Rebalancing')
    plt.title('Portfolio Value: Rebalancing vs Buy-and-Hold Strategy', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rebalancing_analysis.png')
    print("Saved rebalancing analysis as 'rebalancing_analysis.png'")
    
    # Calculate performance metrics
    buy_hold_return = (buy_hold_portfolio['value'].iloc[-1] / initial_investment - 1) * 100
    rebalance_return = (rebalanced_portfolio['value'].iloc[-1] / initial_investment - 1) * 100
    
    print(f"\nBuy and Hold Strategy Return: {buy_hold_return:.2f}%")
    print(f"Monthly Rebalancing Strategy Return: {rebalance_return:.2f}%")
    print(f"Difference: {rebalance_return - buy_hold_return:.2f}%")

print("\nPortfolio optimization analysis complete! Check the generated visualizations for insights.")

# Show all figures
plt.show() 
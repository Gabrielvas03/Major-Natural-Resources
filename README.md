# Commodity Market Analysis and Portfolio Optimization

This project provides comprehensive tools for analyzing historical commodity prices, understanding market trends, optimizing portfolios, and forecasting future movements. The analysis spans from basic data visualization to advanced techniques like time series decomposition and portfolio optimization.

## Features

### Basic Market Analysis (`commodity_analysis.py`)
- **Market Trend Analysis**: Visualizes historical closing prices of major commodities
- **Daily Returns Calculation**: Calculates and visualizes daily percentage changes
- **Correlation Analysis**: Generates a heatmap showing correlations between commodity returns

### Exploratory Data Analysis (`commodity_eda.py`)
- **Data Completeness Analysis**: Identifies missing data patterns and visualizes them
- **Time Series Completeness**: Determines when each commodity starts having consistent data
- **Volatility Analysis**: Calculates risk metrics like coefficient of variation and max drawdown
- **Seasonality Analysis**: Detects monthly patterns in commodity prices
- **Correlation Over Time**: Analyzes how correlations between commodities change across different time periods

### Time Series Analysis and Forecasting (`commodity_forecasting.py`)
- **Trend-Seasonality Decomposition**: Breaks down price movements into trend, seasonal, and residual components
- **Autocorrelation Analysis**: Identifies cyclic patterns and dependencies in price changes
- **ARIMA Modeling**: Builds forecasting models with performance evaluation
- **Technical Indicators**: Calculates trading indicators like RSI and MACD
- **Rolling Window Analysis**: Performs moving average analysis to identify trends

### Portfolio Optimization (`portfolio_optimization.py`)
- **Risk-Return Analysis**: Calculates and visualizes risk-return metrics for each commodity
- **Efficient Frontier**: Identifies optimal portfolios with different risk-return profiles
- **Portfolio Allocation Strategies**: Compares maximum Sharpe ratio, minimum volatility, and maximum return portfolios
- **Rebalancing Analysis**: Simulates portfolio performance with different rebalancing strategies

## Generated Visualizations

The scripts generate various visualizations to help understand the data:

### Basic Analysis
- Historical price charts (regular and log scale)
- Daily returns comparison
- Correlation heatmap

### Exploratory Data Analysis
- Missing data patterns visualization
- Seasonal price patterns
- Correlation matrices for different time periods

### Time Series Analysis
- Time series decomposition plots
- Autocorrelation and partial autocorrelation functions
- ARIMA forecast evaluation
- Future price projections with confidence intervals
- Technical indicator charts

### Portfolio Optimization
- Risk-return scatter plot
- Efficient frontier with optimal portfolios
- Portfolio allocation comparison
- Portfolio performance with different rebalancing strategies

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Analysis

Each script can be run independently to focus on different aspects of the analysis:

```bash
# Basic market trend analysis
python commodity_analysis.py

# Exploratory data analysis
python commodity_eda.py

# Time series analysis and forecasting
python commodity_forecasting.py

# Portfolio optimization
python portfolio_optimization.py
```

## Data Format

The scripts expect a CSV file with columns in the following format:
- Date (format: dd/mm/yy)
- Commodity columns named as: `{SYMBOL}_closing_price`
   - e.g., `CL=F_closing_price`, `GC=F_closing_price`, etc.

## Commodity Symbols

- `CL=F`: Crude Oil (WTI)
- `BZ=F`: Brent Crude Oil
- `GC=F`: Gold
- `SI=F`: Silver
- `NG=F`: Natural Gas
- `ZC=F`: Corn
- `ZW=F`: Wheat
- `ZS=F`: Soybeans
- `HG=F`: Copper
- `PL=F`: Platinum
- `PA=F`: Palladium

## Use Cases

This toolkit can be used for:

1. **Investment Research**: Identify optimal commodity allocations for investment portfolios
2. **Risk Management**: Understand volatility and correlations between different commodities
3. **Market Forecasting**: Predict future price movements based on historical patterns
4. **Seasonal Trading**: Identify seasonal patterns that could inform trading strategies
5. **Academic Analysis**: Study the relationships between different natural resources 
#!/usr/bin/env python
# coding: utf-8

# # Unit 5 - Financial Planning
# 

# In[ ]:


# Initial imports
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load .env enviroment variables
load_dotenv()


# ## Part 1 - Personal Finance Planner

# ### Collect Crypto Prices Using the `requests` Library

# In[ ]:


# Set current amount of crypto assets
my_btc = 1.2
my_eth = 5.3
# YOUR CODE HERE!


# In[ ]:


# Crypto API URLs
btc_url = "https://api.alternative.me/v2/ticker/Bitcoin/?convert=CAD"
eth_url = "https://api.alternative.me/v2/ticker/Ethereum/?convert=CAD"
btc_data = requests.get(btc_url).json()
eth_data = requests.get(eth_url).json()


# In[ ]:


# Fetch current BTC price
# YOUR CODE HERE!
btc_price = btc_data['data']['1']['quotes']['USD']['price']


# In[ ]:


# Fetch current ETH price
eth_price = eth_data['data']['1027']['quotes']['USD']['price']


# In[ ]:


# Compute current value of my crpto
btc = btc_price * my_btc
eth = eth_price * my_eth
crypto = btc + eth


# In[ ]:


# Print current crypto wallet balance
print(f"The current value of your {my_btc} BTC is ${btc:0.2f}")
print(f"The current value of your {my_eth} ETH is ${eth:0.2f}")


# ### Collect Investments Data Using Alpaca: `SPY` (stocks) and `AGG` (bonds)

# In[ ]:


# Current amount of shares
my_spy = 50
my_agg = 200


# In[ ]:


# Set Alpaca API key and secret
alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
type(alpaca_secret_key)


# In[ ]:


# Create the Alpaca API object
alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    api_version="v2")


# In[ ]:


# Format current date as ISO format
date = pd.Timestamp("2021-05-28", tz="America/Chicago").isoformat()


# In[ ]:


# Set the tickers
tickers = ["AGG", "SPY"]

# Set timeframe to '1D' for Alpaca API
timeframe = "1D"

# Get current closing prices for SPY and AGG
df_portfolio = alpaca.get_barset(
    tickers,
    timeframe,
    start = date,
    end = date
).df

# Preview DataFrame
df_portfolio


# In[ ]:


# Pick AGG and SPY close prices
df_closing_prices = pd.DataFrame()
df_closing_prices["AGG"] = df_portfolio["AGG"]["close"]
df_closing_prices["SPY"] = df_portfolio["SPY"]["close"]
df_closing_prices.index = df_closing_prices.index.date


# In[ ]:


agg_close_price = df_closing_prices.iat[0, 0]
spy_close_price = df_closing_prices.iat[0, 1]


# In[ ]:


# Print AGG and SPY close prices
print(f"Current AGG closing price: ${agg_close_price}")
print(f"Current SPY closing price: ${spy_close_price}")


# In[ ]:


# Compute the current value of shares
my_spy_value = my_spy * spy_close_price
my_agg_value = my_agg * agg_close_price
shares = my_spy_value + my_agg_value
# Print current value of share
print(f"The current value of your {my_spy} SPY shares is ${my_spy_value:0.2f}")
print(f"The current value of your {my_agg} AGG shares is ${my_agg_value:0.2f}")


# ### Savings Health Analysis

# In[ ]:


# Set monthly household income
monthly_income = 12000
# Create savings DataFrame
data = [['crypto', crypto], ['shares', shares]]
df_savings = pd.DataFrame(data, columns = ['Type', 'Amount'])
df_savings.set_index(df_savings['Type'], inplace=True)
df_savings = df_savings.drop(columns=["Type"])
# Display savings DataFrame
display(df_savings)


# In[ ]:


# Plot savings pie chart
df_pie = df_savings.plot.pie(y='Amount', figsize=(5, 5), title="Savings Amount",)
df_pie


# In[ ]:


# Set ideal emergency fund
emergency_fund = monthly_income * 3
# Calculate total amount of savings
savings = crypto + shares
difference = emergency_fund - savings
# Validate saving health
if savings > emergency_fund:
    print("Congratulations for having enough money in this fund!")
elif savings == emergency_fund:
    print("Congratulations on reaching this financial goal!")
elif savings < emergency_fund:
    print("You are ${difference} from reaching your goal")


# ## Part 2 - Retirement Planning
# 
# ### Monte Carlo Simulation

# In[ ]:


# Set start and end dates of five years back from today.
# Sample results may vary from the solution based on the time frame chosen
start_date = pd.Timestamp('2016-05-20', tz='America/Chicago').isoformat()
end_date = pd.Timestamp('2021-05-20', tz='America/Chicago').isoformat()


# In[ ]:


# Get 5 years' worth of historical data for SPY and AGG
df_prediction = alpaca.get_barset(
    tickers,
    timeframe,
    start = start_date,
    end = end_date
).df

# Display sample data
df_prediction


# In[ ]:


# Configuring a Monte Carlo simulation to forecast 30 years cumulative returns
import numpy as np
import pandas as pd
import os
import alpaca_trade_api as tradeapi
import datetime as dt
import pytz

class MCSimulation:
    def __init__(self, portfolio_data, weights="", num_simulation=1000, num_trading_days=252):
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")
            
        # Set weights if empty, otherwise make sure sum of weights equals one.
        if weights == "":
            num_stocks = len(portfolio_data.columns.get_level_values(0).unique())
            weights = [1.0/num_stocks for s in range(0,num_stocks)]
        else:
            if round(sum(weights),2) < .99:
                raise AttributeError("Sum of portfolio weights must equal one.")
        
        # Calculate daily return if not within dataframe
        if not "daily_return" in portfolio_data.columns.get_level_values(1).unique():
            close_df = portfolio_data.xs('close',level=1,axis=1).pct_change()
            tickers = portfolio_data.columns.get_level_values(0).unique()
            column_names = [(x,"daily_return") for x in tickers]
            close_df.columns = pd.MultiIndex.from_tuples(column_names)
            portfolio_data = portfolio_data.merge(close_df,left_index=True,right_index=True).reindex(columns=tickers,level=0)    
        
        # Set class attributes
        self.portfolio_data = portfolio_data
        self.weights = weights
        self.nSim = num_simulation
        self.nTrading = num_trading_days
        self.simulated_return = ""
        
    def calc_cumulative_return(self):
        """
        Calculates the cumulative return of a stock over time using a Monte Carlo simulation (Brownian motion with drift).

        """
        
        # Get closing prices of each stock
        last_prices = self.portfolio_data.xs('close',level=1,axis=1)[-1:].values.tolist()[0]
        
        # Calculate the mean and standard deviation of daily returns for each stock
        daily_returns = self.portfolio_data.xs('daily_return',level=1,axis=1)
        mean_returns = daily_returns.mean().tolist()
        std_returns = daily_returns.std().tolist()
        
        # Initialize empty Dataframe to hold simulated prices
        portfolio_cumulative_returns = pd.DataFrame()
        
        # Run the simulation of projecting stock prices 'nSim' number of times
        for n in range(self.nSim):
        
            if n % 10 == 0:
                print(f"Running Monte Carlo simulation number {n}.")
        
            # Create a list of lists to contain the simulated values for each stock
            simvals = [[p] for p in last_prices]
    
            # For each stock in our data:
            for s in range(len(last_prices)):

                # Simulate the returns for each trading day
                for i in range(self.nTrading):
        
                    # Calculate the simulated price using the last price within the list
                    simvals[s].append(simvals[s][-1] * (1 + np.random.normal(mean_returns[s], std_returns[s])))
    
            # Calculate the daily returns of simulated prices
            sim_df = pd.DataFrame(simvals).T.pct_change()
    
            # Use the `dot` function with the weights to multiply weights with each column's simulated daily returns
            sim_df = sim_df.dot(self.weights)
    
            # Calculate the normalized, cumulative return series
            portfolio_cumulative_returns[n] = (1 + sim_df.fillna(0)).cumprod()
        
        # Set attribute to use in plotting
        self.simulated_return = portfolio_cumulative_returns
        
        # Calculate 95% confidence intervals for final cumulative returns
        self.confidence_interval = portfolio_cumulative_returns.iloc[-1, :].quantile(q=[0.025, 0.975])
        
        return portfolio_cumulative_returns
    
    def plot_simulation(self):
        """
        Visualizes the simulated stock trajectories using calc_cumulative_return method.

        """ 
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()
            
        # Use Pandas plot function to plot the return data
        plot_title = f"{self.nSim} Simulations of Cumulative Portfolio Return Trajectories Over the Next {self.nTrading} Trading Days."
        return self.simulated_return.plot(legend=None,title=plot_title)
    
    def plot_distribution(self):
        """
        Visualizes the distribution of cumulative returns simulated using calc_cumulative_return method.

        """
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()
        
        # Use the `plot` function to create a probability distribution histogram of simulated ending prices
        # with markings for a 95% confidence interval
        plot_title = f"Distribution of Final Cumuluative Returns Across All {self.nSim} Simulations"
        plt = self.simulated_return.iloc[-1, :].plot(kind='hist', bins=10,density=True,title=plot_title)
        plt.axvline(self.confidence_interval.iloc[0], color='r')
        plt.axvline(self.confidence_interval.iloc[1], color='r')
        return plt
    
    def summarize_cumulative_return(self):
        """
        Calculate final summary statistics for Monte Carlo simulated stock data.
        
        """
        
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()
            
        metrics = self.simulated_return.iloc[-1].describe()
        ci_series = self.confidence_interval
        ci_series.index = ["95% CI Lower","95% CI Upper"]
        return metrics.append(ci_series)


# In[ ]:


# Printing the simulation input data
MC_even_dist = MCSimulation(
    portfolio_data = df_prediction,
    weights = [.6,.4],
    num_simulation = 500,
    num_trading_days = 252*30
)
MC_even_dist.portfolio_data.head()


# In[ ]:


# Running a Monte Carlo simulation to forecast 30 years cumulative returns
MC_even_dist.calc_cumulative_return()


# In[ ]:


# Plot simulation outcomes
line_plot = MC_even_dist.plot_simulation()


# In[ ]:


# Plot probability distribution and confidence intervals
dist_plot = MC_even_dist.plot_distribution()


# ### Retirement Analysis

# In[ ]:


# Fetch summary statistics from the Monte Carlo simulation results
sum_stats = MC_even_dist.summarize_cumulative_return()

# Print summary statistics
print(sum_stats)


# ### Calculate the expected portfolio return at the 95% lower and upper confidence intervals based on a `$20,000` initial investment.

# In[ ]:


# Set initial investment
initial_investment = 20000

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $20,000
ci_lower = round(sum_stats[8]*initial_investment,2)
ci_upper = round(sum_stats[9]*initial_investment,2)
# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 30 years will end within in the range of"
      f" ${ci_lower} and ${ci_upper}")


# ### Calculate the expected portfolio return at the `95%` lower and upper confidence intervals based on a `50%` increase in the initial investment.

# In[ ]:


# Set initial investment
initial_investment2 = 20000 * 1.5

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $30,000
ci_lower = round(sum_stats[8]*initial_investment2,2)
ci_upper = round(sum_stats[9]*initial_investment2,2)
# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment2} in the portfolio"
      f" over the next 30 years will end within in the range of"
      f" ${ci_lower} and ${ci_upper}")


# ## Optional Challenge - Early Retirement
# 
# 
# ### Five Years Retirement Option

# In[ ]:


# Configuring a Monte Carlo simulation to forecast 5 years cumulative returns
# YOUR CODE HERE!


# In[ ]:


# Running a Monte Carlo simulation to forecast 5 years cumulative returns
# YOUR CODE HERE!


# In[ ]:


# Plot simulation outcomes
# YOUR CODE HERE!


# In[ ]:


# Plot probability distribution and confidence intervals
# YOUR CODE HERE!


# In[ ]:


# Fetch summary statistics from the Monte Carlo simulation results
# YOUR CODE HERE!

# Print summary statistics
# YOUR CODE HERE!


# In[ ]:


# Set initial investment
# YOUR CODE HERE!

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $60,000
# YOUR CODE HERE!

# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 5 years will end within in the range of"
      f" ${ci_lower_five} and ${ci_upper_five}")


# ### Ten Years Retirement Option

# In[ ]:


# Configuring a Monte Carlo simulation to forecast 10 years cumulative returns
# YOUR CODE HERE!


# In[ ]:


# Running a Monte Carlo simulation to forecast 10 years cumulative returns
# YOUR CODE HERE!


# In[ ]:


# Plot simulation outcomes
# YOUR CODE HERE!


# In[ ]:


# Plot probability distribution and confidence intervals
# YOUR CODE HERE!


# In[ ]:


# Fetch summary statistics from the Monte Carlo simulation results
# YOUR CODE HERE!

# Print summary statistics
# YOUR CODE HERE!


# In[ ]:


# Set initial investment
# YOUR CODE HERE!

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $60,000
# YOUR CODE HERE!

# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 10 years will end within in the range of"
      f" ${ci_lower_ten} and ${ci_upper_ten}")


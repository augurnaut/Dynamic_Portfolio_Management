#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st

import pandas as pd
import hvplot.pandas
import numpy as np
import fire
import questionary
import pytz
import datetime as dt


# from data_retrieval import alpaca
# Loading alpaca from data retrieval

import os
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as trade_api
from datetime import date

class Alpaca:
    """
    ///Only for standard stocks and bonds///
    A class that accepts inputs for tickers, the api key, 
    it's secret key, a time frame as well as the amount of 
    years of historical data requested. The object then 
    returns a cleaned dataframe of just close prices and volume.
    Inputs:
        ticker_list = list of tickers that we want data for 
        alpaca_api, alpaca_secret_key = name of alpaca api saved to env variable
        timeframe = expressed as ("1(D,W,M,H,S)")
        endpoint_years = accepts an int amount of years
    Output:
        Cleaned dataframe that stores the close and volume values for each stock
    """
    def __init__(self, ticker_list, timeframe, endpoint_years, alpaca_api=None, alpaca_secret_key=None):
        # Init class variables and set them equal to inputs

        self.ticker_list = ticker_list
        self.alpaca_api = str(alpaca_api)
        self.alpaca_secret_key = str(alpaca_secret_key)
        self.timeframe = timeframe
        self.endpoint_years = endpoint_years

    def load_variables(self):
        # Load environment variables to make api calls

        load_dotenv()

        if self.alpaca_api and self.alpaca_secret_key:
            self.alpaca_api = os.getenv(self.alpaca_api)
            self.alpaca_secret_key = os.getenv(self.alpaca_secret_key)
        else:
            self.alpaca_api = os.getenv("ALPACA_API_KEY_ENV")
            self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY_ENV")


    def create_api_call(self):
        # Set up alpaca api call

        self.alpaca = trade_api.REST(
        self.alpaca_api,
        self.alpaca_secret_key,
        api_version="v2")

        return self.alpaca

    def clean_tickers(self):
        # Take the list of tickers ipnut by the user and properly format for api call

        self.tickers = []
        for ticker in self.ticker_list:
            ticker = ticker.upper()
            self.tickers.append(ticker)

        return self.tickers

    def get_dataframe(self):
        # Make the api call

        # Using the date module and pandas Timedelta we get the current day as the start
        # And using the input from the user we go back x amount of years
        end_date = date.today() - pd.Timedelta(days=1)
        start_date = (end_date - pd.Timedelta(weeks=52*(self.endpoint_years)))

        # Make alpaca api call
        self.df_portfolio = self.alpaca.get_barset(
        self.tickers,
        self.timeframe,
        start = pd.Timestamp(start_date, tz="America/New_York").isoformat(),
        end = pd.Timestamp(end_date, tz="America/New_York").isoformat(),
        limit=1000
        ).df

        return self.df_portfolio

    def clean_dataframe(self):
        # Drop NaN values and set index to a pure date (yyyy-mm-dd)
        self.df_portfolio = self.df_portfolio.dropna()
        self.df_portfolio.index = self.df_portfolio.index.date
        clean_ticker_df = []
        columns = []

        # Set column names and parse dataframe down to just close and volume
        for ticker in self.tickers:

            close = f"{ticker}_close"
            volume = f"{ticker}_volume"

            columns.append(close)
            columns.append(volume)

            ticker = self.df_portfolio[ticker][['close', 'volume']]
            
            clean_ticker_df.append(ticker)

        # Concat resulting dataframes into one main dataframe
        self.clean_df = pd.concat(clean_ticker_df, axis='columns', join='inner')
        self.clean_df.columns = columns

        return self.clean_df

    def run(self):

        self.load_variables()

        self.create_api_call()

        self.clean_tickers()

        self.get_dataframe()

        self.clean_dataframe()

        # print(self.clean_df)
        
        return self.clean_df

tickers = ['aapl']
alpaca_api = "ALPACA_API_KEY_ENV"
alpaca_secret_api = "ALPACA_SECRET_KEY_ENV"
timeframe = "1D"
years = 3


if __name__ == '__main__':
    object = Alpaca(tickers, timeframe, years, alpaca_api, alpaca_secret_api)
    # object = Alpaca(tickers, timeframe, years)
    object.run()

# from data_retrieval import crypto_market

import pandas_datareader.data as pdr
import pandas as pd
from datetime import date, datetime

def cmc200(years):
    """
    Retrieves historical data from the coinmarketcap top 200 cryptocurrencies index.
    Does not return days the stockmarket is closed (May alter analysis but fine for initial purposes)
    Inputs:
        Years (an integer representing the number of years of historical data the user needs for analysis)
    Output:
        A dataframe with the adj_close price of the cmc200 index
    """
    # Set and format date strings for the datareader call.
    end_date = date.today() - pd.Timedelta(days=1)
    start_date = (end_date - pd.Timedelta(weeks=52*(int(years))))

    # Using pandas datareader we get data for the cmc200 index
    df = pdr.DataReader('^CMC200', 'yahoo', start=str(start_date), end=str(end_date))
    df = df.drop(columns=['High', 'Low', 'Open', 'Volume', 'Close'])
    df = df.rename(columns={'Adj Close':'Close'})

    return df

# from data_retrieval import getbinance

import time


def Binance(symbol, years):
    """
    Gets historical crypto data from the binance API
    Inputs:
        Symbol as shown on exchanges, years(int) of historical data to get the
    Outputs:
        A dataframe with date as index, close and volume data
    """

        
    years = int(years)

    # Format symbol for api call
    symbol = symbol.upper()
    binance_symbol = f"{symbol}USDT"

    # Load environment vairables
    # Set variables to binance api key env files
    load_dotenv()
    binance_api = os.getenv("BINANCE_API")
    binance_secret = os.getenv("BINANCE_SECRET")

    # Create api client variable
    client = Client(binance_api, binance_secret)

    # Create and format date
    start_date = datetime.date.today() - pd.Timedelta(weeks=52*years)
    start_date = start_date.strftime("%d %b, %Y")

    # make api call and get returned data
    candles = client.get_historical_klines(binance_symbol, Client.KLINE_INTERVAL_1DAY, limit=1000, start_str=start_date) 

    # Create and format dataframe 
    # Returns close and volume with date as index
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote asset volume', 
    'number of trades', 'taker buy base asset volume', 'taker buy quote asset volume', 'ignore']
    df = pd.DataFrame(candles, columns=columns)

    mills = df['date']
    date = pd.Series([datetime.datetime.fromtimestamp(mill/1000) for mill in mills])

    # Combine dataframes and drop columns
    df = pd.concat([date, df], axis=1, join='inner')
    df = df.drop(columns=['date', 'open', 'high', 'low', 'close time', 'quote asset volume', 'number of trades', 
    'taker buy base asset volume', 'taker buy quote asset volume', 'ignore'], axis=1)

    # Set data type to float
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    # Rename columns with ticker values
    df = df.rename(columns={0:'date', 'close':f'{binance_symbol}_Close', 'volume':f'{binance_symbol}_Volume'})

    # Set date index
    df = df.set_index(['date'])
    df.index = df.index.date

    return df

# from data_retrieval import sp500_index

import pandas_datareader.data as pdr
import pandas as pd
from datetime import date, datetime

def sp500(years):
    """
    Retrieves historical data from the coinmarketcap top 200 cryptocurrencies index.
    Does not return days the stockmarket is closed (May alter analysis but fine for initial purposes)
    Inputs:
        Years (an integer representing the number of years of historical data the user needs for analysis)
    Output:
        A dataframe with the adj_close price of the cmc200 index
    """
    # Set and format date strings for the datareader call.
    end_date = date.today() - pd.Timedelta(days=1)
    start_date = (end_date - pd.Timedelta(weeks=52*(int(years))))

    # Using pandas datareader we get data for the cmc200 index
    df = pdr.DataReader('^GSPC', 'yahoo', start=str(start_date), end=str(end_date))
    df = df.drop(columns=['High', 'Low', 'Open', 'Volume', 'Close'])
    df = df.rename(columns={'Adj Close':'Close'})

    return df

# from functions_graphs import functions

class Functions():
    """
    Class defining calculations for risk analysis
    Inputs: 
        Dataframe of a specific ticker, 
        column name of the close columns of the dataframe,
        lastly if there is a market it gets passed
    Outputs:
        New dataframes with daily/cumulative returns
        STD, Anual STD, Monthly STD, Anual average returns, Sharpe Ratio
    """
    def __init__(self, dataframe, column_name, market=None):
        # Init global variables
        self.dataframe = dataframe
        self.market = market
        self.column_name = column_name

    def daily_returns(self):
        # Make a daily returns column
        # Add error handling for instances where column name is incorrect
        self.dataframe['daily_returns'] = self.dataframe[self.column_name].pct_change()
        self.dataframe.dropna(inplace=True)
        return self.dataframe

    def cumulative_returns(self):
        # Make a cumulative returns column
        self.dataframe['cumulative_returns'] = (1 + self.daily_returns(self.dataframe)).cumprod()
        return self.dataframe

    def standard_deviation(self):
        # Calculate std for the dataframe
        return self.dataframe['daily_returns'].std()

    def annual_standard_deviation(self):
        # Calc annual std
        return self.dataframe['daily_returns'] * np.sqrt(252)

    def monthly_standard_deviation(self):
        # Calc monthly std
        return self.dataframe['daily_returns'] * np.sqrt(12)

    def annualize_average_returns(self):
        # Calc anual average means       
        return self.dataframe['daily_returns'].mean() * 252

    def sharpe_ratio(self):
        # Calc sharpe ratio
        return self.annualize_average_returns() / self.annual_standard_deviation()

# from user_input import risk_assesment

def get_user_risk_tolerance_port():
    """
    Get user information to determine risk tolerance
    """
    risk_tolerance = questionary.select("What's your risk tolerance", 
    choices=["Low", "Medium", "High"]).ask()

    indexes = questionary.select("Of the following indexes, select the index that best reflects your current portfolio.", 
    choices=["NASDAQ", "Russel", "S&P 500", "EAFE"]).ask()

    # crypto_benchmark = questionary.select("Would you like to benchmark your crypto against an" + 
    # " index, a specific crypto, or a composite of cryptos?", choices=["Index", "Crypto", "Crypto Composite"]).ask()

    goals = questionary.text("Would you like to acomplish a dollar goal or a percentage return goal?" +
    " (Start with a '$' if dollar goal and with '%' if percent goal").ask()

    invest_amount = int(questionary.text("How much would you like to invest?").ask())

    stock_portfolio = {}
    adding = True
    while adding:
        sp = questionary.text("Enter stock from stock portfolio: ").ask()
        sw = int(questionary.text("Enter Share Amount: ").ask())
        cont = questionary.confirm("Continue?").ask()
        stock_portfolio[sp] = sw
        if cont == False:
            adding = False

    user_dictionary = {"risk tolerance": risk_tolerance,
                        "index": indexes,
                        "Investment Goals": goals,
                        "Investment Amount": invest_amount,
                        "Stock Portfolio": stock_portfolio}

    return user_dictionary

# from data_retrieval import MCForecastTools as mc

class MCSimulation:
    """
    A Python class for runnning Monte Carlo simulation on portfolio price data. 
    
    ...
    
    Attributes
    ----------
    portfolio_data : pandas.DataFrame
        portfolio dataframe
    weights: list(float)
        portfolio investment breakdown
    nSim: int
        number of samples in simulation
    nTrading: int
        number of trading days to simulate
    simulated_return : pandas.DataFrame
        Simulated data from Monte Carlo
    confidence_interval : pandas.Series
        the 95% confidence intervals for simulated final cumulative returns
        
    """
    
    def __init__(self, portfolio_data, weights="", num_simulation=1000, num_trading_days=252):
        """
        Constructs all the necessary attributes for the MCSimulation object.
        Parameters
        ----------
        portfolio_data: pandas.DataFrame
            DataFrame containing stock price information from Alpaca API
        weights: list(float)
            A list fractions representing percentage of total investment per stock. DEFAULT: Equal distribution
        num_simulation: int
            Number of simulation samples. DEFAULT: 1000 simulation samples
        num_trading_days: int
            Number of trading days to simulate. DEFAULT: 252 days (1 year of business days)
        """
        
        # Check to make sure that all attributes are set
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

from collections import Counter

st.title('Dynamic Portfolio Management')



import pandas as pd
import numpy as np

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
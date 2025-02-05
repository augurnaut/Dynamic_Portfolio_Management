from data_retrieval import alpaca
from data_retrieval import crypto_market
from data_retrieval import getbinance
from data_retrieval import sp500_index

from functions_graphs import functions
from user_input import risk_assesment
from data_retrieval import MCForecastTools as mc

from collections import Counter

import pandas as pd
import hvplot.pandas
import numpy as np

# Assign a weight of 70%
# Crypto list = ADA, BNB, BTC, DOT, ETH, LINK, LTC, VET, XLM, XRP
# .041-.089
# Create bins for risk tolerance level - If std is below some_low_std() goes into low risk
# 0.0 -.057 Low
# .057 - .073 Med
# .073 - 1 High

#Assign a weight of 30%
# S&P 500 - Blue chips, large cap, top 500 representative of stock market (Broad index)
# Nasdaq - Smaller cap, a lot of overlap between these two
# Russel - Smallest cap
# Pick crypto based of market cap realtion to index

# Functions for program operations
def get_crypto_dict():
    """Returns a dictionary with ticker as index and its dataframe as a value"""

    cryptos = ['ADA', 'BNB', 'BTC', 'DOT', 'ETH', 'LINK', 'LTC', 'VET', 'XLM', 'XRP']
    crypto_dict = {}

    for ticker in cryptos:
        df = getbinance.Binance(ticker, 3)
        df = pd.DataFrame(df)
        crypto_dict[ticker] = df

    return crypto_dict

def get_std(crypto_dfs):
    """Returns a dictionary of all crypto stds"""

    cryptos = ['ADA', 'BNB', 'BTC', 'DOT', 'ETH', 'LINK', 'LTC', 'VET', 'XLM', 'XRP']
    crypto_std = {}

    for ticker in cryptos:
        df = crypto_dfs[ticker]
        c_name = f"{ticker}USDT_Close"
        funct = functions.Functions(df, c_name)
        daily_returns = funct.daily_returns()
        daily_returns.dropna(inplace=True)
        crypto_std[ticker] = funct.standard_deviation()

    return crypto_std

def get_sharpe(crypto_dfs):
    """Set a list of all the cryptos"""
    cryptos = ['ADA', 'BNB', 'BTC', 'DOT', 'ETH', 'LINK', 'LTC', 'VET', 'XLM', 'XRP']
    crypto_sharpe = {}

    # Loop through list and make a dictionary of sharpe ratios
    for ticker in cryptos:
        df = crypto_dfs[ticker]
        c_name = f"{ticker}USDT_Close"
        funct = functions.Functions(df, c_name)
        crypto_sharpe[ticker] = funct.sharpe_ratio()
    
    return crypto_sharpe

def sort_crypto_std(crypto_std):
    """Returns a dictionary with all crypto std's sorted from low to high risk"""

    high_risk = {}
    med_risk = {}
    low_risk = {}

    # Separates tickers based on std
    for ticker, std in crypto_std.items():
        if float(std) <= 0.057:
            low_risk[ticker] = std
        elif float(std) > 0.057 and float(std) <= 0.073:
            med_risk[ticker] = std
        elif float(std) > 0.073:
            high_risk[ticker] = std

    return {"low":low_risk, "med":med_risk, "high":high_risk}

def sort_crypto_sharpe(crypto_sharpe):
    """Returns a dictionary with all crypto sharpe's sorted from low to high risk"""

    high_risk = {}
    med_risk = {}
    low_risk = {}

    # Is supposed to separate sharpes based on their value,
    # but it's reading sharpe as a dataframe
    for ticker, sharpe in crypto_sharpe.items():
        if float(sharpe) <= 0.5:
            high_risk[ticker] = sharpe
        elif float(sharpe) > 0.5 and float(sharpe) <= 1.5:
            med_risk[ticker] = sharpe
        elif float(sharpe) > 1.5:
            low_risk[ticker] = sharpe

    return {"low":low_risk, "med":med_risk, "high":high_risk}


print("----------Dynamic Portfolio Management - Version 1.0----------")

user_info = risk_assesment.get_user_risk_tolerance_port()


print("----------User Info----------")
# Print user risk assessment dictionary
for i, v in user_info.items():
    print(f"{i}: {v}")
print("\n")

# Stores a "crypto ticker" : value, dictionary in a variable
# And calls the sort function to get
dictionary = get_crypto_dict()
sorted_std = sort_crypto_std(get_std(dictionary))

weights_ = [] # Will be used in the future to automate weight calculation
crypto_sharpe = get_sharpe(dictionary) 

# Until the sharpe ratio function this keeps the code running
try:
    sorted_sharpe = sort_crypto_sharpe(dictionary)
    print(sorted_sharpe)
except:
    pass


print("\n")
print("----------Crypto Risk----------")

# Prints out all of the crypto bins
for t,v in sorted_std['low'].items():
    print(f"Low Risk Cryptos - {t}:{v}")
print("----------")
for t,v in sorted_std['med'].items():
    print(f"Medium Risk Cryptos - {t}:{v}")  
print("----------")
for t,v in sorted_std['high'].items():
    print(f"High Risk Cryptos - {t}:{v}")

# Get user initial investment
i_amount = user_info["Investment Amount"]

# takes input from the user and selects the appropriate bin
len_crypto = []

# Make high bin
if user_info['risk tolerance'] == 'High':
    for t, v in sorted_std['high'].items():
        len_crypto.append(t)
        for t, v in sorted_std['med'].items():
            len_crypto.append(t)
            for t, v in sorted_std['low'].items():
                len_crypto.append(t)

# Make medium bin               
elif user_info['risk tolerance'] == 'Medium':
    for t, v in sorted_std['med'].items():
        len_crypto.append(t)
        for t, v in sorted_std['low'].items():
            len_crypto.append(t)

# Make low bin
elif user_info['risk tolerance'] == 'Low':
    for t, v in sorted_std['low'].items():
        len_crypto.append(t)

# Calculate weighting for risk portion of portfolio
# using weighting get price per crypto
crypto_risk_allotment = i_amount * 0.7
price_per_crypto = crypto_risk_allotment / len(len_crypto)


print("\n")
print("----------Order Book----------")

# Print risk order book
num_orders = Counter(len_crypto)
set_crypto = set(len_crypto)

weight_calc = {}
for ticker in set_crypto:
    print(f"Purchase ${price_per_crypto * num_orders[ticker]:.2f} of {ticker}")
    # Add to dictionary to calculate weights
    weight_calc[ticker] = float(price_per_crypto * num_orders[ticker])

print("\n")
print(f"This should equal 70%, ${crypto_risk_allotment}, of your stated initial investment amount of {user_info['Investment Amount']}")
print("\n")

# Using BTC, ETH and LTC as a benchmark for the s&p we allot the last 30% to these cryptos
# i = index
snp_crypto = ['BTC', 'ETH', 'LTC']
crypto_index_allotment = i_amount * 0.3
price_per_crypto_i = crypto_index_allotment / len(snp_crypto)

# for ticker in snp_crypto:
#     weight_calc[ticker] += float(price_per_crypto_i)

# print_weights = [f"{t}:{v}" for t, v in weight_calc.items()]
# print(print_weights)

# Print index order book
for ticker in snp_crypto:
    print(f"Purchase ${price_per_crypto_i:.2f} of {ticker}")
print("\n")
print(f"This should equal 30%, ${crypto_index_allotment:.2f}, of your stated initial investment amount of {user_info['Investment Amount']}")

# Pull info for user's stock portfolio
tickers = []
for ticker, shares in user_info['Stock Portfolio'].items():
    tickers.append(ticker)

# Api env variables
# For anyone running on their on computer be sure to change these env variable to reflect the names you use
alpaca_api = "ALPACA_API_KEY_ENV"
alpaca_secret_api = "ALPACA_SECRET_KEY_ENV"

# Make the call to alpaca
alpaca_call = alpaca.Alpaca(tickers, '1D', 1, alpaca_api, alpaca_secret_api)
stocks_df = alpaca_call.run()

print("\n")
print("----------Stock Portfolio Value----------")
# Set empty list and dictionary to 
close_values = []
stock_port = {}

# Prints the last closing price of each stock in the portfolio
for ticker in tickers:
    close = f"{ticker.upper()}_close"
    stock_port[ticker] = stocks_df[close]
    close_v = stocks_df[close][-1]
    print(f"{ticker} closed at {close_v}")
    close_values.append(close_v)

print("\n")
# Multiply last closing price by share amount and store in list
portfolio_value = []
for value in close_values:
    for ticker, share in user_info['Stock Portfolio'].items():
        ticker = float(value) * float(share)
        portfolio_value.append(ticker)

# Sum the values in the list to get total portfolio value
portfolio_value = sum(portfolio_value)

print(f"The current value of your stock portfolio is ${portfolio_value:.2f}")
print("\n")

# Get CMC close
cmc = crypto_market.cmc200(1)
cmc.columns = ['CMC_close']

cmc_dr = cmc['CMC_close'].pct_change().dropna()

cmc_cum = (1 + cmc_dr).cumprod()

# Get S&P 500 close
sp500 = sp500_index.sp500(1)
sp500.columns = ['SP500_close']

sp500_dr = sp500['SP500_close'].pct_change().dropna()

sp500_cum = (1 + sp500_dr).cumprod()

# Concat dataframe of indexes
index_cm = pd.concat([sp500_cum, cmc_cum], axis=1)
index_cm.dropna(inplace=True)

# Plot of both indexes daily close
index_overlay_plot = index_cm.hvplot.line(title='CMC200 and S&P500 Daily Returns Overlayed')

# Rolling mean of both indexes
sp_cmc = index_cm.rolling(window=30).mean().dropna()

# Plot of rolling 30 mean indexes
index_dr_30_plot = sp_cmc.hvplot.line(title='CMC200 and S&P500 Overlayed - 30 Day')

# Aggregate average of indexes 30 day
sp_cmc_avg = index_cm.mean(axis=1)
sp_cmc_avg_30 = sp_cmc_avg.rolling(window=30).mean()

# Create a list of dataframes to concat stocks with cryptos
list_df = []

# Store stock dfs
for t, df in stock_port.items():
    df = pd.DataFrame(df)
    list_df.append(df)
    # print(df.head())

# Add crypto dfs
for t, df in dictionary.items():
    df = pd.DataFrame(df)
    df = df[f"{t}USDT_Close"]
    list_df.append(df)

# Get new joined dataframe
port_closes = pd.concat(list_df, axis=1, join='inner')

# Drop NaN values
port_daily_returns = port_closes.pct_change().dropna()

# Get cumulative returns
port_cum_returns = (1 + port_daily_returns).cumprod()

# Get rolling cumulative returns
port_cum_30 = port_cum_returns.rolling(window=30).mean()

# Average the returns across columns to get average total returns
combined_cum_returns = port_cum_returns.mean(axis=1)
combined_cum_returns30 = port_cum_30.mean(axis=1)

# Concat portfolio with index to get new dataframe
index_returns_df = pd.concat([combined_cum_returns, index_cm, sp_cmc_avg], axis=1)
index_returns_df = index_returns_df.dropna()
index_returns_df.columns = ['Portfolio Cum. Returns', 'S&P 500', 'CMC200', 'Avg Index']

# Concat average portfolio with averaged index - 30 day
index_returns_30_df = pd.concat([combined_cum_returns30, sp_cmc_avg_30], axis=1)
index_returns_30_df = index_returns_30_df.dropna()
index_returns_30_df.columns = ['Portfolio Cum. Returns', 'Avg Index']

# montecarlo = mc.MCSimulation(port_cum_returns, weights="", num_simulation=1000, num_trading_days=252)

# Plot graphs in separate window
hvplot.show(
    index_returns_df.hvplot(y=['Portfolio Cum. Returns', 'S&P 500', 'CMC200', 'Avg Index'], 
    value_label='Cumulative Returns', xlabel='Date', title='Portfolio Returns vs Indexes')
    +
    index_returns_30_df.hvplot(y=['Portfolio Cum. Returns', 'Avg Index'], 
    value_label='Cumulative Returns', xlabel='Date', title='Portfolio Returns vs Average Index - 30 Day')
)
import yfinance as yf
import yfinance as yf
from tabulate import tabulate
import pandas as pd
import numpy as np

ticker = 'tqqq'
period = "1000d"
historical_data = []

symbol = yf.Ticker(ticker)
data = symbol.history(period, interval="1d")
historical_data.append(data)



def calcRsi(data, column='Close', period=14, ema_period=14):
    delta = data[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    ema_rsi = rsi.ewm(span=ema_period, adjust=False).mean()

    return rsi, ema_rsi

def rsi_greater_than_ema(rsi, ema_rsi):
    if rsi >= ema_rsi:
        return 1
    else:
        return 0

def calcVWAP(data, period=1):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (tp * data['Volume']).rolling(window=period, min_periods=1).sum() / data['Volume'].rolling(window=period, min_periods=1).sum()
    return vwap

def calcSMA(data, period=5):
    sma = data['Close'].rolling(window=period, min_periods=1).mean()
    return sma

def vwap_greater_than_sma(vwap, sma):
    if vwap > sma:
        return 1
    else:
        return 0

for i in range(len(historical_data)):
    print (historical_data[i])
    rsi, ema_rsi = calcRsi(historical_data[i])

    historical_data[i]['RSI'] = rsi
    historical_data[i]['EMA_RSI'] = ema_rsi
    historical_data[i]['VWAP'] = calcVWAP(historical_data[i])
    historical_data[i]['SMA'] = calcSMA(historical_data[i])

    x = 0
    buyPrice =0
    sellPrice = 0
    buyTime = None
    sellTime = None

    buyPriceArray = []
    sellPriceArray = []
    buyTimeArray = []
    sellTimeArray = []
    profitArray = []

    for index, row in historical_data[i].iterrows():
        total = 0
        closeValue = row['Close']
        volumeValue = row['Volume']
        rsiValue = row['RSI']
        emaRsiValue = row['EMA_RSI']
        vwapValue = row['VWAP']
        smaValue = row['SMA']


        RSIEMAX = rsi_greater_than_ema(rsiValue, emaRsiValue)
        VWAPSMAX = vwap_greater_than_sma(vwapValue, smaValue)
        print(f"Date: {index}, Close: {closeValue}, RSI: {rsiValue}, EMA_RSI: {emaRsiValue},  RSI > EMA_RSI: {RSIEMAX}, VWAP: {vwapValue}, SMA: {smaValue}, VWAP > SMA: {VWAPSMAX}")
        total = RSIEMAX + VWAPSMAX
        if total == 2 and x == 0:
            buyPrice = closeValue
            buyPriceArray.append(buyPrice)
            buyTime = index
            buyTimeArray.append(buyTime)
            x = 1
        elif (total == 0 or total == 1) and x == 1:
            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            x = 0

print("\n")
headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit"]
data = list(zip(buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray))
print(tabulate(data, headers=headers))
print("Total Profit: ", sum(profitArray))

            

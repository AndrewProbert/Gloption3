import yfinance as yf
import yfinance as yf
from tabulate import tabulate
import pandas as pd
import numpy as np

ticker = 'spy'
#period = "1000d"
start_date = "2022-10-01"
end_date = "2023-11-10"
historical_data = []

symbol = yf.Ticker(ticker)
#data = symbol.history(period, interval="1d")
#data = symbol.history(start=start_date, interval="1d")
data = symbol.history(start=start_date, end=end_date, interval="1d")
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

def calcMACD(data, period_long=26, period_short=12, period_signal=9):
    ema_long = data['Close'].ewm(span=period_long, adjust=False).mean()
    ema_short = data['Close'].ewm(span=period_short, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=period_signal, adjust=False).mean()
    return macd, signal

def macd_greater_than_signal(macd, signal):
    if macd > signal:
        return 1
    else:
        return 0
    

def ema_greater_than_knn(ema, knn_ma):
    if ema > knn_ma:
        return 1
    else:
        return 0



ma_len = 5
ema_len_5 = 5

def calculate_knn_ma(price_values, ma_len):
    knn_ma = [np.mean(price_values[i-ma_len:i]) for i in range(ma_len, len(price_values))]
    knn_ma = [0]*ma_len + knn_ma
    return knn_ma


def calculate_knn_prediction(price_values, ma_len, num_closest_values=3, smoothing_period=50):
    def mean_of_k_closest(value, target, num_closest):
        closest_values = []
        for i in range(len(value)):
            distances = [abs(target[i] - v) for v in closest_values]
            if len(distances) < num_closest or min(distances) < min(distances):
                closest_values.append(value[i])
            if len(distances) >= num_closest:
                max_dist_index = distances.index(max(distances))
                if distances[max_dist_index] > min(distances):
                    closest_values[max_dist_index] = value[i]
        return sum(closest_values) / len(closest_values)

    knn_ma = [mean_of_k_closest(price_values[i-ma_len:i], price_values[i-ma_len:i], num_closest_values)
              for i in range(ma_len, len(price_values))]

    if len(knn_ma) < smoothing_period:
        return []

    knn_smoothed = np.convolve(knn_ma, np.ones(smoothing_period) / smoothing_period, mode='valid')

    def knn_prediction(price, knn_ma, knn_smoothed):
        pos_count = 0
        neg_count = 0
        min_distance = 1e10
        nearest_index = 0
        
        # Check if there are enough elements in knn_ma and knn_smoothed
        if len(knn_ma) < 2 or len(knn_smoothed) < 2:
            return 0  # Return 0 for neutral if there aren't enough elements
        
        for j in range(1, min(10, len(knn_ma))):
            distance = np.sqrt((knn_ma[j] - price) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_index = j
                
                # Check if there are enough elements to compare
                if nearest_index >= 1:
                    if knn_smoothed[nearest_index] > knn_smoothed[nearest_index - 1]:
                        pos_count += 1
                    if knn_smoothed[nearest_index] < knn_smoothed[nearest_index - 1]:
                        neg_count += 1
        
        return 1 if pos_count > neg_count else -1

    knn_predictions = [knn_prediction(price_values[i], knn_ma[i - smoothing_period:i], knn_smoothed[i - smoothing_period:i])
                       for i in range(smoothing_period, len(price_values))]

    return knn_predictions



def calculate_ema(price_values, ema_len):
    ema = np.zeros(len(price_values))
    ema[ema_len-1] = np.mean(price_values[:ema_len])
    multiplier = 2 / (ema_len + 1)
    
    for i in range(ema_len, len(price_values)):
        ema[i] = (price_values[i] - ema[i-1]) * multiplier + ema[i-1]

    

    return ema

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
    
def connors_rsi(data, period=3, ema_period=2):
    delta = data['Close'].diff(1)
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)

    up_sma = up.rolling(window=period, min_periods=1).mean()
    down_sma = down.rolling(window=period, min_periods=1).mean()

    rsi = up_sma / (up_sma + down_sma)
    rsi = rsi.ewm(span=ema_period, adjust=False).mean()

    return rsi

#See if connors rsi is between 0.2 and 0.8
def connors_rsi_between_20_80(connors_rsi):
    if connors_rsi >= 0.2 and connors_rsi <= 0.7:
        return 1
    else:
        return 0
    

#function to see if price is over or under the 200 day moving average
def price_over_ma(data):
    if data['Close'] > data['200ma']:
        return 1
    else:
        return 0

for i in range(len(historical_data)):
    print (historical_data[i])
    rsi, ema_rsi = calcRsi(historical_data[i])

    historical_data[i]['RSI'] = rsi
    historical_data[i]['EMA_RSI'] = ema_rsi
    macd, signal = calcMACD(historical_data[i])
    historical_data[i]['MACD'] = macd
    historical_data[i]['SIGNAL'] = signal
    historical_data[i]['KNN_MA'] = calculate_knn_ma(historical_data[i]['Close'], ma_len)
    historical_data[i]['EMA'] = calculate_ema(historical_data[i]['Close'], ema_len_5)
    historical_data[i]['VWAP'] = calcVWAP(historical_data[i])
    historical_data[i]['SMA'] = calcSMA(historical_data[i])
    historical_data[i]['CONNORS_RSI'] = connors_rsi(historical_data[i])






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
    table_data = []

    for index, row in historical_data[i].iterrows():
        total = 0
        closeValue = row['Close']
        volumeValue = row['Volume']
        rsiValue = row['RSI']
        emaRsiValue = row['EMA_RSI']
        macdValue = row['MACD']
        signalValue = row['SIGNAL']
        knn_ma = row['KNN_MA']
        ema = row['EMA']
        vwap = row['VWAP']
        sma = row['SMA']
        connors_rsi = row['CONNORS_RSI']
        




        RSIEMAX = rsi_greater_than_ema(rsiValue, emaRsiValue)
        MACDSMAX = macd_greater_than_signal(macdValue, signalValue)
        KNNEMAX = ema_greater_than_knn(ema, knn_ma)
        VWAPSMAX = vwap_greater_than_sma(vwap, sma)
        CONNORSRSI = connors_rsi_between_20_80(connors_rsi)

        total = RSIEMAX + MACDSMAX + KNNEMAX #+ VWAPSMAX + CONNORSRSI
        table_data.append([index, closeValue, rsiValue, emaRsiValue, RSIEMAX, macdValue, signalValue, MACDSMAX, knn_ma, ema, KNNEMAX,
                           vwap, sma, VWAPSMAX, CONNORSRSI, total])


        stop_loss = 0.005 # 5% stop loss
        stop_profit = 1 # 10% stop profit

        if total == 0 and x == 0:
            buyPrice = closeValue
            buyPriceArray.append(buyPrice)
            buyTime = index
            buyTimeArray.append(buyTime)
            x = 1
        elif (total != 0 or (closeValue - buyPrice) / buyPrice <= -stop_loss) and x == 1:

            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            x = 0


headers = ["Date", "Close", "RSI", "EMA_RSI", "RSI > EMA_RSI", "MACD", "SIGNAL", "MACD > SIGNAL", "KNN_MA", "EMA", "EMA > KNN_MA", "VWAP", "SMA", "VWAP > SMA", "Connors", "Total"]
print(tabulate(table_data, headers=headers))    

print("\n")
headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit"]
data = list(zip(buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray))
print(tabulate(data, headers=headers))
print("Total Profit: ", sum(profitArray))

            

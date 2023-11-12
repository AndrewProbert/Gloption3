import yfinance as yf
import yfinance as yf
from tabulate import tabulate
import pandas as pd
import numpy as np

ticker = 'qqq'
#period = "1000d"
start_date = "1999-11-05"
end_date = "2023-11-11"
historical_data = []

symbol = yf.Ticker(ticker)
#data = symbol.history(period, interval="1d")
#data = symbol.history(start=start_date, interval="1d")
data = symbol.history(start=start_date, end=end_date, interval="1d")
historical_data.append(data)



def calcRsi(data, column='Close', period=30, ema_period=21):
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

def calcMACD(data, period_long=52, period_short=12, period_signal=9):
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


ma_len = 21
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

def calcVWAP(data, period=14):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (tp * data['Volume']).rolling(window=period, min_periods=1).sum() / data['Volume'].rolling(window=period, min_periods=1).sum()
    return vwap

def calcSMA(data, period=1):#used to be 5 now 1
    sma = data['Close'].rolling(window=period, min_periods=1).mean()
    return sma

def vwap_greater_than_sma(vwap, sma): 
    if vwap > sma:
        return 1
    else:
        return 0
    
def calcADX(data, period=14):
    # Calculate the True Range
    data['TR'] = 0
    for i in range(1, len(data)):
        data['TR'][i] = max(data['High'][i] - data['Low'][i], abs(data['High'][i] - data['Close'][i-1]), abs(data['Low'][i] - data['Close'][i-1]))
    
    # Calculate the Directional Movement
    data['DM+'] = 0
    data['DM-'] = 0
    for i in range(1, len(data)):
        data['DM+'][i] = data['High'][i] - data['High'][i-1] if data['High'][i] - data['High'][i-1] > data['Low'][i-1] - data['Low'][i] else 0
        data['DM-'][i] = data['Low'][i-1] - data['Low'][i] if data['Low'][i-1] - data['Low'][i] > data['High'][i] - data['High'][i-1] else 0
    
    # Calculate the Directional Indicators
    data['DI+'] = 100 * data['DM+'].rolling(window=period, min_periods=1).sum() / data['TR'].rolling(window=period, min_periods=1).sum()
    data['DI-'] = 100 * data['DM-'].rolling(window=period, min_periods=1).sum() / data['TR'].rolling(window=period, min_periods=1).sum()
    
    # Calculate the Directional Index
    data['DX'] = 100 * abs(data['DI+'] - data['DI-']) / (data['DI+'] + data['DI-'])
    
    # Calculate the ADX
    data['ADX'] = data['DX'].rolling(window=period, min_periods=1).mean()
    
    return data['ADX']

#return 1 if adx > 25 else 0
def adx_greater_than_25(adx):
    if adx > 25:
        return 1
    else:
        return 0
    

def calcATR(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def maCrossOver(data, period1 = 9, period2 = 21):
    data['MA1'] = data['Close'].rolling(window=period1, min_periods=1).mean()
    data['MA2'] = data['Close'].rolling(window=period2, min_periods=1).mean()
    data['MA_Crossover'] = np.where(data['MA1'] > data['MA2'], 1, 0)
    return data['MA_Crossover']

def stochastic(data, period=14):
    data['14-high'] = data['High'].rolling(window=period, min_periods=1).max()
    data['14-low'] = data['Low'].rolling(window=period, min_periods=1).min()
    data['%K'] = 100 * (data['Close'] - data['14-low']) / (data['14-high'] - data['14-low'])
    data['%D'] = data['%K'].rolling(window=3, min_periods=1).mean()
    return data['%K'], data['%D']


def calcAroon(data, period=25):
    data['25-high'] = data['High'].rolling(window=period, min_periods=1).max()
    data['25-low'] = data['Low'].rolling(window=period, min_periods=1).min()
    data['Aroon Up'] = 14 * (data['25-high'] - data['Close']) / period
    data['Aroon Down'] = 14 * (data['25-low'] - data['Close']) / period
    return data['Aroon Up'], data['Aroon Down']

#check if aroon up is greater than aroon down
def aroon_up_greater_than_aroon_down(aroon_up, aroon_down):
    if aroon_up > aroon_down:
        return 1
    else:
        return 0
    

def price_greater_than_sma(price, sma):
    if price > sma:
        return 1
    else:
        return 0
    

def calcSuperTrend(data, period=10, multiplier=3):
    data['ATR'] = calcATR(data, period)
    data['Upper Basic'] = (data['High'] + data['Low']) / 2 + multiplier * data['ATR']
    data['Lower Basic'] = (data['High'] + data['Low']) / 2 - multiplier * data['ATR']
    data['Upper Band'] = data['Upper Basic']
    data['Lower Band'] = data['Lower Basic']
    for i in range(period, len(data)):
        if data['Close'][i-1] <= data['Upper Band'][i-1]:
            data['Upper Band'][i] = min(data['Upper Basic'][i], data['Upper Band'][i-1])
        else:
            data['Upper Band'][i] = data['Upper Basic'][i]
    for i in range(period, len(data)):
        if data['Close'][i-1] >= data['Lower Band'][i-1]:
            data['Lower Band'][i] = max(data['Lower Basic'][i], data['Lower Band'][i-1])
        else:
            data['Lower Band'][i] = data['Lower Basic'][i]
    data['SuperTrend'] = 0
    for i in range(period, len(data)):
        if data['Close'][i] <= data['Upper Band'][i]:
            data['SuperTrend'][i] = data['Upper Band'][i]
        elif data['Close'][i] > data['Upper Band'][i]:
            data['SuperTrend'][i] = data['Lower Band'][i]
    data['SuperTrend Direction'] = np.where(data['SuperTrend'] > data['SuperTrend'].shift(), 1, 0)
    return data['SuperTrend'], data['SuperTrend Direction']

#if supertrend is bullish return 1 else 0
def supertrend_greater_than_0(supertrend):
    if supertrend > 0:
        return 1
    else:
        return 0
    

def calcCCI(data, period=20):
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['MA'] = data['TP'].rolling(window=period, min_periods=1).mean()
    data['MD'] = data['TP'].rolling(window=period, min_periods=1).std()
    data['CCI'] = (data['TP'] - data['MA']) / (0.015 * data['MD'])
    return data['CCI']

def CCI_greater_than_100(CCI):
    if CCI > 100:
        return 1
    else:
        return 0
    
def fibonnaci_levels(data):
    data['High-Low'] = data['High'] - data['Low']
    data['38.2%'] = data['High'] - data['High-Low'] * 0.382
    data['50.0%'] = data['High'] - data['High-Low'] * 0.5
    data['61.8%'] = data['High'] - data['High-Low'] * 0.618
    
    return data['38.2%'], data['50.0%'], data['61.8%']

def calc_fib_bullish(data):
    if data['Close'] > data['38.2%'] and data['Close'] > data['50.0%'] and data['Close'] > data['61.8%']:
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
    historical_data[i]['ADX'] = calcADX(historical_data[i])
    atr_values = calcATR(historical_data[i])
    aroonUp, aroonDown = calcAroon(historical_data[i])
    historical_data[i]['Aroon Up'] = aroonUp
    historical_data[i]['Aroon Down'] = aroonDown
    #superTrend, superTrendDirection = calcSuperTrend(historical_data[i])
    #historical_data[i]['SuperTrend'] = superTrend
    #historical_data[i]['SuperTrend Direction'] = superTrendDirection
    historical_data[i]['CCI'] = calcCCI(historical_data[i])
    cci = historical_data[i]['CCI'] 
    historical_data[i]['38.2%'], historical_data[i]['50.0%'], historical_data[i]['61.8%'] = fibonnaci_levels(historical_data[i])
    historical_data[i]['Fib Bullish'] = historical_data[i].apply(calc_fib_bullish, axis=1)




  












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
    total_gain = []
    total_loss = []
    profit_by_year = {}
    highestTradePrice = 0


    for index, row in historical_data[i].iterrows():
        if (x == 1) and (closeValue > highestTradePrice):
            highestTradePrice = closeValue
            #print("Highest Trade Price: ", highestTradePrice)

        atr_value = atr_values.loc[index]





        total = 0
        openValue = row['Open']
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
        
        adx = row['ADX']

        aroonUp = row['Aroon Up']
        aroonDown = row['Aroon Down']
        #superTrend = row['SuperTrend']
        #superTrendDirection = row['SuperTrend Direction']
        cci = row['CCI']
        fibBullish = row['Fib Bullish']



        
        


        


        




        RSIEMAX = rsi_greater_than_ema(rsiValue, emaRsiValue)
        MACDSMAX = macd_greater_than_signal(macdValue, signalValue)
        KNNEMAX = ema_greater_than_knn(ema, knn_ma)
        VWAPSMAX = vwap_greater_than_sma(vwap, sma)
        ADX25 = adx_greater_than_25(adx)
        MACROSS = maCrossOver(historical_data[i])
        AROONUP = aroon_up_greater_than_aroon_down(aroonUp, aroonDown)
        #SUPERX = supertrend_greater_than_0(superTrend)
        CCIX = CCI_greater_than_100(cci)
        FIBBULL = fibBullish
            

        total = KNNEMAX + RSIEMAX   + VWAPSMAX  + MACDSMAX 
        table_data.append([index, closeValue, rsiValue, emaRsiValue, RSIEMAX, macdValue, signalValue, MACDSMAX, knn_ma, ema, KNNEMAX,
                           vwap, sma, VWAPSMAX, adx, ADX25,  total])


        #use and (MACROSS[index] == 1) for leveraged
        if ((total == 2 and x == 0) and ADX25 == 1) and (FIBBULL == 0): 
            buyPrice = closeValue
            buyPriceArray.append(buyPrice)
            buyTime = index
            buyTimeArray.append(buyTime)
            x = 1
        elif (total < 1 and x == 1) or (closeValue > buyPrice * 1.5 and x == 1) :

            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            if profit > 0:
                total_gain.append(profit)
            else:
                total_loss.append(profit)
            
            x = 0
            
            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)


            
        elif closeValue < buyPrice * 0.01 and x == 1:
            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            if profit > 0:
                total_gain.append(profit)
            else:
                total_loss.append(profit)
            x = 0
            
            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)

        '''

           # Check for ATR stop loss
        if x == 1 and closeValue < buyPrice - 1.45 * atr_value:  # You can adjust the multiplier as needed
            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            if profit > 0:
                total_gain.append(profit)
            else:
                total_loss.append(profit)
            x = 0
            highestTradePrice = 0
                        # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)
            
            print("Stopped out with ATR stop loss")
        
    '''


        

        if (x == 1) and (closeValue < highestTradePrice * 0.0):
            sellPrice = closeValue
            sellPriceArray.append(sellPrice)
            sellTime = index
            sellTimeArray.append(sellTime)
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            if profit > 0:
                total_gain.append(profit)
            else:
                total_loss.append(profit)
            x = 0
            highestTradePrice = 0
            
            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)



                    


headers = ["Date", "Close", "RSI", "EMA_RSI", "RSI > EMA_RSI", "MACD", "SIGNAL", "MACD > SIGNAL", "KNN_MA", "EMA", "EMA > KNN_MA", "VWAP", "SMA", "VWAP > SMA", "ADX", "ADX > 25", "SuperTrend", "SuperTrend > 0", "Total"]
print(tabulate(table_data, headers=headers))    

print("\n")
headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit"]
data = list(zip(buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray))
print(tabulate(data, headers=headers))
print("Total Profit: ", sum(profitArray))
print("Total Gain: ", sum(total_gain))
print("Total Loss: ", sum(total_loss))
print("Total Trades: ", len(profitArray))





for year in sorted(profit_by_year.keys()):
    print(f"{year}: {sum(profit_by_year[year])}")

if x == 1:
    print ("Still holding")
    print ("Buy Price: ", buyPrice, "Buy Time: ", buyTime)
    print ("Current Price: ", closeValue, "Current Time: ", index, "Additional Profit: ", closeValue - buyPrice)
    print ("Total Profit: ", sum(profitArray) + closeValue - buyPrice)  


    # Count the number of profitable trades
    num_profitable_trades = sum(profit > 0 for profit in profitArray)

    # Calculate the success rate
    success_rate = num_profitable_trades / len(profitArray)

    # Print the success rate
    print("Success rate: {:.2%}".format(success_rate))

    

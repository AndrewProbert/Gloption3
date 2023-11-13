from itertools import count
import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from datetime import datetime
import os
from datetime import datetime, timedelta


def run_program(end_date):
    ticker_symbol = ['tqqq']

    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = '2022-12-01'
    period = "100d"
    historical_data = []

    for i in ticker_symbol:
        ticker = yf.Ticker(i)
        data = ticker.history(start=start_date, end=end_date, interval="1d")
        historical_data.append(data)

    def calcRSI(data, period = 14):
        rsi = []
        for i in range(len(data)):
            if i < period:
                rsi.append(0)
            else:
                avgGain = 0
                avgLoss = 0
                for j in range(i - period, i):
                    change = data[j + 1] - data[j]
                    if change >= 0:
                        avgGain += change
                    else:
                        avgLoss += change
                avgGain = avgGain / period
                avgLoss = avgLoss / period

                if avgLoss == 0:
                    rsi.append(100)
                else:
                    rs = avgGain / abs(avgLoss)
                    rsi.append(100 - (100 / (1 + rs)))
        return rsi

    def calcRSIEMA(data, period=14):
        multiplier = 2 / (period + 1)
        ema = []

        for index, value in enumerate(data):
            if index == 0:
                ema.append(value)
            else:
                ema.append((value - ema[index - 1]) * multiplier + ema[index - 1])

        return ema


    def rsiemaCross (rsi, rsiEMA):
        if rsi[i] > rsiEMA[i]:
            return 1
        else:
            return 0

    def calcSMA (data, period=5):
        sma = []

        for index, value in enumerate(data):
            if index < period - 1:
                sma.append(0)
            else:
                sum_values = sum(data[index - period + 1 : index + 1])
                sma.append(sum_values / period)

        return sma


    def calcVWAP(prices, volumes):
        if len(prices) != len(volumes):
            raise ValueError("Price and volume lists must have the same length")

        vwap = []

        for i in range(len(prices)):
            if i == 0:
                vwap.append(prices[i])  # VWAP for the first data point is the price itself
            else:
                cumulative_price_volume = sum(prices[:i+1] * volumes[:i+1])
                cumulative_volume = sum(volumes[:i+1])
                vwap.append(cumulative_price_volume / cumulative_volume)

        return vwap


    def smavwapCross(sma, vwap):
        if sma[i] <= vwap[i]:
            return 1
        else:
            return 0



    def calcMACD(data, short_period=12, long_period=26, signal_period=9):
        short_ema = calcEMA(data, short_period)
        long_ema = calcEMA(data, long_period)

        macd = [short - long for short, long in zip(short_ema, long_ema)]

        signal = calcEMA(macd, signal_period)

        return macd, signal

    def calcEMA(data, period):
        multiplier = 2 / (period + 1)
        ema = [data[0]]

        for i in range(1, len(data)):
            ema_val = (data[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_val)

        return ema    

    def macdCross(macd, signal):
        if macd[i] > signal[i]:
            return 1
        else:
            return 0




    def calculate_knn_ma(price_values, ma_len):
        knn_ma = [np.mean(price_values[i-ma_len:i]) for i in range(ma_len, len(price_values))]
        return knn_ma

    def calculate_ema(price_values, ema_len):
        ema = np.zeros(len(price_values))
        ema[ema_len-1] = np.mean(price_values[:ema_len])
        multiplier = 2 / (ema_len + 1)

        for i in range(ema_len, len(price_values)):
            ema[i] = (price_values[i] - ema[i-1]) * multiplier + ema[i-1]



        return ema




    def calculate_knn_prediction(price_values, ma_len, num_closest_values=3, smoothing_period=20):
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

    def knnemaCross(knn_ma, ema_5, i):
        if i < len(knn_ma) and i < len(ema_5):
            if knn_ma[i] < ema_5[i]:
                return 1
        return 0

    ma_len = 5
    ema_len = 5

    total = []

    table_data = []

    x = 0
    buyPrice = 0
    buyTime = None
    sellPrice = 0
    sellTime = None
    profitArray = []

    buyPriceArray = []
    buyTimeArray = []
    sellPriceArray = []
    sellTimeArray = []
    tradeLengthArray = []



    for i in tqdm(range(len(historical_data[0]))):

        timestamp = historical_data[0].index[i]
        openPrice = historical_data[0]['Open'][i]
        high = historical_data[0]['High'][i]
        low = historical_data[0]['Low'][i]
        close = historical_data[0]['Close'][i]
        volume = historical_data[0]['Volume'][i]

        rsi = calcRSI(historical_data[0]['Close'])
        rsiEMA = calcRSIEMA(rsi)
        rsiemaX = rsiemaCross(rsi, rsiEMA)

        sma = calcSMA(historical_data[0]['Close'])
        vwap = calcVWAP(historical_data[0]['Close'], historical_data[0]['Volume'])
        smaemaX = smavwapCross(sma, vwap)


        macd, signal = calcMACD(historical_data[0]['Close'])
        macdX = macdCross(macd, signal)


        #add a supertrend indicator    

        knn_ma = calculate_knn_ma(historical_data[0]['Close'], ma_len)
        ema_5 = calculate_ema(historical_data[0]['Close'], ema_len)
        knnemaX = knnemaCross(knn_ma, ema_5, i)


        total.append(rsiemaX + macdX + smaemaX)
        totalDiff = total[i] - total[i - 1]


        if total[i] == 0 and x != 1:
            x = 1
            index = i
            buyPrice = close
            buyTime = timestamp
            buyPriceArray.append(buyPrice)
            buyTimeArray.append(buyTime)
        elif total[i] != 0 and x == 1:
            x = 0
            tradeIndex = i - index
            sellPrice = close
            sellTime = timestamp
            profit = sellPrice - buyPrice
            profitArray.append(profit)
            sellPriceArray.append(sellPrice)
            sellTimeArray.append(sellTime)
            tradeLengthArray.append(tradeIndex)




        table_data.append([i, timestamp, openPrice, high, low, close, volume, rsiemaX, smaemaX, macdX, knnemaX ,total[i], totalDiff ])
        progress = (i + 1) / len(historical_data[0]) * 100



    headers = ["Index", "Timestamp", "Open", "High", "Low", "Close", "Volume", "RSIEMA Cross", "SMA/VWAP Cross", "MACD Cross", "KNN EMA X","Total", "Total Diff"]

    table = (tabulate(table_data, headers, tablefmt="grid"))
    print(table , "\n")

    if x == 1:
        print("Still holding position")
        print("Buy Price: ", buyPrice, " Buy Date: ", buyTime , "\n")

    headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit", "Trade Length"]
    data = [buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray, tradeLengthArray]
    print(tabulate(zip(*data), headers=headers))

    print("Total Profit: ", sum(profitArray), "Median Trade Length: ", np.median(tradeLengthArray), "Average Trade Length: ", np.mean(tradeLengthArray))



    # Define the output file name
    output_folder = "historicalData"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    ticker_folder = os.path.join(output_folder, ticker_symbol[0])
    if not os.path.exists(ticker_folder):
        os.mkdir(ticker_folder)

    period_folder = os.path.join(ticker_folder, period)
    if not os.path.exists(period_folder):
        os.mkdir(period_folder)

    output_file = f"{ticker_symbol[0]}_{period}_{pd.Timestamp.now().strftime('%Y-%m-%d')}.txt"
    output_path = os.path.join(period_folder, output_file)

    # Save the output to the text file
    with open(output_path, "w") as file:
        file.write("Trading Output for " + ticker_symbol[0] + " for the last 300 days\n")

        file.write(table + "\n\n")
        if x == 1:
            file.write("Still holding position\n")
            file.write("Buy Price: " + str(buyPrice) + " Buy Date: " + str(buyTime) + "\n\n")
        file.write(tabulate(zip(*data), headers=headers) + "\n")
        file.write("Total Profit: " + str(sum(profitArray)) + " Median Trade Length: " + str(np.median(tradeLengthArray)) + " Average Trade Length: " + str(np.mean(tradeLengthArray)) + "\n")


# Start the program with the initial end_date
end_date = "2023-11-11"
run_program(end_date)

# Allow the user to increment the end_date and rerun the program by pressing enter
while True:
    input_str = input("Press enter to increment the end date and rerun the program, or type 'exit' to quit: ")
    if input_str.lower() == "exit":
        break
    end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    run_program(end_date)


#Add the actual knn prediction script
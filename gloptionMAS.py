import yfinance as yf
import numpy as np
from tabulate import tabulate

def ema_greater_than_knn(ema, knn_ma):
    if ema > knn_ma:
        return 1
    else:
        return 0


ma_len = 5
ema_len_5 = 5



def calculate_ema(price_values, ema_len):
    ema = np.zeros(len(price_values))
    ema[ema_len-1] = np.mean(price_values[:ema_len])
    multiplier = 2 / (ema_len + 1)
    
    for i in range(ema_len, len(price_values)):
        ema[i] = (price_values[i] - ema[i-1]) * multiplier + ema[i-1]


    return ema


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


#Ticker Detailss
historical_data = []
tradeOpen = False
buyPrice = 0
sellPrice = 0
buyTime = None
sellTime = None

buyPriceArray = []
sellPriceArray = []
buyTimeArray = []
sellTimeArray = []
profitArray = []
positive = []
negative = []
profit_by_year = {}


ticker = yf.Ticker('aapl')
start_date = "2010-11-05"
end_date = "2023-11-11"
data = ticker.history(start=start_date, end=end_date, interval="1d")
historical_data.append(data)






for i in range(len(historical_data)):
    historical_data[i]['EMA_5'] = calculate_ema(historical_data[i]['Close'], ema_len_5)
    historical_data[i]['KNN_MA'] = calculate_knn_ma(historical_data[i]['Close'], ma_len)


    table = []

    for index, row in historical_data[i].iterrows():
        
        date = index
        open_price = row['Open']
        close_price = row['Close']
        volume = row['Volume']
        ema = row['EMA_5']
        knn_ma = row['KNN_MA']


        if ema != None and knn_ma != None:
            KnnEmaX = ema_greater_than_knn(ema, knn_ma)
        else:
            KnnEmaX = None



        if (KnnEmaX == 1) and (tradeOpen == False):
            buyPrice = close_price
            buyTime = date
            tradeOpen = True
            print("Buy at: ", buyPrice, "on: ", buyTime)
        elif ((KnnEmaX == 0) and (tradeOpen == True)):
            sellPrice = close_price
            sellTime = date
            tradeOpen = False
            print("Sell at: ", sellPrice, "on: ", sellTime)
            profit = sellPrice - buyPrice
            print("Profit: ", profit)
            buyPriceArray.append(buyPrice)
            sellPriceArray.append(sellPrice)
            buyTimeArray.append(buyTime)
            sellTimeArray.append(sellTime)
            profitArray.append(profit)

            if profit > 0:
                positive.append(profit)
            else:
                negative.append(profit)


            # Record profit by year
            year = index.year
            if year not in profit_by_year:
                profit_by_year[year] = []
            profit_by_year[year].append(profit)







        table.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX])

header = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX']
print(tabulate(table, headers=header, tablefmt='orgtbl'))



print("\n")
headers = ["Buy Price", "Buy Time", "Sell Price", "Sell Time", "Profit"]
data = list(zip(buyPriceArray, buyTimeArray, sellPriceArray, sellTimeArray, profitArray))
print(tabulate(data, headers=headers))
print("Total Profit: ", sum(profitArray))
print("Total Trades: ", len(profitArray))
print("Positive Trades: ", len(positive))
print("Negative Trades: ", len(negative))
print("Suceess Rate: ", len(positive)/len(profitArray)*100, "%")

#for year in profit_by_year:
 #   print("Year", year, "Profit:", sum(profit_by_year[year]))












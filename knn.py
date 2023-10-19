from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def predictNextPriceKNN(prices, volumes, k=3):
    if len(prices) != len(volumes):
        raise ValueError("Price and volume lists must have the same length")

    # Create a feature matrix by stacking prices and volumes as columns
    feature_matrix = np.column_stack((prices, volumes))

    # Initialize the k-NN regressor
    knn_regressor = KNeighborsRegressor(n_neighbors=k)

    # Fit the model to your data
    knn_regressor.fit(feature_matrix, prices)

    # Predict the next price (assuming you have historical data up to the last point)
    last_price = prices[-1]
    last_volume = volumes[-1]
    next_feature = np.array([last_price, last_volume]).reshape(1, -1)
    predicted_next_price = knn_regressor.predict(next_feature)

    return predicted_next_price[0]

# Example usage:
prices = [10, 12, 14, 16, 18, 20, 22]
volumes = [1000, 1500, 2000, 1800, 2500, 2200, 3000]
k = 3
predicted_next_price = predictNextPriceKNN(prices, volumes, k)
print(predicted_next_price)

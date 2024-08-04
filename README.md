# 📊 Stock Market Analysis & Prediction using GRU 📈

## Introduction
In this project, I analyze stock market data and predict future stock prices using a Gated Recurrent Unit (GRU) neural network. The GRU is a variant of recurrent neural networks that is particularly effective for time series forecasting tasks like stock market prediction.

I'll be answering the following questions along the way:

- What was the change in price of the stock over time?
- What was the daily return of the stock on average?
- What was the moving average of the various stocks?
- What was the correlation between different stocks'?
- How much value do we put at risk by investing in a particular stock?
- How can we attempt to predict future stock behavior? (Predicting the closing price stock price of APPLE inc using GRU)

## Data Exploration and Preprocessing
The dataset used in this project includes historical stock prices with features such as open, high, low, close prices, and trading volume. Data preprocessing steps include handling missing values, normalizing the data, and creating sequences of data points to be used as inputs for the GRU model.

```python
# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)


company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)
```
### Train/Test Split
<img width="600" alt="image" src="https://github.com/user-attachments/assets/78859fb1-7862-42e3-ac75-9f3b4b50b336">

## Model Building and Training
The GRU model is constructed using deep learning frameworks, with careful consideration of hyperparameters such as the number of GRU layers, the number of units per layer, and the dropout rate. The model is trained on the processed dataset, with the aim of minimizing the mean squared error between the predicted and actual stock prices.
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout

# Build the LSTM model
model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(GRU(64, return_sequences=False))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(1))

model.summary()
```
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3a53b2ee-728d-428b-87ff-d1505ba2f07d">


## Prediction and Results
After training, the model is evaluated on a test set to assess its performance. The results demonstrate the model's ability to capture the trends in stock prices, although, as with any financial model, there are limitations due to the inherent unpredictability of markets.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/97123cd4-6838-4643-b347-9b44af57087c">


## Conclusion
The GRU-based model provides a useful tool for stock price prediction, leveraging historical data to forecast future trends. While the model shows promising results, further improvements could be made by integrating additional market indicators or using more advanced neural network architectures.

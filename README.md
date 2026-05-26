# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 26-05-2026

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/content/tea_vs_coffee_global_final.csv')
data['year'] = pd.to_datetime(data['year'])
plt.plot(data['year'], data['cups_per_day'])
plt.xlabel('year')
plt.ylabel('Cups per Day')
plt.title('Cups Time Series')
plt.show()
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
check_stationarity(data['cups_per_day'])

plot_acf(data['cups_per_day'])
plt.show()
plot_pacf(data['cups_per_day'])
plt.show()
sarima_model = SARIMAX(data['cups_per_day'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
sarima_result = sarima_model.fit()
train_size = int(len(data) * 0.8)
train, test = data['cups_per_day'][:train_size], data['cups_per_day'][train_size:]
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
plt.plot(data['year'].loc[test.index], test, label='Actual')
plt.plot(data['year'].loc[test.index], predictions, color='red', label='Predicted')
plt.xlabel('year')
plt.ylabel('Cups per Day')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:
<img width="798" height="681" alt="image" src="https://github.com/user-attachments/assets/0055bdc1-9395-4be0-97ad-779d7fb6e684" />

<img width="804" height="568" alt="image" src="https://github.com/user-attachments/assets/4ec9eec1-6d07-4aaa-a9cd-8c9da7f1d258" />


### RESULT:
Thus the program run successfully based on the SARIMA model.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Unknown_Stock_Data.csv")
adjClose = data['Adj Close']
movingAvg = []
window = 14
for i in range(len(adjClose)):
    movingAvg.append(sum((adjClose[i-window : i]))/window)
movingAvg[-1]
last14 = adjClose[len(adjClose) - 14: len(adjClose)]
last14 = last14.dropna()
movingAvgPred = (movingAvg[-1] * 14) - sum(last14)
xVal = adjClose.shift(1)[1:len(adjClose) - 1]
yVal = adjClose[1:-1]
df = pd.concat([xVal, yVal], axis = 1)
model = LinearRegression()
model.fit(np.array(xVal).reshape(-1, 1), yVal)
predVal = model.predict(np.array(yVal[-1:]).reshape(-1, 1))
lag1 = adjClose.shift(1)[3:len(adjClose) - 1]
lag2 = adjClose.shift(2)[3:len(adjClose) - 1]
lag3 = adjClose.shift(3)[3:len(adjClose) - 1]
xVal1 = pd.concat([lag1, lag2, lag3], axis = 1)
yVal1 = adjClose[3:-1]
df = pd.concat([xVal1, yVal1], axis = 1)
model1 = LinearRegression()
model1.fit(np.array(xVal1).reshape(-1, 3)[0:len(xVal1) - 1], yVal[0:len(yVal1) - 1])
predInputs = xVal1[-1:]
predVal3 = model1.predict(np.array(predInputs).reshape(-1, 3))
print("Regression Prediction: " + str(float(predVal)))
print("Three Lag Regression Prediction: " + str(float(predVal3)))
print("Moving Average Prediction: " + str(movingAvgPred))

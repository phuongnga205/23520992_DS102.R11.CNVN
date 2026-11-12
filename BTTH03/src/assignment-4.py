import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/forestfires.csv")

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
df['month'] = df['month'].map(month_map)
df['day'] = df['day'].map(day_map)

df = df.drop(columns=['DC'])

X_data = df.drop(columns=['area']).values
y_data = df['area'].values.reshape(-1, 1)

np.random.seed(42)
indices = np.arange(len(X_data))
np.random.shuffle(indices)

train_size = int(len(X_data) * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X_data[train_idx]
y_train = y_data[train_idx]
X_test = X_data[test_idx]
y_test = y_data[test_idx]

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
sigma[sigma == 0] = 1e-15

X_train_std = (X_train - mu) / sigma
X_test_std = (X_test - mu) / sigma

model = LinearRegression()
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("="*10 + " ASSIGNMENT 4: SKLEARN LINEAR REGRESSION " + "="*10)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
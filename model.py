import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('gld_price_data.csv')

print(f"Data shape: {data.shape}") # number of rows and columns
print(data.info())
print(f"\nNull object num: \n{data.isnull().sum()}")
print(f"\nStatistical values:\n {data.describe()}")


data['Date'] = pd.to_datetime(data['Date'])
plt.plot(data['Date'], data['GLD'], color="Blue")
plt.xlabel("Year")
plt.ylabel("Gold Price")
plt.show()

# Checking the correlation to understand the relationship between features
data_for_cor = data.drop(['Date'], axis=1)
corr = data_for_cor.corr()
plt.figure(figsize = (8,8))
sns.heatmap(corr, annot=True, cmap="Blues")
plt.show()

# Splitting the data for feature and target values
features = data.drop(["Date", "GLD"], axis=1) # axis=1 to drop a column, axis=0 to drop a row
targets = data['GLD']

# Splitting the data for training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, targets, test_size=0.3,random_state=2)

# Model
model = RandomForestRegressor()
model.fit(features_train, labels_train)
prediction = model.predict(features_test)


r2score = r2_score(labels_test, prediction)
print(f"r2 score is: {r2score}")


# Displaying the result
labels_test = list(labels_test)
plt.plot(labels_test, color="Green", label="Actual Value")
plt.plot(prediction, color="Blue", label="Predicted Value")
plt.xlabel("Number of values")
plt.ylabel("Gold Price")
plt.legend()
plt.show()


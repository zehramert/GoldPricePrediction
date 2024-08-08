import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('gld_price_data.csv')

data = data.drop(['SPX','USO','SLV','EUR/USD'], axis=1)

data['Date'] = pd.to_datetime(data['Date'])
# Extract useful features from the 'Date' column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Lag1'] = data['GLD'].shift(1)
data['Lag2'] = data['GLD'].shift(2)
# Prepare features and target
features = data.drop(columns=['Date', 'GLD'])
target = data['GLD']



# Train test split
features_train, features_test, labels_train, labels_test = train_test_split(features, target, test_size=0.3,random_state=2)

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


# Predicting user's input
import re  #re module is to split a string by multiple delimiters
new_data = input("Enter a day, month and year to predict the gold price:")
day, month, year = map(int,re.split(r"[/.]", new_data)) # Ensures that the new data splitted and converted to int
user_data = pd.DataFrame(
    {
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Lag1' :[data['GLD'].iloc[-1]],
        'Lag2': [data['GLD'].iloc[-2]]
    }
)

pred = model.predict(user_data)
print(f"Gold price prediction for the date {new_data} is: {pred[0]}")


# Gold Price Prediction for the next 10 years:
year1 = 2025
year2 = 2035
feature_year_count = len(list(range(year1,year2)))
future_data = pd.DataFrame({
    'Year': list(range(year1,year2)),
    'Month': [1] * feature_year_count,
    'Day': [1] * feature_year_count,
    'Lag1': [data['GLD'].iloc[-1]] * feature_year_count,
    'Lag2': [data['GLD'].iloc[-2]] * feature_year_count
})

future_predictions = []
for i in range(feature_year_count):
    future_data.loc[i, 'Lag1'] = future_predictions[i - 1] if i > 0 else data['GLD'].iloc[-1]
    future_data.loc[i, 'Lag2'] = future_predictions[i - 2] if i > 1 else data['GLD'].iloc[-2]
    pred = model.predict(future_data.iloc[i:i+1])
    future_predictions.append(pred[0])

# Print and Plot future predictions
future_data['Predicted_GLD'] = future_predictions
print(future_data)
# Plot future predictions
plt.plot(future_data['Year'], future_predictions, color="Green", label="Gold Price Prediction")
plt.xlabel("Years")
plt.ylabel("Gold Price")
plt.legend()
plt.show()


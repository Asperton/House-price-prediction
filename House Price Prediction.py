import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = {
    'Area': [500, 750, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
    'Price': [1500000, 2200000, 3000000, 3500000, 4500000, 5200000, 6000000, 7500000, 9000000]
}
df = pd.DataFrame(data)
X = df[['Area']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)
area = float(input("Enter house area (sq ft): "))
predicted_price = model.predict([[area]])
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price (₹)")
plt.title("House Price Prediction")
plt.legend()
plt.show()

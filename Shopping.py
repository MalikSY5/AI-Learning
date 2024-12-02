import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv('shopping_trends.csv')
#print(data.head())

# Find missing values
#print(data.isnull())

#plot points
# plt.scatter(data['Category'], data['Review'])
# plt.xlabel('Category')
# plt.ylabel('Season')
# plt.title('Category per season')
# plt.show()

#Preparing Data
X = data[['Age', 'Previous Purchases']]
y = data['Purchase Amount (USD)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train LR Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#Evaluate Model
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')
plt.title('Actual vs Predicted Spending')
plt.show()

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_})
print(feature_importance.sort_values(by='Importance', ascending=False))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load dataset
df = pd.read_csv('/storage/emulated/0/Download/Car_Details.csv')

# Step 2: Rename and clean (if needed)
df.rename(columns={
    'Driven_kms': 'Kms_Driven',
    'Selling_type': 'Selling_Type'  # just for consistency
}, inplace=True)

# Step 3: Data info
print("Columns:", df.columns)
print(df.head())

# Step 4: Preprocessing
df['Current_Year'] = 2025
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Year', 'Car_Name', 'Current_Year'], axis=1, inplace=True)

# Step 5: Handle categorical data
df = pd.get_dummies(df, drop_first=True)

# Step 6: Split features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Step 7: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict & evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Optional: Plot prediction vs actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Price')
plt.grid()
plt.show()
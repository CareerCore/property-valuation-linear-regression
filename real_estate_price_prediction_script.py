"""
Real Estate Price Prediction (ML)
Exploratory Data Analysis and Linear Regression modeling on residential housing data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use("seaborn-v0_8")

# ---------------------------------------------------------
# 1. Data Extraction & Inspection
# ---------------------------------------------------------
DATA_PATH = "housing_data.csv"  # Update this path to your local dataset file
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("\nDataset Info:")
print(df.info())

# ---------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------
df.dropna(inplace=True)

# ---------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
# Pearson Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), vmax=0.9, cmap="viridis")
plt.title("Pearson Correlation")
plt.tight_layout()
plt.show()

# Suburb Distribution
plt.figure(figsize=(10, 8))
df["Suburb"].iloc[0:900].value_counts().sort_values().plot(
    kind="barh", color="dodgerblue"
)
plt.xlabel("Count")
plt.ylabel("Suburb")
plt.title("Property Distribution by Suburb")
plt.tight_layout()
plt.show()

# Top Sellers
plt.figure(figsize=(8, 6))
sns.countplot(x=df["SellerG"].iloc[0:12])
plt.title("Property Count by Seller")
plt.tight_layout()
plt.show()

# Property Type Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=df["Type"], palette="Set1")
plt.title("Distribution of Property Types")
plt.tight_layout()
plt.show()

# Property Type vs. Price
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Type", y="Price", palette="Set2")
plt.title("Property Type vs. Price")
plt.tight_layout()
plt.show()

# Sale Method Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=df["Method"])
plt.title("Distribution of Sale Methods")
plt.tight_layout()
plt.show()

# Sale Method vs. Price
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Method", y="Price", palette="Set1")
plt.title("Sale Method vs. Price")
plt.tight_layout()
plt.show()

# Council Area Distribution
plt.figure(figsize=(10, 8))
df["CouncilArea"].iloc[0:2000].value_counts().sort_values().plot(
    kind="barh", color="darkorange"
)
plt.xlabel("Count")
plt.ylabel("Council Area")
plt.title("Property Distribution by Council Area")
plt.tight_layout()
plt.show()

# Region Distribution
plt.figure(figsize=(8, 6))
chart = sns.countplot(x=df["Regionname"], palette="Set2")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Property Distribution by Region")
plt.tight_layout()
plt.show()

# Region vs. Price
plt.figure(figsize=(8, 6))
chart = sns.boxplot(data=df, x="Regionname", y="Price")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.title("Region vs. Price")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Feature Encoding & Train-Test Split
# ---------------------------------------------------------
for col in df.columns.tolist():
    df[col] = df[col].astype("category").cat.codes

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print("\nTrain/Test Shapes:")
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)

# ---------------------------------------------------------
# 5. Linear Regression Model & Evaluation
# ---------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

r2_score = lr.score(X_test, y_test)
y_pred = lr.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nR^2 Score: {r2_score:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Actual vs. Predicted Plot
plt.figure(figsize=(8, 6))
plt.title("Linear Regression Model: Actual vs. Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(y_test, y_pred, "o")
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m * y_test + b)
plt.tight_layout()
plt.show()

# Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df["Price"], kde=True, color="g")
plt.title("Distribution of Encoded Price")
plt.tight_layout()
plt.show()

# Feature Histograms
df.hist(figsize=(15, 13), color="r")
plt.tight_layout()
plt.show()
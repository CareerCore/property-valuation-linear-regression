# Real Estate Price Prediction (ML)

A machine learning project that performs exploratory data analysis (EDA) on residential housing market data and predicts property prices using a **Linear Regression** model in Python.

## Dataset Overview

* **Format:** Structured CSV housing dataset (`housing_data.csv`)
* **Initial Size:** 13,580 rows $\times$ 21 columns
* **Key Features:** `Suburb`, `Rooms`, `Type`, `Method`, `SellerG`, `Distance`, `Postcode`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `CouncilArea`, `Regionname`, and `Propertycount`.

## Project Workflow

1. **Data Extraction & Inspection:** Loaded the dataset using Pandas and inspected its shape, summary statistics, and data types.
2. **Data Cleaning:** Removed rows containing missing values (`dropna()`) and encoded columns into numeric category codes for modeling.
3. **Exploratory Data Analysis (EDA):**
   * Pearson correlation heatmap across numerical features.
   * Distribution of properties by Suburb, Seller, Property Type, Sale Method, Council Area, and Region.
   * Boxplots comparing `Price` against `Type`, `Method`, and `Regionname`.
   * Target price distribution plot and feature histograms.
4. **Model Training & Evaluation:**
   * Split the cleaned dataset into **70% training** (4,337 rows) and **30% testing** (1,859 rows) sets.
   * Trained a Scikit-learn `LinearRegression` model and plotted actual vs. predicted values with a regression line.

## Model Performance

| Metric | Value |
| :--- | :--- |
| **$R^2$ Score** | `0.7975` |
| **Mean Absolute Error (MAE)** | `145.55` |
| **Mean Squared Error (MSE)** | `33717.01` |

## Technologies Used

* **Python 3**
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/CareerCore/real-estate-price-prediction-ml.git
   ```
2. Install the required libraries:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn jupyter
   ```
3. Place your housing dataset CSV in the project directory and update the file path in `real-estate-price-prediction.ipynb` (`pd.read_csv(...)`) to match its local path.
4. Launch Jupyter Notebook and open `real-estate-price-prediction.ipynb`:
   ```bash
   jupyter notebook
   ```
# Movies Rating Regression

## 📌 Project Overview

**Movies Rating Regression** is a machine learning regression project that uses movie-related information to predict a movie's **IMDb rating**.

The project is based on a dataset containing **500 movies** and combines numerical and categorical features such as release year, runtime, budget, number of votes, Metascore, genre, and language.

### 🎯 Objective

The main objective is to build a regression model that can estimate:

> **Target variable: `IMDb_Rating`**

The project can be used to practice the complete machine learning workflow:

1. Load and inspect the dataset
2. Perform exploratory data analysis (EDA)
3. Clean and preprocess the data
4. Encode categorical variables
5. Split data into training and testing sets
6. Train regression models
7. Evaluate model performance
8. Compare different regression algorithms
9. Use the trained model to predict IMDb ratings

---

## 📂 Dataset

**File:** `movies_rating_regression_500.csv`

### Dataset summary

| Property | Value |
|---|---:|
| Number of rows | 500 |
| Number of columns | 9 |
| Target column | `IMDb_Rating` |
| Numerical columns | 6 |
| Categorical columns | 2 |
| ID column | 1 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Columns

| Column | Type | Description |
|---|---|---|
| `Movie_ID` | Integer | Unique identifier for each movie |
| `Release_Year` | Integer | Year in which the movie was released |
| `Runtime_Min` | Integer | Movie duration in minutes |
| `Budget_Million` | Integer | Movie budget in million units |
| `Votes` | Integer | Number of votes received |
| `Metascore` | Integer | Metacritic score |
| `Genre` | Categorical | Movie genre |
| `Language` | Categorical | Movie language |
| `IMDb_Rating` | Float | IMDb rating; this is the prediction target |

---

## 📊 Dataset Characteristics

The dataset contains:

- **500 movie records**
- **10 unique genres**
- **6 unique languages**
- Release years ranging from **2000 to 2024**
- Runtime ranging from **80 to 180 minutes**
- Budget ranging from **5 to 200 million**
- Metascore ranging from **35 to 95**
- IMDb ratings ranging from **4.5 to 8.3**

The categorical features are:

- `Genre`
- `Language`

The numerical predictors are:

- `Release_Year`
- `Runtime_Min`
- `Budget_Million`
- `Votes`
- `Metascore`

`Movie_ID` is an identifier and generally should not be used as a predictive feature.

---

## 🧠 Machine Learning Problem

This is a **supervised regression problem**.

### Input Features

```text
Release_Year
Runtime_Min
Budget_Million
Votes
Metascore
Genre
Language
```

### Target

```text
IMDb_Rating
```

The model learns the relationship between the movie attributes and its IMDb rating.

---

## 🔄 Recommended ML Workflow

### 1. Load the data

```python
import pandas as pd

df = pd.read_csv("movies_rating_regression_500.csv")
print(df.head())
print(df.info())
```

### 2. Check data quality

```python
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
```

### 3. Separate features and target

```python
X = df.drop(columns=["IMDb_Rating", "Movie_ID"])
y = df["IMDb_Rating"]
```

`Movie_ID` is removed because it is an identifier rather than a meaningful movie characteristic.

### 4. Encode categorical features

For `Genre` and `Language`, one-hot encoding can be used:

```python
X = pd.get_dummies(X, columns=["Genre", "Language"], drop_first=True)
```

### 5. Train-test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates an 80/20 training-testing split.

### 6. Train a regression model

Example using Linear Regression:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 7. Make predictions

```python
y_pred = model.predict(X_test)
```

### 8. Evaluate the model

Recommended regression metrics:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
```

---

## 🤖 Models You Can Try

To compare model performance, the following regression algorithms can be tested:

### Linear Regression

Simple baseline model that assumes a linear relationship between features and the target.

### Ridge Regression

Linear regression with L2 regularization. Useful when there are many correlated features after encoding.

### Lasso Regression

Linear regression with L1 regularization. It can also perform feature selection by shrinking some coefficients toward zero.

### Random Forest Regressor

An ensemble of decision trees that can capture non-linear relationships.

### Gradient Boosting Regressor

Builds trees sequentially to improve prediction errors from previous trees.

### Extra Trees Regressor

Uses randomized decision trees and can be useful for capturing complex feature relationships.

---

## 📈 Exploratory Data Analysis

Before training models, useful visualizations include:

### IMDb Rating Distribution

```python
import matplotlib.pyplot as plt

df["IMDb_Rating"].hist()
plt.xlabel("IMDb Rating")
plt.ylabel("Number of Movies")
plt.title("Distribution of IMDb Ratings")
plt.show()
```

### Metascore vs IMDb Rating

```python
plt.scatter(df["Metascore"], df["IMDb_Rating"])
plt.xlabel("Metascore")
plt.ylabel("IMDb Rating")
plt.title("Metascore vs IMDb Rating")
plt.show()
```

### Votes vs IMDb Rating

```python
plt.scatter(df["Votes"], df["IMDb_Rating"])
plt.xlabel("Votes")
plt.ylabel("IMDb Rating")
plt.title("Votes vs IMDb Rating")
plt.show()
```

### Average Rating by Genre

```python
df.groupby("Genre")["IMDb_Rating"].mean().sort_values().plot(kind="bar")
plt.ylabel("Average IMDb Rating")
plt.title("Average IMDb Rating by Genre")
plt.show()
```

---

## 📁 Suggested Project Structure

```text
movies-rating-regression/
│
├── data/
│   └── movies_rating_regression_500.csv
│
├── notebooks/
│   └── movie_rating_regression.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── README.md
└── requirements.txt
```

The exact structure can be simplified if the project is being developed only as a Jupyter Notebook.

---

## 🛠️ Requirements

Recommended Python packages:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

Install them with:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Or create a `requirements.txt` file:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

Then run:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1 — Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run the cells from top to bottom.

### Option 2 — Python Script

If the project is implemented as a Python script:

```bash
python train.py
```

Make sure the CSV file is available at the expected path.

---

## 🔍 Model Evaluation

When comparing regression models, focus on the following:

| Metric | Interpretation |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Penalizes larger errors more heavily |
| RMSE | Error metric in the same unit as IMDb rating |
| R² | Proportion of target variance explained by the model |

For this project, the metrics should be calculated on the **test set** so that model performance is measured on previously unseen data.

---

## ⚠️ Important Notes

- `Movie_ID` is an identifier and should normally be excluded from model training.
- `Genre` and `Language` are categorical variables and need encoding before most scikit-learn regression models can use them.
- Numerical features may have very different scales. Scaling can be useful for models such as Linear Regression, Ridge, and Lasso.
- Tree-based models generally do not require feature scaling.
- Model performance should be compared using the same train/test split or a consistent cross-validation strategy.
- Avoid data leakage by fitting preprocessing steps only on the training data.

For a production-quality implementation, a scikit-learn `Pipeline` and `ColumnTransformer` are recommended.

---

## 🔬 Recommended Advanced Version

A more robust implementation can use a preprocessing pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

numeric_features = [
    "Release_Year",
    "Runtime_Min",
    "Budget_Million",
    "Votes",
    "Metascore"
]

categorical_features = [
    "Genre",
    "Language"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge())
    ]
)
```

This approach keeps preprocessing and model training together and helps reduce the risk of preprocessing mistakes.

---

## 📌 Project Status

**Dataset:** Ready  
**Problem Type:** Regression  
**Target:** `IMDb_Rating`  
**Records:** 500  
**Features:** Movie metadata and audience/critic indicators  
**Recommended Next Step:** Perform EDA, train multiple regression models, compare evaluation metrics, and select a model based on the project's stated evaluation criteria.

---

## 👨‍💻 Author

Add your name, GitHub profile, and project details here.

```text
Author: Your Name
GitHub: https://github.com/your-username
Project: Movies Rating Regression
```

---

## 📄 License

Add the appropriate license for your project if you plan to publish it publicly.

Example:

```text
MIT License
```

---

## ⭐ Project Summary

This project demonstrates how movie metadata can be used to build a **regression model for IMDb rating prediction**. It covers the key stages of a typical machine learning workflow: data inspection, preprocessing, categorical encoding, model training, evaluation, and model comparison.

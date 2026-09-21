# Movies Rating Regression

##  Project Overview

Movies Rating Regression is a machine learning project that predicts a movie's **IMDb rating** using different movie-related features.

The dataset contains **500 movies** with information such as release year, runtime, budget, votes, Metascore, genre, and language.

The project follows a complete machine learning workflow including data analysis, preprocessing, regression modeling, evaluation, and comparison.

---

##  Objectives

The main objectives of this project are:

* To analyze movie-related data.
* To identify factors related to IMDb ratings.
* To preprocess numerical and categorical data.
* To build regression models for predicting IMDb ratings.
* To evaluate model performance using standard regression metrics.
* To compare different regression algorithms.
* To understand how machine learning can be applied to movie rating prediction.

---

##  Dataset Description

**Dataset:** `movies_rating_regression_500.csv`

The dataset contains **500 movie records** and **9 columns**.

| Feature        | Description                     |
| -------------- | ------------------------------- |
| Movie_ID       | Unique ID of the movie          |
| Release_Year   | Year of movie release           |
| Runtime_Min    | Movie duration in minutes       |
| Budget_Million | Movie budget in million units   |
| Votes          | Number of votes received        |
| Metascore      | Metacritic score                |
| Genre          | Movie genre                     |
| Language       | Movie language                  |
| IMDb_Rating    | IMDb rating and target variable |

### Dataset Characteristics

* Records: **500**
* Genres: **10**
* Languages: **6**
* Release years: **2000–2024**
* Runtime: **80–180 minutes**
* Budget: **5–200 million**
* Metascore: **35–95**
* IMDb Rating: **4.5–8.3**
* Missing values: **0**
* Duplicate rows: **0**

---

##  Problem Statement

The objective of this project is to develop a machine learning regression model that can predict the **IMDb rating of a movie** based on its available characteristics.

The target variable is:

**IMDb_Rating**

The input features include:

* Release Year
* Runtime
* Budget
* Votes
* Metascore
* Genre
* Language

`Movie_ID` is treated as an identifier and is not used as a predictive feature.

---

##  Data Preprocessing

The following preprocessing steps are performed:

1. Dataset is loaded and inspected.
2. Missing values are checked.
3. Duplicate records are checked.
4. The target variable is separated from the input features.
5. `Movie_ID` is removed because it is only an identifier.
6. Categorical features such as Genre and Language are converted into numerical form.
7. The dataset is divided into training and testing sets.
8. Numerical features can be scaled when required by the selected model.

The dataset uses an **80:20 training-testing split**.

---

##  Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset and identify relationships between variables.

The main visualizations include:

* IMDb Rating Distribution
* Metascore vs IMDb Rating
* Votes vs IMDb Rating
* Average IMDb Rating by Genre
* Box Plot for numerical features
* Comparison of actual and predicted ratings

These visualizations help understand the distribution of ratings and relationships between movie characteristics and IMDb ratings.


##  Model Evaluation

The models are evaluated using the following metrics:

| Metric   | Description                                                           |
| -------- | --------------------------------------------------------------------- |
| MAE      | Measures the average absolute prediction error                        |
| MSE      | Measures the average squared prediction error                         |
| RMSE     | Measures prediction error in the same unit as IMDb rating             |
| R² Score | Measures how much variation in IMDb ratings is explained by the model |

The models are evaluated using the **test dataset** so that their performance can be measured on previously unseen data.

---

##  Expected Analysis

The project analyzes how different movie characteristics relate to IMDb ratings.

Important factors considered include:

* Metascore
* Number of Votes
* Movie Runtime
* Budget
* Release Year
* Genre
* Language

The regression models are compared using their evaluation metrics to understand their prediction performance.


---

##  Project Outcome

This project demonstrates how machine learning regression techniques can be used to predict movie ratings.

It provides practical experience in:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Categorical data handling
* Regression modeling
* Model evaluation
* Data visualization
* Machine learning workflow

---

##  Limitations

The project has some limitations:

* The dataset contains only **500 movies**.
* IMDb ratings may depend on many factors that are not included in the dataset.
* The dataset may not represent all movies or audiences.
* Genre and language alone may not fully explain rating differences.
* Regression models cannot perfectly predict subjective human ratings.
* The results depend on the quality and characteristics of the available dataset.
* The project is intended mainly for academic and learning purposes.

---

##  Future Scope

The project can be improved in the future by:

* Increasing the dataset size.
* Adding more movie features.
* Including actors, directors, production companies, and reviews.
* Using advanced ensemble models.
* Applying cross-validation and hyperparameter tuning.
* Deploying the model as a web application.
* Creating an interactive movie rating prediction system.


##  Tools and Technologies

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn


---

## Conclusion

The Movies Rating Regression project demonstrates the application of supervised machine learning to predict IMDb movie ratings.

The project covers the complete workflow from **data collection and preprocessing to exploratory analysis, model training, evaluation, and interpretation**.

By comparing different regression approaches, the project provides an understanding of how movie-related features can be used to estimate IMDb ratings.

Overall, the project provides practical knowledge of **regression analysis, data visualization, preprocessing, and machine learning model evaluation**.



## Project Summary

**Movies Rating Regression** is a machine learning project that uses movie metadata to predict IMDb ratings. The project demonstrates data preprocessing, exploratory data analysis, visualization, regression modeling, and model evaluation using a dataset of 500 movies.

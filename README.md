CODECRAFT_ML_01: House Price Prediction using Linear Regression

Objective :

This project is part of my Machine Learning internship at CODE CRAFT.  
The goal is to build a **Linear Regression model** to predict house prices based on:

- Square footage ('GrLivArea')
- Number of bedrooms ('BedroomAbvGr')
- Number of full bathrooms ('FullBath')

Dataset:

The dataset used is from the [Kaggle House Prices: Advanced Regression Techniques competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

- File used: 'train.csv'
- Total features in dataset: 163
- Features selected for this project:
  - 'GrLivArea' (Above grade living area square feet)
  - 'BedroomAbvGr' (Bedrooms above ground)
  - 'FullBath' (Full bathrooms above grade)
  - 'SalePrice' (Target variable)

Steps Performed:

> Loaded data from 'train.csv' 
> Selected relevant features  
> Checked for missing values  
> Split data into training and testing sets  
> Built a Linear Regression model  
> Evaluated model performance  
> Visualized predictions

How to Run:

1. Clone this repository:
   git clone https://github.com/your-username/CODECRAFT_ML_01.git

2. Place 'train.csv' file in the same directory as the notebook/script.

3. Install required Python packages:
   pip install pandas numpy matplotlib seaborn scikit-learn

4. Run the Python script 'task1.py' or Jupyter notebook to train and evaluate the model.
---

Results:

- R² Score: 0.6341  
- RMSE: 52,975.71

This model explains approximately 63% of the variance in house prices using only three features. Including more features from the dataset could further improve accuracy.

Sample Output:

   GrLivArea  FullBath  BedroomAbvGr  SalePrice
0       1710         2             3     208500
1       1262         2             3     181500
2       1786         2             3     223500
3       1717         1             3     140000
4       2198         2             4     250000
GrLivArea       0
FullBath        0
BedroomAbvGr    0
SalePrice       0
dtype: int64
R² Score: 0.6341189942328371
RMSE: 52975.71771338122

Author:

- Name: Sanjaya S  
- Internship: Machine Learning Intern at CODE CRAFT


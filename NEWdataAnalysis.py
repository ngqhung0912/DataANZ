# IMPORTING LIBRARY ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# READ INPUT ----------------------------------------
path = "/Users/macbook/Learning/DataANZ/ANZData.xlsx"
ANZ_df = pd.read_excel(path)
ANZ_df.set_index('transaction_id', inplace = True)
ANZ_df['weekday'] = ANZ_df['date'].dt.day_name()


# NEW DATATFRAME BASED ON CUSTOMER -------------------

annual_salary=[]
age=[]
median_balance=[]
annual_spending=[]
gender = []
# DIVIDE DATA ----------------------------------------
ANZ_df_salary = ANZ_df[ANZ_df.txn_description == 'PAY/SALARY']
ANZ_df_pos_sales = ANZ_df[(ANZ_df.txn_description == 'POS') | (ANZ_df.txn_description == 'SALES-POS')]

# CREATE NEW DATAFRAME BASED ON CUSTOMER ----------------------------------------

customers = ANZ_df.customer_id.unique()
for i in customers:
    salary_for_user=ANZ_df_salary[ANZ_df_salary['customer_id']==i]
    annual_salary.append(salary_for_user['amount'].values.sum())
    age.append(salary_for_user['age'].iloc[0])
    gender.append(salary_for_user['gender'].iloc[0])
    median_balance.append(np.median(salary_for_user['balance'].values))
    spending=ANZ_df_pos_sales[ANZ_df_pos_sales['customer_id']==i]
    annual_spending.append(spending['amount'].values.sum())
    

customer_df = pd.DataFrame(
    {
        'customer_id' : customers, 
        'annual_salary' : annual_salary, 
        'median_balance' : median_balance, 
        'age' : age, 
        'gender': gender, 
    }
)
customer_df.set_index('customer_id', inplace=True)

#SPLIT TO TRAIN-TEST DATA -------------------------
X = customer_df.drop([ "annual_salary"], axis = 1)
y = customer_df.annual_salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# PRE-PROCESSING  --------------------------
ohe = OneHotEncoder(sparse = False)
scaler = StandardScaler()
column_transform = make_column_transformer((ohe, ["gender"]), (scaler, ['median_balance','age' ]))
 
# LINEAR REGRESSION MODEL ------------------
linear_model = LinearRegression()
linear_pipeline = make_pipeline(column_transform, linear_model)

linear_pipeline.fit(X_train, y_train)
linear_pred = linear_pipeline.predict(X_test)
linear_MAE = mean_absolute_error(y_test, linear_pred)

linear_scores = -1 * cross_val_score(linear_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print('linear scores: ')
print(linear_scores.mean())


# DECISION TREE MODEL ---------------------------------
tree_model = RandomForestRegressor(n_estimators = 10 ,random_state = 1, max_depth = 10)

tree_pipeline = make_pipeline(column_transform, tree_model)

tree_pipeline.fit(X_train, y_train)
tree_pred = tree_pipeline.predict(X_test)
tree_MAE = mean_absolute_error(y_test, linear_pred)

tree_scores = -1 * cross_val_score(tree_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print('tree scores: ')
print(tree_scores.mean())


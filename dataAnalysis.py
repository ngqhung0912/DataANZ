# IMPORTING LIBRARY ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from data_func import *
from data_func_2 import *

# READ INPUT ----------------------------------------
path = "/Users/macbook/Learning/DataANZ/ANZData.xlsx"
ANZ_df = pd.read_excel(path)
ANZ_df.set_index('transaction_id', inplace = True)
ANZ_df['weekday'] = ANZ_df['date'].dt.day_name()

# DIVIDE DATA ----------------------------------------
ANZ_df_salary = ANZ_df[ANZ_df.txn_description == 'PAY/SALARY']
ANZ_df_pos_sales = ANZ_df[(ANZ_df.txn_description == 'POS') | (ANZ_df.txn_description == 'SALES-POS')]

# DIVIDE DATA ----------------------------------------

customer_salary_data = group_by_data(ANZ_df_salary,['customer_id'])


# ANNUAL SALARY, BALANCE AND SPENDING ----------------
annual_salary=[]
age=[]
annual_balance=[]
annual_spending=[]

for i in customer_salary_data['customer_id'].values:
    salary_for_user=ANZ_df_salary[ANZ_df_salary['customer_id']==i]
    annual_salary.append(salary_for_user['amount'].values.sum())
    age.append(salary_for_user[['amount','age','customer_id','first_name','date','balance']]['age'].iloc[0])
    annual_balance.append(salary_for_user[['amount','age','customer_id','first_name','date','balance']]['balance'].values.mean())
    spending=ANZ_df_pos_sales[ANZ_df_pos_sales['customer_id']==i]
    annual_spending.append(spending['amount'].values.sum())

# RESHAPING THE ARRAY ------------------------------
annual_salary=np.array(annual_salary).reshape(-1,1)
age=np.array(age).reshape(-1,1)
annual_balance=np.array(annual_balance).reshape(-1,1)
annual_spending=np.array(annual_spending).reshape(-1,1)

# SCALING ANNUAL SALARY, BALANCE AND SPENDING TO SCALE 0-100. 
salary_scaler = MinMaxScaler(feature_range = (0,100)) #create scaler. Change the scale of list from min -> max to 0 -> 100. 
scaled_annual_salary = []
salary_scaler.fit(annual_salary)
for customer_salary in annual_salary: 
    scaled_annual_salary.append(salary_scaler.transform([customer_salary])[0][0])
scaled_annual_salary=np.array(scaled_annual_salary).reshape(-1,1).flatten()

balance_scaler = MinMaxScaler(feature_range = (0,100)) #create scaler. Change the scale of list from min -> max to 0 -> 100. 
scaled_annual_balance = []
balance_scaler.fit(annual_balance)
for customer_balance in annual_balance: 
    scaled_annual_balance.append(balance_scaler.transform([customer_balance])[0][0])
scaled_annual_balance=np.array(scaled_annual_balance).reshape(-1,1).flatten()

spending_scaler = MinMaxScaler(feature_range = (0,100)) #create scaler. Change the scale of list from min -> max to 0 -> 100. 
scaled_annual_spending = []
spending_scaler.fit(annual_spending)
for customer_spending in annual_spending: 
    scaled_annual_spending.append(spending_scaler.transform([customer_spending])[0][0])
scaled_annual_spending=np.array(scaled_annual_spending).reshape(-1,1).flatten()


age_scaler = MinMaxScaler(feature_range = (0,100)) #create scaler. Change the scale of list from min -> max to 0 -> 100. 
scaled_age = []
age_scaler.fit(age)
for customer_age in age: 
    scaled_age.append(age_scaler.transform([[customer_age[0]]])[0][0])
scaled_age=np.array(scaled_age).reshape(-1,1).flatten()

#CREATE NEW DATAFRAME ----------------------
customer_df = pd.DataFrame()

customer_df['customer_id'] = customer_salary_data['customer_id']
customer_df['scaled_age'] = scaled_age
customer_df['scaled_annual_balance'] = scaled_annual_balance
customer_df['scaled_annual_salary'] = scaled_annual_salary
customer_df['scaled_annual_spending'] = scaled_annual_spending
customer_df.set_index('customer_id',inplace=True)

y = customer_df.scaled_annual_salary
X = customer_df.drop('scaled_annual_salary', axis=1)


## SCORING FUNCTIONS 
def get_score(test_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    model = LinearRegression()
    model.fit(x_train,y_train)
    test_score = model.score(x_test,y_test)
    scores = -1 * cross_val_score(model, X, y,
                              cv=4,
                              scoring='neg_mean_absolute_error')


    return scores.mean(), test_score




# MACHINE LEARNING ---------------------------




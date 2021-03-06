# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
print(bank.head())
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)



# code ends here


# --------------
# code starts here
print(bank.head())
banks = bank.drop(columns = 'Loan_ID')
print(banks.head())
print(banks.isnull().sum())
#print(banks)
bank_mode = banks.mode().iloc[0]
print(bank_mode)
banks.fillna(bank_mode,inplace=True)
print(banks.loc[16])
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, values = 'LoanAmount', index=['Gender','Married','Self_Employed'],aggfunc = np.mean )
print(avg_loan_amount)
# code ends here



# --------------
# code starts here
df1= banks[(banks.Self_Employed=='Yes') & (banks.Loan_Status == 'Y')]
loan_approved_se = df1[['Self_Employed','Loan_Status']].count()
df2 = banks[(banks.Self_Employed=='No') & (banks.Loan_Status == 'Y')]
loan_approved_nse = df2[['Self_Employed','Loan_Status']].count()
percentage_se = (loan_approved_se.Self_Employed/614)*100
percentage_nse = (loan_approved_nse.Self_Employed/614)*100
print(percentage_se)
print(percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12)
#print(banks.head())
big_loan_term = len(loan_term[loan_term>=25])
print(big_loan_term)




# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')
print(loan_groupby.head())
loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]
print(loan_groupby.first())
mean_values = loan_groupby.mean()
print(mean_values.head())



# code ends here



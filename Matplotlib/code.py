# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()
plt.bar(loan_status.index, loan_status)

#Code starts here


# --------------
#Code starts here
property_and_loan = data.groupby(['Property_Area', 'Loan_Status'])
#print(property_and_loan.head())
property_and_loan = property_and_loan.size().unstack()
print(property_and_loan)
property_and_loan.plot(kind='bar', stacked=False, figsize=(20,15))
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation =45);


# --------------
#Code starts here



education_and_loan  = data.groupby(['Education', 'Loan_Status'])
education_and_loan = education_and_loan.size().unstack()
print(education_and_loan)
education_and_loan.plot(kind='bar', stacked=False, figsize=(20,15))
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation = 45);


# --------------
#Code starts here
import seaborn as sns



graduate = data[data['Education'] == 'Graduate']
print(graduate.head())

not_graduate = data[data['Education'] == 'Not Graduate']
print(not_graduate.head())

graduate.plot(kind = 'density', label = 'Graduate')
not_graduate.plot(kind = 'density', label = 'Not Graduate')











#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here




fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3, ncols = 1)
ax_1.scatter(data.ApplicantIncome, data.LoanAmount, color = 'r')
ax_1.set_title('Applicat Income')
ax_2.scatter(data.CoapplicantIncome, data.LoanAmount, color = 'g')
ax_2.set_title('Coapplicant Income')

data['TotalIncome'] = data['ApplicantIncome']+data['CoapplicantIncome']

ax_3.scatter(data.TotalIncome, data.LoanAmount, color = 'b')
ax_3.set_title('Total Income')





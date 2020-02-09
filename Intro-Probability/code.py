# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df  = pd.read_csv(path)
#print (df)
p_a = df[df['fico'].astype(float) >700].shape[0]/df.shape[0]
print(p_a)
p_b = df[df['purpose'] == 'debt_consolidation'].shape[0]/df.shape[0]
print(p_b)
df1 = df[df['purpose'] == 'debt_consolidation']
p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
print(p_a_b)
result = (p_a == p_a_b)
print (result)
# code ends here


# --------------
# code starts here

prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]

prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]

new_df = df[df['paid.back.loan'] == 'Yes']

prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0]/new_df.shape[0]

bayes = prob_pd_cs*prob_lp/prob_cs

print(bayes)
# code ends here


# --------------
# code starts here
df.purpose.value_counts(normalize=True).plot(kind='bar')
df1 = df[df['paid.back.loan'] == 'No']
df1.purpose.value_counts(normalize=True).plot(kind='bar')

# code ends here


# --------------
# code starts here
from statistics import median
inst_median = median(df.installment)
print(inst_median)
inst_mean = df.installment.mean()
print(inst_mean)
plt.hist(df.installment, normed=True, bins=30)
plt.hist(df['log.annual.inc'], normed=True, bins=30)


# code ends here



# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'].replace('-', 'Agender',inplace=True)
gender_count = data['Gender'].value_counts()
print('gender_count')
gender_count.plot.bar()

#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
print(alignment)
plt.pie(alignment)
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']]
sc_covariance = sc_df.Strength.cov(sc_df.Combat)
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_combat*sc_strength)


ic_df = data[['Intelligence', 'Combat']]
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_covariance = ic_df.Intelligence.cov(ic_df.Combat)
ic_pearson = ic_covariance/(ic_combat*ic_intelligence)

print("sc_covariance:", sc_covariance)
print("sc_strength:", sc_strength)
print("sc_combat:", sc_combat)
print("sc_pearson", sc_pearson)
print("ic_intelligence", ic_intelligence)
print("ic_covariance", ic_covariance)


# --------------
#Code starts here
total_high = data['Total'].quantile(q=0.99)
print(total_high)
super_best = data[data['Total'] > total_high]
print(super_best.head())
super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(3)
ax_1.plot(data['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.plot(data['Speed'])
ax_2.set_title('Speed')
ax_3.plot(data['Power'])
ax_3.set_title('Power')



# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].plot(kind='hist')
plt.show()
data = data[data['Rating'] <= 5]
data['Rating'].plot(kind='hist')

#Code ends here


# --------------
# code starts here


total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null,percent_null],axis=1, keys=['Total','Percent'])
print(missing_data)
data.dropna(axis = 1, how ='any', inplace = True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1, keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
catplot = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)
catplot.set_xticklabels(rotation=90)
catplot.fig.suptitle("Rating vs Category [BoxPlot]")

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = pd.to_numeric(data['Installs'])

le = LabelEncoder()
le.fit(data['Installs'])
data['Installs'] = le.fit_transform(data['Installs'])
reg = sns.regplot(x="Installs", y="Rating", data=data)
reg.set_title("Rating vs Category [BoxPlot]")
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$','')
data['Price'] = pd.to_numeric(data['Price'])
reg1 = sns.regplot(x="Price", y="Rating", data=data)
reg1.set_title("Rating vs Price [BoxPlot]")

#Code ends here


# --------------

#Code starts here
ser = pd.Series(data['Genres'])
ser.unique()
data.Genres = data.Genres.str.split(';').str[0]
gr_mean = data[['Genres','Rating']].groupby("Genres", as_index=False).mean()
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(by = ['Rating'])
print(gr_mean)


#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
regplot1 = sns.regplot(x="Last Updated Days", y="Rating", data=data)
regplot1.set_title('Rating vs Last Updated [RegPlot]')
#Code ends here



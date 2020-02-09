# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
#print(df.head())

df.INCOME = df.INCOME.str.replace('$','').str.replace(',','')
df.HOME_VAL = df.HOME_VAL.str.replace('$','').str.replace(',','')
df.BLUEBOOK = df.BLUEBOOK.str.replace('$','').str.replace(',','')
df.OLDCLAIM = df.OLDCLAIM.str.replace('$','').str.replace(',','')
df.CLM_AMT = df.CLM_AMT.str.replace('$','').str.replace(',','')
print(df.head())
print(df.info())
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
count = y.value_counts()
print(count)
X_train, X_test, y_test, y_train = train_test_split(X,y, test_size=0.3, random_state=6)

# Code ends here


# --------------
# Code starts here
X_train.INCOME = X_train.INCOME.astype(float)
X_train.HOME_VAL = X_train.HOME_VAL.astype(float)
X_train.BLUEBOOK = X_train.BLUEBOOK.astype(float)
X_train.OLDCLAIM = X_train.OLDCLAIM.astype(float)
X_train.CLM_AMT = X_train.CLM_AMT.astype(float)

X_test.INCOME = X_test.INCOME.astype(float)
X_test.HOME_VAL = X_test.HOME_VAL.astype(float)
X_test.BLUEBOOK = X_test.BLUEBOOK.astype(float)
X_test.OLDCLAIM = X_test.OLDCLAIM.astype(float)
X_test.CLM_AMT = X_test.CLM_AMT.astype(float)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
# Code starts here
print(X_train.shape)
print(X_test.shape)
X_train = X_train.dropna(subset=['YOJ','OCCUPATION'])
X_test = X_test.dropna(subset=['YOJ','OCCUPATION'])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']] = X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].mean(), inplace = True)
X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']] = X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']].mean(), inplace = True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# code starts here 
print(X_train.isnull().sum())
print(y_train.isnull().sum())
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

# Code ends here



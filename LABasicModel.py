import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Reading data
data = pd.read_csv('USW00093134.csv')

'''
# Preparing data
LAdata = data.loc[:, 'MM/DD/YYYY':'PRCP']
LAdata = LAdata.loc[LAdata['YEAR'] >= 2000]

LAdata = LAdata.drop(['MM/DD/YYYY', 'ID', 'TMAX_FLAGS', 'TMIN_FLAGS'], axis = 1)

LAdata = LAdata[(LAdata.MONTH == 3)] # For the month of march


# Linear Regression

# Input variable
X = LAdata[['YEAR', 'DAY']]
X = sm.add_constant(X) 
y = LAdata.PRCP

lr_model1 = sm.OLS(y, X).fit()

print(lr_model1.summary())


# Logistic Regression

X = LAdata[['YEAR', 'DAY']]

LAdata['RAIN'] = np.where(LAdata['PRCP'] >= 1, 1, 0)    # Converting PRCP to categorical variable

y = LAdata.RAIN

def log_classify(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    #Creating Logisitic Regression model
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    
    # Predictions on test set
    log_pred = logmodel.predict(X_test)

    # Accuracy of logmodel1
    return(accuracy_score(y_test, log_pred))


# For 100 trials
log_accuracy = []
for i in range(101):
    
    log_accuracy.append(log_classify(X, y))
    
print('Logistic Regression Results')    
print(log_accuracy)
print(np.mean(log_accuracy))

# Prediction - 13th and 14th of March 2021 Logistic Regression
LA_x = [[2021, 13], [2021, 14]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

LA_pred = logmodel.predict(LA_x)

print(LA_pred)


# kNN
def knn_classify(X, y):
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    knn_pred = knn.predict(X_test)
    
    return(accuracy_score(y_test, knn_pred))

knn_accuracy = []
for i in range(101):
    
    knn_accuracy.append(knn_classify(X, y))

print('kNN Results')    
print(knn_accuracy)
print(np.mean(knn_accuracy))


# Prediction - 13th and 14th of March 2021 kNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

LA_pred = knn.predict(LA_x)
print(LA_pred)


# Random Forest
def rf_classify(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train.values.ravel())
    
    rfc_pred = rfc.predict(X_test)
    
    return(accuracy_score(y_test, rfc_pred))

rf_accuracy = []
for i in range(101):
    
    rf_accuracy.append(rf_classify(X, y))
    
print('Random Forest Results')    
print(rf_accuracy)
print(np.mean(rf_accuracy))

# Prediction - 13th and 14th of March 2021 Random Forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train.values.ravel())
    
LA_pred = rfc.predict(LA_x)
print(LA_pred)









# Iterative Model

# Reading data
data = pd.read_csv('USW00093134.csv')

data = data.loc[:, 'MM/DD/YYYY':'PRCP']
data = data.loc[data['YEAR'] >= 2000]

# Dropping all NaN values and reseting the indices of the dataframe
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data = data.drop(['MM/DD/YYYY', 'ID', 'TMAX_FLAGS', 'TMIN_FLAGS'], axis = 1)
data = data[(data.MONTH == 3)]

# Setting the number of days to look back on
days_back = 5
# Variables to look back on
var_list = ['TMAX', 'TMIN', 'PRCP']
# Creating new columns with previous weather information
for i in range(days_back):
    new_colnames = [j+'_'+str(i+1)+'_DAY' for j in var_list]
    data[new_colnames] = data[var_list].shift(i+1)
    
# Creating PRCP_TF column to either true or false depending on whether or not any rain occured
data['PRCP_TF'] = data['PRCP'] > 0
data['PRCP_TF'].value_counts()

# Establishing predictor variables for model
X = data[['YEAR', 'DAY', 'TMAX_1_DAY', 'TMIN_1_DAY', 'PRCP_1_DAY',
         'TMAX_2_DAY', 'TMIN_2_DAY', 'PRCP_2_DAY',
         'TMAX_3_DAY', 'TMIN_3_DAY', 'PRCP_3_DAY',
         'TMAX_4_DAY', 'TMIN_4_DAY', 'PRCP_4_DAY',
         'TMAX_5_DAY', 'TMIN_5_DAY', 'PRCP_5_DAY']]

X = X.fillna(0)

# Establishing response variables
y = data[['PRCP_TF']]

def log_classify(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    #Creating Logisitic Regression model
    logmodel = LogisticRegression(max_iter = 200)
    logmodel.fit(X_train,y_train)
    
    # Predictions on test set
    log_pred = logmodel.predict(X_test)

    # Accuracy of logmodel1
    return(accuracy_score(y_test, log_pred))


# For 100 trials
log_accuracy = []
for i in range(101):
    
    log_accuracy.append(log_classify(X, y))
    
print('Logistic Regression Results')    
print(log_accuracy)
print(np.mean(log_accuracy))

'''
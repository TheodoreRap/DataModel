import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("dataModel.csv")
print(df)

f=["x","y","z"]
X=df[f]
print(X)

Y=df["apot"]
Y2=df["onoff"]

print(Y)

Xtrain=X.iloc[0:60] # Επιλογή των 61 πρώτων γραμμών των δεδομένων μας
Ytrain=Y.iloc[0:60]
Y2train=Y2.iloc[0:60]

reg = linear_model.LinearRegression() # Το μοντέλο φτίαχνεται βάσει των 61 πρώτων
reg.fit(Xtrain,Ytrain) # τιμών και μετά το χρησιμοποιώ για να υπολογίσω τις υπόλοιπες τιμές F=ax+by+cz

print("Συντελεστές: ",reg.coef_) # Συντελεστές

y=reg.predict([[9,1,7]]) # Πρόβλεψη αποτελέσματος με μικρή απόκλιση απο το πραγματικό
print("Πρόβλεψη αποτελέσματος: ",y)

Xtest=X.iloc[61:99] 
Ytest=Y.iloc[61:99]
Y2test=Y2.iloc[61:99]

print("Επιτυχία :",reg.score(Xtest,Ytest)) # Επιτυχία πρόβλεψης των τιμών

regr = MLPRegressor(random_state=1,hidden_layer_sizes=(10,20), max_iter=1000).fit(Xtrain, Ytrain)
regr.fit(Xtrain,Ytrain) # Νέο μοντέλο βάσει των 61 πρώτων γραμμών με apot

print("\nΕπιτυχία νέου μοντέλου: ",regr.score(Xtest,Ytest)) # Επιτυχία πρόβλεψης των τιμών βάσει του νέου μοντέλου

reg2 = linear_model.Ridge(alpha=.5)
reg2.fit(Xtrain,Ytrain)

print("\nΣυντελεστές: ",reg2.coef_) # Συντελεστές
print("Επιτυχία νέου μοντέλου: ",reg2.score(Xtest,Ytest))


regr1 = MLPClassifier(random_state=1,hidden_layer_sizes=(10,20), max_iter=1000)
regr1.fit(Xtrain,Y2train) # Νέο μοντέλο βάσει των 61 πρώτων γραμμών με onoff

print("\nΕπιτυχία νέου μοντέλου: ",regr1.score(Xtest,Y2test))
print("Πρόβλεψη αποτελέσματος [0]: ",regr1.predict([[6,3,2]])) # Πρόβλεψη αποτελέσματος onoff -> 0 ή 1 
print("Πρόβλεψη αποτελέσματος [1]: ",regr1.predict([[7,10,9]])) # Πρόβλεψη αποτελέσματος onoff -> 0 ή 1 

regr2 = DecisionTreeClassifier()
regr2.fit(Xtrain,Y2train) # Νέο μοντέλο βάσει των 61 πρώτων γραμμών με onoff

print("\nΕπιτυχία νέου μοντέλου: ",regr2.score(Xtest,Y2test))

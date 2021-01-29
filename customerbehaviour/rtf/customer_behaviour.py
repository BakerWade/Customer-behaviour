from os import replace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.concat import concat
import seaborn as sns

cube = pd.read_csv('Customer_Behaviour.csv')

sex = pd.get_dummies(cube['Gender'],drop_first=True)
cube2 = pd.concat([cube.drop('Gender',axis=1),sex],axis=1)

cube2 =cube2.drop('User ID', axis=1)

X = cube2.drop('Purchased',axis=1)
y = cube2['Purchased']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier()
dec.fit(X_train,y_train)

pred = dec.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))"""

from sklearn.ensemble import RandomForestClassifier

ran =RandomForestClassifier()
ran.fit(X_train,y_train)
predict = ran.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))

"""sns.jointplot(x='EstimatedSalary', y='Purchased', data=cube, hue='Gender', palette='coolwarm',kind='hex')
plt.title('EstimatedSalary vs Purcahsed')
plt.show()"""


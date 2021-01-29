from os import replace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.concat import concat
import seaborn as sns
from tensorflow.python.keras.backend import dropout

cube = pd.read_csv('Customer_Behaviour.csv')

sex = pd.get_dummies(cube['Gender'],drop_first=True)
cube2 = pd.concat([cube.drop('Gender',axis=1),sex],axis=1)

cube2 =cube2.drop('User ID', axis=1)

"""g = sns.FacetGrid(cube2,col='Purchased',row='Male')
g.map(plt.scatter,'Age','EstimatedSalary')
plt.show()"""

X = cube2.drop('Purchased',axis=1)
y = cube2['Purchased']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(4,activation='relu'))
#model.add(Dropout(0.45))

model.add(Dense(2,activation='relu'))
#model.add(Dropout(0.45))

model.add(Dense(1,activation='sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=800, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

from tensorflow.keras.models import load_model
model.save('behaviour_model.h5')  


pred = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

"""cm= confusion_matrix(y_test,pred, labels= log.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= log.classes_)

disp.plot()"""




"""sns.jointplot(x='EstimatedSalary', y='Purchased', data=cube, hue='Gender', palette='coolwarm',kind='hex')
plt.title('EstimatedSalary vs Purcahsed')
plt.show()"""


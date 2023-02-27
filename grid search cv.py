import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C://Users\hp\Downloads\Social_Network_Ads.csv')
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)



from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
acc

from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000], 'kernel':['linear']},
            {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1
                         )
grid_search=grid_search.fit(X_train,y_train)
accuracy=grid_search.best_score_
accuracy
grid_search.best_params_


#now we train our model with new parameter
classifier=SVC(kernel='rbf',gamma=0.9)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm
accuracy=(sum(np.diag(cm))/len(y_test))
accuracy


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
acc


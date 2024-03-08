import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 


global filename
global dataset
global  X_train, X_test, y_train, y_test
global classifier
global rbf_classifier

def upload():
    global filename
    global dataset
    filename ="./Dataset/datasets.csv"
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    label = dataset.groupby('Level').size()
    colors = ["red","green","yellow"]
    label.plot(kind="bar", color = colors)
    plt.show()

def processDataset():
    global X, Y
    global dataset
    le = LabelEncoder()
    dataset['Level'] = pd.Series(le.fit_transform(dataset['Level'].astype(str)))
    dataset['Patient Id'] = pd.Series(le.fit_transform(dataset['Patient Id'].astype(str)))
    X = dataset.values[:,1:dataset.shape[1]-1]
    Y = dataset.values[:,dataset.shape[1]-1]
    Y = Y.astype('int')
    X = normalize(X)
    print("Processing data completed")
    
    
def runEnsemble():
    global classifier
    global  X_train, X_test, y_train, y_test
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    dt = DecisionTreeClassifier()
    ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
    mlp = MLPClassifier(max_iter=200,hidden_layer_sizes=100,random_state=42)
    knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) 

    vc = VotingClassifier(estimators=[('dt', dt), ('ab', ada_boost), ('mlp', mlp),  ('knn',knn)] ,voting='soft')
    vc.fit(X_train, y_train)
    predict = vc.predict(X_test) 
    p = precision_score(y_test, predict,average='micro') * 100
    r = recall_score(y_test, predict,average='micro') * 100
    f = f1_score(y_test, predict,average='micro') * 100
    a = accuracy_score(y_test,predict)*100
    classifier = vc

    
    dt.fit(X_train,y_train)
    mlp.fit(X_train,y_train)
    ada_boost.fit(X_train,y_train)
    knn.fit(X_train,y_train)
    
    dtr = dt.predict(X_test)
    mlr = mlp.predict(X_test)
    ada = ada_boost.predict(X_test)
    knr = knn.predict(X_test)

    print("dt result           mlp result          ada result         knn result")
    for x in range(0,200):
        print(dtr[x] , end="                    ")
        print(mlr[x] , end="                    ")
        print(ada[x] , end="                    ")
        print(knr[x])

    print("Ensemble completed")
    

def predict():
    global classifier
    #text.delete('1.0', END)
    filename = "./Dataset/test.csv"
    test = pd.read_csv(filename)
    data = test.values
    data = data[:,1:data.shape[1]]
    data = normalize(data)
    predict = classifier.predict(data)
    print(predict)
    test = test.values
    for i in range(len(predict)):
        result = 'High'
        if predict[i] == 0:
            result = 'High. CT Scan Required'
        if predict[i] == 1:
            result = 'Low. CT Scan Not Required'
        if predict[i] == 2:
            result = 'Medium. CT Scan Maybe Required'



upload()
processDataset()
runEnsemble()
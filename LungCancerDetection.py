from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog,ttk
import tkinter
from pandastable import Table
import numpy as np
import time
from tkinter import filedialog
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
import pickle
import os
import cv2
from Shadow import Shadow

global main

main = tkinter.Tk()
main.title("Lung Cancer Detection Using Ensemble Algorithm")
main.geometry("1920x1080")


global filename
global dataset
global  X_train, X_test, y_train, y_test
global classifier
global rbf_classifier
global pt
global text

def upload():
    global filename
    global dataset
    global main
    global pt
    global text
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)


    '''text.insert(END,str(dataset.head())+"\n\n")
    text.insert(END,"Dataset contains total records    : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Dataset contains total attributes : "+str(dataset.shape[1])+"\n")'''

    pt = Table(text,dataframe = dataset,width = 850, height=500)
    pt.autoResizeColumns()
    pt.show()
    

    label = dataset.groupby('Level').size()
    colors = ["red","green","yellow"]

    plt.bar(label.keys(), label.values,color = colors)
    plt.ylabel('Values')
    plt.title("Lung Cancer Dataset")
    plt.show()

def processDataset():
    global X, Y
    global dataset
    global pt
    global text
    
    pt.remove()
    text=Text(main,height=30,width=120)
    text.config(font= ('times', 12, 'bold'))
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=160)
    text.delete('1.0', END)

    le = LabelEncoder()
    dataset['Level'] = pd.Series(le.fit_transform(dataset['Level'].astype(str)))
    dataset['Patient Id'] = pd.Series(le.fit_transform(dataset['Patient Id'].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    X = dataset.values[:,1:dataset.shape[1]-1]
    Y = dataset.values[:,dataset.shape[1]-1]
    Y = Y.astype('int')
    X = normalize(X)
    print(X)
    print(Y)

    pt = Table(text,dataframe = dataset,width = 850, height=500)
    pt.autoResizeColumns()
    pt.show()

    #text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")
    #text.insert(END,"Dataset contains total Features: "+str(X.shape[1])+"\n")
    
    
def runEnsemble():
    global classifier,text
    global  X_train, X_test, y_train, y_test
    global X, Y

    pt.remove()
    text=Text(main,height=30,width=120)
    text.config(font= ('times', 12, 'bold'))
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=160)
    text.delete('1.0', END)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total dataset records : "+str(X.shape[0])+"\n" )
    text.insert(END,"Total dataset records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total dataset records used to test algorithms  : "+str(X_test.shape[0])+"\n\n")
    dt = DecisionTreeClassifier()
    ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
    mlp = MLPClassifier(max_iter=200,hidden_layer_sizes=100,random_state=42)
    knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) 

    knn.fit(X_train,y_train)
    print("Knn result")
    print(knn.predict(X_test))


    vc = VotingClassifier(estimators=[('dt', dt), ('ab', ada_boost), ('mlp', mlp), ('knn',knn)], voting='soft')
    vc.fit(X_train, y_train)
    predict = vc.predict(X_test) 
    p = precision_score(y_test, predict,average='micro') * 100
    r = recall_score(y_test, predict,average='micro') * 100
    f = f1_score(y_test, predict,average='micro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,"Ensemble of Decision Tree, MLP and AdaBoost Performance Result\n\n")
    text.insert(END,"Ensemble Algorithms Precision : "+str(p)+"\n")
    text.insert(END,"Ensemble Algorithms Recall    : "+str(r)+"\n")
    text.insert(END,"Ensemble Algorithms FMeasure  : "+str(f)+"\n")
    text.insert(END,"Ensemble Algorithms Accuracy  : "+str(a)+"\n")
    classifier = vc
    

def predict():
    global classifier
    global text
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
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
        text.insert(END,"Test Values : "+str(test[i])+" Predicted Disease Status : "+result+"\n\n")
        

def trainRBF():
    global rbf_classifier,text               
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            rbf_classifier = pickle.load(file)
        file.close()
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        predict = rbf_classifier.predict(X_test)
        svm_acc = accuracy_score(y_test,predict)*100
        text.insert(END,"RBF training accuracy : "+str(svm_acc)+"\n\n")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (10,10))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(10,10,3)
                    X.append(im2arr)
                    if name == 'normal':
                        Y.append(0)
                    if name == 'abnormal':
                        Y.append(1)
        X = np.asarray(X)
        Y = np.asarray(Y)
        print(Y.shape)
        print(X.shape)
        print(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        rbf_classifier = svm.SVC(kernel='rbf') 
        rbf_classifier.fit(X, Y)
        predict = rbf_classifier.predict(X_test)
        svm_acc = accuracy_score(y_test,predict)*100
        text.insert(END,"RBF training accuracy : "+str(svm_acc)+"\n\n")
        with open('model/model.txt', 'wb') as file:
            pickle.dump(rbf_classifier, file)
        file.close()
               
def predictCTscan():
    global rbf_classifier,text
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (10,10))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(10,10,3)
    X = np.asarray(im2arr)
    X = X.astype('float32')
    X = X/255
    XX = []
    XX.append(X)
    XX = np.asarray(XX)
    print(XX.shape)
    X = np.reshape(XX, (XX.shape[0],(XX.shape[1]*XX.shape[2]*XX.shape[3])))
    print(X.shape)
    predict = rbf_classifier.predict(X)
    if predict == 0:
        msg = "Uploaded CT Scan is Normal"
    if predict == 1:
        msg = "Uploaded CT Scan is Abnormal"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    if(msg == "Uploaded CT Scan is Normal"):
        cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (51, 204, 51), 2)
    else:
        cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,0,0), 2)
    #cv2.imshow(msg, img)
    #cv2.waitKey(0)   
    plt.imshow(img) 
    plt.show()
    



def changeOnHover(button, colorOnHover, colorOnLeave):
    button.bind("<Enter>", func=lambda e: button.config(
        background=colorOnHover))
 
    button.bind("<Leave>", func=lambda e: button.config(
        background=colorOnLeave))
    

font = ('times', 22, 'bold')
title = Label(main, text='Lung Cancer Detection Using Ensemble Algorithm')
title.config(bg='#3F3E3E', fg='white')  
title.config(font=font)           
title.config(height=3, width=100)       
title.place(x=0,y=10)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Lung Cancer Dataset", command=upload)
upload.place(x=1040,y=260)
upload.config(font=font1)
changeOnHover(upload, "#6C9DDA", "white")  
Shadow(upload, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=1040,y=320)
processButton.config(font=font1) 
changeOnHover(processButton, "#6C9DDA", "white")  
Shadow(processButton, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})
    

eaButton = Button(main, text="Run Ensemble Algorithms", command=runEnsemble)
eaButton.place(x=1040,y=380)
eaButton.config(font=font1) 
changeOnHover(eaButton, "#6C9DDA", "white")
Shadow(eaButton, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})

predictButton = Button(main, text="Predict Lung Cancer Disease", command=predict)
predictButton.place(x=1040,y=440)
predictButton.config(font=font1)
changeOnHover(predictButton, "#6C9DDA", "white")
Shadow(predictButton, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})

rbfButton = Button(main, text="Train RBF on Lungs CT-Scan Images", command=trainRBF)
rbfButton.place(x=1040,y=510)
rbfButton.config(font=font1)
changeOnHover(rbfButton, "#6C9DDA", "white")
Shadow(rbfButton, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})

predictButton = Button(main, text="Predict Cancer from CT-Scan", command=predictCTscan)
predictButton.place(x=1040,y=570)
predictButton.config(font=font1)
changeOnHover(predictButton, "#6C9DDA", "white")
Shadow(predictButton, size=2, offset_x=2, offset_y=2, onhover={'size':4, 'offset_x':4, 'offset_y':4})

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=160)
text.config(font=font1)


main.config(bg="#3F3E3E")
main.mainloop()


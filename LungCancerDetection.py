from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
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
import pickle
import os
import cv2

main = tkinter.Tk()
main.title("Lung Cancer Detection Using Ensemble Algorithm")
main.geometry("1920x1080")


global filename
global dataset
global  X_train, X_test, y_train, y_test
global classifier
global rbf_classifier

def upload():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head())+"\n")
    text.insert(END,"Dataset contains total records    : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Dataset contains total attributes : "+str(dataset.shape[1])+"\n")
    label = dataset.groupby('Level').size()
    colors = ["red","green","yellow"]
    label.plot(kind="bar", color = colors)
    plt.show()

def processDataset():
    global X, Y
    global dataset
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
    text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")
    text.insert(END,"Dataset contains total Features: "+str(X.shape[1])+"\n")
    
    
def runEnsemble():
    global classifier
    global  X_train, X_test, y_train, y_test
    global X, Y
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total dataset records : "+str(X.shape[0])+"\n")
    text.insert(END,"Total dataset records used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total dataset records used to test algorithms  : "+str(X_test.shape[0])+"\n\n")
    dt = DecisionTreeClassifier()
    ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
    mlp = MLPClassifier(max_iter=200,hidden_layer_sizes=100,random_state=42)
    vc = VotingClassifier(estimators=[('dt', dt), ('ab', ada_boost), ('mlp', mlp)], voting='soft')
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
    global rbf_classifier                         
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
    global rbf_classifier
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
        cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)    
    

font = ('times', 16, 'bold')
title = Label(main, text='Lung Cancer Detection Using Ensemble Algorithm')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Lung Cancer Dataset", command=upload)
upload.place(x=950,y=100)
upload.config(font=font1)  

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=950,y=150)
processButton.config(font=font1) 

eaButton = Button(main, text="Run Ensemble Algorithms", command=runEnsemble)
eaButton.place(x=950,y=200)
eaButton.config(font=font1) 

predictButton = Button(main, text="Predict Lung Cancer Disease", command=predict)
predictButton.place(x=950,y=250)
predictButton.config(font=font1)

rbfButton = Button(main, text="Train RBF on Lungs CT-Scan Images", command=trainRBF)
rbfButton.place(x=950,y=300)
rbfButton.config(font=font1)

predictButton = Button(main, text="Predict Cancer from CT-Scan", command=predictCTscan)
predictButton.place(x=950,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()

#!/usr/bin/env python
# coding: utf-8

#https://stackoverflow.com/questions/28467068/add-row-to-dataframe

import numpy as np
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import reciprocal, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.utils import np_utils
import keras
import numpy
import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



f4 = open('/Users/varungupta/Documents/MachineLearning/project/Out1_3grams.csv','w+')
f4.write("Id,Expected,Expected"+"\n")
WordReader = csv.reader(open('/Users/varungupta/Documents/MachineLearning/project/words.txt'), delimiter=',')
for word,diac in WordReader:
    
    my_stop_words=[]
    my_stop_words.append(word)
    my_stop_words.append(diac)
    
    cv=TfidfVectorizer(min_df=10, stop_words=my_stop_words,lowercase=True)
    #Import of train Filed   ata=data["Message"]
    train=[]
    dfnew = pd.DataFrame()
    trainfile="/Users/varungupta/Documents/MachineLearning/project/kaggleS19/files/"+word+".txt"
    df=pd.read_csv(trainfile, sep='\t',names=['Status','Message'])
    df.head()
    df.loc[df["Status"]=='WORD',"Status"]=1
    df.loc[df["Status"]=='DIAC',"Status"]=0
    df_x=df["Message"]
    df_y=df["Status"]
    df.tail()
    
    #import of testfile
    testfile="/Users/varungupta/Documents/MachineLearning/project/kaggleS19/files/"+word+"test.txt"
    #data = pd.read_csv(testfile,names=['Message'])
    data = pd.read_csv(testfile,sep='<TAB>',names=['linenum','Message'])
    #data=data["Message"]
    data_x=data["Message"]
    data_y=data["linenum"]
   
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.10, random_state=4)
    x_traincv=cv.fit_transform(x_train).todense()
    x_test=cv.transform(x_test).todense()
    data_cv=cv.transform(data_x).todense()
    data_cv.shape
    y_train=np_utils.to_categorical(y_train, 2)
    
    nottty,dim = x_traincv.shape


    def baseline_model(input_dim):
        model = Sequential()
        model.add(Dense(1000, input_dim=input_dim, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, input_dim=dim, epochs=10, batch_size=800, verbose=0)
    estimator.fit(x_traincv, y_train)
    testing2=[]
    for i in range(len(data_y)):
        #print(i)
        testing2.append(str(data_y[i])+"same")
        testing2.append(estimator.predict_proba(data_cv[i]))
        testing2.append("#")
        #print(data_cv[0])
    #new=cv.transform(data[i]).toarray()
    for j in testing2:
        print(j)
        f4.write(str(j))
    
f4.close()
    
#

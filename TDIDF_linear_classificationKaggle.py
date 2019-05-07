#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf


# In[2]:


###################### Does Not involve the data preprocessing steps ###################
###
f4 = open('/Users/varungupta/Documents/MachineLearning/project/OutLastgrams.csv','w+')
f4.write("Id,Expected,Expected"+"\n")
WordReader = csv.reader(open('/Users/varungupta/Documents/MachineLearning/project/words.txt'), delimiter=',')
for word,diac in WordReader:
    
    my_stop_words=[]
    my_stop_words.append(word)
    my_stop_words.append(diac)
    cv=TfidfVectorizer(min_df=15, stop_words=my_stop_words,lowercase=True,ngram_range=(1,3))
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
   
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
    x_traincv=cv.fit_transform(x_train)
    x_testcv=cv.transform(x_test)
    data_cv=cv.transform(data_x)
    data_cv.shape
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    ###Log Reg
    #Fitting into the model
    logreg = LogisticRegression(C=0.1)#####0.1 with mindf=10 good
    logreg.fit(x_traincv,y_train)#####
    #Logreg = LogisticRegressionCV()
    #Logreg=LogisticRegressionCV(cv=10, random_state=4).fit(x_traincv, y_train)
    
    #pred= logreg.predict(x_testcv)
    #x=metrics.accuracy_score(pred,y_test)
    #print(x)

    testing2=[]
    
    for i in range(len(data_y)):
        #print(i)
        testing2.append(str(data_y[i])+"same")
        testing2.append(logreg.predict_proba(data_cv[i]))#####
        testing2.append("#")
        #print(data_cv[0])
    #new=cv.transform(data[i]).toarray()
    for j in testing2:
        print(j)
        f4.write(str(j))
    

    
f4.close()
      


# In[6]:


#svmmoDEL

train=[]
### To be deleted for testing purpose

#testdelete.txt 

###
f4 = open('/Users/varungupta/Documents/MachineLearning/project/SVM1_3grams.csv','w+')
f4.write("Id,Expected,Expected"+"\n")
WordReader = csv.reader(open('/Users/varungupta/Documents/MachineLearning/project/words.txt'), delimiter=',')
for word,diac in WordReader:
    
    my_stop_words=[]
    #WordReader = csv.reader(open('/Users/varungupta/Documents/MachineLearning/project/kaggleS19/words.txt'), delimiter=',')
    #for word,diac in WordReader:
    my_stop_words.append(word)
    my_stop_words.append(diac)
    cv=TfidfVectorizer(min_df=10, stop_words=my_stop_words,lowercase=True)#, ngram_range=(1,3))
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
   
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
    x_traincv=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    data_cv=cv.transform(data_x)
    data_cv.shape
    y_train=y_train.astype('int')
    
    #Fitting into the model
    #logreg = LogisticRegression()
    #
    ################# SVM start#############
    #svm_model = svm.SVC(probability=True)  ##SVM
    #svm_model = svm_model.fit(x_traincv, y_train) 
    ################## SVM ends #############
    
    ##############Randome Forest ##############
    ##1st
    #rf = RandomForestClassifier(100)
    #rf_model = rf.fit(x_traincv, y_train)
    
    ##2nd
    rf2 = RandomForestClassifier(1000)
    rf2_model = rf2.fit(x_traincv, y_train)
    ##############Random forest###############
    
    
    
    testing2=[]
    
    for i in range(len(data_y)):
        #print(i)
        testing2.append(str(data_y[i])+"same")
        #predicted = pd.DataFrame(rf_model.predict(data_cv[i]))  
        #probs = pd.DataFrame(rf_model.predict_proba(data_cv[i]))
        #testing2.append(svm_model.predict_proba(data_cv[i])) ## For SVM
        testing2.append(rf2_model.predict_proba(data_cv[i])) ## For Random forest
        #testing2.append(NB.predict_proba(data_cv[i])) ## For Random forest
        #print(predicted)
        #break
        testing2.append("#")
        #print(data_cv[0])
        #new=cv.transform(data[i]).toarray()
    for j in testing2:
        print(j)
        f4.write(str(j))
    

    
f4.close()


# In[22]:



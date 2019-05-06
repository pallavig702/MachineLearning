#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML



from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn import cluster 
from sklearn import metrics
from scipy.cluster.hierarchy import cophenet,fcluster
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ConfusionMatrix


import numpy as np
import pandas as pd
import keras, sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import classification_report


# In[45]:
#################### Unsupervised Clustering Starts Here ##################
########################################################################


df=pd.read_csv('disease_clusters.csv')
df=df[['CHF', 'Cardiac_dysrhythmias', 'Heart_valve_disorders',
       'Pulmonary_heart_disease', 'Peripheral_Heart', 'Hypertension',
       'Conduction', 'Nerves', 'Chronic_Pulmonary', 'Diabetes',
       'Thyroid_disorder', 'Renal', 'Liver', 'Gastroduodenal_ulcer', 'HIV',
       'arthritis', 'Coagulation', 'Endocrine', 'Nutritional_deficiencies',
       'Fluid_electrolyte', 'Alcohol_Abuse', 'Substance_Abuse',
       'Schizophrenia_psychoses', 'Anxiety', 'Affective']]


# In[46]:


#################### Hierarchial Clustering Starts Here ##################


df_transpose=pd.DataFrame(df.transpose())
metric_types=['braycurtis', 'correlation','hamming',
              'minkowski','cosine', 'euclidean', 'l1', 'l2','manhattan']
for met in metric_types:
    D=metrics.pairwise_distances(df_transpose, metric=met) #Pairwise metrics
    D_linkage=linkage(D,method='ward')

    diseases=df_transpose.index
    plt.figure(figsize=(10, 10))
    plt.title('Hierarchical Clustering Dendrogram. Metric: '+ met)
    plt.ylabel('Disease index')
    plt.xlabel('Distance')
    dendrogram(D_linkage,labels=df_transpose.index,orientation='left',)
    #plt.show()
    name=met+".png"
    plt.savefig(met,bbox_inches='tight')
    plt.savefig('sample.pdf')
    #savefig('test.png', bbox_inches='tight')
    #fig.save('/Users/varungupta/Downloads/Cancer..Wirth/CLUSTERING.png')

df_transpose.describe()

df.describe()

plt.scatter(X[:,0],X[:,1], label='True Position') 


# In[7]:
#################### Hierarchial Clustering Ends Here ##################


#################### ######### k MEANS Starts ################################

sse = {}
for k in range(2, 14):
    kmeans = KMeans(n_clusters=k, max_iter=14).fit(df_transpose)
    df_transpose["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[111]:


sse = {}
for k in range(2, 14):
    kmeans = KMeans(n_clusters=k, max_iter=30).fit(df)
    df["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.title('k-Means Clustering(SSE_vs_k)')
#plt.ylabel('k')
#plt.xlabel('SSE')
#>>
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster (k)")
plt.ylabel("SSE")
plt.show()
plt.savefig('/Users/varungupta/Downloads/Cancer..Wirth/kmeans.png',bbox_inches='tight')

"""
lt.figure(figsize=(10, 10))
    plt.title('Hierarchical Clustering Dendrogram. Metric: '+ met)
    plt.ylabel('Disease index')
    plt.xlabel('Distance')
    dendrogram(D_linkage,labels=df_transpose.index,orientation='left',)
    #plt.show()
    name=met+".png"
    plt.savefig(met,bbox_inches='tight')
    plt.savefig('sample.pdf')
    #savefig('test.png', bbox_inches='tight')
    #fig.save('/Users/varungupta/Downloads/Cancer..Wirth/CLUSTERING.png')

"""
#################### Kmean Clustering Ends Here ##################

#########################################################################
#################### Unsupervised Clustering Ends Here ##################


#################### Supervised Clustering Starts Here ##################
####################### Import of labelled data #########################

df=pd.read_csv('/Users/varungupta/Downloads/Cancer..Wirth/NewFile3.csv')
df_x=df[['CHF', 'Cardiac_dysrhythmias', 'Heart_valve_disorders',
       'Pulmonary_heart_disease', 'Peripheral_Heart', 'Hypertension',
       'Conduction', 'Nerves', 'Chronic_Pulmonary', 'Diabetes',
       'Thyroid_disorder', 'Renal', 'Liver', 'Gastroduodenal_ulcer', 'HIV',
       'arthritis', 'Coagulation', 'Endocrine', 'Nutritional_deficiencies',
       'Fluid_electrolyte', 'Alcohol_Abuse', 'Substance_Abuse',
       'Schizophrenia_psychoses', 'Anxiety', 'Affective']]


df_y=df[['kmodes']]

#print(df['kmodes'].unique())

x_train, x_test, y_train, y_test= train_test_split(df_x, df_y, test_size=0.20, random_state=42)
y_train=y_train.astype('int')


# In[158]:


############################
### Random Forest Start####
###########################

#model1 = RandomForestClassifier(n_estimators=10, max_depth=18,criterion='entropy')#.9984
model1 = RandomForestClassifier(n_estimators=10, max_depth=10)#.9984 #10
model1.fit(x_train, y_train)
pred_RF = model1.predict(x_test)
#print(metrics.accuracy_score(prediction,y_test))

##Visualisation###
visualizer = ClassificationReport(model1, classes=['1','2','3'])
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof() 
 
cm = ConfusionMatrix(model1, classes=[1,2,3])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(x_train, y_train)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(x_test, y_test)
# How did we do?
cm.poof()
print ("scores= ", visualizer.score(x_test, y_test)) 
print ('Confusion matrix:\n', confusion_matrix(y_test, pred_RF))


###########################
### Random Forest Ends  ####
###########################






################################
### Logistic Regression starts###
################################

#fit a logistic regression model to the data
model = LogisticRegression(multi_class='ovr')
model.fit(x_test, y_test)
#print(model)
#make predictions
expected = y_test
pred_log = model.predict(x_test)

###Visualisation###
visualizer = ClassificationReport(model, classes=['1','2','3'])
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
g = visualizer.poof() 
 
print ("scores= ", visualizer.score(x_test, y_test))
cm = ConfusionMatrix(model, classes=[1,2,3])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(x_train, y_train)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(x_test, y_test)
# How did we do?
cm.poof()
print ("scores= ", visualizer.score(x_test, y_test)) 
print ('Confusion matrix:\n', confusion_matrix(y_test, pred_RF))
##############################
### Logistic Regression End###
##############################


# In[64]:


print df_confusion = pd.crosstab(x_test, y_pred)
print (y_test)


# In[98]:


#####################################
############ KNN Start###############
#####################################

k_range = range(7, 75)

# list of scores from k_range
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    scores
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())

    ###Plot of KNN vs Cross Validation Accuracy####
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')



#knn = KNeighborsClassifier(n_neighbors=10)
#scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
krange = range(1, 50)
sco={}
for i in krange:
    w=0
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train) 
    predKnn=neigh.predict(x_test)
    w=metrics.accuracy_score(predKnn,y_test)
    sco[i]=w
    
plt.figure()
plt.plot(list(sco.keys()), list(sco.values()))
plt.xlabel("k-nearest neighbor")
plt.ylabel("Accuracy")
plt.show()
    


# In[176]:


knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train) 
predKnn=knn.predict(x_test)
#w=metrics.accuracy_score(predKnn,y_test)

visualizer = ClassificationReport(knn, classes=['1','2', '3'])
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
g = visualizer.poof() 
 
print ("scores= ", visualizer.score(x_test, y_test))
print (classification_report(y_test, predKnn, target_names=['1','2','3']))
 
from sklearn.metrics import confusion_matrix
 
cm = ConfusionMatrix(knn, classes=[1,2,3])
# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
cm.fit(x_train, y_train)
# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
# and then creates the confusion_matrix from scikit-learn.
cm.score(x_test, y_test)
print (metrics.accuracy_score(predKnn,y_test))
# How did we do?
cm.poof()
print ("scores= ", visualizer.score(x_test, y_test)) 

#####################################
############ KNN Ends###############
#####################################


# In[5]:

############################### Neural Network starts ####################################
################################################################################

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


seed = 7
numpy.random.seed(seed)

df=pd.read_csv('/Users/varungupta/Downloads/Cancer..Wirth/NewFile3.csv', skiprows=[0])
"""
df=df[['CHF', 'Cardiac_dysrhythmias', 'Heart_valve_disorders',
       'Pulmonary_heart_disease', 'Peripheral_Heart', 'Hypertension',
       'Conduction', 'Nerves', 'Chronic_Pulmonary', 'Diabetes',
       'Thyroid_disorder', 'Renal', 'Liver', 'Gastroduodenal_ulcer', 'HIV',
       'arthritis', 'Coagulation', 'Endocrine', 'Nutritional_deficiencies',
       'Fluid_electrolyte', 'Alcohol_Abuse', 'Substance_Abuse',
       'Schizophrenia_psychoses', 'Anxiety', 'Affective','kmodes']]
"""
df.head()
#dataframe = pandas.read_csv("iris.csv", )
dataset = df.values
df_x = dataset[:,1:26].astype(float)
df_y = dataset[:,26]

encoder = LabelEncoder()
encoder.fit(df_y)
encoded_Y = encoder.transform(df_y)
dummy_y = np_utils.to_categorical(encoded_Y)
#x_train, x_test, y_train, y_test= train_test_split(df_x, dummy_y, test_size=0.20, random_state=42)

def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(8, input_dim=25, activation='relu'))
    model.add(Dense(10, input_dim=25, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #rmsprop
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=20, verbose=1)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#print 
results = cross_val_score(estimator, df_x, dummy_y, cv=10)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#single dense hidden layer(98%)
#2 dense hidden layer tanh (Baseline: 99.76% (0.23%))
#2 dense hidden layer with relu Baseline: 99.73% (0.29%)

############################### Neural Network ends ####################################
###################################################################################


# In[ ]:




